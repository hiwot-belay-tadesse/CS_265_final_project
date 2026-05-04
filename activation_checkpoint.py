import inspect
import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Tuple
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from graph_tracer import SEPFunction


def _extract_subgraph(joint_graph, inputs, outputs):
    """Version-tolerant wrapper around _extract_graph_with_inputs_outputs.

    Newer PyTorch added an ``outputs_descs`` required argument; older builds
    don't have it. Pass an empty list when the parameter exists.
    """
    sig = inspect.signature(_extract_graph_with_inputs_outputs)
    if "outputs_descs" in sig.parameters:
        return _extract_graph_with_inputs_outputs(
            joint_graph=joint_graph,
            inputs=inputs,
            outputs=outputs,
            outputs_descs=[None] * len(outputs),
        )
    return _extract_graph_with_inputs_outputs(
        joint_graph=joint_graph, inputs=inputs, outputs=outputs)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — μ-TWO greedy selection
# ─────────────────────────────────────────────────────────────────────────────

# ATen ops that return *views* (alias the input's storage) — dropping them
# saves no real memory because they don't own any.
_VIEW_OPS = {
    "transpose", "squeeze", "unsqueeze", "view", "reshape",
    "permute", "t", "expand", "slice", "select", "detach",
    "as_strided", "narrow", "unbind", "split",
}


def _is_view_target(target_str: str) -> bool:
    """target looks like ``aten.transpose.int`` — split on dots and match
    any segment against the view-op set."""
    return any(part in _VIEW_OPS for part in target_str.split("."))


# PCIe bandwidth assumption used to score the SWAP option in μ-TWO.
# 12 GB/s is a reasonable default for PCIe Gen3 x16; tune for your hardware.
_PCIE_BANDWIDTH_BYTES_PER_MS = 12 * 1024 * 1024 * 1024 / 1000.0


def _recompute_chain_cost_ms(
    node: "fx.Node",
    stats: Dict[str, dict],
    placeholder_names: set,
    cache: Dict[str, float],
) -> float:
    """Sum of latencies along the recomputation chain from boundary
    (placeholders) up to ``node``. Memoised in ``cache``."""
    if node.name in cache:
        return cache[node.name]
    if node.name in placeholder_names or node.op == "placeholder":
        cache[node.name] = 0.0
        return 0.0

    cost = stats.get(node.name, {}).get("latency_ms", 0.0)
    for inp in node.all_input_nodes:
        cost += _recompute_chain_cost_ms(inp, stats, placeholder_names, cache)
    cache[node.name] = cost
    return cost


def _idle_window_steps(lifetime_fwd_last: int, lifetime_bwd_first: int) -> int:
    """Number of graph steps the activation sits idle between its last
    forward use and its first backward use."""
    return max(0, lifetime_bwd_first - lifetime_fwd_last)


def select_activations_to_recompute(
    stats: Dict[str, dict],
    current_peak_bytes: float,
    target_peak_bytes: float,
    max_recompute: int = 10,
    gm: "fx.GraphModule | None" = None,
    peak_step: int | None = None,
    sep_idx: int | None = None,
    enable_swap: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """μ-TWO algorithm — full per-tensor cost analysis + greedy fill.

    For each candidate activation:
      • idle_window = first_bwd_use - last_fwd_use     (steps it sits idle)
      • recompute_cost_ms = Σ latencies of ops needed to recompute it
      • swap_cost_ms = 2 · size / PCIe_bandwidth        (out + in)
      • feasible options:
          - SWAP if swap_cost_ms <= idle_window_ms (use steps as ms proxy)
          - RECOMPUTE always (when ancestors reach placeholders)
      • chosen action = min-cost feasible option
      • score = size_bytes / chosen_cost_ms

    Then greedy: sort by score desc, pick until peak ≤ budget or cap reached.

    Args:
        stats, current_peak_bytes, target_peak_bytes, max_recompute, gm,
        peak_step: same as before.
        sep_idx: index of the SEPFunction marker; needed to split each user
                 into "forward" vs "backward" so we can compute idle windows.

    Returns:
        nodes_to_recompute: ordered list of activation names whose action is
                            RECOMPUTE (the rewriter only handles this action).
        schedule:           per-candidate action ∈ {RETAIN, RECOMPUTE, SWAP}.
                            SWAP entries are advisory — the rewriter currently
                            implements only RECOMPUTE.
    """
    # ── Step 1: basic candidate filter ─────────────────────────────────
    candidates = [
        (name, s) for name, s in stats.items()
        if s["type"] == "ACT"
        and s["size_bytes"] > 0
        and s["latency_ms"] > 0
        and not _is_view_target(s.get("target", ""))
    ]

    # Need the graph for users + idle window calculation.
    if gm is None or sep_idx is None:
        raise ValueError(
            "μ-TWO requires gm and sep_idx to compute idle windows / chain cost.")

    name_to_node = {n.name: n for n in gm.graph.nodes}
    idx_of = {n: i for i, n in enumerate(gm.graph.nodes)}
    placeholder_names = {n.name for n in gm.graph.nodes if n.op == "placeholder"}

    # ── Step 2: keep only activations alive at peak AND with a backward
    #           consumer past peak. Same logic as before — these are the
    #           tensors whose drop can move the peak. ────────────────────
    if peak_step is not None:
        def _alive_and_useful(act_name: str, lifetime: tuple) -> bool:
            if not (lifetime[0] <= peak_step <= lifetime[1]):
                return False
            node = name_to_node.get(act_name)
            if node is None:
                return False
            user_idxs = [idx_of[u] for u in node.users if u in idx_of]
            after_peak = [i for i in user_idxs if i > peak_step]
            return len(after_peak) > 0
        candidates = [(n, s) for n, s in candidates
                      if _alive_and_useful(n, s["lifetime"])]
        print(f"  [μ-TWO] {len(candidates)} eligible after peak filter")

    # ── Step 2b: shallow-recompute filter. Reject activations whose
    # producer needs other (still-live) candidates as inputs — those would
    # require recomputing chains, ballooning backward memory. We want
    # recomputation to be a single op whose inputs are already retained
    # (placeholders or non-candidate nodes).
    candidate_names_set = {n for n, _ in candidates}

    def _is_shallow(act_name: str) -> bool:
        node = name_to_node.get(act_name)
        if node is None:
            return False
        for inp in node.all_input_nodes:
            if inp.op == "placeholder":
                continue
            if inp.name in candidate_names_set:
                return False    # input would need recomputation too
        return True

    candidates = [(n, s) for n, s in candidates if _is_shallow(n)]
    print(f"  [μ-TWO] {len(candidates)} eligible after shallow filter")

    # ── Step 3: per-tensor cost analysis ───────────────────────────────
    chain_cache: Dict[str, float] = {}
    enriched: list[tuple[str, dict, str, float]] = []  # (name, stat, action, cost_ms)

    for name, s in candidates:
        node = name_to_node.get(name)
        if node is None:
            continue

        # Idle window: between last forward use and first backward use
        fwd_users = [idx_of[u] for u in node.users
                     if u in idx_of and idx_of[u] <= sep_idx]
        bwd_users = [idx_of[u] for u in node.users
                     if u in idx_of and idx_of[u] > sep_idx]
        if not bwd_users:
            continue  # no backward use → not a true activation
        last_fwd = max(fwd_users) if fwd_users else idx_of[node]
        first_bwd = min(bwd_users)
        idle_steps = _idle_window_steps(last_fwd, first_bwd)

        # Recompute cost: sum of latencies along the chain to placeholders
        recompute_cost_ms = _recompute_chain_cost_ms(
            node, stats, placeholder_names, chain_cache)

        # Swap cost: 2 × size / bandwidth (out + in)
        swap_cost_ms = (2.0 * s["size_bytes"]) / _PCIE_BANDWIDTH_BYTES_PER_MS

        # Feasibility: swap only if it fits the idle window. Use 1 step ≈
        # the average op latency as the unit; conservatively, require swap
        # to fit in fewer ms than idle_steps × min_latency_ms.
        # (Approximation: treat idle_steps as ms — sufficient for ranking.)
        swap_feasible = enable_swap and swap_cost_ms <= max(idle_steps, 1)

        options: list[tuple[str, float]] = [("RECOMPUTE", recompute_cost_ms)]
        if swap_feasible:
            options.append(("SWAP", swap_cost_ms))
        action, cost_ms = min(options, key=lambda x: x[1])

        enriched.append((name, s, action, cost_ms))

    # ── Step 4: rank by score and greedy fill ──────────────────────────
    # score = bytes saved per ms paid by chosen option
    enriched.sort(
        key=lambda e: e[1]["size_bytes"] / max(e[3], 1e-3),
        reverse=True,
    )

    nodes_to_recompute: List[str] = []
    schedule: Dict[str, str] = {name: "RETAIN" for name, *_ in enriched}
    projected_peak = current_peak_bytes
    swap_count = 0

    for name, s, action, cost_ms in enriched:
        if projected_peak <= target_peak_bytes:
            break
        if len(nodes_to_recompute) + swap_count >= max_recompute:
            break

        schedule[name] = action
        projected_peak -= s["size_bytes"]
        if action == "RECOMPUTE":
            nodes_to_recompute.append(name)
        else:
            swap_count += 1   # advisory; rewriter doesn't implement SWAP yet

    print(f"[μ-TWO] {len(nodes_to_recompute)} RECOMPUTE + {swap_count} SWAP "
          f"selected ({current_peak_bytes/1024**2:.1f} MB → "
          f"{projected_peak/1024**2:.1f} MB, target "
          f"{target_peak_bytes/1024**2:.1f} MB)")
    return nodes_to_recompute, schedule


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — graph rewriting (the existing manual example)
# ─────────────────────────────────────────────────────────────────────────────


# We define a custom function that takes in two weight matrices that require
# gradients to be computed and an input data matrix. The function returns the
# gradients of the weight matrices with respect to the loss (sum in our
# example). NOTE: The custom function mimics a simple two layer liner neural
# network with relu activation functions and a sum loss function.
def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.mm(w1, x)
    z = nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = nn.functional.relu(z)
    z = z.sum()
    z = SEPFunction.apply(z)
    z.backward()
    return w1.grad, w2.grad


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def _node_indices(gm: fx.GraphModule) -> Dict[fx.Node, int]:
    """Map each node to its position in topological order."""
    return {n: i for i, n in enumerate(gm.graph.nodes)}


def _first_backward_user(
    node: fx.Node, node_idx: Dict[fx.Node, int], sep_idx: int
) -> fx.Node:
    """First user of `node` that lives in the backward region (index > sep_idx)."""
    candidates = [u for u in node.users if node_idx.get(u, -1) > sep_idx]
    if not candidates:
        return None
    return min(candidates, key=lambda u: node_idx[u])


def _compute_recompute_inputs(
    node: fx.Node,
    recompute_names: set,
    visited: set = None,
) -> List[fx.Node]:
    """Walk back through `node`'s ancestors, stopping at placeholders or
    retained (non-recomputed) activations. Returns the boundary set — exactly
    the inputs the recomputation subgraph needs to start from."""
    if visited is None:
        visited = set()
    deps: List[fx.Node] = []
    for inp in node.all_input_nodes:
        if inp in visited:
            continue
        visited.add(inp)
        # Boundary: placeholders and retained activations stop the walk
        if inp.op == "placeholder" or inp.name not in recompute_names:
            deps.append(inp)
        else:
            # This input is also being recomputed → walk through it
            deps.extend(_compute_recompute_inputs(inp, recompute_names, visited))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[fx.Node] = []
    for d in deps:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def activation_checkpointing(
    gm: fx.GraphModule,
    nodes_to_recompute_names: List[str],
    sep_idx: int,
) -> fx.GraphModule:
    """Phase 3 — rewrite the graph so each activation in
    ``nodes_to_recompute_names`` is dropped after its last forward use and
    recomputed just before its first backward use.

    Args:
        gm:                       the traced graph module
        nodes_to_recompute_names: list of activation node names (from Phase 2's
                                  μ-TWO selection)
        sep_idx:                  index of the SEP marker (forward/backward
                                  boundary), from GraphProfiler.sep_idx

    Returns:
        the modified gm (rewritten in place; same object).
    """
    name_to_node = get_name_to_node_map(gm)
    node_idx = _node_indices(gm)
    recompute_set = set(nodes_to_recompute_names)

    for act_name in nodes_to_recompute_names:
        if act_name not in name_to_node:
            print(f"[AC] skipping '{act_name}' — not in graph")
            continue
        node_to_recompute = name_to_node[act_name]

        first_back = _first_backward_user(node_to_recompute, node_idx, sep_idx)
        if first_back is None:
            print(f"[AC] skipping '{act_name}' — no backward user")
            continue

        deps = _compute_recompute_inputs(node_to_recompute, recompute_set)

        # Extract a subgraph that recomputes act_name from its boundary inputs
        recompute_subgraph = _extract_subgraph(
            joint_graph=gm.graph,
            inputs=deps,
            outputs=[node_to_recompute],
        )

        # Splice the recomputation in just before the first backward use
        with gm.graph.inserting_before(first_back):
            for n in recompute_subgraph.nodes:
                if n.op in ("placeholder", "output"):
                    continue
                new_node = gm.graph.node_copy(
                    n, arg_transform=lambda arg: name_to_node[arg.name]
                )
                if n.name == act_name:
                    # Redirect downstream backward uses to the recomputed copy
                    replace_subsequent_uses_of(
                        gm.graph, old_node=node_to_recompute, new_node=new_node
                    )
                name_to_node[n.name] = new_node

        # Indices shifted because we inserted nodes — recompute
        node_idx = _node_indices(gm)

    gm.graph.lint()
    gm.recompile()
    print(f"[AC] rewrote graph: {len(nodes_to_recompute_names)} activations "
          f"checkpointed")
    return gm


if __name__ == "__main__":
    # Create two weight matrices that require gradients and one input data matrix
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    # Create a graph module by tracing the the custom function with the given inputs
    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the traced function
    # NOTE: We have already captured the backward operations during tracing
    # hence we are executing in no grad mode
    with torch.no_grad():
        old_grads = graph_module(w1, w2, x)

    # Apply the activation checkpointing algorithm (check new node 'relu_2')
    # Sep marker index for this hand-built example: scan the graph to find it.
    sep_idx = next(
        (i for i, n in enumerate(graph_module.graph.nodes)
         if n.target == torch.ops.separator.sep.default),
        None,
    )
    new_graph_module = activation_checkpointing(
        graph_module,
        nodes_to_recompute_names=["relu"],
        sep_idx=sep_idx,
    )
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the activation
    # checkpointed function to recalculate them
    with torch.no_grad():
        new_grads = new_graph_module(w1, w2, x)

    # Verify that gradients produced with activation checkpointing equal the
    # ones obtained earlier with no optimization.
    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))
