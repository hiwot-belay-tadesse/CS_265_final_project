import inspect
import numpy as np
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

    # ── Step 2b: chain-reachability filter.
    # The recompute chain must terminate at placeholders or at intermediates
    # still alive at the recomputation point — otherwise the rewriter has
    # nothing to start from. We do *not* try to bound the recompute spike
    # statically here; μ-TWO trusts the post-pick greedy/re-simulation step
    # to verify the new peak. The size check we used to do (single op or
    # cumulative bytes ≤ target_size) was too strict — at deep BERT layers,
    # any chain with more than a couple of ops would fail it.
    def _chain_feasible(act_name: str) -> bool:
        target = name_to_node.get(act_name)
        if target is None:
            return False
        target_size = stats.get(act_name, {}).get("size_bytes", 0)
        if target_size <= 0:
            return False

        bwd_user_idxs = [
            idx_of[u] for u in target.users
            if u in idx_of and idx_of[u] > sep_idx
        ]
        if not bwd_user_idxs:
            return False
        recompute_point = min(bwd_user_idxs)

        # BFS up to placeholders or to surviving intermediates. As long as
        # every leaf of the BFS terminates at one of those, the recompute
        # subgraph is constructible.
        seen: set = set()
        frontier: list = [target]
        while frontier:
            node = frontier.pop()
            if node.name in seen:
                continue
            seen.add(node.name)

            if node is target:
                pass    # walk into target's inputs
            else:
                inp_lifetime = stats.get(node.name, {}).get("lifetime")
                if inp_lifetime is not None and \
                   inp_lifetime[0] <= recompute_point <= inp_lifetime[1]:
                    continue
                if node.op == "placeholder":
                    continue
            for prev in node.all_input_nodes:
                frontier.append(prev)
        return True

    candidates = [(n, s) for n, s in candidates if _chain_feasible(n)]
    print(f"  [μ-TWO] {len(candidates)} eligible after chain-feasibility filter")

    # Boundary helper used later for re-simulation. Mirrors the rewriter's
    # _compute_recompute_inputs assuming all other selected acts will also be
    # recomputed (worst-case boundary).
    def _rewriter_boundary(act_name: str, recompute_set: set) -> set:
        target = name_to_node.get(act_name)
        if target is None:
            return set()
        boundary: set = set()
        seen: set = set()
        frontier: list = [target]
        while frontier:
            node = frontier.pop()
            for inp in node.all_input_nodes:
                if inp.name in seen:
                    continue
                seen.add(inp.name)
                if inp.op == "placeholder" or inp.name not in recompute_set:
                    boundary.add(inp.name)
                else:
                    frontier.append(inp)
        return boundary

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

    # ── Step 4: rank by score and greedy fill, with re-simulation ───────
    # μ-TWO doesn't trust a static "peak − size" estimate: dropping an
    # activation also extends its boundary inputs' lifetimes (they're now
    # consumed at the recompute splice). Build a real timeline and re-derive
    # peak after each tentative pick.
    enriched.sort(
        key=lambda e: e[1]["size_bytes"] / max(e[3], 1e-3),
        reverse=True,
    )

    # Build the baseline delta array from current stats. Deltas are indexed
    # by graph step; +size at first_use, -size at last_use+1. Cumsum gives
    # live bytes; max gives peak.
    n_steps = len(idx_of) + 2
    deltas = np.zeros(n_steps, dtype=np.float64)
    # Track current effective lifetimes per node so we can update incrementally.
    eff_life: Dict[str, tuple] = {}
    for nm, st in stats.items():
        sz = st.get("size_bytes", 0)
        if sz <= 0:
            continue
        life = st.get("lifetime")
        if life is None:
            continue
        first, last = life
        deltas[first] += sz
        deltas[last + 1] -= sz
        eff_life[nm] = (first, last)

    def _current_peak() -> float:
        return float(np.max(np.cumsum(deltas[:-1])))

    sim_peak = _current_peak()

    def _alive_at_peak(nm: str) -> bool:
        st = stats.get(nm)
        if st is None:
            return False
        if st.get("type") == "PARAM":
            return True   # placeholders span the whole graph
        life = eff_life.get(nm) or st.get("lifetime")
        if life is None:
            return False
        return life[0] <= peak_step <= life[1]

    def _expand_recompute_set(seed: str, current: set) -> set:
        """Cascading recompute: starting from `seed`, walk back through
        boundary inputs that are forward-only (i.e. not alive at peak and
        not placeholders) and pull them into the recompute set too. The
        walk bottoms out when every boundary input is either a placeholder,
        already alive at peak, or already in the set. Returns the *added*
        names (not including the seed if it was already in `current`)."""
        added: set = set()
        worklist = [seed]
        while worklist:
            nm = worklist.pop()
            if nm in current or nm in added:
                continue
            node = name_to_node.get(nm)
            if node is None:
                continue
            # Pull this node into the recompute set.
            added.add(nm)
            # Examine its raw graph inputs (not the closure).
            for inp in node.all_input_nodes:
                if inp.op == "placeholder":
                    continue
                if _alive_at_peak(inp.name):
                    continue   # safe boundary: no lifetime extension at peak
                if inp.name in current or inp.name in added:
                    continue
                # Forward-only boundary input — keep cascading.
                worklist.append(inp.name)
        return added

    def _apply_lifetime(nm: str, new_first: int, new_last: int, log: list) -> None:
        st = stats.get(nm)
        if st is None:
            return
        size = st.get("size_bytes", 0)
        if size <= 0:
            return
        old = eff_life.get(nm, st.get("lifetime"))
        if old is None:
            return
        of, ol = old
        if (of, ol) == (new_first, new_last):
            return
        deltas[of] -= size
        deltas[ol + 1] += size
        deltas[new_first] += size
        deltas[new_last + 1] -= size
        log.append(("life", nm, old))
        eff_life[nm] = (new_first, new_last)

    def _try_pick(name: str, s: dict) -> tuple[bool, float, list]:
        """Cascading-recompute pick. Apply tentatively and re-derive peak;
        roll back if peak doesn't strictly drop."""
        node = name_to_node.get(name)
        if node is None:
            return False, sim_peak, []

        bwd_users = [idx_of[u] for u in node.users
                     if u in idx_of and idx_of[u] > sep_idx]
        if not bwd_users:
            return False, sim_peak, []
        recompute_point = min(bwd_users)

        # 1) Decide the full set of nodes that will be recomputed: `name` plus
        #    every forward-only ancestor reachable through non-peak-alive nodes.
        current_set = set(nodes_to_recompute)
        new_acts = _expand_recompute_set(name, current_set)
        if name not in new_acts:
            return False, sim_peak, []
        full_set = current_set | new_acts

        log: list = []

        # 2) Each newly-recomputed activation's lifetime collapses to its own
        #    forward range (last_use becomes its last fwd user). Its bytes
        #    disappear from the bwd region; instead, its *boundary inputs* —
        #    i.e. the nodes the rewriter's BFS bottoms out at — get extended
        #    to the recompute_point.
        for nm in new_acts:
            n_node = name_to_node.get(nm)
            if n_node is None:
                continue
            cur_first, cur_last = eff_life.get(nm, stats[nm]["lifetime"])
            fwd_uses = [idx_of[u] for u in n_node.users
                        if u in idx_of and idx_of[u] <= sep_idx]
            new_last = max(fwd_uses) if fwd_uses else cur_first
            if new_last < cur_last:
                _apply_lifetime(nm, cur_first, new_last, log)

        # 3) Now extend boundary inputs of the full set. The rewriter walks
        #    back through any node in full_set; the first non-set, non-
        #    placeholder ancestor is the boundary. Those get a use at
        #    recompute_point.
        boundary: set = set()
        seen: set = set()
        frontier = [name_to_node[nm] for nm in new_acts if nm in name_to_node]
        while frontier:
            cur = frontier.pop()
            for inp in cur.all_input_nodes:
                if inp.name in seen:
                    continue
                seen.add(inp.name)
                if inp.op == "placeholder" or inp.name not in full_set:
                    boundary.add(inp.name)
                else:
                    frontier.append(inp)

        for inp in boundary:
            inp_st = stats.get(inp)
            if inp_st is None:
                continue
            if inp_st.get("type") == "PARAM":
                continue
            inp_first, inp_last = eff_life.get(inp, inp_st["lifetime"])
            if recompute_point <= inp_last:
                continue
            _apply_lifetime(inp, inp_first, recompute_point, log)

        new_peak = _current_peak()
        if new_peak >= sim_peak:
            # Roll back
            for kind, nm, old in reversed(log):
                if kind == "life":
                    cur_first, cur_last = eff_life[nm]
                    cur_size = stats[nm].get("size_bytes", 0)
                    deltas[cur_first] -= cur_size
                    deltas[cur_last + 1] += cur_size
                    of, ol = old
                    deltas[of] += cur_size
                    deltas[ol + 1] -= cur_size
                    eff_life[nm] = old
            return False, sim_peak, []
        # Success: also commit cascaded acts so subsequent picks see them
        # in nodes_to_recompute.
        for nm in new_acts:
            if nm != name and nm not in nodes_to_recompute:
                nodes_to_recompute.append(nm)
                schedule[nm] = "RECOMPUTE"
        return True, new_peak, log

    nodes_to_recompute: List[str] = []
    schedule: Dict[str, str] = {name: "RETAIN" for name, *_ in enriched}
    swap_count = 0

    for name, s, action, cost_ms in enriched:
        if sim_peak <= target_peak_bytes:
            break
        if len(nodes_to_recompute) + swap_count >= max_recompute:
            break
        if action != "RECOMPUTE":
            schedule[name] = action
            swap_count += 1
            continue

        accepted, new_peak, _log = _try_pick(name, s)
        if not accepted:
            continue
        schedule[name] = action
        nodes_to_recompute.append(name)
        sim_peak = new_peak

    print(f"[μ-TWO] {len(nodes_to_recompute)} RECOMPUTE + {swap_count} SWAP "
          f"selected ({current_peak_bytes/1024**2:.1f} MB → "
          f"{sim_peak/1024**2:.1f} MB, target "
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

    # Plan all subgraph extractions against the ORIGINAL graph in one pass.
    # Doing this on a per-iteration basis after splices breaks: each extracted
    # subgraph would also pick up the spliced copies from prior iterations,
    # causing the recompute regions to roughly double in size each step.
    plans: list[tuple[str, fx.Node, fx.Node, fx.Graph]] = []
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
        recompute_subgraph = _extract_subgraph(
            joint_graph=gm.graph,
            inputs=deps,
            outputs=[node_to_recompute],
        )
        sub_ops = [n for n in recompute_subgraph.nodes
                   if n.op not in ("placeholder", "output")]
        sub_phs = [n.name for n in recompute_subgraph.nodes if n.op == "placeholder"]
        print(f"  [AC] {act_name}: deps={[d.name for d in deps]} "
              f"placeholders={sub_phs} ops={[n.name for n in sub_ops]}")
        plans.append((act_name, node_to_recompute, first_back, recompute_subgraph))

    for act_name, node_to_recompute, first_back, recompute_subgraph in plans:

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
