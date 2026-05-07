from enum import Enum
import time
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import numpy as np

class OP(str, Enum):
    """
    a simple enum/vocabulary for the different types of fx nodes we expect to see in the graph
    """
    
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        """One-time static analysis of the traced graph.

        Runs once when the profiler is constructed (before any iteration runs).
        Initialises the per-iteration accumulators (``raw_measurements``,
        ``stats``, ``memory_timeline``, ``peak_memory``) and walks the graph
        to:
          1. locate the SEPFunction marker that separates the forward and
             backward regions, and
          2. classify every node as PARAM / ACT / GRAD based on where it sits
             relative to that marker (placeholders → PARAM, pre-sep ops → ACT,
             post-sep ops → GRAD). The classification is stored in
             ``self.node_mapping`` and used later by ``aggregate_stats``.
        """
        super().__init__(module, garbage_collect_values)

        self.raw_measurements = []
        self.node_mapping = {}
        self.stats = {}
        self.use_cuda_events = torch.cuda.is_available()
        self.memory_timeline = np.array([], dtype=float)
        self.peak_memory = 0.0
        self.peak_step = 0

        # Single pass over the graph: locate the fwd/bwd separator, the
        # backward separator (which marks the start of the optimizer phase),
        # build the node→index map, and classify every node as
        # PARAM / ACT / OTHER / GRAD relative to the separator.
        self.sep_idx = None
        sep_node = None ## marks the SEPFunction node that separates forward and backward
        idx_of: Dict[fx.Node, int] = {}
        nodes_in_order: list[fx.Node] = []
        sep_targets: list[str] = []

        for i, node in enumerate(self.module.graph.nodes):
            idx_of[node] = i
            nodes_in_order.append(node)
            target_str = str(node.target)
            if "separator" in target_str:
                sep_targets.append(target_str)
            if sep_node is None and node.target == torch.ops.separator.sep.default:
                sep_node = node
                self.sep_idx = i
            if node.target == torch.ops.separator.sep_backward.default:
                sep_bwd_idx = i

        # Optimizer phase begins immediately after the SEP-backward marker.
        # Falls back to None if the marker isn't present.
        self.optim_start_idx = (sep_bwd_idx + 1) if sep_bwd_idx is not None else None

        sep_idx_local = self.sep_idx if self.sep_idx is not None else float("inf")
        sep_found = False
        for node in nodes_in_order:
            if node is sep_node:
                sep_found = True
                continue
            if node.op == "placeholder":
                self.node_mapping[node] = NodeType.PARAM
            elif not sep_found:
                has_bwd_user = any(
                    idx_of.get(u, -1) > sep_idx_local for u in node.users
                )
                self.node_mapping[node] = (
                    NodeType.ACT if has_bwd_user else NodeType.OTHER
                )
            else:
                self.node_mapping[node] = NodeType.GRAD

        print(f"separator targets in graph: {sep_targets}")
    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        """Entry point for one iteration of graph execution.

        Called by ``graph_transformation`` once per warmup/profile iteration.
        Decides whether CUDA-event timing is usable (only when CUDA is
        available *and* the inputs actually live on a CUDA device), then
        delegates the node-by-node walk to ``fx.Interpreter.run``, which in
        turn calls ``self.run_node`` for every node.
        """
        self.use_cuda_events = (
            torch.cuda.is_available() and self._contains_cuda_tensor(args)
        )
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        """Per-node measurement hook — called once for every node, every iteration.

        Wraps the parent class's actual op execution (``super().run_node``)
        with timing and memory accounting:
          - on CUDA: uses ``torch.cuda.Event`` for timing and
            ``torch.cuda.memory_allocated`` for ground-truth allocation deltas;
          - on CPU: falls back to ``time.perf_counter`` (memory delta is
            unavailable, so it's recorded as 0).
        Each call appends one measurement to ``self.raw_measurements``;
        ``aggregate_stats`` later collapses these per-iteration samples into
        per-node averages. The returned tensor is unchanged so the parent's
        execution loop can keep flowing.
        """
        if self.use_cuda_events:
            mem_before = torch.cuda.memory_allocated()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            result = super().run_node(n)

            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event)
            mem_delta = torch.cuda.memory_allocated() - mem_before
        else:
            start_time = time.perf_counter()
            result = super().run_node(n)
            duration = (time.perf_counter() - start_time) * 1000
            mem_delta = 0

        self.raw_measurements.append({
            "node_name": n.name,
            "latency": duration, ## how long did this node take to execute in ms
            "mem_delta": mem_delta,
            "mem_size": self._result_size_bytes(result),
        })


        return result

    def _result_size_bytes(self, result: Any) -> int:
        """Recursive helper — total tensor footprint (in bytes) of a value.

        A node's output may be a single tensor, a tuple/list of tensors, or a
        dict containing tensors. This walks all of those cases and sums
        ``numel * element_size`` for each tensor leaf, returning 0 for
        non-tensor leaves.
        """
        if isinstance(result, torch.Tensor):
            return result.element_size() * result.nelement()
        if isinstance(result, (list, tuple)):
            return sum(self._result_size_bytes(item) for item in result)
        if isinstance(result, dict):
            return sum(self._result_size_bytes(item) for item in result.values())
        return 0

    def _contains_cuda_tensor(self, value: Any) -> bool:
        """Recursive helper — does any tensor in this nested value live on CUDA?

        Used by ``run`` to decide between CUDA-event timing and CPU
        ``perf_counter`` timing. Returns True only if at least one tensor leaf
        has ``is_cuda == True``.
        """
        if isinstance(value, torch.Tensor):
            return value.is_cuda
        if isinstance(value, (list, tuple)):
            return any(self._contains_cuda_tensor(item) for item in value)
        if isinstance(value, dict):
            return any(self._contains_cuda_tensor(item) for item in value.values())
        return False

    def aggregate_stats(self) -> None:
        """Collapse per-iteration samples into per-node statistics.

        Called once after the profile iterations are done. For each node:
          - averages its ``latency`` samples (mean across the y profile runs);
          - takes the max of its ``mem_size`` samples as its size;
          - computes its lifetime ``(first_use, last_use)`` from graph
            structure: first_use = the node's own index, last_use = the max
            index among its consumers (``node.users``);
          - records its NodeType classification from ``self.node_mapping``.
        It then builds the memory timeline by treating each tensor as
        contributing +size at first_use and −size at last_use+1, taking the
        prefix sum, and recording the running peak (``self.peak_memory``,
        ``self.peak_step``).
        """
        ## dictionary to map node to its index in the graph
        node_to_idx = {node: i for i, node in enumerate(self.module.graph.nodes)}

        total_steps = len(node_to_idx)

        # Track the 'changes' in memory at each step
        deltas = [0.0] * (total_steps + 1)

        for node in self.module.graph.nodes:
            node_measurements = [
                measurement
                for measurement in self.raw_measurements
                if measurement["node_name"] == node.name
            ]
            size = max(
                (measurement["mem_size"] for measurement in node_measurements),
                default=0,
            )
            latency = float(np.mean([
                measurement["latency"] for measurement in node_measurements
            ])) if node_measurements else 0.0
            first_use = node_to_idx[node]
            last_use = max([node_to_idx[u] for u in node.users], default=first_use)
            
            # 2. Mark the boundaries
            deltas[first_use] += size
            deltas[last_use + 1] -= size
            
            # 3. Save to stats dict
            self.stats[node.name] = {
                "size_bytes": size,
                "latency_ms": latency,
                "lifetime": (first_use, last_use),
                "type": self.node_mapping.get(node, NodeType.OTHER).name,
                "target": str(node.target),
            }

        
        self.memory_timeline = np.cumsum(deltas[:-1])
        if self.memory_timeline.size:
            self.peak_memory = float(np.max(self.memory_timeline))
            self.peak_step = int(np.argmax(self.memory_timeline))
        else:
            self.peak_memory = 0.0
            self.peak_step = 0
                        
        n_grads = sum(1 for s in self.stats.values() if s["type"] == "GRAD")
        print(f"GRAD count: {n_grads}, peak_step: {self.peak_step}, sep_idx: {self.sep_idx}")

        # Peak restricted to forward + backward (excludes optimizer phase)
        if self.optim_start_idx and self.optim_start_idx <= len(self.memory_timeline):
            region = self.memory_timeline[: self.optim_start_idx]
            self.training_peak_memory = float(np.max(region))
            self.training_peak_step = int(np.argmax(region))
        else:
            self.training_peak_memory = self.peak_memory
            self.training_peak_step = self.peak_step

        # Breakdown at the training peak (where activations are alive)
        self.peak_breakdown = {t: 0 for t in NodeType}
        for stat in self.stats.values():
            first, last = stat["lifetime"]
            if first <= self.training_peak_step <= last:
                self.peak_breakdown[NodeType[stat["type"]]] += stat["size_bytes"]

        acts_alive = sum(1 for s in self.stats.values()
                         if s["type"] == "ACT"
                         and s["lifetime"][0] <= self.training_peak_step <= s["lifetime"][1])
        print(f"[debug] training_peak_step: {self.training_peak_step}, "
              f"ACTs alive: {acts_alive}, total ACTs: "
              f"{sum(1 for s in self.stats.values() if s['type']=='ACT')}")


    def print_stats(self) -> None:
        """Pretty-print the per-node profile report.

        Sorts the entries in ``self.stats`` by their first-use index (so the
        output reads in topological order) and prints one line per node with
        its size in bytes, average latency in ms, lifetime window, and
        NodeType classification. Requires ``aggregate_stats`` to have been
        called first; produces nothing useful if called on an empty
        ``self.stats``.
        """
        self.stats = dict(sorted(self.stats.items(), key=lambda item: item[1]["lifetime"][0]))
        for node_name, stat in self.stats.items():
            print(
                f"Node: {node_name}, "
                f"Size (Bytes): {stat['size_bytes']}, "
                f"Latency (ms): {stat['latency_ms']:.3f}, "
                f"Lifetime: {stat['lifetime']}, "
                f"Type: {stat['type']}"
            )

    def reset_stats(self) -> None:
        """Wipe per-iteration accumulators between warmup and measurement.

        Called by ``graph_transformation`` after the warmup iterations have
        run, so that the timings/memory samples actually used to produce the
        report come only from the profile iterations (not from cold-cache
        warmup runs). Note: this does *not* reset the static analysis built
        in ``__init__`` (``self.node_mapping``, the SEP boundary) — those are
        graph-structural and don't change between iterations.
        """
        self.raw_measurements = []
        self.stats = {}
        self.memory_timeline = np.array([], dtype=float)
        self.peak_memory = 0.0
        self.peak_step = 0



    def plot_memory(self, output_path: str = "peak_memory.png") -> None:
        """plot the memory-over-time curve for one iteration to a PNG.

        Plots ``self.memory_timeline`` (live memory at each graph node index)
        in MB, marks and annotates the peak point, and saves the figure to
        ``output_path``. Auto-runs ``aggregate_stats`` if it hasn't been
        called yet (so the timeline exists).
        """
        if self.memory_timeline.size == 0:
            self.aggregate_stats()

        timeline_mb = self.memory_timeline / (1024 ** 2)
        peak_memory_mb = self.peak_memory / (1024 ** 2)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timeline_mb, color="#378ADD", linewidth=1.5, label="Estimated live memory")
        
        if timeline_mb.size:
            ax.scatter(self.peak_step, peak_memory_mb, color="#D85A30", zorder=5)
            ax.axhline(
                peak_memory_mb,
                color="#D85A30",
                linestyle="--",
                linewidth=1.0,
                label=f"Peak: {peak_memory_mb:.1f} MB",
            )
            ax.annotate(
                f"{peak_memory_mb:.1f} MB",
                (self.peak_step, peak_memory_mb),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="#D85A30",
            )
        
        ax.set_title("Peak Memory Across FX Graph")
        ax.set_xlabel("Graph node index")
        ax.set_ylabel("Estimated live memory (MB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Peak memory plot saved to {output_path}")

    def plot_peak_breakdown(self, output_path="figures/peak_breakdown.png"):
        '''
        plots the memory consumption breakdown by tensor type as side-by-side bars
        '''
        colors = {NodeType.ACT: "#378ADD", NodeType.PARAM: "#1D9E75",
                  NodeType.GRAD: "#D85A30", NodeType.OTHER: "#888780"}

        # filter out zero-byte categories
        items = [(t, b) for t, b in self.peak_breakdown.items() if b > 0]
        labels = [t.name for t, _ in items]
        values_mb = [b / (1024**2) for _, b in items]
        bar_colors = [colors[t] for t, _ in items]

        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.bar(labels, values_mb, color=bar_colors, width=0.6)

        for bar, mb in zip(bars, values_mb):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{mb:.1f} MB",
                    ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Memory at peak (MB)")
        ax.set_title("Peak memory breakdown by tensor type")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

def plot_peak_memory_vs_batch_size(
    batch_sizes: list[int],
    peak_memories_bytes: list[float],
    output_path: str = "peak_memory_vs_batch_size.png",
) -> None:
    """plot peak memory as a bar chart vs. mini-batch size for multiple iterations

    Standalone helper (not a method on GraphProfiler) so the caller can
    collect peak-memory readings from multiple ``GraphProfiler`` runs at
    different batch sizes, then pass the parallel lists in. Each bar is
    labelled with its peak in MB; saves the figure to ``output_path``.
    """
    if len(batch_sizes) != len(peak_memories_bytes):
        raise ValueError("batch_sizes and peak_memories_bytes must have the same length.")
    if not batch_sizes:
        raise ValueError("At least one batch size is required to plot peak memory.")

    peak_memories_mb = [memory / (1024 ** 2) for memory in peak_memories_bytes]
    labels = [str(batch_size) for batch_size in batch_sizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, peak_memories_mb, color="#378ADD", width=0.65)

    for bar, peak_memory_mb in zip(bars, peak_memories_mb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{peak_memory_mb:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Peak Memory vs Mini-Batch Size")
    ax.set_xlabel("Mini-batch size")
    ax.set_ylabel("Peak memory (MB)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Peak memory vs batch size plot saved to {output_path}")


def plot_with_vs_without_ac(
    records: list[tuple[int, float, float, float, float]],
    output_path: str = "peak_memory_ac.png",
) -> None:
    """Side-by-side bars: peak memory without vs with activation checkpointing.

    Each record is (batch, peak_no_ac_bytes, peak_ac_bytes, lat_no_ac, lat_ac).
    """
    if not records:
        return
    batches = [r[0] for r in records]
    peak_no = [r[1] / (1024 ** 2) for r in records]
    peak_ac = [r[2] / (1024 ** 2) for r in records]

    x = np.arange(len(batches))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, peak_no, w, label="without AC", color="#378ADD")
    ax.bar(x + w / 2, peak_ac, w, label="with AC",    color="#D85A30")

    for i, (no, ac) in enumerate(zip(peak_no, peak_ac)):
        ax.text(i - w / 2, no, f"{no:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, ac, f"{ac:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Mini-batch size")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Peak memory: with vs without activation checkpointing")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"AC peak comparison saved to {output_path}")


def plot_latency_with_vs_without_ac(
    records: list[tuple[int, float, float, float, float]],
    output_path: str = "latency_ac.png",
) -> None:
    """Side-by-side bars: iteration latency without vs with activation checkpointing."""
    if not records:
        return
    batches = [r[0] for r in records]
    lat_no = [r[3] for r in records]
    lat_ac = [r[4] for r in records]

    x = np.arange(len(batches))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, lat_no, w, label="without AC", color="#378ADD")
    ax.bar(x + w / 2, lat_ac, w, label="with AC",    color="#D85A30")

    for i, (no, ac) in enumerate(zip(lat_no, lat_ac)):
        ax.text(i - w / 2, no, f"{no:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, ac, f"{ac:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Mini-batch size")
    ax.set_ylabel("Iteration latency (ms)")
    ax.set_title("Latency: with vs without activation checkpointing")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"AC latency comparison saved to {output_path}")
