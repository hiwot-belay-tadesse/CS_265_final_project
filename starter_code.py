import argparse
import logging
import os
from typing import Any, Callable, NamedTuple

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from graph_prof import (
    GraphProfiler,
    plot_peak_memory_vs_batch_size,
    plot_with_vs_without_ac,
    plot_latency_with_vs_without_ac,
)
from graph_tracer import SEPFunction, compile
from activation_checkpoint import (
    select_activations_to_recompute,
    activation_checkpointing,
)

MODEL_NAMES = ("bert", "resnet152")


class ExperimentSetup(NamedTuple):
    """Bundle of everything one profiling run needs.

    Built once by ``build_experiment`` and consumed by ``run_profile``:
    the model, its optimizer, a synthetic input batch, and the model-specific
    ``train_step`` function the tracer will capture into a graph.
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer
    batch: tuple[torch.Tensor, ...]
    train_step: Callable[
        [torch.nn.Module, torch.optim.Optimizer, tuple[torch.Tensor, ...]], None
    ]


class TinyBertClassifier(nn.Module):
    """Minimal BERT-style classifier used as the LLM benchmark for Phase 1.

    Token + position + token-type embeddings → LayerNorm → stack of
    ``TransformerEncoderLayer`` blocks → linear classification head on the
    [CLS] position. Sized small (a couple of layers, hidden_dim=128) so it
    traces quickly and produces a graph small enough to read by hand.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        num_classes: int,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor):
        """Standard BERT forward — embed → encode → classify on [CLS]."""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        hidden_states = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.encoder(hidden_states)
        cls_hidden_state = hidden_states[:, 0, :]
        return self.classifier(cls_hidden_state)


# We wrap the loss with a separator function to call a
# dummy function 'SEPFunction', which is the separator function, that will call
# an identity operator at the end of the forward pass. This identity operator
# will get recorded in the computational graph and will inform you where the
# backward pass ends.


# This is the train_step function that takes in a model, optimizer and an input
# mini batch and calls the forward pass, loss function and the optimizer step. A
# computational graph corresponding to a train_step will be captured by the
# compiler.


def bert_train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: tuple[torch.Tensor, ...]
):
    """One BERT training iteration — forward + cross-entropy + backward + step.

    Wraps the loss with ``SEPFunction.apply`` to plant the marker the tracer
    uses to delimit the forward/backward boundary in the captured graph.
    """
    input_ids, token_type_ids, labels = batch
    logits = model(input_ids, token_type_ids)
    loss = F.cross_entropy(logits, labels)
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()


def resnet_train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: tuple[torch.Tensor, ...]
):
    """One ResNet-152 training iteration — forward + cross-entropy + backward + step.

    Same pattern as ``bert_train_step`` but with a single image tensor as
    input. ``SEPFunction.apply`` marks the forward/backward boundary for the
    tracer.
    """
    images, labels = batch
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()


# Below is a user defined function that accepts a graph module and arguments of
# used to run the graph. You can essentially do any operation, graph
# modification, profiling etc. inside this function. Subsequent to modifications
# or graph analysis, the function expects you to return the modified graph back.
# In the given example, we just print the graph, and then initilize the graph
# profiler. The graph profiler extends the class fx.Interpreter, that allows you
# to run the graph node by node, more explanation in graph_prof.py.


def make_graph_transformation(
    model_name: str,
    batch_size: int | None = None,
    peak_memory_records: list[tuple[int, float]] | None = None,
    plot_timeline: bool = True,
    print_node_stats: bool = True,
) -> Callable[[fx.GraphModule, Any], fx.GraphModule]:
    """Factory that builds the graph_transformation callback ``compile()`` expects.

    Returns a closure over the per-run options (model name, batch size,
    plotting toggles, optional list to append peak-memory readings into).
    The closure does the actual profiling: 2 warmup iterations → reset →
    3 profile iterations → aggregate → optionally print/plot/record.
    """
    def _profile(profiler: GraphProfiler, args):
        warm_up_iters, profile_iters = 2, 3
        with torch.no_grad():
            for _ in range(warm_up_iters):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(profile_iters):
                profiler.run(*args)
        profiler.aggregate_stats()
        return (
            profiler.training_peak_memory,
            sum(s["latency_ms"] for s in profiler.stats.values()),
        )

    def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        """Profile gm, apply AC, profile again, record both passes."""
        fig_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # ── Pass 1: profile WITHOUT AC ─────────────────────────────────
        profiler_no_ac = GraphProfiler(gm)
        peak_no_ac, lat_no_ac = _profile(profiler_no_ac, args)
        if print_node_stats:
            profiler_no_ac.print_stats()
        if plot_timeline:
            profiler_no_ac.plot_memory(
                os.path.join(fig_dir, f"peak_memory_{model_name}.png"))
            profiler_no_ac.plot_peak_breakdown(
                os.path.join(fig_dir, f"peak_breakdown_{model_name}.png"))

        # ── Phase 2: select activations to drop ────────────────────────
        target = 0.7 * peak_no_ac
        nodes_to_recompute, _ = select_activations_to_recompute(
            profiler_no_ac.stats,
            current_peak_bytes=peak_no_ac,
            target_peak_bytes=target,
            gm=gm,
            peak_step=profiler_no_ac.training_peak_step,
            sep_idx=profiler_no_ac.sep_idx,
        )

        # ── Phase 3: rewrite graph ─────────────────────────────────────
        gm = activation_checkpointing(
            gm, nodes_to_recompute, profiler_no_ac.sep_idx)

        # ── DEBUG: did the rewrite actually free the activations? ─────
        all_nodes = list(gm.graph.nodes)
        sep_pos = next(
            (i for i, n in enumerate(all_nodes)
             if n.target == torch.ops.separator.sep.default),
            -1,
        )
        for act_name in nodes_to_recompute:
            node = next((n for n in all_nodes if n.name == act_name), None)
            if node is None:
                print(f"  [debug] {act_name}: NOT IN GRAPH (erased)")
                continue
            user_indices = sorted(all_nodes.index(u) for u in node.users)
            in_bwd = [i for i in user_indices if i > sep_pos]
            print(f"  [debug] {act_name}: users={user_indices}, in_bwd={in_bwd}")

        # ── Pass 2: profile WITH AC ────────────────────────────────────
        profiler_ac = GraphProfiler(gm)
        peak_ac, lat_ac = _profile(profiler_ac, args)
        print(f"  peak_no_ac: {peak_no_ac/1024**2:.1f} MB")
        print(f"  peak_ac:    {peak_ac/1024**2:.1f} MB")
        print(f"  delta:      {(peak_no_ac - peak_ac)/1024**2:.1f} MB")

        # Did the simulation actually shorten lifetimes? Print before/after
        # for each selected activation.
        for act_name in nodes_to_recompute:
            old = profiler_no_ac.stats.get(act_name, {}).get("lifetime")
            new = profiler_ac.stats.get(act_name, {}).get("lifetime")
            print(f"  [lifetime] {act_name}: {old} → {new}")

        # Guard: if the rewrite increased peak, the recomputation chain cost
        # more than the activations saved. Honest fallback: report baseline.
        if peak_ac >= peak_no_ac:
            print("  [AC] reverted — rewrite increased peak; reporting baseline")
            peak_ac = peak_no_ac
            lat_ac  = lat_no_ac

        # ── Record both for the comparison plots ───────────────────────
        if peak_memory_records is not None and batch_size is not None:
            peak_memory_records.append(
                (batch_size, peak_no_ac, peak_ac, lat_no_ac, lat_ac))

        return gm

    return graph_transformation


def build_bert_experiment(device_str: str, batch_size: int | None) -> ExperimentSetup:
    """Construct an ExperimentSetup for the TinyBert model.

    Builds the model, a synthetic (input_ids, token_type_ids, labels) batch,
    and a plain SGD optimizer (foreach=False to avoid CPU-PyTorch compile
    quirks). Default batch_size=8 if not overridden.
    """
    batch_size = 8 if batch_size is None else batch_size
    seq_len = 128
    vocab_size = 30_522
    hidden_dim = 128
    layers = 2
    heads = 4
    num_classes = 2

    model = TinyBertClassifier(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        num_classes=num_classes,
    ).to(device_str)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device_str)
    token_type_ids = torch.zeros(
        (batch_size, seq_len), dtype=torch.long, device=device_str
    )
    labels = torch.randint(0, num_classes, (batch_size,), device=device_str)
    batch = (input_ids, token_type_ids, labels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    return ExperimentSetup(model, optimizer, batch, bert_train_step)


def build_resnet152_experiment(device_str: str, batch_size: int | None) -> ExperimentSetup:
    """Construct an ExperimentSetup for ResNet-152.

    Builds the model with random weights, a synthetic (images, labels) batch
    of 224x224 RGB inputs, and a plain SGD optimizer. Disables in-place ops
    (e.g. ``ReLU(inplace=True)``) because ``make_fx`` tracing is incompatible
    with in-place mutation on parameter tensors. Default batch_size is 2 on
    CUDA / 1 on CPU.
    """
    from torchvision.models import resnet152

    batch_size = (
        2 if device_str.startswith("cuda") else 1
    ) if batch_size is None else batch_size
    num_classes = 1000

    model = resnet152(weights=None).to(device_str)
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    images = torch.randn(batch_size, 3, 224, 224, device=device_str)
    labels = torch.randint(0, num_classes, (batch_size,), device=device_str)
    batch = (images, labels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    return ExperimentSetup(model, optimizer, batch, resnet_train_step)


def build_experiment(
    model_name: str, device_str: str, batch_size: int | None
) -> ExperimentSetup:
    """Dispatch on model name to the appropriate builder.

    Raises ValueError if ``model_name`` is anything other than ``bert`` /
    ``resnet152``.
    """
    if model_name == "bert":
        return build_bert_experiment(device_str, batch_size)
    if model_name == "resnet152":
        return build_resnet152_experiment(device_str, batch_size)
    raise ValueError(f"Unknown model '{model_name}'. Expected one of {MODEL_NAMES}.")


def initialize_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Pre-populate optimizer state on real tensors before tracing.

    The tracer captures whatever state tensors already live in
    ``optimizer.state``. Optimizers like Adam only allocate their moment
    buffers after their first ``.step()``, so we run one step on random
    fake gradients to materialise them. For SGD without momentum this is a
    no-op state-wise but harmless. Runs *before* ``compile()`` is called.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param)

    optimizer.step()
    optimizer.zero_grad()


# We first initialize the model, pass it to the wrapper model, then create a
# random input mini-batch and initilize the optimizer. We then call the compile
# function that takes in two arguments, a train_step function and a
# graph_transformation function. The train_step function is the one that will be
# traced by the compiler and a computational graph for the same will be created.
# This computational graph is then passed to the graph_transformation function
# to do any graph profiling, modifications and optimizations. This modified
# graph is stored and will be returned as the compiled function. In essence we
# do the following inside the compile function:

# def compile (train_step, graph_transformation):
#     @wraps(train_step)
#     def inner(*args, **kwargs):
#         if not_compiled:
#             original_graph, input_args = graph_tracer(train_step)
#             modified_graph = graph_transformation(original_graph, input_args)
#         output = modified_graph(*args, **kwargs)
#         return output
#     return inner


def run_profile(
    model_name: str,
    batch_size: int | None,
    plot_timeline: bool = True,
    print_node_stats: bool = True,
) -> tuple[int, float, float, float, float]:
    """Run one full profiling pipeline for one (model, batch_size) pair.

    Returns: (batch_size, peak_no_ac, peak_ac, latency_no_ac, latency_ac)
    in bytes / ms. All zeros if profiling failed to produce a record.
    """
    logging.getLogger().setLevel(logging.DEBUG)
    model_name = model_name.lower()

    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    setup = build_experiment(model_name, device_str, batch_size)
    initialize_optimizer_state(setup.model, setup.optimizer)
    # holds (batch_size, peak_no_ac, peak_ac, lat_no_ac, lat_ac) per run
    peak_memory_records: list[tuple[int, float, float, float, float]] = []
    effective_batch_size = setup.batch[0].shape[0]
    graph_transformation = make_graph_transformation(
        model_name,
        batch_size=effective_batch_size,
        peak_memory_records=peak_memory_records,
        plot_timeline=plot_timeline,
        print_node_stats=print_node_stats,
    )
    compiled_fn = compile(setup.train_step, graph_transformation)
    compiled_fn(setup.model, setup.optimizer, setup.batch)
    return peak_memory_records[0] if peak_memory_records else (
        effective_batch_size, 0.0, 0.0, 0.0, 0.0)


def experiment(model_name: str = "bert", batch_size: int | None = None):
    """Top-level single-run entry point.

    Seeds RNG for reproducibility, then runs ``run_profile`` once with both
    the per-iteration timeline plot and the per-node stats table enabled.
    Used when no batch sweep was requested on the command line.
    """
    torch.manual_seed(20)
    run_profile(model_name, batch_size, plot_timeline=True, print_node_stats=True)


def batch_size_sweep(model_name: str, batch_sizes: list[int]) -> None:
    """Run with vs without AC at each batch size; produce comparison plots."""
    records: list[tuple[int, float, float, float, float]] = []

    for batch_size in batch_sizes:
        torch.manual_seed(20)
        record = run_profile(
            model_name, batch_size,
            plot_timeline=False, print_node_stats=False,
        )
        records.append(record)
        bs, peak_no, peak_ac, lat_no, lat_ac = record
        print(
            f"batch_size={bs}  "
            f"peak: {peak_no/1024**2:.1f} → {peak_ac/1024**2:.1f} MB  "
            f"latency: {lat_no:.1f} → {lat_ac:.1f} ms"
        )

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Phase 1 plot — peak memory (no-AC) vs batch size
    plot_peak_memory_vs_batch_size(
        [r[0] for r in records],
        [r[1] for r in records],
        os.path.join(fig_dir, f"peak_memory_vs_batch_size_{model_name}.png"),
    )

    # Phase 3 plots — with vs without AC
    plot_with_vs_without_ac(
        records, os.path.join(fig_dir, f"peak_memory_ac_{model_name}.png"))
    plot_latency_with_vs_without_ac(
        records, os.path.join(fig_dir, f"latency_ac_{model_name}.png"))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=os.environ.get("MODEL_NAME", "bert").lower(),
        help="Model to profile.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the default synthetic batch size.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Run a mini-batch size sweep and plot peak memory as a bar graph.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.batch_sizes:
        batch_size_sweep(model_name=args.model, batch_sizes=args.batch_sizes)
    else:
        experiment(model_name=args.model, batch_size=args.batch_size)
