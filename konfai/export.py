# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Export a frozen KonfAI ``Network`` to a self-contained ONNX graph + manifest.

A trained model becomes ``model.onnx`` (graph + weights, single file) plus ``manifest.json``
(patch geometry, input/output spec) for a Python-free runtime. The export uses the **dynamo**
exporter, reaches the graph through ``named_forward`` and selects a named head, and inlines the
external weight data so the ``.onnx`` is a single self-contained file.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np
import torch

from konfai.utils.errors import PredictorError
from konfai.utils.utils import Placement, owned_destination

if TYPE_CHECKING:
    from konfai.network.network.network import ModuleArgsDict

MANIFEST_VERSION = 5  # 5: patch.stride under the end and spread placements
#: A manifest blending by GaussianFloat16 or by a patch combine of the app's extension: a runtime reading version 5
#: alone refuses it rather than blend it as Cosinus.
FLOAT16_BLEND_VERSION = 6
#: The bytes one ONNX file holds: protobuf's limit for a message, its weights inline.
ONE_FILE = 2**31 - 1


def _require(module: str) -> ModuleType:
    """Import an optional export dependency or raise an actionable error."""
    try:
        return import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise PredictorError(
            f"ONNX export needs the optional dependency '{module}'. Install it with `pip install konfai[export]`.",
        ) from exc


def _routed(model: torch.nn.Module) -> ModuleArgsDict:
    """``model`` as the routed graph the export walks; refused without ``named_forward``."""
    if not hasattr(model, "named_forward"):
        raise PredictorError("export expects a KonfAI Network exposing `named_forward`.")
    return cast("ModuleArgsDict", model)


class _NamedHead(torch.nn.Module):
    """Wrap a routed KonfAI graph to return a single named output tensor.

    ``named_forward`` exposes every module output as ``(dotted_name, tensor)``; this returns the
    tensor of ``output_module``. ``fold_pre`` are per-patch tensor->tensor callables applied to the
    input *inside* the graph, so they trace into the ONNX; only pointwise ops are foldable. ``fold_post``
    are applied to the output alike.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        output_module: str,
        fold_pre: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        fold_post: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.net = _routed(net)
        self.output_module = output_module
        self.fold_pre = list(fold_pre or [])
        self.fold_post = list(fold_post or [])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        # The folds prepare the group the chain belongs to, which is the model's first input.
        first, *rest = inputs
        for pre in self.fold_pre:
            first = pre(first)
        out = first
        for name, tensor in self.net.named_forward(first, *rest):
            if name == self.output_module:
                out = tensor
        for post in self.fold_post:
            out = post(out)
        # An argmax's labels as the float the manifest declares; a float head keeps the graph's precision.
        return out if out.is_floating_point() else out.float()


def list_output_modules(model: torch.nn.Module, *example_input: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(dotted_name, shape)`` for every output of the routed graph, to discover the
    inference head to export. One example per group the model reads, in its graph's order."""
    graph = _routed(model)
    model.eval()
    outputs: list[tuple[str, tuple[int, ...]]] = []
    with torch.no_grad():
        for name, tensor in graph.named_forward(*example_input):
            outputs.append((name, tuple(tensor.shape)))
    return outputs


def _cut_by(
    graph: ModuleArgsDict, prefix: str, inputs: list[str], counts: Counter[str], runs: int, patched: bool = True
) -> list[tuple[str, str]]:
    """What ``graph.named_forward`` yields, by name relative to ``graph``, each with the network whose
    ModelPatch left that tensor one patch of several ("" when it holds the whole input).

    It follows ``named_forward`` without running it: the patch pass and its assembly, then the branch
    routing, gates included. ``inputs`` holds the same for the inputs, ``counts`` how often the example
    pass yielded each full name (``prefix`` is ``graph``'s), and ``graph`` ran ``runs`` times.
    """
    from konfai.network.network.base import accumulator_owner, mark_accumulated, strip_accumulated
    from konfai.network.network.network import ModuleArgsDict

    if patched and getattr(graph, "patch", None):
        marked = prefix + mark_accumulated("")
        patches = min((n for name, n in counts.items() if name.startswith(marked)), default=runs)
        owner = (accumulator_owner(marked) or "the root network") if patches > runs else ""
        per_patch = _cut_by(graph, marked, inputs, counts, patches, patched=False)
        assembled = {name.split(".")[0]: cut for name, cut in per_patch}
        return [(mark_accumulated(name), cut or owner) for name, cut in per_patch] + [
            (name, cut) for name, cut in assembled.items() if graph._modulesArgs[name]._isEnd
        ]
    if not inputs:
        return []
    branches = {str(i): cut for i, cut in enumerate(inputs)}
    out, yielded = inputs[0], []
    for name, module in graph.items():
        args = graph._modulesArgs[name]
        if module is None or not graph._runs(name):
            continue
        reads = [branches.setdefault(branch, inputs[0]) for branch in args.in_branch]
        if isinstance(module, ModuleArgsDict) and not args.isCheckpoint:
            inner = _cut_by(module, f"{prefix}{name}.", reads, counts, runs)
            written = set()
            for key, out in inner:
                matched = set(args.out_branch) & set(
                    module._modulesArgs[strip_accumulated(key.split(".")[0])].out_branch
                )
                branches.update(dict.fromkeys(matched, out))
                written |= matched
            branches.update(dict.fromkeys(set(args.out_branch) - written, out))
            yielded += [(f"{name}.{key}", cut) for key, cut in inner]
        else:
            out = next((cut for cut in reads if cut), "")
            branches.update(dict.fromkeys(args.out_branch, out))
            yielded.append((name, out))
    return yielded


def _refuse_one_patch_of_several(model: torch.nn.Module, outputs: list[tuple[str, tuple[int, ...]]], head: str) -> None:
    """Refuse an export whose ``head`` would see one patch of several (``outputs`` from
    :func:`list_output_modules`).

    A ModelPatch that cuts the example into several patches yields each per-patch output once per
    patch under one name, and the export keeps the last. A network assembles its patches only once
    the model is bound (as Trainer and Predictor bind it), and the root never does, so the head sees
    the last patch when it is a per-patch output or reads one through the branches.
    """
    from konfai.network.network.network import ModuleArgsDict

    if not isinstance(model, ModuleArgsDict):  # no ModelPatch outside a routed graph
        return
    owner = dict(_cut_by(_routed(model), "", [""], Counter(name for name, _ in outputs), 1)).get(head, "")
    if owner:
        raise PredictorError(
            f"The ModelPatch of {owner} cuts the example input into several patches; "
            f"the exported head '{head}' would see the last patch only.",
            "Export with an example input of one patch, or without the ModelPatch (the runtime tiles the volume).",
        )


def select_inference_head(model: torch.nn.Module, *example_input: torch.Tensor) -> str:
    """Pick the head to export: the last **floating-point** output in execution order.

    Integer outputs (an ``Argmax`` label map) are skipped. Pass ``output_module`` to target a
    specific head.
    """
    graph = _routed(model)
    model.eval()
    head: str | None = None
    with torch.no_grad():
        for name, tensor in graph.named_forward(*example_input):
            if torch.is_floating_point(tensor):
                head = name
    if head is None:
        raise PredictorError("no floating-point output to export; pass an explicit output_module.")
    return head


def _scalar_tensor(s: float, dtype: int = 1, layout: str = "", device: str = "", pin_memory: bool = False) -> Any:
    """``aten.scalar_tensor``, the Python scalar of an op, as a constant of its own dtype. The exporter's own makes a
    float a float32 constant cast to that dtype: a float64 graph would hold ``x * 0.95`` as 0.949999988. A float32
    graph comes out the same either way."""
    # ponytail: torch.full's fill value takes the same float32 path; translate aten.full too once a graph needs it.
    import onnx_ir as ir
    from onnxscript import opset18 as op

    # The stub names a TensorProto; the exporter's tracer takes an ir tensor.
    return op.Constant(value=ir.tensor(s, dtype=ir.DataType(dtype)))  # type: ignore[arg-type]


def _write_onnx(
    module: torch.nn.Module,
    examples: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int = 18,
    dynamic: bool = False,
    free_from: int = 2,
) -> None:
    """Write ``module``, traced on ``examples``, as one self-contained ONNX file at ``onnx_path``: the weights go
    inline, also past the 1.5 GiB torch moves to a file beside the graph, up to :data:`ONE_FILE`, past which the
    export is refused. The exporter records every node's Python stack, and with it the paths of the machine that
    exported, which a model handed to others does not carry. ``dynamic``: every axis of every input from
    ``free_from`` on (the spatial axes of ``[N, C, ...]``) is left free, and the graph is checked on other sizes of
    those axes (:func:`_check_free_axes`) before it replaces what ``onnx_path`` held: a graph refused leaves it as it
    was."""
    onnx = _require("onnx")
    _require("onnxscript")  # required by the torch dynamo ONNX exporter
    from onnx_ir import serde  # shipped with the dynamo exporter
    from onnx_ir.passes.common import ClearMetadataAndDocStringPass

    # ``module`` takes its inputs as ``*inputs``: one free shape per example, under that one argument.
    free = (tuple(dict.fromkeys(range(free_from, example.dim()), torch.export.Dim.AUTO) for example in examples),)
    with torch.no_grad():
        program = cast(
            "torch.onnx.ONNXProgram",
            torch.onnx.export(
                module,
                examples,
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                dynamo=True,
                dynamic_shapes=free if dynamic else None,
                custom_translation_table={torch.ops.aten.scalar_tensor.default: _scalar_tensor},
            ),
        )
    ClearMetadataAndDocStringPass()(program.model)
    proto = serde.serialize_model(program.model)  # inline: ``program.save`` moves weights past 1.5 GiB beside it
    if proto.ByteSize() > ONE_FILE:
        raise PredictorError(
            f"The exported graph holds {proto.ByteSize() / 2**30:.2f} GiB, past the 2 GiB one ONNX file holds.",
            "Export a smaller network, or one part of it per head.",
        )
    # A folder of its own (another export to the destination stages beside it), the file made as any the user makes.
    staging = Path(tempfile.mkdtemp(prefix=f".{onnx_path.name}.", suffix=".partial", dir=onnx_path.parent))
    staged = staging / onnx_path.name
    try:
        onnx.save(proto, staged)
        if dynamic:
            # A dynamo export always carries the program it traced, guards and all.
            traced = cast("torch.export.ExportedProgram", program.exported_program)
            _check_free_axes(module, examples, staged, input_names, free_from, traced)
        staged.replace(owned_destination(onnx_path))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _smooth(shape: list[int], dtype: torch.dtype, free_from: int) -> torch.Tensor:
    """An input of ``shape`` that varies slowly: a product of cosines along the axes from ``free_from`` on, the same
    on the axes before. A model with hard decisions (an argmax between networks) gives ONNX and torch the same answer
    on it, as on an image."""
    value = torch.ones(())
    for axis, extent in enumerate(shape[free_from:]):
        line = torch.cos(torch.arange(extent, dtype=torch.float64) / (7.0 + 3.0 * axis))
        value = value.unsqueeze(-1) * line
    return value.expand(shape).to(dtype).contiguous()


def _traced_sizes(
    traced: torch.export.ExportedProgram, shapes: list[list[int]], free_from: int
) -> tuple[list[list[Any]], dict[Any, int]]:
    """Each axis of the inputs of ``shapes`` as the trace holds it, an expression of its own size symbols (a size of one
    input tied to another's is one symbol in both, ``2*s`` and ``s``), a free axis it fixed as a symbol of its own;
    and the value of each symbol at the example, as the trace holds it."""
    import sympy  # torch's own dependency, which it writes sizes in

    users = set(traced.graph_signature.user_inputs)
    held = [node.meta["val"].shape for node in traced.graph.nodes if node.op == "placeholder" and node.name in users]
    sizes: list[list[Any]] = []
    values: dict[Any, int] = {}
    for index, (shape, symbolic) in enumerate(zip(shapes, held, strict=True)):
        sizes.append([])
        for axis, (extent, size) in enumerate(zip(shape, symbolic, strict=True)):
            if axis < free_from:
                expr = sympy.Integer(extent)
            elif isinstance(size, torch.SymInt) and size.node.expr.free_symbols:
                expr = size.node.expr
                env: Any = size.node.shape_env  # torch 2.5 calls backed_var_to_val var_to_val, later ones deprecate it
                traced_values = env.backed_var_to_val if hasattr(env, "backed_var_to_val") else env.var_to_val
                values.update({symbol: int(traced_values[symbol]) for symbol in sorted(expr.free_symbols, key=str)})
            else:  # an int or a symbol of one value (torch 2.5): an axis the trace fixed
                expr = sympy.Symbol(f"fixed_{index}_{axis}")
                values[expr] = extent
            sizes[-1].append(expr)
    return sizes, values


def _check_free_axes(
    module: torch.nn.Module,
    examples: tuple[torch.Tensor, ...],
    onnx_path: Path,
    names: list[str],
    free_from: int,
    traced: torch.export.ExportedProgram,
) -> None:
    """Refuse a graph exported with the axes from ``free_from`` on free that answers for the example's shape alone.

    The shapes it is checked on set the sizes ``traced`` holds (:func:`_traced_sizes`), so a size tied to another
    stays tied. A size its guards keep out (a branch on the size, which the example took one way), or sizes it ties
    moved apart (a branch on two sizes equal), are refused where torch takes them: the graph would answer there with
    the branch it traced. Then, on a smooth input whose sizes are
    each shorter by the largest power of two up to 32 they hold, on one whose sizes are each longer by another
    multiple of it (a model of 32-voxel blocks takes both), and on two that reverse the order of every two sizes (a
    branch on which is the taller), the graph must give torch's shape, and torch's values as closely as on a smooth
    input of the example's own shape. An input torch refuses checks nothing, and torch must take one of them. The inputs are on the examples' device, the graph's on the host."""
    import sympy

    ort = _require("onnxruntime")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    hint = (
        "Compute what depends on the input's shape with symbolic sizes (torch.sym_max, no int()), or give the model a"
        " fixed patch_size."
    )

    def smooth(shapes: list[list[int]]) -> list[torch.Tensor]:
        return [_smooth(shape, x.dtype, free_from).to(x.device) for shape, x in zip(shapes, examples, strict=True)]

    def reference(inputs: list[torch.Tensor]) -> np.ndarray | None:
        """torch's answer on ``inputs``, ``None`` where it refuses them (a size its blocks or its poolings do not
        take), which says nothing of the graph."""
        try:
            with torch.no_grad():
                return module(*inputs).cpu().numpy()
        except Exception:
            return None

    def gap(inputs: list[torch.Tensor], want: np.ndarray) -> tuple[float | None, str]:
        """How far the graph lands from torch's ``want`` on ``inputs``, over their finite values once both hold the
        same nonfinite values at the same voxels (a maximum over a NaN is NaN, which no bound refuses); ``None`` and
        what differs otherwise."""
        try:
            got = session.run(None, {name: x.cpu().numpy() for name, x in zip(names, inputs, strict=True)})[0]
        except Exception as error:  # a graph that fixed an axis refuses the other shape outright
            return None, f"fails ({type(error).__name__})"
        if got.shape != want.shape:
            return None, f"gives {list(got.shape)}"
        finite = np.isfinite(want)
        if not np.array_equal(np.isfinite(got), finite) or not np.array_equal(
            got[~finite], want[~finite], equal_nan=True
        ):
            return None, "gives other nonfinite values"
        return float(np.abs(got[finite].astype(np.float64) - want[finite]).max(initial=0.0)), ""

    shapes = [list(example.shape) for example in examples]
    sizes, values = _traced_sizes(traced, shapes, free_from)

    def at(moved: dict[Any, int]) -> list[list[int]]:
        """The input shapes with the symbols of ``moved`` at those values, the others at the example's."""
        assigned = values | moved
        return [
            [
                int(size) if (size := expr.subs(assigned)).is_Integer else extent
                for expr, extent in zip(row, shape, strict=True)
            ]
            for row, shape in zip(sizes, shapes, strict=True)
        ]

    for symbol, bounds in traced.range_constraints.items():
        if symbol not in values:  # an expression of the symbols (2*s), which moves with them
            continue
        low = int(bounds.lower) if isinstance(bounds.lower, sympy.Integer) and bounds.lower > 2 else None
        high = int(bounds.upper) if isinstance(bounds.upper, sympy.Integer) else None
        if high is None:
            span = f"{low} voxels or more"
        else:
            span = f"at most {high} voxels" if low is None else f"{low} to {high} voxels"
        # A least size of 2 or less is the exporter's own floor, no branch.
        for size in ([low - 1] if low is not None else []) + ([high + 1] if high is not None else []):
            moved = at({symbol: size})
            if moved != shapes and reference(smooth(moved)) is not None:
                raise PredictorError(
                    f"The model exported with free spatial axes was traced for an axis of {span} alone, and torch"
                    f" takes inputs of {moved[0]} too: the forward answers for the example's shape alone.",
                    hint,
                )
    for symbol in values:
        # A size the trace holds on several axes (a branch on two sizes equal, or a tie torch needs itself): one of
        # them alone moved, each in turn: where torch takes it the graph would answer with the branch the tie traced.
        tied = [
            (index, axis)
            for index, row in enumerate(sizes)
            for axis, expr in enumerate(row)
            if symbol in expr.free_symbols
        ]
        if len(tied) < 2:
            continue
        for index, axis in tied:
            untied = [list(shape) for shape in shapes]
            untied[index][axis] += math.gcd(untied[index][axis], 32)
            if reference(smooth(untied)) is not None:
                raise PredictorError(
                    f"The model exported with free spatial axes was traced with sizes tied to one another, and torch"
                    f" takes inputs of {untied[index]} too, where they are not: the forward answers for the example's"
                    " shape alone.",
                    hint,
                )
    inputs = smooth(shapes)
    want = reference(inputs)
    own, answer = gap(inputs, want) if want is not None else (0.0, "")
    # Off torch at the example's own shape by more than its rounding is no tolerance for the others.
    if want is not None and (
        own is None or own > 1e-4 * max(1.0, float(np.abs(want[np.isfinite(want)]).max(initial=0.0)))
    ):
        answer = answer or f"lands {own:.3g} from torch"
        raise PredictorError(
            f"The model exported with free spatial axes {answer} at the example's own shape {shapes[0]}: the graph is"
            " not the forward.",
            "Make the forward the exporter traces the one torch runs (a branch on torch.compiler.is_compiling, an op"
            " the exporter translates otherwise).",
        )
    own = own or 0.0
    checked = False
    # Each size moved by a multiple its first axis's place in its input sets, so that no input grows with the others.
    spread = len(shapes[0]) - free_from
    place = {}
    for row in sizes:
        for axis, expr in enumerate(row[free_from:]):
            place.update({symbol: axis for symbol in sorted(expr.free_symbols, key=str) if symbol not in place})
    shorter = {symbol: max(1, value - math.gcd(value, 32)) for symbol, value in values.items()}
    longer = {symbol: value + math.gcd(value, 32) * (spread - place[symbol]) for symbol, value in values.items()}
    # The order of every two sizes reversed: those of one size at the example lengthened the other way round, the
    # others mirrored between the example's least and greatest.
    ascending = {symbol: value + math.gcd(value, 32) * (place[symbol] + 1) for symbol, value in values.items()}
    least, greatest = min(values.values(), default=0), max(values.values(), default=0)
    mirrored = {symbol: least + greatest - value for symbol, value in values.items()}
    probes = (at(shorter), at(longer), at(ascending), at(mirrored))
    # An axis of one voxel is no shorter: the example's own shape checks nothing of another.
    for other in (shape for shape in probes if shape != shapes):
        inputs = smooth(other)
        want = reference(inputs)
        if want is None:
            continue
        checked = True
        there, answer = gap(inputs, want)
        if there is None or there > max(1e-4, 10.0 * own):
            answer = answer or f"lands {there:.3g} from torch where it lands {own:.3g} at the example's shape"
            raise PredictorError(
                f"The model exported with free spatial axes {answer} on inputs of {other[0]}, where torch gives"
                f" {list(want.shape)}: the forward answers for the example's shape alone.",
                hint,
            )
    if not checked:
        raise PredictorError(
            "Of the inputs the model exported with free spatial axes is checked on, torch takes none but the"
            f" example's {shapes[0]}: nothing says the graph holds at another shape.",
            "Give the model a fixed patch_size.",
        )


class _Folds(torch.nn.Module):
    """Pointwise transforms of an output chain, in order, on ``[1, C, ...]`` in the ``dtype`` konfai hands them:
    what the last hands back, as the float32 the program holds every buffer in."""

    def __init__(self, folds: list[Callable[[torch.Tensor], torch.Tensor]], dtype: torch.dtype) -> None:
        super().__init__()
        self.folds = list(folds)
        self.dtype = dtype

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        # Taken as ``*inputs``, as every graph ``_write_onnx`` frees the axes of.
        out = inputs[0][0].to(self.dtype)
        for fold in self.folds:
            out = fold(out)
        return out[None].to(torch.float32)


def export_folds(
    folds: list[Callable[[torch.Tensor], torch.Tensor]],
    onnx_path: Path,
    channels: int,
    rank: int,
    dtype: str = "float32",
    opset: int = 18,
) -> tuple[int, str]:
    """Export pointwise transforms of an output chain to one ONNX graph at ``onnx_path``, its input ``C =
    channels`` in the ``dtype`` konfai hands them and its ``rank`` spatial axes free; return the channel count and
    the dtype konfai's transforms hand back."""
    example = torch.randn(1, channels, *([8] * rank))
    with torch.no_grad():
        out = example[0].to(getattr(torch, dtype))
        for fold in folds:
            out = fold(out)
    module = _Folds(folds, getattr(torch, dtype)).eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    _write_onnx(module, (example,), onnx_path, ["input"], ["output"], opset, dynamic=True)
    return int(out.shape[0]), str(out.dtype).removeprefix("torch.")


class _Function(torch.nn.Module):
    """A function of tensors as a module, its inputs taken as ``*inputs``, as every graph ``_write_onnx`` frees."""

    def __init__(self, function: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.function(*inputs)


def export_case_statistic(statistic: Any, partial_path: Path, finish_path: Path, rank: int, opset: int = 18) -> None:
    """Export the class a ``CaseStatistic`` names as two float64 graphs: ``partial_path`` takes ``block`` ``[1,
    *spatial]``, one channel (the compiler refuses a group of more), its ``rank`` spatial axes free, to ``partial``
    ``[K]``; ``finish_path`` takes ``total`` ``[K]`` to the 0-d ``statistic``."""
    # A region of 8 voxels and the halo on both sides, as the streamed fold hands it.
    block = torch.rand(1, *([8 + 2 * int(statistic.halo)] * rank), dtype=torch.float64)
    partial = _Function(statistic.partial).eval()
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    _write_onnx(partial, (block,), partial_path, ["block"], ["partial"], opset, dynamic=True, free_from=1)
    with torch.no_grad():
        total = partial(block)
    _write_onnx(_Function(statistic.finish).eval(), (total,), finish_path, ["total"], ["statistic"], opset)


def export_to_onnx(
    model: torch.nn.Module,
    output_dir: str | Path,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    output_module: str | None = None,
    *,
    opset: int = 18,
    input_name: str = "input",
    output_name: str = "output",
    input_group: str | list[str] = "Volume_0",
    output_group: str = "output",
    patch_overlap: list[int] | None = None,
    extend_slice: int = 0,
    pad_value: float | str | None = None,
    placement: Placement = "pad",
    patch_stride: list[float] | None = None,
    fold_pre: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    fold_post: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    extra_manifest: dict[str, Any] | None = None,
    model_filename: str = "model.onnx",
    write_manifest: bool = True,
    dynamic: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Export ``model`` to ``output_dir/model.onnx`` (+ ``manifest.json``).

    Parameters
    ----------
    model:
        A frozen KonfAI ``Network`` (or any module exposing ``named_forward``).
    output_dir:
        Directory to write ``model.onnx`` and ``manifest.json`` into.
    example_input:
        A fixed-shape example patch ``[N, C, (Z), Y, X]``, or one per group the model reads, in the
        order its graph takes them; the ONNX is exported at these exact shapes (no dynamic axes). When a
        ModelPatch of the graph cuts it into several patches, an export whose head would see only the last
        one is refused.
    input_group:
        The group the first example holds, or one group per example. The manifest's ``input`` names the first;
        each later one is a ``reads`` entry, in graph order, unless ``extra_manifest`` states its own ``reads``.
    output_module:
        Dotted name of the inference head to export (see :func:`list_output_modules`).
        When omitted, :func:`select_inference_head` picks the terminal floating-point head.
    patch_overlap:
        Per-axis voxel overlap. None retains KonfAI's case-dependent fixed-patch
        overlap and last-patch assignment. An explicit zero list means disjoint patches.
    placement:
        The ``Patch.placement`` the config states, written into the manifest's patch.
    patch_stride:
        Per-axis unrounded step between windows under placements ``end`` and ``spread``, which ``patch_overlap``
        rounds; written into the manifest's patch when given.
    opset:
        ONNX opset (>= 18 recommended; the dynamo exporter implements 18).
    model_filename:
        ONNX file name to write (a multi-model ensemble exports one file per fold).
    write_manifest:
        Write ``manifest.json`` beside the ONNX. Off for a multi-model bundle, whose
        per-fold manifests are embedded in a single ``program.json`` instead.
    dynamic:
        Leave every spatial axis free: the graph takes the whole case as one window, which the manifest's
        ``patch.size`` of zeros says. The graph is checked on inputs of another shape before it is kept.

    Returns
    -------
    ``(onnx_path, manifest)``: the written ONNX path and the manifest dict (also written
    to ``manifest.json`` when ``write_manifest``).
    """
    if placement not in get_args(Placement):
        raise PredictorError(f"placement '{placement}' is none of {', '.join(get_args(Placement))}.")
    if isinstance(pad_value, str) and pad_value not in ("edge", "min"):
        raise PredictorError(f"pad_value '{pad_value}' is none of 'edge', 'min' or a number.")
    out_dir = Path(output_dir)
    onnx_path = out_dir / model_filename

    model.eval()
    examples = example_input if isinstance(example_input, tuple) else (example_input,)
    groups = [input_group] if isinstance(input_group, str) else list(input_group)
    if len(groups) != len(examples):
        raise PredictorError(
            f"{len(examples)} example inputs, {len(groups)} input group(s): give one group per example."
        )
    if output_module is None:
        output_module = select_inference_head(model, *examples)
    available = list_output_modules(model, *examples)
    matches = [shape for name, shape in available if name == output_module]
    if not matches:
        names = [name for name, _ in available]
        raise PredictorError(
            f"output_module '{output_module}' not found in the graph. Available outputs (last few): {names[-8:]}",
        )
    _refuse_one_patch_of_several(model, available, output_module)
    output_shape = matches[-1]
    if fold_post:
        # What the graph ends on past the head: its channels, read off one voxel.
        voxel = torch.zeros(1, int(output_shape[1]), *[1] * (len(output_shape) - 2))
        for post in fold_post:
            voxel = post(voxel)
        output_shape = (output_shape[0], int(voxel.shape[1]), *output_shape[2:])
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = _NamedHead(model, output_module, fold_pre=fold_pre, fold_post=fold_post).eval()
    names = [input_name] + [f"{input_name}_{at}" for at in range(1, len(examples))]
    _write_onnx(wrapper, examples, onnx_path, names, [output_name], opset, dynamic)

    input_shape = tuple(int(s) for s in examples[0].shape)
    spatial = list(input_shape[2:])
    dim = len(spatial)
    manifest: dict[str, Any] = {
        "konfai_edge_manifest": MANIFEST_VERSION,
        "model": onnx_path.name,
        "opset": opset,
        "output_module": output_module,
        "input": {"name": input_name, "group": groups[0], "channels": input_shape[1], "dtype": "f32"},
        "output": {"name": output_name, "group": output_group, "channels": int(output_shape[1]), "dtype": "f32"},
        "patch": {
            # Zeros: one window per case, the whole of it.
            "size": [0] * len(spatial) if dynamic else spatial,
            "overlap": patch_overlap,
            "dim": dim,
            "extend_slice": extend_slice,
            # What pads a patch past the volume (after it under placement pad, on both sides otherwise): 0 unless
            # the config pins a value, "edge" for the volume's edge voxel, or "min" for the minimum of the window read.
            "pad_value": pad_value if isinstance(pad_value, str) else float(pad_value or 0.0),
            # Where the windows of an axis sit, konfai's Patch.placement: pad, end or spread
            # (konfai.utils.utils.get_patch_slices_from_shape). Under end a Gaussian blend's weight is floored at
            # its min_weight, 1e-3.
            "placement": placement,
        },
        "geometry": {"mode": "preserve_from_input"},
    }
    if patch_stride is not None:
        manifest["patch"]["stride"] = patch_stride
    if len(examples) > 1:
        manifest["reads"] = [
            {"group": group, "channels": int(example.shape[1]), "preprocessing": []}
            for group, example in zip(groups[1:], examples[1:], strict=True)
        ]
    if extra_manifest:
        manifest.update(extra_manifest)
    if patch_overlap is None:
        manifest.setdefault("blend", "Overwrite")
    # A patch combine of the app's extension is named by its classpath.
    if manifest.get("blend") == "GaussianFloat16" or ":" in manifest.get("blend", ""):
        manifest["konfai_edge_manifest"] = FLOAT16_BLEND_VERSION
    if write_manifest:
        owned_destination(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return onnx_path, manifest
