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
from collections import Counter
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np
import torch

from konfai.utils.errors import PredictorError
from konfai.utils.utils import Placement

if TYPE_CHECKING:
    from konfai.network.network.network import ModuleArgsDict

MANIFEST_VERSION = 5  # 5: patch.stride under the end and spread placements
#: A manifest blending by GaussianFloat16: a runtime reading version 5 alone refuses it rather than blend it as Cosinus.
FLOAT16_BLEND_VERSION = 6


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
    input *inside* the graph, so they trace into the ONNX; only pointwise ops are foldable.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        output_module: str,
        fold_pre: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.net = _routed(net)
        self.output_module = output_module
        self.fold_pre = list(fold_pre or [])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        # The folds prepare the group the chain belongs to, which is the model's first input.
        first, *rest = inputs
        for pre in self.fold_pre:
            first = pre(first)
        out = first
        for name, tensor in self.net.named_forward(first, *rest):
            if name == self.output_module:
                out = tensor
        return out


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


def _write_onnx(
    module: torch.nn.Module,
    examples: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int = 18,
    dynamic: bool = False,
) -> None:
    """Write ``module``, traced on ``examples``, as one self-contained ONNX file at ``onnx_path``: the weights go
    inline. The exporter records every node's Python stack, and with it the paths of the machine that exported,
    which a model handed to others does not carry. ``dynamic``: every spatial axis of every input is left free."""
    _require("onnx")
    _require("onnxscript")  # required by the torch dynamo ONNX exporter
    from onnx_ir.passes.common import ClearMetadataAndDocStringPass  # shipped with the dynamo exporter

    # ``module`` takes its inputs as ``*inputs``: one free shape per example, under that one argument.
    free = (tuple(dict.fromkeys(range(2, example.dim()), torch.export.Dim.AUTO) for example in examples),)
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
            ),
        )
    ClearMetadataAndDocStringPass()(program.model)
    program.save(onnx_path, external_data=False)


def _smooth(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    """An input of ``shape`` that varies slowly: a product of cosines along the spatial axes, on every channel. A model
    with hard decisions (an argmax between networks) gives ONNX and torch the same answer on it, as on an image."""
    value = torch.ones(())
    for axis, extent in enumerate(shape[2:]):
        line = torch.cos(torch.arange(extent, dtype=torch.float64) / (7.0 + 3.0 * axis))
        value = value.unsqueeze(-1) * line
    return value.expand(shape).to(dtype).contiguous()


def _check_free_axes(
    module: torch.nn.Module, examples: tuple[torch.Tensor, ...], onnx_path: Path, names: list[str]
) -> None:
    """Refuse a graph exported with free axes that answers for the example's shape alone: on a smooth input of
    another shape (32 voxels more along the first axis, 16 along the last) it must give torch's shape, and torch's
    values as closely as on a smooth input of the example's own shape."""
    ort = _require("onnxruntime")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def gap(inputs: list[torch.Tensor]) -> tuple[float | None, list[int], str]:
        """How far ONNX lands from torch on ``inputs``, ``None`` when it gives another shape or fails."""
        with torch.no_grad():
            want = module(*inputs).numpy()
        try:
            got = session.run(None, {name: tensor.numpy() for name, tensor in zip(names, inputs, strict=True)})[0]
        except Exception as error:  # a graph that fixed an axis refuses the other shape outright
            return None, list(want.shape), f"fails ({type(error).__name__})"
        if got.shape != want.shape:
            return None, list(want.shape), f"gives {list(got.shape)}"
        return float(np.abs(got.astype(np.float64) - want).max(initial=0.0)), list(want.shape), ""

    own, _, _ = gap([_smooth(list(example.shape), example.dtype) for example in examples])
    other = []
    for example in examples:
        shape = list(example.shape)
        shape[2] += 32
        shape[-1] += 16
        other.append(_smooth(shape, example.dtype))
    there, shape, answer = gap(other)
    own = 0.0 if own is None else own
    if there is None or there > max(1e-4, 10.0 * own):
        answer = answer or f"lands {there:.3g} from torch where it lands {own:.3g} at the example's shape"
        raise PredictorError(
            f"The model exported with free spatial axes {answer} on inputs of {list(other[0].shape)}, where torch"
            f" gives {shape}: the forward answers for the example's shape alone.",
            "Compute what depends on the input's shape with symbolic sizes (torch.sym_max, no int()), or give the"
            " model a fixed patch_size.",
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
    _check_free_axes(module, (example,), onnx_path, ["input"])
    return int(out.shape[0]), str(out.dtype).removeprefix("torch.")


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
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = _NamedHead(model, output_module, fold_pre=fold_pre).eval()
    names = [input_name] + [f"{input_name}_{at}" for at in range(1, len(examples))]
    _write_onnx(wrapper, examples, onnx_path, names, [output_name], opset, dynamic)
    if dynamic:
        _check_free_axes(wrapper, examples, onnx_path, names)

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
    if manifest.get("blend") == "GaussianFloat16":
        manifest["konfai_edge_manifest"] = FLOAT16_BLEND_VERSION
    if write_manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return onnx_path, manifest
