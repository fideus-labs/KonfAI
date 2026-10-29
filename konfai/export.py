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
from typing import TYPE_CHECKING, Any, cast

import torch

from konfai.utils.errors import PredictorError

if TYPE_CHECKING:
    from konfai.network.network.network import ModuleArgsDict

MANIFEST_VERSION = 4


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
    pad_value: float | None = None,
    fold_pre: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    extra_manifest: dict[str, Any] | None = None,
    model_filename: str = "model.onnx",
    write_manifest: bool = True,
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
    opset:
        ONNX opset (>= 18 recommended; the dynamo exporter implements 18).
    model_filename:
        ONNX file name to write (a multi-model ensemble exports one file per fold).
    write_manifest:
        Write ``manifest.json`` beside the ONNX. Off for a multi-model bundle, whose
        per-fold manifests are embedded in a single ``program.json`` instead.

    Returns
    -------
    ``(onnx_path, manifest)``: the written ONNX path and the manifest dict (also written
    to ``manifest.json`` when ``write_manifest``).
    """
    onnx = _require("onnx")
    _require("onnxscript")  # required by the torch dynamo ONNX exporter

    out_dir = Path(output_dir)
    onnx_path = out_dir / model_filename

    model.eval()
    examples = example_input if isinstance(example_input, tuple) else (example_input,)
    groups = [input_group] if isinstance(input_group, str) else list(input_group)
    # Each input past the first is bound to its group by a `reads` entry, in graph order: the caller's (which may
    # also name companions past them), else one per group past the first.
    stated = (extra_manifest or {}).get("reads")
    if stated is None:
        covered = len(groups) == len(examples)
    else:
        covered = len(groups) in (1, len(examples)) and len(stated) >= len(examples) - 1
    if not covered:
        raise PredictorError(
            f"{len(examples)} example inputs, {len(groups)} input group(s), {len(stated or [])} stated reads:"
            " give one group per example, or a read per input past the first."
        )
    read_groups = groups[1:] if stated is None else []
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
    with torch.no_grad():
        names = [input_name] + [f"{input_name}_{at}" for at in range(1, len(examples))]
        torch.onnx.export(
            wrapper,
            examples,
            str(onnx_path),
            opset_version=opset,
            input_names=names,
            output_names=[output_name],
            dynamo=True,
        )

    # The dynamo exporter writes weights as external data; inline them so the .onnx is a
    # single self-contained file. It also records every node's Python stack, and with it the
    # paths of the machine that exported, which a model handed to others does not carry.
    exported = onnx.load(str(onnx_path))
    for node in [*exported.graph.node, *(node for function in exported.functions for node in function.node)]:
        kept = [prop for prop in node.metadata_props if prop.key != "pkg.torch.onnx.stack_trace"]
        del node.metadata_props[:]
        node.metadata_props.extend(kept)
    onnx.save(exported, str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

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
            "size": spatial,
            "overlap": patch_overlap,
            "dim": dim,
            "extend_slice": extend_slice,
            # The border-pad value for the last (ragged) patch of an axis; 0 unless the config pins one.
            "pad_value": float(pad_value) if pad_value is not None else 0.0,
        },
        "geometry": {"mode": "preserve_from_input"},
    }
    if read_groups:
        manifest["reads"] = [
            {"group": group, "channels": int(example.shape[1]), "preprocessing": []}
            for group, example in zip(read_groups, examples[1:], strict=True)
        ]
    if extra_manifest:
        manifest.update(extra_manifest)
    if patch_overlap is None:
        manifest.setdefault("blend", "Overwrite")
    if write_manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return onnx_path, manifest
