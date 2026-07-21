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
(patch geometry, input/output spec) for a Python-free runtime. Three constraints shape the code:

* KonfAI ``Network`` overrides ``state_dict()`` with a custom signature that breaks the
  TorchScript exporter, so the **dynamo** exporter is used.
* ``Network.forward`` returns per-output-group results (empty without ``init()``), so the
  graph is reached via ``named_forward`` and a named head is selected.
* The dynamo exporter writes weights as external data; they are inlined so the ``.onnx`` is
  a single self-contained file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from konfai.utils.errors import PredictorError

MANIFEST_VERSION = 2


def _require(module: str) -> ModuleType:
    """Import an optional export dependency or raise an actionable error."""
    try:
        return import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise PredictorError(
            f"ONNX export needs the optional dependency '{module}'. Install it with `pip install konfai[export]`.",
        ) from exc


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
        self.net = net
        self.output_module = output_module
        self.fold_pre = list(fold_pre or [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for pre in self.fold_pre:
            x = pre(x)
        out = x
        for name, tensor in self.net.named_forward(x):
            if name == self.output_module:
                out = tensor
        return out


def list_output_modules(model: torch.nn.Module, example_input: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(dotted_name, shape)`` for every output of the routed graph, to discover the
    inference head to export."""
    if not hasattr(model, "named_forward"):
        raise PredictorError("export expects a KonfAI Network exposing `named_forward`.")
    model.eval()
    outputs: list[tuple[str, tuple[int, ...]]] = []
    with torch.no_grad():
        for name, tensor in model.named_forward(example_input):
            outputs.append((name, tuple(tensor.shape)))
    return outputs


def select_inference_head(model: torch.nn.Module, example_input: torch.Tensor) -> str:
    """Pick the head to export: the last **floating-point** output in execution order.

    Terminal outputs often end in an integer ``Argmax`` label map; a float runtime wants the final
    probability/regression head, so integer outputs are skipped. Pass ``output_module`` to target a
    specific head.
    """
    if not hasattr(model, "named_forward"):
        raise PredictorError("export expects a KonfAI Network exposing `named_forward`.")
    model.eval()
    head: str | None = None
    with torch.no_grad():
        for name, tensor in model.named_forward(example_input):
            if torch.is_floating_point(tensor):
                head = name
    if head is None:
        raise PredictorError("no floating-point output to export; pass an explicit output_module.")
    return head


def export_to_onnx(
    model: torch.nn.Module,
    output_dir: str | Path,
    example_input: torch.Tensor,
    output_module: str | None = None,
    *,
    opset: int = 18,
    input_name: str = "input",
    output_name: str = "output",
    input_group: str = "Volume_0",
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
        A fixed-shape example patch ``[N, C, (Z), Y, X]``; the ONNX is exported at
        this exact shape (no dynamic axes).
    output_module:
        Dotted name of the inference head to export (see :func:`list_output_modules`).
        When omitted, :func:`select_inference_head` picks the terminal floating-point head.
    opset:
        ONNX opset (>= 18 recommended; the dynamo exporter implements 18).
    model_filename:
        ONNX file name to write (a multi-model ensemble exports one file per fold).
    write_manifest:
        Write ``manifest.json`` beside the ONNX. Off for a multi-model bundle, whose
        per-fold manifests are embedded in a single ``program.json`` instead.

    Returns
    -------
    ``(onnx_path, manifest)`` -- the written ONNX path and the manifest dict (also written
    to ``manifest.json`` when ``write_manifest``).
    """
    onnx = _require("onnx")
    _require("onnxscript")  # required by the torch dynamo ONNX exporter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / model_filename

    model.eval()
    if output_module is None:
        output_module = select_inference_head(model, example_input)
    available = list_output_modules(model, example_input)
    matches = [shape for name, shape in available if name == output_module]
    if not matches:
        names = [name for name, _ in available]
        raise PredictorError(
            f"output_module '{output_module}' not found in the graph. Available outputs (last few): {names[-8:]}",
        )
    output_shape = matches[-1]

    wrapper = _NamedHead(model, output_module, fold_pre=fold_pre).eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input,
            str(onnx_path),
            opset_version=opset,
            input_names=[input_name],
            output_names=[output_name],
            dynamo=True,
        )

    # The dynamo exporter writes weights as external data; inline them so the .onnx is a
    # single self-contained file. Remove the now-orphan sidecar.
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    input_shape = tuple(int(s) for s in example_input.shape)
    spatial = list(input_shape[2:])
    dim = len(spatial)
    manifest: dict[str, Any] = {
        "konfai_rs_manifest": MANIFEST_VERSION,
        "model": onnx_path.name,
        "opset": opset,
        "output_module": output_module,
        "input": {"name": input_name, "group": input_group, "channels": input_shape[1], "dtype": "f32"},
        "output": {"name": output_name, "group": output_group, "channels": int(output_shape[1]), "dtype": "f32"},
        "patch": {
            "size": spatial,
            "overlap": patch_overlap if patch_overlap is not None else [0] * dim,
            "dim": dim,
            "extend_slice": extend_slice,
            # The border-pad value for the last (ragged) patch of an axis; 0 unless the config pins one.
            "pad_value": float(pad_value) if pad_value is not None else 0.0,
        },
        "geometry": {"mode": "preserve_from_input"},
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    if write_manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return onnx_path, manifest
