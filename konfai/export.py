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

This is the producer side of the ``konfai-rs`` portable-inference contract: a
trained KonfAI model becomes ``model.onnx`` (graph + weights, single file) plus
``manifest.json`` (patch geometry, input/output spec) that a no-Python runtime
consumes. The non-obvious steps the ``torch -> ONNX -> burn-onnx`` chain requires
are encoded here:

* KonfAI ``Network`` overrides ``state_dict()`` with a custom signature that breaks
  the TorchScript exporter, so the **dynamo** exporter is used.
* ``Network.forward`` returns per-output-group results (empty without ``init()``),
  so the graph is reached via ``named_forward`` and a named head is selected.
* The dynamo exporter writes weights as external data; they are inlined so the
  ``.onnx`` is a single self-contained file (required by ``burn-onnx``).
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from konfai.utils.errors import PredictorError

MANIFEST_VERSION = 1


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

    ``Network.forward`` yields per-output-group results (deep-supervision aware);
    for export we want one specific head. ``named_forward`` exposes every module
    output as ``(dotted_name, tensor)``; this returns the tensor of ``output_module``.
    """

    def __init__(self, net: torch.nn.Module, output_module: str) -> None:
        super().__init__()
        self.net = net
        self.output_module = output_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for name, tensor in self.net.named_forward(x):
            if name == self.output_module:
                out = tensor
        return out


def list_output_modules(model: torch.nn.Module, example_input: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(dotted_name, shape)`` for every output of the routed graph.

    Use this to discover the inference head to export (e.g. ``UNetBlock_0.Head.Softmax``
    or ``Head.Tanh``), skipping deep-supervision heads and integer ``Argmax`` outputs.
    """
    if not hasattr(model, "named_forward"):
        raise PredictorError("export expects a KonfAI Network exposing `named_forward`.")
    model.eval()
    outputs: list[tuple[str, tuple[int, ...]]] = []
    with torch.no_grad():
        for name, tensor in model.named_forward(example_input):
            outputs.append((name, tuple(tensor.shape)))
    return outputs


def export_to_onnx(
    model: torch.nn.Module,
    output_dir: str | Path,
    example_input: torch.Tensor,
    output_module: str,
    *,
    opset: int = 18,
    input_name: str = "input",
    output_name: str = "output",
    input_group: str = "Volume_0",
    output_group: str = "output",
    patch_overlap: list[int] | None = None,
    extend_slice: int = 0,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """Export ``model`` to ``output_dir/model.onnx`` (+ ``manifest.json``).

    Parameters
    ----------
    model:
        A frozen KonfAI ``Network`` (or any module exposing ``named_forward``).
    output_dir:
        Directory to write ``model.onnx`` and ``manifest.json`` into.
    example_input:
        A fixed-shape example patch ``[N, C, (Z), Y, X]``; the ONNX is exported at
        this exact shape (burn-onnx dislikes dynamic shapes).
    output_module:
        Dotted name of the inference head to export (see :func:`list_output_modules`).
    opset:
        ONNX opset (>= 18 recommended; the dynamo exporter implements 18).

    Returns
    -------
    Path to the written ``model.onnx``.
    """
    onnx = _require("onnx")
    _require("onnxscript")  # required by the torch dynamo ONNX exporter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    model.eval()
    available = list_output_modules(model, example_input)
    matches = [shape for name, shape in available if name == output_module]
    if not matches:
        names = [name for name, _ in available]
        raise PredictorError(
            f"output_module '{output_module}' not found in the graph. Available outputs (last few): {names[-8:]}",
        )
    output_shape = matches[-1]

    wrapper = _NamedHead(model, output_module).eval()
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

    # The dynamo exporter writes weights as external data; inline them so the
    # .onnx is a single self-contained file (required by burn-onnx's ModelGen and
    # simpler to serve to ONNX-Runtime-Web). Remove the now-orphan sidecar.
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
        },
        "geometry": "preserve_from_input",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return onnx_path
