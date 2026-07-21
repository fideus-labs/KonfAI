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

"""ONNX export parity (onnxruntime vs torch) across the whole KonfAI YAML catalog."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
CATALOG_MODELS = sorted(CATALOG.glob("*.yml"))


def _example_input(params: dict) -> torch.Tensor:
    """A small fixed-shape patch matching the model's declared dim / channels."""
    dim = int(params.get("dim", 2))
    channels = params.get("channels")
    in_channels = params.get("in_channels") or (channels[0] if isinstance(channels, list) and channels else 1)
    patch_size = params.get("patch_size")
    size = patch_size * 2 if patch_size else (64 if dim == 2 else 48)
    return torch.randn(1, int(in_channels), *([size] * dim))


@pytest.fixture(autouse=True)
def _config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_config_file", str(tmp_path / "config.yml"))


@pytest.mark.slow
@pytest.mark.parametrize("yml", CATALOG_MODELS, ids=lambda p: p.stem)
def test_catalog_model_exports_with_parity(yml, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import _NamedHead, export_to_onnx, select_inference_head
    from konfai.utils.model_builder import build_model_from_yaml

    params = yaml.safe_load(yml.read_text()).get("parameters", {}) or {}
    model = build_model_from_yaml(yaml_path=str(yml)).eval()
    example = _example_input(params)

    head = select_inference_head(model, example)
    assert "argmax" not in head.lower(), f"{yml.stem}: exported an integer label head {head!r}"

    onnx_path, _ = export_to_onnx(model, tmp_path, example)  # output_module auto-selected
    assert onnx_path.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["output_module"] == head
    assert manifest["patch"]["dim"] == int(params.get("dim", 2))

    with torch.no_grad():
        reference = _NamedHead(model, head)(example).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"input": example.numpy().astype(np.float32)})[0]

    assert produced.shape == reference.shape
    assert float(np.mean(np.abs(produced - reference))) < 1e-4


def test_explicit_head_overrides_auto_selection(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from konfai.export import export_to_onnx, list_output_modules
    from konfai.models.python.segmentation.UNet import UNet

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    example = torch.randn(1, 1, 64, 64)

    heads = [name for name, _ in list_output_modules(model, example)]
    head = "UNetBlock_0.Head.Softmax"
    assert head in heads, f"expected full-res head among {heads[-5:]}"

    export_to_onnx(model, tmp_path, example, head, opset=18)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["output_module"] == head


def test_fold_pre_bakes_a_custom_pointwise_op_into_the_graph(tmp_path):
    # A custom pointwise transform the runtime has no primitive for is folded into the ONNX graph.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import _NamedHead, export_to_onnx, select_inference_head
    from konfai.models.python.segmentation.UNet import UNet

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    example = torch.randn(1, 1, 64, 64)

    def custom(t):  # an arbitrary pointwise op, outside the curated op registry
        return torch.clamp(t, -0.5, 0.5) * 2.0 + 1.0

    head = select_inference_head(model, example)
    onnx_path, _ = export_to_onnx(model, tmp_path, example, head, fold_pre=[custom])

    with torch.no_grad():
        reference = _NamedHead(model, head, fold_pre=[custom])(example).numpy()  # custom -> model head
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"input": example.numpy().astype(np.float32)})[0]

    assert produced.shape == reference.shape
    assert float(np.mean(np.abs(produced - reference))) < 1e-4  # the custom op is baked in the graph


def test_export_unknown_head_raises(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from konfai.export import export_to_onnx
    from konfai.models.python.segmentation.UNet import UNet
    from konfai.utils.errors import PredictorError

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    with pytest.raises(PredictorError):
        export_to_onnx(model, tmp_path, torch.randn(1, 1, 64, 64), "Does.Not.Exist")
