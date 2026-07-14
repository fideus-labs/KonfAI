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

"""ONNX export parity test (onnxruntime vs torch) on the example UNet."""

import json

import numpy as np
import pytest
import torch


def test_export_to_onnx_parity(tmp_path, monkeypatch):
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_config_file", str(tmp_path / "config.yml"))

    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import _NamedHead, export_to_onnx, list_output_modules
    from konfai.models.python.segmentation.UNet import UNet

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    example = torch.randn(1, 1, 64, 64)

    heads = [name for name, _ in list_output_modules(model, example)]
    head = "UNetBlock_0.Head.Softmax"
    assert head in heads, f"expected full-res head among {heads[-5:]}"

    onnx_path = export_to_onnx(model, tmp_path, example, head, opset=18)
    assert onnx_path.exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["konfai_rs_manifest"] == 1
    assert manifest["patch"]["size"] == [64, 64]
    assert manifest["patch"]["dim"] == 2
    assert manifest["input"]["channels"] == 1
    assert manifest["output"]["channels"] == 2
    assert manifest["output_module"] == head

    with torch.no_grad():
        reference = _NamedHead(model, head)(example).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"input": example.numpy().astype(np.float32)})[0]

    assert produced.shape == reference.shape
    assert float(np.mean(np.abs(produced - reference))) < 1e-4


def test_export_unknown_head_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_config_file", str(tmp_path / "config.yml"))
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from konfai.export import export_to_onnx
    from konfai.models.python.segmentation.UNet import UNet
    from konfai.utils.errors import PredictorError

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    with pytest.raises(PredictorError):
        export_to_onnx(model, tmp_path, torch.randn(1, 1, 64, 64), "Does.Not.Exist")
