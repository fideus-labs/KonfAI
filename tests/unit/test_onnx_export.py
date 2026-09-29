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


def test_any_module_exposing_named_forward_exports(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import export_to_onnx

    class Plain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)

        def named_forward(self, x):
            y = self.conv(x)
            yield "conv", y
            yield "act", torch.tanh(y)

    model = Plain().eval()
    example = torch.randn(1, 1, 32, 32)
    onnx_path, manifest = export_to_onnx(model, tmp_path, example)

    assert manifest["output_module"] == "act"
    produced = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"input": example.numpy()}
    )[0]
    with torch.no_grad():
        np.testing.assert_allclose(produced, torch.tanh(model.conv(example)).numpy(), atol=1e-5)


def test_export_unknown_head_raises(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from konfai.export import export_to_onnx
    from konfai.models.python.segmentation.UNet import UNet
    from konfai.utils.errors import PredictorError

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    with pytest.raises(PredictorError):
        export_to_onnx(model, tmp_path, torch.randn(1, 1, 64, 64), "Does.Not.Exist")


def test_a_model_patch_cutting_the_example_into_several_patches_is_refused(tmp_path):
    # Each output of a patched pass is yielded once per patch; the export would keep the last patch
    # while the manifest declares the whole example.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.data.patching import ModelPatch
    from konfai.export import export_to_onnx
    from konfai.models.python.segmentation.UNet import UNet
    from konfai.utils.errors import PredictorError

    torch.manual_seed(0)
    plain = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    patched = UNet(dim=2, channels=[1, 8, 16], nb_class=2, patch=ModelPatch([16, 16], overlap=0)).eval()
    patched.load_state_dict(plain.state_dict())

    with pytest.raises(PredictorError, match="ModelPatch"):
        export_to_onnx(patched, tmp_path / "several", torch.randn(1, 1, 32, 32))

    # One patch covering the example: the export holds the whole pass, as the plain network computes it.
    example = torch.randn(1, 1, 16, 16)
    onnx_path, _ = export_to_onnx(patched, tmp_path / "one", example)
    produced = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"input": example.numpy()}
    )[0]
    with torch.no_grad():
        expected = dict(plain.named_forward(example))["UNetBlock_0.Head.Softmax"].numpy()
    np.testing.assert_allclose(produced, expected, atol=1e-5)


def _nested_patch_network(side_branch: bool = False, gated: bool = False):
    """A root network holding ``Inner``, whose ModelPatch cuts a 32x32 example into four 16x16 patches,
    then ``Post``, which reads Inner or, as a side branch Inner writes, the input beside it. The 1x1
    convolutions make the assembled pass equal the unpatched one. ``gated`` skips Act, the module whose
    patches Inner assembles once bound, outside prediction."""
    from konfai.data.patching import ModelPatch
    from konfai.network.network import Network

    class Inner(Network):
        def __init__(self) -> None:
            super().__init__(in_channels=1, dim=2, patch=ModelPatch([16, 16], overlap=0))
            self.add_module("Conv", torch.nn.Conv2d(1, 2, 1))
            self.add_module("Act", torch.nn.Tanh(), training=False if gated else None)

    class Outer(Network):
        def __init__(self) -> None:
            super().__init__(in_channels=1, dim=2)
            self.add_module("Inner", Inner(), out_branch=["aux"] if side_branch else [0])
            self.add_module("Post", torch.nn.Conv2d(1 if side_branch else 2, 2, 1))

    torch.manual_seed(0)
    return Outer().eval()


def test_a_nested_model_patch_is_refused_only_when_the_head_sees_one_patch(tmp_path, monkeypatch):
    # A nested network assembles its patches once the model is bound (Trainer, Predictor, train_model
    # bind it); unbound, it hands its last patch to what reads it.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import export_to_onnx
    from konfai.network.network import NetState
    from konfai.utils.errors import PredictorError
    from konfai.utils.runtime import State

    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    example = torch.randn(1, 1, 32, 32)

    with pytest.raises(PredictorError, match=r"ModelPatch of Inner .* head 'Post'"):
        export_to_onnx(_nested_patch_network(), tmp_path / "unbound", example)
    assert not (tmp_path / "unbound").exists()

    gated = _nested_patch_network(gated=True)
    gated.bind(False, State.PREDICTION, ["x"])
    gated.set_state(NetState.TRAIN)
    with pytest.raises(PredictorError, match=r"ModelPatch of Inner .* head 'Post'"):
        export_to_onnx(gated, tmp_path / "gated", example)

    bound = _nested_patch_network()
    bound.bind(False, State.PREDICTION, ["x"])
    for name, model in (("bound", bound), ("side_branch", _nested_patch_network(side_branch=True))):
        onnx_path, manifest = export_to_onnx(model, tmp_path / name, example)
        assert manifest["output_module"] == "Post"
        produced = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
            None, {"input": example.numpy()}
        )[0]
        model.Inner.patch = None
        with torch.no_grad():
            expected = dict(model.named_forward(example))["Post"].numpy()
        np.testing.assert_allclose(produced, expected, atol=1e-5)
