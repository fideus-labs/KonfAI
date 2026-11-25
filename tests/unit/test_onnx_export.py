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
    assert manifest["patch"]["overlap"] is None
    assert manifest["blend"] == "Overwrite"


def test_the_exported_graph_names_no_path_of_the_machine_that_exported_it(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    import konfai
    from konfai.export import export_to_onnx
    from konfai.models.python.segmentation.UNet import UNet

    model = UNet(dim=2, channels=[1, 8, 16], nb_class=2).eval()
    export_to_onnx(model, tmp_path, torch.randn(1, 1, 64, 64), "UNetBlock_0.Head.Softmax", opset=18)
    graph = str(onnx.load(str(tmp_path / "model.onnx")))
    for installed in (konfai, torch):
        assert str(Path(installed.__file__).resolve().parent) not in graph, installed.__name__


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


@pytest.mark.parametrize(("dtype", "constant"), [(torch.float64, np.float64(0.95)), (torch.float32, np.float32(0.95))])
def test_a_python_float_is_a_constant_of_the_graph_s_own_precision(tmp_path, dtype, constant):
    # The exporter's own scalar_tensor makes 0.95 a float32 constant cast to float64: 0.949999988.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from konfai.export import export_to_onnx
    from onnx import numpy_helper

    class Scale(torch.nn.Module):
        def named_forward(self, x):
            yield "scaled", x * 0.95

    export_to_onnx(Scale(), tmp_path, torch.rand(1, 1, 8, 8, dtype=dtype), "scaled")
    graph = onnx.load(str(tmp_path / "model.onnx")).graph
    assert [node.op_type for node in graph.node] == ["Mul"]
    [initializer] = [numpy_helper.to_array(tensor) for tensor in graph.initializer]
    assert initializer.dtype == constant.dtype and initializer == constant


@pytest.mark.parametrize(("blend", "version"), [("Gaussian", 5), ("GaussianFloat16", 6), ("fusion:Surest", 6)])
def test_a_float16_blend_is_written_at_the_manifest_version_that_carries_it(tmp_path, blend, version):
    # A runtime reading version 5 alone refuses a GaussianFloat16 blend, or a patch combine of an app's extension,
    # rather than blend it as Cosinus.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from konfai.export import export_to_onnx

    class Head(torch.nn.Module):
        def named_forward(self, x):
            yield "head", torch.tanh(x)

    _, manifest = export_to_onnx(Head(), tmp_path, torch.randn(1, 1, 16, 16), "head", extra_manifest={"blend": blend})
    assert manifest["konfai_edge_manifest"] == version


class _Head(torch.nn.Module):
    """``forward`` as the one head of a routed graph."""

    def named_forward(self, x):
        yield "head", self.forward(x)


class _Conv(_Head):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _Fixed(_Conv):  # reads an axis as a number, which fixes it
    def forward(self, x):
        return self.conv(x)[..., : int(x.shape[-2]) // 2, :]


def test_a_model_exported_whole_keeps_its_axes_free_and_one_that_fixes_them_is_refused(tmp_path):
    # One window per case: the manifest's patch is zeros and the graph runs at any shape; a forward that reads
    # its input's shape as a number answers for the example alone, which the export finds and refuses.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Floored(_Conv):  # pads up to a floor in a branch on the size: the export keeps the example's (SynthSeg v1)
        def forward(self, x):
            width = x.shape[-1]
            if width < 32:
                x = torch.nn.functional.pad(x, (0, 32 - width))
            return self.conv(x)

    class Decided(_Conv):  # a hard decision between two networks, as SynthSeg takes one
        def forward(self, x):
            return self.conv(self.conv(x)[:, :1].tanh() * (self.conv(x).argmax(1, keepdim=True) == 1))

    example = torch.randn(1, 1, 8, 10, 12)
    path, manifest = export_to_onnx(_Conv().eval(), tmp_path / "free", example, "head", dynamic=True)
    assert manifest["patch"]["size"] == [0, 0, 0]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.run(None, {"input": np.zeros((1, 1, 5, 7, 9), np.float32)})[0].shape == (1, 2, 5, 7, 9)
    with pytest.raises(PredictorError, match="answers for the example's shape alone"):
        export_to_onnx(_Fixed().eval(), tmp_path / "fixed", example, "head", dynamic=True)
    # Traced on 12 voxels along x, under its floor of 32, which torch takes past.
    with pytest.raises(PredictorError, match=r"at most 31 voxels.*answers for the example's shape alone"):
        export_to_onnx(Floored().eval(), tmp_path / "floored", example, "head", dynamic=True)
    # Checked on a smooth input, against the gap at the example's own shape: a hard decision is not a fixed shape.
    export_to_onnx(Decided().eval(), tmp_path / "decided", example, "head", dynamic=True)


def test_a_graph_refused_leaves_the_one_it_would_replace(tmp_path):
    # A graph that fails its check is never written over the bundle's: the manifest beside it still names a graph
    # that passed.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")  # the check of the free axes runs the graph
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    example = torch.randn(1, 1, 8, 10, 12)
    path, _ = export_to_onnx(_Conv().eval(), tmp_path, example, "head", dynamic=True)
    kept = path.read_bytes()
    with pytest.raises(PredictorError, match="answers for the example's shape alone"):
        export_to_onnx(_Fixed().eval(), tmp_path, example, "head", dynamic=True)
    assert path.read_bytes() == kept and sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "model.onnx"]


def test_two_exports_to_one_destination_stage_their_graphs_apart(tmp_path, monkeypatch):
    # A second export to the bundle while the first checks its graph stages its own: neither takes the other's.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai import export

    check, nested = export._check_free_axes, []

    def checked(*args):
        if not nested:
            nested.append(True)
            export.export_to_onnx(_Conv().eval(), tmp_path, torch.randn(1, 1, 8, 10, 12), "head", dynamic=True)
        return check(*args)

    monkeypatch.setattr(export, "_check_free_axes", checked)
    export.export_to_onnx(_Conv().eval(), tmp_path, torch.randn(1, 1, 8, 10, 12), "head", dynamic=True)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json", "model.onnx"]


def test_the_check_of_the_free_axes_reads_the_trace_through_no_deprecated_name(tmp_path):
    # torch 2.12 renamed the shape environment's var_to_val backed_var_to_val and warns on the old name.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    import warnings

    from konfai.export import export_to_onnx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_to_onnx(_Conv().eval(), tmp_path, torch.randn(1, 1, 8, 10, 12), "head", dynamic=True)
    assert not [str(w.message) for w in caught if "var_to_val" in str(w.message)]


class _Wide(_Conv):  # weights past the 256 bytes torch's exporter keeps inline when it moves the others out
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 8, 3, padding=1)


@pytest.mark.parametrize("dynamic", [False, True])
def test_weights_past_torchs_limit_for_one_file_stay_inline_in_the_one_file_the_bundle_holds(
    tmp_path, monkeypatch, dynamic
):
    # Past 1536 MB (MedicalSAM3's 1.9 GB) torch moves the weights to a file beside the graph, which no consumer of
    # the bundle carries: the export keeps them inline, in a file as readable as any the user makes, twice over.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    import os
    import shutil

    import onnx

    program = pytest.importorskip("torch.onnx._internal.exporter._onnx_program")
    if not hasattr(program, "_LARGE_MODEL_THRESHOLD"):
        pytest.skip("this torch keeps no size limit for one file")
    monkeypatch.setattr(program, "_LARGE_MODEL_THRESHOLD", 0)  # a small conv stands for 1.9 GB of weights
    from konfai.export import export_to_onnx

    example = torch.randn(1, 1, 8, 10, 12)
    bundle = tmp_path / "bundle"
    for _ in range(2):
        export_to_onnx(_Wide().eval(), bundle, example, "head", dynamic=dynamic)
    assert sorted(path.name for path in bundle.iterdir()) == ["manifest.json", "model.onnx"]
    graph = onnx.load(str(bundle / "model.onnx"), load_external_data=False)
    assert not [t.name for t in graph.graph.initializer if t.external_data]
    umask = os.umask(0)
    os.umask(umask)
    assert (bundle / "model.onnx").stat().st_mode & 0o777 == 0o666 & ~umask
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy2(bundle / "model.onnx", moved / "model.onnx")
    session = ort.InferenceSession(str(moved / "model.onnx"), providers=["CPUExecutionProvider"])
    assert session.run(None, {session.get_inputs()[0].name: example.numpy()})[0].shape == (1, 8, 8, 10, 12)


def test_a_graph_past_what_one_onnx_file_holds_is_refused_and_leaves_the_bundle_as_it_was(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from konfai import export
    from konfai.utils.errors import PredictorError

    example = torch.randn(1, 1, 8, 10, 12)
    export.export_to_onnx(_Conv().eval(), tmp_path, example, "head")
    before = (tmp_path / "model.onnx").read_bytes()
    monkeypatch.setattr(export, "ONE_FILE", 1600)  # between the conv's graph (1297 bytes) and the wider one's (1969)
    with pytest.raises(PredictorError, match="past the 2 GiB one ONNX file holds"):
        export.export_to_onnx(_Wide().eval(), tmp_path, example, "head")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json", "model.onnx"]
    assert (tmp_path / "model.onnx").read_bytes() == before


@pytest.mark.parametrize(("eager", "exported"), [(float("nan"), 2.0), (float("-inf"), float("inf"))])
def test_a_nonfinite_value_the_graph_moves_is_refused_and_one_it_keeps_is_not(tmp_path, eager, exported):
    # A NaN torch writes where the graph writes a number, or an infinity of the other sign, is a gap a maximum over
    # the voxels would read as NaN, and pass; both writing the same NaN at the same voxel is no gap.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")  # the check of the free axes runs the graph
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Moved(_Head):  # the smooth input is 1 at its first voxel only
        def forward(self, x):
            value = exported if torch.compiler.is_compiling() else eager
            return torch.where(x > 0.999, value, x * 2)

    class Kept(_Head):
        def forward(self, x):
            return torch.where(x > 0.999, eager, x * 2)

    example = torch.randn(1, 1, 8, 10, 12)
    with pytest.raises(PredictorError, match="nonfinite"):
        export_to_onnx(Moved().eval(), tmp_path / "moved", example, "head", dynamic=True)
    export_to_onnx(Kept().eval(), tmp_path / "kept", example, "head", dynamic=True)


def test_a_graph_off_torch_at_the_example_s_own_shape_is_refused_before_it_sets_the_tolerance(tmp_path):
    # Off by one everywhere, the graph is off by one at the example's shape too: that gap is no tolerance for the
    # other shapes, which it would pass at ten times its size. A right graph's rounding there grows with its values.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Off(_Conv):
        def forward(self, x):
            return self.conv(x) + 1 if torch.compiler.is_compiling() else self.conv(x)

    class Large(_Conv):  # right, in thousands: its rounding is a thousand times a unit's
        def forward(self, x):
            return self.conv(x) * 3000.0

    with pytest.raises(PredictorError, match="at the example's own shape"):
        export_to_onnx(Off().eval(), tmp_path / "off", torch.randn(1, 1, 8, 10, 12), "head", dynamic=True)
    export_to_onnx(Large().eval(), tmp_path / "large", torch.randn(1, 1, 8, 10, 12), "head", dynamic=True)


def test_a_branch_on_a_free_size_the_example_never_takes_is_refused(tmp_path):
    # Traced on 12 voxels along x, the graph keeps the branch of 8 or more and would answer it for 7: the trace's
    # own guard says where it holds, and torch taking a size past it is refused, whatever the probes above it say.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")  # the check of the free axes runs the graph
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Below(_Conv):
        def forward(self, x):
            return self.conv(x) if x.shape[-1] >= 8 else self.conv(x) + 1

    with pytest.raises(PredictorError, match=r"8 voxels or more.*answers for the example's shape alone"):
        export_to_onnx(Below().eval(), tmp_path, torch.randn(1, 1, 8, 10, 12), "head", dynamic=True)


def test_a_branch_on_sizes_the_trace_ties_is_refused_where_torch_takes_them_untied(tmp_path):
    # Traced on a square, the graph keeps the square's branch and ties the two axes to one size: torch takes a
    # rectangle, where the graph would answer with that branch. A tie torch needs itself (two inputs added) holds.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Square(_Conv):
        def forward(self, x):
            return self.conv(x) if x.shape[-1] == x.shape[-2] else self.conv(x) + 1

    with pytest.raises(PredictorError, match=r"tied.*answers for the example's shape alone"):
        export_to_onnx(Square().eval(), tmp_path, torch.randn(1, 1, 8, 12, 12), "head", dynamic=True)


def test_a_branch_on_a_tie_is_refused_on_whichever_tied_axis_torch_takes_moved_alone(tmp_path):
    # Traced on two widths equal, the graph adds z unsliced: torch refuses x alone wider (x + z), and takes z alone
    # wider, which it slices to x's width where the graph does not.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Sliced(torch.nn.Module):
        def named_forward(self, x, z):
            yield "head", x + (z if z.shape[-1] == x.shape[-1] else z[..., : x.shape[-1]])

    examples = (torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    with pytest.raises(PredictorError, match=r"tied.*answers for the example's shape alone"):
        export_to_onnx(Sliced().eval(), tmp_path, examples, "head", input_group=["A", "B"], dynamic=True)


def test_an_example_of_axes_of_one_voxel_is_no_other_shape_to_check_on(tmp_path):
    # A shorter axis of one voxel is that one voxel: the example's own shape checks nothing of another.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class One(_Head):
        def forward(self, x):
            return x.reshape(1, 1, 1, 1) * 2

    with pytest.raises(PredictorError, match="torch takes none"):
        export_to_onnx(One().eval(), tmp_path, torch.randn(1, 1, 1, 1), "head", dynamic=True)


@pytest.mark.parametrize("example", [(1, 1, 8, 12, 12), (1, 1, 8, 16, 8)], ids=["square", "oblong"])
def test_a_branch_on_the_order_of_two_sizes_is_checked_on_shapes_that_reverse_it(tmp_path, example):
    # Traced taller than wide, or as tall, the graph keeps that branch: it is checked on a case wider than tall, which
    # torch answers otherwise.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Tall(_Conv):
        def forward(self, x):
            return self.conv(x) if x.shape[-2] >= x.shape[-1] else self.conv(x) + 1

    with pytest.raises(PredictorError, match="answers for the example's shape alone"):
        export_to_onnx(Tall().eval(), tmp_path, torch.randn(*example), "head", dynamic=True)


def test_the_shapes_a_model_of_many_inputs_is_checked_on_grow_with_the_rank_alone(tmp_path):
    # Each axis of an input is moved by its own multiple of the step, the same for every input: four inputs of three
    # axes are checked on inputs at most four times as long on an axis, not on one 13 times as long.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx

    seen: list[int] = []

    class Means(torch.nn.Module):
        def named_forward(self, *inputs):
            if not torch.compiler.is_compiling():  # the checks' own calls, not the trace's
                seen.extend(x.numel() for x in inputs)
            yield "head", inputs[0] + sum(x.mean() for x in inputs[1:])

    examples = tuple(torch.randn(1, 1, 8, 8, 8) for _ in range(4))
    export_to_onnx(Means().eval(), tmp_path, examples, "head", input_group=["A", "B", "C", "D"], dynamic=True)
    assert max(seen) <= 4**3 * 8**3, max(seen)


def test_a_model_taking_blocks_of_32_voxels_is_checked_on_shapes_it_takes(tmp_path):
    # Probes keep the powers of two of the example's extents up to 32: a model of 32-voxel blocks takes them, and the
    # graph is checked there. A model torch takes at the example's shape alone has no other shape to be checked on.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    class Blocks(_Head):  # each 32 x 32 block less its mean
        def forward(self, x):
            n, c, h, w = x.shape
            blocks = x.reshape(n, c, h // 32, 32, w // 32, 32)
            return (blocks - blocks.mean(dim=(3, 5), keepdim=True)).reshape(n, c, h, w)

    class Exact(_Head):
        def forward(self, x):
            return x.reshape(1, 1, 64, 64) * 2

    path, _ = export_to_onnx(Blocks().eval(), tmp_path / "blocks", torch.randn(1, 1, 64, 64), "head", dynamic=True)
    case = torch.randn(1, 1, 96, 160)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": case.numpy()})[0]
    np.testing.assert_allclose(got, Blocks()(case).numpy(), atol=1e-6)
    with pytest.raises(PredictorError, match="torch takes none"):
        export_to_onnx(Exact().eval(), tmp_path / "exact", torch.randn(1, 1, 64, 64), "head", dynamic=True)


def test_a_model_reading_two_groups_at_sizes_it_ties_is_checked_at_sizes_that_keep_the_tie(tmp_path, monkeypatch):
    # The trace keeps the first group twice the second (2*s, s): the shapes it is checked on move s, and keep the tie.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from konfai import export
    from konfai.export import export_to_onnx

    traced, held = export._traced_sizes, []
    monkeypatch.setattr(export, "_traced_sizes", lambda *args: held.append(traced(*args)) or held[-1])

    class Tied(torch.nn.Module):
        def named_forward(self, x, y):
            yield "head", x + torch.nn.functional.interpolate(y, scale_factor=2)

    examples = (torch.randn(1, 1, 64, 64), torch.randn(1, 1, 32, 32))
    export_to_onnx(Tied().eval(), tmp_path, examples, "head", input_group=["A", "B"], dynamic=True)
    [(sizes, values)] = held
    assert [str(size) for size in sizes[1][2:]] == [str(symbol) for symbol in values] and set(values.values()) == {32}


def test_every_size_the_trace_holds_is_one_the_checks_move() -> None:
    # A free axis an exporter fixed (an int, or a symbol of one value as torch 2.5 holds it, where 2.12 keeps a symbol
    # its guards hold at one value) gets a symbol of its own, so the shapes checked move it and the graph is seen to
    # refuse them. A size held as an expression of a symbol no input holds alone (2*s + 1) moves with that symbol,
    # valued as the trace holds it.
    from types import SimpleNamespace

    import sympy
    from konfai.export import _traced_sizes
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.sym_node import SymNode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    env = ShapeEnv()
    symbol = env.create_symintnode(env.create_symbol(32, ConstantSource("s")), hint=32)
    fixed = torch.SymInt(SymNode(sympy.Integer(10), env, int, 10))
    shape = (1, 1, 10, fixed, 2 * symbol + 1)
    node = SimpleNamespace(op="placeholder", name="x", meta={"val": SimpleNamespace(shape=shape)})
    traced = SimpleNamespace(graph_signature=SimpleNamespace(user_inputs=["x"]), graph=SimpleNamespace(nodes=[node]))
    sizes, values = _traced_sizes(traced, [[1, 1, 10, 10, 65]], 2)
    assert sizes[0][:2] == [1, 1] and values == {sizes[0][2]: 10, sizes[0][3]: 10, symbol.node.expr: 32}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_a_model_on_a_device_is_checked_on_that_device(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")  # the check of the free axes runs the graph
    from konfai.export import export_to_onnx

    example = torch.randn(1, 1, 8, 10, 12, device="cuda")
    export_to_onnx(_Conv().cuda().eval(), tmp_path, example, "head", dynamic=True)


def test_a_case_statistic_s_graphs_land_in_folders_made_for_both_before_either_is_written(tmp_path):
    # The finish graph's folder does not exist: it is made with the partial's before either graph is written, so the
    # partial replaces the previous one only where the finish lands too.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from konfai.export import export_case_statistic

    class Mean:
        halo = 0

        def partial(self, block):
            return torch.stack([block.sum(), torch.ones_like(block).sum()])

        def finish(self, total):
            return total[0] / total[1]

    export_case_statistic(Mean(), tmp_path / "partial.onnx", tmp_path / "graphs" / "finish.onnx", rank=3)
    assert (tmp_path / "partial.onnx").is_file() and (tmp_path / "graphs" / "finish.onnx").is_file()


def test_a_case_statistic_is_traced_past_its_halo_and_checked_on_every_spatial_axis(tmp_path):
    # Its partial takes ``[1, *spatial]``, the region and ``halo`` voxels on both sides: a 9-voxel stencil (halo 4)
    # has a region to trace on, and a partial right on the first spatial axis 48 voxels longer and wrong on it twice as
    # long is refused, on an input that varies along that axis (a mean is the same over a replicated edge otherwise).
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    from konfai.export import export_case_statistic
    from konfai.utils.errors import PredictorError

    class Peak:  # the mean of each region voxel's 9-voxel box maximum
        halo = 4

        def partial(self, block):
            peak = torch.nn.functional.max_pool3d(block[None], 9, stride=1)
            return torch.stack([peak.sum(), torch.ones_like(peak).sum()])

        def finish(self, total):
            return total[0] / total[1]

    class Floored(
        Peak
    ):  # replicates the first axis' edge up to 60 voxels in a branch on its size, which the export keeps
        halo = 0

        def partial(self, block):
            if block.shape[1] < 60:
                block = torch.nn.functional.pad(block[None], (0, 0, 0, 0, 0, 60 - block.shape[1]), "replicate")[0]
            return torch.stack([block.mean(), block.new_ones(())])

    partial, finish = tmp_path / "partial.onnx", tmp_path / "finish.onnx"
    export_case_statistic(Peak(), partial, finish, rank=3)
    block = torch.from_numpy(np.random.default_rng(0).random((1, 11, 14, 12)))
    got = ort.InferenceSession(str(partial), providers=["CPUExecutionProvider"]).run(None, {"block": block.numpy()})[0]
    np.testing.assert_allclose(got, Peak().partial(block).numpy(), rtol=1e-12)
    with pytest.raises(PredictorError, match="answers for the example's shape alone"):
        export_case_statistic(Floored(), partial, finish, rank=3)


def test_a_model_reading_two_groups_names_each_in_its_manifest_and_keeps_a_stated_blend(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    from konfai.export import export_to_onnx

    class Masked(torch.nn.Module):
        def named_forward(self, x, mask):
            yield "masked", torch.tanh(x) * mask

    examples = (torch.randn(1, 2, 16, 16), torch.ones(1, 1, 16, 16))
    _, manifest = export_to_onnx(
        Masked(), tmp_path, examples, input_group=["CT", "MASK"], extra_manifest={"blend": "Gaussian"}
    )

    assert manifest["input"]["group"] == "CT" and manifest["input"]["channels"] == 2
    assert manifest["reads"] == [{"group": "MASK", "channels": 1, "preprocessing": []}]
    assert manifest["blend"] == "Gaussian"
    # An input no group binds is refused before anything is written.
    from konfai.utils.errors import PredictorError

    with pytest.raises(PredictorError, match="example inputs"):
        export_to_onnx(Masked(), tmp_path / "refused", examples, input_group=["CT"])
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("stated", [{"pad_value": "unknown"}, {"placement": "centre"}])
def test_export_refuses_a_pad_or_a_placement_the_manifest_schema_rejects(tmp_path, stated):
    from konfai.export import export_to_onnx
    from konfai.utils.errors import PredictorError

    with pytest.raises(PredictorError, match=r"unknown|centre"):
        export_to_onnx(None, tmp_path, torch.zeros(1, 1, 8, 8), **stated)
    assert not list(tmp_path.iterdir())


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
