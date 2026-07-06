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

"""Tests for ``konfai.data.transform``: Clip, Dilate, Norm, Crop, Standardize, Padding,
ResampleToResolution/ResampleToShape, InferenceStack, and KonfAIInference."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from konfai.data.transform import (
    DEFAULT_INFERENCE_MODEL_NAME,
    DEFAULT_INFERENCE_REPO_ID,
    Clip,
    Crop,
    Dilate,
    InferenceStack,
    KonfAIInference,
    Norm,
    Normalize,
    Padding,
    ResampleToResolution,
    ResampleToShape,
    Standardize,
    Statistics,
)
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError

# --------------------------------------------------------------------------------------
# Clip
# --------------------------------------------------------------------------------------


def test_clip_resolves_min_and_percentile_bounds() -> None:
    # ``min`` (torch scalar) and ``percentile:<p>`` (numpy scalar) bounds must be coerced to float
    # so the in-place clip assignments are valid for a torch tensor.
    tensor = torch.arange(0, 100, dtype=torch.float32)
    clip = Clip(min_value="min", max_value="percentile:90")

    out = clip("case", tensor.clone(), Attribute())

    assert out.min().item() == pytest.approx(0.0)
    assert out.max().item() == pytest.approx(89.1)
    assert out.dtype == torch.float32


def test_clip_fixed_numeric_bounds() -> None:
    tensor = torch.arange(-50, 50, dtype=torch.float32)
    clip = Clip(min_value=-10.0, max_value=10.0)

    out = clip("case", tensor.clone(), Attribute())

    assert out.min().item() == pytest.approx(-10.0)
    assert out.max().item() == pytest.approx(10.0)


# --------------------------------------------------------------------------------------
# Dilate
# --------------------------------------------------------------------------------------


def _dense_cube_dilation(tensor: torch.Tensor, dilate: int) -> torch.Tensor:
    """Reference: dilation via a single dense k**n max-pool (the pre-separable implementation)."""
    data = (tensor > 0).to(torch.float32)
    k = 2 * dilate + 1
    if data.dim() - 1 == 2:
        data = F.max_pool2d(data, kernel_size=k, stride=1, padding=dilate)
    else:
        data = F.max_pool3d(data, kernel_size=k, stride=1, padding=dilate)
    return data.to(tensor.dtype)


@pytest.mark.parametrize("dilate", [1, 2, 5])
@pytest.mark.parametrize("shape", [(1, 24, 30), (2, 12, 20, 18)])
def test_dilate_separable_matches_dense_cube(shape: tuple[int, ...], dilate: int) -> None:
    # The separable 1-D max-pool implementation must be bit-identical to the dense k**n cube it replaces,
    # for both [C,H,W] and [C,D,H,W] inputs and several radii — this is the correctness guarantee that
    # lets the ~14x speedup ship as a transparent optimization.
    torch.manual_seed(0)
    mask = (torch.rand(shape) > 0.7).to(torch.uint8)

    out = Dilate(dilate)("case", mask.clone(), Attribute())
    ref = _dense_cube_dilation(mask, dilate)

    assert torch.equal(out, ref)
    assert out.dtype == mask.dtype
    assert out.shape == mask.shape


def test_dilate_single_voxel_fills_neighbourhood() -> None:
    # A single active voxel dilated by 1 must fill its full 3x3x3 neighbourhood.
    mask = torch.zeros(1, 5, 5, 5, dtype=torch.uint8)
    mask[0, 2, 2, 2] = 1

    out = Dilate(1)("case", mask.clone(), Attribute())

    assert out[0, 1:4, 1:4, 1:4].sum().item() == 27
    assert out.sum().item() == 27


def test_dilate_zero_is_identity() -> None:
    mask = (torch.rand(1, 8, 8, 8) > 0.5).to(torch.uint8)
    out = Dilate(0)("case", mask.clone(), Attribute())
    assert torch.equal(out, mask)


# --------------------------------------------------------------------------------------
# Norm
# --------------------------------------------------------------------------------------


def _stack_attribute() -> Attribute:
    # Geometry of a displacement-field stack: the leading image axis holds the vector components
    # (origin 0 / spacing 1 / identity direction row), the remaining axes carry the fixed grid.
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 1.0, 2.0, 3.0])
    attribute["Spacing"] = np.asarray([1.0, 2.0, 2.0, 2.0])
    direction = np.eye(4)
    direction[1:, 1:] = np.diag([1.0, -1.0, 1.0])
    attribute["Direction"] = direction.flatten()
    return attribute


def test_norm_reduces_trailing_component_axis_and_geometry() -> None:
    # A stack of 2 displacement fields [N=2, D, H, W, C=3] -> per-sample magnitudes [2, D, H, W].
    tensors = torch.randn(2, 4, 5, 6, 3)
    attribute = _stack_attribute()

    out = Norm()("case", tensors, attribute)

    assert list(out.shape) == [2, 4, 5, 6]
    assert torch.allclose(out, torch.linalg.norm(tensors, dim=-1))
    # The reduced trailing tensor axis is the first geometry axis: it must be dropped.
    assert attribute.get_np_array("Origin").tolist() == [1.0, 2.0, 3.0]
    assert attribute.get_np_array("Spacing").tolist() == [2.0, 2.0, 2.0]
    assert attribute.get_np_array("Direction").tolist() == np.diag([1.0, -1.0, 1.0]).flatten().tolist()


def test_norm_transform_shape_drops_trailing_axis() -> None:
    assert Norm().transform_shape("group", "case", [4, 5, 6, 3], Attribute()) == [4, 5, 6]


# --------------------------------------------------------------------------------------
# Crop — transform_shape predicts the spatial crop exactly (patch planning depends on it)
# --------------------------------------------------------------------------------------


def test_crop_transform_shape_matches_spatial_crop() -> None:
    # The pre-fix code treated ``shape[0]`` as a channel dim and paired the crop box with
    # ``shape[1:]``, shifting every axis by one and returning a wrong shape.
    attribute = Attribute()
    attribute["box"] = np.array([[2, 3], [1, 1], [4, 2]])  # (start, end-distance) per spatial axis

    out = Crop().transform_shape("CT", "CASE_001", [10, 20, 30], attribute)

    # 10-2-3, 20-1-1, 30-4-2 — each spatial axis cropped by its own box row.
    assert out == [5, 18, 24]


# --------------------------------------------------------------------------------------
# Standardize
# --------------------------------------------------------------------------------------


def test_standardize_explicit_scalar_stats():
    """#5 Standardize with explicit scalar mean/std must not crash."""
    t = Standardize(lazy=False, mean=[10.0], std=[2.0])
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = t("c", x.clone(), Attribute())
    assert torch.allclose(out, (x - 10.0) / 2.0)


def test_standardize_explicit_per_channel_stats():
    """#5 Per-channel mean/std broadcast over the channel axis."""
    t = Standardize(lazy=False, mean=[10.0, 20.0], std=[2.0, 4.0])
    x = torch.zeros(2, 3, 4)
    x[0] = 10.0
    x[1] = 20.0
    out = t("c", x.clone(), Attribute())
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# --------------------------------------------------------------------------------------
# Padding — origin bookkeeping
# --------------------------------------------------------------------------------------


def test_padding_shifts_origin_along_the_padded_axes(image_attributes):
    """Each F.pad pair (X, Y, Z) must shift the matching (x, y, z) origin component."""
    attributes = image_attributes([10.0, 20.0, 30.0], [1.0, 2.0, 4.0])

    padded = Padding(padding=[1, 0, 2, 0, 3, 0])("case", torch.zeros(1, 5, 5, 5), attributes)

    assert list(padded.shape) == [1, 8, 7, 6]
    np.testing.assert_allclose(
        attributes.get_np_array("Origin"),
        [10.0 - 1 * 1.0, 20.0 - 2 * 2.0, 30.0 - 3 * 4.0],
    )


def test_padding_after_the_data_keeps_origin(image_attributes):
    """Padding only on the high side of each axis must leave the origin untouched."""
    attributes = image_attributes([10.0, 20.0, 30.0], [1.0, 2.0, 4.0])

    padded = Padding(padding=[0, 2, 0, 0, 0, 1])("case", torch.zeros(1, 5, 5, 5), attributes)

    assert list(padded.shape) == [1, 6, 5, 7]
    np.testing.assert_allclose(attributes.get_np_array("Origin"), [10.0, 20.0, 30.0])


# --------------------------------------------------------------------------------------
# ResampleToResolution / ResampleToShape
# --------------------------------------------------------------------------------------


def test_resample_to_resolution_transform_shape_missing_spacing_raises():
    """A tensor without 'Spacing' metadata must surface a TransformError, not fall through."""
    with pytest.raises(TransformError):
        ResampleToResolution().transform_shape("group", "case", [10, 10, 10], Attribute())


def test_resample_to_shape_transform_shape_missing_spacing_raises():
    """ResampleToShape must also raise when 'Spacing' metadata is absent."""
    with pytest.raises(TransformError):
        ResampleToShape().transform_shape("group", "case", [10, 10, 10], Attribute())


def test_resample_to_resolution_transform_shape_dimension_mismatch_message():
    """The dimension-mismatch error is raised and its message interpolates the actual shape."""
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0], dtype=np.float64)
    with pytest.raises(TransformError) as excinfo:
        ResampleToResolution(spacing=[1.0, 1.0]).transform_shape("group", "case", [10, 10, 10], attributes)
    assert "shape=[10, 10, 10]" in str(excinfo.value)


def test_resample_to_shape_transform_shape_dimension_mismatch_message():
    """ResampleToShape raises a formatted (f-string) message on a shape/target mismatch."""
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(TransformError) as excinfo:
        ResampleToShape(shape=[4, 4]).transform_shape("group", "case", [10, 10, 10], attributes)
    message = str(excinfo.value)
    assert "shape=[10, 10, 10]" in message
    assert "target_shape" in message


def test_resample_to_shape_does_not_mutate_config():
    """#9 transform_shape must not write resolved dims back into the shared instance config."""
    resampler = ResampleToShape(shape=[0, 16, 16])
    before = resampler.shape.clone()
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    out = resampler.transform_shape("CT", "case", [8, 16, 16], attributes)
    assert out[0] == 8  # sentinel 0 resolved to the input dim for this call
    assert torch.equal(resampler.shape, before), "self.shape must stay [0, 16, 16] for the next case"


def test_resample_to_shape_inverse_without_spacing_metadata():
    """Inverting a resample must not pop a 'Spacing' the forward pass never pushed."""
    resampler = ResampleToShape(shape=[4, 4, 4])
    attributes = Attribute()  # no image metadata at all
    tensor = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)

    forward = resampler("case", tensor, attributes)
    assert list(forward.shape) == [1, 4, 4, 4]

    restored = resampler.inverse("case", forward, attributes)
    assert list(restored.shape) == [1, 8, 8, 8]


def test_resample_to_shape_inverse_pops_pushed_spacing():
    """When 'Spacing' exists, the inverse removes the version the forward pass pushed."""
    resampler = ResampleToShape(shape=[4, 4, 4])
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    tensor = torch.zeros(1, 8, 8, 8)

    resampler("case", tensor, attributes)
    # image_shape / shape == 2 on every axis, so the resampled spacing doubles.
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [2.0, 2.0, 2.0])

    resampler.inverse("case", torch.zeros(1, 4, 4, 4), attributes)
    # The pushed spacing is popped, restoring the original resolution.
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [1.0, 1.0, 1.0])


# --------------------------------------------------------------------------------------
# InferenceStack / KonfAIInference
# --------------------------------------------------------------------------------------


def test_inference_stack_super_init_enables_dataset_fallback():
    """InferenceStack must inherit Transform's 'datasets' list for the fallback write path."""
    stack = InferenceStack(dataset="", name="pred", mode="mean")
    assert stack.datasets == []

    written: dict[str, object] = {}

    class _FakeDataset:
        def write(self, group, name, data, cache_attribute):
            written["group"] = group
            written["name"] = name
            written["shape"] = tuple(data.shape)

    stack.set_datasets([object(), _FakeDataset()])

    tensors = torch.stack(
        [torch.full((1, 2, 2), 3.0), torch.full((1, 2, 2), 5.0)],
        dim=0,
    )  # shape [2, 1, 2, 2]: two stacked predictions
    out = stack("pred", tensors, Attribute())

    assert written["group"] == "InferenceStack"
    assert written["name"] == "pred"
    assert written["shape"] == (2, 2, 2)
    # 'mean' mode averages the two predictions element-wise.
    assert torch.allclose(out, torch.full((1, 2, 2), 4.0))


def test_konfai_inference_reassembles_channels_in_sorted_order(tmp_path, monkeypatch):
    """Per-channel outputs must be stacked in deterministic (sorted) case order."""
    sitk = pytest.importorskip("SimpleITK")

    output_dir = tmp_path / "Output"
    files = []
    for i in range(3):
        case_dir = output_dir / f"P{i:03d}"
        case_dir.mkdir(parents=True)
        array = np.full((2, 2, 2), float(i * 10), dtype=np.float32)
        path = case_dir / "Volume.mha"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        files.append(path)

    # Simulate an arbitrary (here reversed) filesystem enumeration order.
    scrambled = list(reversed(files))
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(scrambled))

    result = KonfAIInference._reassemble_output(output_dir)

    assert list(result.shape) == [3, 2, 2, 2]
    assert float(result[0].mean()) == 0.0
    assert float(result[1].mean()) == 10.0
    assert float(result[2].mean()) == 20.0


def test_konfai_inference_default_repo_and_model_preserved():
    """Constructing without arguments keeps the current published repo/model default."""
    transform = KonfAIInference()

    assert transform.repo_id == DEFAULT_INFERENCE_REPO_ID
    assert transform.model_name == DEFAULT_INFERENCE_MODEL_NAME
    assert transform.repo_id == "VBoussot/MRSegmentator-KonfAI"
    assert transform.model_name == "MRSegmentator"


def test_konfai_inference_forwards_configured_repo_and_model(monkeypatch):
    """A custom repo/model is forwarded verbatim to the KonfAIApp spec, not the default."""
    captured = {}

    class _FakeKonfAIApp:
        def __init__(self, spec, *args):
            captured["spec"] = spec

        def infer(self, *args, **kwargs):
            captured["infer"] = (args, kwargs)

    fake_module = types.ModuleType("konfai_apps")
    fake_module.KonfAIApp = _FakeKonfAIApp
    monkeypatch.setitem(sys.modules, "konfai_apps", fake_module)

    transform = KonfAIInference(
        repo_id="acme/Custom-KonfAI",
        model_name="CustomModel",
        checkpoints_name=["fold_1"],
    )
    transform.infer_entry(Path("dataset"), Path("output"), [])

    assert captured["spec"] == "acme/Custom-KonfAI:CustomModel"


# --------------------------------------------------------------------------------------
# Finalize transforms are device-transparent (the volume may be blended on the GPU)
# --------------------------------------------------------------------------------------


def test_standardize_inverse_stats_follow_the_volume_and_stay_float32() -> None:
    # The cached stats parse back as float64: without the float32 cast, denormalizing a whole fp16
    # volume would promote it to a float64 copy (4x the memory of the fp16 finalize path).
    attr = Attribute()
    attr["Mean"] = torch.tensor([10.0])
    attr["Std"] = torch.tensor([2.0])
    out = Standardize().inverse("case", torch.ones(1, 4, dtype=torch.float16), attr)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.full((1, 4), 12.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device finalize only applies on CUDA")
def test_finalize_transforms_accept_a_cuda_resident_volume() -> None:
    # GPU accumulation keeps the assembled volume on the device through the whole finalize chain:
    # every default finalize transform must accept a CUDA volume (1.5.8 always handed them CPU tensors).
    volume = torch.rand(1, 4, 4, device="cuda", dtype=torch.float16)

    normalized = Normalize()("case", volume.clone(), Attribute())  # writes Min/Max from CUDA tensors
    assert normalized.device.type == "cuda"

    stats = Attribute()
    stats["Mean"] = torch.tensor([10.0])
    stats["Std"] = torch.tensor([2.0])
    denormalized = Standardize().inverse("case", volume.clone(), stats)
    assert denormalized.device.type == "cuda"

    attr = Attribute()
    Statistics()("case", volume, attr)  # writes ImageMin/Max/Mean/Std from CUDA tensors
    assert "ImageMin" in attr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device finalize only applies on CUDA")
def test_clip_percentile_bounds_accept_a_cuda_resident_volume() -> None:
    # ``np.percentile`` cannot coerce a CUDA tensor; percentile bounds must sample from a host view
    # while the clip itself stays on-device.
    tensor = torch.arange(0, 100, dtype=torch.float16, device="cuda")
    out = Clip(min_value="percentile:10", max_value="percentile:90")("case", tensor, Attribute())
    assert out.device.type == "cuda"
    assert float(out.min()) == pytest.approx(9.9, abs=0.1)
    assert float(out.max()) == pytest.approx(89.1, abs=0.1)
