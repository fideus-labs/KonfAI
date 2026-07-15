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
    Canonical,
    Clip,
    Crop,
    Dilate,
    InferenceStack,
    KonfAIInference,
    LocalityKind,
    Norm,
    Normalize,
    OneHot,
    Padding,
    ResampleToResolution,
    ResampleToShape,
    Squeeze,
    StandardDeviation,
    Standardize,
    Statistics,
    Variance,
)
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError

# --------------------------------------------------------------------------------------
# Clip
# --------------------------------------------------------------------------------------


def test_onehot_inverse_argmaxes_the_class_axis_batched_and_unbatched() -> None:
    # The inverse must argmax the axis sized num_classes -- the channel axis -- never a batch or spatial
    # axis. The predictor feeds a per-sample [num_classes, *spatial] (output[i]); a batched
    # [B, num_classes, *spatial] must work too.
    one_hot = OneHot(num_classes=4)

    unbatched = torch.randn(4, 5, 6, 7)  # [num_classes, *spatial]
    decoded = one_hot.inverse("seg", unbatched, Attribute())
    assert tuple(decoded.shape) == (1, 5, 6, 7)
    assert torch.equal(decoded, unbatched.argmax(0).unsqueeze(0))

    batched = torch.randn(2, 4, 5, 6, 7)  # [B, num_classes, *spatial]
    decoded_b = one_hot.inverse("seg", batched, Attribute())
    assert tuple(decoded_b.shape) == (2, 1, 5, 6, 7)
    assert torch.equal(decoded_b, batched.argmax(1).unsqueeze(1))


@pytest.mark.parametrize("cls", [Variance, StandardDeviation])
def test_ensemble_dispersion_keeps_member_axis_for_single_member(cls) -> None:
    # The N>1 branch does .var/.std(0).unsqueeze(0) -> [1, C, *spatial]; the single-member branch must
    # unsqueeze too, or a 1-member ensemble yields an output one rank short of the multi-member case.
    transform = cls()
    multi = transform("x", torch.randn(3, 2, 4, 4), Attribute())
    single = transform("x", torch.randn(1, 2, 4, 4), Attribute())
    assert single.ndim == multi.ndim
    assert tuple(single.shape) == tuple(multi.shape)


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
# Squeeze — transform_shape must track which axis squeeze() drops, so the patch grid folds it
# --------------------------------------------------------------------------------------


def test_squeeze_channel_axis_leaves_spatial_shape_untouched() -> None:
    # ``shape`` is the channel-stripped spatial shape, so the runtime tensor is [C, 4, 5, 6] and
    # dim 0 squeezes the channel: the spatial grid the patches tile is unchanged.
    assert Squeeze(0).transform_shape("group", "case", [4, 5, 6], Attribute()) == [4, 5, 6]


def test_squeeze_singleton_spatial_axis_is_dropped_from_the_grid() -> None:
    # Runtime tensor [C, 1, 5, 6]; dim 1 is the leading spatial axis (size 1), which squeeze() removes.
    assert Squeeze(1).transform_shape("group", "case", [1, 5, 6], Attribute()) == [5, 6]
    # A negative dim indexes from the back: -1 is the trailing spatial axis.
    assert Squeeze(-1).transform_shape("group", "case", [4, 5, 1], Attribute()) == [4, 5]


def test_squeeze_non_singleton_axis_is_a_no_op() -> None:
    # squeeze() leaves a size>1 axis in place, so the grid must not drop it either.
    assert Squeeze(1).transform_shape("group", "case", [4, 5, 6], Attribute()) == [4, 5, 6]


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

    stats_forward = Attribute()
    stats_forward["Mean"] = torch.tensor([0.5])
    stats_forward["Std"] = torch.tensor([0.25])
    standardized = Standardize()("case", volume.clone(), stats_forward)  # get_tensor stats follow the device
    assert standardized.device.type == "cuda"

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


# --------------------------------------------------------------------------------------
# Canonical. A reorientation between orthogonal direction cosines is a bijection on the
# voxels, so the property to hold it to is not closeness but exactness.
# --------------------------------------------------------------------------------------

# Deliberately non-cubic on every axis: a cube hides an axis swap, and equal extents make a
# wrong permutation look right.
_CANONICAL_SPATIAL = (9, 10, 11)

# LPS is what Canonical reorients ONTO, so it is the one direction that asks for no flip at all.
_LPS = np.diag([-1.0, -1.0, 1.0])
_RAS = np.eye(3)
# Orthogonal, and neither the identity nor a mirroring: it swaps physical x and z.
_PERMUTING = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
_OBLIQUE = np.asarray(
    [
        [np.cos(np.deg2rad(20.0)), -np.sin(np.deg2rad(20.0)), 0.0],
        [np.sin(np.deg2rad(20.0)), np.cos(np.deg2rad(20.0)), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
# A direction is orthonormal by definition, not by construction. None of these is, so none of them
# is a bijection on the voxels -- and each wears one half of a signed permutation's disguise.
# Mixes two axes at unit weight: the column sums to 1 exactly as a permutation's does, and only its
# flattened peak refuses it.
_SHEARING = _LPS @ np.linalg.inv(np.asarray([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]))
# Averages all three axes into each output. Its ROWS sum to unit weight too, so every sum a
# permutation's matrix satisfies, this one satisfies: again only the peak is not there.
_AVERAGING = _LPS @ np.linalg.inv(np.asarray([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]))
# Reads a whole axis and a fraction of another on top: the peak IS a permutation's, and only the
# weight the column carries besides it refuses it.
_SUPERPOSING = _LPS @ np.linalg.inv(np.asarray([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def _canonical_attributes(direction: np.ndarray) -> Attribute:
    # Deliberately anisotropic on every axis, for the reason the extents are non-cubic: a spacing an
    # axis shares with another is a spacing a wrong permutation can carry onto it and look right.
    """
    Create anisotropic image metadata with the specified direction matrix.
    
    Parameters:
    	direction (np.ndarray): Direction matrix to store in flattened form.
    
    Returns:
    	Attribute: Image metadata containing origin, spacing, and direction.
    """
    attributes = Attribute()
    attributes["Origin"] = np.asarray([-3.0, 5.0, 11.0])
    attributes["Spacing"] = np.asarray([1.5, 1.75, 2.0])
    attributes["Direction"] = direction.reshape(-1)
    return attributes


def _ct_like_volume() -> torch.Tensor:
    # Distinct values everywhere: a repeated value could survive a wrong remap by coincidence, and
    # the multiset check below is only as strict as the volume is varied.
    """Create a channel-first volume with deterministic, distinct random values for canonicalization tests.
    
    Returns:
    	torch.Tensor: A float32 volume with shape ``[1, *_CANONICAL_SPATIAL]``.
    """
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.standard_normal(_CANONICAL_SPATIAL).astype(np.float32) * 500.0)[None]


@pytest.mark.parametrize(
    "direction, expected",
    [
        # LPS is what Canonical reorients ONTO: the one direction that asks for nothing at all.
        (_LPS, lambda volume: volume),
        # RAS mirrors physical x and y, and physical axis k is array axis dim - 1 - k: array axes 3 and 2.
        (_RAS, lambda volume: volume.flip((3, 2))),
        # Swapping physical x and z transposes the array axes carrying them (3 and 1), and mirrors the
        # two axes the canonical direction is negative on.
        (_PERMUTING, lambda volume: volume.permute(0, 3, 2, 1).flip((2, 3))),
    ],
    ids=["LPS", "RAS", "permuting"],
)
def test_canonical_is_the_exact_index_remap_bit_for_bit(direction: np.ndarray, expected) -> None:
    volume = _ct_like_volume()

    out = Canonical()("case", volume, _canonical_attributes(direction))

    assert torch.equal(out, expected(volume)), "an orthogonal reorientation must be the remap, not near it"


@pytest.mark.parametrize("direction", [_RAS, _LPS, _PERMUTING], ids=["RAS", "LPS", "permuting"])
def test_canonical_is_a_bijection_on_the_voxels(direction: np.ndarray) -> None:
    # The whole claim, stated as strongly as it can be: reorienting only moves values, so the sorted
    # multiset of them is bit-for-bit the input's. This is what LocalityKind.preserves_statistics lets a
    # later GLOBAL_STAT trust, and it is strictly stronger than comparing statistics -- a sampled
    # reorientation can leave a statistic looking right while having moved the values under it.
    volume = _ct_like_volume()

    out = Canonical()("case", volume, _canonical_attributes(direction))

    assert torch.equal(torch.sort(out.flatten())[0], torch.sort(volume.flatten())[0])
    # Reductions that do not depend on the order the voxels are visited in follow bit for bit.
    assert torch.min(out) == torch.min(volume)
    assert torch.max(out) == torch.max(volume)
    assert torch.std(out) == torch.std(volume)
    # A mean does depend on it: float addition is not associative, so summing the SAME multiset along a
    # mirrored axis can land an ulp from summing it along the original one. That is the reduction's
    # traversal, not the remap -- reduced in one order, or in a width the order cannot reach, it is 0.
    assert torch.mean(out.double()) == torch.mean(volume.double())


@pytest.mark.parametrize("direction", [_RAS, _LPS, _PERMUTING], ids=["RAS", "LPS", "permuting"])
def test_canonical_round_trips_exactly(direction: np.ndarray) -> None:
    # inverse() undoes a remap with a remap: an exact forward paired with a sampled inverse would put
    # the interpolation error back at prediction time, where the inverse is what the user is handed.
    # It restores the source EXTENT as well as the values -- a permutation transposed it on the way out.
    volume = _ct_like_volume()
    attributes = _canonical_attributes(direction)
    transform = Canonical()

    restored = transform.inverse("case", transform("case", volume, attributes), attributes)

    assert restored.shape == volume.shape
    assert torch.equal(restored, volume)
    # And the geometry it popped is the source's, so a chain's stack comes back to the depth it started.
    np.testing.assert_allclose(attributes.get_np_array("Origin"), [-3.0, 5.0, 11.0])
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [1.5, 1.75, 2.0])
    np.testing.assert_allclose(attributes.get_np_array("Direction"), direction.reshape(-1))


def test_canonical_records_the_canonical_geometry_from_the_volume_extent() -> None:
    # The new origin is the corner the reorientation mirrors onto, so it is the far end of the extent
    # on each mirrored axis and untouched on the others. A mirroring moves no axis, so the spacing stays.
    attributes = _canonical_attributes(_RAS)

    Canonical()("case", _ct_like_volume(), attributes)

    np.testing.assert_allclose(attributes.get_np_array("Direction"), np.diag([-1.0, -1.0, 1.0]).reshape(-1))
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [1.5, 1.75, 2.0])
    # Origin (x, y, z) = (-3, 5, 11); spacing (1.5, 1.75, 2.0); extents (x, y, z) = (11, 10, 9).
    np.testing.assert_allclose(
        attributes.get_np_array("Origin"),
        [-3.0 + (11 - 1) * 1.5, 5.0 + (10 - 1) * 1.75, 11.0],
    )


def test_canonical_permuting_records_the_grid_the_remap_lands_on() -> None:
    # An extent and a spacing travel with the axis they belong to, so swapping physical x and z carries
    # the z spacing onto x. What the reorientation preserves is the physical extent -- the volume's
    # centre is fixed -- so the origin is that centre stepped back by the TARGET half-extent.
    attributes = _canonical_attributes(_PERMUTING)

    Canonical()("case", _ct_like_volume(), attributes)

    np.testing.assert_allclose(attributes.get_np_array("Direction"), np.diag([-1.0, -1.0, 1.0]).reshape(-1))
    # Spacing (x, y, z) = (1.5, 1.75, 2.0) with x and z swapped.
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [2.0, 1.75, 1.5])
    # Centre = D @ half_extent_in + origin = (8, 7.875, 7.5) + (-3, 5, 11) = (5, 12.875, 18.5); target
    # half-extent = (8, 7.875, 7.5); LPS steps back negatively on x and y, positively on z.
    np.testing.assert_allclose(attributes.get_np_array("Origin"), [13.0, 20.75, 11.0])


@pytest.mark.parametrize(
    "direction, expected",
    [
        (_LPS, list(_CANONICAL_SPATIAL)),
        (_RAS, list(_CANONICAL_SPATIAL)),
        # Swapping physical x and z transposes the extents they carry: array (9, 10, 11) -> (11, 10, 9).
        (_PERMUTING, [11, 10, 9]),
        # An oblique direction is resampled onto the input's own grid, so it keeps its extent.
        (_OBLIQUE, list(_CANONICAL_SPATIAL)),
    ],
    ids=["LPS", "RAS", "permuting", "oblique"],
)
def test_canonical_folds_the_patch_grid_onto_the_extent_it_produces(direction: np.ndarray, expected: list[int]) -> None:
    # The patch grid is folded from transform_shape, so a permuting case's patches are cut on the grid
    # this returns -- which is only the right grid if it is the extent __call__ actually produces.
    transform = Canonical()
    attributes = _canonical_attributes(direction)

    folded = transform.transform_shape("Group", "case", list(_CANONICAL_SPATIAL), attributes)

    assert folded == expected
    assert list(transform("case", _ct_like_volume(), attributes).shape[1:]) == expected


def test_canonical_shape_fold_leaves_the_case_metadata_alone() -> None:
    # The attribute folded through transform_shape is the case's own, before a voxel has been read, and
    # it is what the streamed chain is later seeded from: writing the target geometry here would hand
    # every patch the reorientation's own result as the description of its input.
    attributes = _canonical_attributes(_PERMUTING)
    before = dict(attributes)

    Canonical().transform_shape("Group", "case", list(_CANONICAL_SPATIAL), attributes)

    assert dict(attributes) == before


@pytest.mark.parametrize(
    "direction, kind",
    [
        (_RAS, LocalityKind.ORIENTATION),
        (_LPS, LocalityKind.ORIENTATION),
        # Permuting is an exact remap too -- onto a grid of its own, which the patch grid is folded onto.
        (_PERMUTING, LocalityKind.ORIENTATION),
        # Genuinely mixes axes: no remap reproduces it, so it is resampled from the whole volume.
        (_OBLIQUE, LocalityKind.WHOLE_VOLUME),
        # Not orthonormal: no remap is a bijection on the voxels, whatever their columns attest.
        (_SHEARING, LocalityKind.WHOLE_VOLUME),
        (_AVERAGING, LocalityKind.WHOLE_VOLUME),
        (_SUPERPOSING, LocalityKind.WHOLE_VOLUME),
    ],
    ids=["RAS", "LPS", "permuting", "oblique", "shearing", "averaging", "superposing"],
)
def test_canonical_declares_orientation_only_where_it_is_an_exact_index_remap(
    direction: np.ndarray, kind: LocalityKind
) -> None:
    assert Canonical().patch_locality(_canonical_attributes(direction)).kind is kind


def test_canonical_declaration_is_total_on_absent_metadata() -> None:
    # The config-time checks probe with an empty Attribute, and a group carries only what its writer
    # stored: a missing direction must fall to the safe kind rather than raise.
    assert Canonical().patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
