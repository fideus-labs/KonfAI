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

"""Tests for ``konfai.data.augmentation``: per-sample state (draw/reset/inverse),
Flip (incl. vector fields), Rotate, Translate, intensity augmentations, and the
SimpleITK-backed Elastix/Mask augmentations."""

from pathlib import Path

import konfai.data.augmentation as augmentation_module
import numpy as np
import pytest
import torch
from konfai.data.augmentation import (
    Brightness,
    CutOUT,
    Elastix,
    Flip,
    Mask,
    Noise,
    Rotate,
    Translate,
)
from konfai.data.transform import LocalityKind
from konfai.utils.dataset import Attribute
from konfai.utils.errors import AugmentationError

# --------------------------------------------------------------------------------------
# Per-sample state: draw caching, reset, and inverse slot lookup
# --------------------------------------------------------------------------------------


def test_hue_axis_rotation_preserves_luma() -> None:
    # Hue rotation is a rotation of RGB about the luma axis: it must be identity at theta=0 and leave a
    # grey pixel unchanged for any angle (an Euler XYZ rotation about the coordinate axes would recolour it).
    from konfai.data.augmentation import _axis_rotation_matrix

    v = torch.tensor([1.0, 1.0, 1.0]) / torch.sqrt(torch.tensor(3.0))
    assert torch.allclose(_axis_rotation_matrix(torch.tensor(0.0), v), torch.eye(4), atol=1e-6)
    grey = torch.tensor([0.5, 0.5, 0.5, 1.0])
    for theta in (0.3, 0.7, 1.5):
        assert torch.allclose(_axis_rotation_matrix(torch.tensor(theta), v) @ grey, grey, atol=1e-5)


def test_saturation_matrix_scales_chroma_not_luma() -> None:
    # v vT + (I - v vT) * s : s=1 is identity, s=0 collapses a colour to its luma (greyscale), and luma is
    # preserved for any s, unlike the old (v vT + (I - v vT)) * s = I * s which was a uniform gain.
    v = torch.tensor([1.0, 1.0, 1.0, 0.0]) / torch.sqrt(torch.tensor(3.0))
    colour = torch.tensor([0.8, 0.2, 0.5, 0.0])
    luma = colour[:3].mean()
    for s in (1.0, 0.0, 2.0):
        matrix = v.ger(v) + (torch.eye(4) - v.ger(v)) * s
        out = matrix @ colour
        assert torch.allclose(out[:3].mean(), luma, atol=1e-5)
    assert torch.allclose((v.ger(v) + (torch.eye(4) - v.ger(v)) * 1.0) @ colour, colour, atol=1e-5)
    grey = (v.ger(v) + (torch.eye(4) - v.ger(v)) * 0.0) @ colour
    assert torch.allclose(grey[:3], torch.full((3,), luma), atol=1e-5)


def test_augmentation_resamples_after_reset_state():
    """#1 Augmentation parameters must be re-sampled each epoch via reset_state.

    Within an epoch ``state_init`` caches the per-case draw so every patch shares
    one transform; ``reset_state`` must clear that cache so the next epoch draws
    fresh parameters (previously ``who_index`` was never cleared → frozen forever).
    """
    aug = Flip([1.0, 1.0, 1.0])
    aug.load(1.0)

    aug.state_init(0, [[4, 4, 4]], [Attribute()])
    first = aug.flip[0]
    # Re-running state_init without a reset returns the cached draw unchanged.
    aug.state_init(0, [[4, 4, 4]], [Attribute()])
    assert aug.flip[0] is first

    aug.reset_state(0)
    assert 0 not in aug.who_index
    aug.state_init(0, [[4, 4, 4]], [Attribute()])
    assert 0 in aug.who_index
    assert aug.flip[0] is not first  # a fresh draw replaced the cached one


def test_augmentation_inverse_uses_local_slot_for_global_index():
    """``inverse(a)`` receives the *global* sample index.

    Per-sample state (``flip``/``matrix``) is stored only for the selected
    samples, in selection order (local slots). The global index must therefore
    be translated to the sample's position within ``who_index`` before indexing
    that state, otherwise a selected sample either reads a neighbour's transform
    (silent corruption) or overruns the list (IndexError).
    """
    aug = Flip([1.0, 0.0, 0.0])
    # Samples 1 and 2 were selected; their flip axes occupy local slots 0 and 1.
    aug.who_index[0] = [1, 2]
    aug.flip[0] = [[1], [2]]  # global 1 -> axis 1, global 2 -> axis 2

    x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    # Unselected sample is passed through untouched.
    assert torch.equal(aug.inverse(0, 0, x.clone()), x)
    # Global 1 -> local slot 0 -> axis 1 (previously read slot 1 -> axis 2).
    assert torch.equal(aug.inverse(0, 1, x.clone()), torch.flip(x, [1]))
    # Global 2 -> local slot 1 -> axis 2 (previously an out-of-range index).
    assert torch.equal(aug.inverse(0, 2, x.clone()), torch.flip(x, [2]))


def test_intensity_augmentation_inverses_are_identity():
    """Value-only augmentations must invert to the identity.

    ColorTransform/Noise/CutOUT do not move voxels, so the inverse applied to a
    prediction is the tensor itself. They previously returned ``None``, which
    crashed the prediction inverse path (``NoneType`` has no ``device``).
    """
    x = torch.randn(3, 4, 5, 6)  # 3 channels for the ColorTransform path

    color = Brightness(b_std=0.5)
    color.who_index[0] = [0]
    assert torch.equal(color.inverse(0, 0, x.clone()), x)

    noise = Noise(n_std=0.1)
    noise.who_index[0] = [0]
    assert torch.equal(noise.inverse(0, 0, x.clone()), x)

    cutout = CutOUT(c_prob=1.0, cutout_size=2, value=0.0)
    cutout.who_index[0] = [0]
    assert torch.equal(cutout.inverse(0, 0, x.clone()), x)


# --------------------------------------------------------------------------------------
# Flip — vector-field awareness
# --------------------------------------------------------------------------------------


def _flip_all_axes(vector_field: bool) -> Flip:
    flip = Flip(f_prob=[1.0, 1.0, 1.0], vector_field=vector_field)
    flip._state_init(0, [[4, 5, 6]], [Attribute()])
    return flip


def test_flip_vector_field_round_trip_is_identity() -> None:
    # TTA un-flips the model output with ``_inverse``: on a displacement field the compose of
    # ``_compute`` and ``_inverse`` must be the identity, component signs included.
    flip = _flip_all_axes(vector_field=True)
    dvf = torch.randn(3, 4, 5, 6)

    augmented = flip._compute("case", 0, 0, dvf.clone())

    assert torch.equal(flip._inverse(0, 0, augmented), dvf)


def test_flip_vector_field_negates_flipped_components() -> None:
    # Mirroring a spatial axis reverses the voxel layout AND the sign of that axis' component channel
    # (channels are (dx, dy, dz) while tensor axes are reversed: dim 3 = x -> channel 0, ...).
    flip = _flip_all_axes(vector_field=True)
    dvf = torch.randn(3, 4, 5, 6)

    augmented = flip._compute("case", 0, 0, dvf.clone())

    layout_only = torch.flip(dvf, dims=[1, 2, 3])
    assert torch.equal(augmented, -layout_only)


def test_flip_scalar_data_is_layout_only() -> None:
    # Single-channel data (images, masks) is mirror-invariant: even with ``vector_field`` enabled the
    # shared Flip instance must not negate intensities.
    flip = _flip_all_axes(vector_field=True)
    volume = torch.randn(1, 4, 5, 6)

    augmented = flip._compute("case", 0, 0, volume.clone())

    assert torch.equal(augmented, torch.flip(volume, dims=[1, 2, 3]))
    assert torch.equal(flip._inverse(0, 0, augmented), volume)


def test_flip_default_stays_layout_only_on_vector_data() -> None:
    # ``vector_field`` is opt-in: existing intensity-TTA bundles keep the historical behaviour.
    flip = _flip_all_axes(vector_field=False)
    dvf = torch.randn(3, 4, 5, 6)

    augmented = flip._compute("case", 0, 0, dvf.clone())

    assert torch.equal(augmented, torch.flip(dvf, dims=[1, 2, 3]))


# --------------------------------------------------------------------------------------
# Rotate
# --------------------------------------------------------------------------------------


def test_rotate_converts_degrees_to_radians():
    """#6 A 90-degree rotation must yield [[0,-1],[1,0]], not cos/sin of 90 radians."""
    rot = Rotate(a_min=90.0, a_max=90.0, is_quarter=False)
    rot._state_init(0, [[8, 8]], [Attribute()])
    block = rot.matrix[0][0][0, :2, :2]
    assert torch.allclose(block, torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-5)


def test_rotate_quarter_builds_one_signed_permutation_per_sample():
    """``is_quarter`` must draw one rotation per sample/axis.

    The previous flat 9-vector produced 0-d angles that crashed
    ``_rotation_3d_matrix`` and ignored the sample count. Each resulting matrix
    must be a proper multiple-of-90-degree rotation, i.e. an orthogonal signed
    permutation matrix (integer entries, determinant +1).
    """
    torch.manual_seed(0)
    rot = Rotate(is_quarter=True)
    shapes = [[8, 8, 8], [8, 8, 8]]
    rot._state_init(0, shapes, [Attribute(), Attribute()])

    assert len(rot.matrix[0]) == len(shapes)  # one matrix per sample, not 9
    identity = torch.eye(3)
    for matrix in rot.matrix[0]:
        assert matrix.shape == (1, 4, 4)
        rotation = matrix[0, :3, :3]
        assert torch.allclose(rotation @ rotation.T, identity, atol=1e-5)
        assert torch.allclose(rotation, rotation.round(), atol=1e-5)
        assert torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-5)


def test_rotate_preserves_label_ids_with_nearest() -> None:
    # A geometric augmentation applies to every group, including uint8 segmentation targets.
    # Bilinear resampling would blend class ids into a non-existent intermediate label (1|3 -> 2);
    # nearest-neighbour must keep the label set unchanged.
    rotate = Rotate(a_min=45.0, a_max=45.0)  # a_max == a_min -> deterministic 45 degrees
    labels = torch.zeros(1, 32, 32, dtype=torch.uint8)
    labels[:, 8:24, 8:24] = 1
    labels[:, 12:20, 12:20] = 3
    rotate._state_init(0, [[32, 32]], [Attribute()])

    out = rotate._compute("case", 0, 0, labels.clone())

    assert out.dtype == torch.uint8
    assert set(out.unique().tolist()).issubset({0, 1, 3})
    assert 2 not in out.unique().tolist()


def _rotate_volume(spatial: tuple[int, ...]) -> torch.Tensor:
    # Distinct values everywhere: a repeated one could survive a wrong remap by coincidence, and the
    # multiset check below is only as strict as the volume is varied.
    """Create a single-channel volume with deterministic, distinct random values.
    
    Parameters:
    	spatial (tuple[int, ...]): Spatial dimensions of the volume.
    
    Returns:
    	torch.Tensor: A float32 tensor with shape `(1, *spatial)`.
    """
    return torch.from_numpy((np.random.default_rng(0).standard_normal(spatial) * 500.0).astype(np.float32))[None]


@pytest.mark.parametrize("seed", range(8))
def test_rotate_quarter_is_an_exact_bijection_on_a_cubic_grid(seed: int) -> None:
    # A multiple of 90 degrees about each axis composes to a signed permutation of the axes, so it only
    # moves voxels: the sorted multiset of values must come back bit for bit. This is what
    # LocalityKind.preserves_statistics lets a later stage trust, and it is strictly stronger than
    # comparing statistics -- a sampled turn can leave one looking right while moving the values under it.
    """
    Verify that a quarter-turn rotation preserves voxel values and matches the sampled transform on a cubic volume.
    
    Parameters:
    	seed (int): Random seed used to generate the test volume.
    """
    torch.manual_seed(seed)
    volume = _rotate_volume((12, 12, 12))
    rotate = Rotate(is_quarter=True)
    rotate._state_init(0, [[12, 12, 12]], [Attribute()])

    out = rotate._compute("case", 0, 0, volume)

    assert out.shape == volume.shape
    assert torch.equal(torch.sort(out.flatten())[0], torch.sort(volume.flatten())[0])
    assert torch.min(out) == torch.min(volume)
    assert torch.max(out) == torch.max(volume)
    assert torch.std(out) == torch.std(volume)
    # And it is the SAME turn the sampler describes, not merely some bijection: a different permutation
    # would disagree by the data's own range, where interpolating this one disagrees by ~1e-3 of it.
    sampled = rotate._sample(rotate._grid_matrix(0, 0, [12, 12, 12]), volume)
    assert (out.double() - sampled.double()).abs().max() < 0.05


@pytest.mark.parametrize("seed", range(8))
def test_rotate_quarter_inverse_restores_the_volume_exactly(seed: int) -> None:
    # An exact turn undone by a sampled one would put the interpolation error back at prediction time.
    torch.manual_seed(seed)
    volume = _rotate_volume((12, 12, 12))
    rotate = Rotate(is_quarter=True)
    rotate._state_init(0, [[12, 12, 12]], [Attribute()])

    assert torch.equal(rotate._inverse(0, 0, rotate._compute("case", 0, 0, volume)), volume)


@pytest.mark.parametrize("seed", range(12))
def test_rotate_quarter_transposes_a_non_cubic_grid_exactly(seed: int) -> None:
    # A 90 degree turn transposes the two extents it swaps, so the copy it draws is cut on the grid the
    # turn lands on -- which _state_init announces and the patch grid is loaded from. The remap stays a
    # bijection on the voxels whatever the extents: only where they sit changes.
    torch.manual_seed(seed)
    volume = _rotate_volume((9, 10, 11))
    rotate = Rotate(is_quarter=True)

    drawn = rotate._state_init(0, [[9, 10, 11]], [Attribute()])
    out = rotate._compute("case", 0, 0, volume)

    assert list(out.shape[1:]) == drawn[0], "the copy must be cut on the grid its draw announced"
    assert sorted(drawn[0]) == [9, 10, 11], "a turn permutes the extents, it does not invent them"
    assert torch.equal(torch.sort(out.flatten())[0], torch.sort(volume.flatten())[0])
    # And the inverse brings back the source extent as well as the values.
    assert torch.equal(rotate._inverse(0, 0, out), volume)


def test_rotate_declares_orientation_from_the_draw_not_from_the_flag() -> None:
    # The declaration is about the turn that was drawn, so a free range pinned to a right angle is just
    # as exact as a quarter draw, and a free angle is not -- whatever the extents it was drawn for.
    torch.manual_seed(0)
    right_angle = Rotate(a_min=90.0, a_max=90.0)
    right_angle._state_init(0, [[9, 10, 11]], [Attribute()])
    assert right_angle._patch_locality(0, 0, Attribute()).kind is LocalityKind.ORIENTATION

    free = Rotate(a_min=10.0, a_max=10.0)
    free._state_init(0, [[12, 12, 12]], [Attribute()])
    assert free._patch_locality(0, 0, Attribute()).kind is LocalityKind.WHOLE_VOLUME


# --------------------------------------------------------------------------------------
# Translate
# --------------------------------------------------------------------------------------


def test_translate_scales_voxel_offset_to_normalized_grid():
    """``t_min``/``t_max`` are voxel offsets.

    ``F.affine_grid`` expects normalized coordinates where a full axis spans
    [-1, 1] (align_corners=True), so a d-voxel shift becomes d * 2 / (size - 1),
    per axis in affine order (x, y, z) = reversed spatial (z, y, x).
    """
    aug = Translate(t_min=5.0, t_max=5.0, is_int=False)  # deterministic 5-voxel shift
    aug._state_init(0, [[4, 6, 10]], [Attribute()])  # spatial (z, y, x)
    column = aug._grid_matrix(0, 0, [4, 6, 10])[0, :3, 3]  # affine order (x, y, z)
    expected = torch.tensor([5.0 * 2 / (10 - 1), 5.0 * 2 / (6 - 1), 5.0 * 2 / (4 - 1)])
    assert torch.allclose(column, expected, atol=1e-6)


def test_translate_is_int_rounds_to_whole_voxels():
    """``is_int`` must round to entire voxels, not to two decimals (0.01)."""
    aug = Translate(t_min=5.3, t_max=5.3, is_int=True)
    aug._state_init(0, [[9, 9, 9]], [Attribute()])
    column = aug._grid_matrix(0, 0, [9, 9, 9])[0, :3, 3]
    expected = torch.full((3,), 5.0 * 2 / (9 - 1))  # round(5.3) == 5, then normalized
    assert torch.allclose(column, expected, atol=1e-6)
    # Neither the pre-fix 0.01 rounding (5.3) nor raw-voxel units survive.
    assert not torch.allclose(column, torch.full((3,), 5.3), atol=1e-6)


# --------------------------------------------------------------------------------------
# SimpleITK-backed augmentations (Elastix / Mask)
# --------------------------------------------------------------------------------------


def test_simpleitk_augmentations_fail_clearly_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(augmentation_module, "sitk", None)

    with pytest.raises(AugmentationError, match="SimpleITK"):
        Elastix()
    with pytest.raises(AugmentationError, match="SimpleITK"):
        Mask("mask.mha", 0)


def test_mask_reads_pixels_only_on_first_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sitk = pytest.importorskip("SimpleITK")
    mask_path = tmp_path / "mask.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2), dtype=np.uint8)), str(mask_path))

    read_count = 0
    original_read_image = sitk.ReadImage

    def counting_read_image(path: str):
        nonlocal read_count
        read_count += 1
        return original_read_image(path)

    monkeypatch.setattr(augmentation_module.sitk, "ReadImage", counting_read_image)
    augmentation = Mask(str(mask_path), 0)
    augmentation._state_init(0, [[2, 2]], [Attribute()])

    assert read_count == 0
    augmentation._compute("case", 0, 0, torch.ones((1, 2, 2)))
    augmentation._compute("case", 0, 0, torch.ones((1, 2, 2)))
    assert read_count == 1
