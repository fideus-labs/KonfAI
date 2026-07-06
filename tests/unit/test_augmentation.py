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
from konfai.utils.dataset import Attribute
from konfai.utils.errors import AugmentationError

# --------------------------------------------------------------------------------------
# Per-sample state: draw caching, reset, and inverse slot lookup
# --------------------------------------------------------------------------------------


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

    augmented = flip._compute("case", 0, [dvf.clone()])[0]

    assert torch.equal(flip._inverse(0, 0, augmented), dvf)


def test_flip_vector_field_negates_flipped_components() -> None:
    # Mirroring a spatial axis reverses the voxel layout AND the sign of that axis' component channel
    # (channels are (dx, dy, dz) while tensor axes are reversed: dim 3 = x -> channel 0, ...).
    flip = _flip_all_axes(vector_field=True)
    dvf = torch.randn(3, 4, 5, 6)

    augmented = flip._compute("case", 0, [dvf.clone()])[0]

    layout_only = torch.flip(dvf, dims=[1, 2, 3])
    assert torch.equal(augmented, -layout_only)


def test_flip_scalar_data_is_layout_only() -> None:
    # Single-channel data (images, masks) is mirror-invariant: even with ``vector_field`` enabled the
    # shared Flip instance must not negate intensities.
    flip = _flip_all_axes(vector_field=True)
    volume = torch.randn(1, 4, 5, 6)

    augmented = flip._compute("case", 0, [volume.clone()])[0]

    assert torch.equal(augmented, torch.flip(volume, dims=[1, 2, 3]))
    assert torch.equal(flip._inverse(0, 0, augmented), volume)


def test_flip_default_stays_layout_only_on_vector_data() -> None:
    # ``vector_field`` is opt-in: existing intensity-TTA bundles keep the historical behaviour.
    flip = _flip_all_axes(vector_field=False)
    dvf = torch.randn(3, 4, 5, 6)

    augmented = flip._compute("case", 0, [dvf.clone()])[0]

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

    out = rotate._compute("case", 0, [labels.clone()])[0]

    assert out.dtype == torch.uint8
    assert set(out.unique().tolist()).issubset({0, 1, 3})
    assert 2 not in out.unique().tolist()


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
    column = aug.matrix[0][0][0, :3, 3]  # affine order (x, y, z)
    expected = torch.tensor([5.0 * 2 / (10 - 1), 5.0 * 2 / (6 - 1), 5.0 * 2 / (4 - 1)])
    assert torch.allclose(column, expected, atol=1e-6)


def test_translate_is_int_rounds_to_whole_voxels():
    """``is_int`` must round to entire voxels, not to two decimals (0.01)."""
    aug = Translate(t_min=5.3, t_max=5.3, is_int=True)
    aug._state_init(0, [[9, 9, 9]], [Attribute()])
    column = aug.matrix[0][0][0, :3, 3]
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
    augmentation._compute("case", 0, [torch.ones((1, 2, 2))])
    augmentation._compute("case", 0, [torch.ones((1, 2, 2))])
    assert read_count == 1
