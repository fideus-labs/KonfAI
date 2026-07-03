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

"""Regression tests for the data-augmentation inverse/state-init fixes."""

import os

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import torch
from konfai.data.augmentation import (
    Brightness,
    CutOUT,
    Flip,
    Noise,
    Rotate,
    Translate,
)
from konfai.utils.dataset import Attribute


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
