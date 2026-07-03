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

import torch
from konfai.data.augmentation import Flip
from konfai.utils.dataset import Attribute


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
