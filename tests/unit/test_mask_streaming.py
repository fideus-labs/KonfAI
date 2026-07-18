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
"""``Mask`` streams slab by slab: ``stream_slab`` reads only the aligned mask region and reassembles
byte-identically to the whole-volume ``__call__``, so a finalize ``Mask`` keeps the case on the
streamed path."""

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai.data.transform import Attribute, LocalityKind, Mask  # noqa: E402


def _write_mask(path, z, y, x, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((z, y, x)) > 0.4).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


@pytest.mark.parametrize("slab", [1, 4, 5])
def test_mask_stream_slab_reassembles_like_whole_volume(tmp_path, slab):
    # Single-channel output (the real case: a masked sCT); a [1,Z,Y,X] mask indexes a 1-channel volume.
    z, y, x = 12, 8, 7
    path = tmp_path / "mask.mha"
    _write_mask(path, z, y, x)
    torch.manual_seed(1)
    volume = torch.randn(1, z, y, x)

    # SLAB declaration is what routes it through the streamed-write dispatcher.
    m = Mask(path=str(path), value_outside=-999)
    assert m.patch_locality(Attribute()).kind is LocalityKind.SLAB

    whole = m("case", volume.clone(), Attribute())

    streamed = volume.clone()
    for z0 in range(0, z, slab):
        z1 = min(z0 + slab, z)
        rows = m.stream_slab("case", streamed[:, z0:z1].clone(), slice(z0, z1), [z, y, x], Attribute())
        streamed[:, z0:z1] = rows

    assert torch.equal(streamed, whole)


def test_mask_stream_slab_masks_the_right_voxels(tmp_path):
    # A concrete check the region is aligned, not just self-consistent: outside the mask -> value_outside,
    # inside -> untouched, and the streamed slabs land on the same voxels as the whole-volume call.
    z, y, x = 6, 5, 4
    path = tmp_path / "m.mha"
    _write_mask(path, z, y, x, seed=3)
    mask = torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(str(path)))).unsqueeze(0)
    m = Mask(path=str(path), value_outside=-1)
    out = torch.ones(1, z, y, x) * 7.0
    for z0 in range(0, z, 2):
        out[:, z0 : z0 + 2] = m.stream_slab("c", out[:, z0 : z0 + 2].clone(), slice(z0, z0 + 2), [z, y, x], Attribute())
    assert torch.equal(out[mask == 0], torch.full_like(out[mask == 0], -1.0))
    assert torch.equal(out[mask != 0], torch.full_like(out[mask != 0], 7.0))
