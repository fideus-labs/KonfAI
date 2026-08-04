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

"""The public data surface: the OME-Zarr readers and the pyramid writer, as a capsule uses them.

These cover what a hand-rolled store reader gets wrong, which is never the happy path: a big-endian
source, a pyramid level that does not exist, and a downsampling default that changes the pixels."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")

from konfai.data import (
    append_ome_zarr_levels,
    create_ome_zarr_store,
    get_ome_zarr_info,
    read_ome_zarr_data_slice,
    write_ome_zarr,
)
from konfai.utils.errors import DatasetManagerError


def _volume(dtype: str = "<f4") -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((1, 8, 12, 16)) * 1000).astype(dtype)


def _big_endian_store(path: Path, volume: np.ndarray) -> Path:
    """A store that genuinely holds big-endian samples, as a light-sheet source does.

    Written through create_ome_zarr_store, because that is the one path that keeps the dtype it is
    given: write_ome_zarr would hand the array to ngff-zarr, which normalises on the way in and so
    could never exercise the reader.
    """
    array = create_ome_zarr_store(path, volume.shape, ">f4", spacing=[2.0, 1.0, 1.0])
    array[:] = volume.astype(">f4")
    return path


def test_the_public_surface_is_importable_from_konfai_data() -> None:
    """The locality vocabulary is the framework's distinctive feature; a user who cannot import it
    cannot make their transform stream at all."""
    import konfai.data as kd

    assert kd.PatchLocality(kd.LocalityKind.HALO, halo=(2,)).halo == (2,)
    for name in ("DatasetManager", "Transform", "Write", "Dataset", "LocalityKind", "PatchLocality"):
        assert name in kd.__all__ and hasattr(kd, name), name


def test_reader_returns_native_byte_order(tmp_path: Path) -> None:
    """Light-sheet sources are big-endian. A non-native array is refused outright by
    torch.from_numpy, and silently reinterpreted by anything that touches the raw buffer."""
    volume = _volume()
    store = _big_endian_store(tmp_path / "big_endian.ome.zarr", volume)

    patch, _ = read_ome_zarr_data_slice(store, (slice(None), slice(0, 4), slice(None), slice(None)))
    assert patch.dtype.isnative
    import torch

    torch.from_numpy(patch)  # the loud half of the bug: this raises on a non-native array
    np.testing.assert_array_equal(patch, volume[:, :4])


def test_info_publishes_the_shape_the_reader_indexes(tmp_path: Path) -> None:
    """`shape` is the store's own axis order and `canonical_shape` is the reader's; sizing slices
    from the wrong one reads a transposed region with the right rank and no error."""
    store = tmp_path / "v.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[2.0, 1.0, 1.0])
    info = get_ome_zarr_info(store)

    assert info["canonical_shape"] == [1, 8, 12, 16]
    patch, _ = read_ome_zarr_data_slice(store, tuple(slice(0, s) for s in info["canonical_shape"]))
    assert list(patch.shape) == info["canonical_shape"]


def test_out_of_range_level_says_so_instead_of_blaming_the_store(tmp_path: Path) -> None:
    store = tmp_path / "pyramid.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[2.0, 1.0, 1.0], scale_factors=[2])

    with pytest.raises(DatasetManagerError, match="level 5 is out of range"):
        get_ome_zarr_info(store, 5)


def test_write_ome_zarr_builds_a_pyramid_by_position(tmp_path: Path) -> None:
    store = tmp_path / "pyramid.ome.zarr"
    # spacing arrives (x, y, z) SimpleITK-style, so this is x=2 um, y=1, z=1 -- and the metadata
    # below is read back (c, z, y, x). The two orders meeting here is the whole reason to assert it.
    write_ome_zarr(store, _volume(), spacing=[2.0, 1.0, 1.0], scale_factors=[2])

    assert get_ome_zarr_info(store)["n_levels"] == 2
    fine, coarse = get_ome_zarr_info(store, 0), get_ome_zarr_info(store, 1)
    assert coarse["canonical_shape"] == [1, 4, 6, 8]
    # Level 1 is level 0 halved, and carries its OWN spacing -- a consumer indexing by position gets
    # a coarser image, not the same image mislabelled.
    assert [round(s, 6) for s in fine["scale"]] == [1.0, 1.0, 1.0, 2.0]
    assert [round(s, 6) for s in coarse["scale"]] == [1.0, 2.0, 2.0, 4.0]
    # And its origin is shifted by HALF the spacing delta, per axis: these stores use a
    # centre-of-voxel convention, so the coarse first voxel sits at the centre of the block it
    # averages. Reusing the fine origin biases every voxel and still looks like a plausible image.
    assert [round(t, 6) for t in coarse["translation"]] == [0.0, 0.5, 0.5, 1.0]


def test_default_downsampling_is_a_block_mean_not_a_gaussian(tmp_path: Path) -> None:
    """The default must not change the pixels: BIN_SHRINK is the block mean a capsule writes by
    hand, where ngff-zarr's own ITKWASM_GAUSSIAN default smooths and crushes the peak."""
    volume = _volume()
    store = tmp_path / "shrunk.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], scale_factors=[2])

    coarse, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    blocks = volume[:, :8, :12, :16].reshape(1, 4, 2, 6, 2, 8, 2)
    np.testing.assert_allclose(coarse, blocks.mean(axis=(2, 4, 6)), rtol=1e-6)


def test_unknown_downsample_method_names_the_valid_ones(tmp_path: Path) -> None:
    with pytest.raises(DatasetManagerError, match="ITKWASM_BIN_SHRINK"):
        write_ome_zarr(
            tmp_path / "x.ome.zarr", _volume(), spacing=[1.0] * 3, scale_factors=[2], downsample_method="NOPE"
        )


def test_append_levels_turns_a_streamed_store_into_a_pyramid(tmp_path: Path) -> None:
    """The region-written path cannot take scale_factors up front — no level exists until the last
    region lands — so the pyramid is derived afterwards, and level 0 must survive it untouched."""
    volume = _volume()
    store = tmp_path / "streamed.ome.zarr"
    write_ome_zarr(store, volume, spacing=[2.0, 1.0, 1.0], attributes={"Direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]})
    assert get_ome_zarr_info(store)["n_levels"] == 1

    append_ome_zarr_levels(store, [2])

    assert get_ome_zarr_info(store)["n_levels"] == 2
    level0, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=0)
    np.testing.assert_array_equal(level0, volume)
    # The sidecar survives the rewrite: losing Direction is silent, the reader falls back to identity.
    assert get_ome_zarr_info(store)["attributes"]["Direction"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_append_levels_without_factors_is_a_no_op(tmp_path: Path) -> None:
    store = tmp_path / "v.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[1.0] * 3)
    append_ome_zarr_levels(store, [])
    assert get_ome_zarr_info(store)["n_levels"] == 1
