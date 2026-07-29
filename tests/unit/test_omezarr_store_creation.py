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

"""Unit tests for ``create_ome_zarr_store``: the properties a region-by-region write depends on.

``write_ome_zarr`` and ``create_ome_zarr_store`` now share ngff-zarr as their single source of
OME-NGFF metadata, which is what these tests are really protecting -- the alternative, a hand-built
``multiscales`` entry, drifts from the spec silently and can only ever describe what its author knew.
What that sharing must not cost is any of the guarantees the streaming path relies on, and most of
them were previously covered only end to end, where a regression shows up as a wrong volume rather
than as a wrong store.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")

import zarr
from konfai.utils.ome_zarr import (
    clear_ome_zarr_cache,
    create_ome_zarr_store,
    get_ome_zarr_info,
)

SPACING = (2.0, 1.5, 0.5)
ORIGIN = (7.0, -3.0, 10.0)
# A non-identity direction: an identity one lets a dropped sidecar pass unnoticed, because the reader
# falls back to identity exactly when the attribute is missing.
DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
SHAPE = (3, 8, 10, 12)


def _create(path: Path, **kwargs: Any) -> Any:
    attributes = {
        "Spacing": " ".join(str(value) for value in SPACING),
        "Origin": " ".join(str(value) for value in ORIGIN),
        "Direction": " ".join(str(value) for value in DIRECTION),
    }
    return create_ome_zarr_store(path, SHAPE, np.int16, spacing=SPACING, origin=ORIGIN, attributes=attributes, **kwargs)


def test_requested_chunks_are_the_store_chunks(tmp_path: Path) -> None:
    """The caller's chunk grid must survive untouched.

    A store whose chunks straddle the region grid turns every region write into a read-modify-write,
    and the grid is the one thing a writer handed only a shape cannot infer. Only the producer knows
    it, so only the producer can choose.
    """
    array = _create(tmp_path / "chunked.ome.zarr", chunks=(3, 4, 5, 6))
    assert tuple(array.chunks) == (3, 4, 5, 6)


def test_store_shape_and_dtype_are_exactly_what_was_asked(tmp_path: Path) -> None:
    array = _create(tmp_path / "typed.ome.zarr")
    assert tuple(array.shape) == SHAPE
    assert array.dtype == np.dtype(np.int16)


def test_unwritten_regions_read_back_as_zero(tmp_path: Path) -> None:
    """Region writes leave the rest of the volume untouched, and untouched must mean zero.

    Anything a stream does not reach has to read back as zero rather than as whatever the store
    happens to default to, because a partial write is a normal outcome here, not a failure.
    """
    array = _create(tmp_path / "sparse.ome.zarr", chunks=(3, 4, 5, 6))
    array[:, 0:4, 0:5, 0:6] = np.full((3, 4, 5, 6), 7, dtype=np.int16)

    np.testing.assert_array_equal(np.asarray(array[:, 0:4, 0:5, 0:6]), 7)
    np.testing.assert_array_equal(np.asarray(array[:, 4:8, 5:10, 6:12]), 0)


def test_store_is_readable_before_any_region_is_written(tmp_path: Path) -> None:
    """Metadata lands first, so a stream interrupted at any point leaves a valid store behind.

    This is also what lets a reader open an entry while it is still being written, which the dataset
    layer relies on when it replaces an existing entry.
    """
    path = tmp_path / "empty.ome.zarr"
    _create(path)
    clear_ome_zarr_cache()

    info = get_ome_zarr_info(path)
    assert info["shape"] == list(SHAPE)
    assert info["dtype"] == "int16"
    # spacing and origin arrive in SimpleITK's xyz order and are stored per axis, so the spatial part
    # of the store geometry reads back reversed. Asserting the reversal rather than a symmetric value
    # is the point: a transposed geometry is the failure this test exists to catch.
    np.testing.assert_allclose(info["scale"][1:], SPACING[::-1])
    np.testing.assert_allclose(info["translation"][1:], ORIGIN[::-1])


def test_konfai_attributes_survive_the_metadata_write(tmp_path: Path) -> None:
    """The sidecar must outlive ngff-zarr's own root write.

    ``to_ngff_zarr(overwrite=True)`` reopens the root group in ``mode="w"`` and drops every attribute
    already there, so ordering is the whole test: written first, Direction disappears, and its loss is
    silent -- the reader substitutes identity and returns a plausibly oriented volume.
    """
    path = tmp_path / "attrs.ome.zarr"
    _create(path)
    clear_ome_zarr_cache()

    stored = get_ome_zarr_info(path)["attributes"]
    assert [float(value) for value in stored["Direction"].split()] == list(DIRECTION)


def test_level_zero_key_is_taken_from_the_metadata_not_a_literal(tmp_path: Path) -> None:
    """Whatever key ngff-zarr chose, the array returned is the one the metadata points at.

    The key moved from ``"0"`` to ``"scale0/image"`` when ngff-zarr took over the metadata; hardcoding
    either spelling makes KonfAI break on a convention that is not its own.

    Reading through the group, rather than through ngff-zarr, is also the only thing here that sees the
    store's consolidated index -- a summary of its own contents that readers trust over the arrays
    themselves, and that is written while the array is still a one-voxel stand-in.
    """
    path = tmp_path / "keyed.ome.zarr"
    array = _create(path)
    array[:] = 3

    group = zarr.open_group(str(path), mode="r")
    multiscales = dict(group.attrs)["multiscales"][0]
    dataset_key = multiscales["datasets"][0]["path"]
    np.testing.assert_array_equal(np.asarray(group[dataset_key][:]), 3)


def test_creating_a_store_writes_no_pixel_bytes(tmp_path: Path) -> None:
    """Describing the store must not cost the volume.

    Handing a writer an array of the target shape, even an empty one, means a pass over every chunk of
    it before the first real voxel arrives -- around 33 s for a 13.6 GiB field, to produce nothing.
    Creation has to stay independent of the extent.
    """
    path = tmp_path / "bytes.ome.zarr"
    _create(path, chunks=(3, 4, 5, 6))

    written = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    metadata_only = 64 << 10
    assert written < metadata_only, f"{written} bytes written for an untouched store"
