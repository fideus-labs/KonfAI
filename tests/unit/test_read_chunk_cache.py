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

"""Read-side caching of chunked stores.

Overlapping patch reads revisit the same chunks: the HDF5 read handle carries a chunk cache sized for
imaging chunks (the library default holds barely one), and the OME-Zarr image handle is memoised per
(store, level) so a streamed run parses the NGFF metadata once, not once per patch — invalidated by
every write path, because a store just written must be re-read."""

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset

h5py = pytest.importorskip("h5py")


def test_h5_read_handle_carries_an_imaging_sized_chunk_cache(tmp_path) -> None:
    dataset = Dataset(f"{tmp_path}/store.h5", "h5")
    dataset.write("group", "CASE_000", np.zeros((1, 4, 4, 4), dtype=np.float32), Attribute())
    with Dataset.File(f"{tmp_path}/store.h5", True, "h5") as backend:
        _, nslots, nbytes, _ = backend.h5.id.get_access_plist().get_cache()
    assert nbytes == Dataset.H5File._READ_CHUNK_CACHE_BYTES
    assert nslots == Dataset.H5File._READ_CHUNK_CACHE_SLOTS


def test_ome_zarr_image_is_memoised_per_store_and_invalidated_by_writes(tmp_path) -> None:
    pytest.importorskip("ngff_zarr")
    from konfai.utils.ome_zarr import _load_image, read_ome_zarr_data_slice, write_ome_zarr

    store = tmp_path / "case.ome.zarr"
    write_ome_zarr(store, np.ones((1, 4, 4, 4), dtype=np.float32))
    first = _load_image(str(store), 0)
    assert _load_image(str(store), 0) is first, "the image handle must be memoised per (store, level)"

    write_ome_zarr(store, np.full((1, 4, 4, 4), 7.0, dtype=np.float32))
    data, _ = read_ome_zarr_data_slice(store, tuple(slice(None) for _ in range(4)))
    assert float(data.max()) == 7.0, "a write must invalidate the memo so the new voxels are read"
