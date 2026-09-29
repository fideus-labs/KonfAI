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

from pathlib import Path

import pytest

# ---------------------------------------------------------------- what a whole-volume scan decodes


def test_a_scan_raises_its_grain_onto_the_store_block_it_would_decode_anyway() -> None:
    """A chunked store decodes whole blocks, so a scan stepping finer decodes the same block again
    at every step inside it -- measured at 85x, and 170x where the step straddled two. Where the
    budget holds a whole block the grain is raised to it, which reads each block once."""
    from konfai.utils.dataset.statistics import _scan_block_on_the_store_grid

    rows, held = _scan_block_on_the_store_grid(rows=3, extent=512, plane=1000, granularity=[64], budget=1 << 30)
    assert rows == 64, "raised onto the grid"
    assert held == 64 * 3 * 1000 * 4, "an aligned block decodes itself and nothing more"


def test_a_scan_that_cannot_afford_a_whole_block_is_charged_for_the_one_it_decodes() -> None:
    """Where the budget cannot hold a stored block the grain stays fine -- and what the store
    decodes is charged, so the plan refuses instead of the kernel."""
    from konfai.utils.dataset.statistics import _scan_block_on_the_store_grid

    tight = 4 * 1000 * 4 * 8  # far under one 64-row block
    rows, held = _scan_block_on_the_store_grid(rows=3, extent=512, plane=1000, granularity=[64], budget=tight)
    assert rows == 3, "the grain the budget bought"
    assert held > 3 * 3 * 1000 * 4, "and the block it really decodes, charged"


def test_an_unchunked_store_is_charged_for_what_it_is_asked_for() -> None:
    from konfai.utils.dataset.statistics import _scan_block_on_the_store_grid

    rows, held = _scan_block_on_the_store_grid(rows=7, extent=512, plane=1000, granularity=None, budget=1 << 30)
    assert (rows, held) == (7, 7 * 3 * 1000 * 4)


# ---------------------------------------------------------------- what a memmapped store serves


def test_a_memmapped_store_declares_the_band_a_region_read_touches(tmp_path: "Path") -> None:
    """A memmap is served band by band: the read maps the outermost axis the window spans and every
    axis below it whole, so a region narrower than a plane touches a plane's pages and the kernel
    counts them. Priced on the voxels it asked for instead, the shape search answered a cube under
    96 MiB and the run held MORE at a smaller budget (77 MiB at 64 against 58 at 96).

    Declared the way a chunked store declares its chunk, so one mechanism prices both.
    """
    import numpy as np
    import SimpleITK as sitk
    from konfai.utils.dataset import Dataset, chunk_hull_voxels

    case = tmp_path / "Dataset" / "CASE_000"
    case.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.zeros((40, 24, 16), dtype=np.int16))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(case / "CT.mha"), useCompression=False)

    granularity = Dataset(str(tmp_path / "Dataset"), "mha").read_granularity("CT", "CASE_000")
    assert granularity == (1, 1, 24, 16), "one step along the banded axis, everything below it whole"

    # And the hull that grain implies IS the band: a cube costs its rows times the whole plane.
    spatial, cube = [40, 24, 16], (slice(8, 16), slice(4, 12), slice(2, 10))
    assert chunk_hull_voxels(cube, granularity[1:], spatial) == 8 * 24 * 16
    slab = (slice(8, 16), slice(0, 24), slice(0, 16))
    assert chunk_hull_voxels(slab, granularity[1:], spatial) == 8 * 24 * 16, "a full-plane slab costs itself"


def test_a_chunked_h5_entry_declares_its_chunk_and_a_contiguous_one_declares_none(tmp_path: "Path") -> None:
    """An HDF5 hyperslab decodes every chunk it touches whole, so a chunked entry (a third-party
    store, or a cache a region write chunked on its region) declares its chunk as the grain a sweep
    is cut on; what ``Dataset.write`` stores is contiguous and declares none: a read costs the
    bytes it covers."""
    import h5py
    import numpy as np
    from konfai.utils.dataset import Attribute, Dataset

    store = tmp_path / "Dataset.h5"
    with h5py.File(store, "w") as file:
        file.create_dataset("CT/CHUNKED", data=np.zeros((1, 128, 16, 16), np.float32), chunks=(1, 64, 16, 16))
    dataset = Dataset(str(store), "h5")
    dataset.write("CT", "PLAIN", np.zeros((1, 8, 16, 16), np.float32), Attribute())

    assert dataset.read_granularity("CT", "CHUNKED") == (1, 64, 16, 16)
    assert dataset.read_granularity("CT", "PLAIN") is None
    assert dataset.bounded_region_reads("CT", "CHUNKED")


# ---------------------------------------------------------------- an entry the backend cannot read


def _one_case(root: "Path", file_format: str) -> None:
    import numpy as np
    from konfai.utils.dataset import Attribute, Dataset

    attribute = Attribute()
    attribute["Origin"], attribute["Spacing"], attribute["Direction"] = np.zeros(3), np.ones(3), np.eye(3).flatten()
    Dataset(str(root), file_format).write("CT", "CASE_000", np.ones((1, 4, 8, 8), dtype=np.float32), attribute)


def _spoil(root: "Path", how: str) -> None:
    for path in (path for path in root.rglob("*") if path.is_file()):
        data = path.read_bytes()
        path.write_bytes(data[: len(data) * 7 // 10] if how == "truncated" else b"not an image\n")


@pytest.mark.parametrize(("file_format", "how"), [("mha", "truncated"), ("mha", "garbage"), ("h5", "garbage")])
def test_an_entry_the_backend_cannot_decode_is_a_case_read_error_naming_it(
    tmp_path: "Path", file_format: str, how: str
) -> None:
    """What the backend's library raises on a corrupt file (SimpleITK's RuntimeError, h5py's OSError)
    comes out as one error the one-pass workflows can set the case aside on, naming case and entry."""
    from konfai.utils.dataset import Dataset
    from konfai.utils.errors import CaseReadError

    _one_case(tmp_path / "Dataset", file_format)
    _spoil(tmp_path, how)

    with pytest.raises(CaseReadError, match=r"'CT' entry of case 'CASE_000'") as raised:
        Dataset(str(tmp_path / "Dataset"), file_format).read_data("CT", "CASE_000")
    assert isinstance(raised.value.__cause__, (RuntimeError, OSError))


def test_an_out_of_memory_in_a_read_is_not_a_read_error(tmp_path: "Path", monkeypatch: pytest.MonkeyPatch) -> None:
    """Only what the backend declares as its read errors is a case's fault: a MemoryError stops the run."""
    from konfai.utils.dataset import Dataset
    from konfai.utils.dataset.sitk_file import SitkFile

    _one_case(tmp_path / "Dataset", "mha")

    def no_memory(self, group, name):
        raise MemoryError()

    monkeypatch.setattr(SitkFile, "file_to_data", no_memory)
    with pytest.raises(MemoryError):
        Dataset(str(tmp_path / "Dataset"), "mha").read_data("CT", "CASE_000")


@pytest.mark.parametrize("file_format", ["mha", "h5", "omezarr", "itktransform"])
def test_a_region_read_records_the_regions_own_geometry(tmp_path: "Path", file_format: str) -> None:
    """Every backend hands a region back with the origin of its first sample, and the spacing a step
    scales, whatever route it serves the region by."""
    import numpy as np
    from konfai.utils.dataset import Attribute, Dataset

    if file_format == "omezarr":
        pytest.importorskip("ngff_zarr")
    attributes = Attribute()
    attributes["Origin"], attributes["Spacing"] = np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])
    attributes["Direction"] = np.eye(3).ravel()
    volume = np.arange(3 * 6 * 5 * 4, dtype=np.float32).reshape(3, 6, 5, 4)  # a field for itktransform
    Dataset(str(tmp_path / "Dataset"), file_format).write("G", "CASE_000", volume, attributes)
    dataset = Dataset(str(tmp_path / "Dataset"), file_format)

    region = (slice(None), slice(2, 4), slice(1, 3), slice(1, 3))
    data, record = dataset.read_data_slice("G", "CASE_000", region)
    np.testing.assert_array_equal(data, volume[region])
    np.testing.assert_allclose(record.get_np_array("Origin"), [11.0, 22.0, 36.0])

    stepped = (slice(None), slice(0, 6, 2), slice(1, 3), slice(1, 3))
    data, record = dataset.read_data_slice("G", "CASE_000", stepped)
    np.testing.assert_array_equal(data, volume[stepped])
    np.testing.assert_allclose(record.get_np_array("Origin"), [11.0, 22.0, 30.0])
    np.testing.assert_allclose(record.get_np_array("Spacing"), [1.0, 2.0, 6.0])
