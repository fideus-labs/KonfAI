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

"""``Dataset.read_data_statistics`` is one fold over ``iter_data_blocks``, whatever the backend.

Each backend used to own its walk (slabs, slices, or the whole volume); those walks are recopied
here as oracles, and the fold is held against them key by key, next to numpy in float64 on the
whole volume. Welford in floating point is not associative, and the fold merges in cache-sized
pieces of its own, so the bound is a few ulp, whatever the backend walked before."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from konfai.utils import dataset as dataset_module
from konfai.utils.budget import set_per_rank_budget
from konfai.utils.dataset import (
    Dataset,
    _finalize_running_statistics,
    _statistics_chunk_length,
    _update_pieces,
    _update_running_statistics,
)
from konfai.utils.errors import DatasetManagerError

sitk = pytest.importorskip("SimpleITK")

_KEYS = ("min", "max", "mean", "std", "min_per_channel", "max_per_channel", "mean_per_channel", "std_per_channel")


def _volume(shape: tuple[int, ...], seed: int = 0, dtype: type = np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape, dtype=np.float32) * 300.0 - 100.0).astype(dtype)


def _numpy_reference(volume: np.ndarray) -> dict[str, Any]:
    values = np.asarray(volume, dtype=np.float64)
    per_channel = values.reshape(values.shape[0], -1)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min_per_channel": per_channel.min(axis=1).tolist(),
        "max_per_channel": per_channel.max(axis=1).tolist(),
        "mean_per_channel": per_channel.mean(axis=1).tolist(),
        "std_per_channel": per_channel.std(axis=1, ddof=1).tolist(),
    }


def _former_walk(chunks: Iterator[np.ndarray], channels: list[int] | None = None) -> dict[str, Any]:
    """What every backend's ``file_to_data_statistics`` did around its own chunks."""
    state = None
    for chunk in chunks:
        state = _update_running_statistics(state, chunk if channels is None else chunk[channels])
    return _finalize_running_statistics(state)


def _assert_within_ulps(new: dict[str, Any], old: dict[str, Any], ulps: int = 64) -> None:
    for key in _KEYS:
        a, b = np.asarray(new[key], dtype=np.float64), np.asarray(old[key], dtype=np.float64)
        assert np.all(np.abs(a - b) <= ulps * np.spacing(np.maximum(np.abs(a), np.abs(b)))), (key, a, b)


def _assert_close_to_numpy(new: dict[str, Any], volume: np.ndarray) -> None:
    reference = _numpy_reference(volume)
    for key in _KEYS:
        np.testing.assert_allclose(new[key], reference[key], rtol=1e-10, atol=0, err_msg=key)


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks of about a thousand elements, folded in pieces of about a hundred: several of each per
    volume, so both grains of the fold are exercised."""
    monkeypatch.setattr(dataset_module, "_STATISTICS_CHUNK_ELEMENTS", 1000)
    monkeypatch.setattr(dataset_module, "_STATISTICS_UPDATE_ELEMENTS", 100)


def _former_h5_walk(path: Path, group: str, name: str) -> Iterator[np.ndarray]:
    """Chunks along axis 1 (axis 0 for a vector), the dataset held open across them."""
    with dataset_module._open_h5(str(path), "r") as file:  # the pool's handle is unlocked: agree with it
        dataset = file[group][name]
        axis = 1 if dataset.ndim > 1 else 0
        length = _statistics_chunk_length(dataset.shape, axis, dataset_module._STATISTICS_CHUNK_ELEMENTS)
        for start in range(0, dataset.shape[axis], length):
            slices = [slice(None)] * dataset.ndim
            slices[axis] = slice(start, min(dataset.shape[axis], start + length))
            yield np.asarray(dataset[tuple(slices)])


@pytest.mark.parametrize("channels", [None, [1]])
def test_h5_folds_within_ulps_of_its_former_walk(
    tmp_path: Path, image_attributes, small_blocks: None, channels: list[int] | None
) -> None:
    pytest.importorskip("h5py")
    volume = _volume((2, 37, 12, 10))
    dataset = Dataset(tmp_path / "store", "h5")
    dataset.write("CT", "P0", volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

    got = dataset.read_data_statistics("CT", "P0", channels)

    _assert_within_ulps(got, _former_walk(_former_h5_walk(tmp_path / "store.h5", "CT", "P0"), channels))
    _assert_close_to_numpy(got, volume if channels is None else volume[channels])


def test_a_vector_takes_the_whole_read_and_folds_within_ulps_of_its_former_pass(
    tmp_path: Path, small_blocks: None
) -> None:
    """A 1-D entry (a parameter list) has no spatial axis to walk: one block, the former one chunk."""
    pytest.importorskip("h5py")
    vector = _volume((50,), dtype=np.float64)
    dataset = Dataset(tmp_path / "store", "h5")
    dataset.write("P", "P0", vector)

    got = dataset.read_data_statistics("P", "P0")

    _assert_within_ulps(got, _former_walk(_former_h5_walk(tmp_path / "store.h5", "P", "P0")))
    _assert_close_to_numpy(got, vector.reshape(1, -1))


def _former_image_slab_walk(directory: Path, name: str, group: str, extension: str) -> Iterator[np.ndarray]:
    """``SitkFile._file_to_image_statistics``: slabs along axis 1 through ``_file_to_image_slice``."""
    path = str(directory / name / f"{group}.{extension}")
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    shape = [reader.GetNumberOfComponents(), *reversed(reader.GetSize())]
    length = _statistics_chunk_length(shape, 1, dataset_module._STATISTICS_CHUNK_ELEMENTS)
    file = Dataset.SitkFile(f"{directory / name}/", True, extension)
    for start in range(0, shape[1], length):
        slices = [slice(None)] * len(shape)
        slices[1] = slice(start, min(shape[1], start + length))
        yield file._file_to_image_slice(group, path, tuple(slices))[0]


def _former_image_whole_pass(directory: Path, name: str, group: str, extension: str) -> Iterator[np.ndarray]:
    """The whole image, channel-first, in one chunk: what a store without region reads got."""
    image = sitk.ReadImage(str(directory / name / f"{group}.{extension}"))
    data = sitk.GetArrayViewFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        yield np.expand_dims(data, 0)
    else:
        yield np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))


@pytest.mark.parametrize(
    ("extension", "former"),
    [("mha", _former_image_slab_walk), ("nii.gz", _former_image_whole_pass)],
    ids=["region reads", "whole read"],
)
@pytest.mark.parametrize("channels", [None, [1]])
def test_an_image_folds_within_ulps_of_its_former_pass(
    tmp_path: Path,
    image_attributes,
    small_blocks: None,
    extension: str,
    former: Callable[..., Iterator[np.ndarray]],
    channels: list[int] | None,
) -> None:
    volume = _volume((2, 37, 12, 10))
    dataset = Dataset(tmp_path / "store", extension)
    dataset.write("CT", "P0", volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    assert dataset.bounded_region_reads("CT", "P0") is (extension == "mha")

    got = dataset.read_data_statistics("CT", "P0", channels)

    _assert_within_ulps(got, _former_walk(former(tmp_path / "store", "P0", "CT", extension), channels))
    _assert_close_to_numpy(got, volume if channels is None else volume[channels])


def test_parameter_rows_take_the_whole_read_and_fold_within_ulps_of_their_former_pass(
    tmp_path: Path, small_blocks: None
) -> None:
    """An ``.itk.txt`` entry is 2-D (one row per transform), but not an image: no region reads."""
    transform = sitk.AffineTransform(3)
    transform.SetParameters(tuple(_volume((12,), dtype=np.float64)))
    dataset = Dataset(tmp_path / "store", "mha")
    dataset.write("T", "P0", transform)
    assert not dataset.bounded_region_reads("T", "P0")

    got = dataset.read_data_statistics("T", "P0")

    rows = Dataset.SitkFile(f"{tmp_path / 'store' / 'P0'}/", True, "mha").file_to_data("", "T")[0]
    _assert_within_ulps(got, _former_walk(iter([rows])))
    _assert_close_to_numpy(got, rows)


def test_a_npy_folds_within_ulps_of_its_former_whole_pass(tmp_path: Path, small_blocks: None) -> None:
    """The former pass ran on the whole map in one update; the fold reads it slab by slab off the map."""
    volume = _volume((2, 37, 12, 10), dtype=np.float64)
    dataset = Dataset(tmp_path / "store", "mha")
    dataset.write("CT", "P0", volume)  # no geometry: stored as CT.npy
    assert dataset.bounded_region_reads("CT", "P0")

    got = dataset.read_data_statistics("CT", "P0")

    former = _former_walk(iter([np.load(tmp_path / "store" / "P0" / "CT.npy", mmap_mode="r")]))
    _assert_within_ulps(got, former)
    _assert_close_to_numpy(got, volume)


def _former_zarr_walk(directory: Path, name: str, group: str) -> Iterator[np.ndarray]:
    file = Dataset.OmeZarrFile(f"{directory / name}/", True)
    shape, _ = file.get_infos("", group)
    length = _statistics_chunk_length(shape, 1, dataset_module._STATISTICS_CHUNK_ELEMENTS)
    for start in range(0, shape[1], length):
        slices = [slice(None)] * len(shape)
        slices[1] = slice(start, min(shape[1], start + length))
        yield file.file_to_data_slice("", group, tuple(slices))[0]


@pytest.mark.parametrize("channels", [None, [1]])
def test_an_ome_zarr_folds_within_ulps_of_its_former_walk(
    tmp_path: Path, image_attributes, small_blocks: None, channels: list[int] | None
) -> None:
    pytest.importorskip("ngff_zarr")
    volume = _volume((2, 37, 12, 10))
    dataset = Dataset(tmp_path / "store", "omezarr")
    dataset.write("CT", "P0", volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

    got = dataset.read_data_statistics("CT", "P0", channels)

    _assert_within_ulps(got, _former_walk(_former_zarr_walk(tmp_path / "store", "P0", "CT"), channels))
    _assert_close_to_numpy(got, volume if channels is None else volume[channels])


def _former_dicom_slice_walk(series: Path) -> Iterator[np.ndarray]:
    from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

    info = get_dicom_info(series)
    for index in range(info["shape"][1]):
        yield read_dicom_series_slice(
            series,
            (slice(None), slice(index, index + 1), slice(None), slice(None)),
            series_uid=info["series_uid"],
            info=info,
        )[0]


def test_a_dicom_series_folds_within_ulps_of_its_former_slice_walk(
    tmp_path: Path, image_attributes, small_blocks: None
) -> None:
    """The former walk read one slice at a time; the fold reads the slices a block holds at once."""
    pytest.importorskip("pydicom")
    volume = np.round(_volume((1, 37, 12, 10)))
    dataset = Dataset(tmp_path / "store", "dicom")
    dataset.write("CT", "P0", volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    stored = dataset.read_data("CT", "P0")[0]  # the series stores a rescaled integer encoding

    got = dataset.read_data_statistics("CT", "P0")

    _assert_within_ulps(got, _former_walk(_former_dicom_slice_walk(tmp_path / "store" / "P0" / "CT")))
    _assert_close_to_numpy(got, stored)


def test_a_displacement_field_folds_within_ulps_of_its_former_whole_pass(
    tmp_path: Path, image_attributes, small_blocks: None
) -> None:
    """The former pass decoded the field whole (float64); the fold reads it row span by row span."""
    pytest.importorskip("h5py")
    field = _volume((3, 37, 12, 10))
    dataset = Dataset(tmp_path / "store", "itktransform")
    dataset.write("Transform", "P0", field, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    assert dataset.bounded_region_reads("Transform", "P0")

    got = dataset.read_data_statistics("Transform", "P0")

    whole = Dataset.ItkTransformFile(f"{tmp_path / 'store' / 'P0'}/", True).file_to_data("", "Transform")[0]
    _assert_within_ulps(got, _former_walk(iter([whole])))
    _assert_close_to_numpy(got, field)


def test_the_fold_reads_a_bounded_store_by_blocks_and_never_whole(
    tmp_path: Path, image_attributes, small_blocks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset(tmp_path / "store", "mha")
    dataset.write("CT", "P0", _volume((1, 37, 12, 10)), image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    monkeypatch.setattr(Dataset, "read_data", lambda *_: pytest.fail("a bounded store is never read whole"))
    blocks: list[tuple[int, ...]] = []
    real = Dataset.read_data_slice

    def counted(self: Dataset, groups: str, name: str, slices: tuple[slice, ...]) -> Any:
        data, attributes = real(self, groups, name, slices)
        blocks.append(data.shape)
        return data, attributes

    monkeypatch.setattr(Dataset, "read_data_slice", counted)
    dataset.read_data_statistics("CT", "P0")
    # Under a declared budget the blocks are priced off a one-voxel read of the store; the blocks
    # themselves are the rest, and they cover the volume whether or not that probe happened.
    blocks = [shape for shape in blocks if int(np.prod(shape)) > 1]
    assert len(blocks) > 1 and all(shape[0] == 1 and shape[2:] == (12, 10) for shape in blocks)
    assert sum(shape[1] for shape in blocks) == 37


def test_the_fold_reads_an_unbounded_store_whole_once(
    tmp_path: Path, image_attributes, small_blocks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset(tmp_path / "store", "nii.gz")
    dataset.write("CT", "P0", _volume((1, 37, 12, 10)), image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    monkeypatch.setattr(Dataset, "read_data_slice", lambda *_: pytest.fail("no region read on an unbounded store"))
    reads: list[str] = []
    real = Dataset.read_data
    monkeypatch.setattr(Dataset, "read_data", lambda self, g, n: reads.append(n) or real(self, g, n))
    dataset.read_data_statistics("CT", "P0")
    assert reads == ["P0"]


def test_a_block_is_folded_in_pieces_along_its_first_spatial_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_module, "_STATISTICS_UPDATE_ELEMENTS", 500)
    block = _volume((2, 37, 12, 10))
    pieces = list(_update_pieces(block))
    assert [piece.shape for piece in pieces] == [(2, 2, 12, 10)] * 18 + [(2, 1, 12, 10)]  # 500 // 240 rows each
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), block)
    assert all(np.shares_memory(piece, block) for piece in pieces)
    vector = _volume((50,))
    assert [piece.shape for piece in _update_pieces(vector)] == [(50,)]


# ---------------------------------------------------------------- the budget is the read grain


@pytest.fixture
def budgeted_scan(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pieces of about a hundred elements, so a block of the small volumes below snaps to whole rows
    rather than to the one piece the whole volume is; and the published budget put back after."""
    monkeypatch.setattr(dataset_module, "_STATISTICS_UPDATE_ELEMENTS", 100)
    yield
    set_per_rank_budget(None)


def _scan_blocks(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """The shapes a scan of ``CT/P0`` reads, and the statistics it answers with."""
    blocks: list[tuple[int, ...]] = []
    real = Dataset.read_data_slice

    def counted(self: Dataset, groups: str, name: str, slices: tuple[slice, ...]) -> Any:
        data, attributes = real(self, groups, name, slices)
        blocks.append(data.shape)
        return data, attributes

    monkeypatch.setattr(Dataset, "read_data_slice", counted)
    return blocks


def test_a_declared_budget_is_what_the_scan_reads_a_block_of(
    tmp_path: Path, image_attributes, monkeypatch: pytest.MonkeyPatch, budgeted_scan: None
) -> None:
    """A fixed read grain made a whole-volume scan cost the same whatever the budget said: 95.6 MiB
    of resident set over a 78 MiB case at 512, 128 and 32 MiB alike."""
    dataset = Dataset(tmp_path / "store", "mha")
    dataset.write("CT", "P0", _volume((1, 64, 40, 40)), image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    read = {}
    for budget in (None, 3 * 4 * 32 * 40 * 40, 3 * 4 * 8 * 40 * 40):
        set_per_rank_budget(budget)
        blocks = _scan_blocks(dataset, monkeypatch)
        dataset.read_data_statistics("CT", "P0")
        read[budget] = max(shape[1] for shape in blocks)

    assert read[3 * 4 * 8 * 40 * 40] < read[3 * 4 * 32 * 40 * 40] <= read[None]


def test_the_scan_answers_the_same_whatever_the_budget_read_it_in(
    tmp_path: Path, image_attributes, budgeted_scan: None
) -> None:
    """The budget is how much is held, never what is computed: the fold sees the same pieces in the
    same order whatever the read grain, because a block is a whole number of them."""
    dataset = Dataset(tmp_path / "store", "mha")
    dataset.write("CT", "P0", _volume((1, 64, 40, 40)), image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    set_per_rank_budget(None)
    reference = dataset.read_data_statistics("CT", "P0")

    for budget in (3 * 4 * 32 * 40 * 40, 3 * 4 * 8 * 40 * 40, 3 * 4 * 4 * 40 * 40):
        set_per_rank_budget(budget)
        got = dataset.read_data_statistics("CT", "P0")
        for key in _KEYS:
            np.testing.assert_array_equal(got[key], reference[key], err_msg=f"{key} at {budget} B")


def test_a_budget_the_shortest_block_of_a_scan_exceeds_is_refused(
    tmp_path: Path, image_attributes, budgeted_scan: None
) -> None:
    """The floor of the read grain is the fold's own piece, so a budget under it refuses naming both
    figures rather than reading a block it cannot hold."""
    dataset = Dataset(tmp_path / "store", "mha")
    dataset.write("CT", "P0", _volume((1, 64, 40, 40)), image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    set_per_rank_budget(4096)

    with pytest.raises(DatasetManagerError, match=r"over the per-rank memory budget \(4.00 KiB\)"):
        dataset.read_data_statistics("CT", "P0")


def test_a_scan_of_a_float64_source_reads_blocks_the_budget_holds(
    tmp_path: Path, image_attributes, monkeypatch: pytest.MonkeyPatch, budgeted_scan: None
) -> None:
    """A block is the bytes the store hands over, not a float32 copy of them.

    Priced at four bytes an element whatever the source, the three blocks a scan holds in flight
    over a float64 store came to twice the budget the run was given, and nothing said so.
    """
    dataset = Dataset(tmp_path / "store", "mha")
    volume = _volume((1, 64, 40, 40), dtype=np.float64)
    dataset.write("CT", "P0", volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    assert dataset.read_data_slice("CT", "P0", (slice(0, 1),) * 4)[0].dtype == np.float64

    budget = dataset_module._STATISTICS_BLOCKS_IN_FLIGHT * 8 * 8 * 40 * 40  # three eight-row float64 blocks
    set_per_rank_budget(budget)
    blocks = _scan_blocks(dataset, monkeypatch)
    got = dataset.read_data_statistics("CT", "P0")

    held = max(int(np.prod(shape)) for shape in blocks) * 8 * dataset_module._STATISTICS_BLOCKS_IN_FLIGHT
    assert held <= budget, f"{held} B held against a {budget} B budget"
    _assert_close_to_numpy(got, volume)
