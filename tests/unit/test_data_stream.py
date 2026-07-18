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

"""Unit tests for ``Dataset.open_data_stream``: incremental region writes must produce entries
indistinguishable from a whole-volume ``write``, remove partial entries on failure, and refuse formats
that cannot serve region writes."""

from pathlib import Path

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset

pytest.importorskip("SimpleITK")


def _image_attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    return attributes


def _volume(channels: int = 2, dtype: type = np.float32) -> np.ndarray:
    return (np.arange(channels * 6 * 5 * 4).reshape(channels, 6, 5, 4) - 30).astype(dtype)


def _write_by_slabs(dataset: Dataset, volume: np.ndarray, attributes: Attribute, slab: int = 2) -> None:
    stream = dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, attributes)
    assert stream is not None
    with stream:
        for start in range(0, volume.shape[1], slab):
            region = slice(start, min(start + slab, volume.shape[1]))
            slices = (slice(0, volume.shape[0]), region, *(slice(0, extent) for extent in volume.shape[2:]))
            stream.write_slice(slices, volume[:, region])


def _skip_unavailable(file_format: str) -> None:
    if file_format == "omezarr":
        pytest.importorskip("zarr")
    if file_format == "h5":
        pytest.importorskip("h5py")


FORMATS = ["mha", "h5", "omezarr"]


@pytest.mark.parametrize("file_format", FORMATS)
@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_stream_matches_whole_volume_write(tmp_path: Path, file_format: str, dtype: type) -> None:
    _skip_unavailable(file_format)
    volume = _volume(dtype=dtype)
    Dataset(tmp_path / "reference", file_format).write("CT", "CASE_001", volume, _image_attributes())
    _write_by_slabs(Dataset(tmp_path / "streamed", file_format), volume, _image_attributes())

    expected, expected_attributes = Dataset(tmp_path / "reference", file_format).read_data("CT", "CASE_001")
    result, result_attributes = Dataset(tmp_path / "streamed", file_format).read_data("CT", "CASE_001")

    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)
    for key in ("Origin", "Spacing", "Direction"):
        np.testing.assert_allclose(result_attributes.get_np_array(key), expected_attributes.get_np_array(key))


def test_mha_stream_single_channel_reads_back_as_scalar_image(tmp_path: Path) -> None:
    volume = _volume(channels=1)
    dataset = Dataset(tmp_path / "streamed", "mha")
    _write_by_slabs(dataset, volume, _image_attributes())

    import SimpleITK as sitk

    image = sitk.ReadImage(str(tmp_path / "streamed" / "CASE_001" / "CT.mha"))
    assert image.GetNumberOfComponentsPerPixel() == 1
    assert image.GetSize() == (4, 5, 6)
    np.testing.assert_allclose(image.GetOrigin(), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), volume[0])


def test_stream_partial_slab_regions_compose_exactly(tmp_path: Path) -> None:
    """Uneven slab sizes and a non-zero channel offset must land where their slices say."""
    volume = _volume(channels=3)
    dataset = Dataset(tmp_path / "streamed", "mha")
    stream = dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, _image_attributes())
    assert stream is not None
    with stream:
        stream.write_slice((slice(0, 3), slice(0, 5), slice(0, 5), slice(0, 4)), volume[:, 0:5])
        stream.write_slice((slice(1, 3), slice(5, 6), slice(0, 5), slice(0, 4)), volume[1:3, 5:6])
        stream.write_slice((slice(0, 1), slice(5, 6), slice(0, 5), slice(0, 4)), volume[0:1, 5:6])

    result, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(result, volume)


@pytest.mark.parametrize("file_format", FORMATS)
def test_stream_removes_partial_entry_on_error(tmp_path: Path, file_format: str) -> None:
    _skip_unavailable(file_format)
    volume = _volume()
    dataset = Dataset(tmp_path / "streamed", file_format)
    stream = dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, _image_attributes())
    assert stream is not None
    with pytest.raises(RuntimeError, match="boom"):
        with stream:
            stream.write_slice(
                (slice(0, volume.shape[0]), slice(0, 2), slice(0, 5), slice(0, 4)),
                volume[:, 0:2],
            )
            raise RuntimeError("boom")

    assert not dataset.is_dataset_exist("CT", "CASE_001")


def test_unstreamable_formats_and_inputs_return_none(tmp_path: Path) -> None:
    geometry = _image_attributes()
    assert Dataset(tmp_path / "a", "nii.gz").open_data_stream("CT", "C", [1, 4, 4, 4], np.float32, geometry) is None
    assert Dataset(tmp_path / "b", "mha").open_data_stream("CT", "C", [1, 4, 4, 4], np.float32, Attribute()) is None
    assert Dataset(tmp_path / "c", "mha").open_data_stream("CT", "C", [1, 4, 4, 4], np.bool_, geometry) is None


def test_mha_float16_is_stored_as_float32_matching_the_whole_volume_path(tmp_path: Path) -> None:
    """MetaImage has no half-float type, so a float16 output streams as float32 -- the exact widening
    the whole-volume writer does too, so streamed and assembled stay byte-identical (not a crash, which
    is what a bare ``open_data_stream`` refusal would cause mid-run)."""
    volume = _volume(channels=2, dtype=np.float16)
    Dataset(tmp_path / "reference", "mha").write("CT", "CASE_001", volume, _image_attributes())
    _write_by_slabs(Dataset(tmp_path / "streamed", "mha"), volume, _image_attributes())

    expected, _ = Dataset(tmp_path / "reference", "mha").read_data("CT", "CASE_001")
    result, _ = Dataset(tmp_path / "streamed", "mha").read_data("CT", "CASE_001")
    assert expected.dtype == np.float32 and result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, volume.astype(np.float32))


@pytest.mark.parametrize("file_format", FORMATS)
def test_entry_is_invisible_until_the_stream_finalizes(tmp_path: Path, file_format: str) -> None:
    """The entry must not exist under its final name while the stream is open: an existence probe
    taken mid-write (another worker resolving the same case) would otherwise stream from a partial
    volume."""
    _skip_unavailable(file_format)
    volume = _volume()
    dataset = Dataset(tmp_path / "streamed", file_format)
    stream = dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, _image_attributes())
    assert stream is not None
    with stream:
        stream.write_slice(
            (slice(0, volume.shape[0]), slice(0, 2), slice(0, 5), slice(0, 4)),
            volume[:, 0:2],
        )
        assert not Dataset(tmp_path / "streamed", file_format).is_dataset_exist("CT", "CASE_001")
        for start in range(2, volume.shape[1], 2):
            region = slice(start, min(start + 2, volume.shape[1]))
            slices = (slice(0, volume.shape[0]), region, *(slice(0, extent) for extent in volume.shape[2:]))
            stream.write_slice(slices, volume[:, region])
    assert dataset.is_dataset_exist("CT", "CASE_001")
    result, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(result, volume)


@pytest.mark.parametrize("file_format", FORMATS)
def test_replaced_entry_stays_readable_until_its_replacement_is_complete(tmp_path: Path, file_format: str) -> None:
    _skip_unavailable(file_format)
    first = _volume()
    second = first + 1
    dataset = Dataset(tmp_path / "streamed", file_format)
    _write_by_slabs(dataset, first, _image_attributes())
    stream = dataset.open_data_stream("CT", "CASE_001", list(second.shape), second.dtype, _image_attributes())
    assert stream is not None
    with stream:
        stream.write_slice(
            (slice(0, second.shape[0]), slice(0, 2), slice(0, 5), slice(0, 4)),
            second[:, 0:2],
        )
        mid_write, _ = dataset.read_data("CT", "CASE_001")
        np.testing.assert_array_equal(mid_write, first)
        for start in range(2, second.shape[1], 2):
            region = slice(start, min(start + 2, second.shape[1]))
            slices = (slice(0, second.shape[0]), region, *(slice(0, extent) for extent in second.shape[2:]))
            stream.write_slice(slices, second[:, region])
    result, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(result, second)


@pytest.mark.parametrize("file_format", ["mha", "omezarr"])
def test_two_concurrent_streams_of_one_entry_publish_a_complete_volume(tmp_path: Path, file_format: str) -> None:
    """Two writers of the same entry (a case landing on two workers) must not share a temporary:
    each owns its own, and whichever finalizes last publishes a COMPLETE volume — never an
    interleaving where one writer's open truncated the other's in-flight file."""
    _skip_unavailable(file_format)
    # DISTINCT volumes: if the two temporaries interleaved into one final file, the result would equal
    # neither whole -- writing the same values both times could not tell an interleaving from a clean
    # publish. ``first`` finalizes last (its ``with`` closes after ``second``'s), so it must win whole.
    volume_a = _volume()
    volume_b = volume_a + 100
    dataset = Dataset(tmp_path / "streamed", file_format)
    shape, dtype = list(volume_a.shape), volume_a.dtype
    first = dataset.open_data_stream("CT", "CASE_001", shape, dtype, _image_attributes())
    assert first is not None
    with first:
        first.write_slice(
            (slice(0, volume_a.shape[0]), slice(0, 3), slice(0, 5), slice(0, 4)),
            volume_a[:, 0:3],
        )
        # A second writer starts while the first is mid-write, with its own values.
        second = dataset.open_data_stream("CT", "CASE_001", shape, dtype, _image_attributes())
        assert second is not None
        with second:
            for start in range(0, volume_b.shape[1], 2):
                region = slice(start, min(start + 2, volume_b.shape[1]))
                slices = (slice(0, volume_b.shape[0]), region, *(slice(0, extent) for extent in volume_b.shape[2:]))
                second.write_slice(slices, volume_b[:, region])
        for start in range(3, volume_a.shape[1], 2):
            region = slice(start, min(start + 2, volume_a.shape[1]))
            slices = (slice(0, volume_a.shape[0]), region, *(slice(0, extent) for extent in volume_a.shape[2:]))
            first.write_slice(slices, volume_a[:, region])

    result, _ = dataset.read_data("CT", "CASE_001")
    # A complete volume, never a per-slab mixture of the two.
    np.testing.assert_array_equal(result, volume_a)


def test_h5_stream_temporary_key_is_invisible_to_name_listing(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    volume = _volume()
    dataset = Dataset(tmp_path / "streamed", "h5")
    dataset.write("CT", "CASE_000", volume, _image_attributes())
    stream = dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, _image_attributes())
    assert stream is not None
    with stream:
        stream.write_slice(
            (slice(0, volume.shape[0]), slice(0, 2), slice(0, 5), slice(0, 4)),
            volume[:, 0:2],
        )
        assert Dataset(tmp_path / "streamed", "h5").get_names("CT") == ["CASE_000"]
        for start in range(2, volume.shape[1], 2):
            region = slice(start, min(start + 2, volume.shape[1]))
            slices = (slice(0, volume.shape[0]), region, *(slice(0, extent) for extent in volume.shape[2:]))
            stream.write_slice(slices, volume[:, region])
    assert sorted(dataset.get_names("CT")) == ["CASE_000", "CASE_001"]


@pytest.mark.parametrize("file_format", FORMATS)
def test_aborted_stream_leaves_an_existing_entry_untouched(tmp_path: Path, file_format: str) -> None:
    _skip_unavailable(file_format)
    first = _volume()
    dataset = Dataset(tmp_path / "streamed", file_format)
    _write_by_slabs(dataset, first, _image_attributes())
    stream = dataset.open_data_stream("CT", "CASE_001", list(first.shape), first.dtype, _image_attributes())
    assert stream is not None
    with pytest.raises(RuntimeError, match="boom"):
        with stream:
            raise RuntimeError("boom")
    result, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(result, first)


@pytest.mark.parametrize("file_format", FORMATS)
def test_abort_after_close_is_a_noop(tmp_path: Path, file_format: str) -> None:
    """The finalize lifecycle is single-shot: streamed Save materialization close()s, then abort()s on
    the error path. The second call must not re-run _close on already-released state (which would try
    to remove the entry it just published, or double-exit the backing file)."""
    _skip_unavailable(file_format)
    volume = _volume()
    dataset = Dataset(tmp_path / "streamed", file_format)
    stream = dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, _image_attributes())
    assert stream is not None
    for start in range(0, volume.shape[1], 2):
        region = slice(start, min(start + 2, volume.shape[1]))
        stream.write_slice((slice(0, volume.shape[0]), region, *(slice(0, e) for e in volume.shape[2:])), volume[:, region])
    stream.close()
    stream.abort(RuntimeError("late"))  # must be inert, not undo the publish
    result, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(result, volume)


def test_can_stream_data_matches_open_support(tmp_path: Path) -> None:
    geometry = _image_attributes()
    assert Dataset(tmp_path / "a", "mha").can_stream_data(geometry)
    assert not Dataset(tmp_path / "a", "mha").can_stream_data(Attribute())
    assert not Dataset(tmp_path / "b", "nii.gz").can_stream_data(geometry)
    assert Dataset(tmp_path / "c", "h5").can_stream_data(Attribute())
    assert Dataset(tmp_path / "d", "omezarr").can_stream_data(Attribute())
