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

import os
from pathlib import Path

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset, is_staging_entry

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


FORMATS = ["mha", "nii", "h5", "omezarr"]


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
    """MetaImage has no half-float type, so a float16 output streams as float32: the exact widening
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
    each owns its own, and whichever finalizes last publishes a COMPLETE volume, never an
    interleaving where one writer's open truncated the other's in-flight file."""
    _skip_unavailable(file_format)
    # DISTINCT volumes: if the two temporaries interleaved into one final file, the result would equal
    # neither whole: writing the same values both times could not tell an interleaving from a clean
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


def test_h5_replace_keeps_the_old_entry_until_the_new_one_is_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HDF5 has no rename-over. Deleting the old entry and then moving the new one in leaves a
    window where a crash loses both; the old entry is moved aside instead, and a publish that fails
    puts it back where it was (a crash inside that window leaves it recoverable under its
    .replaced-<pid> key, which no listing shows as a case)."""
    h5py = pytest.importorskip("h5py")
    first = _volume()
    dataset = Dataset(tmp_path / "streamed", "h5")
    dataset.write("CT", "CASE_000", first, _image_attributes())
    stream = dataset.open_data_stream("CT", "CASE_000", list(first.shape), first.dtype, _image_attributes())
    assert stream is not None
    real_move = h5py.Group.move
    failed_publishes: list[str] = []

    def fail_the_publish(self, source, dest):
        # The publish (the temporary onto the final name) fails once; the restore that follows
        # (the backup onto the final name) goes through.
        if dest == "CASE_000" and not failed_publishes:
            failed_publishes.append(source)
            raise OSError("the publish failed")
        return real_move(self, source, dest)

    monkeypatch.setattr(h5py.Group, "move", fail_the_publish)
    with pytest.raises(OSError, match="publish failed"):
        with stream:
            slices = (slice(0, first.shape[0]), *(slice(0, extent) for extent in first.shape[1:]))
            stream.write_slice(slices, first + 1)
    monkeypatch.undo()
    assert failed_publishes and ".tmp" in failed_publishes[0]
    with h5py.File(tmp_path / "streamed.h5", "r", locking=False) as handle:
        keys = list(handle["CT"].keys())
    assert not any(".replaced-" in key for key in keys), keys
    np.testing.assert_array_equal(dataset.read_data("CT", "CASE_000")[0], first)  # the old entry, back

    # The nominal replace: new data in, no .replaced- key left behind (the crashed stream's own
    # .tmp is that crash's debris, invisible to listings, and not this replace's to clean).
    dataset.write("CT", "CASE_000", first, _image_attributes())
    stream = dataset.open_data_stream("CT", "CASE_000", list(first.shape), first.dtype, _image_attributes())
    assert stream is not None
    with stream:
        stream.write_slice(slices, first + 1)
    np.testing.assert_array_equal(dataset.read_data("CT", "CASE_000")[0], first + 1)
    with h5py.File(tmp_path / "streamed.h5", "r", locking=False) as handle:
        assert [key for key in handle["CT"].keys() if ".replaced-" in key] == []


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
        stream.write_slice(
            (slice(0, volume.shape[0]), region, *(slice(0, e) for e in volume.shape[2:])), volume[:, region]
        )
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


def test_nii_stream_is_the_file_sitk_would_have_written(tmp_path: Path) -> None:
    """The one convention the NIfTI stream owns is the RAS sform; sitk's own writer is the oracle.

    Compared through sitk's reader on an OBLIQUE grid: a dropped or half-applied LPS-to-RAS flip
    reads back as a different Origin/Direction, not as an error.
    """
    import SimpleITK as sitk

    volume = _volume(channels=1)
    attributes = _image_attributes()
    angle = np.deg2rad(30.0)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    attributes["Direction"] = np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]).reshape(-1)

    dataset = Dataset(tmp_path / "streamed", "nii")
    _write_by_slabs(dataset, volume, attributes)

    reference = sitk.GetImageFromArray(volume[0])
    reference.SetOrigin(attributes.get_np_array("Origin").tolist())
    reference.SetSpacing(attributes.get_np_array("Spacing").tolist())
    reference.SetDirection(attributes.get_np_array("Direction").tolist())
    sitk.WriteImage(reference, str(tmp_path / "reference.nii"))

    got = sitk.ReadImage(str(tmp_path / "streamed" / "CASE_001" / "CT.nii"))
    want = sitk.ReadImage(str(tmp_path / "reference.nii"))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(got), sitk.GetArrayFromImage(want))
    np.testing.assert_allclose(got.GetOrigin(), want.GetOrigin(), atol=1e-5)
    np.testing.assert_allclose(got.GetSpacing(), want.GetSpacing(), atol=1e-6)
    np.testing.assert_allclose(got.GetDirection(), want.GetDirection(), atol=1e-6)


def test_nii_stream_multi_channel_reads_back_as_vector_image(tmp_path: Path) -> None:
    """The vector dimension is NIfTI's slowest, so channel-first slabs land without a transpose."""
    import SimpleITK as sitk

    volume = _volume(channels=3)
    dataset = Dataset(tmp_path / "streamed", "nii")
    _write_by_slabs(dataset, volume, _image_attributes())

    image = sitk.ReadImage(str(tmp_path / "streamed" / "CASE_001" / "CT.nii"))
    assert image.GetNumberOfComponentsPerPixel() == 3
    back, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(np.asarray(back), volume)


@pytest.mark.parametrize("file_format", ["mha", "nii", "nii.gz", "h5", "omezarr", "itktransform"])
def test_a_crashed_writer_leaves_debris_and_no_case(tmp_path: Path, file_format: str, monkeypatch) -> None:
    """Every backend's staging entry (a stream's temporary, or the hidden file a whole-volume write
    publishes from) is recognised as staging and invisible to the listing and to the membership probe:
    what a hard kill leaves behind is debris a rerun rewrites, never a case it skips."""
    _skip_unavailable(file_format)
    volume = _volume(channels=3)
    root = tmp_path / "streamed"
    dataset = Dataset(root, file_format)

    def crash(*args, **kwargs):
        raise OSError("killed before publish")

    # Fail the publish step: the staging entry is then on disk, exactly as a hard kill leaves it.
    if file_format == "h5":
        monkeypatch.setattr(pytest.importorskip("h5py").Group, "move", crash)
    elif file_format == "omezarr":
        monkeypatch.setattr(os, "rename", crash)
    else:
        monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="killed"):
        if file_format == "nii.gz":  # no stream: the whole-volume write stages and publishes by rename
            dataset.write("CT", "CASE_001", volume, _image_attributes())
        else:
            _write_by_slabs(dataset, volume, _image_attributes())
    monkeypatch.undo()

    if file_format == "h5":
        with pytest.importorskip("h5py").File(f"{root}.h5", "r", locking=False) as handle:
            debris = list(handle["CT"].keys())
    else:
        debris = [path.name for path in (root / "CASE_001").iterdir()]
    assert debris and all(is_staging_entry(name) for name in debris), debris
    fresh = Dataset(root, file_format)
    assert fresh.get_names("CT") == []
    assert not fresh.is_dataset_exist("CT", "CASE_001")
    if file_format != "h5":  # an h5 group exists as soon as it holds a temporary; a directory lists its files
        assert fresh.get_group() == []


@pytest.mark.parametrize("how", ["stream", "write"])
def test_h5_entry_whose_attributes_fail_is_not_left_in_the_file(tmp_path: Path, how: str, monkeypatch) -> None:
    """The dataset and its attributes are one entry: an interrupt between the two leaves neither an
    attribute-less entry under the final name nor an orphaned temporary (HDF5 never reclaims one)."""
    h5py = pytest.importorskip("h5py")
    volume = _volume()
    dataset = Dataset(tmp_path / "streamed", "h5")
    dataset.write("CT", "CASE_000", volume, _image_attributes())

    def crash(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(h5py.AttributeManager, "update", crash)
    with pytest.raises(KeyboardInterrupt):
        if how == "stream":
            dataset.open_data_stream("CT", "CASE_001", list(volume.shape), volume.dtype, _image_attributes())
        else:
            dataset.write("CT", "CASE_001", volume, _image_attributes())
    monkeypatch.undo()
    with h5py.File(tmp_path / "streamed.h5", "r", locking=False) as handle:
        assert list(handle["CT"].keys()) == ["CASE_000"]


def test_publishing_an_entry_retires_dead_writers_debris_and_keeps_live_ones(tmp_path):
    """Every writer stages under a pid-marked name and publishes by rename, so a hard kill leaves a
    staging file or store the readers skip -- and, until now, nothing ever removed. Publishing the
    entry sweeps the debris of writers that no longer run; a live writer's staging stays."""
    import subprocess
    import sys

    from konfai.utils.dataset import _retire_dead_debris

    final = tmp_path / "CT.ome.zarr"
    final.mkdir()
    import psutil

    dead = 1
    while True:  # a pid nobody holds (psutil: portable where os.kill(pid, 0) is not)
        dead += 7919
        if not psutil.pid_exists(dead):
            break
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / f"CT.ome.zarr.{dead}-3.tmp").mkdir()
        (tmp_path / f"CT.ome.zarr.{dead}.replaced").mkdir()
        (tmp_path / f"CT.ome.zarr.replaced-{dead}").mkdir()
        (tmp_path / f".CT.{dead}.tmp.mha").write_bytes(b"x")
        (tmp_path / f"CT.ome.zarr.{live.pid}-1.tmp").mkdir()
        (tmp_path / "CT_other.ome.zarr").mkdir()  # another entry: not this one's debris
        (tmp_path / f"CT_other.ome.zarr.{dead}-1.tmp").mkdir()
        _retire_dead_debris(final)
        left = sorted(p.name for p in tmp_path.iterdir())
        assert left == sorted(
            ["CT.ome.zarr", f"CT.ome.zarr.{live.pid}-1.tmp", "CT_other.ome.zarr", f"CT_other.ome.zarr.{dead}-1.tmp"]
        )
    finally:
        live.kill()
        live.wait()


@pytest.mark.parametrize("channels", [1, 3])
def test_a_two_dimensional_nifti_streams_like_the_whole_write(tmp_path, channels):
    """A 2-D image is a NIfTI of two dims: the streamed header says so (its third axis a 1, the
    2x2 cosines embedded in the sform) and reads back as the whole write does, voxels and geometry."""
    rng = np.random.default_rng(0)
    volume = rng.random((channels, 7, 9)).astype(np.float32)
    theta = 0.3
    attributes = Attribute(
        {
            "Origin": np.array([-5.0, 2.0]),
            "Spacing": np.array([0.7, 0.9]),
            "Direction": np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).ravel(),
        }
    )
    whole = Dataset(tmp_path / "whole", "nii")
    whole.write("G", "c", volume, attributes)
    streamed = Dataset(tmp_path / "streamed", "nii")
    stream = streamed.open_data_stream("G", "c", list(volume.shape), volume.dtype, attributes)
    assert stream is not None
    with stream:
        for row in range(volume.shape[1]):
            stream.write_slice((slice(None), slice(row, row + 1)), volume[:, row : row + 1])
    expected, header_expected = whole.read_data("G", "c")
    got, header_got = streamed.read_data("G", "c")
    np.testing.assert_array_equal(got, expected)
    for key in ("Origin", "Spacing", "Direction"):
        np.testing.assert_allclose(header_got.get_np_array(key), header_expected.get_np_array(key), atol=1e-6)


def test_replacing_an_h5_entry_keeps_the_old_one_until_the_new_is_in_place(tmp_path, monkeypatch):
    """A rewrite moves the old entry aside, publishes, then drops it: at no instant is the entry
    absent from the file, which is what a crash between the two steps used to leave."""
    h5py = pytest.importorskip("h5py")

    dataset = Dataset(tmp_path / "store", "h5")
    dataset.write("G", "c", np.ones((1, 4, 4, 4), dtype=np.float32), Attribute())
    stream = dataset.open_data_stream("G", "c", [1, 4, 4, 4], np.dtype(np.float32), Attribute())
    assert stream is not None
    original_move = h5py.Group.move
    seen = []

    def spy(self, source, dest):
        original_move(self, source, dest)
        seen.append(("c" in self, dest))

    monkeypatch.setattr(h5py.Group, "move", spy)
    with stream:
        stream.write_slice((slice(None),) * 4, np.zeros((1, 4, 4, 4), dtype=np.float32))
    monkeypatch.undo()
    # After the first move (old aside) the name is free; after the second the new one holds it.
    assert [present for present, _ in seen] == [False, True]
    written, _ = dataset.read_data("G", "c")
    assert float(np.asarray(written).max()) == 0.0
