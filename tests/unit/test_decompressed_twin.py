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

"""A compressed ITK file (``.nii.gz``, a zlib MetaImage) is read by region from an uncompressed twin,
decompressed once per entry and run: the values and headers of today's reads, the decode counts,
the memory and disk it holds, and the processes sharing it."""

import gzip
import multiprocessing
import os
import shutil
import socket
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from konfai import api  # noqa: E402
from konfai.data.data_manager import DatasetIter, Group, GroupTransform  # noqa: E402
from konfai.data.patching import DatasetManager, DatasetPatch  # noqa: E402
from konfai.data.transform import Clip, Write  # noqa: E402
from konfai.utils.dataset import Dataset, decompressed  # noqa: E402
from konfai.utils.dataset.stream import DataStream  # noqa: E402
from konfai.utils.errors import CaseReadError, KonfAIWarning  # noqa: E402

_DTYPES = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float32", "float64"]
_ROTATION_3D = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_ROTATION_2D = (0.0, -1.0, 1.0, 0.0)


def _image(shape: tuple[int, ...], dtype: str, channels: int) -> "sitk.Image":
    rank = len(shape)
    values = np.arange(int(np.prod(shape)) * channels) % 97 - (40 if np.dtype(dtype).kind in "if" else 0)
    array = values.astype(dtype).reshape(*shape, channels) if channels > 1 else values.astype(dtype).reshape(shape)
    image = sitk.GetImageFromArray(array, isVector=channels > 1)
    image.SetOrigin((0.5, -1.0, 2.5)[:rank])
    image.SetSpacing((1.05, 0.95, 1.15)[:rank])
    image.SetDirection(_ROTATION_3D if rank == 3 else _ROTATION_2D)
    return image


def _write(path: Path, image: "sitk.Image") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)
    return path


def _twins(root: Path) -> list[Path]:
    """The twins published under the cache root: never a lock, never a staging file."""
    return sorted(path for path in root.rglob("*") if path.suffix in (".nii", ".mha") and not path.name.startswith("."))


def _assert_same_read(twinned: tuple, today: tuple, vector_nifti: bool = False) -> None:
    """Same bytes, dtype and record. A vector NIfTI is the one exception, and the raw-block route of an
    uncompressed one already makes it: ITK aborts on a region of it, so today's route reads it whole,
    and the first rung of the Origin stack is the volume's there, the region's here. The region's
    origin, the key every reader takes, is the same."""
    (data, attributes), (expected, expected_attributes) = twinned, today
    assert data.dtype == expected.dtype
    np.testing.assert_array_equal(data, expected)
    rungs = ("Origin_0", "Origin_1") if vector_nifti else ()
    np.testing.assert_array_equal(attributes.get_np_array("Origin"), expected_attributes.get_np_array("Origin"))
    assert {k: v for k, v in attributes.items() if k not in rungs} == {
        k: v for k, v in expected_attributes.items() if k not in rungs
    }


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "decompressed"


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Every whole decode, by route: the twin's one decompression, and ITK decoding a compressed file."""
    calls: dict[str, list[str]] = {"twin": [], "itk": []}
    decompress, execute, read_image = decompressed._decompress, sitk.ImageFileReader.Execute, sitk.ReadImage

    def counting_decompress(found, staging):
        calls["twin"].append(found.data)
        return decompress(found, staging)

    def counting_execute(self, *args, **kwargs):
        if decompressed.layout(self.GetFileName()) is not None:
            calls["itk"].append(self.GetFileName())
        return execute(self, *args, **kwargs)

    def counting_read_image(path, *args, **kwargs):
        if decompressed.layout(str(path)) is not None:
            calls["itk"].append(str(path))
        return read_image(path, *args, **kwargs)

    monkeypatch.setattr(decompressed, "_decompress", counting_decompress)
    monkeypatch.setattr(sitk.ImageFileReader, "Execute", counting_execute)
    monkeypatch.setattr(sitk, "ReadImage", counting_read_image)
    return calls


# ---------------------------------------------------------------- the reads are today's reads


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("channels", [1, 3], ids=["scalar", "vector"])
@pytest.mark.parametrize("shape", [(6, 7, 5), (7, 5)], ids=["3-D", "2-D"])
@pytest.mark.parametrize("filename", ["CT.nii.gz", "CT.mha", "CT.mhd"], ids=["nii.gz", "mha", "mhd+zraw"])
def test_a_region_of_the_twin_is_todays_region_of_the_compressed_file(
    tmp_path: Path, cache: Path, filename: str, shape: tuple[int, ...], channels: int, dtype: str
) -> None:
    """Values, dtype and every header key of a region, unit-step and stepped, against the read ITK
    makes of the compressed file itself; and the whole volume against SimpleITK's own read."""
    source = _write(tmp_path / "Dataset" / "P0" / filename, _image(shape, dtype, channels))
    if filename.endswith(".mhd"):
        assert (source.parent / "CT.zraw").exists()
    file = Dataset.SitkFile(f"{source.parent}/", True, "mha")
    regions = [
        (slice(None), *(slice(1, extent - 1) for extent in shape)),
        (slice(None), *(slice(0, extent, 2) for extent in shape)),
    ]
    for region in regions:
        _assert_same_read(
            file.file_to_data_slice("", "CT", region),
            file._image_region("CT", str(source), region),
            vector_nifti=channels > 1 and filename.endswith(".nii.gz"),
        )
    assert len(_twins(cache)) == 1, "the regions were read from the twin"

    whole, _ = file.file_to_data_slice("", "CT", (slice(None),) * (len(shape) + 1))
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(source)))
    np.testing.assert_array_equal(whole, np.moveaxis(array, -1, 0) if channels > 1 else array[None])


# ---------------------------------------------------------------- one decode per entry per run


def _cohort(root: Path, filename: str = "CT.nii.gz", cases: int = 2) -> dict[str, np.ndarray]:
    volumes = {}
    for index in range(cases):
        image = _image((12, 10, 8), "float32", 1)
        volumes[f"P{index}"] = sitk.GetArrayFromImage(image)[None] + index
        _write(root / f"P{index}" / filename, sitk.Cast(image, sitk.sitkFloat32) + float(index))
    return volumes


def _loader(
    root: Path, volumes: dict[str, np.ndarray], single_pass: bool, patch: tuple[int, ...] = (4, 5, 4)
) -> DatasetIter:
    dataset = Dataset(root, "nii.gz")
    groups = {"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})}
    managers = [
        DatasetManager(
            index=index,
            group_src="CT",
            group_dest="CT",
            name=name,
            dataset=dataset,
            patch=DatasetPatch(list(patch)),
            transforms=[],
            data_augmentations_list=[],
        )
        for index, name in enumerate(volumes)
    ]
    mapping = [(case, 0, patch) for case, manager in enumerate(managers) for patch in range(manager.get_size(0))]
    return DatasetIter(
        rank=0,
        data={"CT": managers},
        mapping=mapping,
        groups_src=groups,
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=list(patch),
        overlap=None,
        buffer_size=2,
        use_cache=False,
        single_pass=single_pass,
    )


def test_a_read_of_the_whole_volume_decodes_it_and_makes_no_twin(
    tmp_path: Path, cache: Path, counted: dict[str, list[str]]
) -> None:
    """One patch per case: the read decodes the volume once either way, and a twin would only add its
    write and its read. No twin, one decode per case, and no warning about per-patch decodes."""
    volumes = _cohort(tmp_path / "Dataset")
    loader = _loader(tmp_path / "Dataset", volumes, single_pass=True, patch=(12, 10, 8))
    assert len(loader) == len(volumes)
    with decompressed.run_scope(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for index in range(len(loader)):
            np.testing.assert_array_equal(loader[index]["CT"].tensor.numpy(), volumes[f"P{index}"])
    assert counted["twin"] == []
    assert len(counted["itk"]) == len(volumes)
    assert not [w for w in caught if issubclass(w.category, KonfAIWarning)]


def test_a_patch_read_off_the_twin_is_the_volume(tmp_path: Path, cache: Path) -> None:
    volumes = _cohort(tmp_path / "Dataset", cases=1)
    dataset = Dataset(tmp_path / "Dataset", "nii.gz")
    with decompressed.run_scope():
        region = (slice(None), slice(2, 9), slice(1, 10), slice(3, 7))
        data, _ = dataset.read_data_slice("CT", "P0", region)
    np.testing.assert_array_equal(data, volumes["P0"][region])


@pytest.mark.parametrize("slabs", [True, False], ids=["slabs", "one region per case"])
def test_transform_plans_a_compressed_case_streamed_and_decodes_it_once(
    tmp_path: Path, cache: Path, counted: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch, slabs: bool
) -> None:
    """The plan asks the store before any twin exists and answers STREAM, not a whole-case LOAD.
    Cut into slabs, each case is decompressed once, its slabs swept off the twin, the twin released with
    the case, and nothing is left in the cache. A case that fits the budget is one region, the whole
    volume: one decode of the compressed file, as a LOAD would be, and no twin."""
    monkeypatch.chdir(tmp_path)
    volumes = _cohort(tmp_path / "Raw")
    chains = {"CT": {"CT": [Clip(min_value=0.0, max_value=30.0), Write(dataset="./Out:mha")]}}
    regions: list[str] = []
    read_data_slice = Dataset.read_data_slice
    monkeypatch.setattr(
        Dataset, "read_data_slice", lambda self, g, n, s: regions.append(n) or read_data_slice(self, g, n, s)
    )
    # A budget that cuts each case into slabs: a sweep of the compressed file itself would decode it per slab.
    budget = f"{3 * 12 * 10 * 8 * 4}b" if slabs else "256MiB"
    plan = api.plan_transform(
        "PLAN", "./Raw:nii.gz", chains, transforms_dir=tmp_path / "Transforms", memory_budget=budget, quiet=True
    )
    assert [entry.verdict for entry in plan.entries] == ["STREAM"] * len(volumes)
    assert counted["twin"] == [], "planning decompresses nothing"

    api.transform(
        "RUN", "./Raw:nii.gz", chains, transforms_dir=tmp_path / "Transforms", memory_budget=budget, quiet=True
    )
    if slabs:
        assert len(counted["twin"]) == len(volumes)
        assert counted["itk"] == []
        assert len(regions) > 2 * len(volumes), regions
    else:
        assert counted["twin"] == []
        assert len(counted["itk"]) == len(volumes)
        assert len(regions) == len(volumes), regions
    assert not _twins(cache) and not [path for path in cache.glob("*") if path.is_dir()]
    for name, volume in volumes.items():
        written = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "Out" / name / "CT.mha")))
        np.testing.assert_array_equal(written[None], np.clip(volume, 0.0, 30.0))


# ---------------------------------------------------------------- memory and disk


def test_a_run_removes_its_twins_when_it_raises(tmp_path: Path, cache: Path) -> None:
    _cohort(tmp_path / "Dataset", cases=1)
    with pytest.raises(RuntimeError), decompressed.run_scope():
        assert decompressed.twin(str(tmp_path / "Dataset" / "P0" / "CT.nii.gz"), "P0") is not None
        assert _twins(cache)
        raise RuntimeError("the run fails")
    assert not _twins(cache) and not list(cache.iterdir())


def test_a_killed_runs_directory_is_removed_and_a_live_ones_kept(tmp_path: Path, cache: Path) -> None:
    """Only this host's: on a cache several hosts share, another host's pid says nothing here."""
    child = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(0,))
    child.start()
    child.join()
    host = socket.gethostname()
    dead, live = cache / f"{host}-{child.pid}", cache / f"{host}-{os.getppid()}"
    elsewhere = cache / f"another-{host}-{child.pid}"
    for run in (dead, live, elsewhere):
        run.mkdir(parents=True)
        (run / "0123456789abcdef.nii").write_bytes(b"a twin")
        (run / ".0123456789abcdef.99-0.tmp.nii").write_bytes(b"half a twin")
    with decompressed.run_scope():
        assert not dead.exists()
        assert live.exists() and elsewhere.exists()


def test_a_staging_file_is_never_served(tmp_path: Path, cache: Path) -> None:
    """A writer killed mid-decompression leaves its staging name, never the twin's: the next reader
    decompresses again."""
    _cohort(tmp_path / "Dataset", cases=1)
    source = str(tmp_path / "Dataset" / "P0" / "CT.nii.gz")
    with decompressed.run_scope():
        target, _ = decompressed._target(source, "P0")
        target.parent.mkdir(parents=True)
        Path(DataStream.staging_path(str(target))).write_bytes(b"half a twin")
        assert decompressed.twin(source, "P0") == str(target)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(sitk.ReadImage(str(target))), sitk.GetArrayFromImage(sitk.ReadImage(source))
        )


def test_a_file_changed_since_is_decompressed_again(tmp_path: Path, cache: Path, counted: dict[str, list[str]]) -> None:
    volumes = _cohort(tmp_path / "Dataset", cases=1)
    source = tmp_path / "Dataset" / "P0" / "CT.nii.gz"
    dataset = Dataset(tmp_path / "Dataset", "nii.gz")
    region = (slice(None), slice(0, 4), slice(0, 4), slice(0, 4))
    with decompressed.run_scope():
        first = decompressed.twin(str(source), "P0")
        np.testing.assert_array_equal(dataset.read_data_slice("CT", "P0", region)[0], volumes["P0"][region])
        _write(source, sitk.GetImageFromArray(volumes["P0"][0] * 2))
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))  # a coarse clock may not tick
        second = decompressed.twin(str(source), "P0")
        assert second != first
        np.testing.assert_array_equal(dataset.read_data_slice("CT", "P0", region)[0], volumes["P0"][region] * 2)
    with decompressed.run_scope():  # the next run decompresses again: nothing outlives a run
        decompressed.twin(str(source), "P0")
    assert len(counted["twin"]) == 3


def test_a_cache_that_cannot_be_written_leaves_the_read_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And says which variable moves the cache, once. The store answers as it did before twins, so
    TRANSFORM loads a case that fits rather than streaming regions that each decode the whole stream."""
    monkeypatch.setattr(decompressed, "_refusal_warned", False)
    monkeypatch.chdir(tmp_path)
    volumes = _cohort(tmp_path / "Dataset", cases=1)
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("KONFAI_DECOMPRESSED_DIRECTORY", str(blocker / "decompressed"))
    region = (slice(None), slice(2, 9), slice(1, 10), slice(3, 7))
    dataset = Dataset(tmp_path / "Dataset", "nii.gz")
    assert not dataset.bounded_region_reads("CT", "P0")
    assert dataset.read_granularity("CT", "P0") is None
    chains = {"CT": {"CT": [Clip(min_value=0.0, max_value=30.0), Write(dataset="./Out:mha")]}}
    plan = api.plan_transform("PLAN", "./Dataset:nii.gz", chains, transforms_dir=tmp_path / "T", memory_budget="256MiB")
    assert [entry.verdict for entry in plan.entries] == ["LOAD"]
    with decompressed.run_scope(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert decompressed.twin(str(tmp_path / "Dataset" / "P0" / "CT.nii.gz"), "P0") is None
        data, _ = dataset.read_data_slice("CT", "P0", region)
    np.testing.assert_array_equal(data, volumes["P0"][region])
    messages = [str(w.message) for w in caught if "KONFAI_DECOMPRESSED_DIRECTORY" in str(w.message)]
    assert len(messages) == 1 and "cannot be written" in messages[0]


def test_a_disk_without_room_for_the_twin_leaves_the_read_as_it_was(
    tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan counts on a bounded read only where the twin will be made: without the room for it, the store
    answers as it did before twins."""
    _cohort(tmp_path / "Dataset", cases=1)
    monkeypatch.setattr(decompressed, "_fits", lambda run, needed: False)
    dataset = Dataset(tmp_path / "Dataset", "nii.gz")
    assert not dataset.bounded_region_reads("CT", "P0")
    assert dataset.read_granularity("CT", "P0") is None


def test_a_truncated_stream_publishes_nothing_and_itk_answers(tmp_path: Path, cache: Path) -> None:
    source = _write(tmp_path / "Dataset" / "P0" / "CT.mha", _image((12, 10, 8), "float32", 1))
    source.write_bytes(source.read_bytes()[:-64])
    with decompressed.run_scope():
        assert decompressed.twin(str(source), "P0") is None
        run = Path(os.environ["KONFAI_DECOMPRESSED_RUN"])
        assert [path.suffix for path in run.iterdir()] == [".lock"]
        with pytest.raises(CaseReadError):
            Dataset(tmp_path / "Dataset", "mha").read_data_slice(
                "CT", "P0", (slice(None), slice(0, 2)) + (slice(None),) * 2
            )


# ---------------------------------------------------------------- processes sharing one entry


def _race(source: str, barrier, results) -> None:
    barrier.wait()
    results.put(decompressed.twin(source, "P0"))


@pytest.mark.skipif(sys.platform != "linux", reason="forks the test process; the lock is POSIX")
def test_two_processes_asking_at_once_decompress_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ranks and loader workers of one run share its directory: the first holds the lock, the other
    waits and finds the twin published, never half written."""
    _cohort(tmp_path / "Dataset", cases=1)
    source = str(tmp_path / "Dataset" / "P0" / "CT.nii.gz")
    decompressed.layout(source)  # ITK's header read before the fork, not in the children
    log = tmp_path / "decompressions.log"
    decompress = decompressed._decompress

    def slow_decompress(found, staging):
        with open(log, "a") as file:
            file.write(f"{os.getpid()}\n")
        time.sleep(0.5)  # the other process arrives while this one holds the lock
        return decompress(found, staging)

    monkeypatch.setattr(decompressed, "_decompress", slow_decompress)
    context = multiprocessing.get_context("fork")
    barrier, results = context.Barrier(2), context.Queue()
    with decompressed.run_scope():
        processes = [context.Process(target=_race, args=(source, barrier, results)) for _ in range(2)]
        for process in processes:
            process.start()
        twins = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0
        assert twins[0] == twins[1] is not None
        assert len(log.read_text().split()) == 1
        with gzip.open(source) as stream:
            assert Path(twins[0]).read_bytes() == stream.read()


def test_release_removes_every_twin_read_for_the_case_from_any_root(tmp_path: Path, cache: Path) -> None:
    """A one-pass reader leaving a case takes its mask's twin with it, whichever root and group path
    (here a sub-directory) holds it, and leaves the other cases' twins."""
    _cohort(tmp_path / "Dataset", cases=2)
    for name in ("P0", "P1"):
        (tmp_path / "Masks" / "A" / name).mkdir(parents=True)
        shutil.copy(tmp_path / "Dataset" / name / "CT.nii.gz", tmp_path / "Masks" / "A" / name / "MASK.nii.gz")
    reads = [(Dataset(tmp_path / "Dataset", "nii.gz"), "CT"), (Dataset(tmp_path / "Masks", "nii.gz"), "A/MASK")]
    with decompressed.run_scope():
        for name in ("P0", "P1"):
            for dataset, group in reads:
                dataset.read_data_slice(group, name, (slice(None), slice(0, 2), slice(0, 2), slice(0, 2)))
        assert len(_twins(cache)) == 4
        decompressed.release("P0")
        left = _twins(cache)
        assert [path.name for path in left] == sorted(
            Path(decompressed._target(str(root / "P1" / f"{group}.nii.gz"), "P1")[0]).name
            for root, group in ((tmp_path / "Dataset", "CT"), (tmp_path / "Masks" / "A", "MASK"))
        )
