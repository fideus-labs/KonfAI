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

"""``CaseMaterializer.materialize`` writes a case's chain without ever going through ``get_data``.

Two branches, and the caller is told which one ran: the sweep when the chain streams, the classic
load when it cannot. Both must leave the same bytes, and the streamed one must leave the volume on
disk, never in memory, which is what a whole-volume patch through ``get_data`` would have done."""

from pathlib import Path

import numpy as np
import pytest
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import DatasetManager
from konfai.data.transform import Clip, Permute, Save, Standardize, Transform
from konfai.utils.dataset import Attribute, Dataset

pytest.importorskip("SimpleITK")


def _image_attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attributes


def _source(tmp_path: Path) -> Dataset:
    rng = np.random.default_rng(0)
    volume = (rng.random((1, 12, 10, 8)) * 100).astype(np.float32)
    dataset = Dataset(tmp_path / "source", "mha")
    dataset.write("CT", "CASE_000", volume, _image_attributes())
    return dataset


def _manager(source: Dataset, transforms: list[Transform]) -> DatasetManager:
    # No patch declared: this is exactly the TRANSFORM case, where get_data would cut one
    # whole-volume patch and read back what materialize just wrote.
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=None,
        transforms=transforms,
        data_augmentations_list=[],
    )


def test_streamed_branch_writes_without_holding_the_volume(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0), Save(str(tmp_path / "out"))])

    assert CaseMaterializer(manager).materialize() is Verdict.STREAM
    assert Dataset(tmp_path / "out", "mha").is_dataset_exist("CT", "CASE_000")
    # The volume was never assembled: the sweep read regions and wrote regions.
    assert not manager.loaded
    assert not manager.data


def test_streamed_branch_writes_the_same_bytes_as_the_classic_load(tmp_path: Path) -> None:
    source = _source(tmp_path)
    classic = [Permute("2|1|0"), Clip(0.0, 50.0), Save(str(tmp_path / "classic"))]
    swept = [Permute("2|1|0"), Clip(0.0, 50.0), Save(str(tmp_path / "swept"))]

    _manager(source, classic).load(classic, [], load_augmentations=False)
    assert CaseMaterializer(_manager(source, swept)).materialize() is Verdict.STREAM

    expected, expected_attributes = Dataset(tmp_path / "classic", "mha").read_data("CT", "CASE_000")
    result, result_attributes = Dataset(tmp_path / "swept", "mha").read_data("CT", "CASE_000")
    np.testing.assert_array_equal(result, expected)
    for key in ("Origin", "Spacing", "Direction"):
        np.testing.assert_allclose(
            result_attributes.get_np_array(key), expected_attributes.get_np_array(key), err_msg=key
        )


def test_whole_volume_branch_still_writes_and_says_so(tmp_path: Path) -> None:
    """A statistic taken after a value-changing stage cannot stream: the stored volume's statistic
    is not Standardize's input, so the chain falls back.

    This is the case the driver exists for: ``_stream_ready`` alone would have returned False and
    written nothing at all, silently."""
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0), Standardize(inverse=False), Save(str(tmp_path / "out"))])

    assert manager.can_stream_patch(0) is False
    assert CaseMaterializer(manager).materialize() is Verdict.WHOLE_VOLUME
    assert Dataset(tmp_path / "out", "mha").is_dataset_exist("CT", "CASE_000")
    # The fallback assembled the volume, but released it before returning.
    assert not manager.data


def test_unstreamable_destination_falls_back_and_writes(tmp_path: Path) -> None:
    # nii.gz cannot serve region writes: the sweep refuses, the classic load writes it anyway.
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0), Save(f"{tmp_path / 'out'}:nii.gz")])

    assert CaseMaterializer(manager).materialize() is Verdict.WHOLE_VOLUME
    assert Dataset(tmp_path / "out", "nii.gz").is_dataset_exist("CT", "CASE_000")


def test_materialize_never_reads_the_volume_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory bound, asserted as a bound: the streamed branch must not call the whole-volume read.

    Equality of values would pass even if the streamed path loaded everything; only forbidding the
    full read proves the case never landed in memory."""
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0), Save(str(tmp_path / "out"))])

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("materialize() read the whole volume")

    monkeypatch.setattr(Dataset, "read_data", refuse)
    assert CaseMaterializer(manager).materialize() is Verdict.STREAM


def test_second_run_skips_the_case(tmp_path: Path) -> None:
    """Per-case resume, for free: the Save boundary answers, so nothing is recomputed or rewritten."""
    source = _source(tmp_path)
    transforms = [Clip(0.0, 50.0), Save(str(tmp_path / "out"))]
    assert CaseMaterializer(_manager(source, transforms)).materialize() is Verdict.STREAM

    written = Dataset(tmp_path / "out", "mha")
    before, _ = written.read_data("CT", "CASE_000")
    stamp = next((tmp_path / "out").glob("**/*")).stat().st_mtime_ns

    manager = _manager(source, transforms)
    assert CaseMaterializer(manager).materialize() is Verdict.STREAM
    after, _ = written.read_data("CT", "CASE_000")
    np.testing.assert_array_equal(after, before)
    assert next((tmp_path / "out").glob("**/*")).stat().st_mtime_ns == stamp
