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

"""An unsatisfied ``Save`` whose prefix streams keeps the case streaming: the cache is materialized
slab by slab (the sweep) at first data access, after which the case streams from it exactly as if
the cache had always existed. The swept cache must equal the classically written one, the regime
probe must never write, and every failure must fall back to the whole-volume path."""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.patching import DatasetManager, DatasetPatch
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


def _manager(source: Dataset, transforms: list[Transform], patch_size: list[int] | None = None) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch(patch_size if patch_size is not None else [4, 5, 4]),
        transforms=transforms,
        data_augmentations_list=[],
    )


def _whole_volume_patches(manager: DatasetManager, transforms: list[Transform]) -> list[torch.Tensor]:
    reference = _manager(manager.dataset, transforms)
    reference.load(transforms, [], load_augmentations=False)
    return [
        reference.patch.get_data(reference._get_tensor(0), index, 0, True)
        for index in range(reference.patch.get_size(0))
    ]


def test_regime_probe_answers_yes_without_writing_the_cache(tmp_path: Path) -> None:
    source = _source(tmp_path)
    cache = Dataset(tmp_path / "cache", "mha")
    manager = _manager(source, [Clip(0.0, 50.0), Save(str(tmp_path / "cache"))])

    assert manager.can_stream_patch(0)
    assert not cache.is_dataset_exist("CT", "CASE_000")


def test_sweep_writes_the_same_cache_as_the_whole_volume_load(tmp_path: Path) -> None:
    source = _source(tmp_path)
    classic = [Clip(0.0, 50.0), Save(str(tmp_path / "cache_classic"))]
    swept = [Clip(0.0, 50.0), Save(str(tmp_path / "cache_swept"))]

    _manager(source, classic).load(classic, [], load_augmentations=False)
    manager = _manager(source, swept)
    patch = manager.get_data(0, 0, [], True)

    # The case streamed: the volume was never assembled, and the source boundary is the swept cache.
    assert not manager.loaded
    stream_source = manager._resolve_patch_stream_source(0)
    assert stream_source is not None and not stream_source.pending_sweeps
    assert Path(stream_source.dataset.filename).name == "cache_swept"

    expected, expected_attributes = Dataset(tmp_path / "cache_classic", "mha").read_data("CT", "CASE_000")
    result, result_attributes = Dataset(tmp_path / "cache_swept", "mha").read_data("CT", "CASE_000")
    np.testing.assert_array_equal(result, expected)
    for key in ("Origin", "Spacing", "Direction"):
        np.testing.assert_allclose(
            result_attributes.get_np_array(key), expected_attributes.get_np_array(key), err_msg=key
        )
    assert torch.equal(patch, _whole_volume_patches(manager, classic)[0])


def test_streamed_patches_after_the_sweep_match_the_whole_volume_path(tmp_path: Path) -> None:
    source = _source(tmp_path)
    transforms = [Clip(0.0, 50.0), Save(str(tmp_path / "cache"))]
    manager = _manager(source, transforms)
    reference = _whole_volume_patches(manager, [Clip(0.0, 50.0)])

    for index in range(manager.patch.get_size(0)):
        assert torch.equal(manager.get_data(index, 0, [], True), reference[index]), index
    assert not manager.loaded


def test_sweep_composes_region_stages_before_the_save(tmp_path: Path) -> None:
    # A Permute is an ORIENTATION stage: the sweep's slabs of the Save space pull remapped source
    # regions, and the materialized cache must still equal the whole-volume pass byte for byte.
    source = _source(tmp_path)
    classic = [Permute("2|1|0"), Clip(0.0, 50.0), Save(str(tmp_path / "cache_classic"))]
    swept = [Permute("2|1|0"), Clip(0.0, 50.0), Save(str(tmp_path / "cache_swept"))]

    _manager(source, classic).load(classic, [], load_augmentations=False)
    manager = _manager(source, swept)
    manager.get_data(0, 0, [], True)
    assert not manager.loaded

    expected, _ = Dataset(tmp_path / "cache_classic", "mha").read_data("CT", "CASE_000")
    result, _ = Dataset(tmp_path / "cache_swept", "mha").read_data("CT", "CASE_000")
    np.testing.assert_array_equal(result, expected)


def test_save_unlocks_a_global_statistic_after_a_value_changing_stage(tmp_path: Path) -> None:
    """[Clip, Standardize] cannot stream (the stored volume's statistic is not Standardize's input),
    but [Clip, Save, Standardize] can: the statistic seeds from the materialized post-Clip cache."""
    source = _source(tmp_path)
    assert not _manager(source, [Clip(0.0, 50.0), Standardize(inverse=False)]).can_stream_patch(0)

    transforms = [Clip(0.0, 50.0), Save(str(tmp_path / "cache")), Standardize(inverse=False)]
    manager = _manager(source, transforms)
    assert manager.can_stream_patch(0)
    reference = _whole_volume_patches(manager, [Clip(0.0, 50.0), Standardize(inverse=False)])

    for index in range(manager.patch.get_size(0)):
        got = manager.get_data(index, 0, [], True)
        np.testing.assert_allclose(got.numpy(), reference[index].numpy(), rtol=1e-6, err_msg=str(index))
    assert not manager.loaded


def test_two_saves_materialize_in_order(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first, second = str(tmp_path / "cache_first"), str(tmp_path / "cache_second")
    classic_first, classic_second = str(tmp_path / "classic_first"), str(tmp_path / "classic_second")
    classic = [Clip(0.0, 50.0), Save(classic_first), Clip(10.0, 40.0), Save(classic_second)]
    swept = [Clip(0.0, 50.0), Save(first), Clip(10.0, 40.0), Save(second)]

    _manager(source, classic).load(classic, [], load_augmentations=False)
    manager = _manager(source, swept)
    manager.get_data(0, 0, [], True)
    assert not manager.loaded

    for swept_cache, classic_cache in ((first, classic_first), (second, classic_second)):
        expected, _ = Dataset(Path(classic_cache), "mha").read_data("CT", "CASE_000")
        result, _ = Dataset(Path(swept_cache), "mha").read_data("CT", "CASE_000")
        np.testing.assert_array_equal(result, expected, err_msg=swept_cache)


def test_unstreamable_destination_keeps_the_whole_volume_path(tmp_path: Path) -> None:
    # nii.gz cannot serve region writes: the Save stays on the whole-volume path, as before the sweep.
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0), Save(f"{tmp_path / 'cache'}:nii.gz")])
    assert not manager.can_stream_patch(0)


def test_failed_sweep_falls_back_to_the_whole_volume_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from konfai.utils import dataset as dataset_module

    source = _source(tmp_path)
    transforms = [Clip(0.0, 50.0), Save(str(tmp_path / "cache"))]
    manager = _manager(source, transforms)
    reference = _whole_volume_patches(manager, [Clip(0.0, 50.0)])

    def broken_write(self, slices, data):
        raise OSError("disk full")

    monkeypatch.setattr(dataset_module._MhaDataStream, "write_slice", broken_write)
    with pytest.warns(UserWarning, match="falling back to the whole-volume path"):
        patch = manager.get_data(0, 0, [], True)
    monkeypatch.undo()

    assert manager._sweep_failed
    assert torch.equal(patch, reference[0])
    # The aborted sweep left no debris; the whole-volume fallback wrote the cache classically.
    assert not list((tmp_path / "cache").glob("**/*.tmp"))
    assert Dataset(tmp_path / "cache", "mha").is_dataset_exist("CT", "CASE_000")


def test_kill_switch_keeps_the_whole_volume_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STREAMED_WRITES", "0")
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0), Save(str(tmp_path / "cache"))])
    assert not manager.can_stream_patch(0)
