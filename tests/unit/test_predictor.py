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

"""Tests for ``konfai.predictor``: the background writer overlaps disk writes with the prediction
loop, byte-identically.

Writes are submitted to one worker per output dataset (in order, bounded queue, failures kept and
re-raised), but only when the destination serves disjoint files per entry
(``Dataset.concurrent_write_safe``): a single-store backend (h5, zarr) stays inline, so no store is
ever written from two threads. Pure ``threading``/``queue``, no fork and no signals, so the behaviour
is the same on Linux, macOS and Windows."""

import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from konfai.data.augmentation import Flip
from konfai.data.patching import Accumulator, blend_axes
from konfai.predictor import PREDICTION_CLOCK, OutputDataset, _AsyncWriter, _Predictor
from konfai.utils.dataset import Dataset
from konfai.utils.errors import PredictorError
from konfai.utils.utils import get_patch_slices_from_shape


def test_async_writer_charges_its_writes_to_the_writer_s_own_phase() -> None:
    """The writer thread is the one thread charging ``write``, so the report can set the writer's
    time beside how long the loop waited on it: a submission into a queue with room is no wait."""
    PREDICTION_CLOCK.reset()
    writer = _AsyncWriter()
    writer.submit(lambda: time.sleep(0.05))
    writer.close()
    assert PREDICTION_CLOCK.spent("write") >= 0.05
    assert PREDICTION_CLOCK.spent("wait(write)") == 0.0


def test_async_writer_runs_in_order_and_surfaces_failures() -> None:
    writer = _AsyncWriter()
    done: list[int] = []
    writer.submit(lambda: done.append(1))
    writer.submit(lambda: (_ for _ in ()).throw(PredictorError("destination died")))
    # Operations submitted after a failure drain unexecuted; the failure surfaces at the LATEST at
    # close(), possibly earlier at this submit if the worker already recorded it: accept it wherever
    # it lands, so a run can never end with a write silently missing.
    with pytest.raises(PredictorError, match="destination died"):
        writer.submit(lambda: done.append(2))
        writer.close()
    assert done == [1]


def test_concurrent_write_safety_is_declared_per_backend(tmp_path) -> None:
    assert Dataset(f"{tmp_path}/a", "mha").concurrent_write_safe()
    assert not Dataset(f"{tmp_path}/a.h5", "h5").concurrent_write_safe()
    assert not Dataset(f"{tmp_path}/a", "omezarr").concurrent_write_safe()


def test_async_streamed_writes_match_the_inline_reference(tmp_path, monkeypatch, drive_tta) -> None:
    # A per-file destination goes through the background writer (forced here: the automatic gate
    # also requires a GPU-placed output); the kill-switch runs the same store inline. Same
    # operations, same order: the files must match bit for bit.
    used: list[int] = []
    submit = _AsyncWriter.submit
    monkeypatch.setattr(_AsyncWriter, "submit", lambda self, op: (used.append(1), submit(self, op))[1])
    monkeypatch.setenv("KONFAI_ASYNC_WRITES", "1")
    asynchronous, whole_volume = drive_tta(
        tmp_path / "async", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="mha"
    )
    assert not whole_volume and used, "the mha destination should have taken the background writer"
    monkeypatch.setenv("KONFAI_ASYNC_WRITES", "0")
    inline, _ = drive_tta(
        tmp_path / "inline", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="mha"
    )
    assert torch.equal(asynchronous, inline)


def test_async_gate_stays_inline_for_single_stores_and_cpu_outputs(tmp_path, monkeypatch, drive_tta) -> None:
    used: list[int] = []
    submit = _AsyncWriter.submit
    monkeypatch.setattr(_AsyncWriter, "submit", lambda self, op: (used.append(1), submit(self, op))[1])
    # Even forced, a single-store destination never crosses threads.
    monkeypatch.setenv("KONFAI_ASYNC_WRITES", "1")
    drive_tta(tmp_path / "h5", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="h5")
    assert not used, "an h5 store must never be written from the background thread"
    # Automatic mode on a CPU-placed output stays inline: the blend already saturates the memory
    # bandwidth the writer would consume.
    monkeypatch.delenv("KONFAI_ASYNC_WRITES", raising=False)
    drive_tta(tmp_path / "cpu", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="mha")
    assert not used, "a CPU-only output must stay inline in automatic mode"


def test_a_built_in_reduction_binds_from_its_own_block_like_a_custom_one(write_config, monkeypatch) -> None:
    """``reduction: Mean`` used to build ``Mean()`` directly, so the ``Mean:`` block a resolved config
    carries was read by nothing; every operator now binds from its block, and the write-back says so."""
    import ruamel.yaml
    from konfai.data.reduction import Mean

    path = write_config("Predictor:\n  outputs_dataset:\n    L:\n      OutputDataset:\n        reduction: Mean\n")
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    output = OutputDataset(
        same_as_group="a:b",
        dataset_filename="./Out:mha",
        before_reduction_transforms={},
        after_reduction_transforms={},
        final_transforms={},
        reduction="Mean",
    )
    output.prepare("L")

    assert isinstance(output.reduction, Mean)
    block = ruamel.yaml.YAML().load(path.read_text())["Predictor"]["outputs_dataset"]["L"]["OutputDataset"]
    assert "Mean" in block


def _output_dataset(write_config, monkeypatch) -> OutputDataset:
    """An output dataset with its declared blend window built, as ``prepare`` leaves it."""
    write_config("Predictor:\n  outputs_dataset:\n    L:\n      OutputDataset: {}\n")
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    output = OutputDataset(
        same_as_group="a:b",
        dataset_filename="./Out:mha",
        before_reduction_transforms={},
        after_reduction_transforms={},
        final_transforms={},
    )
    output.prepare("L")
    return output


def _configure_blend(output: OutputDataset, patch_size: list[int], overlap: int) -> None:
    """Hand ``output`` the run's patch config the way the prediction loop does."""
    _Predictor(
        world_size=1,
        global_rank=0,
        local_rank=0,
        autocast=False,
        predict_path=Path("."),
        data_log=None,
        outputs_dataset={"L": output},
        model_composite=SimpleNamespace(module=SimpleNamespace(get_networks=dict)),
        dataloader_prediction=SimpleNamespace(
            dataset=SimpleNamespace(get_patch_config=lambda: (patch_size, overlap), data_augmentations_list=[])
        ),
    )


@pytest.mark.parametrize("patch_size", [[1, 4, 3], [4, 1, 3], [4, 4, 1]])
def test_a_singleton_patch_axis_still_carries_its_blend_window(write_config, monkeypatch, patch_size) -> None:
    """A 2.5-D grid tiles one axis a voxel at a time. The accumulator reads one window per spatial axis,
    so the untiled axes have to be handed over too, as a single broadcast entry: dropping them leaves the
    trailing axis with no window at all."""
    output = _output_dataset(write_config, monkeypatch)
    _configure_blend(output, patch_size, 1)
    assert [window.numel() for window in output.patch_combine.windows_1d] == blend_axes(patch_size)


@pytest.mark.parametrize("patch_size", [[1, 3, 3], [3, 1, 3], [3, 3, 1]])
def test_a_grid_with_a_singleton_patch_axis_reassembles_its_case(write_config, monkeypatch, patch_size) -> None:
    """The same grid, blended: the loader drops the singleton axis of each patch, the accumulator puts it
    back, and the case comes out of the blend as it went in."""
    output = _output_dataset(write_config, monkeypatch)
    _configure_blend(output, patch_size, 0)

    shape = [6, 6, 6]  # tiles exactly at overlap 0: no padded border patch to crop
    volume = torch.arange(math.prod(shape), dtype=torch.float32).reshape(1, *shape)
    slices = get_patch_slices_from_shape(patch_size, shape, 0)
    accumulator = Accumulator(slices, patch_size, output.patch_combine, batch=False)
    for index, patch in enumerate(slices):
        selector = tuple(s.start if s.stop - s.start == 1 else s for s in patch)
        accumulator.add_layer(index, volume[(slice(None), *selector)].clone())
    assert torch.equal(accumulator.assemble(), volume)


def test_the_published_sink_name_resolves_to_the_output_dataset() -> None:
    """Every published Prediction.yml names the sink ``OutSameAsGroupDataset``: the alias must resolve."""
    from konfai.predictor import OutputDataset
    from konfai.utils.utils import get_module

    module, name = get_module("OutSameAsGroupDataset", "konfai.predictor")
    assert getattr(module, name) is OutputDataset
