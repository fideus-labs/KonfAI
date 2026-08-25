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

"""How :class:`Evaluator.update` feeds its metrics: the device moves and the values it records.

An ``Evaluator`` is faked with the attributes ``__init__`` sets, no config or dataset, so the update
paths run on in-memory batches.
"""

from types import SimpleNamespace

import pytest
import torch
from konfai.data.data_manager import BatchDataItem
from konfai.data.patching import DatasetPatch, SweepClock
from konfai.evaluator import Evaluator, Statistics
from konfai.metric.measure import MAE, MSE, SSIM


def _evaluator(metrics: dict[str, dict[str, dict[torch.nn.Module, None]]], streamed: bool = False) -> Evaluator:
    evaluator = object.__new__(Evaluator)
    evaluator.metrics = metrics
    evaluator._device = torch.device("cpu")
    evaluator._clock = SweepClock()
    evaluator._streamed = streamed
    evaluator._halo = 0
    evaluator._pending = {}
    evaluator._pending_name = None
    evaluator._last_result = {}
    evaluator._map_sinks = {}
    return evaluator


def _batch(name: str, p: int = 0, **tensors: torch.Tensor) -> dict[str, BatchDataItem]:
    return {
        group: BatchDataItem(name=[name], tensor=tensor, attribute=[None], x=[0], a=[0], p=[p], is_input=False)
        for group, tensor in tensors.items()
    }


@pytest.fixture
def registration_metrics() -> dict[str, dict[str, dict[torch.nn.Module, None]]]:
    # The shipped Registration evaluation: FIXED is the target of two outputs, and once more with a mask.
    return {
        "MOVED": {"FIXED": {MAE(): None, MSE(): None}},
        "MOVING": {"FIXED": {MAE(): None}, "FIXED;MASK": {MAE(): None}},
    }


def test_a_group_named_by_several_specs_is_moved_once_per_update(monkeypatch, registration_metrics):
    moves: list[str] = []
    original = Evaluator._on

    def counting(tensor, device):
        moves.append(str(tensor.shape))
        return original(tensor, device)

    monkeypatch.setattr(Evaluator, "_on", staticmethod(counting))
    batch = _batch("case", FIXED=torch.rand(1, 1, 4, 4), MOVED=torch.rand(1, 1, 4, 4), MOVING=torch.rand(1, 1, 4, 4))
    batch["MASK"] = _batch("case", MASK=torch.ones(1, 1, 4, 4, dtype=torch.uint8))["MASK"]

    moved = _evaluator(registration_metrics)._groups_on(batch)

    assert set(moved) == {"FIXED", "MOVED", "MOVING", "MASK"}
    assert len(moves) == 4  # FIXED once, not once per spec (three) nor per metric
    assert moved["FIXED"] is batch["FIXED"].tensor  # already on the device: handed back, not copied


def test_update_scores_every_spec_from_the_shared_tensors(registration_metrics):
    torch.manual_seed(0)
    fixed, moved, moving = torch.rand(1, 1, 4, 5), torch.rand(1, 1, 4, 5), torch.rand(1, 1, 4, 5)
    mask = torch.zeros(1, 1, 4, 5, dtype=torch.uint8)
    mask[..., :2, :] = 1
    batch = _batch("case", FIXED=fixed, MOVED=moved, MOVING=moving, MASK=mask)
    statistics = Statistics(None)

    result = _evaluator(registration_metrics).update(batch, statistics)

    assert result["MOVED:FIXED:MAE"] == pytest.approx((moved - fixed).abs().mean().item())
    assert result["MOVED:FIXED:MSE"] == pytest.approx((moved - fixed).pow(2).mean().item())
    assert result["MOVING:FIXED:MAE"] == pytest.approx((moving - fixed).abs().mean().item())
    assert result["MOVING:FIXED;MASK:MAE"] == pytest.approx((moving - fixed).abs()[..., :2, :].mean().item())
    assert statistics.measures["case"] == result
    # A metric never writes into the tensors it is handed: sharing one upload is safe.
    assert torch.equal(batch["FIXED"].tensor, fixed) and torch.equal(batch["MASK"].tensor, mask)


class TestClockReport:
    """Where a split's wall clock went: one line per split, in the sweep's format, above a second."""

    def test_update_charges_the_move_and_each_metric_by_name(self, registration_metrics):
        evaluator = _evaluator(registration_metrics)
        batch = _batch("case", **{g: torch.rand(1, 1, 8, 8) for g in ("FIXED", "MOVED", "MOVING", "MASK")})

        evaluator.update(batch, Statistics(None))

        assert evaluator._clock.spent("h2d") > 0.0
        assert evaluator._clock.spent("MAE") > 0.0 and evaluator._clock.spent("MSE") > 0.0
        assert evaluator._clock.spent("map") == 0.0  # no SaveMap metric wrote anything

    def test_report_is_silent_under_a_second_and_closes_on_other(self, registration_metrics):
        evaluator = _evaluator(registration_metrics)
        with evaluator._clock.phase("split"):
            with evaluator._clock.phase("MAE"):
                pass
            with evaluator._clock.phase("wait(load)"):
                pass

        assert evaluator._clock_report("TRAIN") is None
        report = evaluator._clock_report("TRAIN", min_seconds=0.0)

        assert report.startswith("[KonfAI] evaluation TRAIN ")
        assert " = wait(load) 0.0 + MAE 0.0 + other 0.0" in report  # MSE, h2d, map, flush: nothing to say
        assert "MSE" not in report


def test_streamed_update_moves_each_group_once_per_patch(monkeypatch, registration_metrics):
    moves: list[str] = []
    original = Evaluator._on
    monkeypatch.setattr(Evaluator, "_on", staticmethod(lambda t, d: (moves.append("x"), original(t, d))[1]))
    evaluator = _evaluator(registration_metrics, streamed=True)
    torch.manual_seed(1)
    volumes = {g: torch.rand(1, 1, 6, 5) for g in ("FIXED", "MOVED", "MOVING")}
    volumes["MASK"] = (torch.rand(1, 1, 6, 5) > 0.3).to(torch.uint8)
    statistics = Statistics(None)

    for z in (0, 3):
        evaluator.update(_batch("case", **{g: t[..., z : z + 3, :] for g, t in volumes.items()}), statistics)
    evaluator._flush_pending(statistics)

    assert len(moves) == 8  # four groups, two patches
    whole = _evaluator(registration_metrics).update(_batch("case", **volumes), Statistics(None))
    for key, value in whole.items():
        assert statistics.measures["case"][key] == pytest.approx(value, rel=1e-6)


class TestStreamedUpdateWithAHalo:
    """A grid reading a halo: a metric that declared one is handed the read and its slot's place in
    it, the others the slot alone, and each case ends on the whole-volume value."""

    @staticmethod
    def _grid(shape: list[int], halo: int) -> DatasetPatch:
        patch = DatasetPatch(patch_size=[5, 6], overlap=0)
        patch.pad_to_patch = False
        patch.halo = halo
        patch.load(shape, 0)
        return patch

    @classmethod
    def _stream(cls, metrics, volumes: dict[str, torch.Tensor], halo: int) -> tuple[Statistics, list]:
        """Feed every patch of the grid as the loader would read it, halo included; the MAE states."""
        shape = list(next(iter(volumes.values())).shape[2:])
        patch = cls._grid(shape, halo)
        evaluator = _evaluator(metrics, streamed=True)
        evaluator._halo = halo
        evaluator._iter_dataset = SimpleNamespace(get_dataset_from_index=lambda group, x: SimpleNamespace(patch=patch))
        statistics = Statistics(None)
        states = []
        for index in range(patch.get_size(0)):
            read = patch.read_slices(0, index, shape)
            evaluator.update(_batch("case", index, **{g: t[(..., *read)] for g, t in volumes.items()}), statistics)
            states.append(next(iter(evaluator._pending.values()))[1][-1])
        evaluator._flush_pending(statistics)
        return statistics, states

    @pytest.mark.parametrize("masked", [False, True])
    def test_a_halo_metric_ends_on_the_whole_volume_value(self, masked):
        torch.manual_seed(2)
        volumes = {"CT": torch.rand(1, 1, 17, 23), "sCT": torch.rand(1, 1, 17, 23)}
        target = "CT"
        if masked:
            volumes["MASK"] = (torch.rand(1, 1, 17, 23) > 0.3).to(torch.uint8)
            target = "CT;MASK"
        metrics = {"sCT": {target: {MAE(): None, SSIM(dynamic_range=1.0): None}}}

        statistics, _ = self._stream(metrics, volumes, SSIM.halo)

        whole = _evaluator(metrics).update(_batch("case", **volumes), Statistics(None))
        assert statistics.measures["case"][f"sCT:{target}:SSIM"] == pytest.approx(whole[f"sCT:{target}:SSIM"], rel=1e-9)
        assert statistics.measures["case"][f"sCT:{target}:MAE"] == pytest.approx(whole[f"sCT:{target}:MAE"], rel=1e-6)

    def test_a_metric_without_a_halo_sees_the_slot_alone(self):
        # MAE's partial states with the grid read with SSIM's halo are the states without it: the
        # context past the slot never reaches a metric that did not ask for it.
        torch.manual_seed(3)
        volumes = {"CT": torch.rand(1, 1, 17, 23), "sCT": torch.rand(1, 1, 17, 23)}
        metrics = {"sCT": {"CT": {MAE(): None, SSIM(dynamic_range=1.0): None}}}

        _, with_halo = self._stream(metrics, volumes, SSIM.halo)
        _, without = self._stream({"sCT": {"CT": {MAE(): None}}}, volumes, 0)

        assert with_halo == without
