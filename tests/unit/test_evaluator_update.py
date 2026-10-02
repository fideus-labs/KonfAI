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

import json
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from konfai.data.data_manager import BatchDataItem
from konfai.data.patching import DatasetPatch
from konfai.evaluator import Evaluator, Statistics
from konfai.metric.measure import (
    MAE,
    MSE,
    SSIM,
    CriterionWithAttribute,
    Dice,
    FocalLoss,
    Gram,
    MAESaveMap,
    Mean,
)
from konfai.metric.schedulers import Constant
from konfai.network.network import CriterionsAttr, Measure, ModuleArgsDict
from konfai.utils.clock import SweepClock
from konfai.utils.dataset import Attribute
from konfai.utils.errors import EvaluatorError, KonfAIWarning


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
    evaluator._scored_names = set()
    evaluator.dataset = SimpleNamespace(unreadable=({}, {}))
    return evaluator


def _batch(name: str, p: int = 0, **tensors: torch.Tensor) -> dict[str, BatchDataItem]:
    return {
        group: BatchDataItem(name=[name], tensor=tensor, attribute=[Attribute()], x=[0], a=[0], p=[p], is_input=False)
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


class _ReadsGeometry(CriterionWithAttribute):
    """A metric that places its tensors itself from their headers."""

    def forward(self, output, *targets, attributes):
        return torch.tensor(0.0)


class TestGridMismatch:
    """An output and a target a metric compares voxel to voxel lie on one grid, or the case is refused."""

    @pytest.mark.parametrize("output_z", [16, 1])  # torch raises at 16, broadcasts in silence at 1
    def test_an_output_of_another_shape_is_refused_naming_the_case(self, output_z):
        batch = _batch("CASE_1", sCT=torch.rand(1, 1, output_z, 16, 16), CT=torch.rand(1, 1, 8, 16, 16))
        statistics = Statistics(None)

        with pytest.raises(EvaluatorError, match=r"CASE_1(.|\n)*'sCT'(.|\n)*'CT'"):
            _evaluator({"sCT": {"CT": {MAE(): None}}}).update(batch, statistics)
        assert statistics.measures == {}

    def test_a_mask_of_another_shape_is_refused(self):
        batch = _batch("CASE_1", sCT=torch.rand(1, 1, 8, 8), CT=torch.rand(1, 1, 8, 8), MASK=torch.ones(1, 1, 8, 1))

        with pytest.raises(EvaluatorError, match="'MASK'"):
            _evaluator({"sCT": {"CT;MASK": {MAE(): None}}}).update(batch, Statistics(None))


def _header(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), direction=None) -> Attribute:
    """An image header as a reader returns it."""
    attribute = Attribute()
    attribute["Origin"] = np.asarray(origin, dtype=np.float64)
    attribute["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attribute["Direction"] = np.eye(len(origin)).reshape(-1) if direction is None else np.asarray(direction)
    return attribute


def _headed(batch: dict[str, BatchDataItem], **headers: Attribute) -> dict[str, BatchDataItem]:
    return {
        group: replace(item, attribute=[headers[group]]) if group in headers else item for group, item in batch.items()
    }


def _stored_as(evaluator: Evaluator, **formats: str) -> Evaluator:
    """The evaluator reading each group from a dataset of the given format, ``mha`` by default."""
    evaluator._iter_dataset = SimpleNamespace(
        get_dataset_from_index=lambda group, x: SimpleNamespace(
            dataset=SimpleNamespace(file_format=formats.get(group, "mha"))
        )
    )
    return evaluator


def _geometry_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, KonfAIWarning) and "geometry" in str(w.message)]


class TestGeometryMismatch:
    """An output and a target of one shape but two geometries are warned about once per case, and scored."""

    @pytest.mark.parametrize(
        ("output_header", "key", "values"),
        [
            (_header(origin=(10.0, 0.0, 0.0)), "Origin", ("[10.0, 0.0, 0.0]", "[0.0, 0.0, 0.0]")),
            (_header(origin=(0.01, 0.0, 0.0)), "Origin", ("[0.01, 0.0, 0.0]", "[0.0, 0.0, 0.0]")),  # 1/100 voxel
            (_header(spacing=(2.0, 1.0, 1.0)), "Spacing", ("[2.0, 1.0, 1.0]", "[1.0, 1.0, 1.0]")),
            (_header(direction=np.diag([-1.0, 1.0, 1.0]).reshape(-1)), "Direction", ("[-1.0, 0.0", "[1.0, 0.0")),
        ],
    )
    def test_a_pair_on_two_geometries_is_warned_about_and_scored(self, output_header, key, values):
        torch.manual_seed(0)
        sct, ct = torch.rand(1, 1, 4, 5, 6), torch.rand(1, 1, 4, 5, 6)
        batch = _headed(_batch("CASE_1", sCT=sct, CT=ct), sCT=output_header, CT=_header())
        statistics = Statistics(None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _stored_as(_evaluator({"sCT": {"CT": {MAE(): None, MSE(): None}}})).update(batch, statistics)

        [message] = _geometry_warnings(caught)
        assert "CASE_1" in message and "'sCT'" in message and "'CT'" in message
        assert key in message and all(value in message for value in values)
        assert result["sCT:CT:MAE"] == pytest.approx((sct - ct).abs().mean().item())
        assert statistics.measures["CASE_1"] == result

    @pytest.mark.parametrize(
        ("metric", "output_header", "target_header", "output_format", "output_shape"),
        [
            (MAE(), _header(), _header(), "mha", [2, 4, 4]),
            # Rounding: the origin within 1e-3 x the smallest spacing, the spacing within 1e-6 x the first,
            # the direction within 1e-6.
            (MAE(), _header(origin=(5e-4, 0.0, 0.0), spacing=(1.0 + 5e-7, 1.0, 1.0)), _header(), "mha", [2, 4, 4]),
            (  # the Synthesis demo's 1THA001: its MR, hence its sCT, lies 6.1e-5 mm from its CT
                MAE(),
                _header(origin=(-254.5, -91.5, -134.99993896484375), spacing=(1.0, 1.0, 2.0)),
                _header(origin=(-254.5, -91.5, -135.0), spacing=(1.0, 1.0, 2.0)),
                "mha",
                [2, 4, 4],
            ),
            (MAE(), _header(direction=np.eye(3).reshape(-1) + 5e-7), _header(), "mha", [2, 4, 4]),
            (
                MAE(),
                _header(origin=(1e-4, 0, 0), spacing=(1e3, 1e3, 1e3)),
                _header(spacing=(1e3, 1e3, 1e3)),
                "mha",
                [2, 4, 4],
            ),
            (Mean(), _header(origin=(10.0, 0.0, 0.0)), _header(), "mha", [2, 4, 4]),  # never compares voxels
            (Gram(), _header(origin=(10.0, 0.0, 0.0)), _header(), "mha", [2, 4, 4]),
            (_ReadsGeometry(), _header(origin=(10.0, 0.0, 0.0)), _header(), "mha", [2, 4, 4]),  # places both itself
            (MAE(), _header(origin=(10.0, 0.0, 0.0)), _header(), "png", [2, 4, 4]),  # a png stores no origin
            (MAE(), Attribute(), _header(), "h5", [2, 4, 4]),  # an h5 entry without attributes
            (MAE(), _header(origin=(10.0, 0.0), spacing=(1.0, 1.0)), _header(), "mha", [4, 4]),  # beside [1, 4, 4]
        ],
    )
    def test_no_warning_where_nothing_differs_or_nothing_compares_the_geometry(
        self, metric, output_header, target_header, output_format, output_shape
    ):
        target_shape = [1, 4, 4] if len(output_shape) == 2 else output_shape
        batch = _headed(
            _batch("CASE_1", sCT=torch.ones(1, 1, *output_shape), CT=torch.ones(1, 1, *target_shape)),
            sCT=output_header,
            CT=target_header,
        )
        evaluator = _stored_as(_evaluator({"sCT": {"CT": {metric: None}}}), sCT=output_format)

        with warnings.catch_warnings():
            warnings.simplefilter("error", KonfAIWarning)
            evaluator.update(batch, Statistics(None))

    def test_a_nifti_prediction_against_an_mha_reference_is_not_warned_about(self, tmp_path):
        sitk = pytest.importorskip("SimpleITK")
        # NIfTI stores the origin in float32: -135.1 comes back 6.1e-6 mm away, a voxel of 1 mm has not moved.
        image = sitk.Image([6, 5, 4], sitk.sitkFloat32)
        image.SetOrigin((-254.123456789, -91.987654321, -135.1))
        sitk.WriteImage(image, str(tmp_path / "CT.mha"))
        sitk.WriteImage(image, str(tmp_path / "sCT.nii.gz"))
        headers = {}
        for group, file in (("CT", "CT.mha"), ("sCT", "sCT.nii.gz")):
            read = sitk.ReadImage(str(tmp_path / file))
            headers[group] = _header(read.GetOrigin(), read.GetSpacing(), read.GetDirection())
        assert not np.array_equal(headers["CT"].get_np_array("Origin"), headers["sCT"].get_np_array("Origin"))
        batch = _headed(_batch("CASE_1", sCT=torch.ones(1, 1, 4, 5, 6), CT=torch.ones(1, 1, 4, 5, 6)), **headers)
        evaluator = _stored_as(_evaluator({"sCT": {"CT": {MAE(): None}}}), sCT="nii.gz")

        with warnings.catch_warnings():
            warnings.simplefilter("error", KonfAIWarning)
            evaluator.update(batch, Statistics(None))

    @pytest.mark.parametrize("metric", [Dice(labels=[1]), FocalLoss()])
    def test_a_label_metric_on_the_output_shape_is_warned_about(self, metric):
        # On the output's own shape Dice and FocalLoss take the target as it is: a voxel away, it is scored there.
        label = torch.zeros(1, 1, 4, 5, 6, dtype=torch.uint8)
        label[..., 1:3, :, :] = 1
        output = torch.cat([1 - label, label], 1).float()
        batch = _headed(_batch("CASE_1", PRED=output, SEG=label), PRED=_header(origin=(1.0, 0.0, 0.0)), SEG=_header())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _stored_as(_evaluator({"PRED": {"SEG": {metric: None}}})).update(batch, Statistics(None))

        [message] = _geometry_warnings(caught)
        assert "CASE_1" in message and "'PRED'" in message and "'SEG'" in message and metric.get_name() in message
        assert "[1.0, 0.0, 0.0]" in message and "[0.0, 0.0, 0.0]" in message
        assert f"PRED:SEG:{metric.get_name()}" in result


def test_two_rank_evaluation_refuses_a_single_file_map(tmp_path: Path) -> None:
    """Every rank writes its cases' error maps: into one h5, a map can go missing with exit 0."""
    evaluator = _evaluator({"MOVED": {"FIXED": {MAESaveMap(dataset=f"{tmp_path / 'Maps'}:h5"): None}}})

    with pytest.raises(EvaluatorError, match="single-file store"):
        evaluator.setup(2)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("streamed", [False, True])
def test_two_metrics_of_one_class_keep_a_row_each(streamed: bool) -> None:
    """Two metrics of one class once wrote one row, the second over the first."""
    torch.manual_seed(4)
    volumes = {"SEG": torch.randint(0, 3, (1, 1, 6, 5)), "PRED": torch.randint(0, 3, (1, 1, 6, 5))}
    metrics = {"PRED": {"SEG": {Dice(labels=[1]): None, Dice(labels=[2]): None}}}
    statistics = Statistics(None)
    evaluator = _evaluator(metrics, streamed=streamed)

    evaluator.update(_batch("case", **volumes), statistics)
    evaluator._flush_pending(statistics)

    row = statistics.measures["case"]
    assert row["PRED:SEG:Dice"] == row["PRED:SEG:Dice:1"] == Dice(labels=[1])(volumes["PRED"], volumes["SEG"])[1][1]
    assert row["PRED:SEG:Dice#2"] == row["PRED:SEG:Dice#2:2"] == Dice(labels=[2])(volumes["PRED"], volumes["SEG"])[1][2]


def _who(group: str) -> Attribute:
    attribute = Attribute()
    attribute["Who"] = group
    return attribute


def _score_in_training_and_in_evaluation(criterion, output_group: str, target_group: str) -> None:
    """Score ``criterion`` once through the trainer's ``Measure.update`` and once through
    ``Evaluator.update``, on one case whose every group's attribute carries the group's name."""
    tensors = {group: torch.rand(1, 1, 4, 4) for group in (output_group, *target_group.split(";"))}
    graph = ModuleArgsDict()
    graph.add_module(output_group, torch.nn.Identity())
    attribute = CriterionsAttr()
    attribute.schedulers = {Constant(): None}
    measure = Measure("Net", {})
    measure.outputs_criterions = {output_group: {target_group: {criterion: attribute}}}
    measure.init(graph, list(tensors))
    targets = {group: (tensor, [_who(group)]) for group, tensor in tensors.items() if group != output_group}
    measure.update(output_group, tensors[output_group], targets, it=0, nb_patch=1, training=False)

    batch = {
        group: BatchDataItem(name=["case"], tensor=tensor, attribute=[_who(group)], x=[0], a=[0], p=[0], is_input=False)
        for group, tensor in tensors.items()
    }
    _evaluator({output_group: {target_group: {criterion: None}}}).update(batch, Statistics(None))


def test_a_criterion_gets_the_same_attributes_in_training_and_in_evaluation() -> None:
    """``attributes`` holds the targets' own, in the order of the target group, through both calls.
    The evaluator once put the output's first, so one criterion indexed two different lists."""

    class Recorder(CriterionWithAttribute):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[list[str]] = []

        def forward(self, output, *targets, attributes):
            self.seen.append([samples[0]["Who"] for samples in attributes])
            return torch.zeros(())

    recorder = Recorder()

    _score_in_training_and_in_evaluation(recorder, "sCT", "CT;MASK")

    assert recorder.seen == [["CT", "MASK"], ["CT", "MASK"]]


@pytest.mark.parametrize(
    ("name", "target_group", "training", "evaluation"),
    [
        ("IMPACTReg", "CT;MASK", [("CT", "CT")], [("sCT", "CT")]),
        ("SAM_Perceptual", "CT;MASK", [("CT", "CT")], [("CT", "CT")]),
        ("IMPACTSynth", "CT;MR", [("CT", "CT"), ("MR", "MR")], [("sCT", "CT"), ("MR", "MR")]),
        ("IMPACTSynth", "CT;MR;MASK", [("CT", "CT"), ("MR", "MR")], [("sCT", "CT"), ("MR", "MR")]),
    ],
)
def test_an_impact_criterion_reads_the_statistics_of_the_images_it_compares(
    name: str, target_group: str, training: list, evaluation: list
) -> None:
    """Each image the extractor sees is normalized by the statistics of a group it compares: the
    output by its own when it has any (a prediction on disk, not a model output), else by the
    reference's; never by the mask's. IMPACTReg read the MASK's in training, IMPACTSynth failed there
    (IndexError) or read the MR's and the MASK's, and SAM_Perceptual read the prediction's in
    evaluation."""
    from konfai.metric.measure import impact

    seen: list[tuple[str, str]] = []

    def slice_losses(output, output_attributes, target, target_attributes, mask, loss, project=None):
        seen.append((output_attributes[0]["Who"], target_attributes[0]["Who"]))
        return iter([(torch.tensor(1.0), 1)])

    stub = SimpleNamespace(slice_losses=slice_losses)
    criterion = getattr(impact, name).__new__(getattr(impact, name))
    torch.nn.Module.__init__(criterion)
    criterion.__dict__.update(
        name=name, model=stub, content=stub, style=stub, loss=None, content_loss=None, style_loss=None, pca=0
    )

    _score_in_training_and_in_evaluation(criterion, "sCT", target_group)

    assert seen == training + evaluation


class _Loader(list):
    """The loader ``_evaluate_split`` walks: its batches, and a dataset that loads nothing."""

    dataset = SimpleNamespace(load=lambda label: None)


def test_a_resumed_run_rescores_a_case_scored_before_a_metric_was_added(tmp_path: Path) -> None:
    """An interrupted run scored CASE_0 with MAE alone; the config now also names MSE."""
    evaluator = _evaluator({"sCT": {"CT": {MAE(): None, MSE(): None}}})
    evaluator.metric_path = tmp_path
    statistics = Statistics(tmp_path / "Metric_TRAIN.json")
    row = {"name": "CASE_0", "values": {"sCT:CT:MAE": 0.5}}
    (tmp_path / "Metric_TRAIN.cases.rank0.jsonl").write_text(json.dumps(row) + "\n")
    zeros = torch.zeros(1, 1, 4, 4)
    loader = _Loader([_batch(f"CASE_{i}", sCT=zeros + i, CT=zeros) for i in range(2)])

    evaluator._evaluate_split(loader, statistics, "TRAIN", 1, 0, 0)

    report = json.loads(statistics.filename.read_text())
    assert report["aggregates"]["sCT:CT:MSE"]["count"] == 2
    assert report["case"]["sCT:CT:MAE"] == {"CASE_0": 0.0, "CASE_1": 1.0}


def test_a_resumed_run_reuses_the_cases_scored_with_the_same_metrics(tmp_path: Path) -> None:
    """A row holding every configured metric, a dict-valued one's components included, is not scored again."""
    evaluator = _evaluator({"PRED": {"SEG": {MAE(): None, Dice(labels=[1]): None}}})
    evaluator.metric_path = tmp_path
    statistics = Statistics(tmp_path / "Metric_TRAIN.json")
    row = {"name": "CASE_0", "values": {"PRED:SEG:MAE": 0.5, "PRED:SEG:Dice:1": 0.25, "PRED:SEG:Dice": 0.25}}
    (tmp_path / "Metric_TRAIN.cases.rank0.jsonl").write_text(json.dumps(row) + "\n")
    ones = torch.ones(1, 1, 4, 4, dtype=torch.uint8)
    loader = _Loader([_batch(f"CASE_{i}", PRED=ones, SEG=ones) for i in range(2)])

    evaluator._evaluate_split(loader, statistics, "TRAIN", 1, 0, 0)

    report = json.loads(statistics.filename.read_text())
    assert report["case"]["PRED:SEG:MAE"] == {"CASE_0": 0.5, "CASE_1": 0.0}
    assert report["case"]["PRED:SEG:Dice:1"] == {"CASE_0": 0.25, "CASE_1": 1.0}


def test_a_case_set_aside_mid_stream_is_scored_neither_in_part_nor_after(tmp_path: Path) -> None:
    """A chunk of CASE_1 the loader cannot read: its patches read before it are dropped, the ones read
    after it are skipped, and the report lists it beside the other cases' values."""
    evaluator = _evaluator({"sCT": {"CT": {MAE(): None}}}, streamed=True)
    evaluator.metric_path = tmp_path
    statistics = Statistics(tmp_path / "Metric_TRAIN.json")
    zeros = torch.zeros(1, 1, 2, 4)
    unreadable = {
        group: BatchDataItem([], torch.empty(0), [], [], [], [], False, unreadable=[(1, "CASE_1", "why")])
        for group in ("sCT", "CT")
    }
    loader = _Loader(
        [
            _batch("CASE_0", 0, sCT=zeros + 1, CT=zeros),
            _batch("CASE_1", 0, sCT=zeros + 5, CT=zeros),
            unreadable,
            _batch("CASE_1", 2, sCT=zeros + 7, CT=zeros),
            _batch("CASE_2", 0, sCT=zeros + 2, CT=zeros),
        ]
    )

    with pytest.warns(KonfAIWarning, match="CASE_1"):
        evaluator._evaluate_split(loader, statistics, "TRAIN", 1, 0, 0)

    report = json.loads(statistics.filename.read_text())
    assert report["case"]["sCT:CT:MAE"] == {"CASE_0": 1.0, "CASE_2": 2.0}
    assert report["aggregates"]["sCT:CT:MAE"]["count"] == 2
    assert report["set_aside"] == {"CASE_1": "why"}
