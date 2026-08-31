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

from typing import Any, ClassVar, cast

import numpy as np
import pytest
import torch
import tqdm
from konfai.data.data_manager import BatchDataItem, DatasetIter
from konfai.data.transform import TransformInverse
from konfai.network.network import Network
from konfai.predictor import (
    PREDICTION_CLOCK,
    Mean,
    ModelComposite,
    OutputDataset,
    _colocate_loaded_modules,
    _prediction_report,
    _Predictor,
)
from konfai.utils.clock import SweepClock
from konfai.utils.dataset import Attribute


class DummyPredictNetwork(Network):
    def __init__(self) -> None:
        super().__init__(in_channels=1)
        self.scale = 1.0
        self.load_history: list[float] = []
        self.load_name_history: list[str] = []

    def load(self, state_dict, init: bool = True, ema: bool = False):  # type: ignore[override]
        self.scale = float(state_dict["scale"])
        self.load_history.append(self.scale)
        self.load_name_history.append(self.get_name())

    def forward(self, batch_sample, output_layers=[]):  # type: ignore[override]
        tensor = next(iter(batch_sample.values())).tensor * self.scale
        return [("out", tensor)]


def test_model_composite_streams_ensemble_through_a_single_loaded_model() -> None:
    model = DummyPredictNetwork()
    composite = ModelComposite(model, Mean())
    composite.load([{"scale": 1.0}, {"scale": 3.0}])

    batch_sample = {
        "input": BatchDataItem(
            name=["CASE_000"],
            tensor=torch.ones(1, 1, 2, 2),
            attribute=[Attribute()],
            x=[0],
            a=[0],
            p=[0],
            is_input=True,
        )
    }

    outputs = composite(batch_sample, ["out"])
    streamed_model = composite["Model_0"]

    assert len(list(composite.keys())) == 1
    assert len(outputs) == 1
    assert outputs[0][0] == "out"
    assert outputs[0][1] == [1, 1]
    assert torch.allclose(outputs[0][2], torch.full((1, 1, 2, 2), 2.0, dtype=outputs[0][2].dtype))
    assert isinstance(streamed_model, DummyPredictNetwork)
    assert streamed_model.load_history == [1.0, 3.0]
    assert streamed_model.load_name_history == ["DummyPredictNetwork", "DummyPredictNetwork"]


def test_model_composite_hands_over_a_lone_model_output_and_folds_an_ensemble_in_place() -> None:
    """A single-model run hands the model's own output on: dividing it by one copied every batch
    output (56 MiB per [1, 14, 128^3] fp16 patch; 2 allocations per forward measured against 1).
    An ensemble's mean is folded into the first output in place, the same values as the
    out-of-place quotient to the bit.
    """

    class RecordingNetwork(DummyPredictNetwork):
        def forward(self, batch_sample, output_layers=[]):  # type: ignore[override]
            self.last = next(iter(batch_sample.values())).tensor * self.scale
            return [("out", self.last)]

    def batch() -> dict[str, BatchDataItem]:
        torch.manual_seed(0)
        return {
            "input": BatchDataItem(
                name=["CASE_000"],
                tensor=torch.randn(1, 1, 4, 4, dtype=torch.float16),
                attribute=[Attribute()],
                x=[0],
                a=[0],
                p=[0],
                is_input=True,
            )
        }

    composite = ModelComposite(RecordingNetwork(), Mean())
    composite.load([{"scale": 3.0}])
    output = composite(batch(), ["out"])[0][2]
    assert output is cast(RecordingNetwork, composite["Model_0"]).last

    composite = ModelComposite(RecordingNetwork(), Mean())
    composite.load([{"scale": 1.0}, {"scale": 3.0}, {"scale": 5.0}])
    folded = composite(batch(), ["out"])[0][2]
    outputs = [batch()["input"].tensor * scale for scale in (1.0, 3.0, 5.0)]
    assert torch.equal(folded, (outputs[0] + outputs[1] + outputs[2]) / 3)


def test_model_composite_runs_a_weightless_model_without_a_checkpoint() -> None:
    """A model with no trainable weights (0 parameters) runs once, as constructed, with no state sources: a
    classical/optimisation engine (e.g. registration) needs no checkpoint."""
    model = DummyPredictNetwork()  # scale=1.0, held as a plain float -> 0 parameters
    assert not list(model.parameters())
    composite = ModelComposite(model, Mean())
    composite.load([])  # no checkpoints

    batch_sample = {
        "input": BatchDataItem(
            name=["CASE_000"],
            tensor=torch.ones(1, 1, 2, 2),
            attribute=[Attribute()],
            x=[0],
            a=[0],
            p=[0],
            is_input=True,
        )
    }

    outputs = composite(batch_sample, ["out"])
    ran_model = cast(DummyPredictNetwork, composite["Model_0"])

    assert len(outputs) == 1
    assert outputs[0][0] == "out"
    # Runs with the constructed scale (1.0); no state was ever loaded.
    assert torch.allclose(outputs[0][2], torch.ones(1, 1, 2, 2, dtype=outputs[0][2].dtype))
    assert ran_model.load_history == []


def test_model_composite_refuses_empty_sources_for_a_model_with_weights() -> None:
    """Defense in depth: load([]) is only for a weightless model. A model WITH parameters and no checkpoint
    would run random weights, so the composite refuses it rather than relying on the Predictor's guard."""
    from konfai.utils.errors import PredictorError

    class WeightedNetwork(Network):
        def __init__(self) -> None:
            super().__init__(in_channels=1)
            # KonfAI registers weights through the add_module graph, as a real model does.
            self.add_module("Conv", torch.nn.Conv2d(1, 2, 3), in_branch=[0], out_branch=[0])

    weighted = WeightedNetwork()
    assert list(weighted.parameters())  # sanity: it really has trainable weights
    composite = ModelComposite(weighted, Mean())
    with pytest.raises(PredictorError, match="trainable parameters"):
        composite.load([])


def test_output_dataset_uses_batch_attributes_when_manager_cache_is_cold() -> None:
    class DummyPatch:
        patch_size: ClassVar[list[int]] = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(0, 2), slice(0, 2))]

        @staticmethod
        def get_sweep_axis(index_augmentation: int) -> int:
            # These grids are cut along axis 0, which is what these doubles' slices assume.
            del index_augmentation
            return 0

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        cache_attributes: ClassVar[list[Attribute]] = [Attribute({"Origin": [0.0, 0.0]})]

    class DummyGroupTransform:
        patch_transforms: ClassVar[list[object]] = []

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": DummyGroupTransform()}}

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            assert group_dest == "dest"
            assert index == 0
            return DummyManager()

    output_dataset = OutputDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )

    streamed_attribute = Attribute()
    streamed_attribute["Spacing"] = np.asarray([1.0, 1.0])
    streamed_attribute["Size"] = np.asarray([4, 4])
    streamed_attribute["Size"] = np.asarray([2, 2])

    output_dataset.add_layer(
        index_dataset=0,
        index_augmentation=0,
        index_patch=0,
        layer=torch.zeros(1, 2, 2),
        dataset=cast(DatasetIter, DummyDatasetIter()),
        attribute=streamed_attribute,
    )

    assert "Size" in output_dataset.attributes[0][0][0]
    assert output_dataset.attributes[0][0][0].get_np_array("Size").tolist() == [2.0, 2.0]


def test_output_dataset_offloads_patch_predictions_to_cpu_before_accumulating() -> None:
    class DummyPatch:
        patch_size: ClassVar[list[int]] = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(0, 2), slice(0, 2))]

        @staticmethod
        def get_sweep_axis(index_augmentation: int) -> int:
            # These grids are cut along axis 0, which is what these doubles' slices assume.
            del index_augmentation
            return 0

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        cache_attributes: ClassVar[list[Attribute]] = [Attribute({"Origin": [0.0, 0.0]})]

    class DummyGroupTransform:
        patch_transforms: ClassVar[list[object]] = []

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": DummyGroupTransform()}}

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            assert group_dest == "dest"
            assert index == 0
            return DummyManager()

    class FakeCudaTensor:
        def __init__(self) -> None:
            self.device = torch.device("cuda:0")
            self.cpu_calls = 0

        # Small enough to stay under the pinned-offload threshold, so ``_offload_to_cpu`` takes the
        # plain ``detach().cpu()`` path this test asserts on.
        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 4

        def detach(self):
            return self

        def cpu(self) -> torch.Tensor:
            self.cpu_calls += 1
            return torch.ones(1, 2, 2)

    output_dataset = OutputDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )

    fake_layer = FakeCudaTensor()
    output_dataset.add_layer(
        index_dataset=0,
        index_augmentation=0,
        index_patch=0,
        layer=cast(torch.Tensor, fake_layer),
        dataset=cast(DatasetIter, DummyDatasetIter()),
        attribute=Attribute(),
    )

    # The accumulator blends each patch straight into a running CPU buffer (no per-patch list), so
    # the offloaded patch must have been moved to CPU exactly once before being blended.
    accumulator = output_dataset.output_layer_accumulator[0][0]
    assert fake_layer.cpu_calls == 1
    assert accumulator._result is not None
    assert accumulator._result.device.type == "cpu"


def _single_patch_dataset_iter() -> DatasetIter:
    """A dataset double whose one case is one 2x2 patch, swept along axis 0, with no patch transform."""

    class DummyPatch:
        patch_size: ClassVar[list[int]] = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(0, 2), slice(0, 2))]

        @staticmethod
        def get_sweep_axis(index_augmentation: int) -> int:
            del index_augmentation
            return 0

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        cache_attributes: ClassVar[list[Attribute]] = [Attribute({"Origin": [0.0, 0.0]})]

    class DummyGroupTransform:
        patch_transforms: ClassVar[list[object]] = []

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": DummyGroupTransform()}}

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            assert group_dest == "dest"
            assert index == 0
            return DummyManager()

    return cast(DatasetIter, DummyDatasetIter())


def _three_patch_dataset_iter(patch_transforms: list[object]) -> DatasetIter:
    """A dataset double whose one case is three 2x2 patches down axis 0, mirroring ``patch_transforms``."""

    class DummyPatch:
        patch_size: ClassVar[list[int]] = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(row, row + 2), slice(0, 2)) for row in (0, 2, 4)]

        @staticmethod
        def get_sweep_axis(index_augmentation: int) -> int:
            del index_augmentation
            return 0

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        cache_attributes: ClassVar[list[Attribute]] = [Attribute({"Origin": [0.0, 0.0]})]

    class DummyGroupTransform:
        pass

    group = DummyGroupTransform()
    group.patch_transforms = patch_transforms  # type: ignore[attr-defined]

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": group}}

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            assert group_dest == "dest"
            assert index == 0
            return DummyManager()

    return cast(DatasetIter, DummyDatasetIter())


def test_patch_headers_are_copied_only_for_a_patch_inverse_to_undo_on() -> None:
    """One header per (case, augmentation), at index 0, unless a patch-level inverse reads one per
    patch: then every patch gets a copy of its own, taken before any inverse ran, so what one
    inverse pushes never reaches the next patch's. On a grid of 18,000 thin 2.5D patches with
    nothing to undo the copies cost 473 ms and 18,000 dicts held per (case, augmentation),
    measured; 17 ms and one header now.
    """

    class Undo(TransformInverse):
        def __init__(self) -> None:
            super().__init__(inverse=True)
            self.undone_on: list[Attribute] = []

        def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
            return tensor

        def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
            cache_attribute["Undone"] = 1
            self.undone_on.append(cache_attribute)
            return tensor

    def output_dataset() -> OutputDataset:
        return OutputDataset(
            same_as_group="src:dest",
            dataset_filename="./Output:mha",
            group="out",
            patch_combine=None,
            reduction="Mean",
        )

    plain = output_dataset()
    for index_patch in range(3):
        plain.add_layer(0, 0, index_patch, torch.zeros(1, 2, 2), _three_patch_dataset_iter([]), Attribute())
    assert list(plain.attributes[0][0]) == [0]

    undo = Undo()
    inverted = output_dataset()
    for index_patch in range(3):
        inverted.add_layer(0, 0, index_patch, torch.zeros(1, 2, 2), _three_patch_dataset_iter([undo]), Attribute())
    headers = inverted.attributes[0][0]
    assert list(headers) == [0, 1, 2]
    assert [undone is headers[index_patch] for index_patch, undone in enumerate(undo.undone_on)] == [True] * 3
    assert all("Undone_0" in header and "Undone_1" not in header for header in headers.values())


def test_get_output_hands_the_assembled_volume_on_as_a_view() -> None:
    """One model chunk: the assembled volume reaches the reduction without a copy.

    Stacking a lone chunk copied it: 448 MiB and 54 ms per copy of a [14, 256^3] fp16 case,
    measured, once per augmentation. No reduction writes into what it is handed (a Mean of one is
    the member itself, a fold of several copies first, Median/Vote/Concat build a new tensor), so
    the accumulator's own buffer, which assemble() has already let go of, can be the answer.
    """
    output_dataset = OutputDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    dataset = _single_patch_dataset_iter()
    output_dataset.add_layer(0, 0, 0, torch.arange(12, dtype=torch.float32).reshape(3, 2, 2), dataset, Attribute())
    assembled = output_dataset.output_layer_accumulator[0][0]._result
    assert assembled is not None

    stacked = output_dataset._get_output(0, 0, [3], dataset)

    assert stacked.shape == (1, 3, 2, 2)
    assert stacked.data_ptr() == assembled.data_ptr()
    assert torch.equal(stacked[0], torch.arange(12, dtype=torch.float32).reshape(3, 2, 2))


def test_predict_log_skips_measure_sync_when_tensorboard_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = _Predictor.__new__(_Predictor)
    predictor_any = cast(Any, predictor)
    predictor_any.tb = None
    predictor_any._has_runtime_measures = True
    predictor_any.world_size = 2
    predictor_any.global_rank = 0
    predictor_any.local_rank = 0
    predictor_any.model_composite = object()

    monkeypatch.setattr(
        "konfai.utils.runtime.DistributedObject.get_measure",
        staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected sync"))),
    )

    predictor._predict_log({})


def _loop_doubles(batches: int) -> tuple[_Predictor, Any, dict[str, Any], Any]:
    """A ``_Predictor`` over ``batches`` identical one-patch batches of one case, two outputs done at
    every batch, no TensorBoard: the doubles the loop tests drive."""

    class DummyPredictionDataset:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def load(self, label: str) -> None:
            self.labels.append(label)

    class DummyPredictionLoader:
        def __init__(self, batches: list[dict[str, BatchDataItem]], dataset: DummyPredictionDataset) -> None:
            self._batches = batches
            self.dataset = dataset

        def __iter__(self):
            return iter(self._batches)

        def __len__(self) -> int:
            return len(self._batches)

    class DummyOutputDataset:
        group_dest = "input"

        def __init__(self) -> None:
            self.writes = 0

        def add_layer(self, *args, **kwargs) -> None:
            pass

        def is_done(self, index: int) -> bool:
            assert index == 0
            return True

        def get_output(self, index: int, number_of_channels_per_model: list[int], dataset: DummyPredictionDataset):
            assert index == 0
            assert number_of_channels_per_model == [1]
            assert isinstance(dataset, DummyPredictionDataset)
            return torch.ones(1, 2, 2)

        def write_prediction(self, index: int, name: str, layer: torch.Tensor) -> None:
            assert index == 0
            assert name == "CASE_000"
            assert layer.shape == (1, 2, 2)
            self.writes += 1

        def finalize_writes(self) -> None:
            pass

    class DummyCompositeModule:
        @staticmethod
        def set_state(state) -> None:
            del state

        @staticmethod
        def get_networks() -> dict[str, object]:
            return {}

    class DummyComposite:
        def __init__(self) -> None:
            self.module = DummyCompositeModule()
            self.eval_calls = 0
            self.eval = self._eval

        def _eval(self) -> None:
            self.eval_calls += 1

        def __call__(self, batch_sample, output_layers):
            assert output_layers == ["out_a", "out_b"]
            return [
                ("out_a", [1], torch.ones(1, 1, 2, 2)),
                ("out_b", [1], torch.ones(1, 1, 2, 2) * 2),
            ]

    dataset = DummyPredictionDataset()
    batch = {
        "input": BatchDataItem(
            name=["CASE_000"],
            tensor=torch.ones(1, 1, 2, 2),
            attribute=[Attribute()],
            x=[0],
            a=[0],
            p=[0],
            is_input=True,
        )
    }
    loader = DummyPredictionLoader([batch] * batches, dataset)
    outputs_dataset = {"out_a": DummyOutputDataset(), "out_b": DummyOutputDataset()}
    model_composite = DummyComposite()

    predictor = _Predictor.__new__(_Predictor)
    predictor_any = cast(Any, predictor)
    predictor_any.world_size = 1
    predictor_any.global_rank = 0
    predictor_any.local_rank = 0
    predictor_any.model_composite = model_composite
    predictor_any.dataloader_prediction = loader
    predictor_any.outputs_dataset = outputs_dataset
    predictor_any.autocast = False
    predictor_any.it = 0
    predictor_any.dataset = dataset
    predictor_any.tb = None
    return predictor, dataset, outputs_dataset, model_composite


def test_predictor_runs_prediction_logging_once_per_batch_even_with_multiple_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor, dataset, outputs_dataset, model_composite = _loop_doubles(batches=1)

    log_calls: list[int] = []
    monkeypatch.setattr(predictor, "_predict_log", lambda batch_sample: log_calls.append(len(batch_sample)))
    monkeypatch.setattr("konfai.predictor.loop.description", lambda model: "stub")

    predictor.run()

    assert log_calls == [1]
    assert dataset.labels == ["Prediction"]
    assert model_composite.eval_calls == 1
    assert outputs_dataset["out_a"].writes == 1
    assert outputs_dataset["out_b"].writes == 1


def test_prediction_loop_refreshes_its_status_every_tenth_batch_and_clocks_its_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bar's status (an NVML query, two psutil calls and a forced redraw: 95 us per batch,
    measured) is set every ``_DESCRIPTION_EVERY`` batches without a redraw; the bar redraws on its
    own clock. The loop charges ``PREDICTION_CLOCK``: the fetch of each batch, the forward, the
    case finalize, and the run's wall, which bounds their sum; a run under a second reports nothing.
    """
    predictor, _dataset, outputs_dataset, _model_composite = _loop_doubles(batches=25)
    monkeypatch.setattr(predictor, "_predict_log", lambda batch_sample: None)
    described: list[str] = []
    monkeypatch.setattr("konfai.predictor.loop.description", lambda model: described.append("status") or "status")
    refreshes: list[bool] = []
    set_description = tqdm.tqdm.set_description

    def recording(self, desc=None, refresh=True):
        refreshes.append(refresh)
        return set_description(self, desc, refresh)

    monkeypatch.setattr(tqdm.tqdm, "set_description", recording)

    predictor.run()

    assert outputs_dataset["out_a"].writes == 25
    assert len(described) == 1 + 3, "the bar's own description, then batches 0, 10 and 20"
    assert refreshes == [False] * 3
    clock = PREDICTION_CLOCK
    assert clock.spent("fetch") > 0 and clock.spent("forward") > 0 and clock.spent("finalize(case)") > 0
    assert clock.spent("prediction") >= clock.spent("fetch") + clock.spent("forward") + clock.spent("finalize(case)")
    assert _prediction_report(clock) is None


def test_prediction_report_closes_on_the_loop_s_own_thread() -> None:
    """The sum before the bar is the loop's thread and closes exactly (what no phase names is
    ``other``); after it, the writer's own time and how long the loop waited on it."""
    clock = SweepClock()
    spent = {
        "prediction": 10.0,
        "fetch": 1.0,
        "forward": 5.0,
        "blend": 2.0,
        "finalize(stream)": 0.6,
        "finalize(case)": 0.2,
        "drain": 0.2,
        "write": 3.0,
        "wait(write)": 0.5,
    }
    clock._spent = dict(spent)

    assert _prediction_report(clock) == (
        "[KonfAI] prediction 10.0 s = fetch 1.0 + forward 5.0 + blend 2.0 + finalize(stream) 0.6"
        " + finalize(case) 0.2 + drain 0.2 + other 1.0 | writer 3.0 s, waited on 0.5 s"
    )
    clock._spent = {"prediction": 0.9}
    assert _prediction_report(clock) is None


def test_gate_approved_blend_falls_back_to_cpu_when_the_allocation_fails() -> None:
    # The gate samples free VRAM once per case; another process can reclaim it before the
    # volume-sized first allocation lands. The blend must retry on the memory-safe CPU path instead
    # of killing the run (``get_output`` reconciles mixed devices afterwards).
    class DummyGroupTransform:
        patch_transforms: ClassVar[list[object]] = []

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": DummyGroupTransform()}}

    class OOMThenRecordAccumulator:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def is_empty(self) -> bool:
            return True

        def add_layer(self, index: int, layer: torch.Tensor) -> None:
            if not self.devices and layer.device.type != "cpu":
                self.devices.append("oom")
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            self.devices.append(layer.device.type)

    class FakeCudaTensor:
        def __init__(self) -> None:
            self.device = torch.device("cuda:0")

        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 4

        def detach(self):
            return self

        def cpu(self) -> torch.Tensor:
            return torch.ones(1, 2, 2)

    output_dataset = OutputDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    accumulator = OOMThenRecordAccumulator()
    output_dataset.output_layer_accumulator[0] = {0: cast(Any, accumulator)}
    output_dataset.attributes[0] = {0: {0: Attribute()}}
    output_dataset.names[0] = "CASE_000"
    output_dataset._accum_device[0] = torch.device("cuda", 0)

    output_dataset.add_layer(
        index_dataset=0,
        index_augmentation=0,
        index_patch=0,
        layer=cast(torch.Tensor, FakeCudaTensor()),
        dataset=cast(DatasetIter, DummyDatasetIter()),
        attribute=Attribute(),
    )

    assert accumulator.devices == ["oom", "cpu"]
    assert output_dataset._accum_device[0].type == "cpu"


def test_mid_blend_oom_stays_fatal() -> None:
    # Only the volume-sized FIRST allocation may retry on CPU: once patches are blended into a
    # GPU-resident buffer, a CPU retry would mix devices inside one accumulator: re-raise instead.
    class DummyGroupTransform:
        patch_transforms: ClassVar[list[object]] = []

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": DummyGroupTransform()}}

    class MidBlendOOMAccumulator:
        @staticmethod
        def is_empty() -> bool:
            return False  # a previous patch is already blended in

        @staticmethod
        def add_layer(index: int, layer: torch.Tensor) -> None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    class FakeCudaTensor:
        device = torch.device("cuda:0")

        @staticmethod
        def numel() -> int:
            return 4

        @staticmethod
        def element_size() -> int:
            return 4

    output_dataset = OutputDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    output_dataset.output_layer_accumulator[0] = {0: cast(Any, MidBlendOOMAccumulator())}
    output_dataset.attributes[0] = {0: {0: Attribute()}}
    output_dataset.names[0] = "CASE_000"
    output_dataset._accum_device[0] = torch.device("cuda", 0)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        output_dataset.add_layer(
            index_dataset=0,
            index_augmentation=0,
            index_patch=1,
            layer=cast(torch.Tensor, FakeCudaTensor()),
            dataset=cast(DatasetIter, DummyDatasetIter()),
            attribute=Attribute(),
        )
    assert output_dataset._accum_device[0].type == "cuda"


# ---------------------------------------------------------------------------
# Device co-location of modules a per-model load() adds after placement
# ---------------------------------------------------------------------------
class _LateHeadNetwork(Network):
    """Model whose load() appends a checkpoint-sized module, mimicking TotalSegmentator's Head.Conv."""

    def __init__(self) -> None:
        super().__init__(in_channels=1)
        self.add_module("Stem", torch.nn.Conv3d(1, 2, kernel_size=1))

    def load(self, state_dict, init: bool = True, ema: bool = False):  # type: ignore[override]
        # A head sized from the checkpoint, created at load time -> defaults to CPU.
        self.add_module("Head", torch.nn.Conv3d(2, int(state_dict["nb_class"]), kernel_size=1))

    def forward(self, batch_sample, output_layers=[]):  # type: ignore[override]
        return []


def test_colocate_is_a_safe_noop_when_model_is_all_cpu() -> None:
    # With no device-placed parameter there is nothing to co-locate; the helper must be a no-op.
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1), torch.nn.Conv3d(2, 3, 1))
    _colocate_loaded_modules(model)
    assert all(not p.is_cuda for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device co-location only manifests on GPU")
def test_ensemble_load_colocates_late_added_head_on_gpu() -> None:
    # The TotalSegmentator pattern: the model is placed on the GPU, then a per-model load() appends
    # a Head on CPU. The forward must not hit "Input cuda, weight CPU".
    composite = ModelComposite(_LateHeadNetwork(), Mean())
    Network.to(composite, 0)  # place on cuda:0, exactly as the predictor does before inference

    composite.load([{"nb_class": 5}])  # single source -> triggers _ensure_model_loaded(0)
    model = composite["Model_0"]

    head = dict(model.named_modules())["Head"]
    assert list(head.parameters()), "test setup: Head should have parameters"
    assert all(p.is_cuda for p in head.parameters()), "load-added Head must be co-located onto the GPU"
    # the whole model must live on a single device
    assert len({p.device for p in model.parameters()}) == 1


# ---------------------------------------------------------------------------
# CPU offload of patch predictions (pinned staging buffer)
# ---------------------------------------------------------------------------
def _dataset(monkeypatch: pytest.MonkeyPatch) -> OutputDataset:
    monkeypatch.setenv("KONFAI_config_file", "unused.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    return OutputDataset(same_as_group="default:default", dataset_filename="default|./Dataset:mha")


def test_offload_cpu_tensor_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(monkeypatch)
    x = torch.randn(3, 4, 4)
    out = ds._offload_to_cpu(x)
    assert torch.equal(out, x.detach().cpu())
    assert ds._pin_buffer is None  # no page-locked buffer allocated for a CPU patch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging only applies to CUDA patches")
def test_offload_large_cuda_patch_is_bit_identical_and_pageable(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(monkeypatch)
    # above _PINNED_OFFLOAD_MIN_BYTES (64 MiB): 40M fp16 elements = 80 MiB
    x = torch.empty(40 * 1024 * 1024, dtype=torch.float16, device="cuda").normal_()

    out = ds._offload_to_cpu(x)

    assert torch.equal(out, x.detach().cpu())  # staging through pinned memory changes nothing numerically
    assert out.device.type == "cpu"
    assert not out.is_pinned()  # the stored patch must be pageable, not page-locked
    assert ds._pin_buffer is not None and ds._pin_buffer.is_pinned()
    # the one-patch pinned buffer is reused, not reallocated, for the next same-shape patch
    reused = ds._pin_buffer
    ds._offload_to_cpu(x)
    assert ds._pin_buffer is reused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging only applies to CUDA patches")
def test_offload_small_cuda_patch_takes_plain_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(monkeypatch)
    x = torch.randn(8, 8, 8, device="cuda")  # well under the staging threshold
    out = ds._offload_to_cpu(x)
    assert torch.equal(out, x.detach().cpu())
    assert ds._pin_buffer is None  # small patches never allocate the pinned buffer


def test_a_grid_swept_off_axis_zero_takes_the_whole_volume_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The streamed consumers index their slabs on the first spatial axis; a grid ordered along another
    axis must not stream, or one axis's slab lands in another. Whole-volume is the safe path until the
    consumers carry the axis."""

    class DummyPatch:
        patch_size: ClassVar[list[int]] = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(0, 2), slice(0, 2))]

        @staticmethod
        def get_sweep_axis(index_augmentation: int) -> int:
            del index_augmentation
            return 1

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        cache_attributes: ClassVar[list[Attribute]] = [Attribute({"Origin": [0.0, 0.0]})]

    class DummyGroupTransform:
        patch_transforms: ClassVar[list[object]] = []

    class DummyDatasetIter:
        groups_src: ClassVar[dict[str, dict[str, object]]] = {"src": {"dest": DummyGroupTransform()}}

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            return DummyManager()

    output_dataset = OutputDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    output_dataset._streaming_enabled = True
    # A plan would exist if streaming were allowed to consider this case at all.
    monkeypatch.setattr(OutputDataset, "_plan_stream", lambda self, *a, **k: object(), raising=True)

    output_dataset.add_layer(
        index_dataset=0,
        index_augmentation=0,
        index_patch=0,
        layer=torch.zeros(1, 2, 2),
        dataset=cast(DatasetIter, DummyDatasetIter()),
        attribute=Attribute({"Spacing": np.asarray([1.0, 1.0])}),
    )

    assert output_dataset._stream_plans[0] is None, "an off-axis sweep must fall back to whole-volume"
    from konfai.data.patching import StreamingAccumulator

    assert not isinstance(output_dataset.output_layer_accumulator[0][0], StreamingAccumulator)
