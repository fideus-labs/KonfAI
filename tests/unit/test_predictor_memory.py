from typing import Any, ClassVar, cast

import numpy as np
import pytest
import torch
from konfai.data.data_manager import BatchDataItem, DatasetIter
from konfai.network.network import Network
from konfai.predictor import Mean, ModelComposite, OutSameAsGroupDataset, _Predictor
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


def test_output_dataset_uses_batch_attributes_when_manager_cache_is_cold() -> None:
    class DummyPatch:
        patch_size: ClassVar[list[int]] = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(0, 2), slice(0, 2))]

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

    output_dataset = OutSameAsGroupDataset(
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

    output_dataset = OutSameAsGroupDataset(
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

    # The accumulator now blends each patch straight into a running CPU buffer (no per-patch list), so
    # the offloaded patch must have been moved to CPU exactly once before being blended.
    accumulator = output_dataset.output_layer_accumulator[0][0]
    assert fake_layer.cpu_calls == 1
    assert accumulator._result is not None
    assert accumulator._result.device.type == "cpu"


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
        "konfai.predictor.DistributedObject.get_measure",
        staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected sync"))),
    )

    predictor._predict_log({})


def test_predictor_runs_prediction_logging_once_per_batch_even_with_multiple_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    loader = DummyPredictionLoader([batch], dataset)
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

    log_calls: list[int] = []
    monkeypatch.setattr(predictor, "_predict_log", lambda batch_sample: log_calls.append(len(batch_sample)))
    monkeypatch.setattr("konfai.predictor.description", lambda model: "stub")

    predictor.run()

    assert log_calls == [1]
    assert dataset.labels == ["Prediction"]
    assert model_composite.eval_calls == 1
    assert outputs_dataset["out_a"].writes == 1
    assert outputs_dataset["out_b"].writes == 1


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

    output_dataset = OutSameAsGroupDataset(
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
    # GPU-resident buffer, a CPU retry would mix devices inside one accumulator — re-raise instead.
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

    output_dataset = OutSameAsGroupDataset(
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
