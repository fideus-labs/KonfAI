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
from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import Flip
from konfai.data.patching import Accumulator, blend_axes
from konfai.data.reduction import Reduction
from konfai.network.blocks import ArgMax
from konfai.network.network import ModuleArgsDict, Network
from konfai.predictor import PREDICTION_CLOCK, OutputDataset
from konfai.predictor.loop import _Predictor
from konfai.predictor.output import _AsyncWriter
from konfai.predictor.workflow import Predictor, build_predict
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PredictorError
from konfai.utils.utils import get_patch_slices_from_shape
from ruamel.yaml import YAML


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
    """``reduction: Mean`` binds from the ``Mean:`` block a resolved config carries, as a custom
    operator does, and the write-back records that block."""
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


def test_the_old_sink_name_is_refused_naming_the_output_dataset() -> None:
    """``OutSameAsGroupDataset``, the sink's old name, is refused, and the refusal names the class to
    write instead; ``OutputDataset`` resolves."""
    from konfai.predictor import OutputDataset
    from konfai.utils.errors import ConfigError
    from konfai.utils.utils import get_module, module_attribute

    module, name = get_module("OutSameAsGroupDataset", "konfai.predictor")
    with pytest.raises(ConfigError, match="closest: 'OutputDataset'"):
        module_attribute(module, name)
    assert module_attribute(*get_module("OutputDataset", "konfai.predictor")) is OutputDataset


def test_two_rank_prediction_refuses_a_single_file_output(tmp_path: Path) -> None:
    """Two ranks writing one h5 lost cases with exit 0: its lock serializes the threads of one process
    only. Refused before the first write, as TRANSFORM refuses it."""
    predictor = Predictor.__new__(Predictor)
    predictor.gpu_checkpoints = None
    predictor.outputs_dataset = {"Head": Dataset(tmp_path / "Prediction", "h5")}

    with pytest.raises(PredictorError, match="single-file store"):
        predictor.setup(2)
    assert not any(tmp_path.iterdir())


class TTANet(Network):
    """The smallest network a Predictor builds: the TTA tests below read the draws, not its output."""

    def __init__(self) -> None:
        super().__init__(in_channels=1, dim=2)
        self.add_module("Conv", torch.nn.Conv2d(1, 1, 1))


def _tta_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subset: list[str] | None, manual_seed: int | None = None
) -> dict[str, list[list[int]]]:
    """Build the Predictor of a flip TTA over ``subset`` of a three-case cohort, from a global RNG in
    one fixed state, and answer each case's draw (the axes each of its eight copies flips) by name."""
    for key in ("KONFAI_config_file", "KONFAI_ROOT", "KONFAI_STATE", "KONFAI_CONFIG_MODE"):
        monkeypatch.setenv(key, "")
    monkeypatch.chdir(tmp_path)
    source = Dataset(str(tmp_path / "Dataset"), "mha")
    for name in ("CASE_000", "CASE_001", "CASE_002"):
        source.write("CT", name, np.ones((1, 2, 4, 4), dtype=np.float32), Attribute())
    predictor_tree: dict[str, object] = {
        "Model": {"classpath": "test_predictor:TTANet"},
        "Dataset": {
            "dataset_filenames": ["./Dataset:a:mha"],
            "groups_src": {"CT": {"groups_dest": {"CT": {"is_input": True}}}},
            "augmentations": {
                "DataAugmentation_0": {"nb": 8, "data_augmentations": {"Flip": {"f_prob": [0.5] * 3, "prob": 1}}}
            },
            "Patch": {"patch_size": [1, 4, 4], "overlap": 0},
            "subset": subset if subset is not None else "None",
            "num_workers": 0,
        },
        "outputs_dataset": {
            "Conv": {"OutputDataset": {"same_as_group": "CT:CT", "group": "OUT", "dataset_filename": "Out:mha"}}
        },
    }
    if manual_seed is not None:
        predictor_tree["manual_seed"] = manual_seed
    config = tmp_path / "Prediction.yml"
    YAML().dump({"Predictor": predictor_tree}, config)
    torch.manual_seed(0)
    predictor = cast(Predictor, build_predict([tmp_path / "fold.pt"], config, tmp_path / "Predictions"))
    flip = predictor.dataset.data_augmentations_list["DataAugmentation_0"].data_augmentations[0]
    assert isinstance(flip, Flip)
    return {manager.name: flip.flip[manager.index] for manager in next(iter(predictor.dataset._managers.values()))}


def test_a_case_s_tta_draw_does_not_depend_on_the_cases_predicted_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draws were taken from the global RNG in case order, so CASE_001 predicted alone was handed
    the copies CASE_000 gets in the cohort, and its prediction changed with the subset."""
    alone = _tta_draws(tmp_path, monkeypatch, ["CASE_001"])
    cohort = _tta_draws(tmp_path, monkeypatch, None)
    assert alone["CASE_001"] == cohort["CASE_001"]
    assert cohort["CASE_000"] != cohort["CASE_001"], "two cases are handed two draws"


def test_an_output_key_that_names_no_module_is_refused_under_its_own_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal named `outputs_criterions`, a training key, for a key of `outputs_dataset`."""
    for key in ("KONFAI_config_file", "KONFAI_ROOT", "KONFAI_STATE", "KONFAI_CONFIG_MODE"):
        monkeypatch.setenv(key, "")
    monkeypatch.chdir(tmp_path)
    Dataset(str(tmp_path / "Dataset"), "mha").write("CT", "CASE_000", np.ones((1, 2, 4, 4), np.float32), Attribute())
    tree = {
        "Model": {"classpath": "test_predictor:TTANet"},
        "Dataset": {
            "dataset_filenames": ["./Dataset:a:mha"],
            "groups_src": {"CT": {"groups_dest": {"CT": {"is_input": True}}}},
            "Patch": {"patch_size": [1, 4, 4], "overlap": 0},
            "num_workers": 0,
        },
        "outputs_dataset": {
            "Head": {"OutputDataset": {"same_as_group": "CT:CT", "group": "OUT", "dataset_filename": "Out:mha"}}
        },
    }
    YAML().dump({"Predictor": tree}, tmp_path / "Prediction.yml")
    with pytest.raises(PredictorError, match=r"(?s)'Head' under 'outputs_dataset'.*\['Conv'\]"):
        build_predict([tmp_path / "fold.pt"], tmp_path / "Prediction.yml", tmp_path / "Predictions")


_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "Workflows"
_CASES = ("CASE_000", "CASE_001")


def _tiny_synthesis(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Two MR cases and the TinySynth prediction config that maps them to ``sCT``."""
    import numpy as np
    from konfai.utils.dataset import Attribute

    monkeypatch.syspath_prepend(str(_ASSETS))  # where 'TinySynth:TinySynthNet' resolves
    attributes = Attribute()
    attributes["Origin"] = np.zeros(3)
    attributes["Spacing"] = np.ones(3)
    attributes["Direction"] = np.eye(3).reshape(-1)
    volume = np.linspace(-1.0, 1.0, 2 * 16 * 16, dtype=np.float32).reshape(1, 2, 16, 16)
    for name in _CASES:
        Dataset(root / "Dataset", "mha").write("MR", name, volume, attributes)
    return {
        "Predictor": {
            "Model": {"classpath": "TinySynth:TinySynthNet", "TinySynthNet": {"outputs_criterions": "None"}},
            "Dataset": {
                "groups_src": {"MR": {"groups_dest": {"MR": {"is_input": True}}}},
                "Patch": {"patch_size": [1, 16, 16], "overlap": "None"},
                "dataset_filenames": [f"{root / 'Dataset'}:a:mha"],
                "batch_size": 4,
                "num_workers": 0,
            },
            "outputs_dataset": {
                "Head:Tanh": {
                    "OutputDataset": {"dataset_filename": "Dataset:mha", "group": "sCT", "same_as_group": "MR:MR"}
                }
            },
            "train_name": "RERUN",
        }
    }


def _tiny_checkpoint(path: Path, weight: float) -> Path:
    torch.save(
        {"Model": {"TinySynthNet": {"Projection.weight": torch.tensor([weight]), "Projection.bias": torch.zeros(1)}}},
        path,
    )
    return path


def _predicted(root: Path, name: str):
    return Dataset(root / "Predictions" / "RERUN" / "Dataset", "mha").read_data("sCT", name)[0]


def test_a_rerun_with_the_same_checkpoint_still_resumes(tmp_path: Path, monkeypatch) -> None:
    """The per-case resume itself: a case whose outputs the same checkpoint wrote is skipped, a missing
    one is computed, and so after an overwrite run that replaced another checkpoint's outputs."""
    import numpy as np
    from konfai import api

    config = _tiny_synthesis(tmp_path, monkeypatch)
    first, other = _tiny_checkpoint(tmp_path / "a.pt", 1.0), _tiny_checkpoint(tmp_path / "b.pt", 0.5)
    output = Dataset(tmp_path / "Predictions" / "RERUN" / "Dataset", "mha")

    def predict(model: Path, overwrite: bool = False) -> None:
        api.predict(model, config, cpu=1, quiet=True, overwrite=overwrite, predictions_dir=tmp_path / "Predictions")

    for model, overwrite in ((first, False), (other, True)):
        predict(model, overwrite)
        expected = _predicted(tmp_path, _CASES[1])
        # A sentinel where a skipped case keeps its file, and a case whose output is gone.
        sentinel, attributes = output.read_data("sCT", _CASES[0])
        output.write("sCT", _CASES[0], np.full_like(sentinel, 7.0), attributes)
        (tmp_path / "Predictions" / "RERUN" / "Dataset" / _CASES[1] / "sCT.mha").unlink()

        predict(model)

        np.testing.assert_array_equal(_predicted(tmp_path, _CASES[0]), np.full_like(sentinel, 7.0))
        np.testing.assert_array_equal(_predicted(tmp_path, _CASES[1]), expected)


class _ArgmaxHead(ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Argmax", ArgMax(dim=1))


class ArgmaxSegNet(Network):
    """Two classes and a label-map head, the way the Segmentation example's UNet ends."""

    def __init__(self) -> None:
        super().__init__(in_channels=1, dim=2)
        self.add_module("Logits", torch.nn.Conv2d(1, 2, 1))
        self.add_module("Head", _ArgmaxHead())


def _label_map_prediction(
    root: Path, monkeypatch: pytest.MonkeyPatch, members: int, net: type[Network] = ArgmaxSegNet
) -> list[Path]:
    """The TinySynth cohort predicted by ``net`` (its last module's output), one checkpoint per ensemble member."""
    config = _tiny_synthesis(root, monkeypatch)["Predictor"]
    config["Model"] = {"classpath": f"test_predictor:{net.__name__}"}
    output = "Head:Argmax" if net is ArgmaxSegNet else "Complex"
    config["outputs_dataset"] = {
        output: {"OutputDataset": {"dataset_filename": "Dataset:mha", "group": "SEG", "same_as_group": "MR:MR"}}
    }
    config["combine"] = "Mean"
    checkpoints = []
    for index in range(members):
        torch.manual_seed(index)
        checkpoints.append(root / f"member_{index}.pt")
        torch.save({"Model": net().network_states()}, checkpoints[-1])
    from konfai import api

    api.predict(checkpoints, {"Predictor": config}, cpu=1, quiet=True, predictions_dir=root / "Predictions")
    return sorted((root / "Predictions" / "RERUN" / "Dataset").rglob("SEG.mha"))


def test_a_mean_ensemble_of_label_maps_is_refused(tmp_path: Path, monkeypatch) -> None:
    """``combine: Mean`` summed the members' Argmax labels in the first one's int64 tensor, then died
    dividing it (``result type Float can't be cast to the desired output type Long``); a mean of class
    indices would be a third class anyway."""
    with pytest.raises(PredictorError, match="final_transforms"):
        _label_map_prediction(tmp_path, monkeypatch, members=2)


class Maximum(Reduction):
    """A fold of one's own, the extension point ``custom-models.md`` documents."""

    voxel_local = True

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors).amax(dim=0)


def _tta_prediction_config(dataset: Path) -> dict:
    """TinySynth through a Gamma test-time augmentation of two copies, averaged, under ``manual_seed``."""
    return {
        "Predictor": {
            "Model": {"classpath": "TinySynth:TinySynthNet", "TinySynthNet": {"outputs_criterions": "None"}},
            "Dataset": {
                "groups_src": {
                    "MR": {"groups_dest": {"MR": {"transforms": "None", "patch_transforms": "None", "is_input": True}}}
                },
                "augmentations": {
                    "DataAugmentation_0": {
                        "nb": 2,
                        "data_augmentations": {
                            "Gamma": {"gamma_min": 0.5, "gamma_max": 2.0, "groups": ["MR"], "prob": 1}
                        },
                    }
                },
                "Patch": {"patch_size": [1, 8, 8], "overlap": "None", "pad_value": 0, "extend_slice": 0},
                "subset": "None",
                "dataset_filenames": [f"{dataset}:a:mha"],
                "batch_size": 4,
            },
            "outputs_dataset": {
                "Head:Tanh": {
                    "OutputDataset": {
                        "name_class": "OutputDataset",
                        "before_reduction_transforms": "None",
                        "after_reduction_transforms": "None",
                        "final_transforms": "None",
                        "dataset_filename": "Dataset:mha",
                        "group": "OUT",
                        "same_as_group": "MR:MR",
                        "reduction": "Mean",
                        "Mean": {},
                    }
                }
            },
            "train_name": "TTA",
            "manual_seed": 0,
            "gpu_checkpoints": "None",
            "autocast": False,
            "combine": "Mean",
            "data_log": "None",
        }
    }


def test_a_seeded_prediction_replays_its_test_time_augmentation(tmp_path: Path, monkeypatch) -> None:
    """``manual_seed`` replays a random TTA bit for bit. The copies are drawn while the workflow is
    built, before the run seeds anything, so the draw must come from the seed and not from the state
    the process started in (a fresh interpreter seeds torch at random)."""
    import numpy as np
    from konfai import api
    from konfai.utils.runtime.distributed import preserved_rng, seed_all

    sitk = pytest.importorskip("SimpleITK")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "assets" / "Workflows"))
    dataset = tmp_path / "Dataset"
    ramp = np.linspace(0.0, 1.0, 2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    for case in ("P000", "P001"):
        (dataset / case).mkdir(parents=True)
        sitk.WriteImage(sitk.GetImageFromArray(ramp), str(dataset / case / "MR.mha"))
    checkpoint = tmp_path / "m.pt"
    torch.save(
        {"Model": {"TinySynthNet": {"Projection.weight": torch.ones(1), "Projection.bias": torch.zeros(1)}}}, checkpoint
    )

    outputs = []
    for start in (1, 2):
        with preserved_rng():
            seed_all(start)  # two processes, two generator states
            workspace = api.predict(
                models=[checkpoint],
                config=_tta_prediction_config(dataset),
                cpu=1,
                quiet=True,
                overwrite=True,
                predictions_dir=tmp_path / f"Predictions_{start}",
            )
        outputs.append(sitk.GetArrayFromImage(sitk.ReadImage(str(workspace / "Dataset" / "P000" / "OUT.mha"))))
    assert np.array_equal(outputs[0], outputs[1])


def test_a_checkpoint_path_that_does_not_exist_is_refused_before_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--models nope.pt`` listed every case, printed 'Running on CPU' and then raised a bare
    ValueError from setup; the config it read was rewritten on the way."""
    from konfai.predictor import predict

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "Prediction.yml"
    YAML().dump(_tiny_synthesis(tmp_path, monkeypatch), config)
    written = config.read_bytes()

    with pytest.raises(PredictorError) as refused:
        predict(models=["nope.pt"], quiet=True, prediction_file=config, predictions_dir=tmp_path / "Predictions")

    message = str(refused.value)
    assert "[Predictor] Checkpoint 'nope.pt' does not exist" in message and str(tmp_path / "nope.pt") in message
    assert config.read_bytes() == written
    assert not (tmp_path / "Predictions").exists()


def test_a_models_path_that_is_no_checkpoint_is_refused_by_name(tmp_path: Path) -> None:
    """A missing path and a directory each read as one [Predictor] line naming the path; a directory
    lists the checkpoints it holds."""
    run = tmp_path / "Checkpoints" / "RUN"
    run.mkdir(parents=True)
    (run / "2026_01_01.pt").touch()
    (run / "resume_latest.pt").touch()
    predictor = Predictor.__new__(Predictor)

    predictor.set_models([str(tmp_path / "Checkpoints" / "NOPE.pt")])
    with pytest.raises(PredictorError, match=r"NOPE\.pt' does not exist"):
        predictor._load()

    predictor.set_models([str(run)])
    with pytest.raises(PredictorError, match="is a directory") as refused:
        predictor._load()
    assert "2026_01_01.pt, resume_latest.pt" in str(refused.value)


class _WidthNet(Network):
    """The same class at two widths: a checkpoint of one does not fit the other."""

    def __init__(self, width: int) -> None:
        super().__init__(in_channels=1, dim=2)
        self.add_module("Conv", torch.nn.Conv2d(1, width, 1))


def test_a_checkpoint_of_another_architecture_is_refused_by_name() -> None:
    from konfai.data.reduction import Mean
    from konfai.predictor import ModelComposite

    composite = ModelComposite(_WidthNet(2), Mean())
    with pytest.raises(PredictorError, match="size mismatch") as refused:
        composite.load([{"Model": _WidthNet(1).network_states()}])
    assert "Predictor.Model._WidthNet" in str(refused.value)


@pytest.mark.parametrize("same_as_group", ["CT", "a:b:c"])
def test_same_as_group_names_a_source_and_a_destination_group(same_as_group: str) -> None:
    """``same_as_group`` is ``<group_src>:<group_dest>``: the default was ``default`` and failed on an
    unpacking ValueError; a value of another form is refused by name."""
    from konfai.utils.errors import PredictorError

    with pytest.raises(PredictorError, match="<group_src>:<group_dest>"):
        OutputDataset(same_as_group=same_as_group, dataset_filename="./Out:mha")
    assert OutputDataset(dataset_filename="./Out:mha").group_src == "default"
