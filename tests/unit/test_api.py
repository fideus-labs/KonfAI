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

"""The Python front door (:mod:`konfai.api`): live objects and the YAML file are two spellings of
one engine. Pins the kwargs recording, the object->tree serialization, the run contract (raise, not
exit; the process env left as found; one workflow at a time), and byte-identity between the two
spellings of the same run."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai import api  # noqa: E402
from konfai.data.reduction import Std  # noqa: E402
from konfai.data.transform import Clip, Magnitude, Resample, Save, Write  # noqa: E402
from konfai.metric.measure import MAE, Dice  # noqa: E402
from konfai.utils.errors import ConfigError, KonfAIError  # noqa: E402

# --------------------------------------------------------------------------- recording and trees


def test_a_stage_records_the_arguments_as_given() -> None:
    stage = Clip(min_value=-100.0, max_value=300.0)
    assert stage._konfai_given == {"min_value": -100.0, "max_value": 300.0}


def test_a_criterion_records_too() -> None:
    assert "labels" in Dice(labels=[1, 2])._konfai_given


def test_a_subclass_with_no_init_of_its_own_records_the_inherited_one() -> None:
    """Accuracy inherits Criterion's constructor whole: the recording must come with it."""
    from konfai.metric.measure import Accuracy

    assert Accuracy()._konfai_given == {}


def test_a_repeated_mapping_stage_is_qualified_by_resolution() -> None:
    """The second occurrence of a bare mapping name gets the module the binder would resolve."""
    tree = api._chain_tree(
        [{"Clip": {"min_value": 0.0}}, {"Clip": {"max_value": 1.0}}],
        api._STAGE_MODULES,
        "chains.CT.CT",
    )
    assert list(tree) == ["Clip", "konfai.data.transform:Clip"]


def test_a_numpy_scalar_is_spelled_as_a_plain_scalar() -> None:
    """np.float64 IS a float subclass; unspelled, ruamel refuses it at dump time."""
    spelled = api._yaml_safe(np.float64(1.5), "chains.CT.CT.Clip.min_value")
    assert type(spelled) is float and spelled == 1.5


def test_a_config_file_is_copied_not_rewritten(tmp_path: Path) -> None:
    """Reading a config rewrites it; a caller's file is not this call's to rewrite."""
    source = tmp_path / "Prediction.yml"
    source.write_text("Predictor: {}\n", encoding="utf-8")
    copy = api._config_copy(source)
    assert copy != source
    assert Path(copy).read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_a_subclass_delegating_to_super_keeps_its_own_spelling() -> None:
    """The recorded spelling is the caller's: a subclass expanding into ``Resample`` arguments
    inside ``super().__init__`` records its OWN kwargs, so the tree references the subclass with
    what the caller wrote and rebinds identically."""

    class FieldBeside(Resample):
        def __init__(self, group: str) -> None:
            super().__init__(field_group=group)

    assert FieldBeside(group="DVF")._konfai_given == {"group": "DVF"}


def test_the_chain_tree_is_the_yaml_subtree() -> None:
    tree = api._chain_tree([Clip(min_value=0.0), Write(dataset="./Out:mha")], api._STAGE_MODULES, "chains.CT.CT")
    assert tree == {"Clip": {"min_value": 0.0}, "Write": {"dataset": "./Out:mha"}}


def test_an_unrecordable_stage_is_refused_by_name() -> None:
    class VarArgs(Clip):
        def __init__(self, *bounds: float) -> None:
            super().__init__(min_value=min(bounds))

    with pytest.raises(ConfigError, match="VarArgs"):
        api._chain_tree([VarArgs(1.0, 2.0)], api._STAGE_MODULES, "chains.CT.CT")


def test_a_non_spellable_argument_is_refused_by_name() -> None:
    with pytest.raises(ConfigError, match="max_value"):
        api._chain_tree([Clip(max_value=np.ones(2))], api._STAGE_MODULES, "chains.CT.CT")


# ------------------------------------------------------------------------------------ run contract


def _write_case(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(values), str(path))


@pytest.fixture()
def cohort(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    for case in ("P000", "P001"):
        _write_case(tmp_path / "Raw" / case / "CT.mha", rng.normal(0.0, 200.0, (6, 7, 8)).astype(np.float32))
    return tmp_path


def test_objects_and_yaml_are_two_spellings_of_one_run(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(cohort)
    result = api.transform(
        "BY_OBJECTS",
        "./Raw:mha",
        {"CT": {"CT": [Clip(min_value=-50.0, max_value=100.0), Write(dataset="./OutA:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    api.transform(
        "BY_TREE",
        "./Raw:mha",
        {"CT": {"CT": {"Clip": {"min_value": -50.0, "max_value": 100.0}, "Write": {"dataset": "./OutB:mha"}}}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    for case in ("P000", "P001"):
        by_objects = (cohort / "OutA" / case / "CT.mha").read_bytes()
        by_tree = (cohort / "OutB" / case / "CT.mha").read_bytes()
        assert by_objects == by_tree
    assert result.workspace == cohort / "Transforms" / "BY_OBJECTS"
    assert result.outputs[0]["dataset"] == str(cohort / "OutA")
    assert result.outputs[0]["path"] == str(cohort / "OutA")
    assert result.config.is_file()


def test_a_designed_refusal_raises_instead_of_exiting(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(cohort)
    with pytest.raises(KonfAIError):
        api.transform(
            "NO_WRITE",
            "./Raw:mha",
            {"CT": {"CT": [Clip(min_value=0.0)]}},
            transforms_dir=cohort / "Transforms",
            quiet=True,
        )


def test_the_environment_is_left_as_found(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(cohort)
    for key in [key for key in os.environ if key.startswith("KONFAI")]:
        monkeypatch.delenv(key)
    api.transform(
        "ENV",
        "./Raw:mha",
        {"CT": {"CT": [Write(dataset="./OutEnv:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    assert [key for key in os.environ if key.startswith("KONFAI")] == []


def test_the_output_is_not_left_open_in_the_callers_process(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's pooled h5 read handles are released when the call returns: HDF5 refuses to open for
    writing a file this process still holds for reading, so a notebook could not append to the
    output it just produced (nor to the source it just read)."""
    h5py = pytest.importorskip("h5py")
    monkeypatch.chdir(cohort)
    result = api.transform(
        "H5",
        "./Raw:mha",
        {"CT": {"CT": [Save(dataset="./Cache:h5"), Clip(min_value=0.0), Write(dataset="./OutH5:h5")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    for name in ("Cache.h5", "OutH5.h5"):
        with h5py.File(cohort / name, "a") as handle:
            assert "CT" in handle
    assert result.outputs[0]["dataset"] == str(cohort / "OutH5") and result.outputs[0]["path"] == str(
        cohort / "OutH5.h5"
    )


def test_one_workflow_at_a_time_per_process(cohort: Path) -> None:
    assert api._ACTIVE.acquire(blocking=False)
    try:
        with pytest.raises(ConfigError, match="already running"):
            api.transform("BUSY", "./Raw:mha", {"CT": {"CT": [Write(dataset="./Out:mha")]}})
    finally:
        api._ACTIVE.release()


def test_the_reference_follows_the_case(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``reference: '{case}'`` adopts, per case, the grid of that case's own entry: two cases whose
    Ref grids differ land each on their own, not both on a memoized first."""
    monkeypatch.chdir(cohort)
    grids = {"P000": ((1.0, 1.2, 0.8), (5.0, -3.0, 2.0)), "P001": ((2.0, 0.7, 1.1), (-8.0, 4.0, 0.5))}
    for case, (spacing, origin) in grids.items():
        reference = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.float32))
        reference.SetSpacing(spacing)
        reference.SetOrigin(origin)
        sitk.WriteImage(reference, str(cohort / "Raw" / case / "Ref.mha"))
    api.transform(
        "PER_CASE",
        "./Raw:mha",
        {"CT": {"Moved": [Resample(reference="{case}", reference_group="Ref"), Write(dataset="./Moved:mha")]}},
        transforms_dir=cohort / "Transforms",
        on_fallback="error",
        quiet=True,
    )
    for case, (spacing, origin) in grids.items():
        moved = sitk.ReadImage(str(cohort / "Moved" / case / "Moved.mha"))
        assert moved.GetSpacing() == pytest.approx(spacing)
        assert moved.GetOrigin() == pytest.approx(origin)
        assert moved.GetSize() == (7, 6, 5)


def test_evaluate_scores_the_stored_values_when_no_chain_is_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent ``transforms`` key is not an absent chain: the binder materializes its own default
    (``Normalize``), and two groups each rescaled to [-1, 1] by their own extrema no longer differ
    where they did. On a pair related by ``0.9 x + 0.05`` that reported MAE 1.9e-8 for 0.025."""
    monkeypatch.chdir(tmp_path)
    truth = np.linspace(0.0, 1.0, 6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    prediction = (0.9 * truth + 0.05).astype(np.float32)
    _write_case(tmp_path / "Raw" / "P000" / "CT.mha", truth)
    _write_case(tmp_path / "Raw" / "P000" / "sCT.mha", prediction)

    result = api.evaluate(
        "STORED_VALUES",
        "./Raw:mha",
        {"sCT": {"CT": [MAE()]}},
        evaluations_dir=tmp_path / "Evaluations",
        quiet=True,
        overwrite=True,
    )

    assert result.metrics["TRAIN"]["case"]["sCT:CT:MAE"]["P000"] == pytest.approx(
        float(np.abs(prediction - truth).mean()), rel=1e-5
    )


# --------------------------------------------------------------------------- uncertainty vocabulary


def test_std_reduction_matches_torch_incrementally() -> None:
    rng = np.random.default_rng(3)
    members = [torch.from_numpy(rng.normal(size=(1, 4, 5, 6)).astype(np.float32)) for _ in range(5)]
    expected = torch.stack(members).std(0)

    torch.testing.assert_close(Std()(list(members)), expected)

    incremental = Std()
    incremental.start()
    for member in members:
        incremental.accumulate(member)
    torch.testing.assert_close(incremental.finalize(), expected)


def test_std_of_a_single_case_is_zero() -> None:
    member = torch.ones(1, 2, 3)
    assert Std()([member]).abs().max() == 0.0


def test_magnitude_is_the_channel_norm_and_pointwise() -> None:
    from konfai.data.transform import LocalityKind
    from konfai.utils.dataset import Attribute

    field = torch.tensor([[[3.0]], [[4.0]]])
    stage = Magnitude()
    torch.testing.assert_close(stage("case", field, Attribute()), torch.tensor([[[5.0]]]))
    assert stage.patch_locality(Attribute()).kind is LocalityKind.POINTWISE


# ------------------------------------------------------------------------------------- config tree


def test_a_config_tree_must_hold_the_workflow_root() -> None:
    from konfai.utils.runtime.environment import _materialized_config

    with pytest.raises(ConfigError, match="Transformer"):
        _materialized_config({"Trainer": {}}, "Transformer")
    path = _materialized_config({"Transformer": {"name": "X"}}, "Transformer")
    assert path.is_file()


# ---------------------------------------------------------------------------- component discovery


def test_list_components_names_the_config_vocabulary() -> None:
    """The catalog answers with the exact spelling a YAML config references each component by."""
    transforms = {component.name: component for component in api.list_components("transforms")}
    assert transforms["Resample"].config_reference == "Resample" and transforms["Resample"].doc

    assert {"Dice", "MAE"} <= {component.name for component in api.list_components("criteria")}
    assert "Median" in {component.name for component in api.list_components("reductions")}
    assert "Flip" in {component.name for component in api.list_components("augmentations")}
    assert "Conv" in {component.name for component in api.list_components("blocks")}

    models = {component.config_reference for component in api.list_components("models")}
    assert "default|UNet.yml" in models  # the declarative catalog
    assert "segmentation.UNet.UNet" in models  # the Python catalog, in Model.classpath spelling


def test_list_components_refuses_an_unknown_kind() -> None:
    with pytest.raises(ConfigError, match="component kind"):
        api.list_components("optimizers")


# ------------------------------------------------------------- a model YAML named by a relative path


def test_a_relative_model_yaml_is_anchored_to_the_config_files_directory(tmp_path: Path) -> None:
    """The copy lives in a scratch directory, and a relative model YAML resolves next to the
    config file that names it: anchored here, or the shipped examples' ``classpath: UNet.yml``
    would be looked for in the scratch directory."""
    (tmp_path / "UNet.yml").write_text("name: UNet\n", encoding="utf-8")
    source = tmp_path / "Config.yml"
    source.write_text("Trainer:\n  Model:\n    classpath: UNet.yml  # the shipped spelling\n", encoding="utf-8")
    copy = api._config_copy(source)
    assert f"classpath: {tmp_path.resolve() / 'UNet.yml'}" in Path(copy).read_text(encoding="utf-8")
    assert "Trainer:\n  Model:\n    classpath: UNet.yml" in source.read_text(encoding="utf-8")  # untouched


def test_a_tree_anchors_a_relative_model_yaml_to_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tree = api._config_copy({"Predictor": {"Model": {"classpath": "nets/UNet.yml"}}})
    assert tree["Predictor"]["Model"]["classpath"] == str(tmp_path.resolve() / "nets" / "UNet.yml")


def test_catalog_absolute_and_class_spellings_are_left_alone(tmp_path: Path) -> None:
    tree = {
        "Trainer": {
            "Model": {"classpath": "default|UNet.yml"},
            "Other": {"classpath": str(tmp_path / "abs.yml")},
            "Class": {"classpath": "Model:MyNet"},
        }
    }
    assert api._config_copy(dict(tree)) == tree
    source = tmp_path / "Prediction.yml"
    source.write_text("Predictor:\n  Model:\n    classpath: default|UNet.yml\n", encoding="utf-8")
    assert Path(api._config_copy(source)).read_bytes() == source.read_bytes()


def test_a_call_releases_its_scratch_config_and_restores_the_callers_rng(
    cohort: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree is materialized under a scratch directory a spawned rank re-reads; it was removed at
    interpreter exit only, so a notebook's thousandth call held a thousand. And building a workflow
    draws (the split, the init): the caller's generators come back as they were."""
    import random
    import tempfile

    from konfai.utils.runtime.environment import _SCRATCH_CONFIGS

    monkeypatch.chdir(cohort)
    scratch_root = Path(tempfile.gettempdir())
    before = {p.name for p in scratch_root.glob("konfai_transformer_*")}
    random.seed(3)
    torch.manual_seed(5)
    states = (random.getstate(), torch.get_rng_state().clone(), np.random.get_state()[1].copy())
    registered = len(_SCRATCH_CONFIGS)

    api.transform(
        "SCOPED",
        "./Raw:mha",
        {"CT": {"CT": [Write(dataset="./OutScoped:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )

    assert {p.name for p in scratch_root.glob("konfai_transformer_*")} == before
    assert len(_SCRATCH_CONFIGS) == registered
    assert random.getstate() == states[0]
    assert torch.equal(torch.get_rng_state(), states[1])
    assert np.array_equal(np.random.get_state()[1], states[2])


def test_a_third_stage_of_one_class_is_spelled_by_occurrence() -> None:
    tree = api._chain_tree(
        [Clip(min_value=0.0), Clip(max_value=1.0), Clip(min_value=0.5)], api._STAGE_MODULES, "chains.CT.CT"
    )
    assert list(tree) == ["Clip", "konfai.data.transform:Clip", "Clip#3"]


def test_three_clips_run_as_one_chain(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The third occurrence once refused with 'split the chain'; it binds under Clip#3 and runs in
    order: three nested clips equal the innermost."""
    monkeypatch.chdir(cohort)
    api.transform(
        "THREE",
        "./Raw:mha",
        {
            "CT": {
                "CT": [
                    Clip(min_value=-100.0, max_value=100.0),
                    Clip(min_value=-50.0, max_value=80.0),
                    Clip(min_value=-20.0, max_value=60.0),
                    Write(dataset="./OutThree:mha"),
                ]
            }
        },
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    api.transform(
        "ONE",
        "./Raw:mha",
        {"CT": {"CT": [Clip(min_value=-20.0, max_value=60.0), Write(dataset="./OutOne:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    for case in ("P000", "P001"):
        three = sitk.GetArrayFromImage(sitk.ReadImage(str(cohort / "OutThree" / case / "CT.mha")))
        one = sitk.GetArrayFromImage(sitk.ReadImage(str(cohort / "OutOne" / case / "CT.mha")))
        np.testing.assert_array_equal(three, one)


@pytest.mark.parametrize("live_objects", [False, True])
def test_repeated_qualified_stages_preserve_their_module_and_application_order(
    cohort: Path, monkeypatch: pytest.MonkeyPatch, live_objects: bool
) -> None:
    import sys
    from types import ModuleType

    modules = [ModuleType("first_clip_module"), ModuleType("second_clip_module")]
    for module in modules:
        module.Clip = type("Clip", (Clip,), {"__module__": module.__name__})
        monkeypatch.setitem(sys.modules, module.__name__, module)
    kwargs = [{"min_value": -100.0}, {"max_value": 80.0}, {"min_value": -20.0}, {"max_value": 60.0}]
    stages = [
        modules[index % 2].Clip(**arguments) if live_objects else {f"{modules[index % 2].__name__}:Clip": arguments}
        for index, arguments in enumerate(kwargs)
    ]
    output = cohort / "Qualified"
    api.transform(
        "QUALIFIED",
        f"{cohort / 'Raw'}:mha",
        {"CT": {"CT": [*stages, Write(dataset=f"{output}:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    for case in ("P000", "P001"):
        source = sitk.GetArrayFromImage(sitk.ReadImage(str(cohort / "Raw" / case / "CT.mha")))
        predicted = sitk.GetArrayFromImage(sitk.ReadImage(str(output / case / "CT.mha")))
        np.testing.assert_array_equal(predicted, np.clip(source, -20.0, 60.0))


@pytest.mark.parametrize("workflow", ["plan", "predict"])
@pytest.mark.parametrize("missing_input", [False, True])
def test_plans_and_live_predictions_release_scratch_on_success_and_failure(
    cohort: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, missing_input: bool
) -> None:
    import random
    import tempfile

    from konfai.utils.runtime.environment import _SCRATCH_CONFIGS

    created: list[Path] = []
    mkdtemp = tempfile.mkdtemp

    def tracked_mkdtemp(*args, **kwargs):
        path = Path(mkdtemp(*args, **kwargs))
        if path.name.startswith("konfai_"):
            created.append(path)
        return str(path)

    monkeypatch.setattr(tempfile, "mkdtemp", tracked_mkdtemp)
    source = f"{cohort / ('Missing' if missing_input else 'Raw')}:mha"
    output = cohort / "ScopedOutput"
    model = torch.nn.Conv3d(1, 1, 1)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    registered = len(_SCRATCH_CONFIGS)

    def call():
        if workflow == "plan":
            return api.plan_transform(
                "SCOPED_PLAN",
                source,
                {"CT": {"CT": [Write(dataset=f"{output}:mha")]}},
                transforms_dir=cohort / "Transforms",
                quiet=True,
            )
        return api.predict_model(
            model,
            source,
            inputs="CT",
            patch=[6, 7, 8],
            output=f"{output}:mha",
            predictions_dir=cohort / "Predictions",
            quiet=True,
        )

    if missing_input:
        with pytest.raises(KonfAIError, match="Group source 'CT'"):
            call()
    else:
        result = call()
        assert result is not None
        if workflow == "predict":
            predicted = sitk.ReadImage(str(output / "P000" / "PRED.mha"))
            assert predicted.GetSize() == (8, 7, 6)
    assert created, "the workflow exercised its actual scratch-config path"
    assert not any(path.exists() for path in created)
    assert len(_SCRATCH_CONFIGS) == registered
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])


def test_released_scratch_does_not_accumulate_exit_callbacks(tmp_path: Path) -> None:
    import atexit

    from konfai.utils.runtime.environment import _SCRATCH_CONFIGS, register_scratch_config, release_scratch_configs

    before = atexit._ncallbacks()
    for index in range(3):
        scratch = tmp_path / str(index)
        scratch.mkdir()
        mark = len(_SCRATCH_CONFIGS)
        register_scratch_config(scratch)
        release_scratch_configs(mark)
        assert not scratch.exists()
    assert atexit._ncallbacks() == before


# ------------------------------------------------------------------------------ bring your model


def test_a_model_built_in_python_trains_and_predicts_in_ten_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ten-line path: an ``nn.Module`` with one tensor in and one out, a dataset root, the
    groups it reads and scores, a loss, a patch. Trained one epoch on the CPU, then predicted twice:
    from the checkpoint the training wrote, and from the weights the module holds in memory."""
    from konfai.data.transform import TensorCast
    from konfai.metric.measure import CrossEntropyLoss

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(3)
    for case in ("P000", "P001"):
        _write_case(tmp_path / "Raw" / case / "CT.mha", rng.normal(0.0, 100.0, (4, 8, 8)).astype(np.float32))
        _write_case(tmp_path / "Raw" / case / "SEG.mha", rng.integers(0, 2, (4, 8, 8)).astype(np.uint8))
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 2, 1))

    checkpoints = api.train_model(
        model,
        "./Raw:mha",
        inputs="CT",
        targets="SEG",
        loss=CrossEntropyLoss(),
        patch=[1, 8, 8],
        epochs=1,
        batch_size=2,
        transforms={"SEG": [TensorCast(dtype="int64")]},
        validation=0.5,
        name="TEN_LINES",
        manual_seed=1,
        checkpoints_dir=tmp_path / "Checkpoints",
        statistics_dir=tmp_path / "Statistics",
        quiet=True,
    )
    saved = sorted(checkpoints.glob("*.pt"))
    assert checkpoints == tmp_path / "Checkpoints" / "TEN_LINES" and saved, "one epoch wrote a checkpoint"
    record = (tmp_path / "Statistics" / "TEN_LINES" / "Trainer.yml").read_text(encoding="utf-8")
    assert "konfai.api:live_model" in record, (
        "the run record names the live model, as every run keeps its resolved config"
    )

    workspace = api.predict_model(
        model,
        "./Raw:mha",
        inputs="CT",
        patch=[1, 8, 8],
        output="./Pred:mha",
        checkpoints=saved[-1],
        name="TEN_LINES",
        predictions_dir=tmp_path / "Predictions",
        quiet=True,
    )
    assert workspace == tmp_path / "Predictions" / "TEN_LINES"
    # A relative output root lands under the run's workspace, as it does for every prediction.
    predicted = sitk.GetArrayFromImage(sitk.ReadImage(str(workspace / "Pred" / "P000" / "PRED.mha")))
    assert predicted.shape == (4, 8, 8, 2), "the two logit channels, as a vector image on the case's grid"

    # No checkpoint named: the weights the module holds are what predicts, written as one for the run.
    live_workspace = api.predict_model(
        model,
        "./Raw:mha",
        inputs="CT",
        patch=[1, 8, 8],
        output=f"{tmp_path / 'PredLive'}:mha",  # absolute: written where it says
        name="TEN_LINES_LIVE",
        predictions_dir=tmp_path / "Predictions",
        quiet=True,
    )
    assert live_workspace == tmp_path / "Predictions" / "TEN_LINES_LIVE"
    live = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "PredLive" / "P000" / "PRED.mha")))
    assert live.shape == predicted.shape
    (
        np.testing.assert_allclose(live, predicted, rtol=1e-5, atol=1e-5),
        "the checkpoint holds the weights the module holds",
    )


def test_a_live_model_token_is_released_when_the_run_returns() -> None:
    model = object()
    with api._registered_live_model(model) as token:
        assert api.live_model(token) is model
    assert token not in api._LIVE_MODELS
    with pytest.raises(ConfigError, match="No live model"):
        api.live_model(token)


def test_a_live_model_refuses_several_ranks() -> None:
    with pytest.raises(ConfigError, match="one rank"):
        api.train_model(
            torch.nn.Identity(), "./Raw:mha", inputs="CT", targets="SEG", loss=[], patch=[1, 8, 8], gpu=[0, 1]
        )


def test_a_monai_unet_trains_and_predicts_through_the_same_ten_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library model the adoption page promises: MONAI's UNet, untouched, through train_model
    and predict_model."""
    monai_nets = pytest.importorskip("monai.networks.nets")
    from konfai.data.transform import TensorCast
    from konfai.metric.measure import CrossEntropyLoss

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(5)
    for case in ("P000", "P001"):
        _write_case(tmp_path / "Raw" / case / "CT.mha", rng.normal(0.0, 100.0, (4, 16, 16)).astype(np.float32))
        _write_case(tmp_path / "Raw" / case / "SEG.mha", rng.integers(0, 3, (4, 16, 16)).astype(np.uint8))
    model = monai_nets.UNet(spatial_dims=2, in_channels=1, out_channels=3, channels=(4, 8, 16), strides=(2, 2))

    checkpoints = api.train_model(
        model,
        "./Raw:mha",
        inputs="CT",
        targets="SEG",
        loss=CrossEntropyLoss(),
        patch=[1, 16, 16],
        epochs=1,
        batch_size=2,
        transforms={"SEG": [TensorCast(dtype="int64")]},
        validation=0.5,
        name="MONAI",
        manual_seed=1,
        checkpoints_dir=tmp_path / "Checkpoints",
        statistics_dir=tmp_path / "Statistics",
        quiet=True,
    )
    workspace = api.predict_model(
        model,
        "./Raw:mha",
        inputs="CT",
        patch=[1, 16, 16],
        output="./Pred:mha",
        checkpoints=sorted(checkpoints.glob("*.pt"))[-1],
        name="MONAI",
        predictions_dir=tmp_path / "Predictions",
        quiet=True,
    )
    predicted = sitk.GetArrayFromImage(sitk.ReadImage(str(workspace / "Pred" / "P001" / "PRED.mha")))
    assert predicted.shape == (4, 16, 16, 3)
