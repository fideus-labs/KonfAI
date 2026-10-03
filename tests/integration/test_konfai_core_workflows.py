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

import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from harness import konfai_cli_command, prepare_experiment_dir, replace_once, run_workflow, write_image
from konfai.evaluator import build_evaluate
from konfai.predictor import build_predict
from konfai.trainer import build_train
from konfai.transformer import build_transform
from konfai.utils.errors import DatasetManagerError

pytestmark = pytest.mark.integration

#: Windows opens no process group, and KonfAI refuses the ranks that would need one.
_SEVERAL_RANKS = pytest.mark.skipif(os.name == "nt", reason="several ranks need a process group")

SimpleITK = pytest.importorskip("SimpleITK")


def _create_prediction_dataset_stub(predictions_dataset_dir: Path) -> None:
    predictions_dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        case_dir = predictions_dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        sct = np.zeros((3, 16, 16), dtype=np.float32)
        write_image(case_dir / "sCT.mha", sct, SimpleITK.sitkFloat32)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _assert_experiment_outputs(
    dataset_dir: Path,
    checkpoints_dir: Path,
    predictions_dir: Path,
    evaluations_dir: Path,
    train_name: str,
) -> None:
    expected_cases = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    checkpoints = sorted((checkpoints_dir / train_name).glob("*.pt"))
    assert checkpoints
    predicted = sorted((predictions_dir / train_name / "Dataset").rglob("sCT.mha"))
    assert len(predicted) == len(expected_cases)
    assert sorted(path.parent.name for path in predicted) == expected_cases
    for path in predicted:
        image = SimpleITK.ReadImage(str(path))
        array = SimpleITK.GetArrayFromImage(image)
        assert array.shape == (3, 16, 16)
        assert np.isfinite(array).all()
    metrics_path = evaluations_dir / train_name / "Metric_TRAIN.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "case" in metrics
    assert any(key.endswith("MAE") for key in metrics["case"])
    for metric_name, case_values in metrics["case"].items():
        assert sorted(case_values) == expected_cases
        assert all(isinstance(value, (int, float)) for value in case_values.values()), metric_name
        assert all(np.isfinite(value) for value in case_values.values()), metric_name


def test_konfai_cli_user_path(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment_cli"
    train_name = "CLI"
    paths = prepare_experiment_dir(experiment_dir, train_name)
    cli = konfai_cli_command()

    run_workflow(
        [
            *cli,
            "TRAIN",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "Config.yml",
            "--checkpoints-dir",
            "Checkpoints",
            "--statistics-dir",
            "Statistics",
        ],
        experiment_dir,
    )
    checkpoints = sorted((paths["checkpoints_dir"] / train_name).glob("*.pt"))
    assert checkpoints

    run_workflow(
        [
            *cli,
            "PREDICTION",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "Prediction.yml",
            "--models",
            *[str(path) for path in checkpoints],
            "--predictions-dir",
            "Predictions",
        ],
        experiment_dir,
    )
    run_workflow(
        [
            *cli,
            "EVALUATION",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "Evaluation.yml",
            "--evaluations-dir",
            "Evaluations",
        ],
        experiment_dir,
    )
    _assert_experiment_outputs(
        paths["dataset_dir"],
        paths["checkpoints_dir"],
        paths["predictions_dir"],
        paths["evaluations_dir"],
        train_name,
    )


def test_konfai_build_steps_construct_workflows_without_execution(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment_build"
    train_name = "BUILD"
    paths = prepare_experiment_dir(experiment_dir, train_name)
    _create_prediction_dataset_stub(paths["predictions_dir"] / train_name / "Dataset")

    sys.path.insert(0, str(experiment_dir))
    try:
        with _working_directory(experiment_dir):
            trainer = build_train(
                config=experiment_dir / "Config.yml",
                checkpoints_dir=paths["checkpoints_dir"],
                statistics_dir=experiment_dir / "Statistics",
            )
            predictor = build_predict(
                models=[experiment_dir / "dummy.pt"],
                prediction_file=experiment_dir / "Prediction.yml",
                predictions_dir=paths["predictions_dir"],
            )
            evaluator = build_evaluate(
                evaluations_file=experiment_dir / "Evaluation.yml",
                evaluations_dir=paths["evaluations_dir"],
            )
    finally:
        sys.path.remove(str(experiment_dir))

    assert trainer.name == train_name
    assert predictor.name == train_name
    assert evaluator.name == train_name


# What the builders export to the environment.
_WORKFLOW_ENVIRONMENT = (
    "KONFAI_ROOT",
    "KONFAI_STATE",
    "KONFAI_CHECKPOINTS_DIRECTORY",
    "KONFAI_STATISTICS_DIRECTORY",
    "KONFAI_PREDICTIONS_DIRECTORY",
    "KONFAI_EVALUATIONS_DIRECTORY",
    "KONFAI_TRANSFORMS_DIRECTORY",
)

_TRANSFORM_CONFIG = """\
Transformer:
  name: WRITE_BACK
  Dataset:
    dataset_filenames:
      - {dataset}:a:mha
    groups_src:
      MR:
        groups_dest:
          MR_out:
            transforms:
              Clip:
                min_value: 0.0
                max_value: 1.0
              Write:
                dataset: {out}:mha
"""


# Each workflow's builder, as its CLI command calls it, over the config at `path` in the experiment `root`.
_BUILDS = {
    "Config.yml": lambda path, root: build_train(
        config=path, checkpoints_dir=root / "Checkpoints", statistics_dir=root / "Statistics"
    ),
    "Prediction.yml": lambda path, root: build_predict(
        models=[root / "dummy.pt"], prediction_file=path, predictions_dir=root / "Predictions"
    ),
    "Evaluation.yml": lambda path, root: build_evaluate(evaluations_file=path, evaluations_dir=root / "Evaluations"),
    "Transform.yml": lambda path, root: build_transform(transform_file=path, transforms_dir=root / "Transforms"),
}


@pytest.mark.parametrize("name", list(_BUILDS))
def test_a_failed_build_leaves_the_config_as_written_and_a_successful_one_writes_it(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow's config, built with its dataset directory missing, then as written: the failed build
    leaves the file byte-identical, the successful one writes the resolved defaults back. Each of the
    four wrote what resolved before the error (Config.yml 75 -> 88 lines)."""
    # Pinned, so what the builders export is taken back after the test.
    for key in _WORKFLOW_ENVIRONMENT:
        monkeypatch.setenv(key, "sentinel")
        monkeypatch.delenv(key)
    experiment_dir = tmp_path / "experiment_write_back"
    paths = prepare_experiment_dir(experiment_dir, "WRITE_BACK")
    dataset = paths["dataset_dir"]
    _create_prediction_dataset_stub(paths["predictions_dir"] / "WRITE_BACK" / "Dataset")
    (experiment_dir / "Transform.yml").write_text(
        _TRANSFORM_CONFIG.format(dataset=dataset, out=experiment_dir / "Out"), encoding="utf-8"
    )
    monkeypatch.chdir(experiment_dir)
    monkeypatch.syspath_prepend(str(experiment_dir))
    path = experiment_dir / name
    written = path.read_text(encoding="utf-8")
    path.write_text(replace_once(written, f"{dataset}:a:mha", f"{dataset}_moved:a:mha"), encoding="utf-8")
    broken = path.read_bytes()

    with pytest.raises(DatasetManagerError, match="not found in any dataset"):
        _BUILDS[name](path, experiment_dir)
    assert path.read_bytes() == broken

    path.write_text(written, encoding="utf-8")
    _BUILDS[name](path, experiment_dir)
    assert len(path.read_text(encoding="utf-8").splitlines()) > len(written.splitlines())


def _validation_scores(tmp_path: Path, ranks: int, batch_size: int) -> set[float]:
    """The scores of the checkpoints a TRAIN at ``lr: 0`` saves: every one is the validation value of
    the untrained model, over the three patches of the one validation case."""
    experiment_dir = tmp_path / f"ranks_{ranks}_batch_{batch_size}"
    paths = prepare_experiment_dir(experiment_dir, "VAL")
    config = experiment_dir / "Config.yml"
    content = replace_once(config.read_text(encoding="utf-8"), "lr: 0.001", "lr: 0.0")
    config.write_text(replace_once(content, "batch_size: 16", f"batch_size: {batch_size}"), encoding="utf-8")
    run_workflow(
        [
            *konfai_cli_command(),
            "TRAIN",
            "-y",
            "--cpu",
            str(ranks),
            "-q",
            "-c",
            "Config.yml",
            "--checkpoints-dir",
            "Checkpoints",
            "--statistics-dir",
            "Statistics",
        ],
        experiment_dir,
    )
    checkpoints = sorted((paths["checkpoints_dir"] / "VAL").glob("*.pt"))
    assert checkpoints
    return {float(torch.load(path, map_location="cpu", weights_only=False)["loss"]) for path in checkpoints}


@_SEVERAL_RANKS
def test_two_rank_validation_scores_every_patch_once(tmp_path: Path) -> None:
    # Three validation patches: two ranks score one and two, three ranks one each. A batch of one
    # patch leaves the first of two ranks a batch short, and its padding is run but never scored.
    (reference,) = _validation_scores(tmp_path, 1, 16)
    for ranks, batch_size in ((2, 16), (3, 16), (2, 1)):
        (score,) = _validation_scores(tmp_path, ranks, batch_size)
        assert score == pytest.approx(reference, rel=1e-6), (ranks, batch_size)


@_SEVERAL_RANKS
def test_two_rank_evaluation_scores_a_validation_split_smaller_than_the_ranks(tmp_path: Path) -> None:
    """One validation case over two ranks: the rank that holds none still takes part in the split."""
    experiment_dir = tmp_path / "experiment_eval_ranks"
    train_name = "EVAL_RANKS"
    paths = prepare_experiment_dir(experiment_dir, train_name)
    _create_prediction_dataset_stub(paths["predictions_dir"] / train_name / "Dataset")
    config = experiment_dir / "Evaluation.yml"
    config.write_text(replace_once(config.read_text(), "validation: None", "validation: [CASE_003]"))

    completed = run_workflow(
        [*konfai_cli_command(), "EVALUATION", "-y", "--cpu", "2", "-q", "-c", "Evaluation.yml"],
        experiment_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    root = paths["evaluations_dir"] / train_name
    train = json.loads((root / "Metric_TRAIN.json").read_text())
    validation = json.loads((root / "Metric_VALIDATION.json").read_text())
    assert sorted(train["case"]["sCT:CT:MAE"]) == ["CASE_000", "CASE_001", "CASE_002"]
    assert sorted(validation["case"]["sCT:CT:MAE"]) == ["CASE_003"]


@pytest.fixture(scope="module")
def trained_synthesis(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Path]]:
    """One tiny synthesis model, trained once for the tests that only predict with it."""
    experiment_dir = tmp_path_factory.mktemp("trained") / "experiment"
    paths = prepare_experiment_dir(experiment_dir, "UNREADABLE")
    run_workflow(
        [*konfai_cli_command(), "TRAIN", "-y", "--cpu", "1", "-q", "-c", "Config.yml"],
        experiment_dir,
    )
    return experiment_dir, sorted((paths["checkpoints_dir"] / "UNREADABLE").glob("*.pt"))


def _spoil(path: Path, how: str) -> bytes:
    """A truncated file keeps its header and fails at the voxels, a garbage one fails at the header."""
    original = path.read_bytes()
    path.write_bytes(original[: int(len(original) * 0.7)] if how == "truncated" else b"not an image\n")
    return original


def _run(experiment_dir: Path, *arguments: str):
    return run_workflow(
        [*konfai_cli_command(), *arguments], experiment_dir, check=False, capture_output=True, text=True
    )


def _said(completed, workspace: Path) -> str:
    """What the run said: the console, and each rank's log, where a rank past the first writes."""
    return completed.stdout + completed.stderr + "".join(log.read_text() for log in workspace.glob("log_*.txt"))


# A batch of 16 holds every patch of the cohort, so the case is set aside before any of its patches is
# forwarded; a batch of 1 streams the slices read before the failing one to the sink, which is aborted.
@pytest.mark.parametrize(
    ("how", "num_workers", "batch_size", "streamed", "ranks"),
    [
        ("truncated", 0, 16, False, 1),
        ("truncated", 2, 16, False, 1),
        ("truncated", 0, 16, True, 1),
        ("truncated", 0, 1, True, 1),
        ("truncated", 2, 1, True, 1),
        ("truncated", 0, 1, True, 2),
        ("garbage", 0, 16, False, 1),
        ("garbage", 2, 1, True, 1),
    ],
)
def test_prediction_sets_aside_an_unreadable_case_and_predicts_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_synthesis: tuple[Path, list[Path]],
    how: str,
    num_workers: int,
    batch_size: int,
    streamed: bool,
    ranks: int,
) -> None:
    """One unreadable input among four: the three others are predicted, the run exits 0 naming the
    case, and the case has no output at all, so a relaunch predicts it once the file is fixed."""
    experiment_dir, checkpoints = trained_synthesis
    dataset = tmp_path / "Dataset"
    shutil.copytree(experiment_dir / "Dataset", dataset)
    shutil.copy(experiment_dir / "TinySynth.py", tmp_path)
    original = _spoil(dataset / "CASE_002" / "MR.mha", how)
    config = (experiment_dir / "Prediction.yml").read_text().replace(str(experiment_dir / "Dataset"), str(dataset))
    (tmp_path / "Prediction.yml").write_text(
        replace_once(config, "batch_size: 16", f"batch_size: {batch_size}\n    num_workers: {num_workers}")
    )
    if streamed:
        monkeypatch.setenv("KONFAI_STREAM_WORTH_THRESHOLD", "0")  # the toy volumes stream their slabs to the sink
    arguments = ["PREDICTION", "--cpu", str(ranks), "-c", "Prediction.yml", "--models", *map(str, checkpoints)]
    completed = _run(tmp_path, *arguments, "-y")
    predictions = tmp_path / "Predictions" / "UNREADABLE" / "Dataset"
    written = sorted(path.parent.name for path in predictions.rglob("sCT.mha"))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert written == ["CASE_000", "CASE_001", "CASE_003"]
    assert not list(predictions.glob("CASE_002/*")), "no partial output is left for the case set aside"
    output = _said(completed, tmp_path / "Predictions" / "UNREADABLE")
    assert "CASE_002" in output and "MR" in output and "set aside" in output

    (dataset / "CASE_002" / "MR.mha").write_bytes(original)
    relaunched = _run(tmp_path, *arguments)
    assert relaunched.returncode == 0, relaunched.stdout + relaunched.stderr
    assert "3/4 case(s) already written" in relaunched.stdout + relaunched.stderr
    assert sorted(path.parent.name for path in predictions.rglob("sCT.mha")) == [f"CASE_{i:03d}" for i in range(4)]


@pytest.mark.parametrize(
    ("how", "num_workers", "memory_budget", "ranks"),
    [
        ("truncated", 0, "None", 1),
        ("truncated", 2, "None", 1),
        ("truncated", 0, "4000b", 1),
        ("truncated", 2, "4000b", 1),
        pytest.param("truncated", 0, "4000b", 2, marks=_SEVERAL_RANKS),
        ("garbage", 0, "None", 1),
        ("garbage", 2, "4000b", 1),
        pytest.param("garbage", 0, "None", 2, marks=_SEVERAL_RANKS),
    ],
)
def test_evaluation_sets_aside_an_unreadable_case_and_records_it(
    tmp_path: Path, how: str, num_workers: int, memory_budget: str, ranks: int
) -> None:
    """One unreadable reference among four, whole or in patches: the three others are scored, the run
    exits 0 naming the case, and Metric_TRAIN.json lists it beside the aggregate that leaves it out."""
    experiment_dir = tmp_path / "experiment_unreadable"
    paths = prepare_experiment_dir(experiment_dir, "UNREADABLE")
    _create_prediction_dataset_stub(paths["predictions_dir"] / "UNREADABLE" / "Dataset")
    _spoil(paths["dataset_dir"] / "CASE_001" / "CT.mha", how)
    config = experiment_dir / "Evaluation.yml"
    config.write_text(
        replace_once(
            config.read_text(),
            "batch_size: 4",
            f"batch_size: 4\n    num_workers: {num_workers}\n    memory_budget: {memory_budget}",
        )
    )

    completed = _run(experiment_dir, "EVALUATION", "-y", "--cpu", str(ranks), "-c", "Evaluation.yml")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = _said(completed, paths["evaluations_dir"] / "UNREADABLE")
    assert "CASE_001" in output and "set aside" in output
    if memory_budget != "None":
        assert "disjoint patches" in output
    report = json.loads((paths["evaluations_dir"] / "UNREADABLE" / "Metric_TRAIN.json").read_text())
    assert sorted(report["case"]["sCT:CT:MAE"]) == ["CASE_000", "CASE_002", "CASE_003"]
    assert report["aggregates"]["sCT:CT:MAE"]["count"] == 3
    assert list(report["set_aside"]) == ["CASE_001"]
    assert "CT" in report["set_aside"]["CASE_001"]


@pytest.mark.parametrize("memory_budget", ["None", "4000b"])
def test_evaluation_warns_about_a_prediction_on_another_geometry_and_scores_it(
    tmp_path: Path, memory_budget: str
) -> None:
    """CASE_001's prediction has its reference's shape but lies 10 mm away: one warning names the case,
    both groups and both origins, whole or in patches, and every case is scored."""
    experiment_dir = tmp_path / "experiment_geometry"
    paths = prepare_experiment_dir(experiment_dir, "GEOMETRY")
    predictions = paths["predictions_dir"] / "GEOMETRY" / "Dataset"
    _create_prediction_dataset_stub(predictions)
    shifted = SimpleITK.ReadImage(str(predictions / "CASE_001" / "sCT.mha"))
    shifted.SetOrigin((10.0, 0.0, 0.0))
    SimpleITK.WriteImage(shifted, str(predictions / "CASE_001" / "sCT.mha"))
    config = experiment_dir / "Evaluation.yml"
    config.write_text(
        replace_once(config.read_text(), "batch_size: 4", f"batch_size: 4\n    memory_budget: {memory_budget}")
    )

    completed = _run(experiment_dir, "EVALUATION", "-y", "--cpu", "1", "-c", "Evaluation.yml")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    console = completed.stdout + completed.stderr  # the rank log holds the same lines
    warned = [line for line in console.splitlines() if "geometry" in line and "Case '" in line]
    assert len(warned) == 1, console
    assert "CASE_001" in warned[0] and "'sCT'" in warned[0] and "'CT'" in warned[0]
    assert "[10.0, 0.0, 0.0]" in warned[0] and "[0.0, 0.0, 0.0]" in warned[0]
    report = json.loads((paths["evaluations_dir"] / "GEOMETRY" / "Metric_TRAIN.json").read_text())
    assert sorted(report["case"]["sCT:CT:MAE"]) == [f"CASE_{i:03d}" for i in range(4)]


def _predict_spoiled(tmp_path: Path, trained_synthesis: tuple[Path, list[Path]], spoils: str, ranks: int):
    """Predict the four cases, the input of case i spoiled as ``spoils[i]`` says: ``t`` truncated,
    ``g`` garbage, ``-`` left readable."""
    experiment_dir, checkpoints = trained_synthesis
    dataset = tmp_path / "Dataset"
    shutil.copytree(experiment_dir / "Dataset", dataset)
    shutil.copy(experiment_dir / "TinySynth.py", tmp_path)
    for index, how in enumerate(spoils):
        if how != "-":
            _spoil(dataset / f"CASE_{index:03d}" / "MR.mha", "truncated" if how == "t" else "garbage")
    config = (experiment_dir / "Prediction.yml").read_text().replace(str(experiment_dir / "Dataset"), str(dataset))
    (tmp_path / "Prediction.yml").write_text(config)
    arguments = ["PREDICTION", "-y", "--cpu", str(ranks), "-c", "Prediction.yml", "--models", *map(str, checkpoints)]
    return _run(tmp_path, *arguments)


# A truncated file fails at its voxels, on the rank that holds it: no rank alone sees the cohort fail.
@pytest.mark.parametrize(("spoils", "ranks"), [("tttt", 1), ("gggg", 1), ("gggg", 2), ("tttt", 2), ("ggtt", 2)])
def test_prediction_of_a_cohort_that_reads_nowhere_fails(
    tmp_path: Path, trained_synthesis: tuple[Path, list[Path]], spoils: str, ranks: int
) -> None:
    """Every input unreadable is a wrong tree or an unreadable disk, not a bad file: the run fails."""
    completed = _predict_spoiled(tmp_path, trained_synthesis, spoils, ranks)

    assert completed.returncode != 0
    assert "None of the 4 case(s) could be read" in _said(completed, tmp_path / "Predictions" / "UNREADABLE")
    assert not list((tmp_path / "Predictions" / "UNREADABLE").rglob("sCT.mha"))


@pytest.mark.parametrize(
    ("how", "memory_budget", "ranks"),
    [("truncated", "None", 1), pytest.param("truncated", "4000b", 2, marks=_SEVERAL_RANKS)],
)
def test_evaluation_of_a_cohort_that_reads_nowhere_fails(
    tmp_path: Path, how: str, memory_budget: str, ranks: int
) -> None:
    """Every reference unreadable, whole or in patches, on one rank or two: the run fails, no report."""
    experiment_dir = tmp_path / "experiment_unreadable"
    paths = prepare_experiment_dir(experiment_dir, "UNREADABLE")
    _create_prediction_dataset_stub(paths["predictions_dir"] / "UNREADABLE" / "Dataset")
    for path in paths["dataset_dir"].glob("CASE_*/CT.mha"):
        _spoil(path, how)
    config = experiment_dir / "Evaluation.yml"
    config.write_text(
        replace_once(config.read_text(), "batch_size: 4", f"batch_size: 4\n    memory_budget: {memory_budget}")
    )

    completed = _run(experiment_dir, "EVALUATION", "-y", "--cpu", str(ranks), "-c", "Evaluation.yml")

    assert completed.returncode != 0
    assert "None of the 4 TRAIN case(s) could be read" in _said(completed, paths["evaluations_dir"] / "UNREADABLE")
    assert not (paths["evaluations_dir"] / "UNREADABLE" / "Metric_TRAIN.json").exists()
