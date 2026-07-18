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
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from konfai_mcp import runner as mcp_runner  # noqa: E402
from konfai_mcp.server_experiments import SessionService  # noqa: E402
from konfai_mcp.server_jobs import JobRegistry  # noqa: E402
from konfai_mcp.server_support import WorkspaceLayout  # noqa: E402


def _service(tmp_path: Path) -> SessionService:
    repo_root = Path(__file__).resolve().parents[2]
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    return SessionService(
        repo_root=repo_root,
        examples_root=repo_root / "examples",
        workspace_layout=layout,
        job_registry=JobRegistry({"queued", "running"}, workspace_layout=layout),
        max_log_tail_lines=20,
        active_job_states={"queued", "running"},
        validation_levels={"instantiate", "setup"},
        workflows={"train", "prediction", "evaluation"},
    )


def test_default_prediction_checkpoint_is_scoped_to_the_run(tmp_path: Path) -> None:
    # In a sweep, Checkpoints/ holds several runs. The default checkpoint for run A's prediction must come
    # from Checkpoints/RUN_A, not the globally-newest .pt (which belongs to whichever run trained last).
    service = _service(tmp_path)
    checkpoints = service.workspace_layout.checkpoints_dir()
    (checkpoints / "RUN_A").mkdir(parents=True)
    (checkpoints / "RUN_B").mkdir(parents=True)
    run_a_ckpt = checkpoints / "RUN_A" / "epoch.pt"
    run_b_ckpt = checkpoints / "RUN_B" / "epoch.pt"
    run_a_ckpt.write_text("a", encoding="utf-8")
    run_b_ckpt.write_text("b", encoding="utf-8")
    os.utime(run_a_ckpt, (1000, 1000))  # older
    os.utime(run_b_ckpt, (2000, 2000))  # newer -> the globally-newest checkpoint

    # Scoped to RUN_A: its own checkpoint, NOT run B's newer one.
    assert service.discover_model_paths(run_name="RUN_A") == [run_a_ckpt]
    # Unscoped falls back to the global newest (the previous behaviour), used only when no run is known.
    assert service.discover_model_paths() == [run_b_ckpt]


def test_extensionless_dicom_series_directory_is_detected(tmp_path: Path) -> None:
    # PACS exports store DICOM slices with no extension; a suffix-only scan misses them. The magic-byte
    # sniff must classify such a directory as a dicom group, without false-positiving on a plain folder.
    service = _service(tmp_path)
    extensions = service._supported_extensions()

    dicom_dir = tmp_path / "CT"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001").write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 64)
    assert service._classify_directory_entry(dicom_dir, extensions) == ("CT", "dicom")

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "readme").write_text("hello", encoding="utf-8")
    assert service._classify_directory_entry(notes_dir, extensions) is None


def test_label_statistics_cover_many_class_segmentations_and_flag_intensity_groups(tmp_path: Path) -> None:
    # A whole-body segmentation (e.g. TotalSegmentator, ~117 classes) must still get per-label stats -- the
    # old <=64 cap silently dropped them. An intensity image stored as int16 (thousands of values) must NOT
    # produce a huge label dict; it is flagged as high-cardinality instead of silently omitted.
    sitk = pytest.importorskip("SimpleITK")
    service = _service(tmp_path)
    dataset_dir = tmp_path / "Dataset"
    for case in range(2):
        case_dir = dataset_dir / f"CASE_{case:03d}"
        case_dir.mkdir(parents=True)
        seg = (np.arange(1 * 40 * 40).reshape(1, 40, 40) % 117).astype(np.uint8)  # 117 classes present
        sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(seg), sitk.sitkUInt8), str(case_dir / "SEG.mha"))
        intensity = (np.arange(1 * 40 * 40).reshape(1, 40, 40) % 2000).astype(np.int16)  # ~1600 values
        sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(intensity), sitk.sitkInt16), str(case_dir / "HU.mha"))

    seg_stats = service.compute_dataset_group_statistics(dataset_dir, "SEG")
    assert seg_stats["labels"]["count"] == 117  # not dropped by a low cap

    hu_stats = service.compute_dataset_group_statistics(dataset_dir, "HU")
    assert "labels" not in hu_stats  # not treated as a label map
    assert "high_cardinality_integer_group" in hu_stats  # but not silently omitted either


def test_infer_dataset_structure_is_task_neutral(tmp_path: Path) -> None:
    service = _service(tmp_path)
    case_dir = tmp_path / "Dataset" / "case_001"
    case_dir.mkdir(parents=True)
    (case_dir / "CT.mha").write_text("", encoding="utf-8")
    (case_dir / "SEG.mha").write_text("", encoding="utf-8")
    (case_dir / "metadata.json").write_text("{}", encoding="utf-8")

    payload = service.infer_dataset_structure_payload(case_dir.parent)

    assert payload["layout"] == "per_case_directories"
    assert any(path.endswith("metadata.json") for path in payload["ignored_files"])
    assert payload["dataset_entry"] == f"{case_dir.parent}:a:mha"
    assert "recommended_templates" not in payload


def test_design_config_strategy_requires_explicit_task_context(tmp_path: Path) -> None:
    service = _service(tmp_path)
    case_dir = tmp_path / "Dataset" / "case_001"
    case_dir.mkdir(parents=True)
    (case_dir / "MR.mha").write_text("", encoding="utf-8")
    (case_dir / "CT.mha").write_text("", encoding="utf-8")
    (case_dir / "MASK.mha").write_text("", encoding="utf-8")

    payload = service.design_config_strategy_payload(
        case_dir.parent,
        task="synthesis",
        group_roles={"MR": "input", "CT": "target", "MASK": "support"},
        workflows=["train", "prediction"],
        modeling_intent="2.5d",
        example="Synthesis",
    )

    assert payload["task"] == "synthesis"
    assert payload["group_roles"]["input"] == ["MR"]
    assert payload["selected_example"]["name"] == "Synthesis"
    assert payload["customization_options"]["can_write_local_components"] is True
    assert payload["customization_options"]["signature_tool"] == "inspect_object_signature"
    assert payload["unresolved_questions"] == []


def test_design_config_strategy_supports_multiple_datasets(tmp_path: Path) -> None:
    service = _service(tmp_path)
    train_case = tmp_path / "TrainDataset" / "case_001"
    eval_case = tmp_path / "EvalDataset" / "case_001"
    train_case.mkdir(parents=True)
    eval_case.mkdir(parents=True)
    (train_case / "MR.mha").write_text("", encoding="utf-8")
    (train_case / "CT.mha").write_text("", encoding="utf-8")
    (eval_case / "MR.mha").write_text("", encoding="utf-8")
    (eval_case / "CT.mha").write_text("", encoding="utf-8")

    payload = service.design_config_strategy_payload(
        dataset_dir=None,
        dataset_dirs=[train_case.parent, eval_case.parent],
        task="synthesis",
        group_roles={"MR": "input", "CT": "target"},
        workflows=["train", "evaluation"],
        modeling_intent="2.5d",
        example="Synthesis",
    )

    assert payload["dataset_dir"] is None
    assert payload["dataset_summary"]["count"] == 2
    assert payload["dataset_dirs"] == [str(train_case.parent), str(eval_case.parent)]
    assert len(payload["config_plan"]["dataset_entries"]) == 2
    assert payload["config_plan"]["dataset_entries"][0]["entry"].endswith(":a:mha")
    assert any("provided datasets" in question for question in payload["unresolved_questions"])


def test_leaderboard_payload_reads_latest_metrics(tmp_path: Path) -> None:
    service = _service(tmp_path)
    metric_dir = service.workspace_layout.evaluations_dir() / "RUN_01"
    metric_dir.mkdir(parents=True)
    metrics_path = metric_dir / "Metric_TRAIN.json"
    metrics_path.write_text(json.dumps({"aggregates": {"Dice": {"mean": 0.82}}}), encoding="utf-8")

    payload = service.leaderboard_payload(metric="Dice")

    assert payload["session"] == "default"
    assert payload["selected_metric"] == "Dice"
    assert payload["best"]["run_name"] == "RUN_01"
    assert payload["best"]["value"] == 0.82


def test_leaderboard_ranks_app_evaluation_trials(tmp_path: Path) -> None:
    """An app evaluate/pipeline writes its metrics under AppEvaluations/AppPipelines; the leaderboard must
    rank those tuned trials alongside train-branch runs so a refine loop can compare them."""
    service = _service(tmp_path)
    workspace = service.workspace_layout.workspace_dir()
    for label, score in (("eval_app__iterations_100", 0.70), ("eval_app__iterations_300", 0.88)):
        metric_dir = workspace / "AppEvaluations" / label / "Evaluations" / "RUN"
        metric_dir.mkdir(parents=True)
        (metric_dir / "Metric_TRAIN.json").write_text(
            json.dumps({"aggregates": {"Dice": {"mean": score}}}), encoding="utf-8"
        )

    payload = service.leaderboard_payload(metric="Dice")

    assert payload["selected_metric"] == "Dice"
    assert payload["best"]["value"] == 0.88
    # Both trials are visible, and their metrics_path names which parameters produced each score.
    paths = " ".join(row["metrics_path"] for row in payload["leaderboard"])
    assert "iterations_100" in paths
    assert "iterations_300" in paths


def test_session_summary_blocks_evaluation_without_prediction_artifacts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.workspace_layout.train_config_path().write_text(
        "Trainer:\n  train_name: DEMO\n  Dataset:\n    dataset_filenames:\n      - ./Dataset:a:mha\n",
        encoding="utf-8",
    )
    service.workspace_layout.prediction_config_path().write_text(
        "Predictor:\n  train_name: DEMO\n  Dataset:\n    dataset_filenames:\n      - ./Dataset:a:mha\n",
        encoding="utf-8",
    )
    service.workspace_layout.evaluation_config_path().write_text(
        "Evaluator:\n"
        "  train_name: DEMO\n"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        "      - ./Dataset:a:mha\n"
        "      - ./Predictions/DEMO/Dataset:i:mha\n",
        encoding="utf-8",
    )
    (service.workspace_dir() / "Dataset").mkdir()

    summary = service.session_summary()

    assert summary["readiness"]["train"] is True
    assert summary["readiness"]["prediction"] is False
    assert summary["readiness"]["evaluation"] is False
    assert "run_evaluation" not in summary["next_actions"]


def test_validation_runner_creates_runtime_directories_for_setup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validate_root = tmp_path / "validate"
    workspace_dir = tmp_path / "session"
    workspace_dir.mkdir(parents=True)
    config_path = workspace_dir / "Config.yml"
    config_path.write_text("Trainer:\n  train_name: DEMO\n", encoding="utf-8")
    monkeypatch.setenv("KONFAI_MCP_VALIDATE_ROOT", str(validate_root))

    class DummyWorkflow:
        name = "DEMO"
        dataloader: ClassVar[list[object]] = []
        size = 1

        def setup(self, *_args) -> None:
            assert (validate_root / "Checkpoints").exists()
            assert (validate_root / "Statistics").exists()
            assert (validate_root / "Predictions").exists()
            assert (validate_root / "Evaluations").exists()

    monkeypatch.setattr(mcp_runner, "build_train", lambda **_kwargs: DummyWorkflow())
    payload = mcp_runner.validate_workflow_api(
        workflow="train",
        level="setup",
        workspace_dir=str(workspace_dir),
        config=str(config_path),
    )

    assert payload["ok"] is True


def test_validate_semantics_blocks_missing_local_model_source(tmp_path: Path) -> None:
    service = _service(tmp_path)
    config_path = service.workspace_layout.train_config_path()
    config_path.write_text(
        "Trainer:\n"
        "  Model:\n"
        "    classpath: Model:UNetpp5\n"
        "    UNetpp5:\n"
        "      outputs_criterions: None\n"
        "      Patch: None\n"
        "      dim: 2\n"
        "  Dataset:\n"
        "    groups_src:\n"
        "      MR:\n"
        "        groups_dest:\n"
        "          MR:\n"
        "            transforms: None\n"
        "            patch_transforms: None\n"
        "            is_input: true\n"
        "    Patch:\n"
        "      patch_size: [1, 32, 32]\n"
        "      extend_slice: 0\n"
        "    dataset_filenames:\n"
        "      - ./Dataset:a:mha\n"
        "  train_name: DEMO\n",
        encoding="utf-8",
    )
    (service.workspace_dir() / "Dataset").mkdir(exist_ok=True)

    payload = service.validate_semantics("train", "instantiate")

    assert payload["ok"] is False
    assert payload["blocked"] is True
    assert payload["error_type"] == "SemanticIssue"
    assert payload["blocking_issues"][0]["code"] == "missing_local_model_source"


class _PickleLoader:
    def __init__(self, dataset: object) -> None:
        self.dataset = dataset


class _PickleWorkflow:
    def __init__(self, dataset: object) -> None:
        self.dataloader = [[_PickleLoader(dataset)]]


class _CleanDataset:
    # Module-level on purpose: pickle serialises classes by reference, so a test-local class would
    # fail for the wrong reason (unimportable class) instead of the unpicklable member under test.
    value = 3


class _DirtyDataset:
    def __init__(self) -> None:
        self.transform = lambda x: x  # lambdas cannot be pickled for spawn transfer


def test_worker_spawn_picklability_check_catches_unpicklable_datasets() -> None:
    # Validation loads single-process, so an unpicklable dataset member passes validation and then
    # kills the real run's DataLoader workers at setup; the check reproduces the spawn pickling.
    clean = mcp_runner._check_worker_spawn_picklability(_PickleWorkflow(_CleanDataset()), 4)
    assert clean == {"requested_num_workers": 4, "checked": True, "picklable": True, "datasets": 1}

    dirty = mcp_runner._check_worker_spawn_picklability(_PickleWorkflow(_DirtyDataset()), 4)
    assert dirty["picklable"] is False
    assert "num_workers=4" in dirty["hint"]

    # num_workers=0 -> the real run spawns nothing; the check must not fire.
    skipped = mcp_runner._check_worker_spawn_picklability(_PickleWorkflow(_DirtyDataset()), 0)
    assert skipped == {"requested_num_workers": 0, "checked": False}
