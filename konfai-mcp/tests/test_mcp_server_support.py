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

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import konfai_mcp.server_support as server_support  # noqa: E402


def test_workspace_layout_resolves_paths_inside_session_workspace(tmp_path: Path) -> None:
    layout = server_support.WorkspaceLayout(tmp_path)
    workspace = layout.ensure_session_workspace()

    assert workspace == tmp_path / "sessions" / "default"
    assert layout.config_path("train") == workspace / "Config.yml"
    assert layout.resolve_workspace_relative_path("nested/file.txt") == workspace / "nested" / "file.txt"
    assert layout.job_state_path("job1") == workspace / ".konfai_mcp" / "jobs" / "job1" / "job.json"

    with pytest.raises(ValueError, match="escapes"):
        layout.resolve_workspace_relative_path("../outside.txt")


def test_workspace_layout_ignores_top_level_dataset_dirs(tmp_path: Path) -> None:
    layout = server_support.WorkspaceLayout(tmp_path)
    dataset_dir = tmp_path / "Dataset"
    case_dir = dataset_dir / "CASE_001"
    case_dir.mkdir(parents=True)
    (case_dir / "CT.mha").write_text("", encoding="utf-8")

    assert layout.available_sessions() == []
    assert layout.workspace_dir() == tmp_path / "sessions" / "default"
    assert layout.session_workspace_exists() is False


def test_workspace_layout_uses_session_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KONFAI_MCP_SESSION", "challenge round 1")
    layout = server_support.WorkspaceLayout(tmp_path)

    assert layout.current_session == "challenge_round_1"
    assert layout.session_dir() == tmp_path / "sessions" / "challenge_round_1"


def test_template_helpers_load_existing_example() -> None:
    examples_root = Path(__file__).resolve().parents[2] / "examples"

    assert "Synthesis" in server_support.available_templates(examples_root)
    summary = server_support.template_summary(examples_root, "Synthesis", {"train", "prediction", "evaluation"})
    assert "Config.yml" in summary["yaml_files"]

    configs = server_support.load_template_configs(examples_root, "Synthesis")
    groups = server_support.template_groups(configs)
    assert "train" in groups
    assert "MR" in groups["train"]


def test_classpath_local_file_resolution() -> None:
    assert server_support._classpath_local_file("UNet.yml") == "UNet.yml"
    assert server_support._classpath_local_file("Model:Gan") == "Model.py"
    assert server_support._classpath_local_file("Model.Generator") == "Model.py"
    # Importable konfai paths resolve to a name that does not exist in a template dir.
    assert server_support._classpath_local_file("segmentation.UNet.UNet") == "segmentation.py"


def test_copy_template_subset_pulls_in_referenced_yaml_model(tmp_path: Path) -> None:
    examples_root = Path(__file__).resolve().parents[2] / "examples"
    template = server_support.template_dir(examples_root, "Segmentation")

    copied, skipped_python = server_support.copy_template_subset(
        tmp_path,
        template,
        overwrite=True,
        include_python=False,
        workflows=["train"],
    )

    # The declarative ``UNet.yml`` model is referenced by Config.yml and must come along,
    # even without include_python / include_support_files.
    assert "Config.yml" in copied
    assert "UNet.yml" in copied
    assert (tmp_path / "UNet.yml").exists()
    assert "Evaluation.yml" not in copied
    assert skipped_python == []  # a fully-declarative example seeds runnable with no warning


def test_copy_template_subset_keeps_python_models_opt_in(tmp_path: Path) -> None:
    examples_root = Path(__file__).resolve().parents[2] / "examples"
    template = server_support.template_dir(examples_root, "Synthesis")

    copied, skipped_python = server_support.copy_template_subset(
        tmp_path,
        template,
        overwrite=True,
        include_python=False,
        workflows=["train"],
    )

    # Synthesis references a local ``Model.py``; Python models stay opt-in, but the omission
    # must be REPORTED -- the seeded config cannot resolve without them.
    assert "Config.yml" in copied
    assert "Model.py" not in copied
    assert not (tmp_path / "Model.py").exists()
    assert "Model.py" in skipped_python


def test_mcp_package_metadata_exposes_entrypoint() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")

    assert 'name = "konfai-mcp"' in content
    assert 'konfai-mcp = "konfai_mcp:main"' in content


def test_patch_transforms_lint_skips_evaluator_configs() -> None:
    # Evaluator groups_dest entries bind to GroupTransformMetric (no patch_transforms parameter),
    # so the missing-patch_transforms trap does not exist there and must not be reported.
    evaluator = {"Evaluator": {"metrics": {"PRED": {"targets_criterions": {"SEG": {"groups_dest": {"SEG": {}}}}}}}}
    assert server_support._lint_config_data(evaluator) == []

    trainer = {"Trainer": {"Dataset": {"groups_src": {"CT": {"groups_dest": {"CT": {"transforms": None}}}}}}}
    warnings = server_support._lint_config_data(trainer)
    assert [w["code"] for w in warnings] == ["missing_patch_transforms"]


def test_prediction_default_outputs_criterions_lint() -> None:
    # A Predictor Model without an explicit outputs_criterions binds KonfAI's default, which references
    # target group 'Labels'; if the dataset does not load 'Labels', prediction raises MeasureError.
    trap = {
        "Predictor": {
            "Model": {"classpath": "Model:MyNet", "MyNet": {}},
            "Dataset": {"groups_src": {"CT": {"groups_dest": {"CT": {}}}}},
        }
    }
    # Membership, not equality: an unrelated lint (missing_patch_transforms) also fires on these groups.
    assert "prediction_default_outputs_criterions" in {w["code"] for w in server_support._lint_config_data(trap)}

    # Silent when the trap cannot occur: explicit outputs_criterions, a .yml model builder (defaults to
    # None), a loaded 'Labels' group, or a non-prediction root.
    dataset = {"groups_src": {"CT": {"groups_dest": {"CT": {}}}}}
    for safe in (
        {"Predictor": {"Model": {"classpath": "Model:MyNet", "MyNet": {"outputs_criterions": {}}}, "Dataset": dataset}},
        {"Predictor": {"Model": {"classpath": "UNet.yml", "UNet": {}}, "Dataset": dataset}},
        {
            "Predictor": {
                "Model": {"classpath": "Model:MyNet", "MyNet": {}},
                "Dataset": {"groups_src": {"CT": {"groups_dest": {"Labels": {}}}}},
            }
        },
        {"Trainer": {"Model": {"classpath": "Model:MyNet", "MyNet": {}}, "Dataset": dataset}},
    ):
        assert "prediction_default_outputs_criterions" not in {
            w["code"] for w in server_support._lint_config_data(safe)
        }


def test_summarize_signature_resolves_config_reference_form() -> None:
    # The YAML classpath 'segmentation.UNet.UNet' is not importable as-is; the resolver retries via the
    # builtin mapping instead of failing with "No module named 'segmentation'".
    resolved = server_support.summarize_classpath_signature("segmentation.UNet.UNet")
    assert resolved["ok"] is True
    assert resolved["resolved_classpath"] == "konfai.models.python.segmentation.UNet:UNet"
    assert resolved["parameters"]
    # A genuinely-unknown model still fails with the original limitation.
    bad = server_support.summarize_classpath_signature("segmentation.UNet.NoSuchModel")
    assert bad["ok"] is False


def test_round_floats_trims_precision_and_is_opt_outable(monkeypatch) -> None:
    payload = {"mae": 7717.17822265625, "n": [0.00046867796724351746], "count": 3, "flag": True, "name": "x"}
    rounded = server_support.round_floats(payload)
    assert rounded["mae"] == 7717.18
    assert rounded["n"][0] == 0.000468678
    assert rounded["count"] == 3 and rounded["flag"] is True and rounded["name"] == "x"

    monkeypatch.setattr(server_support, "ROUND_SIGNIFICANT_FIGURES", None)
    assert server_support.round_floats({"mae": 7717.17822265625})["mae"] == 7717.17822265625


def test_inspect_signature_summarizes_a_catalog_yaml_model() -> None:
    # The agent flow: list_components surfaces 'default|<Name>.yml' -> inspect_object_signature must
    # explain it (hyperparameters, loss-attachable named outputs, how to adapt) without importing it
    # as a Python module and without instantiating the model.
    sig = server_support.summarize_classpath_signature("default|VGG16.yml")
    assert sig["ok"] is True
    assert sig["source"] == "yaml_catalog"
    assert [p["name"] for p in sig["parameters"]] == ["dim", "in_channels", "widths"]
    assert sig["terminal_leaves"] == [f"Block_{i}:Out" for i in range(5)]
    assert "outputs_criterions" in sig["how_to_adapt"]
    assert "write_session_file" in sig["how_to_adapt"]  # the structural-edit path is advertised
    assert sig["yaml_content"].startswith("#")

    # Deep-supervision heads resolve to their producing leaf (the outputs_criterions key).
    unet = server_support.summarize_classpath_signature("default|UNet.yml")
    assert "UNetBlock_0:Head:Argmax" in unet["terminal_leaves"]

    with pytest.raises(ValueError, match="Available catalog models"):
        server_support.summarize_classpath_signature("default|DoesNotExist.yml")
