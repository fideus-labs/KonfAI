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

"""WORKFLOW_SPECS is the single source of workflow-kind identity: the static Literal aliases and
every derived registry must match it, so adding a kind cannot silently miss a map."""

from pathlib import Path
from typing import get_args

import pytest
from konfai_mcp import capabilities, experiment_state, runner, server, server_support
from konfai_mcp.workflows import (
    APP_JOB_KINDS,
    JOB_KINDS,
    JOB_RETRY_TOOLS,
    WORKFLOW_SPECS,
    JobKind,
    WorkflowKind,
)


def test_literal_aliases_match_the_table() -> None:
    # A job kind is either a workflow kind or a konfai-apps kind (run_app_* / fine_tune_app), never else.
    assert set(get_args(WorkflowKind)) == set(WORKFLOW_SPECS)
    assert set(get_args(JobKind)) == set(JOB_KINDS)
    assert set(JOB_KINDS) == set(WORKFLOW_SPECS) | set(APP_JOB_KINDS)


def test_derived_registries_come_from_the_table() -> None:
    assert server_support.WORKFLOW_CONFIG_FILES == {k: s.config_file for k, s in WORKFLOW_SPECS.items()}
    assert server_support.WORKFLOW_ROOT_KEYS == {k: s.root_key for k, s in WORKFLOW_SPECS.items()}
    assert server.WORKFLOWS == set(WORKFLOW_SPECS)
    assert capabilities._WORKFLOW_ROOTS == {k: (s.root_key, s.module, s.class_name) for k, s in WORKFLOW_SPECS.items()}
    assert set(JOB_RETRY_TOOLS) == set(JOB_KINDS)
    assert experiment_state._LAUNCHER_CONFIG == {s.retry_tool: s.config_file for s in WORKFLOW_SPECS.values()}


def test_an_example_holding_every_workflow_config_loads_every_workflow(tmp_path: Path) -> None:
    template = tmp_path / "Everything"
    template.mkdir()
    for spec in WORKFLOW_SPECS.values():
        (template / spec.config_file).write_text(f"{spec.root_key}: {{}}\n", encoding="utf-8")

    assert set(server_support.load_template_configs(tmp_path, "Everything")) == set(WORKFLOW_SPECS)


def test_every_workflow_command_builds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for builder in ("build_train", "build_predict", "build_evaluate", "build_transform"):
        monkeypatch.setattr(runner, builder, lambda builder=builder, **_: builder)

    for spec in WORKFLOW_SPECS.values():
        assert runner._build_workflow(spec.command, str(tmp_path / spec.config_file)).startswith("build_")


def test_table_values_pin_the_konfai_contract() -> None:
    """The KonfAI-facing values themselves, so a table edit is a visible, deliberate act."""
    assert WORKFLOW_SPECS["train"].config_file == "Config.yml"
    assert WORKFLOW_SPECS["train"].root_key == "Trainer"
    assert WORKFLOW_SPECS["train"].command == "TRAIN"
    assert WORKFLOW_SPECS["prediction"].config_file == "Prediction.yml"
    assert WORKFLOW_SPECS["prediction"].root_key == "Predictor"
    assert WORKFLOW_SPECS["evaluation"].config_file == "Evaluation.yml"
    assert WORKFLOW_SPECS["evaluation"].root_key == "Evaluator"
    assert capabilities._WORKFLOW_ALIASES["trainer"] == "train"
    assert capabilities._WORKFLOW_ALIASES["eval"] == "evaluation"
