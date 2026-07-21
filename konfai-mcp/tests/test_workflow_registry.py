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

from typing import get_args

import pytest
from konfai_mcp import capabilities, server, server_support
from konfai_mcp.workflows import (
    APP_JOB_KINDS,
    JOB_KINDS,
    JOB_RETRY_TOOLS,
    WORKFLOW_SPECS,
    JobKind,
    WorkflowKind,
)


def test_literal_aliases_match_the_table() -> None:
    assert set(get_args(WorkflowKind)) == set(WORKFLOW_SPECS)
    assert set(get_args(JobKind)) == set(JOB_KINDS)
    assert set(JOB_KINDS) == set(WORKFLOW_SPECS) | set(APP_JOB_KINDS)


def test_derived_registries_come_from_the_table() -> None:
    assert server_support.WORKFLOW_CONFIG_FILES == {k: s.config_file for k, s in WORKFLOW_SPECS.items()}
    assert server_support.WORKFLOW_ROOT_KEYS == {k: s.root_key for k, s in WORKFLOW_SPECS.items()}
    assert server.WORKFLOWS == set(WORKFLOW_SPECS)
    assert capabilities._WORKFLOW_ROOTS == {k: (s.root_key, s.module, s.class_name) for k, s in WORKFLOW_SPECS.items()}
    assert set(JOB_RETRY_TOOLS) == set(JOB_KINDS)


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


def test_app_job_devices_register_the_real_default_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """konfai-apps defaults an omitted gpu to every visible CUDA device; the job registry must
    record that reservation, not 'cpu', or concurrent scheduling double-books the GPUs."""
    monkeypatch.setattr(server, "konfai_pkg", server.konfai_pkg)
    monkeypatch.setattr(server.konfai_pkg, "cuda_visible_devices", lambda: [0, 1], raising=False)
    assert server._app_job_devices(None, None) == ["0", "1"]
    assert server._app_job_devices([], None) == ["cpu"]  # explicit empty list forces CPU
    assert server._app_job_devices([1], None) == ["1"]
