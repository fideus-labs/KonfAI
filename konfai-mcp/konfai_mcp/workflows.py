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

"""The single source of truth for workflow-kind identity.

Adding a workflow kind means: one ``WorkflowSpec`` entry here, extending the two ``Literal``
aliases below (a static ``Literal`` cannot be derived from data), and nothing else, every other
map (config filename, root key, runner command, capabilities class, retry tool) is derived from
this table. ``test_workflow_registry_drift`` pins the aliases to the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WorkflowSpec:
    kind: str  # canonical agent-facing name
    config_file: str  # session YAML filename
    root_key: str  # config root key
    module: str  # konfai module holding the workflow class
    class_name: str
    command: str  # KonfAI CLI command
    retry_tool: str  # MCP tool that relaunches this kind
    aliases: tuple[str, ...]  # accepted alternative spellings


WORKFLOW_SPECS: dict[str, WorkflowSpec] = {
    spec.kind: spec
    for spec in (
        WorkflowSpec(
            "train",
            "Config.yml",
            "Trainer",
            "konfai.trainer",
            "Trainer",
            "TRAIN",
            "run_train",
            ("trainer", "training"),
        ),
        WorkflowSpec(
            "prediction",
            "Prediction.yml",
            "Predictor",
            "konfai.predictor",
            "Predictor",
            "PREDICTION",
            "run_prediction",
            ("predict", "predictor"),
        ),
        WorkflowSpec(
            "evaluation",
            "Evaluation.yml",
            "Evaluator",
            "konfai.evaluator",
            "Evaluator",
            "EVALUATION",
            "run_evaluation",
            ("eval", "evaluate", "evaluator"),
        ),
        WorkflowSpec(
            "transform",
            "Transform.yml",
            "Transformer",
            "konfai.transformer",
            "Transformer",
            "TRANSFORM",
            "run_transform",
            ("transformer", "process"),
        ),
    )
}

# konfai-apps job kinds (no session YAML of their own) -> the tool that relaunches them.
APP_JOB_RETRY_TOOLS: dict[str, str] = {
    "infer": "run_app_infer",
    "finetune": "fine_tune_app",
    "evaluate": "run_app_evaluate",
    "uncertainty": "run_app_uncertainty",
    "pipeline": "run_app_pipeline",
}
APP_JOB_KINDS: tuple[str, ...] = tuple(APP_JOB_RETRY_TOOLS)
JOB_RETRY_TOOLS: dict[str, str] = {
    **{kind: spec.retry_tool for kind, spec in WORKFLOW_SPECS.items()},
    **APP_JOB_RETRY_TOOLS,
}
JOB_KINDS: tuple[str, ...] = tuple(WORKFLOW_SPECS) + APP_JOB_KINDS

# What a *finished* workflow job leads to. A trained model is worth nothing until it has predicted, and a
# prediction nothing until it is scored, so a done job names the step that follows it rather than ending
# the loop. Consumed by the job payload builder; every entry must be a registered tool (drift test).
WORKFLOW_DONE_NEXT: dict[str, list[str]] = {
    "train": ["run_prediction", "read_training_curves", "summarize_session"],
    "prediction": ["run_evaluation", "summarize_session"],
    "evaluation": ["get_run_metrics", "leaderboard", "compare_runs"],
    # A transformed dataset is an input, not a result: what follows is using it.
    "transform": ["inspect_dataset", "summarize_session"],
}


def workflow_choice_description(action: str) -> str:
    """``"<action>: train -> Config.yml (Trainer), ..."``: a tool's ``workflow`` parameter, from the table.

    Generated, because these descriptions are the only view an agent has of what a parameter accepts:
    a workflow missing from the prose is a workflow the agent never tries, whatever the type allows.
    """
    options = ", ".join(
        f"{kind} -> {spec.config_file} ({spec.root_key})" for kind, spec in sorted(WORKFLOW_SPECS.items())
    )
    return f"{action}: {options}."


# Static mirrors of the table for tool signatures; pinned to it by the drift test.
WorkflowKind = Literal["train", "prediction", "evaluation", "transform"]
JobKind = Literal[
    "train", "prediction", "evaluation", "transform", "infer", "finetune", "evaluate", "uncertainty", "pipeline"
]
