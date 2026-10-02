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

"""Every key a workflow binds is named by its configuration guide.

A key is spelled as its constructor parameter, or as the ``@config`` key of the class the parameter
takes (``model: ModelLoader`` is ``Model``). The guide's own page or the configuration index may name
it. The augmentation list and ``pretrained_from`` are documented in their reference pages.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from pathlib import Path

import pytest
from konfai.data.data_manager.groups import GroupTransform
from konfai.data.data_manager.sources import DataMetric, DataPrediction, DataTrain, DataTransform
from konfai.data.patching.grid import DatasetPatch, ModelPatch
from konfai.evaluator import Evaluator
from konfai.network.network.loaders import CriterionsAttr, CriterionsLoader, OptimizerLoader, TargetCriterionsLoader
from konfai.network.network.model import ModelLoader
from konfai.predictor.output import OutputDataset
from konfai.predictor.workflow import Predictor
from konfai.trainer import EarlyStopping, Trainer
from konfai.transformer import Transformer

GUIDES = Path(__file__).resolve().parents[2] / "docs" / "source" / "config_guide"
BOUND = {
    "training.md": [
        Trainer,
        ModelLoader,
        OptimizerLoader,
        CriterionsAttr,
        CriterionsLoader,
        TargetCriterionsLoader,
        DataTrain,
        DatasetPatch,
        ModelPatch,
        GroupTransform,
        EarlyStopping,
    ],
    "prediction.md": [Predictor, OutputDataset, DataPrediction, DatasetPatch],
    "evaluation.md": [Evaluator, DataMetric],
    "transform.md": [Transformer, DataTransform],
}


def _named(text: str) -> set[str]:
    return set(re.findall(r"`([A-Za-z_]\w*)`", text)) | set(re.findall(r"^\s*([A-Za-z_]\w*):", text, re.M))


def _key(parameter: inspect.Parameter) -> str:
    annotation = parameter.annotation
    members = typing.get_args(annotation) if isinstance(annotation, types.UnionType) else (annotation,)
    return next((member._key for member in members if hasattr(member, "_key")), parameter.name)


@pytest.mark.parametrize("guide", sorted(BOUND))
def test_a_guide_names_every_key_its_workflow_binds(guide: str) -> None:
    named = _named((GUIDES / guide).read_text(encoding="utf-8")) | _named(
        (GUIDES / "index.md").read_text(encoding="utf-8")
    )
    missing = {
        f"{cls.__name__}.{_key(parameter)}"
        for cls in BOUND[guide]
        for parameter in list(inspect.signature(cls.__init__).parameters.values())[1:]
        if parameter.kind is not parameter.VAR_KEYWORD and _key(parameter) not in named
    }
    assert not missing, f"{guide} does not name {sorted(missing)}"
