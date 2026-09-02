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


"""Prediction workflow entrypoints and orchestration for KonfAI.

The reductions ``Mean``, ``Median`` and ``Concat`` are re-exported: published configs name them as
``konfai.predictor.<Reduction>``."""

from konfai.data.reduction import Concat as Concat
from konfai.data.reduction import Mean as Mean
from konfai.data.reduction import Median as Median
from konfai.data.reduction import Reduction as Reduction
from konfai.predictor.ensemble import ModelComposite as ModelComposite
from konfai.predictor.output import PREDICTION_CLOCK as PREDICTION_CLOCK
from konfai.predictor.output import OutputDataset as OutputDataset
from konfai.predictor.output import OutputDatasetLoader as OutputDatasetLoader
from konfai.predictor.output import OutSameAsGroupDataset as OutSameAsGroupDataset
from konfai.predictor.workflow import Predictor as Predictor
from konfai.predictor.workflow import build_predict as build_predict
from konfai.predictor.workflow import predict as predict

__all__ = [
    "PREDICTION_CLOCK",
    "Concat",
    "Mean",
    "Median",
    "ModelComposite",
    "OutSameAsGroupDataset",
    "OutputDataset",
    "OutputDatasetLoader",
    "Predictor",
    "Reduction",
    "build_predict",
    "predict",
]
