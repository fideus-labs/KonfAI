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
from konfai.predictor.ensemble import _colocate_loaded_modules as _colocate_loaded_modules
from konfai.predictor.loop import _DESCRIPTION_EVERY as _DESCRIPTION_EVERY
from konfai.predictor.loop import _prediction_report as _prediction_report
from konfai.predictor.loop import _Predictor as _Predictor
from konfai.predictor.output import _STREAM_WORTH_MIN_FRACTION as _STREAM_WORTH_MIN_FRACTION
from konfai.predictor.output import PREDICTION_CLOCK as PREDICTION_CLOCK
from konfai.predictor.output import OutputDataset as OutputDataset
from konfai.predictor.output import OutputDatasetLoader as OutputDatasetLoader
from konfai.predictor.output import OutSameAsGroupDataset as OutSameAsGroupDataset
from konfai.predictor.output import _AsyncWriter as _AsyncWriter
from konfai.predictor.output import _FinalizeStage as _FinalizeStage
from konfai.predictor.output import _RegionState as _RegionState
from konfai.predictor.output import _slab_context as _slab_context
from konfai.predictor.output import _StreamPlan as _StreamPlan
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
