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


"""Data loading: groups and their chains, the patch read order, the samples a loader yields, the sources each workflow reads."""

from konfai.data.data_manager.groups import Group as Group
from konfai.data.data_manager.groups import GroupMetric as GroupMetric
from konfai.data.data_manager.groups import GroupOut as GroupOut
from konfai.data.data_manager.groups import GroupTransform as GroupTransform
from konfai.data.data_manager.groups import GroupTransformMetric as GroupTransformMetric
from konfai.data.data_manager.groups import GroupTransformOut as GroupTransformOut
from konfai.data.data_manager.groups import _chains as _chains
from konfai.data.data_manager.groups import _check_patch_transform_invertible as _check_patch_transform_invertible
from konfai.data.data_manager.groups import _check_patch_transform_locality as _check_patch_transform_locality
from konfai.data.data_manager.groups import _check_patch_transform_shape as _check_patch_transform_shape
from konfai.data.data_manager.order import PatchReadOrder as PatchReadOrder
from konfai.data.data_manager.order import WindowedCaseSampler as WindowedCaseSampler
from konfai.data.data_manager.order import _interleaved_case_entries as _interleaved_case_entries
from konfai.data.data_manager.samples import _CACHE_ELEMENT_BYTES as _CACHE_ELEMENT_BYTES
from konfai.data.data_manager.samples import BatchDataItem as BatchDataItem
from konfai.data.data_manager.samples import BatchSample as BatchSample
from konfai.data.data_manager.samples import DataItem as DataItem
from konfai.data.data_manager.samples import DatasetIter as DatasetIter
from konfai.data.data_manager.samples import Sample as Sample
from konfai.data.data_manager.samples import _cache_worker_count as _cache_worker_count
from konfai.data.data_manager.samples import collate_konfai as collate_konfai
from konfai.data.data_manager.sources import Data as Data
from konfai.data.data_manager.sources import DataMetric as DataMetric
from konfai.data.data_manager.sources import DataPrediction as DataPrediction
from konfai.data.data_manager.sources import DataSources as DataSources
from konfai.data.data_manager.sources import DataTrain as DataTrain
from konfai.data.data_manager.sources import DataTransform as DataTransform
from konfai.data.data_manager.subset import PredictionSubset as PredictionSubset
from konfai.data.data_manager.subset import Subset as Subset

__all__ = [
    "BatchDataItem",
    "BatchSample",
    "Data",
    "DataItem",
    "DataMetric",
    "DataPrediction",
    "DataSources",
    "DataTrain",
    "DataTransform",
    "DatasetIter",
    "Group",
    "GroupMetric",
    "GroupOut",
    "GroupTransform",
    "GroupTransformMetric",
    "GroupTransformOut",
    "PatchReadOrder",
    "PredictionSubset",
    "Sample",
    "Subset",
    "WindowedCaseSampler",
    "collate_konfai",
]
