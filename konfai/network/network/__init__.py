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


"""The model system: routed module graphs, networks with their measures, the loaders that bind a config to them."""

from konfai.network.network.base import NetState as NetState
from konfai.network.network.base import PatchIndexed as PatchIndexed
from konfai.network.network.base import batched_step as batched_step
from konfai.network.network.loaders import CriterionsAttr as CriterionsAttr
from konfai.network.network.loaders import CriterionsLoader as CriterionsLoader
from konfai.network.network.loaders import LossSchedulersLoader as LossSchedulersLoader
from konfai.network.network.loaders import LRSchedulersLoader as LRSchedulersLoader
from konfai.network.network.loaders import OptimizerLoader as OptimizerLoader
from konfai.network.network.loaders import TargetCriterionsLoader as TargetCriterionsLoader
from konfai.network.network.loaders import build_configured_criterions as build_configured_criterions
from konfai.network.network.measure import Measure as Measure
from konfai.network.network.measure import _RunningNanMean as _RunningNanMean
from konfai.network.network.measure import _tail as _tail
from konfai.network.network.model import Model as Model
from konfai.network.network.model import ModelLoader as ModelLoader
from konfai.network.network.network import MinimalModel as MinimalModel
from konfai.network.network.network import ModuleArgsDict as ModuleArgsDict
from konfai.network.network.network import Network as Network
from konfai.network.network.network import OutputsGroup as OutputsGroup
from konfai.network.network.network import _channels_last as _channels_last
from konfai.network.network.network import _flat_downsampling as _flat_downsampling
from konfai.network.network.network import _leaf_spatial_stride as _leaf_spatial_stride

__all__ = [
    "CriterionsAttr",
    "CriterionsLoader",
    "LRSchedulersLoader",
    "LossSchedulersLoader",
    "Measure",
    "MinimalModel",
    "Model",
    "ModelLoader",
    "ModuleArgsDict",
    "NetState",
    "Network",
    "OptimizerLoader",
    "OutputsGroup",
    "PatchIndexed",
    "TargetCriterionsLoader",
    "batched_step",
    "build_configured_criterions",
]
