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


"""Patch-based access to a case: the grid it is cut on, the stages its chain runs, the sweep that streams it, the blend that reassembles it, and the manager that owns all of it."""

from konfai.data.patching.accumulate import Accumulator as Accumulator
from konfai.data.patching.accumulate import SlabAligner as SlabAligner
from konfai.data.patching.accumulate import SlabRegionStream as SlabRegionStream
from konfai.data.patching.accumulate import StreamingAccumulator as StreamingAccumulator
from konfai.data.patching.blend import Cosinus as Cosinus
from konfai.data.patching.blend import Gaussian as Gaussian
from konfai.data.patching.blend import Mean as Mean
from konfai.data.patching.blend import PathCombine as PathCombine
from konfai.data.patching.blend import Trim as Trim
from konfai.data.patching.blend import blend_axes as blend_axes
from konfai.data.patching.blend import blend_overlap as blend_overlap
from konfai.data.patching.budget import CASE_ELEMENT_BYTES as CASE_ELEMENT_BYTES
from konfai.data.patching.budget import FALLBACK_INFLIGHT_FACTOR as FALLBACK_INFLIGHT_FACTOR
from konfai.data.patching.budget import SWEEP_ENGINE_FLOOR_BYTES as SWEEP_ENGINE_FLOOR_BYTES
from konfai.data.patching.budget import SWEEP_SLAB_ROWS as SWEEP_SLAB_ROWS
from konfai.data.patching.budget import HeldMeter as HeldMeter
from konfai.data.patching.budget import device_capped_budget as device_capped_budget
from konfai.data.patching.budget import open_held_meter as open_held_meter
from konfai.data.patching.grid import DatasetPatch as DatasetPatch
from konfai.data.patching.grid import ModelPatch as ModelPatch
from konfai.data.patching.grid import Patch as Patch
from konfai.data.patching.manager import DatasetManager as DatasetManager
from konfai.data.patching.stage import AugmentedStage as AugmentedStage
from konfai.data.patching.stage import PatchReadPlan as PatchReadPlan
from konfai.data.patching.stage import Stage as Stage
from konfai.data.patching.sweep import SWEEP_CLOCK as SWEEP_CLOCK
from konfai.data.patching.sweep import BlockReads as BlockReads
from konfai.data.patching.sweep import RegionWriter as RegionWriter
from konfai.data.patching.sweep import SweepSegment as SweepSegment
from konfai.data.patching.sweep import save_destination as save_destination

__all__ = [
    "CASE_ELEMENT_BYTES",
    "FALLBACK_INFLIGHT_FACTOR",
    "SWEEP_CLOCK",
    "SWEEP_ENGINE_FLOOR_BYTES",
    "SWEEP_SLAB_ROWS",
    "Accumulator",
    "AugmentedStage",
    "BlockReads",
    "Cosinus",
    "DatasetManager",
    "DatasetPatch",
    "Gaussian",
    "HeldMeter",
    "Mean",
    "ModelPatch",
    "Patch",
    "PatchReadPlan",
    "PathCombine",
    "RegionWriter",
    "SlabAligner",
    "SlabRegionStream",
    "Stage",
    "StreamingAccumulator",
    "SweepSegment",
    "Trim",
    "blend_axes",
    "blend_overlap",
    "device_capped_budget",
    "open_held_meter",
    "save_destination",
]
