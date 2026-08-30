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
from konfai.data.patching.budget import _PLATEAU_READ_MARGIN as _PLATEAU_READ_MARGIN
from konfai.data.patching.budget import _STREAM_STAT_KEYS as _STREAM_STAT_KEYS
from konfai.data.patching.budget import _STREAM_STATS as _STREAM_STATS
from konfai.data.patching.budget import _SWEEP_ELEMENT_BYTES as _SWEEP_ELEMENT_BYTES
from konfai.data.patching.budget import _SWEEP_MAX_DEPTH as _SWEEP_MAX_DEPTH
from konfai.data.patching.budget import _SWEEP_SLAB_ROWS_DEVICE as _SWEEP_SLAB_ROWS_DEVICE
from konfai.data.patching.budget import _SWEEP_TILE_MARGIN as _SWEEP_TILE_MARGIN
from konfai.data.patching.budget import _UNRESOLVED as _UNRESOLVED
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
from konfai.data.patching.grid import _PatchGrid as _PatchGrid
from konfai.data.patching.manager import DatasetManager as DatasetManager
from konfai.data.patching.stage import _MAX_HALO_FRACTION as _MAX_HALO_FRACTION
from konfai.data.patching.stage import AugmentedStage as AugmentedStage
from konfai.data.patching.stage import PatchReadPlan as PatchReadPlan
from konfai.data.patching.stage import Stage as Stage
from konfai.data.patching.stage import _drawn_from as _drawn_from
from konfai.data.patching.stage import _halo_radii as _halo_radii
from konfai.data.patching.stage import _HaloPull as _HaloPull
from konfai.data.patching.stage import _is_draw as _is_draw
from konfai.data.patching.stage import _ReadStagePlan as _ReadStagePlan
from konfai.data.patching.stage import _RemapPull as _RemapPull
from konfai.data.patching.stage import _spatial as _spatial
from konfai.data.patching.stage import _stage_name as _stage_name
from konfai.data.patching.sweep import SWEEP_CLOCK as SWEEP_CLOCK
from konfai.data.patching.sweep import BlockReads as BlockReads
from konfai.data.patching.sweep import RegionWriter as RegionWriter
from konfai.data.patching.sweep import SweepSegment as SweepSegment
from konfai.data.patching.sweep import _channel_first_block as _channel_first_block
from konfai.data.patching.sweep import _cubic_tile as _cubic_tile
from konfai.data.patching.sweep import _HostLanding as _HostLanding
from konfai.data.patching.sweep import _open_sweep_stream as _open_sweep_stream
from konfai.data.patching.sweep import _PatchStreamSource as _PatchStreamSource
from konfai.data.patching.sweep import _PendingSweep as _PendingSweep
from konfai.data.patching.sweep import _plateau_rows as _plateau_rows
from konfai.data.patching.sweep import _pull_block_spans as _pull_block_spans
from konfai.data.patching.sweep import _pull_block_voxels as _pull_block_voxels
from konfai.data.patching.sweep import _ReadAhead as _ReadAhead
from konfai.data.patching.sweep import _shares_h5_file as _shares_h5_file
from konfai.data.patching.sweep import _span_voxels as _span_voxels
from konfai.data.patching.sweep import _stage_failure as _stage_failure
from konfai.data.patching.sweep import _stage_failures_explained as _stage_failures_explained
from konfai.data.patching.sweep import _sweep_header as _sweep_header
from konfai.data.patching.sweep import _sweep_pipeline_depth as _sweep_pipeline_depth
from konfai.data.patching.sweep import _sweep_resident_regions as _sweep_resident_regions
from konfai.data.patching.sweep import _sweep_targets as _sweep_targets
from konfai.data.patching.sweep import _SweepMember as _SweepMember
from konfai.data.patching.sweep import _torch_dtype_hint as _torch_dtype_hint
from konfai.data.patching.sweep import _WriteBehind as _WriteBehind
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
