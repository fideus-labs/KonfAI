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

"""Data loading, patching, transform, and augmentation utilities for KonfAI.

This is the surface for driving KonfAI's data machinery from plain Python: no YAML, no environment
variables, no workflow. A chain of transforms applied to a dataset, out-of-core, is::

    from konfai.data import Dataset, DatasetManager, Write
    from konfai.data.transform import Clip   # the concrete stages stay in the module that defines them

    manager = DatasetManager(
        index=0, group_src="CT", group_dest="CT", name="CASE_000",
        dataset=Dataset("./Raw/", "mha"), patch=None,
        transforms=[Clip(0.0, 400.0), Write("./Out:omezarr")],
        data_augmentations_list=[],
    )
    streamed = manager.materialize()      # True when it never held the volume
    print(manager.stream_refusal())       # why not, naming the stage, when it is False

``patch=None`` on purpose: out-of-core is not the same thing as patched, and a whole-volume slab
sweep is both simpler and faster than a tiling at these sizes. Passing ``DatasetPatch()`` instead
would silently impose a 128³ tiling nobody asked for.

Writing a transform that streams needs :class:`LocalityKind` and :class:`PatchLocality`: a transform
that declares neither is correct but loads whole volumes, so they are exported here rather than
buried in the module that happens to define them.
"""

from konfai.data.data_manager import DataMetric, DataPrediction, DatasetIter, DataTrain, DataTransform
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.reduction import Concat, Mean, Median, Reduction, Vote
from konfai.data.transform import (
    Expand,
    LocalityKind,
    PatchLocality,
    Reduce,
    Save,
    Transform,
    TransformInverse,
    Write,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.ome_zarr import (
    append_ome_zarr_levels,
    create_ome_zarr_store,
    get_ome_zarr_info,
    is_displacement_field,
    read_ome_zarr_data_slice,
    write_ome_zarr,
)

__all__ = [
    "Attribute",
    "Concat",
    "DataMetric",
    "DataPrediction",
    "DataTrain",
    "DataTransform",
    "Dataset",
    "DatasetIter",
    "DatasetManager",
    "DatasetPatch",
    "Expand",
    "LocalityKind",
    "Mean",
    "Median",
    "PatchLocality",
    "Reduce",
    "Reduction",
    "Save",
    "Transform",
    "TransformInverse",
    "Vote",
    "Write",
    "append_ome_zarr_levels",
    "create_ome_zarr_store",
    "get_ome_zarr_info",
    "is_displacement_field",
    "read_ome_zarr_data_slice",
    "write_ome_zarr",
]
