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


"""Writing what passes through a chain to a dataset."""

import torch

from konfai.data.transform.base import Transform
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError
from konfai.utils.utils import split_path_spec


class Save(Transform):
    """Write the chain's state here, and become a source boundary.

    ``scale_factors`` writes an OME-NGFF PYRAMID instead of a single level: ``[4]`` adds a level 1 at
    a quarter of the extent per axis, ``[4, 4]`` a level 2 at a sixteenth. Every reader indexes a
    pyramid BY POSITION (``:omezarr@1`` is the second entry, not one named "1"), so the order is
    the contract, 0 finest. It applies on both write paths: assembled in memory, or region by region,
    where the levels are derived once the last region has landed.

    ``downsample_method`` names how the coarse levels are derived, and its default is
    ``DASK_BIN_SHRINK`` (block averaging), NOT ngff-zarr's own ``ITKWASM_GAUSSIAN``. Measured on a
    real volume, the Gaussian holds a 0.9998 correlation while crushing peak intensity by 20 %.
    """

    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    alters_values = False

    def __init__(
        self,
        dataset: str,
        group: str | None = None,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.group = group
        if scale_factors and any(int(factor) < 2 for factor in scale_factors):
            raise TransformError(
                f"'{type(self).__name__}' was given a scale factor below 2 in {list(scale_factors)}.",
                "A pyramid level shrinks its parent, so each factor is 2 or more: scale_factors: [4]"
                " writes one extra level at a quarter of the extent per axis.",
            )
        self.scale_factors = [int(factor) for factor in scale_factors] if scale_factors else None
        self.downsample_method = downsample_method
        self._destination: Dataset | None = None

    # WHOLE_VOLUME by declaration, yet the case may still stream: a Save whose cache exists is a
    # source boundary, and an unsatisfied Save with a streamable prefix is materialized slab by slab
    # first (DatasetManager._materialize_save). Only an unsweepable prefix loads the whole volume.

    @property
    def spec(self) -> tuple[str, str] | None:
        """``(filename, file_format)`` of the dataset this stage names, ``None`` when it names none.

        Parsed only: a parse-time check reads the path without probing the store on disk."""
        if not self.dataset:
            return None
        filename, _flag, file_format = split_path_spec(self.dataset, default_format="mha")
        return filename, file_format

    @property
    def destination(self) -> Dataset | None:
        """The :class:`Dataset` this stage writes into, ``None`` when it names none.

        Built once: the stage is shared by every case's manager, and constructing a Dataset probes
        the destination directory on disk."""
        if self._destination is None and (spec := self.spec) is not None:
            self._destination = Dataset(*spec, self.scale_factors, self.downsample_method)
        return self._destination

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor


class Write(Save):
    """A :class:`Save` that is a deliverable, not a cache.

    Same object, same boundary semantics, one difference that is the point: ``dataset`` has no
    default, so a bare ``Write:`` fails at config time instead of silently writing into the source
    tree (a bare ``Save:`` binds ``dataset`` to nothing and falls back to the manager's own
    dataset). The TRANSFORM workflow plans, resumes and reports on its ``Write`` stages; a ``Save``
    between them stays an opportunistic milestone, never written when a satisfied ``Write``
    downstream lets the boundary skip the whole prefix.
    """

    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    def __init__(
        self,
        dataset: str,
        group: str | None = None,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        if not dataset or not str(dataset).strip():
            raise TransformError(
                "'Write' needs a destination: its 'dataset' is empty.",
                "Declare where the deliverable lands, e.g. Write: {dataset: ./Out:omezarr}. For an"
                " opportunistic cache next to the source, use 'Save' instead.",
            )
        super().__init__(dataset, group, scale_factors, downsample_method)
