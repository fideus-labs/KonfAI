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

"""Utility modules supporting configuration, datasets, ITK, DICOM, OME-Zarr, and runtime helpers."""

import importlib
from enum import Enum


class State(Enum):
    """Workflow state exported through the KonfAI process environment (``KONFAI_STATE``)."""

    TRAIN = "TRAIN"
    RESUME = "RESUME"
    PREDICTION = "PREDICTION"
    EVALUATION = "EVALUATION"
    TRANSFORM = "TRANSFORM"

    def __str__(self) -> str:
        return self.value


# ``dicom`` and ``ome_zarr`` import pydicom and dask/ngff-zarr at module level; resolved on first
# attribute access so ``import konfai`` does not pay for them.
_LAZY_SUBMODULES = ("dicom", "ome_zarr")

__all__ = ["State", *_LAZY_SUBMODULES]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_SUBMODULES))
