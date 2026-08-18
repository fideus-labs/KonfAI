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

"""Tests for the ``konfai.utils`` package itself: what ``import konfai`` loads, and the
submodules and names it resolves lazily."""

import importlib
import subprocess
import sys

import konfai.utils
import pytest


def test_import_konfai_does_not_load_the_imaging_optional_deps() -> None:
    """``konfai.utils.dicom`` and ``ome_zarr`` import pydicom and dask/zarr at module level; the
    package must not pull them in for ``import konfai`` (or for the torch-free ``State``)."""
    script = """
import sys
import konfai
from konfai.utils import State
heavy = ("torch", "pydicom", "dask", "zarr", "ngff_zarr", "konfai.utils.dicom", "konfai.utils.ome_zarr")
loaded = sorted(name for name in heavy if name in sys.modules)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("name", ["dicom", "ome_zarr"])
def test_utils_submodules_resolve_on_attribute_access(name: str) -> None:
    assert getattr(konfai.utils, name) is importlib.import_module(f"konfai.utils.{name}")
    assert name in dir(konfai.utils)


def test_utils_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute 'no_such_module'"):
        _ = konfai.utils.no_such_module


def test_state_is_the_one_runtime_re_exports() -> None:
    from konfai.utils.runtime import State as runtime_state

    assert runtime_state is konfai.utils.State
    assert [str(state) for state in konfai.utils.State] == ["TRAIN", "RESUME", "PREDICTION", "EVALUATION", "TRANSFORM"]
