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

"""The package ``__init__`` files re-export the public surface only.

A private name reachable through a package path invites external code onto internals, and a
re-export of a module global its home module rebinds (``_rank_pool``) is a frozen snapshot that
goes stale after the first rebind. Private names import from their defining submodule."""

import importlib

import pytest

PACKAGES = [
    "konfai.data.augmentation",
    "konfai.data.data_manager",
    "konfai.data.patching",
    "konfai.data.transform",
    "konfai.metric.measure",
    "konfai.network.network",
    "konfai.predictor",
    "konfai.utils.dataset",
    "konfai.utils.runtime",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_init_re_exports_no_private_names(package: str) -> None:
    module = importlib.import_module(package)
    private = sorted(n for n in vars(module) if n.startswith("_") and not n.startswith("__"))
    assert not private, f"{package} re-exports private names: {private}; import them from their defining submodule"
