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


"""The classpath an agent should reference a class by: the outermost package that re-exports it."""

import importlib


def public_module(cls: type) -> str:
    """The shortest module path under which ``cls`` is reachable by its own name: a class defined in
    ``konfai.data.transform.intensity`` and re-exported by ``konfai.data.transform`` is named by the
    package, which is what configs and the documentation name."""
    module = cls.__module__
    parts = module.split(".")
    for depth in range(1, len(parts)):
        candidate = ".".join(parts[:depth])
        try:
            if getattr(importlib.import_module(candidate), cls.__name__, None) is cls:
                return candidate
        except ImportError:
            continue
    return module


def public_classpath(cls: type) -> str:
    """``<public module>:<qualified name>``."""
    return f"{public_module(cls)}:{cls.__qualname__}"
