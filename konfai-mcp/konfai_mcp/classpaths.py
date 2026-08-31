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
    """The package a class is named by: a class defined in ``konfai.data.transform.intensity`` is
    ``konfai.data.transform``'s, because that package re-exports every public name of the module. A
    package that re-exports a few names of a public module (``konfai.data`` and ``Mean``) does not
    rename it."""
    module = importlib.import_module(cls.__module__)
    public = [name for name in vars(module) if not name.startswith("_") and getattr(module, name, None) is not None]
    parts = cls.__module__.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:depth])
        try:
            package = importlib.import_module(candidate)
        except ImportError:
            continue
        exported = set(getattr(package, "__all__", ()))
        defined = [name for name in public if getattr(vars(module)[name], "__module__", None) == cls.__module__]
        if defined and all(name in exported for name in defined) and getattr(package, cls.__name__, None) is cls:
            return candidate
    return cls.__module__


def public_classpath(cls: type) -> str:
    """``<public module>:<qualified name>``."""
    return f"{public_module(cls)}:{cls.__qualname__}"
