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

"""Standalone KonfAI Apps package."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import AbstractKonfAIApp, KonfAIApp, KonfAIAppClient, run_distributed_app, run_remote_job
    from .cli import add_common_konfai_apps, main_apps, main_apps_server
    from .transforms import DEFAULT_INFERENCE_MODEL_NAME, DEFAULT_INFERENCE_REPO_ID, KonfAIInference

# Resolved on first access: importing a submodule (Slicer imports ``konfai_apps.app_repository``) must not
# load torch and SimpleITK through ``app`` and ``transforms``.
_EXPORTS = {
    "AbstractKonfAIApp": "app",
    "KonfAIApp": "app",
    "KonfAIAppClient": "app",
    "run_distributed_app": "app",
    "run_remote_job": "app",
    "add_common_konfai_apps": "cli",
    "main_apps": "cli",
    "main_apps_server": "cli",
    "DEFAULT_INFERENCE_MODEL_NAME": "transforms",
    "DEFAULT_INFERENCE_REPO_ID": "transforms",
    "KonfAIInference": "transforms",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_EXPORTS))


__all__ = [
    "DEFAULT_INFERENCE_MODEL_NAME",
    "DEFAULT_INFERENCE_REPO_ID",
    "AbstractKonfAIApp",
    "KonfAIApp",
    "KonfAIAppClient",
    "KonfAIInference",
    "add_common_konfai_apps",
    "main_apps",
    "main_apps_server",
    "run_distributed_app",
    "run_remote_job",
]
