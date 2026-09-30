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

"""Standalone KonfAI Apps package.

Its names are imported on first use: a light submodule (``konfai_apps.options``) stays light, without torch.
"""

import importlib
from typing import Any

_EXPORTS = {
    "AbstractKonfAIApp": ".app",
    "KonfAIApp": ".app",
    "KonfAIAppClient": ".app",
    "run_distributed_app": ".app",
    "run_remote_job": ".app",
    "add_common_konfai_apps": ".cli",
    "main_apps": ".cli",
    "main_apps_server": ".cli",
    "DEFAULT_INFERENCE_MODEL_NAME": ".transforms",
    "DEFAULT_INFERENCE_REPO_ID": ".transforms",
    "KonfAIInference": ".transforms",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
