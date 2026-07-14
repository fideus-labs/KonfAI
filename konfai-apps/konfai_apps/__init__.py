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

from .app import AbstractKonfAIApp, KonfAIApp, KonfAIAppClient, run_distributed_app, run_remote_job
from .cli import add_common_konfai_apps, main_apps, main_apps_server

__all__ = [
    "AbstractKonfAIApp",
    "KonfAIApp",
    "KonfAIAppClient",
    "add_common_konfai_apps",
    "main_apps",
    "main_apps_server",
    "run_distributed_app",
    "run_remote_job",
]
