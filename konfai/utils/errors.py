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

"""Shared exception types used across KonfAI runtime, datasets, and app tooling."""


class KonfAIError(Exception):
    """Base class for user-facing KonfAI exceptions."""

    TYPE: str | None = None

    def __str__(self) -> str:
        if not self.args:
            return "\n[Error]"

        if isinstance(getattr(self, "TYPE", None), str) and self.TYPE:
            type_error = self.TYPE
            messages = [str(m) for m in self.args]
        else:
            type_error = str(self.args[0])
            messages = [str(m) for m in self.args[1:]]

        if not messages:
            return f"\n[{type_error}]"

        head = f"[{type_error}] {messages[0]}"
        if len(messages) == 1:
            return "\n" + head
        return "\n" + head + "\n→\t" + "\n→\t".join(messages[1:])


class NamedKonfAIError(KonfAIError):
    """Base class for typed KonfAI errors whose label is fixed per subclass."""

    TYPE: str = "Error"


class EvaluatorError(NamedKonfAIError):
    TYPE = "Evaluator"


class ConfigError(NamedKonfAIError):
    TYPE = "Config"


class DatasetManagerError(NamedKonfAIError):
    TYPE = "DatasetManager"


class PatchError(NamedKonfAIError):
    TYPE = "Patch"


class MeasureError(NamedKonfAIError):
    TYPE = "Measure"


class TrainerError(NamedKonfAIError):
    TYPE = "Trainer"


class AugmentationError(NamedKonfAIError):
    TYPE = "Augmentation"


class PredictorError(NamedKonfAIError):
    TYPE = "Predictor"


class TransformError(NamedKonfAIError):
    TYPE = "Transform"


class ReductionError(NamedKonfAIError):
    TYPE = "Reduction"


class AppRepositoryError(NamedKonfAIError):
    TYPE = "App repository"


class AppMetadataError(NamedKonfAIError):
    TYPE = "Model metadata"


class KonfAIAppClientError(NamedKonfAIError):
    TYPE = "KonfAI App client"
