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


"""Loading a model from its classpath or YAML, and the wrapper the workflows run."""

import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any

from konfai import konfai_root
from konfai.data.data_manager import BatchSample
from konfai.data.patching import ModelPatch
from konfai.network.network.loaders import LRSchedulersLoader, OptimizerLoader, TargetCriterionsLoader
from konfai.network.network.network import MinimalModel, Network
from konfai.utils.clock import SweepClock
from konfai.utils.config import apply_config, config
from konfai.utils.errors import ConfigError
from konfai.utils.utils import get_module


@config("Model")
class ModelLoader:
    """Instantiate the root model graph declared in the active configuration."""

    def __init__(self, classpath: str = "default|segmentation.UNet.UNet") -> None:
        self.classpath = classpath

    def _yaml_path(self) -> Path | None:
        raw_path = self.classpath.split("|", maxsplit=1)[-1]
        if Path(raw_path).suffix.lower() not in {".yaml", ".yml"}:
            return None
        if self.classpath.startswith("default|"):
            # 'default|<Name>.yml' selects a model from the shipped catalog (konfai/models/yaml),
            # the declarative counterpart of 'default|segmentation.UNet.UNet' for Python classes. The
            # catalog is a flat directory, so the name must be a bare filename: reject any path
            # separator or '..' that would resolve outside the shipped catalog.
            import konfai.models.yaml as yaml_catalog

            if Path(raw_path).name != raw_path:
                raise ConfigError(
                    f"Invalid catalog model '{raw_path}'.",
                    "A 'default|<Name>.yml' name must be a bare filename from the shipped catalog "
                    "(no path separators). Use a plain path for a model file of your own.",
                )
            path = Path(str(yaml_catalog.__file__)).parent / raw_path
            if not path.is_file():
                available = sorted(entry.name for entry in path.parent.glob("*.yml"))
                raise ConfigError(
                    f"Unknown catalog model '{raw_path}'.",
                    f"Available catalog models: {available}. "
                    "Use 'default|<Name>.yml' for a shipped model or a plain path for your own file.",
                )
        else:
            path = Path(raw_path)
            config_file = os.environ.get("KONFAI_config_file")
            if not path.is_absolute() and config_file:
                path = Path(config_file).resolve().parent / path
        return path.resolve()

    def get_model(
        self,
        train: bool = True,
        konfai_args: str | None = None,
        konfai_without=[
            "optimizer",
            "schedulers",
            "nb_batch_per_step",
            "init_type",
            "init_gain",
        ],
    ) -> Network:
        if not konfai_args:
            konfai_args = f"{konfai_root()}.Model"
        yaml_path = self._yaml_path()
        if yaml_path is not None:
            from konfai.utils.model_builder import build_model_from_yaml

            name = yaml_path.stem

            def builder(
                parameters: dict[str, Any] | None = None,
                optimizer: OptimizerLoader | None = None,
                schedulers: dict[str, LRSchedulersLoader] | None = None,
                outputs_criterions: dict[str, TargetCriterionsLoader] | None = None,
                patch: ModelPatch | None = None,
            ) -> Network:
                return build_model_from_yaml(
                    yaml_path=yaml_path,
                    parameters=parameters,
                    optimizer=optimizer,
                    schedulers=schedulers,
                    outputs_criterions=outputs_criterions,
                    patch=patch,
                )

            model = apply_config(f"{konfai_args}.{name}")(builder)(konfai_without=konfai_without if not train else [])
            return model

        classpath = self.classpath
        # A config that references a built-in model by the absolute path konfai.models.<kind>.<file>:<Class>
        # keeps working: rewrite the prefix once to konfai.models.python, with a deprecation warning,
        # instead of failing on ModuleNotFoundError.
        if classpath.startswith("konfai.models.") and not classpath.startswith(
            ("konfai.models.python.", "konfai.models.yaml.")
        ):
            new_classpath = classpath.replace("konfai.models.", "konfai.models.python.", 1)
            warnings.warn(
                f"Model classpath '{classpath}' uses the pre-1.6.0 package layout; "
                f"use '{new_classpath}'. The old path is accepted for now but will be removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            classpath = new_classpath
        module, name = get_module(classpath, "konfai.models.python")
        cls = getattr(module, name)
        if not hasattr(cls, "_key"):
            konfai_args += "." + name

        model = apply_config(konfai_args)(cls)(konfai_without=konfai_without if not train else [])
        if not isinstance(model, Network):
            model = apply_config(konfai_args)(partial(MinimalModel, model))(
                konfai_without=[*konfai_without, "model"] if not train else []
            )
            model.set_name(name)

        return model


class Model:
    """High-level model wrapper combining networks, criteria, and execution state."""

    def __init__(self, model: Network) -> None:
        self.module = model

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def __call__(
        self,
        batch_sample: BatchSample,
        output_layers: list[str] = [],
        clock: SweepClock | None = None,
    ) -> Any:
        # Passed only when given: the predictor's ModelComposite has a forward of its own, without it.
        if clock is None:
            return self.module(batch_sample, output_layers)
        return self.module(batch_sample, output_layers, clock=clock)
