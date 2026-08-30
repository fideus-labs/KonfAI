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


"""The config-bound loaders of optimizers, schedulers and criteria."""

import importlib
from collections.abc import Callable, Iterator
from typing import Any

import torch

from konfai import konfai_root
from konfai.metric.schedulers import Scheduler
from konfai.network.network.base import batched_step
from konfai.utils.config import apply_config, config
from konfai.utils.errors import TrainerError
from konfai.utils.utils import get_module


@config("optimizer")
class OptimizerLoader:
    """Configuration-aware factory for PyTorch optimizers."""

    def __init__(self, name: str = "AdamW") -> None:
        self.name = name

    def get_optimizer(self, key: str, parameter: Iterator[torch.nn.parameter.Parameter]) -> torch.optim.Optimizer:
        parameters = list(parameter)
        optimizer_class = getattr(importlib.import_module("torch.optim"), self.name)
        return apply_config(f"{konfai_root()}.Model.{key}.optimizer")(batched_step(optimizer_class, parameters))(
            parameters
        )


class LRSchedulersLoader:
    """Configuration-aware factory for learning-rate schedulers."""

    def __init__(self, nb_step: int = 0) -> None:
        self.nb_step = nb_step

    def getschedulers(
        self, key: str, scheduler_classname: str, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        for m in ["torch.optim.lr_scheduler", "konfai.metric.schedulers"]:
            module, name = get_module(scheduler_classname, m)
            if hasattr(module, name):
                return apply_config(f"{konfai_root()}.Model.{key}.schedulers.{scheduler_classname}")(
                    getattr(module, name)
                )(optimizer)
        raise TrainerError(
            f"Unknown scheduler {scheduler_classname}, tried importing from: 'torch.optim.lr_scheduler' and "
            "'konfai.metric.schedulers', but no valid match was found. "
            "Check your YAML config or scheduler name spelling."
        )


class LossSchedulersLoader:
    """Factory for scalar schedulers attached to losses and metrics."""

    def __init__(self, nb_step: int = 0) -> None:
        self.nb_step = nb_step

    def getschedulers(self, key: str, scheduler_classname: str) -> torch.optim.lr_scheduler._LRScheduler:
        return apply_config(f"{key}.{scheduler_classname}")(
            getattr(importlib.import_module("konfai.metric.schedulers"), scheduler_classname)
        )()


def build_configured_criterions(
    criterions_loader: dict[str, Any],
    config_key_prefix: str,
    configure_attr: Callable[[str, Any], None] | None = None,
) -> dict[torch.nn.Module, Any]:
    """Instantiate the criteria a ``criterions_loader`` mapping names, each bound from
    ``<config_key_prefix>.criterions_loader.<classpath>``; ``configure_attr`` (classpath, attribute)
    enriches the attribute before its criterion is built."""
    criterions = {}
    for module_classpath, criterions_attr in criterions_loader.items():
        module, name = get_module(module_classpath, "konfai.metric.measure")
        if configure_attr is not None:
            configure_attr(module_classpath, criterions_attr)
        criterions[
            apply_config(f"{config_key_prefix}.criterions_loader.{module_classpath}")(getattr(module, name))()
        ] = criterions_attr
    return criterions


class CriterionsAttr:
    """Metadata describing how a criterion is applied within the model graph."""

    def __init__(
        self,
        schedulers: dict[str, LossSchedulersLoader] = {"default|Constant": LossSchedulersLoader(0)},
        is_loss: bool = True,
        group: int = 0,
        start: int = 0,
        stop: int | None = None,
        accumulation: bool = False,
    ) -> None:
        self.schedulersLoader = schedulers
        self.is_loss = is_loss
        self.start = start
        self.stop = stop
        self.group = group
        self.accumulation = accumulation
        self.schedulers: dict[Scheduler, int] = {}


class CriterionsLoader:
    """Instantiate the criteria attached to one output/target pair."""

    def __init__(
        self,
        criterions_loader: dict[str, CriterionsAttr] = {"default|torch:nn:CrossEntropyLoss|Dice|NCC": CriterionsAttr()},
    ) -> None:
        self.criterions_loader = criterions_loader

    def get_criterions(
        self, model_classname: str, output_group: str, target_group: str
    ) -> dict[torch.nn.Module, CriterionsAttr]:
        def configure_attr(module_classpath: str, criterions_attr: CriterionsAttr) -> None:
            criterions_attr.schedulers = {}
            for (
                scheduler_classname,
                schedulers,
            ) in criterions_attr.schedulersLoader.items():
                criterions_attr.schedulers[
                    schedulers.getschedulers(
                        f"{konfai_root()}.Model.{model_classname}.outputs_criterions.{output_group}"
                        f".targets_criterions.{target_group}"
                        f".criterions_loader.{module_classpath}.schedulers",
                        scheduler_classname,
                    )
                ] = schedulers.nb_step

        return build_configured_criterions(
            self.criterions_loader,
            (
                f"{konfai_root()}.Model.{model_classname}.outputs_criterions."
                f"{output_group}.targets_criterions.{target_group}"
            ),
            configure_attr=configure_attr,
        )


class TargetCriterionsLoader:
    """Resolve criteria for all targets associated with one model output."""

    def __init__(
        self,
        targets_criterions: dict[str, CriterionsLoader] = {"Labels": CriterionsLoader()},
    ) -> None:
        self.targets_criterions = targets_criterions

    def get_targets_criterions(
        self, output_group: str, model_classname: str
    ) -> dict[str, dict[torch.nn.Module, CriterionsAttr]]:
        targets_criterions = {}
        for target_group, criterions_loader in self.targets_criterions.items():
            targets_criterions[target_group] = criterions_loader.get_criterions(
                model_classname, output_group, target_group
            )
        return targets_criterions
