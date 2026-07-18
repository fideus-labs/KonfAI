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

"""Model graph composition, routing, and optimization helpers for KonfAI."""

import importlib
import inspect
import os
import warnings
from abc import ABC
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

from konfai.utils.dataset import Attribute

try:
    from typing import Self  # type: ignore[attr-defined]  # Python ≥ 3.11
except ImportError:
    from typing_extensions import Self  # Python ≤ 3.10

import numpy as np
import torch
from torch._jit_internal import _copy_to_script_wrapper
from torch.utils.checkpoint import checkpoint

from konfai import konfai_root
from konfai.data.data_manager import BatchSample
from konfai.data.patching import Accumulator, ModelPatch
from konfai.metric.schedulers import Scheduler
from konfai.utils.config import apply_config, config
from konfai.utils.errors import ConfigError, MeasureError, TrainerError
from konfai.utils.runtime import State, get_device, get_gpu_memory
from konfai.utils.utils import get_module


class NetState(Enum):
    """Execution state of a network inside KonfAI workflows."""

    TRAIN = (0,)
    PREDICTION = 1


class PatchIndexed:
    """Track progress while consuming the patches produced by a :class:`ModelPatch`."""

    def __init__(self, patch: ModelPatch, index: int) -> None:
        self.patch = patch
        self.index = index

    def is_full(self) -> bool:
        return len(self.patch.get_patch_slices(0)) == self.index


@config("optimizer")
class OptimizerLoader:
    """Configuration-aware factory for PyTorch optimizers."""

    def __init__(self, name: str = "AdamW") -> None:
        self.name = name

    def get_optimizer(self, key: str, parameter: Iterator[torch.nn.parameter.Parameter]) -> torch.optim.Optimizer:
        return apply_config(f"{konfai_root()}.Model.{key}.optimizer")(
            getattr(importlib.import_module("torch.optim"), self.name)
        )(parameter)


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
    configure_attr: Callable[[str, Any, Any], None] | None = None,
) -> dict[torch.nn.Module, Any]:
    """
    Instantiate criteria from a KonfAI config subtree.

    Parameters
    ----------
    criterions_loader : dict[str, Any]
        Mapping from criterion classpath to criterion attributes.
    config_key_prefix : str
        Config subtree preceding ``.criterions_loader.<classpath>``.
    configure_attr : Callable[[str, Any, Any], None] | None, optional
        Optional callback used to enrich the attribute object with runtime
        metadata before the criterion is instantiated. It receives the
        criterion classpath, the attribute object, and the imported module.

    Returns
    -------
    dict[torch.nn.Module, Any]
        Instantiated criteria mapped to their configuration attributes.
    """
    criterions = {}
    for module_classpath, criterions_attr in criterions_loader.items():
        module, name = get_module(module_classpath, "konfai.metric.measure")
        if configure_attr is not None:
            configure_attr(module_classpath, criterions_attr, module)
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
        self.isTorchCriterion = True
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
        def configure_attr(module_classpath: str, criterions_attr: CriterionsAttr, module: Any) -> None:
            criterions_attr.isTorchCriterion = module.__name__.startswith("torch")
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


class Measure:
    """Collect, validate, and aggregate losses or metrics across model outputs."""

    class Loss:
        def __init__(
            self,
            name: str,
            output_group: str,
            target_group: str,
            group: int,
            is_loss: bool,
            accumulation: bool,
        ) -> None:
            self.name = name
            self.is_loss = is_loss
            self.accumulation = accumulation
            self.output_group = output_group
            self.target_group = target_group
            self.group = group

            self._loss: list[torch.Tensor] = []
            self._weight: list[float] = []
            self._values: list[float] = []

        def reset_loss(self) -> None:
            self._loss.clear()

        def add(self, weight: float, value: torch.Tensor | tuple[torch.Tensor, float | dict[str, float]]) -> None:
            if isinstance(value, tuple):
                loss_value, true_value = value
                if isinstance(true_value, dict):
                    # Per-label/landmark metrics (Dice, TRE) report a dict; the logging windows
                    # nan-mean ``_values``, so store a scalar summary here while the tensor still
                    # carries the metric. Absent labels are NaN and are ignored by the mean.
                    numeric = [v for v in true_value.values() if isinstance(v, int | float)]
                    true_value = float(np.nanmean(numeric)) if numeric else float("nan")
            else:
                loss_value = value
                true_value = value.item()

            self._loss.append(loss_value if self.is_loss else loss_value.detach())
            self._values.append(true_value)
            self._weight.append(weight)

        def get_last_loss(self) -> torch.Tensor:
            return self._loss[-1] * self._weight[-1] if len(self._loss) else torch.zeros(1, requires_grad=True)

        def get_loss(self) -> torch.Tensor:
            if not len(self._loss):
                return torch.zeros(1, requires_grad=True)
            # ``_weight`` accumulates across iterations for the logging windows while ``_loss`` is
            # reset every iteration; align the current losses with their own (trailing) weights so a
            # loss-weight scheduler drives the gradient instead of the first weight ever recorded.
            weights = self._weight[-len(self._loss) :]
            return torch.stack(
                [w * loss_value for w, loss_value in zip(weights, self._loss, strict=False)], dim=0
            ).mean(dim=0)

        def __len__(self) -> int:
            return len(self._loss)

    def __init__(
        self,
        model_classname: str,
        outputs_criterions_loader: dict[str, TargetCriterionsLoader],
    ) -> None:
        super().__init__()
        self.outputs_criterions: dict[str, dict[str, dict[torch.nn.Module, CriterionsAttr]]] = {}
        for output_group, target_criterions_loader in outputs_criterions_loader.items():
            self.outputs_criterions[output_group.replace(":", ".")] = target_criterions_loader.get_targets_criterions(
                output_group, model_classname
            )
        self._loss: dict[int, dict[str, Measure.Loss]] = {}
        self.scaler: torch.amp.GradScaler | None = None

    def init(self, model: torch.nn.Module, group_dest: list[str]) -> None:
        outputs_group_rename = {}

        modules = []
        for i, _, _ in model.named_module_args_dict():
            modules.append(i)

        for output_group in self.outputs_criterions.keys():
            if output_group.replace(";accu;", "") not in modules:
                raise MeasureError(
                    f"The output group '{output_group}' defined in 'outputs_criterions' "
                    "does not correspond to any module in the model.",
                    f"Available modules: {modules}",
                    "Please check that the name matches exactly a submodule or output of your model architecture.",
                )

            for target_group in self.outputs_criterions[output_group]:
                for target_group_tmp in target_group.split(";"):
                    if target_group_tmp not in group_dest:
                        raise MeasureError(
                            f"The target_group {target_group_tmp} defined in "
                            "'outputs_criterions.{output_group}.targets_criterions'"
                            " was not found in the available destination groups.",
                            "This target_group is expected for loss or metric computation, "
                            "but was not loaded in 'group_dest'.",
                            f"Please make sure that the group {target_group_tmp} is defined in "
                            "Dataset:groups_src:...:groups_dest: {target_group_tmp} "
                            "and correctly loaded from the dataset.",
                        )
                for criterion in self.outputs_criterions[output_group][target_group]:
                    # ``criterion`` is the criterion module (dict key); the flag lives on it, not on
                    # the CriterionsAttr value — indexing the dict here would always read False and
                    # silently skip graph-rewiring criteria such as KLDivergence.
                    if getattr(criterion, "accepts_init", False):
                        outputs_group_rename[output_group] = criterion.init(model, output_group, target_group)

        outputs_criterions_bak = self.outputs_criterions.copy()
        for old, new in outputs_group_rename.items():
            self.outputs_criterions.pop(old)
            self.outputs_criterions[new] = outputs_criterions_bak[old]
        for output_group in self.outputs_criterions:
            for target_group in self.outputs_criterions[output_group]:
                for criterion, criterions_attr in self.outputs_criterions[output_group][target_group].items():
                    if criterions_attr.group not in self._loss:
                        self._loss[criterions_attr.group] = {}
                    self._loss[criterions_attr.group][
                        f"{output_group}:{target_group}:{criterion.__class__.__name__}"
                    ] = Measure.Loss(
                        criterion.__class__.__name__,
                        output_group,
                        target_group,
                        criterions_attr.group,
                        criterions_attr.is_loss,
                        criterions_attr.accumulation,
                    )

    def update(
        self,
        output_group: str,
        output: torch.Tensor,
        batch_data_with_attribute: dict[str, tuple[torch.Tensor, list[Attribute]]],
        it: int,
        nb_patch: int,
        training: bool,
    ) -> None:
        for target_group in self.outputs_criterions[output_group]:
            target_data = [
                batch_data_with_attribute[group][0].to(output[0].device).detach()
                for group in target_group.split(";")
                if group in batch_data_with_attribute
            ]
            target_attribute = [
                batch_data_with_attribute[group][1]
                for group in target_group.split(";")
                if group in batch_data_with_attribute
            ]

            for criterion, criterions_attr in self.outputs_criterions[output_group][target_group].items():
                if it >= criterions_attr.start and (criterions_attr.stop is None or it <= criterions_attr.stop):
                    scheduler = self.update_scheduler(criterions_attr.schedulers, it)
                    if getattr(criterion, "accepts_attributes", False):
                        loss = criterion(output, *target_data, attributes=target_attribute)
                    else:
                        loss = criterion(output, *target_data)
                    self._loss[criterions_attr.group][
                        f"{output_group}:{target_group}:{criterion.__class__.__name__}"
                    ].add(scheduler.get_value(), loss)
                    if (
                        training
                        and len(
                            np.unique(
                                [
                                    len(loss_value)
                                    for loss_value in self._loss[criterions_attr.group].values()
                                    if loss_value.accumulation and loss_value.is_loss
                                ]
                            )
                        )
                        == 1
                    ):
                        # Only the accumulation loss that completes the group's per-patch set may fire the
                        # accumulated backward. Without the `accumulation` guard, a plain (non-accumulation)
                        # loss added later in the SAME numeric group re-satisfies the uniform-count test and
                        # re-runs backward over the already-freed accumulation graph (double gradient / crash).
                        if criterions_attr.accumulation and criterions_attr.is_loss:
                            loss = torch.zeros(1, requires_grad=True)
                            for v in [
                                loss_value
                                for loss_value in self._loss[criterions_attr.group].values()
                                if loss_value.accumulation and loss_value.is_loss
                            ]:
                                loss_value = v.get_last_loss()
                                loss = loss.to(loss_value.device) + loss_value
                            loss = loss / nb_patch
                            if self.scaler is not None:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

    def get_loss(self) -> list[torch.Tensor]:
        loss: dict[int, torch.Tensor] = {}
        for group in self._loss.keys():
            loss[group] = torch.zeros(1, requires_grad=True)
            for v in self._loss[group].values():
                if v.is_loss and not v.accumulation:
                    loss_value = v.get_loss()
                    loss[v.group] = loss[v.group].to(loss_value.device) + loss_value
        return list(loss.values())

    def reset_loss(self) -> None:
        for group in self._loss.keys():
            for v in self._loss[group].values():
                v.reset_loss()

    def get_last_values(self, n: int = 1) -> dict[str, float]:
        result = {}
        for group in self._loss.keys():
            result.update(
                {
                    name: np.nanmean(value._values[-n:] if n > 0 else value._values)
                    for name, value in self._loss[group].items()
                    if n < 0 or len(value._values) >= n
                }
            )
        return result

    def get_last_weights(self, n: int = 1) -> dict[str, float]:
        result = {}
        for group in self._loss.keys():
            result.update(
                {
                    name: np.nanmean(value._weight[-n:] if n > 0 else value._weight)
                    for name, value in self._loss[group].items()
                    if n < 0 or len(value._values) >= n
                }
            )
        return result

    def format_loss(self, is_loss: bool, n: int) -> dict[str, tuple[float, float]]:
        result = {}
        for group in self._loss.keys():
            for name, loss in self._loss[group].items():
                if loss.is_loss == is_loss and len(loss._values) >= n:
                    result[name] = (
                        np.nanmean(loss._weight[-n:]),
                        np.nanmean(loss._values[-n:]),
                    )
        return result

    def update_scheduler(self, schedulers: dict[Scheduler, int], it: int) -> Scheduler:
        if not schedulers:
            raise ConfigError(
                f"No scheduler is configured, cannot select one for iteration {it}.",
                "Declare at least one scheduler window in the optimizer configuration.",
            )
        # Pick the window covering `it`; if `it` is past every window, the loop falls
        # through and clamps to the last scheduler (stepped past its last window start).
        step = 0
        _scheduler: Scheduler | None = None
        for _scheduler, value in schedulers.items():
            if value is None or (it >= step and it < step + value):
                break
            step += value
        if _scheduler is None:  # unreachable (schedulers is non-empty); kept for type-narrowing
            raise ConfigError(f"No scheduler matched iteration {it}.")
        _scheduler.step(it - step)
        return _scheduler


def _leaf_spatial_stride(module: torch.nn.Module) -> list[int] | None:
    """Per-axis stride of a leaf that shrinks the grid (a ``Conv``, ``MaxPool`` or ``AvgPool``), else
    ``None``.

    ``ConvTranspose``/``Upsample`` grow the grid, so they read ``None`` and the trace passes their input
    factor straight through. ``AvgPool`` IS a downsampler (a model may pool on its main path) and is
    counted: a residual branch's ``AvgPool`` does not inflate the factor because the branch-aware trace
    merges the parallel main path and shortcut by their per-axis MAX, not their product.
    """
    if isinstance(
        module,
        (
            torch.nn.MaxPool1d,
            torch.nn.MaxPool2d,
            torch.nn.MaxPool3d,
            torch.nn.AvgPool1d,
            torch.nn.AvgPool2d,
            torch.nn.AvgPool3d,
        ),
    ):
        stride = module.stride if module.stride is not None else module.kernel_size
    elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        stride = module.stride
    else:
        return None
    name = type(module).__name__
    ndim = 3 if name.endswith("3d") else 2 if name.endswith("2d") else 1
    return [int(s) for s in (stride if isinstance(stride, (tuple, list)) else [stride] * ndim)]


def _flat_downsampling(module: torch.nn.Module, ndim: int) -> list[int]:
    """Product of every strided ``Conv``/``MaxPool`` inside ``module`` (itself included), each
    trailing-aligned to ``ndim`` -- a leaf of lower dimensionality acts on the LAST axes, so a 2D conv in
    a 3D graph leaves the leading axis untouched.

    This is the factor for an OPAQUE child: a plain torch module whose internal graph the branch trace
    cannot see (a wrapped torchvision/MONAI/smp net added as one ``add_module`` leaf). The flat product
    over-counts a parallel strided shortcut inside it, but over-padding is safe where under-counting
    crashes the model's skip reassembly.
    """
    factor = [1] * ndim
    for leaf in module.modules():
        stride = _leaf_spatial_stride(leaf)
        if stride is None:
            continue
        offset = ndim - len(stride)
        for axis, size in enumerate(stride):
            if axis + offset >= 0:
                factor[axis + offset] *= size
    return factor


class ModuleArgsDict(torch.nn.Module, ABC):
    """Named module graph container supporting KonfAI branch routing metadata."""

    class ModuleArgs:
        def __init__(
            self,
            in_branch: list[str],
            out_branch: list[str],
            pretrained: bool,
            alias: list[str],
            requires_grad: bool | None,
            training: None | bool,
        ) -> None:
            super().__init__()
            self.alias = alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch
            self.in_channels: int | None = None
            self.in_is_channel: bool = True
            self.out_channels: int | None = None
            self.out_is_channel: bool = True
            self.requires_grad = requires_grad
            self.isCheckpoint = False
            self.isGPU_Checkpoint = False
            self.gpu = "cpu"
            self.training = training
            self._isEnd = False

    def __init__(self) -> None:
        super().__init__()
        self._modulesArgs: dict[str, ModuleArgsDict.ModuleArgs] = {}
        self._training = NetState.TRAIN

    def _addindent(self, s_: str, num_spaces: int):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        return first + "\n" + "\n".join(s)

    def __repr__(self):
        extra_lines = []

        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []

        def is_simple_branch(x):
            return len(x) > 1 or x[0] != 0

        for key, module in self._modules.items():
            mod_str = repr(module)

            mod_str = self._addindent(mod_str, 2)
            desc = ""
            if is_simple_branch(self._modulesArgs[key].in_branch) or is_simple_branch(
                self._modulesArgs[key].out_branch
            ):
                desc += f", {self._modulesArgs[key].in_branch}->{self._modulesArgs[key].out_branch}"
            if not self._modulesArgs[key].pretrained:
                desc += ", pretrained=False"
            if self._modulesArgs[key].alias:
                desc += f", alias={self._modulesArgs[key].alias}"
            desc += f", in_channels={self._modulesArgs[key].in_channels}"
            desc += f", in_is_channel={self._modulesArgs[key].in_is_channel}"
            desc += f", out_channels={self._modulesArgs[key].out_channels}"
            desc += f", out_is_channel={self._modulesArgs[key].out_is_channel}"
            desc += f", is_end={self._modulesArgs[key]._isEnd}"
            desc += f", isInCheckpoint={self._modulesArgs[key].isCheckpoint}"
            desc += f", isInGPU_Checkpoint={self._modulesArgs[key].isGPU_Checkpoint}"
            desc += f", requires_grad={self._modulesArgs[key].requires_grad}"
            desc += f", device={self._modulesArgs[key].gpu}"

            child_lines.append(f"({key}{desc}) {mod_str}")

        lines = extra_lines + child_lines

        desc = ""
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                desc += extra_lines[0]
            else:
                desc += "\n  " + "\n  ".join(lines) + "\n"

        return f"{self._get_name()}({desc})"

    def __getitem__(self, key: str) -> torch.nn.Module:
        module = self._modules[key]
        if not module:
            raise ValueError(f"Module '{key}' is None or missing in self._modules")
        return module

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[tuple[str, torch.nn.Module | None]]:
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[torch.nn.Module | None]:
        return self._modules.values()

    def add_module(
        self,
        name: str,
        module: torch.nn.Module,
        in_branch: Sequence[int | str] = [0],
        out_branch: Sequence[int | str] = [0],
        pretrained: bool = True,
        alias: list[str] = [],
        requires_grad: bool | None = None,
        training: None | bool = None,
    ) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs(
            [str(value) for value in in_branch],
            [str(value) for value in out_branch],
            pretrained,
            alias,
            requires_grad,
            training,
        )

    def get_mapping(self):
        results: dict[str, str] = {}
        for name, module_args in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                if len(module_args.alias):
                    count = dict.fromkeys(set(module.get_mapping().values()), 0)
                    if len(count):
                        for k, v in module.get_mapping().items():
                            alias_name = module_args.alias[count[v]]
                            if k == "":
                                results.update({alias_name: name + "." + v})
                            else:
                                results.update({alias_name + "." + k: name + "." + v})
                            count[v] += 1
                    else:
                        for alias in module_args.alias:
                            results.update({alias: name})
                else:
                    results.update({k: name + "." + v for k, v in module.get_mapping().items()})
            else:
                for alias in module_args.alias:
                    results[alias] = name
        return results

    @staticmethod
    def init_func(module: torch.nn.Module, init_type: str, init_gain: float):
        if not isinstance(module, Network):
            if isinstance(module, ModuleArgsDict):
                module.init(init_type, init_gain)
            elif isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):
                if init_type == "normal":
                    torch.nn.init.normal_(module.weight, 0.0, init_gain)
                elif init_type == "xavier":
                    torch.nn.init.xavier_normal_(module.weight, gain=init_gain)
                elif init_type == "kaiming":
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    torch.nn.init.orthogonal_(module.weight, gain=init_gain)
                elif init_type == "trunc_normal":
                    torch.nn.init.trunc_normal_(module.weight, std=init_gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if module.weight is not None:
                    # Normalisation gamma must centre on 1, not 0 (the pix2pix convention): a gamma
                    # near 0 scales the normalised activations to ~0 and stalls early training.
                    torch.nn.init.normal_(module.weight, 1.0, std=init_gain)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def init(self, init_type: str, init_gain: float):
        for module in self._modules.values():
            ModuleArgsDict.init_func(module, init_type, init_gain)

    def named_forward(
        self, *inputs: torch.Tensor, attributes: list[list[Attribute]] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if len(inputs) > 0:
            branchs: dict[str, torch.Tensor] = {}
            attribute_branchs: dict[str, list[Attribute]] = {}
            for i, sinput in enumerate(inputs):
                branchs[str(i)] = sinput
                if attributes is not None and i < len(attributes) and attributes[i] is not None:
                    attribute_branchs[str(i)] = attributes[i]

            out = inputs[0]
            tmp: list[int | str] = []
            for name, module in self.items():
                # Reset per module: ``tmp`` tracks out_branches a nested sibling already filled via
                # inner-match. Kept across siblings, a later sibling sharing that out_branch would skip
                # the fallback below and its output would be silently dropped.
                tmp = []
                if self._modulesArgs[name].training is None or (
                    not (self._modulesArgs[name].training and self._training == NetState.PREDICTION)
                    and not (not self._modulesArgs[name].training and self._training == NetState.TRAIN)
                ):
                    requires_grad = self._modulesArgs[name].requires_grad
                    if requires_grad is not None and module:
                        module.requires_grad_(requires_grad)
                    target_gpu = self._modulesArgs[name].gpu
                    for ib in self._modulesArgs[name].in_branch:
                        if ib not in branchs:
                            branchs[ib] = inputs[0]
                        if target_gpu != "cpu" and str(branchs[ib].device) != f"cuda:{target_gpu}":
                            branchs[ib] = branchs[ib].to(
                                int(target_gpu),
                                non_blocking=branchs[ib].device.type == "cpu",
                            )

                    if self._modulesArgs[name].isCheckpoint:
                        out = checkpoint(
                            module,
                            *[branchs[i] for i in self._modulesArgs[name].in_branch],
                            use_reentrant=True,
                        )
                        for ob in self._modulesArgs[name].out_branch:
                            branchs[ob] = out
                        yield name, out
                    else:
                        if isinstance(module, ModuleArgsDict):
                            for k, out in module.named_forward(
                                *[branchs[i] for i in self._modulesArgs[name].in_branch],
                                attributes=(
                                    [attribute_branchs.get(i, [Attribute()]) for i in self._modulesArgs[name].in_branch]
                                    if attribute_branchs
                                    else None
                                ),
                            ):
                                for ob in self._modulesArgs[name].out_branch:
                                    if ob in module._modulesArgs[k.split(".")[0].replace(";accu;", "")].out_branch:
                                        tmp.append(ob)
                                        branchs[ob] = out
                                yield name + "." + k, out
                            for ob in self._modulesArgs[name].out_branch:
                                if ob not in tmp:
                                    branchs[ob] = out
                        elif isinstance(module, torch.nn.Module):
                            if getattr(module, "accepts_attributes", False):
                                out = module(
                                    *[branchs[i] for i in self._modulesArgs[name].in_branch],
                                    attributes=[
                                        attribute_branchs.get(i, [Attribute()])
                                        for i in self._modulesArgs[name].in_branch
                                    ],
                                )
                            else:
                                out = module(*[branchs[i] for i in self._modulesArgs[name].in_branch])
                            for ob in self._modulesArgs[name].out_branch:
                                branchs[ob] = out
                            yield name, out
            del branchs

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        _v = input
        for _, _v in self.named_forward(*input):
            pass
        return _v

    def named_parameters(
        self, pretrained: bool = False, recurse=False
    ) -> Iterator[tuple[str, torch.nn.parameter.Parameter]]:
        for name, module_args in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                for k, v in module.named_parameters(pretrained=pretrained):
                    yield name + "." + k, v
            elif isinstance(module, torch.nn.Module):
                if not pretrained or not module_args.pretrained:
                    if module_args.training is None or module_args.training:
                        for k, v in module.named_parameters():
                            yield name + "." + k, v

    def parameters(self, pretrained: bool = False):
        for _, v in self.named_parameters(pretrained=pretrained):
            yield v

    def named_module_args_dict(self) -> Iterator[tuple[str, Self, ModuleArgs]]:
        for name, module in self._modules.items():
            yield name, module, self._modulesArgs[name]
            if isinstance(module, ModuleArgsDict):
                for k, v, u in module.named_module_args_dict():
                    yield name + "." + k, v, u

    def _requires_grad(self, keys: list[str]):
        keys = keys.copy()
        for name, module, args in self.named_module_args_dict():
            requires_grad = args.requires_grad
            if requires_grad is not None:
                module.requires_grad_(requires_grad)
            if name in keys:
                keys.remove(name)
                if len(keys) == 0:
                    break

    def _trace_downsampling(self, seeds: list[list[int]], seen: list[list[int]]) -> list[int]:
        """Propagate the per-axis downsampling factor through the branch register, recording each branch
        value in ``seen``. Parallel branches -- a residual shortcut beside the main path -- accumulate from
        the SAME seed and merge at their ``Add`` without multiplying, so a strided projection is not
        double-counted the way a flat ``modules()`` walk would. A child that is NOT a routed block is
        opaque and contributes its flat internal product (``_flat_downsampling``).

        ``seeds`` are this block's input factors, one per positional input; the register is seeded from
        all of them (a decoder block reading ``[upsampled, skip]`` keeps each at its own resolution) and
        an unwritten branch falls back to the first, exactly as ``named_forward`` seeds it. A module
        downsamples along its FIRST input branch; the others only route. Returns the last output's factor.
        """
        branches: dict[str, list[int]] = {str(i): seed for i, seed in enumerate(seeds)}
        default = seeds[0]
        out_f = default
        for name, module in self.items():
            module_args = self._modulesArgs[name]
            in_factors = [branches.get(in_branch, default) for in_branch in module_args.in_branch]
            if isinstance(module, ModuleArgsDict):
                out_f = module._trace_downsampling(in_factors, seen)
            else:
                out_f = [a * b for a, b in zip(in_factors[0], _flat_downsampling(module, len(default)), strict=True)]
            for out_branch in module_args.out_branch:
                branches[out_branch] = out_f
            seen.append(out_f)
        return out_f


class OutputsGroup(list):
    """Container describing one model output and its source modules."""

    def __init__(self, measure: Measure) -> None:
        self.layers: dict[str, torch.Tensor] = {}
        self.measure = measure

    def add_layer(self, name: str, layer: torch.Tensor):
        self.layers[name] = layer

    def is_done(self):
        return len(self) == len(self.layers)

    def clear(self):
        self.layers.clear()


class Network(ModuleArgsDict, ABC):
    """Base class for KonfAI networks participating in a routed model graph."""

    def _apply_network(
        self,
        name_function: Callable[[Self], str],
        networks: list[str],
        key: str,
        function: Callable,
        *args,
        **kwargs,
    ) -> dict[str, object]:
        results: dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                if name_function(module) not in networks:
                    networks.append(name_function(module))
                    for k, v in module._apply_network(
                        name_function,
                        networks,
                        key + "." + name_function(module),
                        function,
                        *args,
                        **kwargs,
                    ).items():
                        results.update({name_function(self) + "." + k: v})
        if len([param.name for param in list(inspect.signature(function).parameters.values()) if param.name == "key"]):
            function = partial(function, key=key)

        results[name_function(self)] = function(self, *args, **kwargs)
        return results

    def _function_network():  # type: ignore[misc]
        def _function_network_d(function: Callable):
            def new_function(self: Self, *args, **kwargs) -> dict[str, object]:
                return self._apply_network(
                    lambda network: network.get_name(),
                    [],
                    self.get_name(),
                    function,
                    *args,
                    **kwargs,
                )

            return new_function

        return _function_network_d

    def __init__(
        self,
        in_channels: int = 1,
        optimizer: OptimizerLoader | None = None,
        schedulers: dict[str, LRSchedulersLoader] | None = None,
        outputs_criterions: dict[str, TargetCriterionsLoader] | None = None,
        patch: ModelPatch | None = None,
        nb_batch_per_step: int = 1,
        init_type: str = "normal",
        init_gain: float = 0.02,
        dim: int = 3,
    ) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.in_channels = in_channels
        self.optimizerLoader = optimizer
        self.optimizer: torch.optim.Optimizer | None = None

        self.lr_schedulers_loader = schedulers
        self.schedulers: dict[torch.optim.lr_scheduler._LRScheduler, int] = {}

        self.outputs_criterions_loader = outputs_criterions
        self.measure: Measure | None = None

        self.patch = patch

        self.nb_batch_per_step = nb_batch_per_step
        self.init_type = init_type
        self.init_gain = init_gain
        self.dim = dim
        self._it = 0
        self._nb_lr_update = 0
        self.outputsGroup: list[OutputsGroup] = []

    @_function_network()
    def state_dict(self) -> dict[str, OrderedDict]:
        destination: OrderedDict[str, Any] = OrderedDict()
        local_metadata = {"version": self._version}
        self._save_to_state_dict(destination, "", False)
        for name, module in self._modules.items():
            if module is not None:
                if not isinstance(module, Network):
                    module.state_dict(destination=destination, prefix="" + name + ".", keep_vars=False)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, "", local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict["_metadata"] = metadata

        def load(module: torch.nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    if not isinstance(child, Network):
                        weight_key = prefix + name + ".weight"
                        if (
                            isinstance(child, (torch.nn.modules.conv._ConvNd, torch.nn.Linear))
                            and weight_key in state_dict
                        ):
                            current_size = child.weight.shape[0]
                            last_size = state_dict[weight_key].shape[0]

                            if current_size != last_size:
                                print(
                                    f"Warning: The size of '{prefix + name}' has changed from {last_size}"
                                    f" to {current_size}. Please check for potential impacts"
                                )
                                ModuleArgsDict.init_func(child, self.init_type, self.init_gain)

                                bias_key = prefix + name + ".bias"
                                # Copy the overlap only. Slicing both sides by min(current, last) keeps the
                                # GROW case (checkpoint smaller -> fill the top rows) working AND fixes the
                                # SHRINK case (checkpoint larger): `weight[:last_size] = ckpt` would pair the
                                # smaller current tensor against the larger checkpoint and crash.
                                overlap = min(current_size, last_size)
                                with torch.no_grad():
                                    child.weight[:overlap] = state_dict[weight_key][:overlap]
                                    if child.bias is not None and bias_key in state_dict:
                                        child.bias[:overlap] = state_dict[bias_key][:overlap]
                                # Skip the normal load for this resized leaf, but keep
                                # loading its siblings (was an early `return` that aborted them).
                                continue
                        load(child, prefix + name + ".")

        load(self)

        if len(unexpected_keys) > 0:
            formatted_keys = ", ".join(f'"{k}"' for k in unexpected_keys)
            error_msgs.insert(
                0,
                f"Unexpected key(s) in state_dict: {formatted_keys}.",
            )
        if len(missing_keys) > 0:
            formatted_keys = ", ".join(f'"{k}"' for k in missing_keys)
            error_msgs.insert(
                0,
                f"Missing key(s) in state_dict: {formatted_keys}.",
            )

        if len(error_msgs) > 0:
            formatted_errors = "\n\t".join(error_msgs)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{formatted_errors}",
            )

    def apply(self, fn: Callable[[torch.nn.Module], None]) -> None:
        """
        Apply ``fn`` to each non-KonfAI child module and finally to ``self``.

        This overrides ``torch.nn.Module.apply`` so the recursive traversal can
        skip nested ``Network`` instances and keep KonfAI's graph semantics
        intact.
        """
        for module in self.children():
            if not isinstance(module, Network):
                module.apply(fn)
        fn(self)

    @_function_network()
    def load(
        self,
        state_dict: dict[str, dict[str, torch.Tensor] | int],
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ):
        if init:
            self.apply(
                partial(
                    ModuleArgsDict.init_func,
                    init_type=self.init_type,
                    init_gain=self.init_gain,
                )
            )
        name = "Model"
        if ema:
            if name + "_EMA" in state_dict:
                name += "_EMA"
        if name in state_dict:
            value = state_dict[name]
            model_state_dict_tmp = {}
            if isinstance(value, dict):
                model_state_dict_tmp = {k.split(".")[-1]: v for k, v in value.items()}[self.get_name()]
            modules_name = self.get_mapping()
            model_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

            for alias in model_state_dict_tmp.keys():
                prefix = ".".join(alias.split(".")[:-1])
                alias_list = [
                    (".".join(prefix.split(".")[: len(i.split("."))]), v)
                    for i, v in modules_name.items()
                    if prefix.startswith(i)
                ]

                if len(alias_list):
                    for a, b in alias_list:
                        model_state_dict[alias.replace(a, b)] = model_state_dict_tmp[alias]
                        break
                else:
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict)
        if f"{self.get_name()}_optimizer_state_dict" in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict[f"{self.get_name()}_optimizer_state_dict"])
        if f"{self.get_name()}_it" in state_dict:
            _it = state_dict.get(f"{self.get_name()}_it")
            if isinstance(_it, int):
                self._it = _it
        if f"{self.get_name()}_nb_lr_update" in state_dict:
            _nb_lr_update = state_dict.get(f"{self.get_name()}_nb_lr_update")
            if isinstance(_nb_lr_update, int):
                self._nb_lr_update = _nb_lr_update

        if override_lr is not None and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = override_lr
                param_group["initial_lr"] = override_lr

        for scheduler in self.schedulers:
            sched: Any = scheduler
            if override_lr is not None:
                if hasattr(sched, "base_lrs"):
                    sched.base_lrs = [override_lr for _ in sched.base_lrs]
                if hasattr(sched, "initial_lr"):
                    sched.initial_lr = override_lr
                sched.last_epoch = 0
                if hasattr(sched, "_last_lr"):
                    sched._last_lr = [override_lr for _ in sched._last_lr]
            else:
                sched.last_epoch = self._nb_lr_update
        self.initialized()

    def _compute_channels_trace(
        self,
        module: ModuleArgsDict,
        in_channels: int,
        gradient_checkpoints: list[str] | None,
        gpu_checkpoints: list[str] | None,
        name: str | None = None,
        in_is_channel: bool = True,
        out_channels: int | None = None,
        out_is_channel: bool = True,
    ) -> tuple[int, bool, int | None, bool]:

        for k1, v1 in module.items():
            if isinstance(v1, ModuleArgsDict):
                for t in module._modulesArgs[k1].out_branch:
                    last = None
                    for k2, _ in v1.items():
                        if t in v1._modulesArgs[k2].out_branch:
                            last = k2
                    if last is not None:
                        v1._modulesArgs[last]._isEnd = True
                    else:
                        v1._modulesArgs[k2]._isEnd = True

        for k, v in module.items():
            if hasattr(v, "in_channels"):
                if v.in_channels:
                    in_channels = v.in_channels
            if hasattr(v, "in_features"):
                if v.in_features:
                    in_channels = v.in_features
            key = name + "." + k if name else k

            if gradient_checkpoints:
                if key in gradient_checkpoints:
                    module._modulesArgs[k].isCheckpoint = True

            if gpu_checkpoints:
                if key in gpu_checkpoints:
                    module._modulesArgs[k].isGPU_Checkpoint = True

            module._modulesArgs[k].in_channels = in_channels
            module._modulesArgs[k].in_is_channel = in_is_channel

            if isinstance(v, ModuleArgsDict):
                in_channels, in_is_channel, out_channels, out_is_channel = self._compute_channels_trace(
                    v,
                    in_channels,
                    gradient_checkpoints,
                    gpu_checkpoints,
                    key,
                    in_is_channel,
                    out_channels,
                    out_is_channel,
                )

            if v.__class__.__name__ == "ToChannels":
                out_is_channel = True

            if v.__class__.__name__ == "ToFeatures":
                out_is_channel = False

            if hasattr(v, "out_channels"):
                if v.out_channels:
                    out_channels = v.out_channels
            if hasattr(v, "out_features"):
                if v.out_features:
                    out_channels = v.out_features

            module._modulesArgs[k].out_channels = out_channels
            module._modulesArgs[k].out_is_channel = out_is_channel

            in_channels = out_channels if out_channels is not None else in_channels
            in_is_channel = out_is_channel

        return in_channels, in_is_channel, out_channels, out_is_channel

    def downsampling_factor(self) -> list[int] | None:
        """Per-axis factor the input spatial size must be a multiple of, or ``None`` if the graph never
        downsamples.

        An encoder/decoder graph (U-Net) only reassembles its skip connections when the input divides
        evenly at every level, so the input must be a multiple of the coarsest downsampling the graph
        reaches. That factor is traced through the branch register: a strided ``Conv`` or a ``MaxPool``
        multiplies the branch it writes, while ``ConvTranspose``/``Upsample`` and a residual branch's
        ``AvgPool`` pass through. Because the trace follows branches, a residual block's strided shortcut
        (parallel to its strided main conv, merged by ``Add``) counts ONCE, not twice. Used to size a
        free (``0``) patch axis to a valid extent (padded up, cropped back after the forward).
        """
        # The graph's spatial rank = the WIDEST strided leaf (a 2D side head in a 3D net must not lock
        # the rank to 2); every leaf stride then aligns to the trailing axes of that rank.
        ndim = max((len(s) for s in map(_leaf_spatial_stride, self.modules()) if s is not None), default=0)
        if ndim == 0:
            return None
        seen: list[list[int]] = []
        self._trace_downsampling([[1] * ndim], seen)
        factor = [max((f[axis] for f in seen), default=1) for axis in range(ndim)]
        return factor if any(f > 1 for f in factor) else None

    @_function_network()
    def init(self, autocast: bool, state: State, group_dest: list[str], key: str) -> None:
        if self.outputs_criterions_loader:
            self.measure = Measure(key, self.outputs_criterions_loader)
            self.measure.init(self, group_dest)
        if self.patch is not None:
            self.patch.init(f"{konfai_root()}.Model.{key}.Patch")
        if state != State.PREDICTION:
            self.scaler = torch.amp.GradScaler("cuda", enabled=autocast)
            if self.measure is not None:
                self.measure.scaler = self.scaler
            if self.optimizerLoader:
                self.optimizer = self.optimizerLoader.get_optimizer(key, self.parameters(False))
                self.optimizer.zero_grad()

            if self.lr_schedulers_loader and self.optimizer:
                for schedulers_classname, schedulers in self.lr_schedulers_loader.items():
                    self.schedulers[schedulers.getschedulers(key, schedulers_classname, self.optimizer)] = (
                        schedulers.nb_step
                    )

    def initialized(self):
        pass

    def named_forward(
        self, *inputs: torch.Tensor, attributes: list[list[Attribute]] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if self.patch:
            self.patch.load(inputs[0].shape[2:])
            accumulators: dict[str, Accumulator] = {}

            patch_iterator = self.patch.disassemble(*inputs)
            buffer = []
            for i, patch_input in enumerate(patch_iterator):
                for name, output_layer in super().named_forward(*patch_input, attributes=attributes):
                    yield f";accu;{name}", output_layer
                    buffer.append((name.split(".")[0], output_layer))
                    if len(buffer) == 2:
                        if buffer[0][0] != buffer[1][0]:
                            if self._modulesArgs[buffer[0][0]]._isEnd:
                                if buffer[0][0] not in accumulators:
                                    accumulators[buffer[0][0]] = Accumulator(
                                        self.patch.get_patch_slices(),
                                        self.patch.patch_size,
                                        self.patch.patch_combine,
                                    )
                                accumulators[buffer[0][0]].add_layer(i, buffer[0][1])
                        buffer.pop(0)
                if self._modulesArgs[buffer[0][0]]._isEnd:
                    if buffer[0][0] not in accumulators:
                        accumulators[buffer[0][0]] = Accumulator(
                            self.patch.get_patch_slices(),
                            self.patch.patch_size,
                            self.patch.patch_combine,
                        )
                    accumulators[buffer[0][0]].add_layer(i, buffer[0][1])
                # The leftover entry must not leak into the next patch iteration: the name-transition
                # branch above would re-add patch i's end-module output at index i+1, and Accumulator
                # blends incrementally so a spurious first add can no longer be overwritten.
                buffer.clear()
            for name, accumulator in accumulators.items():
                yield name, accumulator.assemble()
        else:
            for name, output_layer in super().named_forward(*inputs, attributes=attributes):
                yield name, output_layer

    def get_layers(
        self,
        inputs: list[torch.Tensor],
        layers_name: list[str],
        attributes: list[list[Attribute]] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor, PatchIndexed | None]]:
        layers_name = layers_name.copy()
        output_layer_accumulator: dict[str, Accumulator] = {}
        output_layer_patch_indexed: dict[str, PatchIndexed] = {}
        it = 0
        debug = "KONFAI_DEBUG" in os.environ
        for name_tmp, output_layer in self.named_forward(*inputs, attributes=attributes):
            name = name_tmp.replace(";accu;", "")
            if debug:
                if "KONFAI_DEBUG_LAST_LAYER" in os.environ:
                    os.environ["KONFAI_DEBUG_LAST_LAYER"] = (
                        f"{os.environ['KONFAI_DEBUG_LAST_LAYER']}|{name}:"
                        f"{get_gpu_memory(output_layer.device)}:"
                        f"{str(output_layer.device).replace('cuda:', '')}"
                    )
                else:
                    os.environ["KONFAI_DEBUG_LAST_LAYER"] = (
                        f"{name}:{get_gpu_memory(output_layer.device)}:{str(output_layer.device).replace('cuda:', '')}"
                    )
            it += 1
            if name in layers_name or name_tmp in layers_name:
                if ";accu;" in name_tmp:
                    if name not in output_layer_patch_indexed:
                        network_name = (
                            name_tmp.split(".;accu;")[-2].split(".")[-1]
                            if ".;accu;" in name_tmp
                            else name_tmp.split(";accu;")[-2].split(".")[-1]
                        )
                        module = self
                        network = None
                        if network_name == "":
                            network = module
                        else:
                            for n in name.split("."):
                                module = module[n]
                                if isinstance(module, Network) and n == network_name:
                                    network = module
                                    break

                        if network and network.patch:
                            output_layer_patch_indexed[name] = PatchIndexed(network.patch, 0)

                    if name not in output_layer_accumulator:
                        output_layer_accumulator[name] = Accumulator(
                            output_layer_patch_indexed[name].patch.get_patch_slices(0),
                            output_layer_patch_indexed[name].patch.patch_size,
                            output_layer_patch_indexed[name].patch.patch_combine,
                        )

                    if name_tmp in layers_name:
                        output_layer_accumulator[name].add_layer(output_layer_patch_indexed[name].index, output_layer)
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_accumulator[name].is_full():
                            output_layer = output_layer_accumulator[name].assemble()
                            output_layer_accumulator.pop(name)
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name_tmp)
                            yield name_tmp, output_layer, None

                if name in layers_name:
                    if ";accu;" in name_tmp:
                        yield name, output_layer, output_layer_patch_indexed[name]
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_patch_indexed[name].is_full():
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name)
                    else:
                        layers_name.remove(name)
                        yield name, output_layer, None

            if not len(layers_name):
                break

    def init_outputs_group(self):
        metric_tmp = {
            network.measure: network.measure.outputs_criterions.keys()
            for network in self.get_networks().values()
            if network.measure
        }
        for k, v in metric_tmp.items():
            for a in v:
                outputs_group = OutputsGroup(k)
                outputs_group.append(a)
                for targets_group in k.outputs_criterions[a].keys():
                    if ":" in targets_group:
                        outputs_group.append(targets_group.replace(":", "."))

                self.outputsGroup.append(outputs_group)

    def forward(
        self,
        batch_sample: BatchSample,
        output_layers: list[str] = [],
    ) -> list[tuple[str, torch.Tensor]]:
        if not len(self.outputsGroup) and not len(output_layers):
            return []

        self.reset_loss()
        results = []
        measure_output_layers = set()
        for _outputs_group in self.outputsGroup:
            for name in _outputs_group:
                measure_output_layers.add(name)

        for name, layer, patch_indexed in self.get_layers(
            [batch_data_item.tensor for batch_data_item in batch_sample.values() if batch_data_item.is_input],
            list(set(list(measure_output_layers) + output_layers)),
            attributes=[
                batch_data_item.attribute for batch_data_item in batch_sample.values() if batch_data_item.is_input
            ],
        ):
            outputs_group = [outputs_group for outputs_group in self.outputsGroup if name in outputs_group]

            if len(outputs_group) > 0:
                if patch_indexed is None:
                    batch_data_with_attribute = {
                        k: (batch_data_item.tensor, batch_data_item.attribute)
                        for k, batch_data_item in batch_sample.items()
                    }
                    nb = 1
                else:
                    batch_data_with_attribute = {
                        k: (
                            patch_indexed.patch.get_data(batch_data_item.tensor, patch_indexed.index, 0, False),
                            batch_data_item.attribute,
                        )
                        for k, batch_data_item in batch_sample.items()
                    }
                    nb = patch_indexed.patch.get_size(0)

                for output_group in outputs_group:
                    output_group.add_layer(name, layer)
                    if output_group.is_done():
                        batch_data_with_attribute.update(
                            {
                                k.replace(".", ":"): (batch_data_item, [Attribute()])
                                for k, batch_data_item in output_group.layers.items()
                                if k != output_group[0]
                            }
                        )
                        output_group.measure.update(
                            output_group[0],
                            output_group.layers[output_group[0]],
                            batch_data_with_attribute,
                            self._it,
                            nb,
                            self.training,
                        )
                        output_group.clear()
            if name in output_layers:
                results.append((name, layer))
        return results

    @_function_network()
    def reset_loss(self):
        if self.measure:
            self.measure.reset_loss()

    @_function_network()
    def backward(self, model: Any):
        if self.measure:
            if self.scaler and self.optimizer:
                self._requires_grad(list(self.measure.outputs_criterions.keys()))
                should_step = (self._it + 1) % self.nb_batch_per_step == 0
                sync_context = (
                    model.no_sync()
                    if hasattr(model, "no_sync") and callable(model.no_sync) and not should_step
                    else nullcontext()
                )
                with sync_context:
                    for loss in self.measure.get_loss():
                        self.scaler.scale(loss / self.nb_batch_per_step).backward()

                if should_step:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                self._it += 1

    @_function_network()
    def update_lr(self):
        self._nb_lr_update += 1
        step = 0
        _scheduler = None
        for _scheduler, value in self.schedulers.items():
            if value is None or (self._nb_lr_update >= step and self._nb_lr_update < step + value):
                break
            step += value
        if _scheduler:
            if _scheduler.__class__.__name__ == "ReduceLROnPlateau":
                if self.measure:
                    _scheduler.step(sum(self.measure.get_last_values(0).values()))
            else:
                _scheduler.step()

    @_function_network()
    def get_networks(self) -> Self:
        return self

    @staticmethod
    def to(module: ModuleArgsDict, device: int, _counter: list[int] | None = None):
        # `_counter` is a single-element box holding the next GPU index, shared by
        # reference through the recursion so model-parallel `isGPU_Checkpoint` splits
        # advance it. Each top-level call starts fresh at `device` (replacing a previous
        # `os.environ["device"]` counter that leaked across independent placements).
        if _counter is None:
            _counter = [device]
        for k, v in module.items():
            if module._modulesArgs[k].gpu == "cpu":
                if module._modulesArgs[k].isGPU_Checkpoint:
                    _counter[0] += 1
                module._modulesArgs[k].gpu = str(get_device(_counter[0]))
                if isinstance(v, ModuleArgsDict):
                    v = Network.to(v, _counter[0], _counter)
                else:
                    v = v.to(get_device(_counter[0]))
        if isinstance(module, Network):
            if module.optimizer is not None:
                for state in module.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(get_device(_counter[0]))
        return module

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    def set_state(self, state: NetState):
        for module in self.modules():
            if isinstance(module, ModuleArgsDict):
                module._training = state


class MinimalModel(Network):
    """Small wrapper exposing a single network as a full KonfAI model graph.

    The wrapped model arrives fully constructed — possibly carrying pretrained weights (a
    torchvision/MONAI/SMP class with ``weights=...``). ``load`` therefore never re-initialises:
    ``load(init=True)`` at training start applies ``init_func`` over every descendant and would
    silently destroy those weights with ``init_type`` noise. Models built from scratch keep
    KonfAI's init behaviour; checkpoint loading is unaffected.
    """

    def load(
        self,
        state_dict: dict[str, dict[str, torch.Tensor] | int],
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ):
        del init  # the wrapped model owns its initialisation (possibly pretrained)
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)

    def __init__(
        self,
        model: Network,
        optimizer: OptimizerLoader = OptimizerLoader(),
        schedulers: dict[str, LRSchedulersLoader] = {"default|StepLR": LRSchedulersLoader(0)},
        outputs_criterions: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        dim: int = 3,
        nb_batch_per_step=1,
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__(
            1,
            optimizer,
            schedulers,
            outputs_criterions,
            patch,
            nb_batch_per_step,
            init_type,
            init_gain,
            dim,
        )
        self.add_module("Model", model)


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
            # catalog is a flat directory, so the name must be a bare filename -- reject any path
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
        # v1.6.0 moved konfai/models/<kind> to konfai/models/python/<kind>. A config that referenced a
        # built-in model by its old absolute path (konfai.models.<kind>.<file>:<Class>) keeps working:
        # rewrite the prefix once, with a deprecation warning, instead of failing on ModuleNotFoundError.
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
    ) -> Any:
        return self.module(batch_sample, output_layers)
