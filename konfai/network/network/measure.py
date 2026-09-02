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


"""A network's measure: its criteria over named outputs, and their running values."""

import math
from collections import deque
from collections.abc import Iterator
from itertools import islice
from typing import Any, NamedTuple, TypeAlias

import numpy as np
import torch

from konfai.metric.schedulers import Scheduler
from konfai.network.network.base import strip_accumulated
from konfai.network.network.loaders import CriterionsAttr, TargetCriterionsLoader
from konfai.utils.dataset import Attribute
from konfai.utils.errors import ConfigError, MeasureError


class LabelledValues(NamedTuple):
    """A per-label metric before its host readout: one value per label (NaN for a label the
    reference lacks) and the labels naming them. ``Measure._materialize`` reads the tensor in the
    same batched transfer as the scalar losses; the evaluator turns it into a per-label dict."""

    values: torch.Tensor
    labels: list[Any]


#: The value a criterion reports beside its loss: a float, a 0-d tensor read lazily off its device,
#: a dict of per-label floats, or a :class:`LabelledValues` pair read lazily.
CriterionValue: TypeAlias = float | torch.Tensor | dict[Any, float] | LabelledValues

#: Every shape ``Criterion.forward`` may return; ``CriterionResult.of`` normalizes them all.
CriterionOutput: TypeAlias = (
    torch.Tensor | tuple[torch.Tensor, CriterionValue] | tuple[torch.Tensor, CriterionValue, torch.Tensor]
)


class CriterionResult(NamedTuple):
    """A criterion's forward, normalized: the loss tensor, the reported value, the optional
    per-voxel map. ``of`` is the one place the accepted shapes are checked."""

    loss: torch.Tensor
    value: CriterionValue
    map: torch.Tensor | None = None

    @classmethod
    def of(cls, raw: CriterionOutput, criterion: str = "criterion") -> "CriterionResult":
        if isinstance(raw, torch.Tensor):
            return cls(raw, raw.detach(), None)
        if not isinstance(raw, tuple) or not 2 <= len(raw) <= 3 or not isinstance(raw[0], torch.Tensor):
            raise MeasureError(
                f"'{criterion}' returned {type(raw).__name__} instead of a criterion result.",
                "A criterion returns a loss Tensor, or (loss, value) with value a float, a 0-d "
                "Tensor, a dict of floats or a (values, labels) pair, plus an optional per-voxel "
                "map Tensor third.",
            )
        loss, value = raw[0], raw[1]
        map_ = raw[2] if len(raw) == 3 else None
        if isinstance(value, np.generic):
            value = float(value)
        elif isinstance(value, bool | int):
            value = float(value)
        elif isinstance(value, tuple) and not isinstance(value, LabelledValues):
            if len(value) == 2 and isinstance(value[0], torch.Tensor) and isinstance(value[1], list):
                value = LabelledValues(value[0], value[1])
        if not isinstance(value, float | torch.Tensor | dict | LabelledValues) or (
            map_ is not None and not isinstance(map_, torch.Tensor)
        ):
            raise MeasureError(
                f"'{criterion}' reported a {type(value).__name__} value.",
                "The reported value is a float, a 0-d Tensor, a dict of floats or a "
                "(values, labels) pair, and a map is a Tensor.",
            )
        return cls(loss, value, map_)

    def materialized(self) -> float | dict[Any, float]:
        """The reported value as plain floats, read off their device: for per-case consumers (the
        evaluator records into JSON immediately, where a sync is the cadence anyway)."""
        if isinstance(self.value, LabelledValues):
            return dict(zip(self.value.labels, self.value.values.tolist(), strict=True))
        if isinstance(self.value, torch.Tensor):
            return float(self.value.item())
        return self.value


class _RunningNanMean:
    """The nan-aware mean of everything added, in O(1) per value: what ``np.nanmean`` over the whole
    history returns, up to summation order (measured at most 3.2e-14 relative over 5e5 values)."""

    __slots__ = ("count", "total")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float) -> None:
        if not math.isnan(value):
            self.total += value
            self.count += 1

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def _tail(values: deque[float], n: int) -> list[float]:
    """The last ``n`` values (``n > 0``)."""
    return list(islice(values, max(0, len(values) - n), None))


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

            # This iteration's (weight, loss) pairs: the gradient's, cleared by ``reset_loss``.
            self._loss: list[tuple[float, torch.Tensor]] = []
            # The logging windows, bounded by ``set_window`` to the widest window a consumer reads;
            # the whole-history consumers read the running means instead, so nothing grows with the run.
            self._weight: deque[float] = deque()
            self._values: deque[float] = deque()
            self._mean = _RunningNanMean()
            self._mean_weight = _RunningNanMean()
            self._recorded = 0
            # Values recorded but not yet in ``_values``: a loss is kept as its 0-d tensor (a
            # per-label metric as its LabelledValues), because reading it inside the forward stalls
            # the CPU on the whole graph before backward is enqueued. The consumers read them in one
            # transfer per device (``Measure._materialize``).
            self._unread: list[float | torch.Tensor | LabelledValues] = []

        def reset_loss(self) -> None:
            self._loss.clear()

        @property
        def recorded(self) -> int:
            """How many values this record has been given, read off their device or not."""
            return self._recorded

        def set_window(self, n: int) -> None:
            """Keep at least the last ``n`` values and weights. Grows only; until it is called the
            history is unbounded. A window widened mid-run keeps the values it already held, so a
            window of ``m`` values reaches ``n`` after ``n - m`` more arrive: the reads in between
            average the values held, fewer than they ask for."""
            if self._values.maxlen is not None and self._values.maxlen >= n:
                return
            self._values = deque(self._values, maxlen=n)
            self._weight = deque(self._weight, maxlen=n)

        def _record(self, value: float) -> None:
            self._values.append(value)
            self._mean.add(value)

        def values_mean(self, n: int) -> float:
            """The nan-mean of the last ``n`` values, of the whole history for ``n <= 0``."""
            return float(np.nanmean(_tail(self._values, n))) if n > 0 else self._mean.mean()

        def weights_mean(self, n: int) -> float:
            return float(np.nanmean(_tail(self._weight, n))) if n > 0 else self._mean_weight.mean()

        def add(self, weight: float, value: CriterionOutput) -> None:
            result = CriterionResult.of(value, self.name)
            true_value: float | torch.Tensor | LabelledValues
            if isinstance(result.value, dict):
                # Per-label dicts of plain floats (the hard-label Dice route); the logging windows
                # nan-mean ``_values``, so store the scalar summary. Absent labels are NaN and are
                # ignored by the mean.
                numeric = [v for v in result.value.values() if isinstance(v, int | float)]
                true_value = float(np.nanmean(numeric)) if numeric else float("nan")
            else:
                true_value = result.value

            self._loss.append((weight, result.loss if self.is_loss else result.loss.detach()))
            self._unread.append(true_value)
            self._weight.append(weight)
            self._mean_weight.add(weight)
            self._recorded += 1

        def get_last_loss(self) -> torch.Tensor:
            if not len(self._loss):
                return torch.zeros(1, requires_grad=True)
            weight, loss_value = self._loss[-1]
            return loss_value * weight

        def get_loss(self) -> torch.Tensor:
            if not len(self._loss):
                return torch.zeros(1, requires_grad=True)
            return torch.stack([weight * loss_value for weight, loss_value in self._loss], dim=0).mean(dim=0)

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
        # One forward's targets on the outputs' device, keyed by (group, device) and checked against
        # the source tensor: a group feeding several outputs (two heads, deep supervision) is
        # uploaded once. Released at the end of the forward (``Network.forward``).
        self._targets: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _target(self, group: str, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Not on the step's critical path. Issued instead when the batch arrives, the epoch is 20.4 s
        # against 20.2 on the shipped 2D example and 28.3 against 27.3 on a 3D UNet, and only the
        # phase carrying the wait moves (``criteria`` to ``forward``). Made free outright (pinned
        # memory, an asynchronous copy) the line's host wall falls from 24.0 s an epoch to 0.007 and
        # the epoch still does not move: the host meets the device once a step anyway, where Dice
        # reads its lowest label back.
        moved = self._targets.get((group, device))
        if moved is None or moved[0] is not tensor:
            moved = (tensor, tensor.to(device).detach())
            self._targets[(group, device)] = moved
        return moved[1]

    def release_targets(self) -> None:
        self._targets.clear()

    def init(self, model: torch.nn.Module, group_dest: list[str]) -> None:
        outputs_group_rename = {}

        modules = []
        for i, _, _ in model.named_module_args_dict():
            modules.append(i)

        for output_group in self.outputs_criterions.keys():
            if strip_accumulated(output_group) not in modules:
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
                    # the CriterionsAttr value: indexing the dict here would always read False and
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
            groups = [group for group in target_group.split(";") if group in batch_data_with_attribute]
            target_attribute = [batch_data_with_attribute[group][1] for group in groups]

            for criterion, criterions_attr in self.outputs_criterions[output_group][target_group].items():
                if it >= criterions_attr.start and (criterions_attr.stop is None or it <= criterions_attr.stop):
                    # Criteria live outside the model's module tree, so ``network.to(device)`` never
                    # reaches them: a criterion with its own tensors (``CrossEntropyLoss(weight=...)``)
                    # would stay on CPU while the batch is on the GPU.
                    if getattr(criterion, "_konfai_device", None) != output.device:
                        criterion.to(output.device)
                        setattr(criterion, "_konfai_device", output.device)  # noqa: B010 -- Module.__setattr__ is Tensor-typed
                    scheduler = self.update_scheduler(criterions_attr.schedulers, it)
                    # Below the window gate: a criterion outside its start/stop uploads nothing.
                    target_data = [
                        self._target(group, batch_data_with_attribute[group][0], output.device) for group in groups
                    ]
                    if getattr(criterion, "accepts_attributes", False):
                        loss = criterion(output, *target_data, attributes=target_attribute)
                    else:
                        loss = criterion(output, *target_data)
                    self._loss[criterions_attr.group][
                        f"{output_group}:{target_group}:{criterion.__class__.__name__}"
                    ].add(scheduler.get_value(), loss)
                    # Only the accumulation loss that completes the group's per-patch set may fire the
                    # accumulated backward: a plain (non-accumulation) loss added later in the SAME
                    # numeric group must not re-satisfy the uniform-count test and re-run backward over
                    # the already-freed accumulation graph (double gradient / crash). The criterion's own
                    # flags are the cheap half of that test and gate it: 0.02 us a call against 2.94 for
                    # the count over the group (timeit, 20000 calls).
                    if training and criterions_attr.accumulation and criterions_attr.is_loss:
                        accumulated = [
                            record
                            for record in self._loss[criterions_attr.group].values()
                            if record.accumulation and record.is_loss
                        ]
                        if len({len(record) for record in accumulated}) == 1:
                            loss = torch.zeros(1, requires_grad=True)
                            for record in accumulated:
                                loss_value = record.get_last_loss()
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

    def _records(self) -> Iterator[tuple[str, "Measure.Loss"]]:
        for group in self._loss.values():
            yield from group.items()

    def _materialize(self) -> None:
        """Append every unread value to its record's window, in the order recorded, reading the
        tensors off their device in one transfer per device. A ``LabelledValues`` lands as its
        NaN-skipping mean over the labels: what the eager per-label dict summarized before."""
        records = [record for _, record in self._records() if record._unread]
        tensors: dict[torch.device, list[torch.Tensor]] = {}
        for record in records:
            for value in record._unread:
                tensor = value.values if isinstance(value, LabelledValues) else value
                if isinstance(tensor, torch.Tensor):
                    tensors.setdefault(tensor.device, []).append(tensor.reshape(-1))
        read = {device: iter(torch.cat(batch).tolist()) for device, batch in tensors.items()}
        for record in records:
            for value in record._unread:
                if isinstance(value, LabelledValues):
                    values = list(islice(read[value.values.device], value.values.numel()))
                    record._record(float(np.nanmean(values)) if values else float("nan"))
                elif isinstance(value, torch.Tensor):
                    record._record(next(read[value.device]))
                else:
                    record._record(value)
            record._unread.clear()

    def set_window(self, n: int) -> None:
        """Keep at least the last ``n`` values and weights per criterion: the widest window a consumer
        will read. Grows only; without a call the history is unbounded."""
        for _, record in self._records():
            record.set_window(n)

    def _read(self, n: int) -> Iterator[tuple[str, "Measure.Loss"]]:
        """The records given at least ``n`` values, every value read off its device, and the window
        at least ``n`` wide from now on: a consumer's first read declares what it will keep reading."""
        self._materialize()
        if n > 0:
            self.set_window(n)
        return ((name, record) for name, record in self._records() if record.recorded >= n)

    def get_last_values(self, n: int = 1) -> dict[str, float]:
        return {name: record.values_mean(n) for name, record in self._read(n)}

    def get_last_weights(self, n: int = 1) -> dict[str, float]:
        return {name: record.weights_mean(n) for name, record in self._read(n)}

    def format_loss(self, is_loss: bool, n: int) -> dict[str, tuple[float, float]]:
        return {
            name: (record.weights_mean(n), record.values_mean(n))
            for name, record in self._read(n)
            if record.is_loss == is_loss
        }

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
