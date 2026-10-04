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
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch

from konfai.metric.schedulers import Scheduler
from konfai.network.network.base import strip_accumulated
from konfai.network.network.loaders import CriterionsAttr, TargetCriterionsLoader
from konfai.utils.dataset import Attribute
from konfai.utils.errors import ConfigError, MeasureError

if TYPE_CHECKING:
    from konfai.metric.measure.base import CriterionWithInit
    from konfai.network.network.network import ModuleArgsDict


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
            # Funnel through the tuple path so the bare-loss value passes the same shape check.
            raw = (raw, raw.detach())
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
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            # A multi-element plain value would silently shift the deferred per-device readout.
            raise MeasureError(
                f"'{criterion}' reported a tensor of {value.numel()} elements.",
                "The reported value is a single number: reduce it, or report a (values, labels) "
                "pair for a per-label metric.",
            )
        if isinstance(value, LabelledValues) and (value.values.ndim != 1 or value.values.numel() != len(value.labels)):
            # A misshaped pair passes here but explodes far away, in the materialized() zip.
            raise MeasureError(
                f"'{criterion}' reported {tuple(value.values.shape)} values for {len(value.labels)} labels.",
                "A per-label metric reports a 1-D tensor holding exactly one value per label.",
            )
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


def call_criterion(
    criterion: torch.nn.Module,
    output: torch.Tensor,
    targets: list[torch.Tensor],
    attributes: list[list[Attribute]],
    output_attributes: list[Attribute] | None,
) -> CriterionOutput:
    """``criterion`` scored on ``output`` against ``targets``, the one call of training and evaluation.
    A ``CriterionWithAttribute`` receives ``attributes``, the per-sample attributes of each target in
    the order of the target group, and the output's own as ``output_attributes`` when it declares
    ``accepts_output_attributes``: ``None`` in training, where a model output has none."""
    if not getattr(criterion, "accepts_attributes", False):
        return criterion(output, *targets)
    if getattr(criterion, "accepts_output_attributes", False):
        return criterion(output, *targets, attributes=attributes, output_attributes=output_attributes)
    return criterion(output, *targets, attributes=attributes)


def criterion_keys(output_group: str, target_group: str, names: Iterable[str]) -> list[str]:
    """The key of each criterion of ``output_group`` against ``target_group``, from their names in
    order: ``output:target:Name``, the n-th of one name as ``Name#n`` (as a repeated class binds in a
    chain), so two criteria of one class keep a record each."""
    seen: dict[str, int] = {}
    keys = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        keys.append(f"{output_group}:{target_group}:{name if seen[name] == 1 else f'{name}#{seen[name]}'}")
    return keys


class _RunningNanMean:
    """The nan-aware mean of everything added, each value weighing its ``patches``, in O(1) per
    value: what ``_total`` over the whole history returns, up to summation order (measured at most
    3.2e-14 relative over 5e5 values)."""

    __slots__ = ("count", "total")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float, patches: int = 1) -> None:
        if patches and not math.isnan(value):
            self.total += value * patches
            self.count += patches

    def mean(self) -> float:
        return _ratio((self.total, self.count))


def _tail(values: deque[Any], n: int) -> list[Any]:
    """The last ``n`` values (``n > 0``)."""
    return list(islice(values, max(0, len(values) - n), None))


def _total(values: list[float], patches: list[int]) -> tuple[float, int]:
    """The sum of the values that are not NaN, each times its ``patches``, and the patches they hold:
    ``np.nanmean``'s sum and count, bit for bit, when every value holds one patch."""
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(patches, dtype=np.int64)
    kept = ~np.isnan(value) & (weight > 0)
    return float(np.sum(np.where(kept, value, 0.0) * weight)), int(np.sum(np.where(kept, weight, 0)))


def _summed(losses: Iterable[torch.Tensor]) -> torch.Tensor:
    """Zero plus each loss in turn, the running sum moving to each one's device. The zero is made on the
    first loss's device: one made on the host costs an upload, and on CUDA a stream synchronization."""
    total: torch.Tensor | None = None
    for loss in losses:
        start = torch.zeros(1, device=loss.device, requires_grad=True) if total is None else total.to(loss.device)
        total = start + loss
    return torch.zeros(1, requires_grad=True) if total is None else total


def _ratio(total: tuple[float, int]) -> float:
    return total[0] / total[1] if total[1] else float("nan")


class Measure:
    """Collect, validate, and aggregate losses or metrics across model outputs."""

    # False while a batch is run only to keep the ranks' forwards in step: recorded, weighing nothing.
    scored = True

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
            # The patches each value averages: a mean over batches weighs each by it (see ``Measure.update``).
            self._patches: deque[int] = deque()
            # The minimized quantity beside the reported one: a Dice loss reports the coefficient
            # (higher is better) and minimizes one minus it. Checkpoint selection and a plateau
            # schedule read this window, the boards read ``_values``.
            self._losses: deque[float] = deque()
            self._mean = _RunningNanMean()
            self._mean_weight = _RunningNanMean()
            self._mean_loss = _RunningNanMean()
            self._recorded = 0
            # Values recorded but not yet in ``_values``: a loss is kept as its 0-d tensor (a
            # per-label metric as its LabelledValues), because reading it inside the forward stalls
            # the CPU on the whole graph before backward is enqueued. The consumers read them in one
            # transfer per device (``Measure._materialize``), the loss tensor beside each value.
            self._unread: list[tuple[float | torch.Tensor | LabelledValues, torch.Tensor, int]] = []

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
            self._losses = deque(self._losses, maxlen=n)
            self._patches = deque(self._patches, maxlen=n)

        def _record(self, value: float, loss: float, patches: int) -> None:
            self._values.append(value)
            self._mean.add(value, patches)
            self._losses.append(loss)
            self._mean_loss.add(loss, patches)
            self._patches.append(patches)

        def values_total(self, n: int) -> tuple[float, int]:
            """The last ``n`` values as ``_total`` sums them, the whole history for ``n <= 0``."""
            if n <= 0:
                return self._mean.total, self._mean.count
            return _total(_tail(self._values, n), _tail(self._patches, n))

        def losses_total(self, n: int) -> tuple[float, int]:
            """The last ``n`` minimized values as ``_total`` sums them, the whole history for ``n <= 0``."""
            if n <= 0:
                return self._mean_loss.total, self._mean_loss.count
            tail = _tail(self._losses, n)
            return _total(tail, _tail(self._patches, len(tail)))

        def values_mean(self, n: int) -> float:
            """The nan-mean of the last ``n`` values, each weighing its patches, of the whole history
            for ``n <= 0``."""
            return _ratio(self.values_total(n))

        def loss_mean(self, n: int) -> float:
            """The nan-mean of the last ``n`` minimized values (the loss the criterion returned, lower
            is better whatever it reports), each weighing its patches, of the whole history for ``n <= 0``."""
            return _ratio(self.losses_total(n))

        def learns(self, n: int) -> bool:
            """Whether any of the last ``n`` minimized values is finite."""
            return any(math.isfinite(loss) for loss in _tail(self._losses, n))

        def weights_mean(self, n: int) -> float:
            return float(np.nanmean(_tail(self._weight, n))) if n > 0 else self._mean_weight.mean()

        def add(self, weight: float, value: CriterionOutput, patches: int = 1) -> None:
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
            self._unread.append((true_value, result.loss.detach(), patches))
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
        # Uploaded when a criterion first reads it: the host waits on the device once a step anyway,
        # where Dice reads its lowest label back, so an earlier or asynchronous copy saves no time.
        moved = self._targets.get((group, device))
        if moved is None or moved[0] is not tensor:
            moved = (tensor, tensor.to(device).detach())
            self._targets[(group, device)] = moved
        return moved[1]

    def release_targets(self) -> None:
        self._targets.clear()

    def init(self, model: "ModuleArgsDict", group_dest: list[str]) -> None:
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
                    # A target spelled with ':' names a model output, which the forward hands over.
                    is_output = ":" in target_group_tmp and target_group_tmp.replace(":", ".") in modules
                    if target_group_tmp not in group_dest and not is_output:
                        raise MeasureError(
                            f"The target_group {target_group_tmp} defined in "
                            f"'outputs_criterions.{output_group}.targets_criterions'"
                            " was not found in the available destination groups.",
                            "This target_group is expected for loss or metric computation, "
                            "but was not loaded in 'group_dest'.",
                            f"Please make sure that the group {target_group_tmp} is defined in "
                            f"Dataset:groups_src:...:groups_dest: {target_group_tmp} "
                            "and correctly loaded from the dataset, or that a model output target "
                            f"names a module path with ':' (available modules: {modules}).",
                        )
                for criterion in self.outputs_criterions[output_group][target_group]:
                    # ``criterion`` is the criterion module (dict key); the flag lives on it, not on
                    # the CriterionsAttr value: indexing the dict here would always read False and
                    # silently skip graph-rewiring criteria such as KLDivergence.
                    if getattr(criterion, "accepts_init", False):
                        outputs_group_rename[output_group] = cast("CriterionWithInit", criterion).init(
                            model, output_group, target_group
                        )

        outputs_criterions_bak = self.outputs_criterions.copy()
        for old, new in outputs_group_rename.items():
            self.outputs_criterions.pop(old)
            self.outputs_criterions[new] = outputs_criterions_bak[old]
        for output_group in self.outputs_criterions:
            for target_group, criteria in self.outputs_criterions[output_group].items():
                keys = criterion_keys(output_group, target_group, (type(criterion).__name__ for criterion in criteria))
                for (criterion, criterions_attr), key in zip(criteria.items(), keys, strict=True):
                    if criterions_attr.group not in self._loss:
                        self._loss[criterions_attr.group] = {}
                    self._loss[criterions_attr.group][key] = Measure.Loss(
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
        for target_group, criteria in self.outputs_criterions[output_group].items():
            groups = [group for group in target_group.split(";") if group in batch_data_with_attribute]
            target_attribute = [batch_data_with_attribute[group][1] for group in groups]
            keys = criterion_keys(output_group, target_group, (type(criterion).__name__ for criterion in criteria))

            for (criterion, criterions_attr), key in zip(criteria.items(), keys, strict=True):
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
                    # A reporting-only metric never backpropagates. Detaching its result afterwards
                    # is too late to avoid saving the intermediate tensors for an unused backward.
                    with torch.set_grad_enabled(torch.is_grad_enabled() and criterions_attr.is_loss):
                        loss = call_criterion(criterion, output, target_data, target_attribute, None)
                    # What the value averages: the batch's patches (``Criterion.batch_mean``), else the batch.
                    patches = (output.shape[0] if getattr(criterion, "batch_mean", False) else 1) if self.scored else 0
                    self._loss[criterions_attr.group][key].add(scheduler.get_value(), loss, patches)
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
                            loss = _summed(record.get_last_loss() for record in accumulated) / nb_patch
                            if self.scaler is not None:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

    def get_loss(self) -> list[torch.Tensor]:
        return [
            _summed(v.get_loss() for v in group.values() if v.is_loss and not v.accumulation)
            for group in self._loss.values()
        ]

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
            for value, loss, _ in record._unread:
                tensor = value.values if isinstance(value, LabelledValues) else value
                if isinstance(tensor, torch.Tensor):
                    tensors.setdefault(tensor.device, []).append(tensor.reshape(-1))
                tensors.setdefault(loss.device, []).append(loss.reshape(-1))
        read = {device: iter(torch.cat(batch).tolist()) for device, batch in tensors.items()}
        for record in records:
            for value, loss, patches in record._unread:
                if isinstance(value, LabelledValues):
                    values = list(islice(read[value.values.device], value.values.numel()))
                    reported = float(np.nanmean(values)) if values else float("nan")
                elif isinstance(value, torch.Tensor):
                    reported = next(read[value.device])
                else:
                    reported = value
                record._record(reported, float(np.nanmean(list(islice(read[loss.device], loss.numel())))), patches)
            record._unread.clear()

    def set_window(self, n: int) -> None:
        """Keep at least the last ``n`` values and weights per criterion: the widest window a consumer
        will read. Grows only; without a call the history is unbounded."""
        for _, record in self._records():
            record.set_window(n)

    def checkpoint_state(self) -> dict[str, Any]:
        """Scalar history for logs and plateau schedules, bounded by the declared logging window.

        Running means retain their totals/counts; gradients and device tensors are never serialized.
        """
        self._materialize()
        return {
            "version": 1,
            "records": {
                group: {
                    name: {
                        "values": [float(value) for value in record._values],
                        "weights": [float(weight) for weight in record._weight],
                        "losses": [float(loss) for loss in record._losses],
                        "patches": list(record._patches),
                        "window": record._values.maxlen,
                        "mean": (float(record._mean.total), record._mean.count),
                        "mean_weight": (float(record._mean_weight.total), record._mean_weight.count),
                        "mean_loss": (float(record._mean_loss.total), record._mean_loss.count),
                        "recorded": record._recorded,
                    }
                    for name, record in records.items()
                }
                for group, records in self._loss.items()
            },
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore scalar histories into the same configured criteria, preserving a widened window."""
        if state.get("version") != 1 or state["records"].keys() != self._loss.keys():
            raise MeasureError("RESUME measure history does not match the configured criterion groups.")
        for group, records in self._loss.items():
            saved = state["records"][group]
            if saved.keys() != records.keys():
                raise MeasureError("RESUME measure history does not match the configured criteria.")
            for name, record in records.items():
                entry = saved[name]
                window = max(record._values.maxlen or 0, entry["window"] or len(entry["values"]), 1)
                record._values = deque(entry["values"], maxlen=window)
                record._weight = deque(entry["weights"], maxlen=window)
                # A history without minimized losses (written before they were kept) restores none:
                # a reported value is not what the criterion minimizes.
                record._losses = deque(entry.get("losses", ()), maxlen=window)
                # A history saved without patches weighed each value one.
                record._patches = deque(entry.get("patches", [1] * len(entry["values"])), maxlen=window)
                record._mean.total, record._mean.count = entry["mean"]
                record._mean_weight.total, record._mean_weight.count = entry["mean_weight"]
                record._mean_loss.total, record._mean_loss.count = entry.get("mean_loss", (0.0, 0))
                record._recorded = entry["recorded"]
                record._unread.clear()
                record.reset_loss()

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

    def get_last_losses(self, n: int = 1) -> dict[str, float]:
        """The minimized value per criterion, lower is better whatever the criterion reports: what a
        plateau schedule steps on and what selects a checkpoint."""
        return {name: record.loss_mean(n) for name, record in self._read(n)}

    def learns(self, n: int) -> bool:
        """Whether any loss minimized a finite value over the last ``n`` steps: ``False`` once every step
        of the window is NaN or infinite. A window not yet full is not judged, and metrics do not count."""
        losses = [record for _, record in self._read(n) if record.is_loss]
        return not losses or any(record.learns(n) for record in losses)

    def format_loss(self, is_loss: bool, n: int) -> dict[str, tuple[float, float, float]]:
        """Per criterion: the mean weight, the reported value (the board's), the minimized value."""
        return {
            name: (weight, _ratio(values), _ratio(losses))
            for name, (weight, values, losses) in self.format_totals(is_loss, n).items()
        }

    def format_totals(self, is_loss: bool, n: int) -> dict[str, tuple[float, tuple[float, int], tuple[float, int]]]:
        """``format_loss`` before its two divisions: the reported and minimized values as (sum, patches),
        which ranks add up."""
        return {
            name: (record.weights_mean(n), record.values_total(n), record.losses_total(n))
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
        # through and clamps to the last scheduler, stepped from its window's start.
        step = start = 0
        _scheduler: Scheduler | None = None
        for _scheduler, value in schedulers.items():
            start = step
            if value is None or (it >= step and it < step + value):
                break
            step += value
        if _scheduler is None:  # unreachable (schedulers is non-empty); kept for type-narrowing
            raise ConfigError(f"No scheduler matched iteration {it}.")
        _scheduler.step(it - start)
        return _scheduler
