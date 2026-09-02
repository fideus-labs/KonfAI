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


"""The criterion contract: losses and metrics, with or without init and attributes; masking."""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import torch

from konfai.network.network import Network
from konfai.network.network.measure import CriterionOutput as CriterionOutput
from konfai.network.network.measure import CriterionResult as CriterionResult
from konfai.network.network.measure import CriterionValue as CriterionValue
from konfai.network.network.measure import LabelledValues as LabelledValues
from konfai.utils.config import record_given_arguments
from konfai.utils.dataset import Attribute
from konfai.utils.errors import MeasureError

models_register: dict[str, Network] = {}


def _require_optional(module: str, *, criterion: str, extra: str) -> ModuleType:
    """Import an optional criterion dependency or raise an actionable error.

    Several criteria (LPIPS, the IMPACT family) rely on heavyweight optional packages
    that are not part of the base install. Importing them through this helper
    turns a missing dependency into a clear, install-ready message raised at
    criterion construction, instead of a raw ``ImportError`` surfacing mid-run.
    """
    package = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MeasureError(
            f"The '{criterion}' criterion requires the optional dependency '{package}'.",
            f"Install it with `pip install konfai[{extra}]` (or `pip install {package}`).",
        ) from exc


class Criterion(torch.nn.Module, ABC):
    # Natural optimisation direction of this criterion's reported value: False = lower-is-better
    # (the default: losses and distances), True = higher-is-better (score-style metrics like Dice).
    # It is a property of the criterion, not a global mode: consumers (leaderboard ranking, best-metric
    # selection) read it via getattr instead of guessing the direction from the metric's name.
    maximize: bool = False

    # Streamed-evaluation contract, the metric mirror of ``Reduction.voxel_local``: ``True`` declares
    # that this metric's whole-case value can be rebuilt from per-patch PARTIAL states (running sums,
    # never per-patch final values), so evaluation may feed it disjoint patches instead of the whole
    # volume. Default ``False``: an unknown metric evaluates whole: a wrong ``True`` would corrupt
    # the reported value, so only a metric whose ``partial_metric``/``combine_metric`` reproduce
    # ``forward`` exactly may set it.
    reducible: bool = False

    # Voxels of context a partial state needs past a patch's faces on every spatial axis (a window's
    # radius). A reducible metric declaring one is handed patches read that much wider than their
    # grid slot, clamped at the volume's faces, and ``partial_metric`` receives ``core=``, the slot's
    # slices within the patch: it scores the core through the context and nothing outside it.
    halo: int = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        # A metric is config-built like a stage: record its constructor arguments as given, so
        # konfai.api can write the config tree back from live objects (see Transform).
        super().__init_subclass__(**kwargs)
        record_given_arguments(cls)

    def __init__(self) -> None:
        super().__init__()

    def get_name(self):
        return self.__class__.__name__

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> Any:
        """Sufficient statistics of one disjoint patch (only meaningful when ``reducible``)."""
        raise NotImplementedError(f"{self.get_name()} is not reducible: it has no partial state.")

    def combine_metric(self, states: list[Any]) -> Any:
        """Combine per-patch states into exactly what ``forward`` returns on the whole volume."""
        raise NotImplementedError(f"{self.get_name()} is not reducible: it cannot combine states.")

    @abstractmethod
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> CriterionOutput:
        """A loss ``Tensor``, or ``(loss, value)`` / ``(loss, value, map)``: every accepted shape
        is normalized by ``CriterionResult.of`` at the consumers."""
        raise NotImplementedError()


class CriterionWithInit(Criterion):
    accepts_init = True

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def init(self, model: torch.nn.Module, output_group: str, target_group: str) -> str:
        raise NotImplementedError()


class CriterionWithAttribute(Criterion):
    accepts_attributes = True

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(  # type: ignore[override]  # the added keyword is this subclass's contract
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> CriterionOutput:
        raise NotImplementedError()


class MaskedLoss(Criterion):
    def __init__(
        self,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mode_image_masked: bool,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.mode_image_masked = mode_image_masked

    @staticmethod
    def get_mask(targets: list[torch.Tensor]) -> torch.Tensor | None:
        if len(targets) == 0:
            return None

        mask = targets[0]
        for target in targets[1:]:
            mask = mask * target

        return mask

    def _kernel(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        """Per-voxel contribution whose masked per-item sums reproduce ``self.loss`` through
        ``_value``; ``None`` routes the masked forward through the generic per-item loop (a loss
        that is not a pointwise reduction)."""
        return None

    def _value(self, total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """The per-item values ``self.loss`` returns from per-item (total, count): the batched
        twin of ``_finish``."""
        raise NotImplementedError()

    def _masked_forward(self, kernel: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The masked loss batched: per-item masked sums over the flattened axes, finished by
        ``_value``, averaged over the items whose mask holds a voxel. No host readout: an empty
        item is neutralized on the device (its value multiplied by zero, never NaN, so backward
        stays finite) and the reported value is NaN only when every item is empty."""
        items = kernel.shape[0]
        per_item = (kernel * mask).reshape(items, -1).sum(1)
        counts = mask.reshape(items, -1).sum(1) * (kernel[0].numel() // mask[0].numel())
        scored = counts > 0
        values = self._value(torch.where(scored, per_item, per_item.new_ones(())), counts.clamp(min=1))
        scored_nb = scored.sum()
        loss = (values * scored).sum() / scored_nb.clamp(min=1)
        return loss, torch.where(scored_nb > 0, loss, loss.new_tensor(float("nan"))).detach()

    def forward(
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
    ) -> CriterionOutput:

        if len(targets) == 0:
            raise ValueError("MaskedLoss expects at least one target tensor.")

        target = targets[0]
        mask = self.get_mask(list(targets[1:]))

        if mask is None:
            loss_b = self.loss(
                output.float(),
                target.to(device=output.device).float(),
            )
            return loss_b, loss_b.detach()

        target = target.to(device=output.device)
        mask = mask.to(device=output.device) == 1

        kernel = None if self.mode_image_masked else self._kernel(output.float(), target.float())
        if kernel is not None:
            return self._masked_forward(kernel, mask)

        # One readout for the whole batch, where a per-item ``torch.any`` was one sync each.
        scored = torch.any(mask.reshape(mask.shape[0], -1), dim=1).tolist()
        loss = output.new_tensor(0.0)
        for batch in range(output.shape[0]):
            if not scored[batch]:
                continue

            mask_b = mask[batch, ...]
            output_b = output[batch, ...].float()
            target_b = target[batch, ...].float()

            if self.mode_image_masked:
                mask_f = mask_b.to(dtype=output_b.dtype)

                loss_b = self.loss(
                    output_b * mask_f,
                    target_b * mask_f,
                )

            else:
                loss_b = self.loss(
                    torch.masked_select(output_b, mask_b),
                    torch.masked_select(target_b, mask_b),
                )

            loss = loss + loss_b

        true_nb = sum(scored)
        if true_nb == 0:
            return loss, np.nan

        loss = loss / true_nb
        return loss, loss.detach()

    # . Streamed-evaluation hooks -------------------------------------------------------------------
    # A subclass whose ``loss`` reduces to a running sum provides its sufficient statistic and its
    # finisher, and declares itself ``reducible``; the generic partial/combine below then reproduces
    # ``forward`` exactly from disjoint patches (masked and unmasked paths alike).

    def _stat(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Sum-contribution of one (output, target) pair to this loss's running total."""
        kernel = self._kernel(x, y)
        if kernel is None:
            raise NotImplementedError()
        return float(kernel.sum().item())

    def _finish(self, total: float, count: int) -> float:
        """The value ``self.loss`` would return from a running (total, count)."""
        raise NotImplementedError()

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> Any:
        if len(targets) == 0:
            raise ValueError("MaskedLoss expects at least one target tensor.")
        target = targets[0].to(device=output.device)
        mask = self.get_mask(list(targets[1:]))
        if mask is None:
            x, y = output.float(), target.float()
            return ("whole", self._stat(x, y), x.numel())
        mask = mask.to(device=output.device)
        items = []
        for batch in range(output.shape[0]):
            mask_b = mask[batch, ...] == 1
            if not torch.any(mask_b):
                items.append((0.0, 0, False))
                continue
            output_b, target_b = output[batch, ...].float(), target[batch, ...].float()
            if self.mode_image_masked:
                mask_f = mask_b.to(dtype=output_b.dtype)
                items.append((self._stat(output_b * mask_f, target_b * mask_f), output_b.numel(), True))
            else:
                items.append(
                    (
                        self._stat(torch.masked_select(output_b, mask_b), torch.masked_select(target_b, mask_b)),
                        int(mask_b.sum().item()),
                        True,
                    )
                )
        return ("items", items)

    def combine_metric(self, states: list[Any]) -> Any:
        if states[0][0] == "whole":
            total = sum(state[1] for state in states)
            count = sum(state[2] for state in states)
            value = self._finish(total, count)
            return torch.tensor(value), value
        # Masked: sum each batch item's statistic across patches, finish per item, then average the
        # items that saw any masked voxel: the exact structure of ``forward``.
        n_items = len(states[0][1])
        values = []
        for item in range(n_items):
            total = sum(state[1][item][0] for state in states)
            count = sum(state[1][item][1] for state in states)
            if any(state[1][item][2] for state in states):
                values.append(self._finish(total, count))
        if not values:
            return torch.tensor(0.0), np.nan
        value = float(np.mean(values))
        return torch.tensor(value), value
