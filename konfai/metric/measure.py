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

"""Criterion and metric implementations used by KonfAI workflows."""

import copy
import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from konfai.data.patching import ModelPatch
from konfai.network.blocks import LatentDistribution
from konfai.network.network import ModelLoader, Network
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute
from konfai.utils.errors import MeasureError
from konfai.utils.utils import get_module

models_register: dict[str, Network] = {}


def _require_optional(module: str, *, criterion: str, extra: str) -> ModuleType:
    """Import an optional criterion dependency or raise an actionable error.

    Several criteria (SSIM, LPIPS, FID) rely on heavyweight optional packages
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
    # (the default -- losses and distances), True = higher-is-better (score-style metrics like Dice).
    # It is a property of the criterion, not a global mode: consumers (leaderboard ranking, best-metric
    # selection) read it via getattr instead of guessing the direction from the metric's name.
    maximize: bool = False

    # Streamed-evaluation contract, the metric mirror of ``Reduction.voxel_local``: ``True`` declares
    # that this metric's whole-case value can be rebuilt from per-patch PARTIAL states (running sums,
    # never per-patch final values), so evaluation may feed it disjoint patches instead of the whole
    # volume. Default ``False``: an unknown metric evaluates whole -- a wrong ``True`` would corrupt
    # the reported value, so only a metric whose ``partial_metric``/``combine_metric`` reproduce
    # ``forward`` exactly may set it.
    reducible: bool = False

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
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
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
    def forward(  # type: ignore[override]
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> torch.Tensor:
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

    def forward(
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:

        if len(targets) == 0:
            raise ValueError("MaskedLoss expects at least one target tensor.")

        target = targets[0]
        mask = self.get_mask(list(targets[1:]))

        loss = output.new_tensor(0.0)
        true_nb = 0

        if mask is None:
            loss_b = self.loss(
                output.float(),
                target.to(device=output.device).float(),
            )
            return loss_b, loss_b.detach().item()

        target = target.to(device=output.device)
        mask = mask.to(device=output.device)

        for batch in range(output.shape[0]):
            mask_b = mask[batch, ...] == 1

            if not torch.any(mask_b):
                continue

            output_b = output[batch, ...].float()
            target_b = target[batch, ...].float()

            if self.mode_image_masked:
                mask_b = mask_b.to(dtype=output_b.dtype)

                loss_b = self.loss(
                    output_b * mask_b,
                    target_b * mask_b,
                )

            else:
                loss_b = self.loss(
                    torch.masked_select(output_b, mask_b),
                    torch.masked_select(target_b, mask_b),
                )

            loss = loss + loss_b
            true_nb += 1

        if true_nb == 0:
            return loss, np.nan

        loss = loss / true_nb
        return loss, loss.detach().item()

    # -- Streamed-evaluation hooks -------------------------------------------------------------------
    # A subclass whose ``loss`` reduces to a running sum provides its sufficient statistic and its
    # finisher, and declares itself ``reducible``; the generic partial/combine below then reproduces
    # ``forward`` exactly from disjoint patches (masked and unmasked paths alike).

    def _stat(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Sum-contribution of one (output, target) pair to this loss's running total."""
        raise NotImplementedError()

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
        # items that saw any masked voxel -- the exact structure of ``forward``.
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


class MSE(MaskedLoss):
    @staticmethod
    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.MSELoss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MSE._loss, reduction), False)
        self._reduction = reduction
        self.reducible = reduction in ("mean", "sum")

    def _stat(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return float((x - y).pow(2).sum().item())

    def _finish(self, total: float, count: int) -> float:
        return total / count if self._reduction == "mean" else total


class MAE(MaskedLoss):
    @staticmethod
    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.L1Loss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MAE._loss, reduction), False)
        self._reduction = reduction
        self.reducible = reduction in ("mean", "sum")

    def _stat(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return float((x - y).abs().sum().item())

    def _finish(self, total: float, count: int) -> float:
        return total / count if self._reduction == "mean" else total


class ME(MaskedLoss):
    reducible = True

    @staticmethod
    def _loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).mean()

    def __init__(self) -> None:
        super().__init__(ME._loss, False)

    def _stat(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return float((x - y).sum().item())

    def _finish(self, total: float, count: int) -> float:
        return total / count


class MAESaveMap(MAE):
    def __init__(self, reduction: str = "mean", dataset: str | None = None, group: str | None = None) -> None:
        super().__init__(reduction)
        self.dataset = dataset
        self.group = group

    def partial_map(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        """Per-voxel |output - target| (masked where a mask is given). VOXEL-LOCAL by construction:
        a patch's map equals the same region of the whole-case map, which is what lets the streamed
        evaluation write it region by region instead of needing the whole case."""
        if len(targets) == 2:
            return (
                torch.nn.L1Loss(reduction="none")(
                    output.float() * torch.where(targets[1] == 1, 1, 0),
                    targets[0].float() * torch.where(targets[1] == 1, 1, 0),
                )
                .to(output.dtype)
                .cpu()
            )
        return torch.nn.L1Loss(reduction="none")(output.float(), targets[0].float()).to(output.dtype).cpu()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):  # type: ignore[override]
        loss, true_loss = super().forward(output, *targets)
        return loss, true_loss, self.partial_map(output, *targets)

    def get_name(self) -> str:
        return "MAE"


class PSNR(MaskedLoss):
    reducible = True

    @staticmethod
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((x - y).pow(2))
        psnr = 10 * torch.log10(dynamic_range**2 / mse)
        return psnr

    def __init__(self, dynamic_range: float | None = None) -> None:
        dynamic_range = dynamic_range if dynamic_range else 1024 + 3071
        super().__init__(partial(PSNR._loss, dynamic_range), False)
        self._dynamic_range = float(dynamic_range)

    def _stat(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return float((x - y).pow(2).sum().item())

    def _finish(self, total: float, count: int) -> float:
        # The log is a function of the RUNNING mean, applied once at the end -- never per patch.
        return float(10 * np.log10(self._dynamic_range**2 / (total / count)))


class SSIM(MaskedLoss):
    @staticmethod
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        structural_similarity = _require_optional(
            "skimage.metrics", criterion="SSIM", extra="ssim"
        ).structural_similarity
        value = structural_similarity(
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            data_range=dynamic_range,
            channel_axis=0,
            gradient=False,
            full=False,
        )
        return x.new_tensor(float(value))

    def __init__(self, dynamic_range: float | None = None) -> None:
        _require_optional("skimage.metrics", criterion="SSIM", extra="ssim")
        dynamic_range = dynamic_range if dynamic_range else 1024 + 3000
        super().__init__(partial(SSIM._loss, dynamic_range), True)

    def forward(
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if len(targets) == 1:
            targets = (targets[0], torch.ones_like(targets[0], dtype=torch.uint8))
        return super().forward(output, *targets)


class LPIPS(MaskedLoss):
    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)) * 2 - 1

    @staticmethod
    def preprocessing(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat((1, 3, 1, 1))

    @staticmethod
    def _loss(loss_fn_alex, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Follow the input's device (the DDP rank's GPU, or CPU) instead of a hardcoded device 0.
        loss_fn_alex = loss_fn_alex.to(x.device)
        dataset_patch = ModelPatch([1, 320, 320])
        dataset_patch.load(x.shape[2:])

        patch_iterator = dataset_patch.disassemble(LPIPS.normalize(x), LPIPS.normalize(y))
        loss = 0
        with tqdm(
            iterable=enumerate(patch_iterator),
            leave=False,
            total=dataset_patch.get_size(0),
        ) as batch_iter:
            for _, patch_input in batch_iter:
                real, fake = LPIPS.preprocessing(patch_input[0]), LPIPS.preprocessing(patch_input[1])
                loss += loss_fn_alex(real, fake).flatten()[0]
        return loss / dataset_patch.get_size(0)

    def __init__(self, model: str = "alex") -> None:
        lpips = _require_optional("lpips", criterion="LPIPS", extra="lpips")

        super().__init__(partial(LPIPS._loss, lpips.LPIPS(net=model)), True)


class TRE(Criterion):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):
        loss = torch.linalg.norm(output - targets[0], dim=2)
        return loss.mean(), {f"Landmarks_{i}": v.item() for i, v in enumerate(loss.mean(0))}


class Dice(Criterion):
    maximize = True  # reported value is the Dice coefficient (higher-is-better); DiceSaveMap inherits it

    @staticmethod
    def flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute((1, 0, *tuple(range(2, tensor.dim())))).contiguous().view(tensor.size(1), -1)

    @staticmethod
    def dice_per_channel(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tensor = Dice.flatten(tensor)
        target = Dice.flatten(target)
        return (2.0 * (tensor * target).sum() + 1e-6) / (tensor.sum() + target.sum() + 1e-6)

    @staticmethod
    def _loss(labels: list[int] | None, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(targets[0], output.shape[2:], mode="nearest")
        result = {}
        loss = torch.tensor(0, dtype=torch.float32).to(output.device)
        if labels is None:
            labels = [int(label) for label in torch.unique(target) if int(label) != 0]
        count = 0
        for label in labels:
            tp = target == label
            if tp.any().item():
                if output.shape[1] > 1:
                    pp = output[:, label].unsqueeze(1)
                else:
                    pp = output == label
                loss_tmp = Dice.dice_per_channel(pp.float(), tp.float())
                loss += loss_tmp
                count += 1
                result[label] = loss_tmp.item()
            else:
                result[label] = np.nan
        if count == 0:
            return loss, result
        return 1 - loss / count, result

    reducible = True

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self._labels = labels
        self.loss = partial(Dice._loss, labels)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        mask = MaskedLoss.get_mask(list(targets[1:]))
        if mask is not None:
            mask = torch.where(mask == 1, 1, 0)
            return self.loss(
                output * mask.to(output.dtype),
                targets[0] * mask.to(targets[0].dtype),
            )
        else:
            return self.loss(output, targets[0])

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> Any:
        mask = MaskedLoss.get_mask(list(targets[1:]))
        target = targets[0]
        if mask is not None:
            mask01 = torch.where(mask == 1, 1, 0)
            output = output * mask01.to(output.dtype)
            target = target * mask01.to(target.dtype)
        if tuple(target.shape[2:]) != tuple(output.shape[2:]):
            raise MeasureError(
                "Dice can only stream patches when output and target share the spatial grid",
                f"output {tuple(output.shape[2:])} vs target {tuple(target.shape[2:])}",
                "Evaluate this pair whole (unset the memory budget) or resample the prediction first.",
            )
        labels = self._labels if self._labels is not None else [int(v) for v in torch.unique(target) if int(v) != 0]
        state: dict[int, tuple[float, float, float, bool]] = {}
        for label in labels:
            tp = target == label
            pp = output[:, label].unsqueeze(1) if output.shape[1] > 1 else (output == label)
            ppf, tpf = pp.float(), tp.float()
            state[label] = (
                float((ppf * tpf).sum().item()),
                float(ppf.sum().item()),
                float(tpf.sum().item()),
                bool(tp.any().item()),
            )
        return state

    def combine_metric(self, states: list[Any]) -> Any:
        # The whole volume's label set is exactly the union of the patches' label sets (sorted, as
        # torch.unique returns them); every dice is the ratio of GLOBAL sums, the smooth term applied
        # once here -- never a mean of per-patch dices.
        labels = self._labels if self._labels is not None else sorted({label for state in states for label in state})
        result: dict[int, float] = {}
        total = 0.0
        count = 0
        for label in labels:
            inter = sum(state[label][0] for state in states if label in state)
            output_sum = sum(state[label][1] for state in states if label in state)
            target_sum = sum(state[label][2] for state in states if label in state)
            if any(state[label][3] for state in states if label in state):
                dice = (2.0 * inter + 1e-6) / (output_sum + target_sum + 1e-6)
                result[label] = dice
                total += dice
                count += 1
            else:
                result[label] = np.nan
        if count == 0:
            return torch.tensor(0.0), result
        return torch.tensor(1 - total / count), result


class DiceSaveMap(Dice):
    def __init__(self, labels: list[int] | None = None, dataset: str | None = None, group: str | None = None) -> None:
        super().__init__(labels)
        self.dataset = dataset
        self.group = group

    def partial_map(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        """Per-voxel label disagreement (masked where a mask is given). VOXEL-LOCAL by construction:
        a patch's map equals the same region of the whole-case map, which is what lets the streamed
        evaluation write it region by region instead of needing the whole case."""
        if len(targets) == 2:
            return (
                torch.nn.L1Loss(reduction="none")(
                    output * torch.where(targets[1] == 1, 1, 0), targets[0] * torch.where(targets[1] == 1, 1, 0)
                )
                .to(torch.uint8)
                .cpu()
            )
        return torch.nn.L1Loss(reduction="none")(output, targets[0]).to(torch.uint8).cpu()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):  # type: ignore[override]
        loss, true_loss = super().forward(output, *targets)
        return loss, true_loss, self.partial_map(output, *targets)

    def get_name(self) -> str:
        return "Dice"


class GradientImages(Criterion):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _image_gradient_2d(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return dx, dy

    @staticmethod
    def _image_gradient_3d(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
        return dx, dy, dz

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target_0 = targets[0]
        if len(output.shape) == 5:
            dx, dy, dz = GradientImages._image_gradient_3d(output)
            if target_0 is not None:
                dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient_3d(target_0)
                dx -= dx_tmp
                dy -= dy_tmp
                dz -= dz_tmp
            return dx.norm() + dy.norm() + dz.norm()
        else:
            dx, dy = GradientImages._image_gradient_2d(output)
            if target_0 is not None:
                dx_tmp, dy_tmp = GradientImages._image_gradient_2d(target_0)
                dx -= dx_tmp
                dy -= dy_tmp
            return dx.norm() + dy.norm()


class BCE(Criterion):
    def __init__(self, target: float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer("target", torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, target.to(output.device).expand_as(output))


class PatchGanLoss(Criterion):
    def __init__(self, target: float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.register_buffer("target", torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, (torch.ones_like(output) * target).to(output.device))


class WGP(Criterion):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((output - 1) ** 2)


class Gram(Criterion):
    @staticmethod
    def compute_gram(tensor: torch.Tensor):
        (_b, ch, w) = tensor.size()
        with torch.amp.autocast("cuda", enabled=False):
            return tensor.bmm(tensor.transpose(1, 2)).div(ch * w)

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction="sum")

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = targets[0]
        if len(output.shape) > 3:
            output = output.view(output.shape[0], output.shape[1], int(np.prod(output.shape[2:])))
        if len(target.shape) > 3:
            target = target.view(target.shape[0], target.shape[1], int(np.prod(target.shape[2:])))
        return self.loss(Gram.compute_gram(output), Gram.compute_gram(target))


class PerceptualLoss(Criterion):
    class Module:
        def __init__(self, losses: dict[str, float] = {"Gram": 1, "torch:nn:L1Loss": 1}) -> None:
            self.losses = losses
            self.konfai_args = os.environ["KONFAI_CONFIG_PATH"] if "KONFAI_CONFIG_PATH" in os.environ else ""

        def get_loss(self) -> dict[torch.nn.Module, float]:
            result: dict[torch.nn.Module, float] = {}
            for loss, loss_value in self.losses.items():
                module, name = get_module(loss, "konfai.metric.measure")
                result[apply_config(self.konfai_args)(getattr(module, name))()] = loss_value
            return result

    def __init__(
        self,
        model_loader: ModelLoader = ModelLoader(),
        path_model: str = "name",
        modules: dict[str, Module] = {
            "UNetBlock_0.DownConvBlock.Activation_1": Module({"Gram": 1, "torch:nn:L1Loss": 1})
        },
        shape: list[int] = [128, 128, 128],
    ) -> None:
        super().__init__()
        self.path_model = path_model
        if self.path_model not in models_register:
            self.model = model_loader.get_model(
                train=False,
                konfai_args=os.environ["KONFAI_CONFIG_PATH"].split("PerceptualLoss")[0] + "PerceptualLoss.Model",
                konfai_without=[
                    "optimizer",
                    "schedulers",
                    "nb_batch_per_step",
                    "init_type",
                    "init_gain",
                    "outputs_criterions",
                    "drop_p",
                ],
            )
            if path_model.startswith("https"):
                state_dict = torch.hub.load_state_dict_from_url(path_model)
                state_dict = {"Model": {self.model.get_name(): state_dict["model"]}}
            else:
                state_dict = torch.load(path_model, weights_only=True)
            self.model.load(state_dict)
            models_register[self.path_model] = self.model
        else:
            self.model = models_register[self.path_model]

        self.shape = shape
        self.mode = "trilinear" if len(shape) == 3 else "bilinear"
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}
        for name, losses in modules.items():
            self.modules_loss[name.replace(":", ".")] = losses.get_loss()

        self.model.eval()
        self.model.requires_grad_(False)
        self.models: dict[int, torch.nn.Module] = {}

    def preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _compute(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        output_preprocessing = self.preprocessing(output)
        targets_preprocessing = [self.preprocessing(target) for target in targets]
        for zipped_output in zip([output_preprocessing], *[[target] for target in targets_preprocessing], strict=False):
            output = zipped_output[0]
            targets = zipped_output[1:]

            for zipped_layers in list(
                zip(
                    self.models[output.device.index].get_layers([output], set(self.modules_loss.keys()).copy()),
                    *[
                        self.models[output.device.index].get_layers([target], set(self.modules_loss.keys()).copy())
                        for target in targets
                    ],
                    strict=False,
                )
            ):
                output_layer = zipped_layers[0][1].view(
                    zipped_layers[0][1].shape[0],
                    zipped_layers[0][1].shape[1],
                    int(np.prod(zipped_layers[0][1].shape[2:])),
                )
                # Apply every configured loss to every target layer. Zipping the losses against the
                # targets instead dropped losses whenever there were fewer targets than losses -- so the
                # default {Gram, L1Loss} on a single reference silently used only Gram.
                for target_entry in zipped_layers[1:]:
                    target_layer = target_entry[1].view(
                        target_entry[1].shape[0],
                        target_entry[1].shape[1],
                        int(np.prod(target_entry[1].shape[2:])),
                    )
                    for loss_function, loss_value in self.modules_loss[zipped_layers[0][0]].items():
                        loss = (
                            loss
                            + loss_value
                            * loss_function(output_layer.float(), target_layer.float())
                            / output_layer.shape[0]
                        )
        return loss

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        if output.device.index not in self.models:
            # `Network.to` resets its GPU-index counter per call, so the perceptual model is
            # placed starting at this device.
            self.models[output.device.index] = Network.to(copy.deepcopy(self.model).eval(), output.device.index).eval()
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and len(self.shape) == 2:
            for i in range(output.shape[2]):
                loss = loss + self._compute(output[:, :, i, ...], *[t[:, :, i, ...] for t in targets]) / output.shape[2]
        else:
            loss = self._compute(output, *targets)
        return loss.to(output)


class KLDivergence(CriterionWithInit):
    def __init__(self, shape: list[int], dim: int = 100, mu: float = 0, std: float = 1) -> None:
        super().__init__()
        self.latent_dim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
        self.modelDim = 3
        self.shape = shape
        self.loss = torch.nn.KLDivLoss()

    def init(self, model: Network, output_group: str, target_group: str) -> str:
        model._compute_channels_trace(model, model.in_channels, None, None)

        last_module = model
        for name in output_group.split(".")[:-1]:
            last_module = last_module[name]

        modules = last_module._modules.copy()
        last_module._modules.clear()

        for name, value in modules.items():
            last_module._modules[name] = value
            if name == output_group.split(".")[-1]:
                last_module.add_module(
                    "LatentDistribution",
                    LatentDistribution(shape=self.shape, latent_dim=self.latent_dim),
                )
        return ".".join(output_group.split(".")[:-1]) + ".LatentDistribution.Concat"

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        mu = output[:, 0, :]
        log_std = output[:, 1, :]
        return torch.mean(-0.5 * torch.sum(1 + log_std - mu**2 - torch.exp(log_std), dim=1), dim=0)


class Accuracy(Criterion):
    maximize = True  # reported value is the accuracy fraction (higher-is-better)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        # Return this batch's accuracy; the logging window means it over the batches and resets between
        # train and validation. Accumulating n/corrects on the instance instead reported one lifetime
        # fraction that blended every epoch and both splits.
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
        return (predicted == targets[0]).float().mean()


class TripletLoss(Criterion):
    def __init__(self) -> None:
        super().__init__()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.triplet_loss(output[0], output[1], output[2])


class L1LossRepresentation(Criterion):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def _variance(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.clamp(1 - torch.var(features, dim=0), min=0))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.loss(output[0], output[1]) + self._variance(output[0]) + self._variance(output[1])


class FocalLoss(Criterion):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] = [0.5, 2.0, 0.5, 0.5, 1],
        reduction: str = "mean",
    ):
        super().__init__()
        raw_alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = raw_alpha / raw_alpha.sum() * len(raw_alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(targets[0], output.shape[2:], mode="nearest").long()

        logpt = F.log_softmax(output, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, target)
        pt = pt.gather(1, target)

        # alpha[target] is already [B, 1, *spatial] (matching pt/logpt); do not add an axis.
        at = self.alpha.to(target.device)[target]
        loss = -at * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FID(Criterion):
    class InceptionV3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            torchvision_models = _require_optional("torchvision.models", criterion="FID", extra="fid")
            inception_v3 = torchvision_models.inception_v3
            Inception_V3_Weights = torchvision_models.Inception_V3_Weights

            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            self.model.fc = torch.nn.Identity()
            self.model.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    def __init__(self) -> None:
        super().__init__()
        _require_optional("scipy.linalg", criterion="FID", extra="fid")
        self.inception_model = FID.InceptionV3().cuda()

    @staticmethod
    def preprocess_images(image: torch.Tensor) -> torch.Tensor:
        # resize/normalise-with-mean-std live in torchvision.transforms.functional, not torch.nn.functional
        # (which has no ``resize`` and whose ``normalize`` takes no mean/std) -- the old calls raised at once.
        tvf = _require_optional("torchvision.transforms.functional", criterion="FID", extra="fid")
        resized = tvf.resize(image, [299, 299]).repeat((1, 3, 1, 1))
        return tvf.normalize(resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def get_features(images: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
        with torch.no_grad():
            features = model(images).cpu().numpy()
        return features

    @staticmethod
    def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        mu2 = np.mean(generated_features, axis=0)
        sigma2 = np.cov(generated_features, rowvar=False)

        diff = mu1 - mu2
        linalg = _require_optional("scipy.linalg", criterion="FID", extra="fid")

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        real_images = FID.preprocess_images(targets[0].squeeze(0).permute([1, 0, 2, 3]))
        generated_images = FID.preprocess_images(output.squeeze(0).permute([1, 0, 2, 3]))

        real_features = FID.get_features(real_images, self.inception_model)
        generated_features = FID.get_features(generated_images, self.inception_model)

        return FID.calculate_fid(real_features, generated_features)


class MutualInformationLoss(torch.nn.Module):
    def __init__(
        self,
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        super().__init__()
        bin_centers = torch.linspace(0.0, 1.0, num_bins)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.num_bins = num_bins
        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_weight, pred_probability = self.parzen_windowing_gaussian(pred)
        target_weight, target_probability = self.parzen_windowing_gaussian(target)
        return pred_weight, pred_probability, target_weight, target_probability

    def parzen_windowing_gaussian(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(
            -self.preterm.to(img) * (img - self.bin_centers.to(img)) ** 2
        )  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        wa, pa, wb, pb = self.parzen_windowing(pred, targets[0])  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])  # (batch, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
        mi = torch.sum(
            pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr),
            dim=(1, 2),
        )  # (batch)
        return torch.mean(mi).neg()  # average over the batch and channel ndims


class CrossEntropyLoss(Criterion):
    def __init__(self, weight: list[float] | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight) if weight else None, reduction=reduction)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.loss(output, targets[0].squeeze(1))


class IMPACTReg(CriterionWithAttribute):
    class Weights:
        def __init__(self, weights: list[float] = [0, 1]) -> None:
            self.weights = weights

    def __init__(
        self,
        name: str = "Reg",
        model_name: str = "TS/M291.pt",
        shape: list[int] = [0, 0],
        in_channels: int = 3,
        loss: str = "torch:nn:L1Loss",
        weights: list[float] = [0, 1],
        pca: int = 0,
    ) -> None:
        super().__init__()
        if model_name is None:
            return
        self.name = name
        self.in_channels = in_channels
        self.nb_layer = len(weights)
        module, name = get_module(loss, "konfai.metric.measure")
        self.loss = apply_config(os.environ["KONFAI_CONFIG_PATH"])(getattr(module, name))()

        self.weights = weights
        self.pca = int(pca)
        self.model_path = hf_hub_download(
            repo_id="VBoussot/impact-torchscript-models", filename=model_name, repo_type="model", revision=None
        )  # nosec B615
        self.model: torch.nn.Module = torch.jit.load(self.model_path, map_location=torch.device("cpu"))  # nosec B614
        self.dim = len(shape)
        self.shape = shape if all(s > 0 for s in shape) else None
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}

        # The sanity check only probes the model's output structure and then discards it, so run it on the
        # CPU: forcing device 0 crashed on a CPU-only host and pinned every DDP rank to the same GPU.
        dummy_input = torch.zeros((1, self.in_channels, *(self.shape if self.shape else [224] * self.dim)))
        try:
            out = self.model(dummy_input, torch.tensor([self.nb_layer]))
            if not isinstance(out, (list, tuple)):
                raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
            if len(weights) != len(out):
                raise ValueError(
                    f"Loss '{loss}': mismatch between the number of weights "
                    f"({len(weights)}) and the number of model outputs "
                    f"({len(out)}). Each output must have a corresponding weight."
                )
        except Exception as e:
            msg = (
                f"[Model Sanity Check Failed]\n"
                f"Input shape attempted: {dummy_input.shape}\n"
                f"Error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e
        self.model = None

    def preprocessing(self, tensor: torch.Tensor, attribute: list[Attribute]) -> list[torch.Tensor]:
        if tensor.shape[1] != self.in_channels:
            tensor = tensor.repeat(tuple([1, 3] + [1 for _ in range(self.dim)]))

        return [
            tensor,
            torch.tensor([self.nb_layer]),
            torch.tensor(
                [
                    [
                        float(attr["ImageMin"]),
                        float(attr["ImageMean"]),
                        float(attr["ImageMax"]),
                        float(attr["ImageStd"]),
                    ]
                    for attr in attribute
                ]
            ),
        ]

    def get_name(self):
        return self.name

    @staticmethod
    def _pca_transform(feature: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Project a feature map ``[B, C, spatial...]`` onto a PCA basis ``[C, K]`` -> ``[B, K, spatial...]``,
        centring the input by its own per-channel mean first (itk-impact ``pca_transform``)."""
        shape = feature.shape
        flat = feature.reshape(shape[0], shape[1], -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        projected = torch.einsum("bcn,ck->bkn", flat, basis)
        return projected.reshape(shape[0], basis.shape[1], *shape[2:])

    def _pca_project(
        self, output_feature: torch.Tensor, target_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce both feature maps to their top-``pca`` principal components. For every batch sample the
        basis is fitted on the TARGET (reference) features and reused for the output — a channel-covariance
        eigendecomposition (``eigh`` is ascending, so the largest components live at the end), which
        reproduces itk-impact's per-image ``pca_fit`` by construction. itk-impact fits one basis per image,
        and a batch mixes unrelated cases, so the basis is fitted per sample: a shared basis would project
        every sample after the first into another case's feature space."""
        channels = target_feature.shape[1]
        k = min(self.pca, channels)
        flat = target_feature.detach().reshape(target_feature.shape[0], channels, -1).float()
        projected_output: list[torch.Tensor] = []
        projected_target: list[torch.Tensor] = []
        for b in range(flat.shape[0]):
            centered = flat[b] - flat[b].mean(dim=1, keepdim=True)
            covariance = centered @ centered.t() / max(flat.shape[2] - 1, 1)
            _, eigenvectors = torch.linalg.eigh(covariance)
            basis = eigenvectors[:, channels - k :].to(target_feature.dtype)  # {C, K}, largest-eigenvalue
            projected_output.append(self._pca_transform(output_feature[b : b + 1], basis))
            projected_target.append(self._pca_transform(target_feature[b : b + 1], basis))
        return torch.cat(projected_output), torch.cat(projected_target)

    def _compute(
        self,
        output: torch.Tensor,
        output_attributes: list[Attribute],
        target: torch.Tensor,
        target_attributes: list[Attribute],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)

        output = self.preprocessing(output, output_attributes)
        target = self.preprocessing(target, target_attributes)

        true_nb = 0

        if self.shape is not None:
            model_patch = ModelPatch(self.shape)
            model_patch.load(output[0].shape[2:])
            for index in range(model_patch.get_size(0)):
                mask_patch = model_patch.get_data(mask, index, 0, True) if mask is not None else None

                if mask is None or (mask_patch is not None and torch.any(mask_patch == 1)):
                    for i, zipped_output in enumerate(
                        zip(
                            self.model(model_patch.get_data(output[0], index, 0, True), output[1], output[2]),
                            self.model(model_patch.get_data(target[0], index, 0, True), target[1], target[2]),
                            strict=False,
                        )
                    ):
                        if self.weights[i] == 0:
                            continue
                        output_feature = zipped_output[0]
                        target_feature = zipped_output[1]
                        if self.pca > 0:
                            output_feature, target_feature = self._pca_project(output_feature, target_feature)
                        if mask is not None:
                            if mask_patch is None:
                                raise RuntimeError("LPIPS mask patch is unexpectedly missing.")
                            mask_patch_tensor = cast(torch.Tensor, mask_patch)
                            mask_index_resampled = (
                                torch.nn.functional.interpolate(
                                    mask_patch_tensor.float(), mode="nearest", size=tuple(output_feature.shape[2:])
                                ).repeat((1, output_feature.shape[1], *([1] * self.dim)))
                                == 1
                            )
                            if torch.any(mask_index_resampled):
                                loss_value = self.weights[i] * self.loss(
                                    torch.masked_select(output_feature, mask_index_resampled).float(),
                                    torch.masked_select(target_feature, mask_index_resampled).float(),
                                )
                                if loss_value.isnan():
                                    continue
                                loss += loss_value
                            else:
                                continue
                        else:
                            loss_value = self.weights[i] * self.loss(output_feature.float(), target_feature.float())
                            if loss_value.isnan():
                                continue
                            loss += loss_value
                    true_nb += 1
        else:
            if mask is None or torch.any(mask == 1):
                for i, zipped_output in enumerate(zip(self.model(*output), self.model(*target), strict=False)):
                    if self.weights[i] == 0:
                        continue
                    output_feature = zipped_output[0]
                    target_feature = zipped_output[1]
                    if self.pca > 0:
                        output_feature, target_feature = self._pca_project(output_feature, target_feature)
                    if mask is not None:
                        mask_index_resampled = (
                            torch.nn.functional.interpolate(
                                mask.float(), mode="nearest", size=tuple(output_feature.shape[2:])
                            ).repeat((1, output_feature.shape[1], *([1] * self.dim)))
                            == 1
                        )
                        if torch.any(mask_index_resampled):
                            loss += self.weights[i] * self.loss(
                                torch.masked_select(output_feature, mask_index_resampled).float(),
                                torch.masked_select(target_feature, mask_index_resampled).float(),
                            )
                        else:
                            true_nb -= 1
                    else:
                        loss += self.weights[i] * self.loss(output_feature.float(), target_feature.float())
                true_nb += 1
        return loss, true_nb

    def forward(  # type: ignore[override]
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> tuple[torch.Tensor, float]:
        mask = targets[-1] if targets[-1].dtype == torch.uint8 else None

        # The prediction and the target share the same intensity space, so a single target attribute
        # (single-group target such as ``CT``) is reused to normalize both output and target; a second
        # attribute set is honored when the target is multi-group.
        output_attributes = attributes[0]
        target_attributes = attributes[1] if len(attributes) > 1 else attributes[0]

        if self.model is None:
            self.model = torch.jit.load(self.model_path)  # nosec B614
        self.model.to(output.device)
        self.model.eval()

        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and self.dim == 2:
            true_nb = 0
            for i in range(output.shape[2]):
                loss_tmp, true_nb_tmp = self._compute(
                    output[:, :, i, ...],
                    output_attributes,
                    targets[0][:, :, i, ...],
                    target_attributes,
                    mask[:, :, i, ...] if mask is not None else None,
                )
                loss += loss_tmp
                true_nb += true_nb_tmp
        else:
            loss, true_nb = self._compute(output, output_attributes, targets[0], target_attributes, mask)
        return loss / true_nb, np.nan if true_nb == 0 else loss.item() / true_nb


class IMPACTSynth(CriterionWithAttribute):
    class Weights:
        def __init__(self, weights: list[float] = [0, 1]) -> None:
            self.weights = weights

    def _test_model(self, model_path_content: str, in_channels: int, shape: list[int], weights: list[float]):
        model: torch.nn.Module = torch.jit.load(model_path_content, map_location=torch.device("cpu"))  # nosec B614

        dummy_input = torch.zeros((1, in_channels, *shape)).to(0)
        try:
            out = model.to(0)(dummy_input, torch.tensor([len(weights)]))
            if not isinstance(out, (list, tuple)):
                raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
            if len(weights) != len(out):
                raise ValueError(
                    f"Loss '{model_path_content}': mismatch between the number of weights "
                    f"({len(weights)}) and the number of model outputs "
                    f"({len(out)}). Each output must have a corresponding weight."
                )
        except Exception as e:
            msg = (
                f"[Model Sanity Check Failed]\n"
                f"Input shape attempted: {dummy_input.shape}\n"
                f"Error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e

    def __init__(
        self,
        model_content_name: str,
        model_style_name: str,
        shape_content: list[int] = [0, 0],
        shape_style: list[int] = [0, 0],
        in_channels_content: int = 1,
        in_channels_style: int = 1,
        weights_criterion_content: list[float] = [0, 0, 1],
        weights_criterion_style: list[float] = [1, 1, 1],
    ) -> None:
        super().__init__()
        if model_content_name is None:
            return
        self.in_channels_content = in_channels_content
        self.in_channels_style = in_channels_style

        self.weights_criterion_content = weights_criterion_content
        self.weights_criterion_style = weights_criterion_style

        self.loss_content_function = torch.nn.MSELoss()
        self.loss_style_function = Gram()

        self.model_path_content = hf_hub_download(
            repo_id="VBoussot/impact-torchscript-models", filename=model_content_name, repo_type="model", revision=None
        )  # nosec B615

        self.model_path_style = hf_hub_download(
            repo_id="VBoussot/impact-torchscript-models", filename=model_style_name, repo_type="model", revision=None
        )  # nosec B615

        self.shape_content = shape_content if all(s > 0 for s in shape_content) else None
        self.shape_style = shape_style if all(s > 0 for s in shape_style) else None

        self._test_model(
            self.model_path_content,
            self.in_channels_content,
            self.shape_content if self.shape_content else [224] * len(shape_content),
            weights_criterion_content,
        )
        self._test_model(
            self.model_path_style,
            self.in_channels_style,
            self.shape_style if self.shape_style else [224] * len(shape_style),
            weights_criterion_style,
        )
        self.model_content: torch.nn.Module | None = None
        self.model_style: torch.nn.Module | None = None

    def _preprocessing(
        self, tensor: torch.Tensor, in_channels: int, nb_layer: int, attribute: list[Attribute]
    ) -> list[torch.Tensor]:
        if tensor.shape[1] != in_channels:
            tensor = tensor.repeat(tuple([1, in_channels] + [1 for _ in range(tensor.dim() - 2)]))

        if "Mean" in attribute[0] and "Std" in attribute[0]:
            mean_value = torch.tensor([float(a["Mean"]) for a in attribute], device=tensor.device).view(
                -1, *([1] * (tensor.dim() - 1))
            )
            std_value = torch.tensor([float(a["Std"]) for a in attribute], device=tensor.device).view(
                -1, *([1] * (tensor.dim() - 1))
            )
            tensor = tensor * std_value + mean_value
        elif "Min" in attribute[0] and "Max" in attribute[0]:
            min_value = torch.tensor([float(a["Min"]) for a in attribute], device=tensor.device).view(
                -1, *([1] * (tensor.dim() - 1))
            )
            max_value = torch.tensor([float(a["Max"]) for a in attribute], device=tensor.device).view(
                -1, *([1] * (tensor.dim() - 1))
            )
            tensor = (tensor + 1) / 2 * (max_value - min_value) + min_value

        return [
            tensor,
            torch.tensor([nb_layer]),
            torch.tensor(
                [
                    [
                        float(attr["ImageMin"]),
                        float(attr["ImageMean"]),
                        float(attr["ImageMax"]),
                        float(attr["ImageStd"]),
                    ]
                    for attr in attribute
                ]
            ),
        ]

    def _loss_compute(
        self,
        tensor: list[torch.Tensor],
        target: list[torch.Tensor],
        weights: list[float],
        shape: list[int] | None,
        mask: torch.Tensor | None,
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
    ) -> tuple[torch.Tensor, int]:
        loss = torch.zeros((1), requires_grad=True).to(tensor[0].device, non_blocking=False).type(torch.float32)
        true_nb = 0
        if shape is not None:
            model_patch = ModelPatch(shape)
            model_patch.load(tensor[0].shape[2:])
            for index in range(model_patch.get_size(0)):
                mask_patch = model_patch.get_data(mask, index, 0, True) if mask is not None else None

                if mask is None or (mask_patch is not None and torch.any(mask_patch == 1)):
                    for output_feature, target_feature, weight in zip(
                        model(model_patch.get_data(tensor[0], index, 0, True), tensor[1], tensor[2]),
                        model(model_patch.get_data(target[0], index, 0, True), target[1], target[2]),
                        weights,
                        strict=False,
                    ):
                        if weight == 0:
                            continue
                        if mask is not None:
                            if mask_patch is None:
                                raise RuntimeError("IMPACTSynth mask patch is unexpectedly missing.")
                            mask_patch_tensor = cast(torch.Tensor, mask_patch)
                            mask_index_resampled = (
                                torch.nn.functional.interpolate(
                                    mask_patch_tensor.float(), mode="nearest", size=tuple(output_feature.shape[2:])
                                ).repeat((1, output_feature.shape[1], *([1] * (mask_patch_tensor.dim() - 2))))
                                == 1
                            )
                            if torch.any(mask_index_resampled):
                                loss += weight * loss_function(
                                    torch.masked_select(output_feature, mask_index_resampled).float(),
                                    torch.masked_select(target_feature, mask_index_resampled).float(),
                                )
                            else:
                                true_nb -= 1
                        else:
                            loss += weight * loss_function(output_feature.float(), target_feature.float())
                    true_nb += 1
        else:
            if mask is None or torch.any(mask == 1):
                for output_feature, target_feature, weight in zip(
                    model(*tensor), model(*target), weights, strict=False
                ):
                    if weight == 0:
                        continue
                    if mask is not None:
                        mask_index_resampled = (
                            torch.nn.functional.interpolate(
                                mask.float(), mode="nearest", size=tuple(output_feature.shape[2:])
                            ).repeat((1, output_feature.shape[1], *([1] * (mask.dim() - 2))))
                            == 1
                        )
                        if torch.any(mask_index_resampled):
                            loss += weight * loss_function(
                                torch.masked_select(output_feature, mask_index_resampled).float(),
                                torch.masked_select(target_feature, mask_index_resampled).float(),
                            )
                        else:
                            true_nb -= 1
                    else:
                        loss += weight * loss_function(output_feature.float(), target_feature.float())
                true_nb += 1
        return loss, true_nb

    def forward(  # type: ignore[override]
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> tuple[torch.Tensor, float]:
        if len(targets) < 2:
            raise ValueError("At least two target tensors are required.")

        if self.model_content is None:
            self.model_content = torch.jit.load(self.model_path_content, map_location=torch.device("cpu"))  # nosec B614
            self.model_content.eval()
        if self.model_style is None:
            self.model_style = torch.jit.load(self.model_path_style, map_location=torch.device("cpu"))  # nosec B614
            self.model_style.eval()
        model_content = self.model_content
        model_style = self.model_style
        if model_content is None or model_style is None:
            raise RuntimeError("IMPACTSynth models were not initialized correctly.")
        model_content.to(output.device)
        model_style.to(output.device)

        output_content = self._preprocessing(
            output, self.in_channels_content, len(self.weights_criterion_content), attributes[0]
        )
        output_style = self._preprocessing(
            output, self.in_channels_style, len(self.weights_criterion_style), attributes[2]
        )

        target_content = self._preprocessing(
            targets[0], self.in_channels_content, len(self.weights_criterion_content), attributes[1]
        )
        target_style = self._preprocessing(
            targets[1], self.in_channels_style, len(self.weights_criterion_style), attributes[2]
        )

        mask = targets[2] if len(targets) == 3 and targets[2].dtype == torch.uint8 else None

        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and len(self.weights_criterion_content) == 2:
            true_nb = 0
            for i in range(output.shape[2]):
                loss_content, true_nb_content = self._loss_compute(
                    self._preprocessing(
                        output[:, :, i, ...],
                        self.in_channels_content,
                        len(self.weights_criterion_content),
                        attributes[0],
                    ),
                    target_content,
                    self.weights_criterion_content,
                    self.shape_content,
                    mask[:, :, i, ...] if mask is not None else None,
                    model=model_content,
                    loss_function=self.loss_content_function,
                )
                loss += loss_content
                true_nb += true_nb_content
        else:
            loss_content, true_nb_content = self._loss_compute(
                output_content,
                target_content,
                self.weights_criterion_content,
                self.shape_content,
                mask if mask is not None else None,
                model=model_content,
                loss_function=self.loss_content_function,
            )
            loss = loss_content
            true_nb = true_nb_content

        if len(output.shape) == 5 and len(self.weights_criterion_style) == 2:
            true_nb = 0
            for i in range(output.shape[2]):
                loss_style, true_nb_style = self._loss_compute(
                    self._preprocessing(
                        output[:, :, i, ...], self.in_channels_style, len(self.weights_criterion_style), attributes[2]
                    ),
                    target_style,
                    self.weights_criterion_style,
                    self.shape_style,
                    mask[:, :, i, ...] if mask is not None else None,
                    model=model_style,
                    loss_function=self.loss_style_function,
                )
                loss += loss_style
                true_nb += true_nb_style
        else:
            loss_style, true_nb_style = self._loss_compute(
                output_style,
                target_style,
                self.weights_criterion_style,
                self.shape_style,
                mask if mask is not None else None,
                model=model_style,
                loss_function=self.loss_style_function,
            )
            loss += loss_style
            true_nb += true_nb_style
        return loss / true_nb, np.nan if true_nb == 0 else loss.item() / true_nb


class SAM_Perceptual(CriterionWithAttribute):
    """SAM-feature perceptual criterion usable both as a metric and as a training loss.

    With ``train=False`` (a **metric**) it uses the metric-tuned model
    ``VBoussot/ImpactSynth/<model_name>`` over all feature layers. With ``train=True`` (a **loss**) it
    uses the raw feature extractor ``VBoussot/impact-torchscript-models`` / ``SAM2.1/<model_name>`` and
    applies per-layer ``weights`` (e.g. ``[0, 1, 1, 0]``); a weight of ``0`` skips that layer.
    """

    def __init__(
        self,
        train: bool = False,
        model_name: str = "SAM2.1_Small.pt",
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.model: torch.nn.Module | None = None
        self.loss = torch.nn.L1Loss()
        self.weights = weights
        self.nb_layer = len(weights) if weights is not None else 4
        if train:
            repo_id, filename = "VBoussot/impact-torchscript-models", f"SAM2.1/{model_name}"
        else:
            repo_id, filename = "VBoussot/ImpactSynth", model_name
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", revision=None)  # nosec B615

    def preprocessing(self, tensor: torch.Tensor, attribute: list[Attribute]) -> list[torch.Tensor]:
        tensor = tensor.repeat(1, 3, 1, 1)
        return [
            tensor,
            torch.tensor([self.nb_layer]),
            torch.tensor(
                [
                    [
                        float(attr["ImageMin"]),
                        float(attr["ImageMean"]),
                        float(attr["ImageMax"]),
                        float(attr["ImageStd"]),
                    ]
                    for attr in attribute
                ]
            ),
        ]

    def _compute(
        self, output: torch.Tensor, target: torch.Tensor, target_attributes: list[Attribute], mask: torch.Tensor | None
    ) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        model = self.model
        if model is None:
            raise RuntimeError("SAM perceptual model is not initialized.")

        output = self.preprocessing(output, target_attributes)
        target = self.preprocessing(target, target_attributes)
        true_nb = 0
        model_patch = ModelPatch([512, 512])
        model_patch.load(output[0].shape[2:])
        for index in range(model_patch.get_size(0)):
            mask_patch = model_patch.get_data(mask, index, 0, True) if mask is not None else None

            if mask is None or (mask_patch is not None and torch.any(mask_patch == 1)):
                for i, zipped_output in enumerate(
                    zip(
                        model(model_patch.get_data(output[0], index, 0, True), output[1], output[2]),
                        model(model_patch.get_data(target[0], index, 0, True), target[1], target[2]),
                        strict=False,
                    )
                ):
                    weight = self.weights[i] if self.weights is not None else 1.0
                    if weight == 0:
                        continue
                    output_feature = zipped_output[0]
                    target_feature = zipped_output[1]
                    if mask_patch is not None:
                        mask_patch_tensor = cast(torch.Tensor, mask_patch)
                        mask_index_resampled = (
                            torch.nn.functional.interpolate(
                                mask_patch_tensor.float(), mode="nearest", size=tuple(output_feature.shape[2:])
                            ).repeat((1, output_feature.shape[1], 1, 1))
                            == 1
                        )
                        if torch.any(mask_index_resampled):
                            loss += weight * self.loss(
                                torch.masked_select(output_feature, mask_index_resampled).float(),
                                torch.masked_select(target_feature, mask_index_resampled).float(),
                            )
                        else:
                            continue
                    else:
                        loss += weight * self.loss(output_feature.float(), target_feature.float())
                true_nb += 1
        return loss, true_nb

    def forward(  # type: ignore[override]
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> tuple[torch.Tensor, float]:
        mask = targets[-1] if targets[-1].dtype == torch.uint8 else None
        # ``targets[0]`` is the reference (e.g. CT), normalized with its own stats; the same stats
        # normalize the prediction since both live in the same intensity space.
        target_attributes = attributes[0]

        if self.model is None:
            self.model = torch.jit.load(self.model_path, map_location=torch.device("cpu"))  # nosec B614
        model = self.model
        if model is None:
            raise RuntimeError("SAM perceptual model failed to load.")
        model.eval()
        model.to(output.device)

        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5:
            true_nb = 0
            for i in range(output.shape[2]):
                loss_tmp, true_nb_tmp = self._compute(
                    output[:, :, i, ...],
                    targets[0][:, :, i, ...],
                    target_attributes,
                    mask[:, :, i, ...] if mask is not None else None,
                )
                loss += loss_tmp
                true_nb += true_nb_tmp
        else:
            loss, true_nb = self._compute(output, targets[0], target_attributes, mask)
        return loss / true_nb, np.nan if true_nb == 0 else loss.item() / true_nb


class Variance(Criterion):
    def __init__(self, name: str = "Variance") -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        output = output.float()
        if output.shape[1] > 1:
            variance = output.var(1).mean()
        else:
            variance = torch.zeros((), device=output.device, dtype=output.dtype)
        return variance, variance.item()


class Mean(Criterion):
    def __init__(self, name: str = "Mean") -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        loss = output.float().mean()
        return loss, loss.item()
