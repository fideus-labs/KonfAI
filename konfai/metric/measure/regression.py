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


"""Pointwise, structural and information criteria over intensities."""

from collections.abc import Iterator
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from konfai.data.patching import ModelPatch
from konfai.metric.measure.base import Criterion, CriterionWithInit, MaskedLoss, _require_optional
from konfai.metric.measure.segmentation import Dice
from konfai.network.blocks import LatentDistribution
from konfai.network.network import Network
from konfai.utils.errors import MeasureError


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

    @staticmethod
    def _difference(output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-voxel ``|output - target|``, zero outside the mask when one is given, and the bool mask.

        The one buffer both readouts come from: ``L1Loss`` reduces exactly this tensor, so the
        scalar and the map agree by construction. The mask multiplies in place as bool: one byte
        per voxel, once.
        """
        target = targets[0].to(device=output.device)
        difference = (output.float() - target.float()).abs()
        mask = MaskedLoss.get_mask(list(targets[1:]))
        if mask is None:
            return difference, None
        mask = mask.to(device=output.device) == 1
        return difference.mul_(mask), mask

    @staticmethod
    def _reduce(difference: torch.Tensor, reduction: str) -> torch.Tensor:
        """What ``L1Loss(reduction)`` returns from the differences it reduces: the same kernel."""
        if reduction == "mean":
            return difference.mean()
        if reduction == "sum":
            return difference.sum()
        return difference

    def partial_map(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        """Per-voxel ``|output - target|`` (masked where a mask is given). VOXEL-LOCAL by construction:
        a patch's map equals the same region of the whole-case map, which is what lets the streamed
        evaluation write it region by region instead of needing the whole case."""
        return self._difference(output, *targets)[0].to(output.dtype).cpu()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):  # type: ignore[override]
        difference, mask = self._difference(output, *targets)
        if mask is None:
            loss = self._reduce(difference, self._reduction)
            return loss, loss.detach().item(), difference.to(output.dtype).cpu()
        # Per batch item over its masked voxels, averaged over the items that have any: the
        # structure of MaskedLoss.forward, read off the one difference buffer.
        loss = output.new_tensor(0.0)
        true_nb = 0
        for batch in range(output.shape[0]):
            mask_b = mask[batch, ...]
            if not torch.any(mask_b):
                continue
            loss = loss + self._reduce(torch.masked_select(difference[batch, ...], mask_b), self._reduction)
            true_nb += 1
        map_ = difference.to(output.dtype).cpu()
        if true_nb == 0:
            return loss, np.nan, map_
        loss = loss / true_nb
        return loss, loss.detach().item(), map_

    def get_name(self) -> str:
        return "MAE"


class PSNR(MaskedLoss):
    reducible = True
    maximize = True  # reported value is the peak signal-to-noise ratio in dB (higher-is-better)

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
        # The log is a function of the RUNNING mean, applied once at the end, never per patch.
        return float(10 * np.log10(self._dynamic_range**2 / (total / count)))


class SSIM(MaskedLoss):
    """Structural similarity as ``skimage.metrics.structural_similarity`` computes it with its
    defaults: a 7-wide uniform window, K1 0.01, K2 0.03, the sample covariance, the mean over the
    map cropped by the window's radius, one map per channel (``channel_axis=0``) all averaged. In
    torch on the tensors' device, where skimage's numpy route was single-threaded and pulled the
    volumes back to the host. The map is built slab by slab along the first spatial axis, so the
    five window statistics it needs hold a few MiB at a time whatever the case.

    A voxel's window reaches the window's radius past a patch's faces: reducible from patches read
    with that radius of halo, each scoring the map voxels centred in its own grid slot.
    """

    maximize = True  # reported value is the structural similarity index (higher-is-better)
    reducible = True
    window = 7
    halo = (window - 1) // 2
    k1 = 0.01
    k2 = 0.03
    #: Bytes of one statistic map per slab. Measured at 512^3 float32, 12 threads / one GPU: 0.94 s /
    #: 0.12 s at 4 MiB, 1.08 / 0.09 at 8, 2.29 / 0.23 at 32, 7.58 / 0.28 at 64 (past glibc's 32 MiB
    #: mmap threshold every map is fresh pages); at 256^3 a 1 MiB slab is 6.1 s of per-op overhead
    #: against 0.15 s at 2 MiB.
    slab_bytes = 8 << 20

    def __init__(self, dynamic_range: float | None = None) -> None:
        dynamic_range = dynamic_range if dynamic_range else 1024 + 3000
        super().__init__(partial(SSIM._loss, dynamic_range), True)
        self._dynamic_range = float(dynamic_range)

    @staticmethod
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(SSIM._ssim(x.float(), y.float(), None, dynamic_range))

    @staticmethod
    def _box_sum(tensor: torch.Tensor, window: int) -> torch.Tensor:
        """The sum over the window along every spatial axis of ``[N, *spatial]``, valid part only:
        window - 1 shifted in-place adds per axis (6 ms per 8 MiB map on 12 threads, where a conv3d
        with a (7, 1, 1) kernel took 27)."""
        for axis in range(1, tensor.dim()):
            length = tensor.shape[axis] - window + 1
            total = tensor.narrow(axis, 0, length).clone()
            for shift in range(1, window):
                total += tensor.narrow(axis, shift, length)
            tensor = total
        return tensor

    @staticmethod
    def _map_sum(x: torch.Tensor, y: torch.Tensor, data_range: float) -> tuple[torch.Tensor, int]:
        """The sum (float64) and voxel count of skimage's SSIM map over the valid region of one
        ``[N, *spatial]`` pair, its operations in skimage's order."""
        window = SSIM.window
        voxels = window ** (x.dim() - 1)
        cov_norm = voxels / (voxels - 1)
        c1, c2 = (SSIM.k1 * data_range) ** 2, (SSIM.k2 * data_range) ** 2
        # float64: a variance is the difference of two moments near the square of the mean, and
        # float32 keeps 0.4 of a moment of 6e6 (a mean of 2500 HU) where the variance is a few
        # thousand. Measured against a float64 reference on a constant 1000 with a 5 HU step:
        # 7e-6 off in float32 (skimage 6.5e-8), 1.3e-7 here.
        x, y = x.to(torch.float64), y.to(torch.float64)
        ux = SSIM._box_sum(x, window) / voxels
        uy = SSIM._box_sum(y, window) / voxels
        vx = (SSIM._box_sum(x * x, window) / voxels).sub_(ux * ux).mul_(cov_norm)
        vy = (SSIM._box_sum(y * y, window) / voxels).sub_(uy * uy).mul_(cov_norm)
        vxy = (SSIM._box_sum(x * y, window) / voxels).sub_(ux * uy).mul_(cov_norm)
        a1 = (2 * ux * uy).add_(c1)
        b1 = (ux * ux).add_(uy * uy).add_(c1)
        a2 = vxy.mul_(2).add_(c2)
        b2 = vx.add_(vy).add_(c2)
        s = a1.mul_(a2).div_(b1.mul_(b2))
        return s.sum(dtype=torch.float64), s.numel()

    @staticmethod
    @torch.no_grad()
    def _map_sum_over(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None,
        data_range: float,
        core: tuple[slice, ...] | None,
    ) -> tuple[float, int]:
        """The sum and voxel count of one ``[C, *spatial]`` pair's SSIM map over the map voxels
        centred in ``core`` (slices of the spatial axes; ``None`` is the whole pair). Each slab reads
        the window's radius past the core and nothing more: a patch read with that radius of halo
        scores its grid slot exactly as the whole volume does. A mask multiplies both, slab by slab,
        and the count runs over the whole cropped extent, as the masked mode always was."""
        radius = (SSIM.window - 1) // 2
        spatial = x.shape[1:]
        if core is None:
            core = tuple(slice(0, extent) for extent in spatial)
        # Per axis, the centres whose window lies inside the pair, restricted to the core.
        spans = [(max(c.start, radius), min(c.stop, extent - radius)) for c, extent in zip(core, spatial, strict=True)]
        if any(stop <= start for start, stop in spans):
            return 0.0, 0
        inner = tuple(slice(start - radius, stop + radius) for start, stop in spans[1:])
        plane = x.shape[0] * int(np.prod([stop - start + 2 * radius for start, stop in spans[1:]]))
        planes = max(2 * radius + 2, SSIM.slab_bytes // (plane * 8))  # the maps are float64
        total = torch.zeros((), dtype=torch.float64, device=x.device)
        count = 0
        first, last = spans[0]
        for start in range(first, last, planes):
            rows = slice(start - radius, min(last, start + planes) + radius)
            xs, ys = x[(slice(None), rows, *inner)], y[(slice(None), rows, *inner)]
            if mask is not None:
                # torch.where, not a float x bool product: on a cache-resident slab the mixed-dtype
                # product runs an unvectorised cast path (10.4 ms per 14 MiB slab against 1.3).
                keep = mask[(slice(None), rows, *inner)]
                xs, ys = torch.where(keep, xs, 0.0), torch.where(keep, ys, 0.0)
            slab_total, slab_count = SSIM._map_sum(xs, ys, data_range)
            total += slab_total
            count += slab_count
        return float(total), count

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None, data_range: float) -> float:
        """The SSIM of one ``[C, *spatial]`` pair: the mean of its map over the cropped extent."""
        if any(size < SSIM.window for size in x.shape[1:]):
            raise MeasureError(
                f"SSIM needs every spatial extent to be at least its {SSIM.window}-voxel window.",
                f"Got a {tuple(x.shape[1:])} volume.",
            )
        total, count = SSIM._map_sum_over(x, y, mask, data_range, None)
        return total / count

    def _pairs(
        self, output: torch.Tensor, targets: tuple[torch.Tensor, ...]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None]:
        """Per batch item, the float pair and its bool mask; ``None`` for an item its mask empties."""
        if len(targets) == 0:
            raise ValueError("SSIM expects at least one target tensor.")
        target = targets[0].to(device=output.device)
        mask = self.get_mask(list(targets[1:]))
        mask = None if mask is None else mask.to(device=output.device) == 1
        for batch in range(output.shape[0]):
            mask_b = None if mask is None else mask[batch, ...]
            if mask_b is not None and not torch.any(mask_b):
                yield None
            else:
                yield output[batch].float(), target[batch].float(), mask_b

    def forward(
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        values = [SSIM._ssim(*pair, self._dynamic_range) for pair in self._pairs(output, targets) if pair is not None]
        if not values:
            return output.new_tensor(0.0), np.nan
        value = float(np.mean(values))
        return output.new_tensor(value), value

    def partial_metric(
        self, output: torch.Tensor, *targets: torch.Tensor, core: tuple[slice, ...] | None = None
    ) -> Any:
        """Each batch item's map sum and count over ``core`` (the whole patch when ``None``), in
        the ``items`` form ``MaskedLoss.combine_metric`` finishes per item and averages."""
        items = []
        for pair in self._pairs(output, targets):
            if pair is None:
                items.append((0.0, 0, False))
            else:
                items.append((*SSIM._map_sum_over(*pair, self._dynamic_range, core), True))
        return ("items", items)

    def _finish(self, total: float, count: int) -> float:
        if count == 0:
            raise MeasureError(f"SSIM needs every spatial extent to be at least its {SSIM.window}-voxel window.")
        return total / count


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


class KLDivergence(CriterionWithInit):
    def __init__(self, shape: list[int], dim: int = 100, mu: float = 0, std: float = 1) -> None:
        super().__init__()
        self.latent_dim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
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
        # train and validation. Accumulating n/corrects on the instance instead would report one lifetime
        # fraction that blends every epoch and both splits.
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
        target = Dice.on_grid(output, targets[0]).long()

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


class MutualInformationLoss(Criterion):
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

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        wa, pa, wb, pb = self.parzen_windowing(output, targets[0])  # (batch, num_sample, num_bin), (batch, 1, num_bin)
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
