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

import math
import weakref
from collections.abc import Iterator
from functools import partial
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.patching import ModelPatch
from konfai.metric.measure.base import (
    Criterion,
    CriterionOutput,
    CriterionWithAttribute,
    CriterionWithInit,
    LabelledValues,
    MaskedLoss,
    _require_optional,
)
from konfai.metric.measure.segmentation import Dice
from konfai.network.blocks import LatentDistribution
from konfai.network.network import Network
from konfai.utils.errors import MeasureError
from konfai.utils.ITK import JacobianSums, field_jacobian_sums, field_regularity_statistics, jacobian_statistics

#: The 12-bit CT convention, -1024..3071 HU: the one default dynamic range PSNR and SSIM share, written
#: into a resolved config as a number, and what a ``dynamic_range: None`` in a published config names.
CT_DYNAMIC_RANGE = 4095.0


class MSE(MaskedLoss):
    @staticmethod
    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.MSELoss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MSE._loss, reduction), False)
        self._reduction = reduction
        self.reducible = reduction in ("mean", "sum")
        self.batch_mean = reduction == "mean"

    def _kernel(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        return (output - target).pow(2) if self.reducible else None

    def _value(self, total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        return total / count if self._reduction == "mean" else total

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
        self.batch_mean = reduction == "mean"

    def _kernel(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        return (output - target).abs() if self.reducible else None

    def _value(self, total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        return total / count if self._reduction == "mean" else total

    def _finish(self, total: float, count: int) -> float:
        return total / count if self._reduction == "mean" else total


class ME(MaskedLoss):
    reducible = True
    batch_mean = True

    @staticmethod
    def _loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).mean()

    def __init__(self) -> None:
        super().__init__(ME._loss, False)

    def _kernel(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        return output - target

    def _value(self, total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        return total / count

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
        mask = mask.to(device=output.device)
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

    def forward(
        self, output: torch.Tensor, *targets: torch.Tensor
    ) -> tuple[torch.Tensor, float | torch.Tensor, torch.Tensor]:
        difference, mask = self._difference(output, *targets)
        if mask is None:
            loss = self._reduce(difference, self._reduction)
            return loss, loss.detach(), difference.to(output.dtype).cpu()
        map_ = difference.to(output.dtype).cpu()
        if self.reducible:
            # The difference is already zero outside the mask, so the batched masked forward
            # reads the exact bits MAE reads: the scalar and the map agree by construction.
            loss, value = self._masked_forward(difference, mask)
            return loss, value, map_
        # Per batch item over its masked voxels, averaged over the items that have any: the
        # structure of MaskedLoss.forward, read off the one difference buffer.
        scored = torch.any(mask.reshape(mask.shape[0], -1), dim=1).tolist()
        loss = output.new_tensor(0.0)
        for batch in range(output.shape[0]):
            if not scored[batch]:
                continue
            loss = loss + self._reduce(torch.masked_select(difference[batch, ...], mask[batch, ...]), self._reduction)
        true_nb = sum(scored)
        if true_nb == 0:
            return loss, np.nan, map_
        loss = loss / true_nb
        return loss, loss.detach(), map_

    def get_name(self) -> str:
        return "MAE"


class PSNR(MaskedLoss):
    reducible = True
    maximize = True  # reported value is the peak signal-to-noise ratio in dB (higher-is-better)
    loss_capable = False  # a metric: minimising it would push the output away from the target

    @staticmethod
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((x - y).pow(2))
        psnr = 10 * torch.log10(dynamic_range**2 / mse)
        return psnr

    def __init__(self, dynamic_range: float | None = CT_DYNAMIC_RANGE) -> None:
        dynamic_range = CT_DYNAMIC_RANGE if dynamic_range is None else dynamic_range
        super().__init__(partial(PSNR._loss, dynamic_range), False)
        self._dynamic_range = float(dynamic_range)

    def _kernel(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        return (output - target).pow(2)

    def _value(self, total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        return 10 * torch.log10(self._dynamic_range**2 / (total / count))

    def _finish(self, total: float, count: int) -> float:
        # The log is a function of the RUNNING mean, applied once at the end, never per patch.
        return float(10 * np.log10(self._dynamic_range**2 / (total / count)))


class SSIM(MaskedLoss):
    """Structural similarity as ``skimage.metrics.structural_similarity`` computes it with its
    defaults: a 7-wide uniform window, K1 0.01, K2 0.03, the sample covariance, the mean over the
    map cropped by the window's radius, one map per channel (``channel_axis=0``) all averaged, in
    torch on the tensors' device. The map is built slab by slab along the first spatial axis, so the
    five window statistics it needs hold a few MiB at a time whatever the case.

    Under a mask, both volumes are set to 0 outside it and the map is averaged over the mask's voxels
    of the cropped extent, the map as it is (a window at the mask's edge reads the zeros past it): the
    masked SSIM of SynthRAD's evaluation (``crop(ssim_map, 3)[crop(mask, 3)].mean()``). The voxels
    outside the mask are not scored: each of them would compare two zeros and score 1.

    A voxel's window reaches the window's radius past a patch's faces: reducible from patches read
    with that radius of halo, each scoring the map voxels centred in its own grid slot.

    A metric unless its training config writes ``is_loss: true``; the loss is then ``1 - SSIM``, with
    its gradient, and the reported value stays the SSIM.
    """

    maximize = True  # reported value is the structural similarity index (higher-is-better)
    default_is_loss = False
    reducible = True
    batch_mean = True
    window = 7
    halo = (window - 1) // 2
    k1 = 0.01
    k2 = 0.03
    #: Bytes of one statistic map per slab. Measured at 512^3 float32, 12 threads / one GPU: 0.94 s /
    #: 0.12 s at 4 MiB, 1.08 / 0.09 at 8, 2.29 / 0.23 at 32, 7.58 / 0.28 at 64 (past glibc's 32 MiB
    #: mmap threshold every map is fresh pages); at 256^3 a 1 MiB slab is 6.1 s of per-op overhead
    #: against 0.15 s at 2 MiB.
    slab_bytes = 8 << 20

    def __init__(self, dynamic_range: float | None = CT_DYNAMIC_RANGE) -> None:
        dynamic_range = CT_DYNAMIC_RANGE if dynamic_range is None else dynamic_range
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
    def _map_sum(
        x: torch.Tensor, y: torch.Tensor, data_range: float, keep: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | int]:
        """The sum (float64) and voxel count of skimage's SSIM map over the valid region of one
        ``[N, *spatial]`` pair, its operations in skimage's order; with ``keep`` (bool, the valid
        region's shape, broadcast over the channels), over the map voxels it keeps."""
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
        if keep is None:
            return s.sum(dtype=torch.float64), s.numel()
        keep = keep.expand_as(s)
        return torch.where(keep, s, 0.0).sum(dtype=torch.float64), keep.sum()

    @staticmethod
    def _map(x: torch.Tensor, y: torch.Tensor, data_range: float) -> torch.Tensor:
        """The map ``_map_sum`` sums, out of place so autograd can differentiate it."""
        window = SSIM.window
        voxels = window ** (x.dim() - 1)
        cov_norm = voxels / (voxels - 1)
        c1, c2 = (SSIM.k1 * data_range) ** 2, (SSIM.k2 * data_range) ** 2
        ux = SSIM._box_sum(x, window) / voxels
        uy = SSIM._box_sum(y, window) / voxels
        vx = (SSIM._box_sum(x * x, window) / voxels - ux * ux) * cov_norm
        vy = (SSIM._box_sum(y * y, window) / voxels - uy * uy) * cov_norm
        vxy = (SSIM._box_sum(x * y, window) / voxels - ux * uy) * cov_norm
        return (2 * ux * uy + c1) * (2 * vxy + c2) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))

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
        scores its grid slot exactly as the whole volume does. A mask sets both to 0 outside it, slab
        by slab, and the sum and the count run over the map voxels centred in the mask."""
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
        count: torch.Tensor | int = 0
        first, last = spans[0]
        for start in range(first, last, planes):
            rows = slice(start - radius, min(last, start + planes) + radius)
            xs, ys = x[(slice(None), rows, *inner)], y[(slice(None), rows, *inner)]
            centres = None
            if mask is not None:
                # torch.where, not a float x bool product: on a cache-resident slab the mixed-dtype
                # product runs an unvectorised cast path (10.4 ms per 14 MiB slab against 1.3).
                keep = mask[(slice(None), rows, *inner)]
                xs, ys = torch.where(keep, xs, 0.0), torch.where(keep, ys, 0.0)
                centres = keep[(slice(None), *(slice(radius, size - radius) for size in keep.shape[1:]))]
            slab_total, slab_count = SSIM._map_sum(xs, ys, data_range, centres)
            total += slab_total
            count = count + slab_count
        return float(total), int(count)

    @staticmethod
    def _check_extent(x: torch.Tensor) -> None:
        if any(size < SSIM.window for size in x.shape[1:]):
            raise MeasureError(
                f"SSIM needs every spatial extent to be at least its {SSIM.window}-voxel window.",
                f"Got a {tuple(x.shape[1:])} volume.",
            )

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None, data_range: float) -> float:
        """The SSIM of one ``[C, *spatial]`` pair: the mean of its map over the cropped extent, or over
        the mask's voxels in it."""
        SSIM._check_extent(x)
        total, count = SSIM._map_sum_over(x, y, mask, data_range, None)
        return total / count

    @staticmethod
    def _differentiable_ssim(
        x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None, data_range: float
    ) -> torch.Tensor:
        """``_ssim`` with its gradient: the whole map at once, which autograd keeps for the backward."""
        SSIM._check_extent(x)
        if mask is None:
            return SSIM._map(x.double(), y.double(), data_range).mean()
        x, y = torch.where(mask, x, 0.0), torch.where(mask, y, 0.0)
        ssim_map = SSIM._map(x.double(), y.double(), data_range)
        centres = mask[(slice(None), *(slice(SSIM.halo, size - SSIM.halo) for size in mask.shape[1:]))]
        return torch.where(centres, ssim_map, 0.0).sum() / centres.expand_as(ssim_map).sum()

    def _pairs(
        self, output: torch.Tensor, targets: tuple[torch.Tensor, ...]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None]:
        """Per batch item, the float pair and its bool mask; ``None`` for an item whose mask holds no voxel of the
        cropped extent, where the map is: the three reductions then score the same items."""
        self._require_target(targets)
        target = targets[0].to(device=output.device)
        mask = self.get_mask(list(targets[1:]))
        mask = None if mask is None else mask.to(device=output.device)
        for batch in range(output.shape[0]):
            mask_b = scored = None if mask is None else mask[batch, ...]
            if mask_b is not None and all(size >= SSIM.window for size in mask_b.shape[1:]):
                scored = mask_b[(slice(None), *(slice(SSIM.halo, size - SSIM.halo) for size in mask_b.shape[1:]))]
            if scored is not None and not torch.any(scored):
                yield None
            else:
                yield output[batch].float(), target[batch].float(), mask_b

    def forward(
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        pairs = (pair for pair in self._pairs(output, targets) if pair is not None)
        if self.as_loss and torch.is_grad_enabled():
            ssims = [SSIM._differentiable_ssim(*pair, self._dynamic_range) for pair in pairs]
            if not ssims:
                return output.new_tensor(0.0), np.nan
            ssim = torch.stack(ssims).mean()
            return (1 - ssim).float(), ssim.detach()
        values = [SSIM._ssim(*pair, self._dynamic_range) for pair in pairs]
        if not values:
            return output.new_tensor(0.0), np.nan
        value = float(np.mean(values))
        return output.new_tensor(1 - value if self.as_loss else value), value

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
                total, count = SSIM._map_sum_over(*pair, self._dynamic_range, core)
                # a masked item is scored where its mask holds a voxel of the cropped extent
                items.append((total, count, pair[2] is None or count > 0))
        return ("items", items)

    def _finish(self, total: float, count: int) -> float:
        if count == 0:
            raise MeasureError(f"SSIM needs every spatial extent to be at least its {SSIM.window}-voxel window.")
        return total / count


class LPIPS(MaskedLoss):
    """Learned perceptual distance (``lpips`` package), tiled a slice at a time.

    The output and the target are each rescaled to [-1, 1] with their own min and max, taken over the
    whole batch: in training one image's value depends on the other images of its batch, and a batch
    of one is rescaled per image. The rescaling removes any global gain and offset of intensity. A
    constant output or target (a background patch in a batch of one) has no range: as a metric it
    scores NaN, which the averages skip; as a loss it is rescaled to 0, so its value and gradient stay
    finite. Under a mask the voxels outside it are set to 0 in both, and each image of the batch is
    rescaled and scored on its own.
    """

    @staticmethod
    def normalize(tensor: torch.Tensor, finite: bool = False) -> torch.Tensor:
        """``tensor`` rescaled to [-1, 1]; a constant one is NaN, or 0 when ``finite``."""
        shifted = tensor - torch.min(tensor)
        span = torch.max(tensor) - torch.min(tensor)
        if finite:
            # A safe denominator, not a NaN branch masked out: its gradient would still be NaN.
            flat = span == 0
            return torch.where(flat, 0.0, shifted / torch.where(flat, 1.0, span) * 2 - 1)
        return shifted / span * 2 - 1

    @staticmethod
    def preprocessing(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat((1, 3, 1, 1))

    @staticmethod
    def _loss(
        loss_fn_alex, dataset_patch: ModelPatch, x: torch.Tensor, y: torch.Tensor, finite: bool = False
    ) -> torch.Tensor:
        # Follow the input's device (the DDP rank's GPU, or CPU) instead of a hardcoded device 0.
        loss_fn_alex = loss_fn_alex.to(x.device)
        if x.dim() == 4:  # a 2-D image is tiled as a volume of one slice
            x, y = x.unsqueeze(2), y.unsqueeze(2)
        dataset_patch.load(list(x.shape[2:]))

        loss = x.new_tensor(0.0)
        for patch_input in dataset_patch.disassemble(LPIPS.normalize(x, finite), LPIPS.normalize(y, finite)):
            real, fake = LPIPS.preprocessing(patch_input[0]), LPIPS.preprocessing(patch_input[1])
            # One distance per batch sample: the mean scores them all, where ``.flatten()[0]``
            # silently kept only the first.
            loss = loss + loss_fn_alex(real, fake).mean()
        return loss / dataset_patch.get_size(0)

    def __init__(self, model: str = "alex") -> None:
        lpips = _require_optional("lpips", criterion="LPIPS", extra="lpips")

        super().__init__(self._distance, True)
        self._score = partial(LPIPS._loss, lpips.LPIPS(net=model), ModelPatch([1, 320, 320]))

    def _distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # As a loss a constant image must stay finite: a NaN gradient reaches the optimizer.
        return self._score(x, y, finite=self.as_loss)


class TRE(Criterion):
    """The target registration error: the distance between paired landmarks, their mean with each landmark's own.
    With ``percentile`` it is that percentile of a case's landmark errors instead (95: the error all but the worst
    twentieth of its landmarks stay under), linearly interpolated, as one value."""

    batch_mean = True

    def __init__(self, percentile: float | None = None) -> None:
        super().__init__()
        if percentile is not None and not 0 <= percentile <= 100:
            raise MeasureError(f"TRE takes a percentile between 0 and 100, got {percentile}.")
        self._percentile = percentile

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> CriterionOutput:
        loss = torch.linalg.norm(output - targets[0], dim=2)
        if self._percentile is not None:
            value = torch.quantile(loss.double(), self._percentile / 100, dim=1).mean()
            return value.to(loss.dtype), value.detach()
        per_landmark = loss.mean(0).detach()
        return loss.mean(), LabelledValues(per_landmark, [f"Landmarks_{i}" for i in range(per_landmark.shape[0])])


def _point_spread(members: torch.Tensor) -> torch.Tensor:
    """Per landmark, how far an ensemble's members put it from each other: the norm of the per-axis standard
    deviation across the members (``[M, N, 3]`` in, ``[N]`` out), the root mean square distance to their mean point
    (N - 1); 0 for one member."""
    if members.shape[0] < 2:
        return members.new_zeros(members.shape[1])
    return torch.linalg.norm(members.double().std(0), dim=-1)


class PointSpread(Criterion):
    """The spread of landmarks pushed through every member of an ensemble (``TransformPoints`` through a stack,
    ``[B, M, N, 3]``), in mm: per landmark, the root mean square distance of the members' points to their mean; the
    reported value is their mean. The target is not read: name it ``None`` under ``targets_criterions``."""

    any_target_grid = True

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, LabelledValues]:
        del targets
        if output.dim() != 4 or output.shape[-1] != 3:
            raise MeasureError(
                f"PointSpread reads the points of an ensemble's members, [batch, members, points, 3]; got {list(output.shape)}.",
                "Push the landmarks through a stack of transforms (TransformPoints) first.",
            )
        spread = torch.stack([_point_spread(case) for case in output]).mean(0)
        return spread.mean().float(), LabelledValues(spread.float(), [f"Landmarks_{i}" for i in range(spread.shape[0])])


def _ranks(values: torch.Tensor) -> torch.Tensor:
    """The ranks Spearman's correlation takes: tied values share the mean of the ranks they span."""
    _, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    first = torch.cumsum(counts, 0) - counts  # each distinct value's first rank, from 0
    return (first + (counts - 1) / 2).to(values.dtype)[inverse]


class SpreadCalibration(Criterion):
    """Whether an ensemble's spread predicts its error, from landmarks: the Spearman rank correlation, across the
    landmarks, between their spread (``PointSpread``) and the error of the ensemble's mean point against the target's
    landmark (its TRE). 1 when the most uncertain landmarks are the worst registered; about 0 when the spread says
    nothing. The output is the members' points (``[B, M, N, 3]``), the target the landmarks they should land on."""

    maximize = True
    any_target_grid = True

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        if not targets:
            raise MeasureError("SpreadCalibration compares the members' points with the target's and was handed none.")
        correlations = []
        for members, reference in zip(output, targets[0], strict=True):
            spread = _point_spread(members)
            error = torch.linalg.norm(members.double().mean(0) - reference.double(), dim=-1)
            if spread.numel() < 3 or float(spread.std()) == 0 or float(error.std()) == 0:
                correlations.append(float("nan"))
                continue
            a, b = _ranks(spread), _ranks(error)
            correlations.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
        value = float(np.nanmean(correlations)) if not all(np.isnan(correlations)) else float("nan")
        return torch.tensor(1 - value), value


class Quantile(Criterion):
    """The ``q`` quantile of the output's finite values (0.95: its 95th percentile), a map read whole: with ``Mean``,
    how large an uncertainty map runs, and how far its tail goes. The target is not read."""

    any_target_grid = True

    def __init__(self, q: float = 0.95, name: str | None = None) -> None:
        super().__init__()
        if not 0 <= q <= 1:
            raise MeasureError(f"Quantile takes q in [0, 1], got {q}.")
        self.q = float(q)
        self.name = name or f"Quantile{round(self.q * 100)}"

    def get_name(self) -> str:
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        del targets
        values = output.detach().float().cpu().numpy().ravel()
        values = values[np.isfinite(values)]
        value = float(np.quantile(values, self.q)) if values.size else float("nan")
        return torch.tensor(value), value


def _selected(
    criterion: Criterion, output: torch.Tensor, targets: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The output's and the first target's voxels as two flat float64 vectors, inside the mask the other targets
    make (every voxel without one)."""
    criterion._require_target(targets)
    target = targets[0].to(output.device)
    mask = MaskedLoss.get_mask(list(targets[1:]))
    if mask is None:
        return output.double().flatten(), target.double().flatten()
    mask = mask.to(output.device).expand_as(output)
    return torch.masked_select(output, mask).double(), torch.masked_select(target, mask).double()


class NCC(Criterion):
    """The normalised cross-correlation of the output and the target, over the image or inside a mask (a second
    target): 1 when one is an increasing linear map of the other's intensities. For images of one modality. Its
    sums add up over disjoint patches, so it streams."""

    maximize = True
    reducible = True
    loss_capable = False  # a value from summed moments, no gradient

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[float, ...]:
        a, b = _selected(self, output, targets)  # float64: sums of squares far from 0
        return tuple(
            float(value) for value in (a.numel(), a.sum(), b.sum(), (a * b).sum(), (a * a).sum(), (b * b).sum())
        )

    def combine_metric(self, states: list[Any]) -> tuple[torch.Tensor, float]:
        n, sa, sb, sab, saa, sbb = map(sum, zip(*states, strict=True))
        if not n:  # an empty mask
            return torch.tensor(float("nan")), float("nan")
        covariance, var_a, var_b = sab - sa * sb / n, saa - sa * sa / n, sbb - sb * sb / n
        ncc = covariance / math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else float("nan")
        return torch.tensor(1 - ncc), ncc

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        return self.combine_metric([self.partial_metric(output, *targets)])


class NMI(Criterion):
    """The normalised mutual information ``(H(A) + H(B)) / H(A, B)`` of the output's and the target's intensities,
    over the image or inside a mask (a second target): 1 for independent images, 2 for one image a one-to-one map
    of the other's, whatever their modalities. Intensities fall in ``bins`` bins over each image's own range, or
    over ``output_range`` and ``target_range`` when both are given ([low, high], a value outside in the edge bin):
    the joint histograms of disjoint patches then add up, and it streams."""

    maximize = True
    loss_capable = False  # a value from a histogram, no gradient

    def __init__(
        self, bins: int = 64, output_range: list[float] | None = None, target_range: list[float] | None = None
    ) -> None:
        super().__init__()
        if bins < 2:
            raise MeasureError(f"NMI needs at least two bins, got {bins}.")
        self.bins = int(bins)
        self.output_range, self.target_range = output_range, target_range
        self.reducible = output_range is not None and target_range is not None

    def _histogram(
        self, a: torch.Tensor, b: torch.Tensor, a_range: tuple[float, float], b_range: tuple[float, float]
    ) -> torch.Tensor:
        def binned(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
            scale = self.bins / (high - low) if high > low else 0.0
            return ((values - low) * scale).long().clamp_(0, self.bins - 1)

        joint = binned(a, *a_range) * self.bins + binned(b, *b_range)
        return torch.bincount(joint, minlength=self.bins * self.bins).double().cpu()

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        a, b = _selected(self, output, targets)
        return self._histogram(a, b, tuple(self.output_range), tuple(self.target_range))  # type: ignore[arg-type]

    def combine_metric(self, states: list[Any]) -> tuple[torch.Tensor, float]:
        joint = torch.stack(states).sum(0).view(self.bins, self.bins)
        p = joint / joint.sum() if joint.sum() > 0 else joint

        def entropy(q: torch.Tensor) -> float:
            q = q[q > 0]
            return float(-(q * q.log()).sum())

        h_joint = entropy(p)
        nmi = (entropy(p.sum(1)) + entropy(p.sum(0))) / h_joint if h_joint > 0 else float("nan")
        return torch.tensor(2 - nmi), nmi

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        a, b = _selected(self, output, targets)
        if not a.numel():
            return torch.tensor(float("nan")), float("nan")
        a_range = tuple(self.output_range) if self.output_range else (float(a.min()), float(a.max()))
        b_range = tuple(self.target_range) if self.target_range else (float(b.min()), float(b.max()))
        return self.combine_metric([self._histogram(a, b, a_range, b_range)])  # type: ignore[arg-type]


class LNCC(Criterion):
    """The local normalised cross-correlation of the output and the target: the mean, over the image or inside a mask
    (a second target), of the squared correlation of each ``kernel``-wide cube of voxels, ``cov^2 / (var_a var_b)``
    (ANTs' and VoxelMorph's form, in [0, 1]). For images of one modality whose intensities drift across the field.
    A window reaches ``kernel // 2`` voxels past a patch's faces: it streams from patches read with that halo."""

    maximize = True
    reducible = True
    loss_capable = False  # a value from pooled moments, no gradient

    def __init__(self, kernel: int = 9) -> None:
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise MeasureError(f"LNCC takes an odd kernel, got {kernel}.", "A window centred on its voxel: 5, 7, 9...")
        self.kernel = int(kernel)
        self.halo = self.kernel // 2

    def _map(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The squared local correlation at each voxel of ``a`` and ``b``, ``[N, C, (Z), Y, X]``."""
        pool = {4: F.avg_pool2d, 5: F.avg_pool3d}.get(a.dim())
        if pool is None:
            raise MeasureError(f"LNCC reads 2-D or 3-D images, got {list(a.shape[1:])} (channels first).")

        def total(values: torch.Tensor) -> torch.Tensor:
            # The sum over each window, cut at the faces: padded by hand, since the pool refuses an image smaller
            # than its kernel (a thin volume, a boundary patch).
            return pool(F.pad(values, (self.halo,) * (2 * (a.dim() - 2))), self.kernel, stride=1, divisor_override=1)

        # In float64: the moments' differences cancel in float32 on intensities far from 0 (an MR at 1000 with a
        # local spread of a few units scored 1672 against itself plus noise).
        a, b = a.double(), b.double()
        count = total(torch.ones_like(a))
        mean_a, mean_b = total(a) / count, total(b) / count
        covariance = total(a * b) / count - mean_a * mean_b
        variance = (total(a * a) / count - mean_a**2).clamp_min(0) * (total(b * b) / count - mean_b**2).clamp_min(0)
        return (covariance**2 / (variance + 1e-5)).clamp_max(1.0)

    def partial_metric(
        self, output: torch.Tensor, *targets: torch.Tensor, core: tuple[slice, ...] | None = None
    ) -> tuple[float, int]:
        self._require_target(targets)
        values = self._map(output, targets[0].to(output.device))
        mask = MaskedLoss.get_mask(list(targets[1:]))
        selected = (
            torch.ones_like(values, dtype=torch.bool) if mask is None else mask.to(values.device).expand_as(values)
        )
        if core is not None:
            values, selected = values[(slice(None), slice(None), *core)], selected[(slice(None), slice(None), *core)]
        return float(values[selected].sum()), int(selected.sum())

    def combine_metric(self, states: list[Any]) -> tuple[torch.Tensor, float]:
        total, count = sum(state[0] for state in states), sum(state[1] for state in states)
        lncc = total / count if count else float("nan")
        return torch.tensor(1 - lncc), lncc

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        return self.combine_metric([self.partial_metric(output, *targets)])


def _statistic_name(criterion: str, statistic: str, name: str | None) -> str:
    """The name a ``criterion`` reports one ``statistic`` under (``mean``, ``p95`` or ``max``): ``name`` when given,
    else on ``PointDistanceP95``'s pattern."""
    if statistic not in ("mean", "p95", "max"):
        raise MeasureError(f"{criterion} has no statistic '{statistic}'.", "Choose mean, p95 or max.")
    return name or f"{criterion}{statistic.capitalize()}"


class PointDistance(Criterion):
    """The distance between the output's and the target's vectors (one per voxel, its components on the channel
    axis): their ``mean``, 95th percentile (``p95``) or maximum (``max``) over the voxels where both are finite. On
    two coordinate images (``WorldCoordinates``), one sent through a registration and the one registered back, it is
    the round trip's error, in mm; a voxel the round trip leaves the grid from (``Resample``'s ``fill: .nan``) is
    left out."""

    loss_capable = False  # a percentile of a readout, no gradient

    def __init__(self, statistic: str = "mean", name: str | None = None) -> None:
        super().__init__()
        self.statistic = statistic
        self.name = _statistic_name("PointDistance", statistic, name)

    def get_name(self) -> str:
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        self._require_target(targets)
        distance = torch.linalg.norm(output.double() - targets[0].to(output.device).double(), dim=1)
        values = distance[torch.isfinite(distance)]
        if not values.numel():
            return torch.tensor(float("nan")), float("nan")
        array = values.cpu().numpy()
        value = {"mean": array.mean, "p95": lambda: np.percentile(array, 95), "max": array.max}[self.statistic]()
        return torch.tensor(float(value)), float(value)


def _field_of(field: torch.Tensor, attribute: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    """One displacement field ``[C, (Z), Y, X]`` as the ``[3, Z, Y, X]`` float64 array :func:`field_jacobian_sums`
    and :func:`field_regularity_statistics` take, with the matrix taking a world offset to indices of the grid its
    ``attribute`` describes. A 2-D field is one plane thick, with no z displacement: the z row of J is the identity's
    and every z derivative 0."""
    rank = field.dim() - 1
    if attribute is None or rank not in (2, 3) or field.shape[0] != rank:
        raise MeasureError(
            f"{name} measures a displacement field, one component per spatial axis; got {list(field.shape)}.",
            "Score the group of a dense field read with its geometry: an affine's Jacobian is its matrix, everywhere.",
        )
    values = field.detach().cpu().numpy().astype(np.float64)
    spacing = attribute.get_np_array("Spacing").astype(np.float64)
    direction = attribute.get_np_array("Direction").astype(np.float64).reshape(rank, rank)
    if rank == 2:
        values = np.concatenate([values, np.zeros_like(values[:1])])[:, None]
        spacing, direction = np.append(spacing, 1.0), np.pad(direction, (0, 1))
        direction[2, 2] = 1.0
    return values, np.linalg.inv(direction @ np.diag(spacing))


class _FieldStatistic(CriterionWithAttribute):
    """One statistic of a displacement field, the output, read with its geometry (:func:`_field_of`). The target is
    not read: name it ``None`` under ``targets_criterions``."""

    accepts_output_attributes = True
    any_target_grid = True
    loss_capable = False  # a statistic of a numpy pass, no gradient
    statistic = ""


class _JacobianStatistic(_FieldStatistic):
    """One statistic of the determinant of a displacement field's Jacobian ``J = I + du/dx``, as
    :func:`konfai.utils.ITK.jacobian_statistics` takes it. Reducible under a memory budget: each patch, read with one
    voxel of halo, carries the sums of its slot, so the field is never held whole."""

    reducible = True
    halo = 1

    def forward(  # type: ignore[override]
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
        attributes: list[list[Any]],
        output_attributes: list[Any] | None = None,
    ) -> torch.Tensor:
        del targets, attributes
        return self.combine_metric([self.partial_metric(output, output_attributes=output_attributes)])

    def partial_metric(  # type: ignore[override]
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
        core: tuple[slice, ...] | None = None,
        output_attributes: list[Any] | None = None,
    ) -> list[JacobianSums]:
        """The sums of each sample's field over ``core``, its slot within a patch read with the halo (the whole
        tensor when ``None``), the derivatives running through the voxels around it."""
        del targets
        per_sample = output_attributes if output_attributes is not None else [None] * output.shape[0]
        spatial = tuple(core) if core is not None else ()
        sums = []
        for field, attribute in zip(output, per_sample, strict=True):
            values, to_index = _field_of(field, attribute, self.get_name())
            sums.append(field_jacobian_sums(values, (slice(None),) * (3 - len(spatial)) + spatial, to_index))
        return sums

    def combine_metric(self, states: list[list[JacobianSums]]) -> torch.Tensor:
        """The statistic of each sample from the sums of every patch, averaged over the batch."""
        per_sample = zip(*states, strict=True)
        return torch.tensor(float(np.mean([jacobian_statistics(sums)[self.statistic] for sums in per_sample])))


class _RegularityStatistic(_FieldStatistic):
    """One statistic of :func:`konfai.utils.ITK.field_regularity_statistics`, the field read whole: a percentile of
    the displacements has no partial state."""

    # ponytail: a one-slot memo, so the regularity criteria of one field share one pass; a case at a time, as
    # evaluation runs.
    _last: ClassVar[tuple[Any, list[dict[str, float]]] | None] = None

    def forward(  # type: ignore[override]
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
        attributes: list[list[Any]],
        output_attributes: list[Any] | None = None,
    ) -> torch.Tensor:
        del targets, attributes
        last = _RegularityStatistic._last
        if last is None or last[0]() is not output:
            per_sample = output_attributes if output_attributes is not None else [None] * output.shape[0]
            stats = []
            for field, attribute in zip(output, per_sample, strict=True):
                values, to_index = _field_of(field, attribute, self.get_name())
                stats.append(field_regularity_statistics(values.__getitem__, list(values.shape), to_index))
            _RegularityStatistic._last = (weakref.ref(output), stats)
        else:
            stats = last[1]
        return torch.tensor(float(np.mean([case[self.statistic] for case in stats])))


class BendingEnergy(_RegularityStatistic):
    """The bending energy of a displacement field: the mean over its voxels of every squared second derivative, in
    1/mm^2 for a field in mm. 0 for an affine map; how far from smooth the deformation is."""

    statistic = "bending_energy"


class Displacement(_RegularityStatistic):
    """The length of a displacement field's vectors, in its unit (mm): their ``mean``, 95th percentile (``p95``) or
    maximum (``max``). A registration that moved nothing, or one that went far, shows here first."""

    def __init__(self, statistic: str = "mean", name: str | None = None) -> None:
        super().__init__()
        self.name = _statistic_name("Displacement", statistic, name)
        self.statistic = f"displacement_{statistic}"

    def get_name(self) -> str:
        return self.name


class JacobianFolding(_JacobianStatistic):
    """The fraction of a displacement field's voxels where the map folds: its Jacobian's determinant at or below 0."""

    statistic = "folded_fraction"


class JacobianMin(_JacobianStatistic):
    """The lowest determinant of a displacement field's Jacobian: below 0 the map folds, near 0 it crushes a volume."""

    statistic = "min"
    maximize = True


class JacobianLogStd(_JacobianStatistic):
    """The standard deviation of the log of a displacement field's Jacobian determinant (clipped to ``[1e-9, 1e9]``,
    so a fold counts as a vanishing volume): how unevenly the map stretches."""

    statistic = "sd_log"


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
    batch_mean = True
    any_target_grid = True  # scores the output against a constant, never a target

    def __init__(self, target: float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer("target", torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = self.get_buffer("target")
        return self.loss(output, target.to(output.device).expand_as(output))


class KLDivergence(CriterionWithInit):
    batch_mean = True
    any_target_grid = True  # scores the output's latent distribution, never a target

    def __init__(self, shape: list[int], dim: int = 100, mu: float = 0, std: float = 1) -> None:
        """``mu`` and ``std`` are the mean and standard deviation of the Gaussian prior."""
        super().__init__()
        if std <= 0:
            raise MeasureError(f"KLDivergence: the prior's std must be positive, got {std}.")
        self.latent_dim = dim
        self.prior_mu = float(mu)
        self.prior_var = float(std) ** 2
        self.shape = shape

    def init(self, model: torch.nn.Module, output_group: str, target_group: str) -> str:
        if not isinstance(model, Network):
            raise MeasureError(f"KLDivergence rewires a KonfAI Network, got {type(model).__name__}.")
        model._compute_channels_trace(model, model.in_channels, None, None)

        last_module = model.get_submodule(".".join(output_group.split(".")[:-1]))

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
        # The LatentDistribution channels: mu, then the log-variance (its module is named log_std).
        mu = output[:, 0, :]
        log_var = output[:, 1, :]
        kl = (
            1
            + log_var
            - math.log(self.prior_var)
            - (mu - self.prior_mu) ** 2 / self.prior_var
            - torch.exp(log_var) / self.prior_var
        )
        return torch.mean(-0.5 * torch.sum(kl, dim=1), dim=0)


class Accuracy(Criterion):
    maximize = True  # reported value is the accuracy fraction (higher-is-better)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        # Return this batch's accuracy; the logging window means it over the batches and resets between
        # train and validation. Accumulating n/corrects on the instance instead would report one lifetime
        # fraction that blends every epoch and both splits.
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
        target = targets[0]
        if target.dim() == output.dim():  # the dataset's channel axis, which the argmax removed
            target = target.squeeze(1)
        return (predicted == target).float().mean()


class FocalLoss(Criterion):
    resamples_target = True  # a target of another shape is put on the output's grid (Dice.on_grid)

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        alpha_tensor = None
        if alpha is not None:
            raw_alpha = torch.tensor(alpha, dtype=torch.float32)
            alpha_tensor = raw_alpha / raw_alpha.sum() * len(raw_alpha)
        # A buffer, so ``criterion.to(device)`` moves it once instead of a per-call upload.
        self.register_buffer("alpha", alpha_tensor)
        self.gamma = gamma
        self.reduction = reduction
        self.batch_mean = reduction == "mean"

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = Dice.on_grid(output, targets[0]).long()

        logpt = F.log_softmax(output, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, target)
        pt = pt.gather(1, target)

        loss = -((1 - pt) ** self.gamma) * logpt
        alpha = self._buffers["alpha"]
        if alpha is not None:
            if output.shape[1] > alpha.numel():
                raise MeasureError(
                    f"FocalLoss got {output.shape[1]} output classes for {alpha.numel()} alpha weights.",
                    "Give `alpha:` one weight per class, or omit it to weight every class equally.",
                )
            # alpha[target] is already [B, 1, *spatial] (matching pt/logpt); do not add an axis.
            loss = alpha.to(target.device)[target] * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CrossEntropyLoss(Criterion):
    def __init__(self, weight: list[float] | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight) if weight else None, reduction=reduction)
        # Class weights normalize by the batch's own weight sum.
        self.batch_mean = reduction == "mean" and not weight

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.loss(output, targets[0].squeeze(1))


class Variance(Criterion):
    batch_mean = True
    any_target_grid = True  # scores the output alone

    def __init__(self, name: str = "Variance") -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = output.float()
        if output.shape[1] > 1:
            variance = output.var(1).mean()
        else:
            variance = torch.zeros((), device=output.device, dtype=output.dtype)
        return variance, variance.detach()


class Mean(Criterion):
    batch_mean = True
    any_target_grid = True  # scores the output alone

    def __init__(self, name: str = "Mean") -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss = output.float().mean()
        return loss, loss.detach()
