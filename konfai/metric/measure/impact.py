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


"""The IMPACT feature criteria over TorchScript extractors."""

import contextlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from konfai.data.patching import Accumulator, Cosinus, ModelPatch, blend_axes, blend_overlap
from konfai.metric.measure.adversarial import Gram
from konfai.metric.measure.base import CriterionWithAttribute, _require_optional
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute
from konfai.utils.errors import MeasureError
from konfai.utils.utils import get_module
from konfai.utils.vram import halve_on_oom


def _hf_hub_download(criterion: str):
    """The ``hf_hub_download`` callable, imported at the call site: huggingface_hub is only needed
    by the IMPACT criteria, never by the rest of the metric package."""
    return _require_optional("huggingface_hub", criterion=criterion, extra="all").hf_hub_download


@contextlib.contextmanager
def no_texpr_fuser() -> Iterator[None]:
    """Disable torch's TensorExpr (NNC) fuser for the block, restoring the caller's setting: the IMPACT TorchScript
    models' orientation branch has shape ops (``aten::size``) that crash its alias analysis under autograd. The
    profiling executor stays on."""
    previous = torch._C._jit_texpr_fuser_enabled()
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_fuser_enabled(previous)


def _sniffed_mask(targets: tuple[torch.Tensor, ...], candidate: torch.Tensor) -> torch.Tensor | None:
    """The uint8-mask convention, checked: a target sniffed as a mask must be a {0, 1} map and a
    tensor of its own, never the scored target itself (an 8-bit intensity target would otherwise be
    consumed as a mask in silence)."""
    if candidate.dtype != torch.uint8:
        return None
    if candidate is targets[0]:
        raise MeasureError(
            "The only target is uint8, so it would be read as both the scored target and its mask.",
            "Pass the image target first and the {0, 1} uint8 mask last, or cast the image off uint8.",
        )
    if bool(torch.any(candidate > 1)):
        raise MeasureError(
            "A uint8 target is read as a foreground mask, but it holds values above 1.",
            "IMPACT masks are {0, 1} uint8 maps; cast an 8-bit intensity target to another dtype.",
        )
    return candidate


def _check_feature_model(
    model_path: str, in_channels: int, shape: list[int], weights: list[float], gradient: bool = False
) -> None:
    """Probe a TorchScript feature extractor on the CPU: one output feature map per layer weight, or raise; with
    ``gradient``, also a gradient through every weighted layer, for a loss differentiated through the network.

    Runs on the CPU only: the probe result is discarded, and a GPU here would pin every DDP rank to one device.
    """
    model: torch.nn.Module = torch.jit.load(model_path, map_location=torch.device("cpu"))  # nosec B614
    dummy_input = torch.zeros((1, in_channels, *shape), requires_grad=gradient)
    try:
        with torch.set_grad_enabled(gradient), no_texpr_fuser():
            out = model(dummy_input, torch.tensor([len(weights)]))
        if not isinstance(out, (list, tuple)):
            raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
        if len(weights) != len(out):
            raise ValueError(
                f"'{model_path}': mismatch between the number of weights ({len(weights)}) and the number of "
                f"model outputs ({len(out)}). Each output must have a corresponding weight."
            )
    except Exception as e:
        raise RuntimeError(
            f"[Model Sanity Check Failed]\nInput shape attempted: {dummy_input.shape}\nError: {type(e).__name__}: {e}"
        ) from e
    detached = [
        str(index + 1) for index, (w, o) in enumerate(zip(weights, out, strict=True)) if w and not o.requires_grad
    ]
    if gradient and detached:
        raise MeasureError(
            f"'{model_path}': the weighted layer {', '.join(detached)} carries no gradient (a label map, such as a "
            "segmentation head), and this loss is differentiated through the network.",
            "Weigh feature layers instead, or compare the maps themselves (a Static registration).",
        )


#: The Hugging Face repository of the IMPACT feature models and of their registry, ``models.json``.
MODELS_REPO = "VBoussot/impact-torchscript-models"

#: The models and their registry are read at this revision of ``MODELS_REPO``, so a re-export on the Hub cannot change
#: a result silently. A ref names another revision with ``repo@revision:path``.
MODELS_REVISION = "47ffad660e67cebbac8d74aecf0fa181c5dac042"


def _is_local_ref(ref: str) -> bool:
    """A model ref without a ``:`` is a local file, and so is a Windows drive-letter path (``C:/models/m.pt``)."""
    return ":" not in ref or bool(re.match(r"^[A-Za-z]:[\\/]", ref))


def split_model_ref(ref: str) -> tuple[str, str, str]:
    """``repo[@revision]:path`` -> ``(repo, revision, path)``, the revision defaulting to ``MODELS_REVISION``."""
    repo, filename = ref.split(":", 1)
    repo, _, revision = repo.partition("@")
    return repo, revision or MODELS_REVISION, filename


def fetch_model(ref: str) -> Path:
    """The file of a model ref: a local path, which must exist, or a Hugging Face ``repo[@revision]:path`` fetched at
    its pinned revision."""
    if _is_local_ref(ref):
        local = Path(ref).expanduser().resolve()
        if not local.is_file():
            raise MeasureError(f"The local model ref '{ref}' does not exist (resolved to '{local}').")
        return local
    repo, revision, filename = split_model_ref(ref)
    download = _hf_hub_download("IMPACT")
    return Path(download(repo_id=repo, filename=filename, revision=revision, repo_type="model"))  # nosec B615


def models_registry() -> dict:
    """``models.json``: per model, its ``dimension``, input ``numberofchannels``, per-layer receptive field ``fov``
    (null for a model that sees whole images only) and, when its input must divide by some size, that ``multiple``.
    ``KONFAI_IMPACT_MODELS_REGISTRY`` names a local registry instead, for local models or offline."""
    local = os.environ.get("KONFAI_IMPACT_MODELS_REGISTRY", "")
    path = Path(local) if local else fetch_model(f"{MODELS_REPO}:models.json")
    return json.loads(path.read_text(encoding="utf-8"))


def model_key(ref: str) -> str:
    """The registry key of a model ref: its path in the repository, or a local ref itself."""
    return ref if _is_local_ref(ref) else ref.split(":", 1)[1]


#: The IMPACT distances, as itk-impact defines them: every one positive and 0 at a perfect match, on feature maps
#: ``[B, C, *spatial]`` (channel axis 1), averaged over the voxels a mask ``[B, 1, *spatial]`` weighs (all without one).
DISTANCES = ("L1", "L2", "Dice", "Cosine", "L1Cosine", "NCC", "LNCC")
_EPS = 1e-6
#: itk-impact's L1Cosine: the cosine damped by exp(-lambda |f - m|) per channel.
_L1COSINE_LAMBDA = 0.1
#: The floor of a local variance in the LNCC, as FireANTs' own cross-correlation.
_LNCC_VARIANCE_FLOOR = 1e-5


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """The mean of ``values`` [B, K, *spatial] over its K maps and the voxels ``mask`` [B, 1, *spatial] weighs."""
    if mask is None:
        return values.mean()
    return (values * mask).sum() / (mask.sum() * values.shape[1]).clamp_min(1.0)


def _lncc_piece(moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None, kernel: int) -> torch.Tensor:
    """1 - the squared local correlation of each channel over a window of ``kernel`` voxels a side, averaged."""
    pool = F.avg_pool3d if moved.dim() == 5 else F.avg_pool2d
    mean = lambda t: pool(t, kernel, stride=1, padding=kernel // 2, count_include_pad=False)  # noqa: E731
    moved_mean, fixed_mean = mean(moved), mean(fixed)
    covariance = mean(moved * fixed) - moved_mean * fixed_mean
    moved_var = (mean(moved * moved) - moved_mean * moved_mean).clamp_min(_LNCC_VARIANCE_FLOOR)
    fixed_var = (mean(fixed * fixed) - fixed_mean * fixed_mean).clamp_min(_LNCC_VARIANCE_FLOOR)
    return 1.0 - _masked_mean(covariance * covariance / (moved_var * fixed_var), mask)


def _lncc(moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None, kernel: int, chunk: int) -> torch.Tensor:
    """The LNCC ``chunk`` channels a pass (0: all at once), each pass run again in the backward instead of keeping its
    windowed sums, so the peak follows the chunk rather than the channel count, for the same value and gradient."""
    channels = moved.shape[1]
    step = chunk if 0 < chunk < channels else channels
    total: torch.Tensor | None = None
    for start in range(0, channels, step):
        stop = min(start + step, channels)
        piece = torch.utils.checkpoint.checkpoint(
            _lncc_piece, moved[:, start:stop], fixed[:, start:stop], mask, kernel, use_reentrant=False
        )
        weighted = piece * (stop - start) / channels
        total = weighted if total is None else total + weighted
    if total is None:
        raise RuntimeError("the feature maps carry no channel to correlate")
    return total


def distance(
    name: str,
    moved: torch.Tensor,
    fixed: torch.Tensor,
    mask: torch.Tensor | None = None,
    kernel: int = 5,
    chunk: int = 0,
) -> torch.Tensor:
    """One layer's IMPACT distance between two feature maps, ``mask`` weighing their voxels (its channel axis 1)."""
    # Unmasked, L1 and L2 keep no map of the differences for the backward. torch.dist sums in the maps' type, which
    # half precision overflows: half maps take the mean.
    if name == "L1":
        if mask is None and moved.dtype != torch.float16:
            return torch.dist(moved, fixed, 1) / moved.numel()
        return _masked_mean((moved - fixed).abs(), mask)
    if name == "L2":
        if mask is None:
            return F.mse_loss(moved, fixed)
        return _masked_mean((moved - fixed).pow(2), mask)
    if name in ("Cosine", "L1Cosine"):
        norms = moved.norm(2, 1, keepdim=True) * fixed.norm(2, 1, keepdim=True)
        cosine = (moved * fixed).sum(1, keepdim=True) / (norms + _EPS)
        if name == "Cosine":
            return 1.0 - _masked_mean(cosine, mask)
        return 1.0 - _masked_mean(cosine * torch.exp(-_L1COSINE_LAMBDA * (moved - fixed).abs()), mask)
    if name == "Dice":
        # Soft, on the raw activations: a voxel with no activation on either side counts as a match.
        intersection, union = (moved * fixed).sum(1, keepdim=True), (moved + fixed).sum(1, keepdim=True)
        empty = union == 0
        overlap = torch.where(empty, torch.ones_like(union), 2 * intersection / torch.where(empty, 1.0, union))
        return 1.0 - _masked_mean(overlap, mask)
    if name == "NCC":
        # Each channel's correlation over every weighed voxel, averaged over the channels.
        axes = [0, *range(2, moved.dim())]
        weight = torch.ones_like(moved[:, :1]) if mask is None else mask
        count = weight.sum(axes, keepdim=True).clamp_min(1.0)
        moved_c = moved - (moved * weight).sum(axes, keepdim=True) / count
        fixed_c = fixed - (fixed * weight).sum(axes, keepdim=True) / count
        norms = ((moved_c * moved_c * weight).sum(axes) * (fixed_c * fixed_c * weight).sum(axes)).sqrt()
        return 1.0 - ((moved_c * fixed_c * weight).sum(axes) / (norms + _EPS)).mean()
    if name == "LNCC":
        return _lncc(moved, fixed, mask, kernel, chunk)
    raise MeasureError(f"Unknown IMPACT distance '{name}'.", f"Choose one of {', '.join(DISTANCES)}.")


class Distance(torch.nn.Module):
    """An IMPACT distance as a loss module: a mask weighs the voxels of both maps, nearest-resampled to them."""

    def __init__(self, name: str, kernel: int = 5) -> None:
        super().__init__()
        if name not in DISTANCES:
            raise MeasureError(f"Unknown IMPACT distance '{name}'.", f"Choose one of {', '.join(DISTANCES)}.")
        self.name, self.kernel = name, int(kernel)

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = F.interpolate(mask.float(), size=tuple(moved.shape[2:]), mode="nearest")
        return distance(self.name, moved, fixed, mask, self.kernel)


def _feature_mask(mask: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """The voxels of one sample's feature map a {0,1} mask keeps: the mask nearest-resampled to the map's
    spatial size, flattened to one boolean per voxel."""
    return F.interpolate(mask.float(), mode="nearest", size=tuple(feature.shape[2:])).reshape(-1) == 1


def _patch_views(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, patch_shape: list[int] | None
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Yield aligned (output, target, mask) views: ``ModelPatch`` tiles when ``patch_shape`` is set,
    the whole tensors otherwise."""
    if patch_shape is None:
        yield output, target, mask
        return
    model_patch = ModelPatch(patch_shape)
    model_patch.load(list(output.shape[2:]))
    for index in range(model_patch.get_size(0)):
        yield (
            model_patch.get_data(output, index, 0, True),
            model_patch.get_data(target, index, 0, True),
            model_patch.get_data(mask, index, 0, True) if mask is not None else None,
        )


def _patch_feature_loss(
    model: torch.nn.Module,
    output_patch: torch.Tensor,
    target_patch: torch.Tensor,
    mask_patch: torch.Tensor | None,
    output_rest: list[torch.Tensor],
    target_rest: list[torch.Tensor],
    weights: list[float],
    loss_function: torch.nn.Module,
    project: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
) -> torch.Tensor:
    """The weighted per-layer feature distance of one patch."""
    loss = torch.zeros(1, device=output_patch.device)
    for weight, output_feature, target_feature in zip(
        weights, model(output_patch, *output_rest), model(target_patch, *target_rest), strict=False
    ):
        if weight == 0:
            continue
        if project is not None:
            output_feature, target_feature = project(output_feature, target_feature)
        if isinstance(loss_function, Distance):  # the mask weighs the voxels of each layer
            layer_loss = weight * loss_function(output_feature.float(), target_feature.float(), mask_patch)
            if not layer_loss.isnan():
                loss = loss + layer_loss
            continue
        if mask_patch is not None:
            selection = _feature_mask(mask_patch, output_feature)
            if not torch.any(selection):
                continue
            # Voxels are selected, not elements: the features stay [1, C, voxels], so a distance that reduces
            # over the channel axis (cosine, Dice, per-channel NCC) still finds it.
            output_feature = output_feature.flatten(2)[..., selection]
            target_feature = target_feature.flatten(2)[..., selection]
        layer_loss = weight * loss_function(output_feature.float(), target_feature.float())
        if not layer_loss.isnan():
            loss = loss + layer_loss
    return loss


def _masked_feature_loss(
    model: torch.nn.Module,
    output: list[torch.Tensor],
    target: list[torch.Tensor],
    weights: list[float],
    loss_function: torch.nn.Module,
    mask: torch.Tensor | None,
    patch_shape: list[int] | None,
    project: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    checkpoint: bool = False,
) -> tuple[torch.Tensor, int]:
    """Weighted per-layer feature distance between two preprocessed inputs, tiled and masked.

    ``output`` / ``target`` are ``[tensor, nb_layer, stats]`` triples as fed to an IMPACT TorchScript
    extractor. A patch without a mask voxel is skipped; a layer whose resampled mask vanishes, or whose
    loss is NaN, contributes nothing. Returns the summed loss and the number of scored patches: the
    caller divides. With ``checkpoint``, each patch keeps only its inputs until the backward, which
    recomputes its activations: a registration that differentiates through the network holds one
    patch's activations at a time instead of all of them.
    """
    loss = torch.zeros(1, device=output[0].device, requires_grad=True)
    true_nb = 0
    for output_patch, target_patch, mask_patch in _patch_views(output[0], target[0], mask, patch_shape):
        if mask_patch is not None and not torch.any(mask_patch == 1):
            continue
        args = (model, output_patch, target_patch, mask_patch, output[1:], target[1:], weights, loss_function, project)
        if checkpoint:
            patch_loss = torch.utils.checkpoint.checkpoint(_patch_feature_loss, *args, use_reentrant=False)
        else:
            patch_loss = _patch_feature_loss(*args)
        loss = loss + patch_loss
        true_nb += 1
    return loss, true_nb


def _feature_loss_mean(slices: Iterable[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, float | torch.Tensor]:
    """The slice losses summed and divided by the number of scored patches, with the value to report
    as a detached 0-d tensor read off its device lazily (``Measure._materialize``). No scored patch
    (a mask with no foreground) would divide by zero: the loss is then its zero seed, returned
    as-is, and the value is NaN."""
    losses, counts = zip(*slices, strict=True)
    loss, true_nb = reduce(torch.add, losses), sum(counts)
    if true_nb == 0:
        return loss, np.nan
    loss = loss / true_nb
    return loss, loss.detach()


def grid_size(extent: list[float] | None, voxel_size: list[float], shape: tuple[int, ...]) -> tuple[int, ...]:
    """The tensor shape ``[S, P, L]`` of the grid at ``voxel_size`` (mm, ITK order x y z) over ``extent`` (mm along L,
    P, S), rounded as itk-impact's ImageToTensorFilter rounds it; the voxels' own shape stands in for a missing
    extent."""
    extent = extent if extent is not None else [float(size) for size in reversed(shape)]
    return tuple(
        max(1, int(side / step + 0.5)) for side, step in zip(reversed(extent), reversed(voxel_size), strict=True)
    )


def resampled(
    tensor: torch.Tensor, size: tuple[int, ...], mode: str = "bilinear", padding: str = "zeros"
) -> torch.Tensor:
    """``tensor`` [B, C, *spatial] on a grid of ``size`` over the same extent, as itk-impact resamples an image for a
    model (ImageToTensorFilter): new voxel i of an axis at old continuous index i * old / new, the first voxels aligned,
    linearly interpolated and not smoothed; differentiable. The same mapping brings a model's features back onto the
    image (``padding`` "border" there, past the last feature voxel)."""
    if tuple(tensor.shape[2:]) == tuple(size):
        return tensor
    old_sizes = tensor.shape[2:]
    axes = [
        2 * (torch.arange(new, dtype=torch.float64) * old / new) / max(old - 1, 1) - 1
        for old, new in zip(old_sizes, size, strict=True)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1], dim=-1)  # grid_sample reads (x, y, z)
    grid = grid.to(tensor.device, tensor.dtype if tensor.is_floating_point() else torch.float32)
    source = tensor if tensor.is_floating_point() else tensor.float()
    return F.grid_sample(
        source, grid[None].expand(tensor.shape[0], *grid.shape), mode=mode, padding_mode=padding, align_corners=True
    )


def swept_order(seed: int) -> list[int]:
    """The axes of a ``[B, C, D, H, W]`` volume with the spatial axis a 2-D network sweeps first, drawn for ``seed``:
    a dense loss that draws no points sees the three orientations over its evaluations, as elastix draws a plane per
    point."""
    axis = int(torch.randint(3, (1,), generator=torch.Generator().manual_seed(seed)))
    return [0, 1, 2 + axis, *(2 + other for other in range(3) if other != axis)]


#: A 2-D model's PCA basis is fitted on the target features of at most this many slices, spread evenly along the axis
#: it sweeps, as itk-impact fits it.
PCA_SLICES = 32


def _pca_transform(feature: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project a feature map ``[B, C, spatial...]`` onto a PCA basis ``[C, K]`` -> ``[B, K, spatial...]``, centring the
    input by its own per-channel mean first (itk-impact ``pca_transform``)."""
    shape = feature.shape
    flat = feature.reshape(shape[0], shape[1], -1)
    flat = flat - flat.mean(dim=2, keepdim=True)
    projected = torch.einsum("bcn,ck->bkn", flat, basis)
    return projected.reshape(shape[0], basis.shape[1], *shape[2:])


def pca_project(
    output_feature: torch.Tensor, target_feature: torch.Tensor, components: int, dimension: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both feature maps reduced to their top ``components`` principal components, fitted on the TARGET for every batch
    sample (a batch mixes unrelated cases): a channel-covariance eigendecomposition, as itk-impact's per-image
    ``pca_fit``. A 3-D model's basis is fitted on the whole map; a 2-D model's volume on ``PCA_SLICES`` slices spread
    along its first spatial axis, each image centred by its own mean over those slices."""
    channels = target_feature.shape[1]
    k = min(components, channels)
    if dimension == 3 or target_feature.dim() < 5:  # sampled points: every point, as itk-impact
        flat = target_feature.detach().reshape(target_feature.shape[0], channels, -1).float()
        projected_output: list[torch.Tensor] = []
        projected_target: list[torch.Tensor] = []
        for b in range(flat.shape[0]):
            centered = flat[b] - flat[b].mean(dim=1, keepdim=True)
            covariance = centered @ centered.t() / max(flat.shape[2] - 1, 1)
            del centered  # a full copy of the target's features, not needed past the covariance
            _, eigenvectors = torch.linalg.eigh(covariance)
            basis = eigenvectors[:, channels - k :].to(target_feature.dtype)  # {C, K}, largest-eigenvalue
            projected_output.append(_pca_transform(output_feature[b : b + 1], basis))
            projected_target.append(_pca_transform(target_feature[b : b + 1], basis))
        return torch.cat(projected_output), torch.cat(projected_target)
    depth = target_feature.shape[2]
    index = torch.linspace(0, depth - 1, min(depth, PCA_SLICES)).round().long().to(target_feature.device)
    projected: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    for b in range(target_feature.shape[0]):
        fixed_sample = target_feature[b].detach().index_select(1, index).reshape(channels, -1).float()
        moving_sample = output_feature[b].detach().index_select(1, index).reshape(channels, -1).float()
        centred = fixed_sample - fixed_sample.mean(dim=1, keepdim=True)
        covariance = centred @ centred.t() / max(centred.shape[1] - 1, 1)
        del centred
        basis = torch.linalg.eigh(covariance)[1][:, channels - k :].to(target_feature.dtype)  # largest components
        for side, feature, sample in ((0, output_feature[b], moving_sample), (1, target_feature[b], fixed_sample)):
            flat = feature.reshape(channels, -1) - sample.mean(dim=1, keepdim=True).to(feature.dtype)
            projected[side].append(torch.einsum("cn,ck->kn", flat, basis).reshape(k, *feature.shape[1:]))
    return torch.stack(projected[0]), torch.stack(projected[1])


class _SliceSweep(torch.nn.Module):
    """A 2-D network swept over a volume ``[B, C, D, H, W]`` slice by slice along its first spatial axis, as itk-impact
    sweeps it for its feature maps, each slice told the plane's own axes. Each output layer comes back as a volume
    ``[B, C_l, D, H_l, W_l]``, like a 3-D network's."""

    def __init__(self, network: torch.nn.Module, batch: int = 8) -> None:
        super().__init__()
        self.network, self.batch = network, batch

    def forward(self, image: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor, *_) -> list[torch.Tensor]:
        batch, channels, depth = image.shape[:3]
        slices = image.movedim(2, 1).reshape(batch * depth, channels, *image.shape[3:])
        plane = torch.eye(2, dtype=torch.int16)
        pieces = []
        for start in range(0, slices.shape[0], self.batch):
            try:
                pieces.append(self.network(slices[start : start + self.batch], nb_layers, stats, plane))
            except RuntimeError as error:
                raise RuntimeError(
                    f"a 2D feature network refused slices of {tuple(slices.shape[2:])} voxels: "
                    f"{str(error).strip().splitlines()[-1]}"
                ) from error
        layers = [torch.cat([piece[index] for piece in pieces]) for index in range(len(pieces[0]))]
        return [layer.reshape(batch, depth, *layer.shape[1:]).movedim(1, 2) for layer in layers]


def _statistics(tensor: torch.Tensor) -> list[Attribute]:
    """Each sample's own ``[ImageMin, ImageMax, ImageMean, ImageStd]``, for a caller with no KonfAI attributes."""
    detached = tensor.detach().float()  # not flattened: a permuted tensor is reduced in place, not copied
    return [
        Attribute(
            {
                "ImageMin": float(sample.min()),
                "ImageMax": float(sample.max()),
                "ImageMean": float(sample.mean()),
                "ImageStd": float(sample.std()),
            }
        )
        for sample in detached
    ]


def _denormalized(tensor: torch.Tensor, attributes: list[Attribute]) -> torch.Tensor:
    """``tensor`` mapped back to intensities from the per-sample ``Mean``/``Std`` (``Standardize``) or
    ``Min``/``Max`` (``Normalize``) attributes; untouched when neither is recorded."""

    def per_sample(key: str) -> torch.Tensor:
        values = [float(attribute[key]) for attribute in attributes]
        return torch.tensor(values, device=tensor.device).view(-1, *([1] * (tensor.dim() - 1)))

    if "Mean" in attributes[0] and "Std" in attributes[0]:
        return tensor * per_sample("Std") + per_sample("Mean")
    if "Min" in attributes[0] and "Max" in attributes[0]:
        return (tensor + 1) / 2 * (per_sample("Max") - per_sample("Min")) + per_sample("Min")
    return tensor


#: The smallest tile, per axis, a feature pass is cut into when the whole image does not fit the card.
MIN_TILE = 16
#: The tile a volume extraction falls back to when the whole image does not fit the card; halved after.
FIRST_FEATURE_TILE = 256
#: Input voxels one batch of a sampled pass feeds the network (patches x patch volume); halved when it does not fit.
PATCH_BUDGET = 1 << 22


def normalized_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    """Each voxel's feature vector scaled as itk-impact's NormalizeFeatureChannels: 'l2' to unit length,
    'standardized' to zero mean and unit (unbiased) deviation over its channels, anything else as it is."""
    if mode == "l2":
        return F.normalize(features, dim=1)
    if mode == "standardized":
        return (features - features.mean(1, keepdim=True)) / features.std(1, keepdim=True).clamp_min(1e-6)
    return features


def channel_subset(
    moved: torch.Tensor, fixed: torch.Tensor, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same ``count`` channels of both maps, drawn from ``seed`` (0, or at least all of them: every channel)."""
    channels = moved.shape[1]
    if count <= 0 or count >= channels:
        return moved, fixed
    chosen = torch.randperm(channels, generator=torch.Generator().manual_seed(seed))[:count].to(moved.device)
    return moved.index_select(1, chosen), fixed.index_select(1, chosen)


def draw_centres(
    shape: tuple[int, ...],
    mask: torch.Tensor | None,
    patch: int,
    fraction: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor | None:
    """The points one sampled evaluation reads, as [N, 3] voxel indices drawn anew with ``generator``: ``fraction`` of
    the voxels a ``patch`` cube around the point keeps inside the image (elastix's SampleCheck; a patch of 1 is every
    voxel), and of those the mask keeps. None when the mask keeps none."""
    low = patch // 2
    extents = [size - patch + 1 for size in shape]
    if min(extents) < 1:
        raise ValueError(
            f"voxel_sampling in Jacobian mode runs the network on {patch}-voxel patches, larger than the image of this "
            f"scale {list(shape)}: drop the coarsest 'scales' level, or use mode Static."
        )
    if mask is None:
        count = max(1, round(fraction * math.prod(extents)))
        axes = [torch.randint(extent, (count,), generator=generator) + low for extent in extents]
        return torch.stack(axes, dim=1).to(device)
    window: tuple[int | slice, ...] = (0, 0, *[slice(low, low + extent) for extent in extents])
    inner = mask[window]
    candidates = torch.nonzero(inner >= 0.5)
    if len(candidates) == 0:
        return None
    count = max(1, round(fraction * len(candidates)))
    return candidates[torch.randint(len(candidates), (count,), generator=generator).to(candidates.device)] + low


def _patches(image: torch.Tensor, centres: torch.Tensor, patch: int) -> torch.Tensor:
    """The ``patch`` cube of ``image`` [1, C, D, H, W] around each of ``centres`` [n, 3], as [n, C, patch, patch,
    patch]: a gather, so the gradient flows back to the image."""
    _, height, width = image.shape[2:]
    offsets = torch.arange(patch, device=centres.device) - patch // 2
    z = (centres[:, 0, None] + offsets)[:, :, None, None]
    y = (centres[:, 1, None] + offsets)[:, None, :, None]
    x = (centres[:, 2, None] + offsets)[:, None, None, :]
    return image.flatten(2)[0][:, (z * height + y) * width + x].movedim(0, 1)


def _spacing(shape: tuple[int, ...], extent: list[float] | None) -> tuple[torch.Tensor, torch.Tensor]:
    """The voxels' count and side in mm along (L, P, S), a missing ``extent`` counting one mm a voxel."""
    size = torch.tensor(list(reversed(shape)), dtype=torch.float64)
    return size, torch.tensor(extent, dtype=torch.float64) / size if extent is not None else torch.ones(3).double()


def _plane_grids(
    centres: torch.Tensor,
    patch: int,
    shape: tuple[int, ...],
    extent: list[float] | None,
    seed: int,
    steps: list[float] | None = None,
) -> torch.Tensor:
    """For a 2D network, the ``patch`` x ``patch`` square each of ``centres`` [n, 3] is cut on, on a plane drawn at
    random for each point, as grid_sample coordinates [n, patch, patch, 3] (align_corners, ``x, y, z`` = L, P, S).

    elastix's metric draws each point's plane from three angles uniform in [0, 2 pi), the first two columns of Rz Ry
    Rx (itk-impact's PatchPlane), so that the network sees the anatomy in every orientation. The square is laid out in
    mm, one step ``steps`` along each of the plane's two axes, or the finest voxel side without them: ``extent`` is
    the image's size in mm along (L, P, S), the voxels' shape standing in when None."""
    size, spacing = _spacing(shape, extent)
    a, b, c = (
        torch.rand(3, len(centres), generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 2 * math.pi
    )
    u = torch.stack([c.cos() * b.cos(), c.sin() * b.cos(), -b.sin()], dim=1)
    v = torch.stack(
        [
            c.cos() * b.sin() * a.sin() - c.sin() * a.cos(),
            c.sin() * b.sin() * a.sin() + c.cos() * a.cos(),
            b.cos() * a.sin(),
        ],
        dim=1,
    )
    side_u, side_v = steps if steps is not None else (float(spacing.min()),) * 2
    ticks = torch.arange(patch, dtype=torch.float64) - (patch - 1) / 2
    offsets = (ticks * side_u)[None, :, None, None] * u[:, None, None, :] + (ticks * side_v)[None, None, :, None] * v[
        :, None, None, :
    ]
    index = centres.detach().cpu().flip(1).double()[:, None, None, :] + offsets / spacing
    return (2 * index / (size - 1) - 1).float().to(centres.device)


def _cube_grids(
    centres: torch.Tensor, patch: int, shape: tuple[int, ...], extent: list[float] | None, voxel_size: list[float]
) -> torch.Tensor:
    """For a 3D network with a ``voxel_size``, the ``patch`` cube each of ``centres`` [n, 3] is cut on, one step that
    voxel_size (mm, ITK order x y z) along each axis, as grid_sample coordinates [n, patch, patch, patch, 3]
    (align_corners, ``x, y, z`` = L, P, S): elastix's patch at the model's resolution."""
    size, spacing = _spacing(shape, extent)
    ticks = torch.arange(patch, dtype=torch.float64) - (patch - 1) / 2
    step = torch.tensor(voxel_size, dtype=torch.float64) / spacing  # voxels of the image per model voxel, (L, P, S)
    along_s, along_p, along_l = torch.meshgrid(ticks * step[2], ticks * step[1], ticks * step[0], indexing="ij")
    offsets = torch.stack([along_l, along_p, along_s], dim=-1)  # [patch, patch, patch, 3] in (L, P, S) voxels
    index = centres.detach().cpu().flip(1).double()[:, None, None, None, :] + offsets[None]
    return (2 * index / (size - 1) - 1).float().to(centres.device)


def _grid_patches(image: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """``image`` [1, C, D, H, W] sampled on each point's grid, as [n, C, patch, patch] for a plane (``_plane_grids``)
    or [n, C, patch, patch, patch] for a cube (``_cube_grids``): trilinear, so the gradient flows back to the image;
    outside the image, zeros."""
    count, *sides = grids.shape[:-1]
    flat = grids.reshape(1, count * sides[0], *sides[1:], 3) if len(sides) == 3 else grids[None]
    sampled = F.grid_sample(image, flat.to(image.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[0].reshape(image.shape[1], count, *sides).movedim(1, 0)


def _centre_features(
    network: torch.nn.Module,
    kept: list[int],
    patch: int,
    moved: torch.Tensor,
    fixed: torch.Tensor,
    centres: torch.Tensor,
    moved_rest: list[torch.Tensor],
    fixed_rest: list[torch.Tensor],
    grids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Each kept layer's feature vector at the centre of the patch around each point, [n, C], moved layers then fixed
    ones: the network runs on the patches, and a layer the network downsamples keeps its middle voxel, as itk-impact's
    GetCentersIndexLayers does. With ``grids``, each patch is sampled on its point's grid: a square on a random plane
    for a 2D network (``_plane_grids``), a cube at the model's voxel_size for a 3D one (``_cube_grids``)."""
    features = []
    for image, rest in ((moved, moved_rest), (fixed, fixed_rest)):
        if grids is None:
            layers = network(_patches(image, centres, patch), *rest)
        elif grids.dim() == 4:  # a 2D network's direction is the plane's own axes
            layers = network(_grid_patches(image, grids), *rest[:2], torch.eye(2, dtype=torch.int16))
        else:
            layers = network(_grid_patches(image, grids), *rest)
        for layer in kept:
            output = layers[layer]
            features.append(output[(slice(None), slice(None), *[size // 2 for size in output.shape[2:]])].float())
    return tuple(features)


def _feature_grid(shape: tuple[int, ...], patch: int, overlap: float, multiple: int) -> ModelPatch:
    """The grid one extraction runs on, cut as the predictor cuts a network's input: ``patch`` 0, or an axis no longer
    than it, is one tile spanning that axis (a cube wider than the axis would pad it up to the tile); the tiles share
    ``overlap`` of their width under a raised-cosine window that sums to one; a free axis rounds up to ``multiple``
    (an encoder-decoder's skip connections), cropped back off at blend time."""
    size = [int(patch) if 0 < patch < extent else 0 for extent in shape]
    share: float | int = float(overlap) if patch > 0 else 0
    grid = ModelPatch(size, share)
    if multiple > 1:
        grid.free_axis_multiple = [int(multiple)] * len(shape)
    grid.patch_combine = Cosinus()
    kept = blend_axes(grid.patch_size)
    grid.patch_combine.set_patch_config(kept, blend_overlap(share, kept))
    grid.load(list(shape))
    return grid


class ImpactFeatureModel:
    """An IMPACT TorchScript feature extractor and what its inputs need: the channel count it expects
    (a narrower input is repeated to it), the per-layer weights, the tile it is fed (``None`` = the
    whole tensor) and whether a standardized input is mapped back to intensities first. The model is
    loaded on first use."""

    def __init__(
        self,
        model_path: str,
        in_channels: int,
        weights: list[float],
        shape: list[int] | None,
        dim: int,
        denormalize: bool = False,
    ) -> None:
        self.model_path = model_path
        self.in_channels = in_channels
        self.weights = weights
        self.shape = shape
        self.dim = dim
        self.denormalize = denormalize
        # The registry's per-layer receptive field (None: whole images only) and input multiple (0: any size).
        self.fov: list[int] | None = None
        self.multiple = 0
        # The direction of the images it is fed, handed to the network, which reorients them as it was trained (None:
        # as given); the grid it sees them on, in mm (ITK order; a 2-D network's two plane axes; None: as given);
        # and float16 on a GPU.
        self.direction: torch.Tensor | None = None
        self.voxel_size: list[float] | None = None
        self.half = False
        # Each tile under a checkpoint (see _masked_feature_loss): for a caller that differentiates through
        # the network on volumes too large to keep every tile's activations.
        self.checkpoint = False
        # The patches one batch of ``sampled`` holds, once one has run out of memory (0: PATCH_BUDGET's).
        self.batch = 0
        self.model: torch.nn.Module | None = None

    @classmethod
    def download(
        cls,
        filename: str,
        in_channels: int,
        weights: list[float],
        shape: list[int],
        repo_id: str = "VBoussot/impact-torchscript-models",
        denormalize: bool = False,
    ) -> "ImpactFeatureModel":
        """The model ``filename`` of the HuggingFace ``repo_id``, probed once on the CPU. ``shape`` is the
        tile, its length the dimension; an entry ``<= 0`` scores the whole tensor instead."""
        download = _hf_hub_download("IMPACT")
        model_path = download(repo_id=repo_id, filename=filename, repo_type="model", revision=None)  # nosec B615
        tile = shape if all(s > 0 for s in shape) else None
        _check_feature_model(model_path, in_channels, tile or [224] * len(shape), weights)
        return cls(model_path, in_channels, weights, tile, len(shape), denormalize)

    @classmethod
    def from_ref(cls, ref: str, weights: list[float]) -> "ImpactFeatureModel":
        """The model ``ref`` names (``repo[@revision]:path`` at ``MODELS_REVISION``, or a local file), shaped by its
        registry entry (a local model without one: 3-D, one channel, no receptive field). It sees the whole tensor
        it is handed; ``check`` probes it."""
        path = str(fetch_model(ref))
        local = _is_local_ref(ref) and not os.environ.get("KONFAI_IMPACT_MODELS_REGISTRY")
        entry = {} if local else models_registry().get(model_key(ref), {})
        model = cls(path, int(entry.get("numberofchannels", 1)), weights, None, int(entry.get("dimension", 3)))
        model.fov, model.multiple = entry.get("fov"), int(entry.get("multiple", 0))
        return model

    def check(self, gradient: bool = False) -> None:
        """Probe the model once on the CPU, 64 voxels a side: one output per weight, and with ``gradient`` a gradient
        through every weighted layer, for a loss differentiated through the network."""
        _check_feature_model(self.model_path, self.in_channels, [64] * self.dim, self.weights, gradient)

    @property
    def receptive_field(self) -> int:
        """The receptive field, in voxels, of the deepest weighted layer: the patch a point needs around it."""
        if self.fov is None:
            raise MeasureError(f"The registry gives '{self.model_path}' no receptive field: it sees whole images only.")
        deepest = max(index for index, weight in enumerate(self.weights) if weight)
        if deepest >= len(self.fov):
            raise MeasureError(f"'{self.model_path}' weighs its layer {deepest + 1}, past the {len(self.fov)} it has.")
        return int(self.fov[deepest])

    def _half(self, device: torch.device) -> bool:
        # float16 has no 3-D pooling on the CPU.
        return self.half and device.type == "cuda"

    def network(self, device: torch.device) -> torch.nn.Module:
        """The TorchScript network on ``device``, in float16 there with ``half``; a 2-D one swept over a volume's slices
        along its first spatial axis."""
        if self.model is None:
            self.model = torch.jit.load(self.model_path, map_location="cpu").eval()  # nosec B614
        network = self.model.to(device)
        network = network.half() if self._half(device) else network.float()
        return _SliceSweep(network) if self.dim == 2 else network

    def inputs(self, tensor: torch.Tensor, attribute: Attribute) -> list[torch.Tensor]:
        """The ``[tensor, nb_layer, stats(, direction)]`` inputs the extractor takes, for ONE sample: the flat
        ``[ImageMin, ImageMax, ImageMean, ImageStd]`` itk-impact passes, which a model reads only when
        ``stats.numel() == 4`` and otherwise replaces with statistics of the tensor it is handed."""
        if tensor.shape[1] != self.in_channels:
            tensor = tensor.repeat(1, self.in_channels, *([1] * (tensor.dim() - 2)))
        if self.denormalize:
            tensor = _denormalized(tensor, [attribute])
        if self._half(tensor.device):
            tensor = tensor.half()
        stats = [float(attribute[key]) for key in ("ImageMin", "ImageMax", "ImageMean", "ImageStd")]
        direction = [] if self.direction is None else [self.direction]
        return [tensor, torch.tensor([len(self.weights)]), torch.tensor(stats), *direction]

    @property
    def kept(self) -> list[int]:
        """The weighted layers."""
        return [index for index, weight in enumerate(self.weights) if weight]

    @torch.no_grad()
    def _volume(self, image: torch.Tensor, normalization: str, patch: int, overlap: float) -> list[torch.Tensor]:
        """``volume`` in tiles of ``patch``: the intensity statistics are the whole image's, so every tile is normalised
        alike, and a coarser layer (a segmentation network's deeper ones) is brought onto the tile to be blended."""
        network = self.network(image.device)
        statistics = _statistics(image)[0]
        grid = _feature_grid(tuple(image.shape[2:]), patch, overlap, self.multiple)
        accumulator = Accumulator(grid.get_patch_slices(), grid.patch_size, grid.patch_combine)
        channels: list[int] = []
        for index, (tile,) in enumerate(grid.disassemble(image)):
            outputs = network(*self.inputs(tile, statistics))
            layers = [
                normalized_features(
                    layer
                    if layer.shape[2:] == tile.shape[2:]
                    else F.interpolate(layer, size=tile.shape[2:], mode="trilinear", align_corners=False),
                    normalization,
                )
                for layer in (outputs[kept].float() for kept in self.kept)
            ]
            channels = [layer.shape[1] for layer in layers]
            accumulator.add_layer(index, layers[0] if len(layers) == 1 else torch.cat(layers, dim=1))
            del layers, outputs
        return list(torch.split(accumulator.assemble(), channels, dim=1))

    def volume(
        self, image: torch.Tensor, normalization: str = "none", patch: int = 0, overlap: float = 0.25
    ) -> tuple[list[torch.Tensor], int]:
        """Each weighted layer's features over ``image`` [1, C, *spatial], normalised (``normalized_features``) and on
        its grid, and the tile they took: ``patch`` a side (0: the whole image), tiles sharing ``overlap`` of their
        width blended by a cosine window. A pass that does not fit the card runs again in tiles of
        ``FIRST_FEATURE_TILE`` (or its largest halving shorter than the image), then half as wide each time."""
        tile = patch

        def narrow() -> bool:
            nonlocal tile
            if 0 < tile <= 2 * MIN_TILE:
                return False
            if tile == 0:
                tile = FIRST_FEATURE_TILE
                while tile > 2 * MIN_TILE and tile >= max(image.shape[2:]):
                    tile //= 2
            else:
                tile //= 2
            return True

        return halve_on_oom(lambda: self._volume(image, normalization, tile, overlap), narrow, image.is_cuda), tile

    def sampled(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        centres: torch.Tensor,
        patch: int,
        normalization: str = "none",
        extent: list[float] | None = None,
        seed: int = 0,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each weighted layer's moved and fixed features at ``centres`` [N, 3], as [1, C, N] maps, elastix's Jacobian
        scheme: the network runs on the ``patch`` cube around each point, a batch of patches at a time under a
        checkpoint, and only each layer's centre voxel is kept. A 2D network runs on the square around each point on a
        plane drawn at random for it (``_plane_grids``, from ``seed``); a model with a ``voxel_size`` sees its patch at
        that resolution (``_cube_grids``), laid out in mm over ``extent`` (L, P, S). A batch that does not fit the card
        is halved, and ``batch`` keeps it."""
        network, grids = self.network(moved.device), None
        if self.dim == 2:
            network = cast(torch.nn.Module, self.model)  # the network itself: one plane per point, not a sweep
            grids = _plane_grids(centres, patch, tuple(moved.shape[2:]), extent, seed, self.voxel_size)
        elif self.voxel_size is not None:
            grids = _cube_grids(centres, patch, tuple(moved.shape[2:]), extent, self.voxel_size)
        moved_inputs = self.inputs(moved, _statistics(moved)[0])
        fixed_inputs = self.inputs(fixed, _statistics(fixed)[0])
        kept = self.kept
        self.batch = self.batch or max(1, PATCH_BUDGET // patch**3)

        def run() -> list[tuple[torch.Tensor, ...]]:
            # No early stop: the centre features are the last tensors the recomputation saves, inside the TorchScript
            # network, which turns the checkpoint's internal stop into an empty RuntimeError.
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                return [
                    torch.utils.checkpoint.checkpoint(
                        _centre_features,
                        network,
                        kept,
                        patch,
                        moved_inputs[0],
                        fixed_inputs[0],
                        centres[start : start + self.batch],
                        moved_inputs[1:],
                        fixed_inputs[1:],
                        None if grids is None else grids[start : start + self.batch],
                        use_reentrant=False,
                    )
                    for start in range(0, len(centres), self.batch)
                ]

        with no_texpr_fuser():
            pieces = halve_on_oom(run, self.narrow_batch, moved.is_cuda)
        # [1, C, N]: the points stand for a map's voxels, which every point-wise distance reads alike.
        return [
            tuple(  # type: ignore[misc]
                normalized_features(torch.cat([piece[side + index] for piece in pieces]).t()[None], normalization)
                for side in (0, len(kept))
            )
            for index in range(len(kept))
        ]

    def narrow_batch(self) -> bool:
        """Half as many patches a batch of ``sampled``; False once a batch is one patch."""
        if self.batch <= 1:
            return False
        self.batch = (self.batch + 1) // 2
        return True

    def slice_losses(
        self,
        output: torch.Tensor,
        output_attributes: list[Attribute],
        target: torch.Tensor,
        target_attributes: list[Attribute],
        mask: torch.Tensor | None,
        loss_function: torch.nn.Module,
        project: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> Iterator[tuple[torch.Tensor, int]]:
        """The weighted feature distance and the number of scored patches, per sample, and per 2-D slice
        of it when the extractor is 2-D.

        One sample at a time because a model reads the statistics it is given only for one image
        (``stats.numel() == 4``): a whole batch at once would leave every sample normalized by the
        batch's own min/max (MIND) or mean/std (the MRI TS models), which is another case's intensities.
        """
        if self.model is None:
            self.model = torch.jit.load(self.model_path, map_location="cpu").eval()  # nosec B614
        self.model.to(output.device)
        slices = range(output.shape[2]) if output.dim() == 5 and self.dim == 2 else (slice(None),)
        for sample in range(output.shape[0]):
            for z in slices:
                yield _masked_feature_loss(
                    self.model,
                    self.inputs(output[sample : sample + 1, :, z], output_attributes[sample]),
                    self.inputs(target[sample : sample + 1, :, z], target_attributes[sample]),
                    self.weights,
                    loss_function,
                    mask[sample : sample + 1, :, z] if mask is not None else None,
                    self.shape,
                    project,
                    self.checkpoint,
                )


class IMPACTReg(CriterionWithAttribute):
    """The IMPACT loss: each weighted layer of a feature model compared between the output and the target, the weighted
    layers summed. ``distance`` compares every layer with an itk-impact distance (``DISTANCES``), a mask weighing its
    voxels; left unset, ``loss`` (a classpath) compares the voxels a mask keeps. Plain torch as well: without
    attributes each image is normalized by its own statistics."""

    def __init__(
        self,
        name: str = "Reg",
        model_name: str = "TS/M291.pt",
        shape: list[int] = [0, 0],
        in_channels: int = 3,
        loss: str = "torch:nn:L1Loss",
        weights: list[float] = [0, 1],
        pca: int = 0,
        distance: str | None = None,
        lncc_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.name = name
        self.loss: torch.nn.Module
        if distance is not None:
            self.loss = Distance(distance, lncc_kernel)
        else:
            loss_module, loss_class = get_module(loss, "konfai.metric.measure")
            self.loss = apply_config(os.environ.get("KONFAI_CONFIG_PATH"))(getattr(loss_module, loss_class))()
        self.pca = int(pca)
        self.model = ImpactFeatureModel.download(model_name, in_channels, weights, shape)

    def get_name(self):
        return self.name

    def _pca_project(
        self, output_feature: torch.Tensor, target_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both feature maps reduced to their top-``pca`` principal components, fitted on the target (``pca_project``)."""
        return pca_project(output_feature, target_feature, self.pca)

    def forward(  # type: ignore[override]  # the added keyword is CriterionWithAttribute's contract
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]] | None = None
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        mask = _sniffed_mask(targets, targets[-1])
        if attributes is None:
            attributes = [_statistics(output), _statistics(targets[0])]
        # The prediction and the target share the same intensity space, so a single target attribute
        # (single-group target such as ``CT``) is reused to normalize both output and target; a second
        # attribute set is honored when the target is multi-group.
        target_attributes = attributes[1] if len(attributes) > 1 else attributes[0]
        return _feature_loss_mean(
            self.model.slice_losses(
                output,
                attributes[0],
                targets[0],
                target_attributes,
                mask,
                self.loss,
                project=self._pca_project if self.pca > 0 else None,
            )
        )


class IMPACTSynth(CriterionWithAttribute):
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
        self.content = ImpactFeatureModel.download(
            model_content_name, in_channels_content, weights_criterion_content, shape_content, denormalize=True
        )
        self.style = ImpactFeatureModel.download(
            model_style_name, in_channels_style, weights_criterion_style, shape_style, denormalize=True
        )
        self.content_loss = torch.nn.MSELoss()
        self.style_loss = Gram()

    def forward(  # type: ignore[override]  # the added keyword is CriterionWithAttribute's contract
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        if len(targets) < 2:
            raise ValueError("At least two target tensors are required.")
        mask = _sniffed_mask(targets, targets[2]) if len(targets) == 3 else None
        return _feature_loss_mean(
            chain(
                self.content.slice_losses(output, attributes[0], targets[0], attributes[1], mask, self.content_loss),
                self.style.slice_losses(output, attributes[2], targets[1], attributes[2], mask, self.style_loss),
            )
        )


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
        self.loss = torch.nn.L1Loss()
        if train:
            repo_id, filename = "VBoussot/impact-torchscript-models", f"SAM2.1/{model_name}"
        else:
            repo_id, filename = "VBoussot/ImpactSynth", model_name
        download = _hf_hub_download("SAM_Perceptual")
        model_path = download(repo_id=repo_id, filename=filename, repo_type="model", revision=None)  # nosec B615
        self.model = ImpactFeatureModel(model_path, 3, [1.0] * 4 if weights is None else weights, [512, 512], 2)

    def forward(  # type: ignore[override]  # the added keyword is CriterionWithAttribute's contract
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        mask = _sniffed_mask(targets, targets[-1])
        # ``targets[0]`` is the reference (e.g. CT), normalized with its own stats; the same stats
        # normalize the prediction since both live in the same intensity space.
        return _feature_loss_mean(
            self.model.slice_losses(output, attributes[0], targets[0], attributes[0], mask, self.loss)
        )
