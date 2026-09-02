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

import os
from collections.abc import Callable, Iterable, Iterator
from functools import reduce
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.patching import ModelPatch
from konfai.metric.measure.adversarial import Gram
from konfai.metric.measure.base import CriterionWithAttribute, _require_optional
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute
from konfai.utils.errors import MeasureError
from konfai.utils.utils import get_module


def _hf_hub_download(criterion: str):
    """The ``hf_hub_download`` callable, imported at the call site: huggingface_hub is only needed
    by the IMPACT criteria, never by the rest of the metric package."""
    return _require_optional("huggingface_hub", criterion=criterion, extra="all").hf_hub_download


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


def _check_feature_model(model_path: str, in_channels: int, shape: list[int], nb_layer: int) -> None:
    """Probe a TorchScript feature extractor on the CPU: one output feature map per layer weight, or raise.

    Runs on the CPU only: the probe result is discarded, and touching a GPU here crashed CPU-only hosts
    and pinned every DDP rank to the same device.
    """
    model: torch.nn.Module = torch.jit.load(model_path, map_location=torch.device("cpu"))  # nosec B614
    dummy_input = torch.zeros((1, in_channels, *shape))
    try:
        out = model(dummy_input, torch.tensor([nb_layer]))
        if not isinstance(out, (list, tuple)):
            raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
        if nb_layer != len(out):
            raise ValueError(
                f"'{model_path}': mismatch between the number of weights ({nb_layer}) and the number of "
                f"model outputs ({len(out)}). Each output must have a corresponding weight."
            )
    except Exception as e:
        raise RuntimeError(
            f"[Model Sanity Check Failed]\nInput shape attempted: {dummy_input.shape}\nError: {type(e).__name__}: {e}"
        ) from e


def _feature_mask(mask: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """Nearest-resample a {0,1} mask to a feature map's spatial size, repeated over its channels."""
    resampled = F.interpolate(mask.float(), mode="nearest", size=tuple(feature.shape[2:]))
    return resampled.repeat((1, feature.shape[1], *([1] * (mask.dim() - 2)))) == 1


def _patch_views(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, patch_shape: list[int] | None
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Yield aligned (output, target, mask) views: ``ModelPatch`` tiles when ``patch_shape`` is set,
    the whole tensors otherwise."""
    if patch_shape is None:
        yield output, target, mask
        return
    model_patch = ModelPatch(patch_shape)
    model_patch.load(output.shape[2:])
    for index in range(model_patch.get_size(0)):
        yield (
            model_patch.get_data(output, index, 0, True),
            model_patch.get_data(target, index, 0, True),
            model_patch.get_data(mask, index, 0, True) if mask is not None else None,
        )


def _masked_feature_loss(
    model: torch.nn.Module,
    output: list[torch.Tensor],
    target: list[torch.Tensor],
    weights: list[float],
    loss_function: torch.nn.Module,
    mask: torch.Tensor | None,
    patch_shape: list[int] | None,
    project: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, int]:
    """Weighted per-layer feature distance between two preprocessed inputs, tiled and masked.

    ``output`` / ``target`` are ``[tensor, nb_layer, stats]`` triples as fed to an IMPACT TorchScript
    extractor. A patch without a mask voxel is skipped; a layer whose resampled mask vanishes, or whose
    loss is NaN, contributes nothing. Returns the summed loss and the number of scored patches: the
    caller divides.
    """
    loss = torch.zeros(1, device=output[0].device, requires_grad=True)
    true_nb = 0
    for output_patch, target_patch, mask_patch in _patch_views(output[0], target[0], mask, patch_shape):
        if mask_patch is not None and not torch.any(mask_patch == 1):
            continue
        for weight, output_feature, target_feature in zip(
            weights, model(output_patch, *output[1:]), model(target_patch, *target[1:]), strict=False
        ):
            if weight == 0:
                continue
            if project is not None:
                output_feature, target_feature = project(output_feature, target_feature)
            if mask_patch is not None:
                selection = _feature_mask(mask_patch, output_feature)
                if not torch.any(selection):
                    continue
                output_feature = torch.masked_select(output_feature, selection)
                target_feature = torch.masked_select(target_feature, selection)
            layer_loss = weight * loss_function(output_feature.float(), target_feature.float())
            if not layer_loss.isnan():
                loss = loss + layer_loss
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
        _check_feature_model(model_path, in_channels, tile or [224] * len(shape), len(weights))
        return cls(model_path, in_channels, weights, tile, len(shape), denormalize)

    def inputs(self, tensor: torch.Tensor, attributes: list[Attribute]) -> list[torch.Tensor]:
        """The ``[tensor, nb_layer, stats]`` triple the extractor takes: one
        ``[ImageMin, ImageMean, ImageMax, ImageStd]`` row per sample."""
        if tensor.shape[1] != self.in_channels:
            tensor = tensor.repeat(1, self.in_channels, *([1] * (tensor.dim() - 2)))
        if self.denormalize:
            tensor = _denormalized(tensor, attributes)
        stats = [[float(a[key]) for key in ("ImageMin", "ImageMean", "ImageMax", "ImageStd")] for a in attributes]
        return [tensor, torch.tensor([len(self.weights)]), torch.tensor(stats)]

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
        """The weighted feature distance and the number of scored patches: per 2-D slice of a 3-D batch
        when the extractor is 2-D, once for the whole batch otherwise."""
        if self.model is None:
            self.model = torch.jit.load(self.model_path, map_location="cpu").eval()  # nosec B614
        self.model.to(output.device)
        for z in range(output.shape[2]) if output.dim() == 5 and self.dim == 2 else (slice(None),):
            yield _masked_feature_loss(
                self.model,
                self.inputs(output[:, :, z], output_attributes),
                self.inputs(target[:, :, z], target_attributes),
                self.weights,
                loss_function,
                mask[:, :, z] if mask is not None else None,
                self.shape,
                project,
            )


class IMPACTReg(CriterionWithAttribute):
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
        self.name = name
        loss_module, loss_class = get_module(loss, "konfai.metric.measure")
        self.loss = apply_config(os.environ["KONFAI_CONFIG_PATH"])(getattr(loss_module, loss_class))()
        self.pca = int(pca)
        self.model = ImpactFeatureModel.download(model_name, in_channels, weights, shape)

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
        basis is fitted on the TARGET (reference) features and reused for the output: a channel-covariance
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

    def forward(  # type: ignore[override]  # the added keyword is CriterionWithAttribute's contract
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Attribute]]
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        mask = _sniffed_mask(targets, targets[-1])
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
