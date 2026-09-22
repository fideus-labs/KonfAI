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
#
# This wrapper does NOT copy any FireANTs source: it only calls the public FireANTs API of the
# separately-installed ``fireants`` wheel (PyPI). FireANTs is distributed under the FireANTs License
# v1.0 and must be cited: see the NOTICE file in this directory for the license, copyright and
# bibliography that ship with this app.

"""FireANTs registration as a self-contained KonfAI model (shared by the FireANTs presets).

Same idiomatic ``add_module`` graph and the same output contract as the ConvexAdam preset
(``DisplacementField`` on the FIXED grid), so the
orchestrator / app.json / ensemble / uncertainty are unchanged. The engine chains FireANTs' own
composable stages (GPU, Riemannian Adam), each seeding the next like ANTs' ``-t`` stages:

    Rigid (MI, centre-of-mass init) -> Affine (MI, seeded by the rigid) -> deformable

Two mirrored knobs specialise this shared module into the different presets (as ConvexAdam's shared
module is specialised by ``stages``). ``deformable_method`` picks the deformable stage:

    "syn"    symmetric diffeomorphic SyN (CC) (invertible, higher quality, averages cleanly for ensembling
    "greedy" greedy diffeomorphic (CC)) one-directional, faster / lower VRAM
    "none"   no deformable: the linear stage IS the transform (FireANTs_Affine)

and ``linear_method`` picks the linear one:

    "rigid_affine"  Rigid then Affine, as ANTs (the default
    "rigid"         Rigid only) no free scale or shear
    "none"          deformable from identity, for a pair already globally aligned

They compose, so ``deformable_method="none"`` runs whichever linear stage was asked for rather than
always Rigid+Affine: with ``linear_method="rigid"`` it is a rigid-only registration. The one
combination refused is both set to ``"none"``: that leaves nothing to optimise, and the engine
raises ``ValueError`` at build time rather than returning an identity transform.

Masks: the optional Fixed/Moving masks restrict the metric to a region. FireANTs implements this by
carrying the mask as the last image channel and prefixing the metric with ``masked_``; a mask is only
honoured when it actually restricts (some voxels in, some out), so the common mask-free path is
unchanged (an absent optional mask arrives as a whole-image default and is treated as no mask).

The deformable stages produce the single TOTAL displacement field on the fixed grid (the linear
pre-align is baked in via ``init_affine``, ANTs convention); ``none`` uses the affine matrix directly.
the emitted ``DisplacementField`` is rebuilt from that transform with SimpleITK (the same output path as the
ConvexAdam engine), so all presets/engines are interchangeable in an
ensemble. FireANTs' output-transform writer only serialises to a file, so the deformable field is
round-tripped through a temporary NIfTI (no FireANTs internals are reimplemented here).

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on
runtime-evaluated annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import contextlib
import gc
import json
import os
import tempfile
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import SimpleITK as sitk
import torch
from konfai.metric.measure import ImpactFeatureModel, IMPACTReg
from konfai.network import network
from konfai.utils.config import Choices, Range
from konfai.utils.dataset import Attribute, data_to_image, image_to_data

from .elastix import _is_local_ref

DIM = 3

#: Linear stages a preset may ask for. The engine checks against this rather than trusting the
#: ``Literal`` annotation, which only binds a config-driven call and not a direct Python one.
_LINEAR_METHODS = ("rigid_affine", "rigid", "none")

# Feature-model registry (models.json): the available IMPACT feature models, fetched from HF (NOT bundled).
# Only consulted by the "impact" deformable metric; ``KONFAI_IMPACT_MODELS_REGISTRY`` (a local path) wins
# for dev/offline. Mirrors the ConvexAdam preset so the same 30-model catalogue and picker are shared.
_IMPACT_MODELS_REGISTRY = "VBoussot/impact-torchscript-models:models.json"

# Feature distances, mirroring the itk-impact C++ metric (ITKIMPACT ImpactLoss.h) so FireANTs offers the same
# set as the ConvexAdam / elastix presets. The channel axis is dim 1 (features are [B, C, *spatial]). itk-impact
# computes gradients analytically; FireANTs optimises by autograd, so each loss is the plain differentiable
# value, for Dice this means the SOFT overlap (the C++ rounds activations to {0, 1} and cannot be autograd'd).
_EPS = 1e-6


class _CosineDistance(torch.nn.Module):
    """Per-voxel cosine distance over channels: minimise ``-cos`` (itk-impact ``Cosine``)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cosine = (x * y).sum(1) / (x.norm(2, 1) * y.norm(2, 1) + _EPS)
        return -cosine.mean()


class _SoftDiceDistance(torch.nn.Module):
    """Soft (differentiable) Dice over channels: ``1 - dice`` on clamped activations (itk-impact ``Dice`` rounds
    to {0, 1} and uses an explicit gradient; autograd needs the round dropped)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=0.0)
        y = y.clamp(min=0.0)
        intersection = (x * y).sum(1)
        union = (x + y).sum(1)
        return 1.0 - ((2 * intersection + _EPS) / (union + _EPS)).mean()


class _NCCDistance(torch.nn.Module):
    """Per-channel normalised cross-correlation across all voxels: minimise ``-NCC`` (itk-impact ``NCC``)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        xf = x.transpose(0, 1).reshape(channels, -1)
        yf = y.transpose(0, 1).reshape(channels, -1)
        xf = xf - xf.mean(1, keepdim=True)
        yf = yf - yf.mean(1, keepdim=True)
        ncc = (xf * yf).sum(1) / (torch.sqrt(xf.pow(2).sum(1) * yf.pow(2).sum(1)) + _EPS)
        return -ncc.mean()


_DISTANCES: dict[str, type[torch.nn.Module]] = {
    "L1": torch.nn.L1Loss,
    "L2": torch.nn.MSELoss,
    "Dice": _SoftDiceDistance,
    "Cosine": _CosineDistance,
    "NCC": _NCCDistance,
}


def _fireants_git_ref() -> str:
    """Best-effort FireANTs git ref whose ``fused_ops`` matches the installed ``fireants``.

    Overridable with ``FIREANTS_FUSED_OPS_REF``; falls back to ``main`` if the version is unknown.
    """
    try:
        import importlib.metadata

        version = importlib.metadata.version("fireants").strip()
        if version:
            return version if version.startswith("v") else f"v{version}"
    except Exception:
        pass
    return "main"


def ensure_fireants_runtime(build_kernels: bool = True) -> None:
    """Make the ``fireants`` runtime importable before a preset uses it: best-effort, never fatal.

    A plain ``pip install fireants`` fails inside a host like 3D Slicer for two reasons, both handled
    here with a clear one-line status at each step:

    1. **fireants won't install.** It pins ``simpleitk==2.2.1`` (no wheel on modern Python), while the
       host already ships a newer SimpleITK. We install it with ``--no-deps`` so that pin is ignored
       and the host's SimpleITK/torch are reused; its light deps ship in ``requirements.txt``.
    2. **The fused CUDA kernels** (``fireants_fused_ops``) that make registration fast and
       memory-light are OPTIONAL. Without them fireants runs in pure PyTorch (correct, only slower).
       We enable them only when a CUDA compiler (``nvcc``) is present, compiling from the upstream
       FireANTs source at install time: nothing is vendored into this app. No compiler, or a failed
       build, simply falls back to pure PyTorch.
    """
    import importlib
    import shutil
    import subprocess
    import sys

    def _log(message: str) -> None:
        print(f"[FireANTs] {message}", flush=True)

    # 1) fireants itself: install --no-deps to sidestep its unsatisfiable ``simpleitk==2.2.1`` pin.
    try:
        importlib.import_module("fireants")
    except Exception:
        _log("installing fireants (--no-deps, reusing the host's SimpleITK/torch)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fireants"])
            importlib.invalidate_caches()
            importlib.import_module("fireants")
            _log("fireants installed.")
        except Exception as exc:
            raise RuntimeError(
                "Could not install 'fireants'. Install it manually with:\n"
                f"    {sys.executable} -m pip install --no-deps fireants\n"
                "(its hydra-core/nibabel/pandas dependencies ship in this app's requirements.txt).\n"
                f"Original error: {exc}"
            ) from exc

    if not build_kernels or os.environ.get("FIREANTS_SKIP_FUSED_OPS", "").strip().lower() in {"1", "true", "yes"}:
        _log("skipping the fused CUDA kernels -> pure PyTorch (correct, slower and more memory).")
        return

    # 2) fused CUDA kernels: optional accelerator.
    try:
        importlib.import_module("fireants_fused_ops")
        _log("fused CUDA kernels already available.")
        return
    except Exception:
        pass

    # 2a) a prebuilt wheel matching this torch's CUDA build, if one is ever published.
    cuda_tag = ""
    try:
        cuda_tag = (torch.version.cuda or "").replace(".", "")
    except Exception:
        pass
    if cuda_tag:
        wheel = f"fireants-fused-ops-cu{cuda_tag}"
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", wheel])
            importlib.invalidate_caches()
            importlib.import_module("fireants_fused_ops")
            _log(f"installed prebuilt CUDA kernels ({wheel}).")
            return
        except Exception:
            pass  # no matching wheel -> try a local build

    # 2b) local build is OPT-IN: compiling CUDA kernels is heavy and can exhaust RAM on the user's
    # machine, so it NEVER runs by default. Set FIREANTS_BUILD_KERNELS=1 to enable it (devs with a
    # CUDA toolkit); it then builds ONE file at a time (MAX_JOBS=1) to keep memory bounded. The clean
    # path for end users is a prebuilt wheel (2a): a local compile also needs Python dev headers
    # (absent from some embedded Pythons, e.g. Slicer) and a CUDA-compatible host compiler.
    if os.environ.get("FIREANTS_BUILD_KERNELS", "").strip().lower() not in ("1", "true", "yes"):
        _log(
            "no prebuilt kernels for this platform -> running FireANTs in pure PyTorch (correct, slower "
            "and heavier). Set FIREANTS_BUILD_KERNELS=1 to compile them locally (needs a CUDA toolkit)."
        )
        return
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc") or (os.path.join(cuda_home, "bin", "nvcc") if cuda_home else None)
    if not nvcc or not os.path.exists(nvcc) or shutil.which("git") is None:
        _log("FIREANTS_BUILD_KERNELS set but no CUDA toolkit (nvcc) / git found -> pure PyTorch.")
        return

    ref = os.environ.get("FIREANTS_FUSED_OPS_REF") or _fireants_git_ref()
    _log(f"nvcc found ({nvcc}); compiling the FireANTs CUDA kernels one file at a time (ref '{ref}')...")
    import tempfile

    env = os.environ.copy()
    env["MAX_JOBS"] = "1"  # one compile at a time -> bounded RAM (prevents OOM on large hosts)
    env.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")  # tolerate a newer host compiler
    tmp = tempfile.mkdtemp(prefix="fireants_fused_ops_")
    try:
        # Shallow clone WITHOUT --recursive: FireANTs' submodules (an SSH-only 'cookbook') are unrelated
        # to fused_ops and would otherwise abort the build with a public-key/permission error.
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--branch", ref, "https://github.com/rohitrango/FireANTs.git", tmp]
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", os.path.join(tmp, "fused_ops")],
            env=env,
        )
        importlib.invalidate_caches()
        importlib.import_module("fireants_fused_ops")
        _log("compiled and installed the fast CUDA kernels.")
    except Exception as exc:
        _log(
            "kernel build failed -> running in pure PyTorch (correct, only speed/memory affected). Set "
            "FIREANTS_FUSED_OPS_REF to a compatible FireANTs tag, or install a prebuilt fireants-fused-ops "
            f"wheel, to enable them. Details: {exc}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def registry_choices() -> list[str]:
    """The per-model ``ref`` picker's values: model refs (``repo:path``) from the feature-model registry."""
    repo = _IMPACT_MODELS_REGISTRY.split(":", 1)[0]
    return [f"{repo}:{key}" for key in load_models_registry()]


def load_models_registry(ref: str = _IMPACT_MODELS_REGISTRY) -> dict:
    """Load ``models.json`` (available feature models). ``KONFAI_IMPACT_MODELS_REGISTRY`` (local path) wins
    for dev/offline; otherwise ``ref`` is a ``repo:file`` Hugging Face reference (fetched, not bundled)."""
    from huggingface_hub import hf_hub_download

    local = os.environ.get("KONFAI_IMPACT_MODELS_REGISTRY", "")
    if local:
        path = Path(local)
    elif ":" in ref:
        repo, filename = ref.split(":", 1)
        path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="model"))  # nosec B615
    else:
        raise ValueError(
            f"models_registry '{ref}' must be a 'repo:file' Hugging Face reference: or set "
            "KONFAI_IMPACT_MODELS_REGISTRY to a local file for offline use."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_specs(mapping: dict) -> list:
    """A dict keyed by string indices ('0','1',...) -> its values in numeric order."""
    return [mapping[k] for k in sorted(mapping, key=lambda key: int(key))]


@dataclass
class ModelSpec:
    """One IMPACT feature model in the deformable metric (several are fused). ``ref`` picks the model; the
    rest are its per-model knobs: the same as the ConvexAdam / elastix ``ModelSpec`` except ``voxel_size``
    (an itk-impact resampling knob) has no meaning for FireANTs' geometry-free torch ``custom_loss`` and is
    intentionally absent."""

    ref: Annotated[
        str,
        Choices(registry_choices),
        "IMPACT feature model driving the 'impact' deformable metric (TorchScript 'repo:file' on Hugging Face); "
        "different models capture different anatomy/contrast. Suggested priors (from the IMPACT study, not "
        "forced): TotalSegmentator (TS/M730) is the general default; a model trained on the target structure "
        "(e.g. lung or vessels) sharpens local alignment there; add MIND for MR/CT to recover intra-organ detail.",
    ]
    layers_mask: Annotated[
        str,
        "Per-layer on/off bitmask over the feature model's layers ('1' = use, '0' = skip), one char per layer; "
        "selects which feature depths drive the metric. Suggested priors (not forced): CT/CBCT favours EARLY "
        "layers (they denoise and enhance anatomical structures across modalities, robust to artifacts); MR/CT "
        "favours HIGH-LEVEL layers (contour/segmentation-driven alignment).",
    ] = "01"
    layers_weight: Annotated[
        float, "Relative weight of this feature model in the multi-model fusion (all models are compared jointly)."
    ] = 1.0
    pca: Annotated[
        int,
        Range(0, 100),
        "Number of PCA components the feature channels are reduced to before matching (0 = keep all); "
        "trims redundant/noisy channels and cost.",
    ] = 0
    distance: Annotated[
        Literal["L1", "L2", "Dice", "Cosine", "NCC"],
        "Per-feature distance combined into the IMPACT similarity (Dice is the differentiable soft-Dice). "
        "Suggested prior (not forced): when the task is scored on Dice, choosing 'Dice' aligns the loss with "
        "the metric.",
    ] = "L1"


@contextlib.contextmanager
def _no_texpr_fuser():
    """Disable the TensorExpr JIT fuser while IMPACT's TorchScript feature model runs under autograd.

    The IMPACT feature models are TorchScript; run under FireANTs' gradient optimisation the TensorExpr
    fuser trips on shape ops (``aten::size`` INTERNAL ASSERT). Scoped and restored so no other torch/JIT
    user is affected; the modern profiling executor stays on (this is NOT the legacy executor).
    """
    prev = torch._C._jit_texpr_fuser_enabled()
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_fuser_enabled(prev)


class _ImpactCore(IMPACTReg):
    """One IMPACT feature model, exposed as a FireANTs ``forward(moved, fixed)``.

    Reuses KonfAI's ``ImpactFeatureModel`` verbatim (the stats-normalised feature extraction (the model
    wants per-image ``[min, mean, max, std]``) and the per-layer weighted distance) and ``IMPACTReg``'s
    PCA reduction, so the metric is exactly KonfAI's, not a re-derivation. Only KonfAI's config-binding
    ``__init__`` and its ``Attribute``-based geometry are replaced: FireANTs passes raw tensors at the
    current pyramid scale, so the intensity statistics are computed from those tensors directly.
    """

    def __init__(self, ref: str, in_channels: int, weights: list[float], distance: str, pca: int) -> None:
        from huggingface_hub import hf_hub_download

        torch.nn.Module.__init__(self)  # bypass IMPACTReg.__init__ (KONFAI_CONFIG_PATH / apply_config binding)
        self.name = "Reg"
        self.loss = _DISTANCES[distance]()
        self.pca = int(pca)
        if _is_local_ref(ref):  # otherwise a "repo:path" HF reference
            model_path = ref
        else:
            repo, filename = ref.split(":", 1)
            model_path = hf_hub_download(repo, filename, repo_type="model")  # nosec B615
        # shape=None: the whole (downsampled) tensor is scored, no ModelPatch tiling.
        self.model = ImpactFeatureModel(model_path, int(in_channels), [float(w) for w in weights], None, DIM)

    def pca_project(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """IMPACTReg's own PCA reduction, for the static path: the basis is fitted on ``target``."""
        return self._pca_project(output, target)

    @staticmethod
    def _stats(tensor: torch.Tensor) -> dict:
        detached = tensor.detach()
        return {
            "ImageMin": float(detached.min()),
            "ImageMean": float(detached.mean()),
            "ImageMax": float(detached.max()),
            "ImageStd": float(detached.std()),
        }

    def forward(  # type: ignore[override]
        self, moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with _no_texpr_fuser():
            losses, counts = zip(
                *self.model.slice_losses(
                    moved,
                    [self._stats(moved)],
                    fixed,
                    [self._stats(fixed)],
                    mask,
                    self.loss,
                    project=self._pca_project if self.pca > 0 else None,
                ),
                strict=True,
            )
        return reduce(torch.add, losses) / max(sum(counts), 1)


class ImpactFeatureLoss(torch.nn.Module):
    """FireANTs ``custom_loss`` = the KonfAI IMPACT metric fused over several feature models.

    ``forward(moved, fixed)`` sums each model's ``layers_weight * IMPACT(model)``. A model's per-layer
    weights come from its ``layers_mask`` bitmask; its input channel count is read from the registry
    (``models.json`` ``numberofchannels``) so it never has to be configured by hand.

    ``masked`` mirrors the engine's own decision to run FireANTs' masked mode: the images then carry
    the mask as one extra trailing channel (``apply_mask_to_image``), which nothing about the tensors
    themselves announces. The feature models want the image alone, so the channel is split off once
    and handed to KonfAI's masked feature loss (nearest-resampled onto every feature layer): the
    metric is evaluated inside the fixed mask, which is what an elastix/ITK mask means too.
    """

    def __init__(self, specs: list["ModelSpec"], masked: bool = False) -> None:
        super().__init__()
        registry = load_models_registry()
        self._cores = torch.nn.ModuleList()
        self._model_weights: list[float] = []
        self._masked = masked
        for spec in specs:
            in_channels = int(registry.get(spec.ref.split(":", 1)[-1], {}).get("numberofchannels", 1))
            weights = [1.0 if char == "1" else 0.0 for char in spec.layers_mask]
            self._cores.append(_ImpactCore(spec.ref, in_channels, weights, spec.distance, spec.pca))
            self._model_weights.append(float(spec.layers_weight))

    @property
    def cores(self) -> list["_ImpactCore"]:
        """The per-model feature cores, for the static path, which extracts instead of scoring."""
        return [cast("_ImpactCore", core) for core in self._cores]

    @property
    def model_weights(self) -> list[float]:
        """Each model's weight in the fusion, applied to its features in the static path."""
        return self._model_weights

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor | None = None
        if self._masked:
            mask = (fixed[:, -1:] > 0.5).to(torch.uint8)
            moved, fixed = moved[:, :-1], fixed[:, :-1]
        total: torch.Tensor | None = None
        for weight, core in zip(self._model_weights, self._cores, strict=True):
            term = weight * core(moved, fixed, mask)
            total = term if total is None else total + term
        return total


class _FeatureCC(torch.nn.Module):
    """Local cross-correlation over feature channels, a few channels at a time.

    The static path hands FireANTs volumes of features, and comparing all their channels at once is what
    sets the peak: the windowed sums of a 28-channel pair at full resolution are several times the volume
    itself. This evaluates ``chunk`` channels per pass and re-runs each pass during the backward instead
    of keeping its intermediates, so the peak follows ``chunk`` rather than the channel count, for the
    same objective and the same gradient.

    The correlation itself is the usual windowed one, ``cov(a, b)^2 / (var(a) var(b))`` over a cube of
    ``kernel`` voxels, averaged over channels and voxels and negated, since FireANTs minimises.
    """

    def __init__(self, kernel: int, chunk: int, masked: bool = False) -> None:
        super().__init__()
        self._kernel = int(kernel)
        self._chunk = max(1, int(chunk))
        self._masked = masked

    def _windowed(self, moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        kernel, padding = self._kernel, self._kernel // 2
        mean = lambda t: torch.nn.functional.avg_pool3d(t, kernel, stride=1, padding=padding, count_include_pad=False)  # noqa: E731
        moved_mean, fixed_mean = mean(moved), mean(fixed)
        covariance = mean(moved * fixed) - moved_mean * fixed_mean
        moved_var = (mean(moved * moved) - moved_mean * moved_mean).clamp_min(1e-5)
        fixed_var = (mean(fixed * fixed) - fixed_mean * fixed_mean).clamp_min(1e-5)
        correlation = covariance * covariance / (moved_var * fixed_var)
        if mask is not None:
            return -(correlation * mask).sum() / mask.sum().clamp_min(1.0) / correlation.shape[1]
        return -correlation.mean()

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor | None = None
        if self._masked:
            mask = (fixed[:, -1:] > 0.5).to(moved.dtype)
            moved, fixed = moved[:, :-1], fixed[:, :-1]
        channels = moved.shape[1]
        total: torch.Tensor | None = None
        for start in range(0, channels, self._chunk):
            stop = min(start + self._chunk, channels)
            piece = torch.utils.checkpoint.checkpoint(
                self._windowed, moved[:, start:stop], fixed[:, start:stop], mask, use_reentrant=False
            )
            weighted = piece * (stop - start) / channels
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("the feature volumes carry no channel to correlate")
        return total


@torch.no_grad()
def _one_volume(core: "_ImpactCore", weight: float, image: torch.Tensor, patch: int, overlap: float) -> torch.Tensor:
    """One model's selected feature layers for one image, tiled and blended.

    The intensity statistics come from the WHOLE image, never from a tile, so every tile is normalised
    identically -- a per-tile normalisation would make the same anatomy score differently on either side
    of a seam. Tiles share ``overlap`` of their width and cross-fade through a cosine window.
    """
    model = core.model
    if model.model is None:
        model.model = torch.jit.load(model.model_path, map_location="cpu").eval()  # nosec B614
    model.model.to(image.device)
    stats = torch.tensor(
        [float(image.min()), float(image.max()), float(image.mean()), float(image.std())], device=image.device
    )
    nb_layers = torch.tensor([len(model.weights)], device=image.device)
    tensor = image
    if tensor.shape[1] != model.in_channels:
        tensor = tensor.repeat(1, model.in_channels, *([1] * (tensor.dim() - 2)))

    extracted: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    for window in _tiles(tuple(tensor.shape[2:]), patch, overlap):
        tile = tensor[(slice(None), slice(None), *window)]
        layers = [
            layer
            for layer_weight, layer in zip(model.weights, model.model(tile, nb_layers, stats), strict=False)
            if layer_weight != 0
        ]
        tile_features = weight * torch.nn.functional.normalize(torch.cat(layers, dim=1), dim=1)
        if extracted is None:  # the channel count is only known once a tile has been through
            shape = (tile_features.shape[0], tile_features.shape[1], *tensor.shape[2:])
            extracted = torch.zeros(shape, device=image.device, dtype=tile_features.dtype)
        if weights is None:
            weights = torch.zeros((1, 1, *tensor.shape[2:]), device=image.device, dtype=tile_features.dtype)
        blend = _cosine_window(tuple(tile_features.shape[2:]), image.device, tile_features.dtype)
        extracted[(slice(None), slice(None), *window)] += tile_features * blend
        weights[(slice(None), slice(None), *window)] += blend
        del tile_features, layers
    if extracted is None or weights is None:
        raise RuntimeError(f"no tile covered an image of shape {tuple(tensor.shape[2:])}")
    return extracted / weights.clamp_min(1e-6)


@torch.no_grad()
def _feature_volumes(
    loss: "ImpactFeatureLoss", fixed: torch.Tensor, moving: torch.Tensor, patch: int, overlap: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fixed and moving feature volumes of every model, concatenated along the channel axis.

    This is the static path: each network runs once per image here, instead of once per optimiser step
    inside the loss, and the registration then works on the feature volumes themselves. It is what makes
    a large pair affordable -- no autograd graph through an extractor is kept -- and what a whole-image
    model needs, since the features are never differentiated with respect to the warp.

    A model asking for ``pca`` has both its volumes projected onto the basis of the FIXED one, exactly as
    the online metric fits its basis on the reference side. The channel count is what the comparison then
    costs, so this is also the lever when a pair of models does not fit.
    """
    fixed_sides: list[torch.Tensor] = []
    moving_sides: list[torch.Tensor] = []
    for weight, core in zip(loss.model_weights, loss.cores, strict=True):
        fixed_features = _one_volume(core, weight, fixed, patch, overlap)
        moving_features = _one_volume(core, weight, moving, patch, overlap)
        if core.pca > 0:
            moving_features, fixed_features = core.pca_project(moving_features, fixed_features)
        fixed_sides.append(fixed_features)
        moving_sides.append(moving_features)
    return torch.cat(fixed_sides, dim=1), torch.cat(moving_sides, dim=1)


def _tiles(shape: tuple[int, ...], patch: int, overlap: float):
    """The windows the extraction runs on: the whole image when ``patch`` is 0, otherwise tiles of
    ``patch`` voxels stepping by ``patch * (1 - overlap)``, the last one flush with the far face."""
    if patch <= 0 or all(size <= patch for size in shape):
        yield tuple(slice(0, size) for size in shape)
        return
    step = max(1, round(patch * (1.0 - overlap)))
    starts = [
        sorted({*range(0, max(size - patch, 0) + 1, step), max(size - patch, 0)}) if size > patch else [0]
        for size in shape
    ]
    for first in starts[0]:
        for second in starts[1]:
            for third in starts[2]:
                yield tuple(
                    slice(start, min(start + patch, size))
                    for start, size in zip((first, second, third), shape, strict=True)
                )


def _cosine_window(shape: tuple[int, ...], device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """A separable cosine taper, 1 at the tile's centre and near 0 at its faces, so overlapping tiles
    cross-fade instead of stepping."""
    window = torch.ones((1, 1, *shape), device=device, dtype=dtype)
    for axis, size in enumerate(shape):
        taper = torch.hann_window(size + 2, periodic=False, device=device, dtype=dtype)[1:-1].clamp_min(1e-3)
        window = window * taper.reshape([1, 1] + [size if i == axis else 1 for i in range(len(shape))])
    return window


def _mask_on_grid(mask: "sitk.Image", image: "sitk.Image", device: str):
    """A FireANTs ``Image`` of ``mask`` carrying ``image``'s geometry.

    The mask is defined on the image's grid, but its header can differ by float rounding once it has
    crossed KonfAI's Attribute round-trip; FireANTs' ``concatenate`` then rejects it with a spurious
    "different physical spaces" (surfacing as a ``TypeError`` in its ``check_and_raise_cond``). Reusing
    the image's spacing/origin/direction makes the two spaces identical by construction.
    """
    from fireants.io import Image

    return Image(
        mask,
        device=device,
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
    )


class FireANTsEngine:
    """Register a fixed/moving pair with FireANTs (Rigid -> Affine -> [SyN | Greedy | none]); return
    the displacement field on the fixed grid.

    ``fireants`` is imported lazily inside :meth:`register` so this module can be imported for config
    /signature introspection (SlicerImpactReg reads the tuning knobs off the ``RegistrationNet``
    annotations) on a machine without a GPU or without FireANTs installed.
    """

    def __init__(
        self,
        scales: list[int],
        affine_iterations: list[int],
        deformable_iterations: list[int],
        cc_kernel: int,
        affine_metric: str,
        affine_lr: float,
        moments_init: str,
        linear_method: str,
        deformable_method: str,
        deformable_metric: str,
        deformable_lr: float,
        integrator_n: int,
        smooth_warp_sigma: float,
        smooth_grad_sigma: float,
        seed: int,
        impact_specs: list["ModelSpec"],
        impact_mode: str = "online",
        feature_patch: int = 0,
        feature_chunk: int = 0,
    ) -> None:
        self._scales = [int(s) for s in scales]
        self._affine_iterations = [int(i) for i in affine_iterations]
        self._deformable_iterations = [int(i) for i in deformable_iterations]
        self._cc_kernel = int(cc_kernel)
        self._affine_metric = affine_metric
        self._affine_lr = float(affine_lr)
        self._moments_init = moments_init
        self._linear_method = linear_method
        self._deformable_method = deformable_method
        # Both checks are at BUILD time, as the missing-feature-model one above is: the stages they
        # guard run for minutes, and a run that reaches them has already paid for the read.
        if linear_method not in _LINEAR_METHODS:
            # Named, not silently ignored. Every unrecognised value would otherwise fall through to
            # the rigid-then-affine branch, so a typo ("affine", say) would register with a stage
            # the caller did not ask for and produce a perfectly plausible result.
            raise ValueError(
                f"Unknown linear_method '{linear_method}' (expected {', '.join(map(repr, _LINEAR_METHODS))})."
            )
        if linear_method == "none" and deformable_method == "none":
            # Left to run this optimises nothing and returns the identity: a Moved equal to the moving
            # image and a zero field, which no downstream check tells apart from a pair that needed no
            # moving.
            raise ValueError("linear_method='none' with deformable_method='none' leaves nothing to optimise.")
        self._deformable_metric = deformable_metric
        self._deformable_lr = float(deformable_lr)
        self._integrator_n = int(integrator_n)
        self._smooth_warp_sigma = float(smooth_warp_sigma)
        self._smooth_grad_sigma = float(smooth_grad_sigma)
        self._seed = int(seed)
        # IMPACT deformable metric (only used when deformable_metric == "impact"): KonfAI IMPACT feature
        # models drive the SyN/greedy stage instead of the analytic CC/MI/MSE.
        self._impact_specs = impact_specs
        self._impact_mode = impact_mode
        self._feature_patch = int(feature_patch)
        self._feature_chunk = int(feature_chunk)
        if impact_mode not in ("online", "static"):
            raise ValueError(f"Unknown impact_mode '{impact_mode}' (expected 'online' or 'static').")

    @staticmethod
    def _center_of_mass_translation(
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: "sitk.Image | None",
        moving_mask: "sitk.Image | None",
        device: str,
    ) -> torch.Tensor:
        """Seed translation ``com_moving - com_fixed`` from intensity-weighted centres of mass.

        Computed here rather than through FireANTs' ``MomentsRegistration``: on anisotropic-spacing
        volumes that class mis-estimates the centre of mass (measured ~18-25 voxels off in Y/Z on
        ExaSPIM light-sheet data), and the wrong seed pushes the affine into an anisotropic minimum
        that looks like a metric failure and is not one.

        Each subject's centre is taken in its OWN physical space (mask-restricted when a real mask is
        given), so origin/spacing/direction differences between the two frames are handled exactly.
        Only the centroid is used: second-order moments (rotation/scale) were tried and hurt, because a
        near-symmetric subject has ambiguous principal axes.

        Returns the ``[1, 3]`` physical-space translation ``RigidRegistration(init_translation=...)``
        expects.
        """

        def com_phys(img: sitk.Image, mask: "sitk.Image | None") -> np.ndarray:
            array = sitk.GetArrayFromImage(img).astype(np.float64)  # (z, y, x)
            array = np.clip(array, 0, None)
            if mask is not None:
                array = array * (sitk.GetArrayFromImage(mask).astype(np.float64) > 0.5)
            positive = array > 0
            if not positive.any():
                # An all-zero (or all-negative) subject has no centre of mass to speak of; the frame
                # centre is the only defensible answer and matches what "cof" would have done.
                size = img.GetSize()
                return np.asarray(img.TransformContinuousIndexToPhysicalPoint([(extent - 1) / 2.0 for extent in size]))
            index = np.array(np.nonzero(positive))  # (3, N) in z, y, x
            weight = array[positive]
            centre_voxel = (index * weight).sum(axis=1) / weight.sum()  # z, y, x
            return np.asarray(
                img.TransformContinuousIndexToPhysicalPoint(
                    [float(centre_voxel[2]), float(centre_voxel[1]), float(centre_voxel[0])]
                )
            )

        translation = com_phys(moving, moving_mask) - com_phys(fixed, fixed_mask)
        return torch.tensor(translation, device=device, dtype=torch.float32).reshape(1, 3)

    @staticmethod
    def _is_partial_mask(mask: "sitk.Image | None") -> bool:
        """True only for a mask that actually restricts the region: some voxels in, some out. An absent
        optional mask arrives as a whole-image (all-ones) default and an all-zero mask is degenerate; both
        are treated as no mask so the plain (non-masked) metric path is used."""
        if mask is None:
            return False
        arr = sitk.GetArrayViewFromImage(mask)
        return bool((arr > 0).any()) and bool((arr == 0).any())

    @staticmethod
    def _affine_to_sitk(affine_matrix: "torch.Tensor") -> sitk.AffineTransform:
        """FireANTs' physical (LPS) linear matrix -> SimpleITK AffineTransform (fixed -> moving points),
        the same convention FireANTs writes into an ANTs ``0GenericAffine.mat``."""
        matrix = affine_matrix.float().cpu().numpy()[0]
        affine = sitk.AffineTransform(DIM)
        affine.SetMatrix(matrix[:DIM, :DIM].flatten().astype(np.float64))
        affine.SetTranslation(matrix[:DIM, DIM].astype(np.float64))
        return affine

    def _total_field_transform(self, reg) -> sitk.Transform:
        """Optimise a deformable stage and return its TOTAL displacement (affine baked in) as a
        SimpleITK ``DisplacementFieldTransform`` on the fixed grid.

        FireANTs serialises the total field (ANTs convention, fixed grid) only to a file, so it is
        round-tripped through a temporary NIfTI: its public API, no internals reimplemented."""
        reg.optimize()
        with tempfile.TemporaryDirectory() as tmp:
            warp_path = os.path.join(tmp, "total_warp.nii.gz")
            reg.save_as_ants_transforms(warp_path)
            total_field = sitk.ReadImage(warp_path, sitk.sitkVectorFloat64)
        return sitk.DisplacementFieldTransform(total_field)  # consumes total_field

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> np.ndarray:
        """Register ``moving`` onto ``fixed``; return the displacement field, channel-first, on the fixed grid."""
        ensure_fireants_runtime()
        from fireants.io import BatchedImages, Image
        from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.rigid import RigidRegistration

        torch.manual_seed(self._seed)
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        # FireANTs' Image ctor accepts a SimpleITK image directly, so the fixed/moving cross into
        # FireANTs in-memory (no file load) with their geometry preserved.
        fixed_img = Image(fixed, device=device)
        moving_img = Image(moving, device=device)

        # Masked metric only when a mask genuinely restricts the region. FireANTs' masked mode wants the
        # mask as the last channel of BOTH images (all-ones where one side has none) and a ``masked_``
        # metric prefix; the plain path is untouched when no real mask is present.
        use_fixed_mask = self._is_partial_mask(fixed_mask)
        use_moving_mask = self._is_partial_mask(moving_mask)
        masked = use_fixed_mask or use_moving_mask
        if masked:
            fmask = (
                _mask_on_grid(fixed_mask, fixed, device) if use_fixed_mask else generate_image_mask_allones(fixed_img)
            )
            mmask = (
                _mask_on_grid(moving_mask, moving, device)
                if use_moving_mask
                else generate_image_mask_allones(moving_img)
            )
            fixed_img = apply_mask_to_image(fixed_img, fmask)
            moving_img = apply_mask_to_image(moving_img, mmask)

        bf = BatchedImages([fixed_img])
        bm = BatchedImages([moving_img])
        affine_loss = f"masked_{self._affine_metric}" if masked else self._affine_metric
        deformable_loss = f"masked_{self._deformable_metric}" if masked else self._deformable_metric

        # Linear: Rigid(MI) -> Affine(MI, seeded by the rigid), mirroring ANTs. The affine seeds the
        # deformable stage (or is the whole transform when deformable_method == "none").
        #
        # ``linear_method`` is ``deformable_method``'s mirror. "none" skips the linear stage entirely
        # and starts the deformable from identity: for a pair already globally aligned: a tiled
        # refinement at full resolution, where each patch sees only local anatomy, so a per-patch
        # rigid has no global meaning and neighbouring patches would each estimate a different one
        # and tear the blended field at the seams.
        affine_matrix: torch.Tensor | None
        if self._linear_method == "none":
            affine_matrix = None  # the deformable stage builds its own identity init
        else:
            # The rigid's starting translation mirrors ANTs' ``-r [fixed,moving,N]``: "cof" is the
            # centre of FRAME (N=0), "com" the centre of MASS (N=1). "cof" aligns the image frames,
            # not the subjects, so a subject sitting off its frame centre starts the chain misplaced.
            init_translation: str | torch.Tensor = "cof"
            if self._moments_init == "none":
                # Identity, not a seed: the pair arrives centred and the rigid starts where it is.
                init_translation = torch.zeros((1, 3), device=device, dtype=torch.float32)
            elif self._moments_init == "com":
                init_translation = self._center_of_mass_translation(
                    fixed,
                    moving,
                    fixed_mask if use_fixed_mask else None,
                    moving_mask if use_moving_mask else None,
                    device,
                )
            rigid = RigidRegistration(
                scales=self._scales,
                iterations=self._affine_iterations,
                fixed_images=bf,
                moving_images=bm,
                loss_type=affine_loss,
                optimizer="Adam",
                optimizer_lr=self._affine_lr,
                cc_kernel_size=self._cc_kernel,
                init_translation=init_translation,
            )
            rigid.optimize()
            rigid_matrix = rigid.get_rigid_matrix().detach()

            if self._linear_method == "rigid":
                # The rigid (rotation and translation, no scale or shear) IS the linear transform.
                # For inspecting the linear stage without the affine's free scale, and for a pair whose
                # scale difference is known to be nothing the affine should be free to invent.
                affine_matrix = rigid_matrix
            else:
                affine = AffineRegistration(
                    scales=self._scales,
                    iterations=self._affine_iterations,
                    fixed_images=bf,
                    moving_images=bm,
                    loss_type=affine_loss,
                    optimizer="Adam",
                    optimizer_lr=self._affine_lr,
                    cc_kernel_size=self._cc_kernel,
                    init_rigid=rigid_matrix,
                )
                affine.optimize()
                affine_matrix = affine.get_affine_matrix().detach()

        # Deformable stage (or none). SyN and Greedy share the same constructor surface; both warm-start
        # from the affine so their TOTAL transform already bakes in the linear pre-align.
        if self._deformable_method == "none":
            transform: sitk.Transform = self._affine_to_sitk(affine_matrix)
        else:
            if self._deformable_method == "syn":
                from fireants.registration.syn import SyNRegistration as Deformable
            elif self._deformable_method == "greedy":
                from fireants.registration.greedy import GreedyRegistration as Deformable
            else:
                raise ValueError(
                    f"Unknown deformable_method '{self._deformable_method}' (expected 'syn', 'greedy' or 'none')."
                )
            # "impact" swaps the analytic metric for a KonfAI IMPACT feature loss on the deformable stage
            # (the linear pre-align keeps its own affine_metric); the fixed mask restricts it too.
            loss_type: str = deformable_loss
            custom_loss: torch.nn.Module | None = None
            if self._deformable_metric == "impact" and self._impact_mode == "static":
                # Static: extract once, then register the feature volumes with FireANTs' own metric. The
                # images are re-read unmasked because the masked pair carries the mask as a channel, and
                # the mask is concatenated back afterwards so ``masked_`` still means what it says.
                extractor = ImpactFeatureLoss(self._impact_specs).to(device)
                volumes = _feature_volumes(
                    extractor,
                    Image(fixed, device=device).array,
                    Image(moving, device=device).array,
                    self._feature_patch,
                )
                for image, features in ((fixed_img, volumes[0]), (moving_img, volumes[1])):
                    if masked:
                        features = torch.cat([features, image.array[:, -1:]], dim=1)
                    image.array = features
                    image.channels = features.shape[1]
                del extractor
                gc.collect()
                bf, bm = BatchedImages([fixed_img]), BatchedImages([moving_img])
                # The channels ARE the features now, so a local cross-correlation compares them
                # (``cc_kernel`` sets its window): "impact" names where the channels came from, not a
                # metric FireANTs knows. With ``feature_chunk`` the correlation runs a few channels at a
                # time, which is what lets two feature models share one card.
                if self._feature_chunk > 0:
                    loss_type = "custom"
                    custom_loss = _FeatureCC(self._cc_kernel, self._feature_chunk, masked=masked)
                else:
                    loss_type = "masked_cc" if masked else "cc"
            elif self._deformable_metric == "impact":
                loss_type = "custom"
                custom_loss = ImpactFeatureLoss(self._impact_specs, masked=masked)
            reg = Deformable(
                scales=self._scales,
                iterations=self._deformable_iterations,
                fixed_images=bf,
                moving_images=bm,
                loss_type=loss_type,
                custom_loss=custom_loss,
                cc_kernel_size=self._cc_kernel,
                deformation_type="compositive",
                integrator_n=self._integrator_n,
                smooth_warp_sigma=self._smooth_warp_sigma,
                smooth_grad_sigma=self._smooth_grad_sigma,
                optimizer="Adam",
                optimizer_lr=self._deformable_lr,
                init_affine=affine_matrix,
            )
            transform = self._total_field_transform(reg)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # The DVF is rebuilt from the single transform on the fixed grid: the ConvexAdam output path,
        # so every FireANTs preset emits identical-shaped results.
        dvf = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            fixed.GetSize(),
            fixed.GetOrigin(),
            fixed.GetSpacing(),
            fixed.GetDirection(),
        )
        dvf_np, _ = image_to_data(dvf)
        return dvf_np


class FireANTsRegistration(torch.nn.Module):
    """Graph module: (fixed, moving) tensors + their geometry -> moved image + DVF on the fixed grid.

    ``accepts_attributes = True`` opts this module into receiving the per-branch ``Attribute`` list
    alongside the tensors (same convention as the ConvexAdam / elastix engines); registration needs the
    physical geometry, and the mask branches restrict the metric.
    """

    accepts_attributes = True

    def __init__(self, engine: FireANTsEngine) -> None:
        super().__init__()
        self._engine = engine

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        attributes: list[list[Attribute]],
    ) -> torch.Tensor:
        # attributes = [fixed, moving, fixed_mask, moving_mask] branch attrs; each a list[Attribute] over
        # the batch. Returns, per sample, the moved image (1 channel) channel-stacked with the
        # displacement field (DIM channels); downstream ChannelSelect modules split them. A whole-image
        # mask (the default when none is supplied) restricts nothing.
        fixed_attrs, moving_attrs, fmask_attrs, mmask_attrs = attributes
        device_index = fixed.device.index if fixed.device.type == "cuda" else -1
        combined = []
        # FireANTs runs a gradient-based instance optimisation (Riemannian Adam over the warp); the
        # predictor calls forward under torch.inference_mode(), which forbids autograd. The image tensors
        # have already crossed to numpy/SimpleITK here, so re-enable grad for the optimisation.
        with torch.inference_mode(False), torch.enable_grad():
            for b in range(fixed.shape[0]):
                fixed_img = data_to_image(fixed[b].detach().cpu().numpy(), fixed_attrs[b])
                moving_img = data_to_image(moving[b].detach().cpu().numpy(), moving_attrs[b])
                fixed_mask_img = data_to_image(fixed_mask[b].detach().cpu().numpy(), fmask_attrs[b])
                moving_mask_img = data_to_image(moving_mask[b].detach().cpu().numpy(), mmask_attrs[b])
                try:
                    dvf_np = self._engine.register(fixed_img, moving_img, device_index, fixed_mask_img, moving_mask_img)
                finally:
                    # FireANTs' registration objects keep their CUDA tensors in reference cycles that only the cyclic
                    # collector frees; it runs on Python allocation counts, not device memory, so without a collection
                    # per call several tiles' worth of fields stay allocated and a tiled run runs out of the card. A
                    # failed call collects too: an OOM restart retries with a smaller patch on the same card.
                    gc.collect()
                combined.append(torch.from_numpy(dvf_np))
        return torch.stack(combined, dim=0).to(fixed.device)


class ChannelSelect(torch.nn.Module):
    """Select a channel slice ``[start:stop]`` (splits the registration output into moved / DVF)."""

    def __init__(self, start: int, stop: int) -> None:
        super().__init__()
        self._start = start
        self._stop = stop

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self._start : self._stop]


class RegistrationNet(network.Network):
    """Pairwise FireANTs registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1,
    fixed mask = 2, moving mask = 3; masks restrict the metric, whole-image = no restriction).

    Output on the fixed grid: ``DisplacementField``
    (the DIM-component displacement field, in mm). Geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``. The knobs below are read straight from these annotations by the
    UI: ``Annotated[.., Range]`` gives numeric spin bounds; ``Literal`` a dropdown. ``deformable_method``
    is the knob that specialises this shared model into each FireANTs preset.
    """

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        scales: Annotated[
            list[int],
            "Multi-resolution pyramid: downsampling factor per level, coarse to fine (e.g. [4,2,1]); the "
            "affine/deformable iteration lists are indexed by these levels.",
        ] = [4, 2, 1],
        affine_iterations: Annotated[
            list[int], "Affine-stage iterations per pyramid level (one entry per 'scales' level)."
        ] = [200, 100, 50],
        deformable_iterations: Annotated[
            list[int], "Deformable-stage iterations per pyramid level (one entry per 'scales' level)."
        ] = [200, 100, 50],
        cc_kernel: Annotated[
            int,
            Range(1, 21),
            "Radius (voxels) of the local cross-correlation window when a 'cc' metric is used; larger = more "
            "spatial context, slower.",
        ] = 5,
        affine_metric: Annotated[
            Literal["mi", "cc", "mse"], "Similarity metric optimised during the affine (global) stage."
        ] = "mi",
        affine_lr: Annotated[
            float,
            Range(0.0, 10.0),
            "Gradient step size of the affine optimisation; higher converges faster but risks overshoot.",
        ] = 0.003,
        moments_init: Annotated[
            Literal["cof", "com", "none"],
            "Initial translation seeding the rigid stage, mirroring ANTs' -r [fixed,moving,N]: 'cof' = centre "
            "of frame (N=0), 'com' = intensity-weighted centre of mass (N=1). Prefer 'com' when the subject "
            "sits off its frame centre, where aligning frames starts the chain misplaced. 'none' seeds "
            "nothing and starts the rigid from identity: for a pair the caller has already centred, where "
            "'com' re-estimates a translation that is zero and 'cof' undoes the centring by aligning "
            "frames instead of subjects.",
        ] = "cof",
        linear_method: Annotated[
            Literal["rigid_affine", "rigid", "none"],
            "Linear stage before the deformable: 'rigid_affine' (rigid then affine, as ANTs), 'rigid' to stop "
            "after the rigid and leave scale and shear alone, or 'none' to skip it and start the deformable "
            "from identity. Use 'none' only on a pair already globally aligned: a tiled pass at full "
            "resolution, where a patch sees no global context and a per-patch linear has no global meaning.",
        ] = "rigid_affine",
        deformable_method: Annotated[
            Literal["none", "syn", "greedy"],
            "Deformable algorithm: 'syn' (symmetric diffeomorphic), 'greedy', or 'none' to stop after the linear "
            "stage, whichever 'linear_method' selected, with linear_method='rigid' that is a rigid-only "
            "registration. Both set to 'none' is refused: it would optimise nothing.",
        ] = "syn",
        deformable_metric: Annotated[
            Literal["cc", "mi", "mse", "impact"],
            "Similarity metric for the deformable stage; 'impact' uses the IMPACT feature models under 'models'.",
        ] = "cc",
        deformable_lr: Annotated[float, Range(0.0, 10.0), "Gradient step size of the deformable optimisation."] = 0.25,
        integrator_n: Annotated[
            int,
            Range(1, 100),
            "Velocity-field integration steps for the diffeomorphic (SyN) update; higher = more accurate "
            "integration and invertibility, slower.",
        ] = 10,
        smooth_warp_sigma: Annotated[
            float,
            Range(0.0, 100.0),
            "Gaussian sigma (voxels) smoothing the displacement/warp field each step; higher = smoother, more "
            "regular deformation.",
        ] = 0.5,
        smooth_grad_sigma: Annotated[
            float,
            Range(0.0, 100.0),
            "Gaussian sigma (voxels) smoothing the update gradient each step; higher = more stable but slower "
            "convergence.",
        ] = 1.0,
        seed: Annotated[int, "Random seed for the optimisation, for reproducible runs."] = 42,
        impact_mode: Annotated[
            Literal["online", "static"],
            "How the IMPACT deformable metric reads its features. 'online' extracts them inside the loss at "
            "every optimiser step, so the warp is differentiated through the network. 'static' extracts them "
            "once per image and registers the feature volumes themselves: far less device memory, no network "
            "in the optimisation loop, and the only path a whole-image feature model can take.",
        ] = "online",
        feature_patch: Annotated[
            int,
            "Static mode only: the cube of voxels each feature extraction pass sees, 0 for the whole image "
            "at once. Tiles share a quarter of their width and are blended by a cosine window, so a volume "
            "larger than the card still goes through.",
            Range(0, 1024),
        ] = 0,
        feature_chunk: Annotated[
            int,
            "Static mode only: how many feature channels the local cross-correlation compares at a time, 0 "
            "for all of them at once. A smaller chunk trades a little time for a peak that follows the chunk "
            "instead of the channel count, which is what lets several feature models share one card.",
            Range(0, 64),
        ] = 0,
        models: dict[str, ModelSpec] = {},
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=3,
        )
        # Fail at build time: with no feature model the IMPACT loss would surface as a None-loss crash
        # deep in the deformable stage, minutes after the rigid/affine stages already ran. With
        # deformable_method 'none' the metric is never consumed, so a stale 'impact' stays harmless.
        if deformable_method != "none" and deformable_metric == "impact" and not models:
            raise ValueError("deformable_metric='impact' requires at least one feature model under 'models'.")
        engine = FireANTsEngine(
            scales,
            affine_iterations,
            deformable_iterations,
            cc_kernel,
            affine_metric,
            affine_lr,
            moments_init,
            linear_method,
            deformable_method,
            deformable_metric,
            deformable_lr,
            integrator_n,
            smooth_warp_sigma,
            smooth_grad_sigma,
            seed,
            _sorted_specs(models),
            impact_mode,
            feature_patch,
            feature_chunk,
        )
        self.add_module(
            "Registration", FireANTsRegistration(engine), in_branch=[0, 1, 2, 3], out_branch=["registration"]
        )
        self.add_module("DisplacementField", ChannelSelect(0, 3), in_branch=["registration"], out_branch=["dvf"])
