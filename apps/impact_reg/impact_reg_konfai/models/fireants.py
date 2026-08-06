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
# v1.0 and must be cited — see the NOTICE file in this directory for the license, copyright and
# bibliography that ship with this app.

"""FireANTs registration as a self-contained KonfAI model (shared by the FireANTs presets).

Same idiomatic ``add_module`` graph and the same output contract as the ConvexAdam preset
(``MovedImage`` + ``DisplacementField`` on the FIXED grid, split by two ``ChannelSelect``), so the
orchestrator / app.json / ensemble / uncertainty are unchanged. The engine chains FireANTs' own
composable stages (GPU, Riemannian Adam), each seeding the next like ANTs' ``-t`` stages:

    Rigid (MI, centre-of-mass init) -> Affine (MI, seeded by the rigid) -> deformable

Two mirrored knobs specialise this shared module into the different presets (as ConvexAdam's shared
module is specialised by ``stages``). ``deformable_method`` picks the deformable stage:

    "syn"    symmetric diffeomorphic SyN (CC)   — invertible, higher quality, averages cleanly for ensembling
    "greedy" greedy diffeomorphic (CC)          — one-directional, faster / lower VRAM
    "none"   no deformable                      — the linear stage IS the transform (FireANTs_Affine)

and ``linear_method`` picks the linear one:

    "rigid_affine"  Rigid then Affine, as ANTs   — the default
    "rigid"         Rigid only                   — no free scale or shear
    "none"          deformable from identity     — for a pair already globally aligned

They compose, so ``deformable_method="none"`` runs whichever linear stage was asked for rather than
always Rigid+Affine: with ``linear_method="rigid"`` it is a rigid-only registration. The one
combination refused is both set to ``"none"`` -- that leaves nothing to optimise, and the engine
raises ``ValueError`` at build time rather than returning an identity transform.

Masks: the optional Fixed/Moving masks restrict the metric to a region. FireANTs implements this by
carrying the mask as the last image channel and prefixing the metric with ``masked_``; a mask is only
honoured when it actually restricts (some voxels in, some out), so the common mask-free path is
unchanged (an absent optional mask arrives as a whole-image default and is treated as no mask).

The deformable stages produce the single TOTAL displacement field on the fixed grid (the linear
pre-align is baked in via ``init_affine``, ANTs convention); ``none`` uses the affine matrix directly.
``MovedImage`` and the emitted ``DisplacementField`` are rebuilt from that transform with SimpleITK —
the same output path as the ConvexAdam engine — so all presets/engines are interchangeable in an
ensemble. FireANTs' output-transform writer only serialises to a file, so the deformable field is
round-tripped through a temporary NIfTI (no FireANTs internals are reimplemented here).

NOTE: do NOT add ``from __future__ import annotations`` — KonfAI's config engine relies on
runtime-evaluated annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import SimpleITK as sitk
import torch
from konfai.metric.measure import IMPACTReg
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
# value -- for Dice this means the SOFT overlap (the C++ rounds activations to {0, 1} and cannot be autograd'd).
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
    """Make the ``fireants`` runtime importable before a preset uses it — best-effort, never fatal.

    A plain ``pip install fireants`` fails inside a host like 3D Slicer for two reasons, both handled
    here with a clear one-line status at each step:

    1. **fireants won't install.** It pins ``simpleitk==2.2.1`` (no wheel on modern Python), while the
       host already ships a newer SimpleITK. We install it with ``--no-deps`` so that pin is ignored
       and the host's SimpleITK/torch are reused; its light deps ship in ``requirements.txt``.
    2. **The fused CUDA kernels** (``fireants_fused_ops``) that make registration fast and
       memory-light are OPTIONAL. Without them fireants runs in pure PyTorch (correct, only slower).
       We enable them only when a CUDA compiler (``nvcc``) is present, compiling from the upstream
       FireANTs source at install time — nothing is vendored into this app. No compiler, or a failed
       build, simply falls back to pure PyTorch.
    """
    import importlib
    import shutil
    import subprocess
    import sys

    def _log(message: str) -> None:
        print(f"[FireANTs] {message}", flush=True)

    # 1) fireants itself — install --no-deps to sidestep its unsatisfiable ``simpleitk==2.2.1`` pin.
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

    # 2) fused CUDA kernels — optional accelerator.
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
    # path for end users is a prebuilt wheel (2a) — a local compile also needs Python dev headers
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
    """The per-model ``ref`` picker's values — model refs (``repo:path``) from the feature-model registry."""
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
            f"models_registry '{ref}' must be a 'repo:file' Hugging Face reference — or set "
            "KONFAI_IMPACT_MODELS_REGISTRY to a local file for offline use."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_specs(mapping: dict) -> list:
    """A dict keyed by string indices ('0','1',...) -> its values in numeric order."""
    return [mapping[k] for k in sorted(mapping, key=lambda key: int(key))]


@dataclass
class ModelSpec:
    """One IMPACT feature model in the deformable metric (several are fused). ``ref`` picks the model; the
    rest are its per-model knobs — the same as the ConvexAdam / elastix ``ModelSpec`` except ``voxel_size``
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

    Reuses ``IMPACTReg._compute`` / ``preprocessing`` verbatim — the stats-normalised feature extraction
    (the model wants per-image ``[min, mean, max, std]``) and the per-layer weighted distance — so the
    metric is exactly KonfAI's, not a re-derivation. Only KonfAI's config-binding ``__init__`` and its
    ``Attribute``-based geometry are replaced: FireANTs passes raw tensors at the current pyramid scale, so
    the intensity statistics are computed from those tensors directly. ``pca`` (absent from KonfAI's torch
    ``IMPACTReg``) is added here as a per-layer feature-space reduction matching itk-impact.
    """

    def __init__(self, ref: str, in_channels: int, weights: list[float], distance: str, pca: int) -> None:
        from huggingface_hub import hf_hub_download

        torch.nn.Module.__init__(self)  # bypass IMPACTReg.__init__ (KONFAI_CONFIG_PATH / apply_config binding)
        self.name = "Reg"
        self.in_channels = int(in_channels)
        self.weights = [float(w) for w in weights]
        self.nb_layer = len(self.weights)
        self.loss = _DISTANCES[distance]()
        self.pca = int(pca)  # PCA lives in KonfAI's IMPACTReg._compute (same behaviour as itk-impact)
        self.dim = DIM
        self.shape = None  # score the whole (downsampled) tensor — no ModelPatch tiling
        if _is_local_ref(ref):  # otherwise a "repo:path" HF reference
            self.model_path = ref
        else:
            repo, filename = ref.split(":", 1)
            self.model_path = hf_hub_download(repo, filename, repo_type="model")  # nosec B615
        self.model = None  # lazy-loaded on the first forward, like IMPACTReg

    @staticmethod
    def _stats(tensor: torch.Tensor) -> dict:
        detached = tensor.detach()
        return {
            "ImageMin": float(detached.min()),
            "ImageMean": float(detached.mean()),
            "ImageMax": float(detached.max()),
            "ImageStd": float(detached.std()),
        }

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.model is None:
            self.model = torch.jit.load(self.model_path)  # nosec B614
        self.model.to(moved.device).eval()
        with _no_texpr_fuser():
            loss, true_nb = self._compute(moved, [self._stats(moved)], fixed, [self._stats(fixed)], None)
        return loss / max(true_nb, 1)


class ImpactFeatureLoss(torch.nn.Module):
    """FireANTs ``custom_loss`` = the KonfAI IMPACT metric fused over several feature models.

    ``forward(moved, fixed)`` sums each model's ``layers_weight * IMPACT(model)``. A model's per-layer
    weights come from its ``layers_mask`` bitmask; its input channel count is read from the registry
    (``models.json`` ``numberofchannels``) so it never has to be configured by hand.
    """

    def __init__(self, specs: list["ModelSpec"]) -> None:
        super().__init__()
        registry = load_models_registry()
        self._cores = torch.nn.ModuleList()
        self._model_weights: list[float] = []
        for spec in specs:
            in_channels = int(registry.get(spec.ref.split(":", 1)[-1], {}).get("numberofchannels", 1))
            weights = [1.0 if char == "1" else 0.0 for char in spec.layers_mask]
            self._cores.append(_ImpactCore(spec.ref, in_channels, weights, spec.distance, spec.pca))
            self._model_weights.append(float(spec.layers_weight))

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        total: torch.Tensor | None = None
        for weight, core in zip(self._model_weights, self._cores, strict=True):
            term = weight * core(moved, fixed)
            total = term if total is None else total + term
        return total


class FireANTsEngine:
    """Register a fixed/moving pair with FireANTs (Rigid -> Affine -> [SyN | Greedy | none]); return
    (moved, dvf) on the fixed grid.

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
            # the rigid-then-affine branch, so a typo -- "affine", say -- would register with a stage
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
        """True only for a mask that actually restricts the region — some voxels in, some out. An absent
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
        round-tripped through a temporary NIfTI — its public API, no internals reimplemented."""
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Register ``moving`` onto ``fixed``; return (moved, dvf) as channel-first arrays on the fixed grid."""
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
            fmask = Image(fixed_mask, device=device) if use_fixed_mask else generate_image_mask_allones(fixed_img)
            mmask = Image(moving_mask, device=device) if use_moving_mask else generate_image_mask_allones(moving_img)
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
        # and starts the deformable from identity: for a pair already globally aligned -- a tiled
        # refinement at full resolution, where each patch sees only local anatomy, so a per-patch
        # rigid has no global meaning and neighbouring patches would each estimate a different one
        # and tear the blended field at the seams.
        affine_matrix: "torch.Tensor | None"
        if self._linear_method == "none":
            affine_matrix = None  # the deformable stage builds its own identity init
        else:
            # The rigid's starting translation mirrors ANTs' ``-r [fixed,moving,N]``: "cof" is the
            # centre of FRAME (N=0), "com" the centre of MASS (N=1). "cof" aligns the image frames,
            # not the subjects, so a subject sitting off its frame centre starts the chain misplaced.
            init_translation: "str | torch.Tensor" = "cof"
            if self._moments_init == "com":
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
                # The rigid -- rotation and translation, no scale or shear -- IS the linear transform.
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
            # (the linear pre-align keeps its own affine_metric); masks do not restrict the IMPACT metric.
            if self._deformable_metric == "impact":
                loss_type: str = "custom"
                custom_loss: torch.nn.Module | None = ImpactFeatureLoss(self._impact_specs)
            else:
                loss_type, custom_loss = deformable_loss, None
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

        # Rebuild moved + DVF from the single transform on the fixed grid — the ConvexAdam output path,
        # so every FireANTs preset emits identical-shaped results.
        moved = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        dvf = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            fixed.GetSize(),
            fixed.GetOrigin(),
            fixed.GetSpacing(),
            fixed.GetDirection(),
        )
        moved_np, _ = image_to_data(moved)
        dvf_np, _ = image_to_data(dvf)
        return moved_np, dvf_np


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
                moved_np, dvf_np = self._engine.register(
                    fixed_img, moving_img, device_index, fixed_mask_img, moving_mask_img
                )
                combined.append(torch.from_numpy(np.concatenate([moved_np, dvf_np], axis=0)))
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

    Outputs on the fixed grid: ``MovedImage`` (moving resampled onto fixed) and ``DisplacementField``
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
            Literal["cof", "com"],
            "Initial translation seeding the rigid stage, mirroring ANTs' -r [fixed,moving,N]: 'cof' = centre "
            "of frame (N=0), 'com' = intensity-weighted centre of mass (N=1). Prefer 'com' when the subject "
            "sits off its frame centre, where aligning frames starts the chain misplaced.",
        ] = "cof",
        linear_method: Annotated[
            Literal["rigid_affine", "rigid", "none"],
            "Linear stage before the deformable: 'rigid_affine' (rigid then affine, as ANTs), 'rigid' to stop "
            "after the rigid and leave scale and shear alone, or 'none' to skip it and start the deformable "
            "from identity. Use 'none' only on a pair already globally aligned -- a tiled pass at full "
            "resolution, where a patch sees no global context and a per-patch linear has no global meaning.",
        ] = "rigid_affine",
        deformable_method: Annotated[
            Literal["none", "syn", "greedy"],
            "Deformable algorithm: 'syn' (symmetric diffeomorphic), 'greedy', or 'none' to stop after the linear "
            "stage, whichever 'linear_method' selected -- with linear_method='rigid' that is a rigid-only "
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
        )
        self.add_module(
            "Registration", FireANTsRegistration(engine), in_branch=[0, 1, 2, 3], out_branch=["registration"]
        )
        self.add_module("MovedImage", ChannelSelect(0, 1), in_branch=["registration"], out_branch=["moved"])
        self.add_module("DisplacementField", ChannelSelect(1, 4), in_branch=["registration"], out_branch=["dvf"])
