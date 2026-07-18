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

"""ConvexAdam (itk-impact) registration as a self-contained KonfAI model.

Same idiomatic ``add_module`` graph and the same output contract as the elastix preset
(``MovedImage`` + ``DisplacementField`` on the FIXED grid, split by two ``ChannelSelect``),
so the orchestrator / app.json / ensemble / uncertainty are unchanged. The engine here is
the native, in-memory itk-impact ConvexAdam pipeline (``pip install itk-impact``) instead of
the elastix binary:

    (optional) moments + affine Mattes-MI          [ITKv4 linear pre-align]
      -> ImpactCoarseRegistration                   [coupled-convex init, IMPACT features]
      -> ImpactFineRegistration                     [Adam instance optimisation, IMPACT features]

The IMPACT feature models (e.g. MIND) are TorchScript ``.pt`` files fetched from Hugging Face
and wrapped as ``itk.ModelConfiguration`` — the same models the elastix presets use.

NOTE: do NOT add ``from __future__ import annotations`` — KonfAI's config engine relies on
runtime-evaluated annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import itk
import numpy as np
import SimpleITK as sitk
import torch
import tqdm
from huggingface_hub import hf_hub_download
from konfai.network import network
from konfai.utils.config import Choices, Range
from konfai.utils.dataset import Attribute, data_to_image, image_to_data

DIM = 3
# The feature model's input channel count is an intrinsic property of the pretrained model (grayscale
# medical images), not a tunable — so it's fixed here, never a config/signature parameter.
NUM_CHANNELS = 1

# A UI reads the tuning knobs straight from the TYPES on ``RegistrationNet.__init__`` and ``ModelSpec``:
# ``Annotated[.., Range]`` gives numeric spin bounds; ``Literal`` / ``Annotated[str, Choices]`` a dropdown.
# ``models`` is a dict-of-objects (one ``ModelSpec`` per feature model) — the same shape as the elastix presets,
# so SlicerKonfAI renders each model as a repeatable block with a ``ref`` / ``distance`` combo box.
_IMAGE_F = itk.Image[itk.F, DIM]

_IMPACT_MODELS_REGISTRY = "VBoussot/impact-torchscript-models:models.json"


def registry_choices() -> list[str]:
    """The per-model ``ref`` picker's values — model refs (``repo:path``) from the feature-model registry the
    engine already fetches (offline-first). A user may still point ``ref`` at a local model path."""
    repo = _IMPACT_MODELS_REGISTRY.split(":", 1)[0]
    return [f"{repo}:{key}" for key in load_models_registry()]


def load_models_registry(ref: str = _IMPACT_MODELS_REGISTRY) -> dict:
    """Load ``models.json`` (the available feature models) from the model repo on Hugging Face. The registry is
    NOT bundled: ``KONFAI_IMPACT_MODELS_REGISTRY`` (a local path) wins for dev/offline; otherwise ``ref`` must be
    a ``repo:file`` Hugging Face reference."""
    local = os.environ.get("KONFAI_IMPACT_MODELS_REGISTRY", "")
    if local:
        path = Path(local)
    elif ":" in ref:
        repo, filename = ref.split(":", 1)
        path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="model"))  # nosec B615
    else:
        raise ValueError(
            f"models_registry '{ref}' must be a 'repo:file' Hugging Face reference (fetched from HF, not "
            "bundled) — or set KONFAI_IMPACT_MODELS_REGISTRY to a local file for offline use."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_specs(mapping: dict) -> list:
    """A dict keyed by string indices ('0','1',...) -> its values in numeric order."""
    return [mapping[k] for k in sorted(mapping, key=lambda key: int(key))]


@dataclass
class ModelSpec:
    """One feature model in the ConvexAdam multi-feature fusion. ``ref`` picks the model; the rest are its
    per-model knobs (all models are compared jointly by the IMPACT metric). Same idea as the elastix ``ModelSpec``."""

    ref: Annotated[
        str,
        Choices(registry_choices),
        "IMPACT feature model that drives the similarity (TorchScript 'repo:file' on Hugging Face); "
        "different models capture different anatomy/contrast.",
    ]
    voxel_size: Annotated[
        list[float],
        "Working resolution (mm) the pair is resampled to before feature extraction; larger = coarser and "
        "faster, smaller = finer and slower.",
    ] = field(default_factory=lambda: [3.0, 3.0, 3.0])
    layers_mask: Annotated[
        str,
        "Per-layer on/off bitmask over the feature model's layers ('1' = use, '0' = skip), one char per layer; "
        "selects which feature depths drive the metric.",
    ] = "1"
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
        "Similarity measure compared on the extracted features between fixed and moving.",
    ] = "L1"


@contextlib.contextmanager
def _no_texpr_fuser():
    """Disable ONLY torch's TensorExpr (NNC) fuser for the block. The TS feature-model graphs have a direction/
    orientation branch whose shape ops (aten::dim / aten::size) crash the fuser's alias analysis under itk-impact
    ("INTERNAL ASSERT ... We don't have an op for aten::size" in FuseTensorExprs). It cannot be fixed model-side:
    itk-impact passes an *undefined* direction, so guarding it needs a shape op — the very fuser trigger. The
    modern profiling executor stays on (NOT the legacy executor); measured cost ~1% of a registration (the fuser
    only touches the few feature forwards, not the C++ optimisation loop). The caller's setting is restored."""
    prev = torch._C._jit_texpr_fuser_enabled()
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_fuser_enabled(prev)


def _coarse_registration_type():
    """The coupled-convex initializer, tolerant to the two names the wrapping has shipped under."""
    cls = getattr(itk, "ImpactCoarseRegistration", None) or getattr(itk, "ImpactConvexAdamInitializer", None)
    if cls is None:
        raise RuntimeError(
            "itk-impact does not expose ImpactCoarseRegistration / ImpactConvexAdamInitializer; "
            "install a build with the ConvexAdam registration filters."
        )
    return cls[_IMAGE_F, _IMAGE_F]


def _fine_registration_type():
    """The Adam instance-optimisation stage, tolerant to the two names the wrapping has shipped under."""
    cls = getattr(itk, "ImpactFineRegistration", None) or getattr(itk, "ImpactTorchAdamRegistration", None)
    if cls is None:
        raise RuntimeError(
            "itk-impact does not expose ImpactFineRegistration / ImpactTorchAdamRegistration; "
            "install a build with the ConvexAdam registration filters."
        )
    return cls[_IMAGE_F, _IMAGE_F]


def _sitk_to_itk(image: sitk.Image) -> "itk.Image":
    """Copy a scalar SimpleITK image (with its geometry) into an ``itk.Image[F, 3]``."""
    itk_image = itk.image_from_array(sitk.GetArrayFromImage(image).astype(np.float32))
    itk_image.SetOrigin([float(v) for v in image.GetOrigin()])
    itk_image.SetSpacing([float(v) for v in image.GetSpacing()])
    itk_image.SetDirection(itk.matrix_from_array(np.asarray(image.GetDirection(), dtype=float).reshape(DIM, DIM)))
    return itk_image


def _itk_field_to_sitk_transform(field: "itk.Image", reference: sitk.Image) -> sitk.Transform:
    """Wrap an itk displacement field (on the fixed grid) as a SimpleITK ``DisplacementFieldTransform``."""
    array = itk.array_from_image(field).astype(np.float64)  # [Z, Y, X, 3]
    sitk_field = sitk.GetImageFromArray(array, isVector=True)
    sitk_field.CopyInformation(reference)
    return sitk.DisplacementFieldTransform(sitk.Cast(sitk_field, sitk.sitkVectorFloat64))


def _itk_affine_to_sitk(affine: "itk.AffineTransform") -> sitk.AffineTransform:
    """Convert an ``itk.AffineTransform[D, 3]`` into a SimpleITK ``AffineTransform`` (same LPS convention)."""
    sitk_affine = sitk.AffineTransform(DIM)
    sitk_affine.SetMatrix([float(v) for v in itk.array_from_matrix(affine.GetMatrix()).flatten()])
    sitk_affine.SetTranslation([float(v) for v in affine.GetTranslation()])
    sitk_affine.SetCenter([float(v) for v in affine.GetCenter()])
    return sitk_affine


class ConvexAdamEngine:
    """Register a fixed/moving pair with the itk-impact ConvexAdam pipeline; return (moved, dvf) on the fixed grid.

    The IMPACT feature models are downloaded once (``repo:filename`` on Hugging Face) and reused across cases.
    Masks are accepted for signature compatibility with the elastix engine but ignored: the ConvexAdam
    filters optimise over the whole image (no mask API is exposed by the coarse/fine stages).
    """

    def __init__(
        self,
        models: list[str],
        voxel_sizes: list[list[float]],
        overlap: int,
        layers_masks: list[list[bool]],
        mixed_precision: bool,
        grid_spacing: int,
        displacement_half_width: int,
        iterations: int,
        learning_rate: float,
        regularization_weight: float,
        grid_shrink: int,
        distance: list[str],
        layers_weight: list[float],
        subset_features: list[int],
        pca: list[int],
        stages: list[str],
        linear: bool,
        linear_iterations: int,
        seed: int,
    ) -> None:
        self._stages = stages
        self._model_paths = self._download_models(models)
        # Built lazily and cached: constructing an itk.ModelConfiguration loads the TorchScript model
        # from disk in C++, so build the list once and reuse it across both stages and every case.
        self._configurations: "list[itk.ModelConfiguration] | None" = None
        self._voxel_sizes = voxel_sizes
        self._overlap = overlap
        self._layers_masks = layers_masks
        self._mixed_precision = mixed_precision
        self._grid_spacing = grid_spacing
        self._displacement_half_width = displacement_half_width
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._regularization_weight = regularization_weight
        self._grid_shrink = grid_shrink
        self._distance = distance
        self._layers_weight = layers_weight
        self._subset_features = subset_features
        self._pca = pca
        self._linear = linear
        self._linear_iterations = linear_iterations
        self._seed = seed

    @staticmethod
    def _download_models(models: list[str]) -> list[str]:
        """Fetch the TorchScript feature models (``repo:filename``); return their local paths."""
        paths = []
        for ref in models:
            repo, filename = ref.split(":", 1)
            paths.append(str(hf_hub_download(repo_id=repo, filename=filename, repo_type="model")))  # nosec B615
        return paths

    def _model_configurations(self) -> list["itk.ModelConfiguration"]:
        """Build one ``ModelConfiguration`` per feature model once, then reuse it across stages and cases.

        Constructing an ``itk.ModelConfiguration`` loads the TorchScript module from disk on the C++ side, so
        it is built lazily and cached. The coarse/fine filters copy each configuration by value in
        ``AddModelConfiguration`` and the copy shares the loaded module through the configuration's internal
        ``shared_ptr`` — so a single build is reused everywhere without any reload.
        """
        if self._configurations is None:
            self._configurations = [
                itk.ModelConfiguration(
                    path,
                    DIM,
                    NUM_CHANNELS,
                    [0, 0, 0],
                    [float(v) for v in voxel_size],
                    self._overlap,
                    list(layers_mask),
                    self._mixed_precision,
                )
                for path, voxel_size, layers_mask in zip(self._model_paths, self._voxel_sizes, self._layers_masks)
            ]
        return self._configurations

    def _linear_align(self, fixed: "itk.Image", moving: "itk.Image") -> "itk.AffineTransform":
        """Moments-initialised rigid + affine (Mattes MI), mapping fixed -> moving physical points."""
        rigid = itk.VersorRigid3DTransform[itk.D].New()
        initializer = itk.CenteredTransformInitializer[itk.VersorRigid3DTransform[itk.D], _IMAGE_F, _IMAGE_F].New(
            Transform=rigid, FixedImage=fixed, MovingImage=moving
        )
        initializer.MomentsOn()
        initializer.InitializeTransform()

        affine = itk.AffineTransform[itk.D, DIM].New()
        affine.SetCenter(rigid.GetCenter())
        affine.SetMatrix(rigid.GetMatrix())
        affine.SetOffset(rigid.GetOffset())
        levels = 3
        metric_type = itk.MattesMutualInformationImageToImageMetricv4[_IMAGE_F, _IMAGE_F]
        metric = metric_type.New()
        metric.SetNumberOfHistogramBins(32)
        optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
        optimizer.SetNumberOfIterations(self._linear_iterations)
        optimizer.SetLearningRate(1.0)
        optimizer.SetMinimumStepLength(1e-5)
        optimizer.SetRelaxationFactor(0.6)
        scales = itk.RegistrationParameterScalesFromPhysicalShift[metric_type].New()
        scales.SetMetric(metric)
        optimizer.SetScalesEstimator(scales)
        registration = itk.ImageRegistrationMethodv4[_IMAGE_F, _IMAGE_F].New(
            FixedImage=fixed, MovingImage=moving, Metric=metric, Optimizer=optimizer, InitialTransform=affine
        )
        registration.SetNumberOfLevels(levels)
        registration.SetShrinkFactorsPerLevel([2 ** (levels - 1 - i) for i in range(levels)])
        registration.SetSmoothingSigmasPerLevel([float(levels - 1 - i) for i in range(levels)])
        registration.InPlaceOn()
        registration.Update()
        return affine

    def _coarse(self, fixed: "itk.Image", moving: "itk.Image", device: str) -> "itk.Image":
        """ConvexAdam coarse coupled-convex initializer -> robust low-resolution field on the fixed grid."""
        coarse = _coarse_registration_type().New()
        coarse.SetFixedImage(fixed)
        coarse.SetMovingImage(moving)
        for configuration in self._model_configurations():
            coarse.AddModelConfiguration(configuration)
        coarse.SetGridSpacing(self._grid_spacing)
        coarse.SetDisplacementHalfWidth(self._displacement_half_width)
        coarse.SetDevice(device)
        coarse.SetSeed(self._seed)
        coarse.Update()
        field = coarse.GetOutput()
        field.DisconnectPipeline()
        return field

    def _fine(
        self, fixed: "itk.Image", moving: "itk.Image", initial_field: "itk.Image | None", device: str
    ) -> "itk.Image":
        """Adam instance-optimisation refinement, warm-started from ``initial_field`` (zero if none)."""
        fine = _fine_registration_type().New()
        fine.SetFixedImage(fixed)
        fine.SetMovingImage(moving)
        fine.SetInitialDisplacementField(initial_field if initial_field is not None else self._zero_field(fixed))
        for configuration in self._model_configurations():
            fine.AddModelConfiguration(configuration)
        fine.SetDistance(list(self._distance))
        fine.SetLayersWeight([float(v) for v in self._layers_weight])
        fine.SetSubsetFeatures([int(v) for v in self._subset_features])
        fine.SetPCA([int(v) for v in self._pca])
        fine.SetNumberOfIterations(self._iterations)
        fine.SetLearningRate(self._learning_rate)
        fine.SetRegularizationWeight(self._regularization_weight)
        fine.SetGridShrinkFactor(self._grid_shrink)
        fine.SetDevice(device)
        fine.SetSeed(self._seed)

        # Optional terminal progress over the Adam iterations, driven from the metric trace. ``disable=None``
        # auto-hides it when stderr is not a TTY (e.g. under KonfAI/Slicer, where the outer "Prediction" bar
        # already reports progress), so captured logs stay clean; ``leave=False`` avoids stacking one bar per
        # patch. The observer is best-effort — if the filter emits no IterationEvent the bar just fills at the end.
        progress = tqdm.tqdm(
            total=self._iterations or None, desc="Registration", ncols=0, leave=False, disable=None
        )

        def _update(*_: object) -> None:
            values = list(fine.GetMetricValuesPerIteration())
            progress.n = min(len(values), self._iterations)
            if values:
                progress.set_description(f"Registration : iter {len(values)} | metric {float(values[-1]):.4f}")
            progress.refresh()

        try:
            fine.AddObserver(itk.IterationEvent(), _update)
        except Exception:  # nosec B110 - progress is best-effort; never fail a run over the bar
            pass
        fine.Update()
        progress.n = progress.total or self._iterations  # show completion even if no IterationEvent fired
        progress.refresh()
        progress.close()
        field = fine.GetDisplacementField()
        field.DisconnectPipeline()
        return field

    @staticmethod
    def _zero_field(reference: "itk.Image") -> "itk.Image":
        """An all-zero displacement field on ``reference``'s grid (identity warm-start for a lone fine stage)."""
        field = itk.Image[itk.Vector[itk.F, DIM], DIM].New()
        field.CopyInformation(reference)
        field.SetRegions(reference.GetLargestPossibleRegion())
        field.Allocate()
        zero = itk.Vector[itk.F, DIM]()
        zero.Fill(0)  # itk::Vector default ctor does not zero-initialise
        field.FillBuffer(zero)
        return field

    def _run_stages(self, fixed: "itk.Image", moving: "itk.Image", device: str) -> "itk.Image | None":
        """Run the configured coarse/fine chain; each fine warm-starts from the running field.

        ``coarse`` produces a field from scratch; ``fine`` refines the running field. So ``['coarse']`` is a
        coarse-only app, ``['fine']`` a fine-only app (zero warm-start), and ``['coarse', 'fine']`` chains both
        (the composite, as before). Returns None when no deformable stage runs (e.g. a linear-only chain).
        """
        field: "itk.Image | None" = None
        for stage in self._stages:
            if stage == "coarse":
                field = self._coarse(fixed, moving, device)
            elif stage == "fine":
                field = self._fine(fixed, moving, field, device)
            else:
                raise ValueError(f"Unknown registration stage '{stage}' (expected 'coarse' or 'fine').")
        return field

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Register ``moving`` onto ``fixed``; return (moved, dvf) as channel-first arrays on the fixed grid."""
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        fixed_itk = _sitk_to_itk(fixed)
        moving_itk = _sitk_to_itk(moving)

        # Optional linear pre-align: resample the moving onto the fixed grid so the deformable stage starts close.
        affine = self._linear_align(fixed_itk, moving_itk) if self._linear else itk.AffineTransform[itk.D, DIM].New()
        resampler = itk.ResampleImageFilter[_IMAGE_F, _IMAGE_F].New(
            Input=moving_itk, ReferenceImage=fixed_itk, Transform=affine
        )
        resampler.UseReferenceImageOn()
        resampler.SetInterpolator(itk.LinearInterpolateImageFunction[_IMAGE_F, itk.D].New())
        resampler.Update()
        moving_linear = resampler.GetOutput()

        field = self._run_stages(fixed_itk, moving_linear, device)

        # One transform on the fixed grid = affine then deformable, so the returned DVF/transform warps the
        # ORIGINAL moving. SimpleITK applies the last-added transform first, so [affine, deformable] gives
        # moved(p) = moving(affine(deformable(p))). A linear-only chain (field is None) yields the affine alone.
        chain = [_itk_affine_to_sitk(affine)]
        if field is not None:
            chain.append(_itk_field_to_sitk_transform(field, fixed))
        composite = sitk.CompositeTransform(chain)
        moved = sitk.Resample(moving, fixed, composite, sitk.sitkLinear, 0.0, moving.GetPixelID())
        dvf = sitk.TransformToDisplacementField(
            composite,
            sitk.sitkVectorFloat64,
            fixed.GetSize(),
            fixed.GetOrigin(),
            fixed.GetSpacing(),
            fixed.GetDirection(),
        )
        moved_np, _ = image_to_data(moved)
        dvf_np, _ = image_to_data(dvf)
        return moved_np, dvf_np


class ConvexAdamRegistration(torch.nn.Module):
    """Graph module: (fixed, moving) tensors + their geometry -> moved image + DVF on the fixed grid.

    ``accepts_attributes = True`` opts this module into receiving the per-branch ``Attribute`` list alongside
    the tensors (same convention as ``CriterionWithAttribute``); registration needs the physical geometry.
    """

    accepts_attributes = True

    def __init__(self, engine: ConvexAdamEngine) -> None:
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
        # attributes = [fixed, moving, fixed_mask, moving_mask] branch attrs; each a list[Attribute] over the batch.
        # Returns, per sample, the moved image (1 channel) channel-stacked with the displacement field (DIM
        # channels); downstream ChannelSelect modules split them. Masks are ignored by the ConvexAdam engine.
        fixed_attrs, moving_attrs, _, _ = attributes
        device_index = fixed.device.index if fixed.device.type == "cuda" else -1
        combined = []
        # ConvexAdam runs a gradient-based instance optimisation (Adam over the field) inside itk-impact's
        # .Update(); the predictor calls forward under torch.inference_mode(), which forbids autograd. The
        # image tensors have already crossed to numpy/ITK here, so re-enable grad for the optimisation.
        with torch.inference_mode(False), torch.enable_grad(), _no_texpr_fuser():
            for b in range(fixed.shape[0]):
                fixed_img = data_to_image(fixed[b].detach().cpu().numpy(), fixed_attrs[b])
                moving_img = data_to_image(moving[b].detach().cpu().numpy(), moving_attrs[b])
                moved_np, dvf_np = self._engine.register(fixed_img, moving_img, device_index)
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
    """Pairwise ConvexAdam registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1;
    the mask branches 2/3 are accepted but unused by this engine).

    Outputs on the fixed grid: ``MovedImage`` (moving resampled onto fixed) and ``DisplacementField`` (the
    DIM-component displacement field, in mm). Geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``.
    """

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        models: dict[str, ModelSpec] = {},
        overlap: Annotated[
            int,
            Range(1, 128),
            "Patch overlap of the coarse coupled-convex block matching; higher = denser sampling and a "
            "smoother initialisation, slower.",
        ] = 2,
        mixed_precision: Annotated[
            bool,
            "Run the optimisation in mixed precision (fp16); faster and lighter on VRAM, marginally less precise.",
        ] = False,
        grid_spacing: Annotated[
            int,
            Range(1, 512),
            "Control-point spacing (voxels) of the displacement grid; smaller = a more flexible deformation, "
            "slower and less regular.",
        ] = 4,
        displacement_half_width: Annotated[
            int,
            Range(1, 512),
            "Half-width (voxels) of the discrete displacement search in the coarse stage; raise it to capture "
            "large motion (e.g. deep breathing), at more cost.",
        ] = 6,
        iterations: Annotated[
            int,
            Range(0, 100000),
            "Adam instance-optimisation steps of the fine stage; higher = more converged and accurate, slower.",
        ] = 150,
        learning_rate: Annotated[
            float,
            Range(0.0, 100.0),
            "Adam step size of the fine stage; higher converges faster but can oscillate or diverge.",
        ] = 0.2,
        regularization_weight: Annotated[
            float,
            Range(0.0, 1000.0),
            "Weight of the smoothness (diffusion) regulariser on the displacement field; higher = smoother, "
            "more regular Jacobian, lower = more flexible but risks folding.",
        ] = 1.0,
        grid_shrink: Annotated[
            int,
            Range(1, 128),
            "Downsampling factor of the coarse optimisation grid; higher = a coarser, faster initialisation, "
            "lower = finer.",
        ] = 4,
        subset_features: Annotated[
            list[int],
            "Feature-channel indices to keep (empty = all); a hand-picked subset of channels, NOT a count.",
        ] = [],
        stages: Annotated[
            list[str],
            "Stages to run: 'coarse' (coupled-convex initialisation) then 'fine' (Adam optimisation); drop "
            "'fine' for a fast low-resolution field.",
        ] = ["coarse", "fine"],
        linear: Annotated[
            bool,
            "Run a moments+affine linear pre-alignment before the deformable stages (recommended for large "
            "global offsets).",
        ] = True,
        linear_iterations: Annotated[
            int, Range(0, 100000), "Iterations of the linear pre-alignment; higher = a better global affine fit."
        ] = 200,
        seed: Annotated[int, "Random seed for the optimisation, for reproducible runs."] = 42,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=3,
        )
        specs = _sorted_specs(models)
        engine = ConvexAdamEngine(
            [spec.ref for spec in specs],
            [list(spec.voxel_size) for spec in specs],
            overlap,
            [[c == "1" for c in spec.layers_mask] for spec in specs],
            mixed_precision,
            grid_spacing,
            displacement_half_width,
            iterations,
            learning_rate,
            regularization_weight,
            grid_shrink,
            [spec.distance for spec in specs],
            [float(spec.layers_weight) for spec in specs],
            subset_features,
            [int(spec.pca) for spec in specs],
            stages,
            linear,
            linear_iterations,
            seed,
        )
        self.add_module(
            "Registration", ConvexAdamRegistration(engine), in_branch=[0, 1, 2, 3], out_branch=["registration"]
        )
        self.add_module("MovedImage", ChannelSelect(0, 1), in_branch=["registration"], out_branch=["moved"])
        self.add_module("DisplacementField", ChannelSelect(1, 4), in_branch=["registration"], out_branch=["dvf"])
