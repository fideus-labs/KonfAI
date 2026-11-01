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
(``DisplacementField`` on the FIXED grid),
so the orchestrator / app.json / ensemble / uncertainty are unchanged. The engine here is
the native, in-memory itk-impact ConvexAdam pipeline (``pip install itk-impact``) instead of
the elastix binary:

    (optional) affine Mattes-MI                    [ITKv4 linear pre-align, seeded by the foreground centres]
      -> ImpactCoarseRegistration                   [coupled-convex init, IMPACT features]
      -> ImpactFineRegistration                     [Adam instance optimisation, IMPACT features]

The IMPACT feature models (e.g. MIND) are TorchScript ``.pt`` files fetched from Hugging Face
and wrapped as ``itk.ImpactModelConfiguration``: the same models the elastix and FireANTs presets use, compared by
the IMPACT loss every engine shares (``impact_loss.py``: models, distances, normalization, Static and Jacobian),
which needs itk-impact 0.1.6 or later.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on
runtime-evaluated annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import importlib.metadata
import re
from typing import Annotated

import itk
import numpy as np
import SimpleITK as sitk
import torch
import tqdm
from konfai.network import network
from konfai.utils.config import Range
from konfai.utils.dataset import image_to_data
from konfai.utils.errors import MeasureError

from .impact_loss import (
    FeatureMapUpdateInterval,
    LevelSpec,
    LNCCKernel,
    MixedPrecision,
    Mode,
    ModelSpec,
    Normalize,
    check_models,
    feature_model,
    layer_weights,
    level_models,
    per_kept_layer,
)
from .intensity import EngineRegistration, is_partial_mask, winsorized
from .orientation import world_aligned_pair

DIM = 3
# A UI reads the tuning knobs straight from the TYPES on ``RegistrationNet.__init__`` and ``ModelSpec``:
# ``Annotated[.., Range]`` gives numeric spin bounds; ``Literal`` / ``Annotated[str, Choices]`` a dropdown.
# ``models`` is a dict-of-objects (one ``ModelSpec`` per feature model): the same shape as the elastix presets,
# so SlicerKonfAI renders each model as a repeatable block with a ``ref`` / ``distance`` combo box.
_IMAGE_F = itk.Image[itk.F, DIM]


def _coarse_registration_type():
    """The coupled-convex initializer (itk-impact's ImpactCoarseRegistration)."""
    return itk.ImpactCoarseRegistration[_IMAGE_F, _IMAGE_F]


def _fine_registration_type():
    """The Adam instance-optimisation stage (itk-impact's ImpactFineRegistration)."""
    return itk.ImpactFineRegistration[_IMAGE_F, _IMAGE_F]


def _sitk_to_itk(image: sitk.Image, pixel: type = np.float32) -> "itk.Image":
    """Copy a scalar SimpleITK image (with its geometry) into an ``itk.Image[F, 3]`` (``itk.Image[UC, 3]`` for
    ``np.uint8``)."""
    itk_image = itk.image_from_array(sitk.GetArrayFromImage(image).astype(pixel))
    itk_image.SetOrigin([float(v) for v in image.GetOrigin()])
    itk_image.SetSpacing([float(v) for v in image.GetSpacing()])
    itk_image.SetDirection(itk.matrix_from_array(np.asarray(image.GetDirection(), dtype=float).reshape(DIM, DIM)))
    return itk_image


def _binary_mask(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    """``mask`` as 0/1 (in where not 0) on ``image``'s grid. A mask on that grid can differ from it by float rounding
    once it has crossed KonfAI's Attribute round-trip, and itk-impact would then resample it: it takes the image's
    header, so that every change of grid moves it as it moves the image. A mask on another grid is resampled onto it
    by nearest neighbour, as elastix reads a mask in physical space."""
    binary = mask != 0
    if binary.GetSize() == image.GetSize() and all(
        np.allclose(getattr(binary, query)(), getattr(image, query)(), atol=1e-4)
        for query in ("GetOrigin", "GetSpacing", "GetDirection")
    ):
        binary.CopyInformation(image)
        return binary
    return sitk.Resample(binary, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0)


def _resampled(image: "itk.Image", reference: "itk.Image", transform: "itk.Transform", interpolator) -> "itk.Image":
    """``image`` resampled on ``reference``'s grid through ``transform`` (fixed -> moving points), 0 outside it."""
    image_type = type(image)
    resampler = itk.ResampleImageFilter[image_type, image_type].New(
        Input=image, ReferenceImage=reference, Transform=transform
    )
    resampler.UseReferenceImageOn()
    resampler.SetInterpolator(interpolator[image_type, itk.D].New())
    resampler.Update()
    return resampler.GetOutput()


def _itk_impact_predates_its_stages() -> bool:
    """Whether the installed itk-impact is older than 0.1.6, the first whose stages take what this module sets:
    0.1.5 has ``ImpactModelConfiguration`` already and no ``SetSamplingPercentage``. A source build carries no
    distribution to read a version from, and is taken at its symbols."""
    if not hasattr(itk, "ImpactModelConfiguration"):
        return True
    try:
        installed = importlib.metadata.version("itk-impact")
    except importlib.metadata.PackageNotFoundError:
        return False
    return tuple(int(part) for part in re.findall(r"\d+", installed)[:3]) < (0, 1, 6)


def _itk_field_to_sitk_transform(field: "itk.Image", reference: sitk.Image) -> sitk.Transform:
    """Wrap an itk displacement field (on the fixed grid) as a SimpleITK ``DisplacementFieldTransform``, which
    only takes a float64 field."""
    sitk_field = sitk.GetImageFromArray(itk.array_view_from_image(field).astype(np.float64), isVector=True)
    sitk_field.CopyInformation(reference)
    return sitk.DisplacementFieldTransform(sitk_field)


def _foreground_centre(image: sitk.Image, mask: sitk.Image | None) -> tuple[float, ...]:
    """Physical centre of ``image``'s foreground (Otsu threshold, unweighted), inside ``mask`` (0/1, on its grid) if
    there is one: where the linear stage seeds from.

    Intensity moments divide by the total intensity, which is negative or near zero for a CT in HU or a z-scored
    MR, and clipping the negatives first weights a CT by its bone. An unweighted foreground holds for any intensity
    range; an image with no foreground (a constant one) gives its grid's centre.
    """
    foreground = sitk.OtsuThreshold(image, 0, 1)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.SetComputePerimeter(False)
    stats.Execute(foreground if mask is None else sitk.Mask(foreground, mask))
    if stats.HasLabel(1):
        return stats.GetCentroid(1)
    return image.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in image.GetSize()])


def _itk_affine_to_sitk(affine: "itk.AffineTransform") -> sitk.AffineTransform:
    """Convert an ``itk.AffineTransform[D, 3]`` into a SimpleITK ``AffineTransform`` (same LPS convention)."""
    sitk_affine = sitk.AffineTransform(DIM)
    sitk_affine.SetMatrix([float(v) for v in itk.array_from_matrix(affine.GetMatrix()).flatten()])
    sitk_affine.SetTranslation([float(v) for v in affine.GetTranslation()])
    sitk_affine.SetCenter([float(v) for v in affine.GetCenter()])
    return sitk_affine


class ConvexAdamEngine:
    """Register a fixed/moving pair with the itk-impact ConvexAdam pipeline; return the displacement field on the fixed grid.

    The IMPACT feature models are fetched and probed once, by KonfAI (``repo:filename`` on Hugging Face), and reused
    across cases.
    Masks restrict every stage's similarity to where they are not 0, the linear stage and its seed included, as in the
    FireANTs engine; a fixed mask with no voxel in it (a tile the tissue does not reach) gives a zero field.
    """

    def __init__(
        self,
        stage_models: dict[str, list[ModelSpec]],
        mixed_precision: bool,
        grid_spacing: int,
        displacement_half_width: int,
        iterations: int,
        learning_rate: float,
        regularization_weight: float,
        grid_shrink: int,
        control_grid_smoothing: int,
        stages: list[str],
        linear: bool,
        linear_iterations: int,
        seed: int,
        linear_sampling: float = 1.0,
        mode: str = "Static",
        normalize: bool = True,
        feature_map_update_interval: int = -1,
        lncc_kernel: int = 5,
        voxel_sampling: float = 1.0,
        balance_coarse_layers: bool = False,
    ) -> None:
        # Checked here, before any download: past this point a bad value fails at the first case, or not at all.
        if stages and _itk_impact_predates_its_stages():
            raise MeasureError(
                "The ConvexAdam stages need itk-impact 0.1.6 or later.",
                'Install it beside the torch its wheels are built against: pip install "itk-impact==0.1.6"'
                ' "torch==2.12.*"',
            )
        if any(stage not in ("coarse", "fine") for stage in stages) or "coarse" in stages[1:]:
            # The coarse stage starts from scratch: after a 'fine' it would throw the refinement away.
            raise ValueError(
                f"stages {stages} is not a ConvexAdam chain: 'coarse' (first, once) then 'fine', e.g. "
                "['coarse', 'fine']."
            )
        if not stages and not linear:
            raise ValueError("stages=[] with linear=False leaves nothing to register.")
        for stage in stages:
            # Without a model both itk-impact filters compare raw intensities (SSD, then MSE), a different method
            # from the one the preset describes.
            check_models(stage_models.get(stage, []), f"ConvexAdam {stage} stage", dense=True)
            for spec in stage_models[stage]:
                if spec.voxel_size is not None and len(spec.voxel_size) != DIM:
                    raise ValueError(
                        f"ConvexAdam: model '{spec.ref}' has voxel_size {spec.voxel_size}; give {DIM} "
                        "values, or leave it out for the image as it is."
                    )
        for name, value, low in (
            ("grid_spacing", grid_spacing, 1),
            ("displacement_half_width", displacement_half_width, 1),
            ("grid_shrink", grid_shrink, 1),
            ("iterations", iterations, 0),
            ("control_grid_smoothing", control_grid_smoothing, 0),
            ("linear_iterations", linear_iterations, 0),
        ):
            if value < low:
                # itk-impact takes these unsigned and divides by the spacing and the shrink factor.
                raise ValueError(f"{name} must be at least {low}, got {value}.")
        if not 0 < linear_sampling <= 1:
            raise ValueError(f"linear_sampling is a share of the voxels, in (0, 1]: got {linear_sampling}.")
        if not 0 < voxel_sampling <= 1:
            raise ValueError(f"voxel_sampling is a share of the voxels, in (0, 1]: got {voxel_sampling}.")
        if voxel_sampling < 1 and "fine" in stages:
            # itk-impact refuses it too, but only once the case has been read and the models loaded.
            lncc = [spec.ref for spec in stage_models["fine"] if spec.distance == "LNCC"]
            if lncc:
                raise ValueError(f"voxel_sampling draws points, and LNCC ({lncc}) correlates windows of whole maps.")
        if mode not in ("Static", "Jacobian"):
            raise ValueError(f"mode must be 'Static' or 'Jacobian', got '{mode}'.")
        if lncc_kernel < 1 or lncc_kernel % 2 == 0:
            raise ValueError(f"lncc_kernel must be odd, got {lncc_kernel}: an even window has no centre voxel.")
        self._stages = stages
        self._stage_models = {stage: stage_models[stage] for stage in stages}
        # Each model fetched and shaped by the registry. The fine stage in Jacobian mode differentiates the metric
        # through its networks: probed at the first run, so a layer without a gradient is refused there.
        self._feature_models = {
            stage: [feature_model(spec) for spec in specs] for stage, specs in self._stage_models.items()
        }
        self._unchecked = self._feature_models["fine"] if mode == "Jacobian" and "fine" in stages else []
        # The patch each model runs on: the whole image (0), or, sampled in Jacobian mode, the receptive field around
        # every drawn point, as elastix's metric does (itk-impact's PatchSize).
        self._patches = {stage: [0] * len(specs) for stage, specs in self._stage_models.items()}
        if mode == "Jacobian" and voxel_sampling < 1 and "fine" in stages:
            for index, (spec, model) in enumerate(
                zip(self._stage_models["fine"], self._feature_models["fine"], strict=True)
            ):
                try:
                    self._patches["fine"][index] = model.receptive_field
                except MeasureError as error:
                    raise ValueError(
                        f"voxel_sampling in Jacobian mode runs '{spec.ref}' on the patch of its receptive field "
                        f"around each drawn point, which cannot be sized: {error.args[0]} Use mode Static, or "
                        "voxel_sampling 1."
                    ) from error
        # Built lazily and cached per stage: constructing a model configuration loads the TorchScript model from disk
        # in C++, so each stage's list is built once and reused for every case.
        self._configurations: dict[str, list] = {}
        self._mixed_precision = mixed_precision
        self._grid_spacing = grid_spacing
        self._displacement_half_width = displacement_half_width
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._regularization_weight = regularization_weight
        self._grid_shrink = grid_shrink
        self._control_grid_smoothing = control_grid_smoothing
        self._linear = linear
        self._linear_iterations = linear_iterations
        self._linear_sampling = linear_sampling
        self._voxel_sampling = voxel_sampling
        self._seed = seed
        self._mode = mode
        self._normalize = normalize
        self._feature_map_update_interval = feature_map_update_interval
        self._lncc_kernel = lncc_kernel
        self._balance_coarse_layers = balance_coarse_layers

    def _model_configurations(self, stage: str) -> list:
        """One model configuration per feature model of ``stage``, built once and reused across cases.

        Constructing one loads the TorchScript module from disk on the C++ side; the coarse and fine filters copy
        each configuration by value in ``AddModelConfiguration``, the copy sharing the loaded module. A model
        without voxel_size gets 0 there, which itk-impact reads as the image as it is. Its patch is the whole image
        (0), or, in the sampled Jacobian mode of the fine stage, its receptive field.
        """
        if stage not in self._configurations:
            configurations = []
            for spec, model, patch in zip(
                self._stage_models[stage], self._feature_models[stage], self._patches[stage], strict=True
            ):
                # Patch size 0 is the whole image in one piece: the overlap only places and blends patches.
                voxel = ([float(v) for v in spec.voxel_size] if spec.voxel_size is not None else None) or [0.0] * DIM
                configuration = itk.ImpactModelConfiguration(
                    model.model_path,
                    model.dim,
                    model.in_channels,
                    [patch] * model.dim,
                    voxel,
                    [0] * model.dim,
                    [bit == "1" for bit in spec.layers_mask],
                    self._mixed_precision,
                )
                configuration.SetFeatureNormalization(spec.feature_normalization)
                configurations.append(configuration)
            self._configurations[stage] = configurations
        return self._configurations[stage]

    def _linear_align(
        self,
        fixed: "itk.Image",
        moving: "itk.Image",
        fixed_mask: "itk.Image | None",
        moving_mask: "itk.Image | None",
        fixed_centre: tuple[float, ...],
        moving_centre: tuple[float, ...],
    ) -> "itk.AffineTransform":
        """Affine (Mattes MI) mapping fixed -> moving physical points, seeded by the translation between the
        foreground centres (``_foreground_centre``) or by none, whichever matches better, and centred on the fixed
        one; the metric counts only the points the masks hold, when there are masks."""
        # The scales estimator samples the image at random through ITK's global Mersenne Twister, which ITK seeds from
        # the clock in every process: seeded, two runs give the same affine.
        itk.MersenneTwisterRandomVariateGenerator.GetInstance().SetSeed(self._seed)
        levels = 3
        metric_type = itk.MattesMutualInformationImageToImageMetricv4[_IMAGE_F, _IMAGE_F]

        def new_metric():
            metric = metric_type.New()
            metric.SetNumberOfHistogramBins(32)
            # The registration method reads the metric's masks, for the points it samples too.
            if fixed_mask is not None:
                metric.SetFixedImageMask(itk.ImageMaskSpatialObject[DIM].New(Image=fixed_mask))
            if moving_mask is not None:
                metric.SetMovingImageMask(itk.ImageMaskSpatialObject[DIM].New(Image=moving_mask))
            return metric

        def cost(translation: list[float]) -> float:
            metric = new_metric()
            shift = itk.TranslationTransform[itk.D, DIM].New()
            shift.SetOffset(translation)
            metric.SetFixedImage(fixed)
            metric.SetMovingImage(moving)
            metric.SetMovingTransform(shift)
            try:
                metric.Initialize()
                return float(metric.GetValue())
            except RuntimeError:  # too few points overlap the moving image
                return float("inf")

        # The centres' translation is only a guess: two scans that cover different lengths of the body have their
        # foreground centres apart where the anatomy is not. The seed is the one of it and the images' own placement
        # that matches better.
        seed = min(([float(m - f) for m, f in zip(moving_centre, fixed_centre, strict=True)], [0.0] * DIM), key=cost)
        affine = itk.AffineTransform[itk.D, DIM].New()
        affine.SetCenter([float(v) for v in fixed_centre])
        affine.SetTranslation(seed)
        metric = new_metric()
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
        if self._linear_sampling < 1:
            # Every voxel at every iteration is most of a large pair's linear stage.
            registration.SetMetricSamplingStrategy(itk.ImageRegistrationMethodv4Enums.MetricSamplingStrategy_RANDOM)
            registration.SetMetricSamplingPercentage(self._linear_sampling)
            registration.MetricSamplingReinitializeSeed(self._seed)
        registration.InPlaceOn()
        registration.Update()
        return affine

    @staticmethod
    def _restrict(stage, fixed_mask: "itk.Image | None", moving_mask: "itk.Image | None") -> None:
        """Hand ``stage`` the masks there are, each on its image's grid: without one, the filter runs unmasked."""
        if fixed_mask is not None:
            stage.SetFixedMask(fixed_mask)
        if moving_mask is not None:
            stage.SetMovingMask(moving_mask)

    def _coarse(
        self,
        fixed: "itk.Image",
        moving: "itk.Image",
        fixed_mask: "itk.Image | None",
        moving_mask: "itk.Image | None",
        device: str,
    ) -> "itk.Image":
        """ConvexAdam coarse coupled-convex initializer -> robust low-resolution field on the fixed grid."""
        coarse = _coarse_registration_type().New()
        coarse.SetFixedImage(fixed)
        coarse.SetMovingImage(moving)
        self._restrict(coarse, fixed_mask, moving_mask)
        specs = self._stage_models["coarse"]
        for configuration in self._model_configurations("coarse"):
            coarse.AddModelConfiguration(configuration)
        # Each layer's distance, the weights, the PCA and the channel subset reach it as they reach the fine stage; the
        # coarse stage takes a distance over each cell's window (NCC and LNCC alike, lncc_kernel being the fine stage's).
        coarse.SetDistance(per_kept_layer(specs, lambda spec: spec.distance))
        coarse.SetLayersWeight(layer_weights(specs))
        coarse.SetPCA(per_kept_layer(specs, lambda spec: int(spec.pca)))
        coarse.SetSubsetFeatures(per_kept_layer(specs, lambda spec: int(spec.subset_features)))
        # The coarse cost is never divided by its value at zero displacement: the coupling schedule's coefficients are
        # absolute. It stays raw, its layers weighed by layers_weight, or, with balance_coarse_layers, by their spread
        # over the candidates, their total spread the raw one; `normalize` is the fine stage's.
        coarse.SetNormalizeLosses(False)
        coarse.SetBalanceLosses(self._balance_coarse_layers)
        coarse.SetGridSpacing(self._grid_spacing)
        coarse.SetDisplacementHalfWidth(self._displacement_half_width)
        coarse.SetDevice(device)
        coarse.SetSeed(self._seed)
        coarse.Update()
        field = coarse.GetOutput()
        field.DisconnectPipeline()
        return field

    def _fine(
        self,
        fixed: "itk.Image",
        moving: "itk.Image",
        fixed_mask: "itk.Image | None",
        moving_mask: "itk.Image | None",
        initial_field: "itk.Image | None",
        device: str,
    ) -> "itk.Image":
        """Adam instance-optimisation refinement, warm-started from ``initial_field`` (zero if none)."""
        fine = _fine_registration_type().New()
        fine.SetFixedImage(fixed)
        fine.SetMovingImage(moving)
        self._restrict(fine, fixed_mask, moving_mask)
        fine.SetInitialDisplacementField(initial_field if initial_field is not None else self._zero_field(fixed))
        specs = self._stage_models["fine"]
        for configuration in self._model_configurations("fine"):
            fine.AddModelConfiguration(configuration)
        fine.SetDistance(per_kept_layer(specs, lambda spec: spec.distance))
        fine.SetLayersWeight(layer_weights(specs))
        fine.SetSubsetFeatures(per_kept_layer(specs, lambda spec: int(spec.subset_features)))
        fine.SetPCA(per_kept_layer(specs, lambda spec: int(spec.pca)))
        fine.SetNormalizeLosses(self._normalize)
        fine.SetMode(self._mode)
        fine.SetFeatureMapUpdateInterval(self._feature_map_update_interval)
        fine.SetLNCCKernel(self._lncc_kernel)
        fine.SetSamplingPercentage(self._voxel_sampling)
        fine.SetNumberOfIterations(self._iterations)
        fine.SetLearningRate(self._learning_rate)
        fine.SetRegularizationWeight(self._regularization_weight)
        fine.SetGridShrinkFactor(self._grid_shrink)
        fine.SetControlGridSmoothingIterations(self._control_grid_smoothing)
        fine.SetDevice(device)
        fine.SetSeed(self._seed)

        # Optional terminal progress over the Adam iterations, driven from the metric trace. ``disable=None``
        # auto-hides it when stderr is not a TTY (e.g. under KonfAI/Slicer, where the outer "Prediction" bar
        # already reports progress), so captured logs stay clean; ``leave=False`` avoids stacking one bar per
        # patch. The observer is best-effort: if the filter emits no IterationEvent the bar just fills at the end.
        progress = tqdm.tqdm(total=self._iterations or None, desc="Registration", ncols=0, leave=False, disable=None)

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
        try:
            fine.Update()
        finally:
            # The observer's closure holds ``fine`` and ``fine`` holds the observer: a cycle through C++ that
            # Python's collector cannot see.
            fine.RemoveAllObservers()
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

    def _run_stages(
        self,
        fixed: "itk.Image",
        moving: "itk.Image",
        fixed_mask: "itk.Image | None",
        moving_mask: "itk.Image | None",
        device: str,
    ) -> "itk.Image | None":
        """Run the configured coarse/fine chain; each fine warm-starts from the running field.

        ``coarse`` produces a field from scratch; ``fine`` refines the running field. So ``['coarse']`` is a
        coarse-only app, ``['fine']`` a fine-only app (zero warm-start), and ``['coarse', 'fine']`` chains both
        (the composite). Returns None when no deformable stage runs (e.g. a linear-only chain).
        """
        field: itk.Image | None = None
        for stage in self._stages:
            field = (
                self._coarse(fixed, moving, fixed_mask, moving_mask, device)
                if stage == "coarse"
                else self._fine(fixed, moving, fixed_mask, moving_mask, field, device)
            )
        return field

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> np.ndarray:
        """Register ``moving`` onto ``fixed``; return the displacement field, channel-first, on the fixed grid.

        ``fixed_mask`` and ``moving_mask`` (each on its image's grid, or resampled onto it by nearest neighbour)
        restrict every stage's similarity to where they are not 0, the linear stage and its seed included."""
        if fixed_mask is not None and not sitk.GetArrayViewFromImage(fixed_mask).any():
            # A fixed mask with no voxel in it leaves nothing to register: a zero field, as the elastix engine
            # returns (in a tiled run, every tile the tissue does not reach).
            return np.zeros((DIM, *fixed.GetSize()[::-1]), dtype=np.float32)
        # Only a mask that restricts (some voxels in, some out) reaches the filters, decided on the mask as given:
        # konfai-apps writes an all-ones default on the FIXED grid for both sides.
        fixed_mask = _binary_mask(fixed_mask, fixed) if is_partial_mask(fixed_mask) else None
        moving_mask = _binary_mask(moving_mask, moving) if is_partial_mask(moving_mask) else None
        if fixed_mask is not None and moving_mask is None:
            # As FireANTs gives the side without a mask a whole-image one: warped with the moving image, it drops the
            # fixed voxels the field sends out of it, as elastix does too.
            moving_mask = sitk.Image(moving.GetSize(), sitk.sitkUInt8) + 1
            moving_mask.CopyInformation(moving)
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        for model in self._unchecked:  # before any compute, as the other engines refuse it
            model.check(gradient=True)
        self._unchecked = []
        # Feature networks compute their channels along the voxel axes they are given: both images go in with their
        # voxel axes in LPS order, as in the elastix and FireANTs engines, each mask with its image. Nothing is
        # resampled, so the physical space and every transform stay the same, and the field is sampled on the fixed
        # image as it came.
        grid = fixed
        fixed, moving, fixed_mask, moving_mask = world_aligned_pair(fixed, moving, fixed_mask, moving_mask)
        reoriented = fixed is not grid
        # KonfAI sizes SimpleITK's thread pool to this rank's share of the cores (apply_cpu_thread_budget); ITK
        # Python keeps a pool of its own, which would take every core in every rank.
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
        # Every stage reads the images winsorised: the linear stage's mutual information and Otsu seed bin the image's
        # range, and MIND, the presets' feature model, divides by it (minimum to maximum, its variance floored at 1e-6).
        fixed, moving = winsorized(fixed), winsorized(moving)
        fixed_itk = _sitk_to_itk(fixed)
        moving_itk = _sitk_to_itk(moving)
        fixed_mask_itk = None if fixed_mask is None else _sitk_to_itk(fixed_mask, np.uint8)
        moving_mask_itk = None if moving_mask is None else _sitk_to_itk(moving_mask, np.uint8)
        if not self._linear:
            # The coarse and fine filters bring the moving image and its mask onto the fixed grid themselves when the
            # grids differ (the identity resample this did too), and their field is the answer: nothing to compose it
            # with.
            field = self._run_stages(fixed_itk, moving_itk, fixed_mask_itk, moving_mask_itk, device)
            if not reoriented:
                return np.array(np.moveaxis(itk.array_view_from_image(field), -1, 0), order="C")
            aligned = _itk_field_to_sitk_transform(field, fixed)
            return image_to_data(
                sitk.TransformToDisplacementField(
                    aligned,
                    sitk.sitkVectorFloat32,
                    grid.GetSize(),
                    grid.GetOrigin(),
                    grid.GetSpacing(),
                    grid.GetDirection(),
                )
            )[0]

        # Linear pre-align, masked as the deformable stages: resample the moving onto the fixed grid so the deformable
        # stage starts close, and its mask with it, by nearest neighbour.
        affine = self._linear_align(
            fixed_itk,
            moving_itk,
            fixed_mask_itk,
            moving_mask_itk,
            _foreground_centre(fixed, fixed_mask),
            _foreground_centre(moving, moving_mask),
        )
        moving_linear = _resampled(moving_itk, fixed_itk, affine, itk.LinearInterpolateImageFunction)
        if moving_mask_itk is not None:
            moving_mask_itk = _resampled(
                moving_mask_itk, fixed_itk, affine, itk.NearestNeighborInterpolateImageFunction
            )

        field = self._run_stages(fixed_itk, moving_linear, fixed_mask_itk, moving_mask_itk, device)

        # One transform on the fixed grid = affine then deformable, so the returned DVF/transform warps the
        # ORIGINAL moving. SimpleITK applies the last-added transform first, so [affine, deformable] gives
        # moved(p) = moving(affine(deformable(p))). A linear-only chain (field is None) yields the affine alone.
        chain = [_itk_affine_to_sitk(affine)]
        if field is not None:
            chain.append(_itk_field_to_sitk_transform(field, fixed))
        composite = sitk.CompositeTransform(chain)
        dvf = sitk.TransformToDisplacementField(
            composite,
            sitk.sitkVectorFloat32,
            grid.GetSize(),
            grid.GetOrigin(),
            grid.GetSpacing(),
            grid.GetDirection(),
        )
        dvf_np, _ = image_to_data(dvf)
        return dvf_np


class RegistrationNet(network.Network):
    """Pairwise ConvexAdam registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1, fixed mask = 2,
    moving mask = 3; masks restrict every stage's similarity, whole-image = no restriction).

    Output on the fixed grid: ``DisplacementField`` (the
    DIM-component displacement field, in mm). Geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``.

    ConvexAdam registers on the fixed image's own grid. Its sizes are counted in voxels of the fixed image's finest
    axis, s_min, and derived per axis, so that they are (nearly) isotropic in millimetres: grid_spacing,
    displacement_half_width, grid_shrink, the step learning_rate sets and the gradient regularization_weight
    penalises, and lncc_kernel (along each compared map's finest axis, the nearest odd count of the same length along
    the others). The coarse stage's cost smoothing and its NCC/LNCC window are counted in its cells,
    control_grid_smoothing in control cells. On an isotropic image s_min is its voxel: the presets' +/- 24 s_min are 24 mm at 1 mm, and at 0.8 x 0.8 x 3 mm 19.2 mm in-plane and 24 mm along z,
    whole cells rounding the range up.
    """

    #: Its output is a displacement field in world units: KonfAI blends it in float32, not float16.
    full_precision_outputs = True

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        models: dict[str, ModelSpec] = {},
        levels: Annotated[
            dict[str, LevelSpec],
            "The models of each stage ('0', '1', ...) in 'stages' order (coarse, then fine), in place of 'models' "
            "there; empty = 'models' in every stage.",
        ] = {},
        mode: Mode = "Static",
        normalize: Normalize = True,
        feature_map_update_interval: FeatureMapUpdateInterval = -1,
        lncc_kernel: LNCCKernel = 5,
        overlap: Annotated[
            int | None,
            "Has no effect: the feature model runs on the whole image in one piece, with no patch to overlap. "
            "Accepted so that a configuration setting it still loads.",
        ] = None,
        mixed_precision: MixedPrecision = False,
        grid_spacing: Annotated[
            int,
            Range(1, 512),
            "Coarse stage: the cell size of the grid its discrete search runs on (one displacement per cell, then "
            "upsampled), in voxels of the fixed image's finest axis (s_min): along each axis a cell spans the whole "
            "number of voxels nearest grid_spacing x s_min mm. Smaller = a finer coarse field and far more memory: "
            "(2 x half-width + 1)^3 / grid_spacing^3 floats per voxel on an isotropic image, more along a thick axis, "
            "where a cell spans fewer voxels (3x at 0.8 x 0.8 x 3 mm).",
        ] = 6,
        displacement_half_width: Annotated[
            int,
            Range(1, 512),
            "Coarse stage: half-width of the discrete search in cells of the finest axis, so it captures at least "
            "displacement_half_width x grid_spacing x s_min mm each way, rounded up to whole cells along each axis "
            "(24 mm with the presets' 4 x 6 at 1 mm). Raise it for larger motion; memory grows as "
            "(2 x half-width + 1)^3.",
        ] = 4,
        balance_coarse_layers: Annotated[
            bool,
            "Coarse stage: divide each layer by its spread over the candidate displacements (the mean over the cells "
            "of its cost's range, measured on the first cost volume), times layers_weight, so that every layer moves "
            "the search alike, their total spread staying the raw one the coupling is calibrated on; a layer that "
            "does not vary weighs 0 (off = the raw costs, weighed by layers_weight alone). Experimental.",
        ] = False,
        iterations: Annotated[
            int,
            Range(0, 100000),
            "Fine stage: Adam steps; higher = more converged, slower.",
        ] = 80,
        learning_rate: Annotated[
            float,
            Range(0.0, 100.0),
            "Fine stage: Adam step size, in s_min (the fixed image's finest voxel side) per step along every axis (the "
            "control grid holds full-resolution displacements: ConvexAdam's step 1 on its half-resolution grid is 2 "
            "here). Higher converges faster but can oscillate.",
        ] = 2.0,
        regularization_weight: Annotated[
            float,
            Range(0.0, 1000.0),
            "Fine stage: weight of the diffusion regulariser, the squared gradient of the smoothed control grid in mm "
            "per mm (voxels per voxel on an isotropic image, as ConvexAdam's lambda); higher = smoother, lower = more "
            "flexible but risks folding.",
        ] = 1.25,
        grid_shrink: Annotated[
            int,
            Range(1, 128),
            "Fine stage: the spacing of the Adam control grid, in voxels of the fixed image's finest axis (s_min): "
            "along each axis, the whole number of voxels nearest grid_shrink x s_min mm (ConvexAdam's grid_sp_adam, "
            "2); higher = a coarser, smoother, faster refinement.",
        ] = 2,
        control_grid_smoothing: Annotated[
            int,
            Range(0, 8),
            "Fine stage: 3x3x3 average-pool passes over the control grid at every Adam iteration, in control cells, "
            "which is what keeps the field smooth while it moves: ConvexAdam applies three, and the same smoothing "
            "shapes the regulariser. 0 optimises the grid unsmoothed.",
        ] = 3,
        stages: Annotated[
            list[str],
            "Stages to run: 'coarse' (discrete coupled-convex search, from scratch) then 'fine' (Adam "
            "refinement of the running field, or of zero when it runs alone); drop 'fine' for a fast coarse "
            "field. A 'coarse' after a 'fine' is refused: it would throw the refinement away.",
        ] = ["coarse", "fine"],
        linear: Annotated[
            bool,
            "Run an affine pre-alignment (Mattes MI on the CPU, seeded by the foreground centres) before the "
            "deformable stages, which then refine the pair it brings onto the fixed grid; recommended unless the "
            "pair already overlaps well.",
        ] = True,
        linear_iterations: Annotated[
            int,
            Range(0, 100000),
            "Most iterations of the affine pre-alignment at each of its three resolution levels.",
        ] = 200,
        linear_sampling: Annotated[
            float,
            Range(0.01, 1.0),
            "Share of the voxels the affine pre-alignment's mutual information reads at each iteration, drawn at "
            "random with 'seed' (1 = every voxel): the pre-alignment reads the whole image otherwise, the most of "
            "the registration's time on a large CT.",
        ] = 1.0,
        voxel_sampling: Annotated[
            float,
            Range(0.001, 1.0),
            "Fine stage: share of the voxels the similarity reads at each Adam iteration, drawn anew at random with "
            "'seed' (1 = every voxel). Static compares the features at those points only; Jacobian runs each network on "
            "the patch of its receptive field around each of them, as elastix does. Point-wise distances only (not "
            "LNCC). Experimental.",
        ] = 1.0,
        seed: Annotated[
            int,
            "Seed of torch's generator inside itk-impact and of ITK's, which the affine pre-alignment draws its "
            "samples from. A GPU run is not bit-reproducible whatever the seed (grid_sample's backward accumulates "
            "in any order).",
        ] = 42,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=3,
        )
        per_stage = level_models(models, levels, len(stages), "ConvexAdam")
        engine = ConvexAdamEngine(
            dict(zip(stages, per_stage, strict=True)),
            mixed_precision,
            grid_spacing,
            displacement_half_width,
            iterations,
            learning_rate,
            regularization_weight,
            grid_shrink,
            control_grid_smoothing,
            stages,
            linear,
            linear_iterations,
            seed,
            linear_sampling,
            mode,
            normalize,
            feature_map_update_interval,
            lncc_kernel,
            voxel_sampling,
            balance_coarse_layers,
        )
        self.add_module(
            "Registration",
            EngineRegistration(engine, fuse_texpr=False),
            in_branch=[0, 1, 2, 3],
            out_branch=["registration"],
        )
        # The output module the presets name.
        self.add_module("DisplacementField", torch.nn.Identity(), in_branch=["registration"], out_branch=["dvf"])
