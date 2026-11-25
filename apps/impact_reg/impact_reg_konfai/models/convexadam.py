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

    ImpactCoarseRegistration                   [coupled-convex init, IMPACT features]
      -> ImpactFineRegistration                [Adam instance optimisation, IMPACT features]

The IMPACT feature models (e.g. MIND) are TorchScript ``.pt`` files fetched from Hugging Face
and wrapped as ``itk.ImpactModelConfiguration``: the same models the elastix and FireANTs presets use, compared by
the IMPACT loss every engine shares (``impact_loss.py``: models, distances, normalization, Static and Jacobian),
which needs itk-impact 0.1.6 or later.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on
runtime-evaluated annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import itk
import numpy as np
import SimpleITK as sitk
import torch
import tqdm
from konfai.network import network
from konfai.utils.config import Config, Range
from konfai.utils.dataset import image_to_data
from konfai.utils.errors import ConfigError, MeasureError

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
    plane_model,
)
from .intensity import EngineRegistration, center_of_mass_translation, is_partial_mask
from .orientation import world_aligned_pair

DIM = 3


# A UI reads the tuning knobs straight from the TYPES on ``RegistrationNet.__init__`` and ``ModelSpec``:
# ``Annotated[.., Range]`` gives numeric spin bounds; ``Literal`` / ``Annotated[str, Choices]`` a dropdown.
# ``models`` is a dict-of-objects (one ``ModelSpec`` per feature model): the same shape as the elastix presets,
# so SlicerKonfAI renders each model as a repeatable block with a ``ref`` / ``distance`` combo box.
def _image_f(dimension: int):
    return itk.Image[itk.F, dimension]


#: Bytes of the dense field an ITK streaming writer asks for at once (a slab of whole planes, one at least).
_SLAB_BYTES = 64 << 20


def _coarse_registration_type(dimension: int = DIM):
    """The coupled-convex initializer (itk-impact's ImpactCoarseRegistration)."""
    return itk.ImpactCoarseRegistration[_image_f(dimension), _image_f(dimension)]


def _fine_registration_type(dimension: int = DIM):
    """The Adam instance-optimisation stage (itk-impact's ImpactFineRegistration)."""
    return itk.ImpactFineRegistration[_image_f(dimension), _image_f(dimension)]


def _sitk_to_itk(image: sitk.Image, pixel: type = np.float32) -> "itk.Image":
    """Copy a scalar SimpleITK image (with its geometry) into an ``itk.Image[F, dimension]`` (``itk.Image[UC,
    dimension]`` for ``np.uint8``)."""
    dimension = image.GetDimension()
    itk_image = itk.image_from_array(sitk.GetArrayFromImage(image).astype(pixel))
    itk_image.SetOrigin([float(v) for v in image.GetOrigin()])
    itk_image.SetSpacing([float(v) for v in image.GetSpacing()])
    itk_image.SetDirection(
        itk.matrix_from_array(np.asarray(image.GetDirection(), dtype=float).reshape(dimension, dimension))
    )
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


def _translated(image: sitk.Image, shift: np.ndarray) -> sitk.Image:
    """``image`` placed ``shift`` (physical) back: ``translated(p) = image(p + shift)``, its voxels as they are."""
    moved = sitk.Image(image)
    moved.SetOrigin([float(v) for v in np.subtract(image.GetOrigin(), shift)])
    return moved


def _itk_impact_predates_its_stages() -> bool:
    """Whether the itk-impact imported is older than 0.1.6, the first whose stages take what this module sets, read
    off its symbols: 0.1.5 has ``ImpactModelConfiguration`` already and no ``SetSamplingPercentage``. A source build
    carries no distribution to read a version from, and a distribution may name another install than the one
    imported."""
    if not hasattr(itk, "ImpactModelConfiguration") or not hasattr(itk, "ImpactFineRegistration"):
        return True
    return not hasattr(_fine_registration_type(DIM).New(), "SetSamplingPercentage")


#: The keys of the affine pre-alignment ConvexAdam ran before ``moments_init``: refused by name, never bound.
_LINEAR_KEYS = ("linear", "linear_iterations", "linear_sampling")


def refuse_linear_keys(block) -> None:
    """Refuse a model block that still sets ConvexAdam's former affine pre-alignment: read as unknown keys, the preset
    would register without it a pair it used to bring within the coarse search."""
    given = [key for key in _LINEAR_KEYS if key in block]
    if given:
        raise ConfigError(
            f"ConvexAdam has no affine pre-alignment any more ({', '.join(given)}): the stages read the pair as it"
            " comes.",
            "Remove them. For a pair the coarse search does not reach, moments_init: com (the centres of mass) or cof"
            " (the frames' centres) starts it from the translation between the centres.",
        )


def _itk_field_to_sitk_transform(field: "itk.Image", reference: sitk.Image) -> sitk.Transform:
    """Wrap an itk displacement field (on the fixed grid) as a SimpleITK ``DisplacementFieldTransform``, which
    only takes a float64 field."""
    sitk_field = sitk.GetImageFromArray(itk.array_view_from_image(field).astype(np.float64), isVector=True)
    sitk_field.CopyInformation(reference)
    return sitk.DisplacementFieldTransform(sitk_field)


class ConvexAdamEngine:
    """Register a fixed/moving pair with the itk-impact ConvexAdam pipeline; return the displacement field on the fixed grid.

    The IMPACT feature models are fetched and probed once, by KonfAI (``repo:filename`` on Hugging Face), and reused
    across cases.
    Masks restrict the stages' similarity to where they are not 0, as in the FireANTs engine, unless
    deformable_masked is off; a fixed mask with no voxel in it (a tile the tissue does not reach) gives a zero field when
    the stages read the masks, and is ignored with the image when they do not.
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
        seed: int,
        mode: str = "Static",
        normalize: bool = True,
        feature_map_update_interval: int = -1,
        lncc_kernel: int = 5,
        voxel_sampling: float = 1.0,
        balance_coarse_layers: bool = False,
        deformable_masked: bool = True,
        moments_init: str = "none",
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
        if not stages:
            raise ValueError("stages=[] leaves nothing to register.")
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
        ):
            if value < low:
                # itk-impact takes these unsigned and divides by the spacing and the shrink factor.
                raise ValueError(f"{name} must be at least {low}, got {value}.")
        if not 0 < voxel_sampling <= 1:
            raise ValueError(f"voxel_sampling is a share of the voxels, in (0, 1]: got {voxel_sampling}.")
        if voxel_sampling < 1 and "fine" in stages:
            # itk-impact refuses it too, but only once the case has been read and the models loaded.
            lncc = [spec.ref for spec in stage_models["fine"] if spec.distance == "LNCC"]
            if lncc:
                raise ValueError(f"voxel_sampling draws points, and LNCC ({lncc}) correlates windows of whole maps.")
        if mode not in ("Static", "Jacobian"):
            raise ValueError(f"mode must be 'Static' or 'Jacobian', got '{mode}'.")
        if moments_init not in ("cof", "com", "none"):
            raise ValueError(f"moments_init must be 'cof', 'com' or 'none', got '{moments_init}'.")
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
        self._configurations: dict[tuple[str, int], list] = {}
        self._mixed_precision = mixed_precision
        self._grid_spacing = grid_spacing
        self._displacement_half_width = displacement_half_width
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._regularization_weight = regularization_weight
        self._grid_shrink = grid_shrink
        self._control_grid_smoothing = control_grid_smoothing
        self._voxel_sampling = voxel_sampling
        self._seed = seed
        self._mode = mode
        self._normalize = normalize
        self._feature_map_update_interval = feature_map_update_interval
        self._lncc_kernel = lncc_kernel
        self._balance_coarse_layers = balance_coarse_layers
        self._deformable_masked = deformable_masked
        self._moments_init = moments_init

    def check_gradients(self) -> None:
        """Probe the fine stage's networks once for a gradient through every weighted layer, which Jacobian mode
        differentiates the metric through: before any compute, as the other engines refuse a layer without one."""
        for model in self._unchecked:
            model.check(gradient=True)
        self._unchecked = []

    def _model_configurations(self, stage: str, dimension: int) -> list:
        """One model configuration per feature model of ``stage``, built once and reused across cases.

        Constructing one loads the TorchScript module from disk on the C++ side; the coarse and fine filters copy
        each configuration by value in ``AddModelConfiguration``, the copy sharing the loaded module. A model
        without voxel_size gets 0 there, which itk-impact reads as the image as it is. Its patch is the whole image
        (0), or, in the sampled Jacobian mode of the fine stage, its receptive field. A 2-D image takes a 3-D model's
        2-D version, and the first values of its voxel_size.
        """
        if (stage, dimension) not in self._configurations:
            configurations = []
            for spec, model, patch in zip(
                self._stage_models[stage], self._feature_models[stage], self._patches[stage], strict=True
            ):
                if model.dim > dimension:
                    model = feature_model(plane_model(spec))
                    model.dim = dimension  # its 2-D version, which a local file's absent registry entry calls 3-D
                # Patch size 0 is the whole image in one piece: the overlap only places and blends patches.
                voxel = ([float(v) for v in spec.voxel_size[:dimension]] if spec.voxel_size is not None else None) or [
                    0.0
                ] * dimension
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
            self._configurations[stage, dimension] = configurations
        return self._configurations[stage, dimension]

    @staticmethod
    def _restrict(stage, fixed_mask: "itk.Image | None", moving_mask: "itk.Image | None") -> None:
        """Hand ``stage`` the masks there are, each on its image's grid: without one, the filter runs unmasked."""
        if fixed_mask is not None:
            stage.SetFixedMask(fixed_mask)
        if moving_mask is not None:
            stage.SetMovingMask(moving_mask)

    def _new_coarse(self, fixed, moving, fixed_mask, moving_mask, device: str):
        """The coarse stage's filter, set as the engine says (both routes run it)."""
        dimension = fixed.GetImageDimension()
        coarse = _coarse_registration_type(dimension).New()
        coarse.SetFixedImage(fixed)
        coarse.SetMovingImage(moving)
        self._restrict(coarse, fixed_mask, moving_mask)
        specs = self._stage_models["coarse"]
        for configuration in self._model_configurations("coarse", dimension):
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
        return coarse

    def _coarse(
        self,
        fixed: "itk.Image",
        moving: "itk.Image",
        fixed_mask: "itk.Image | None",
        moving_mask: "itk.Image | None",
        device: str,
    ) -> "itk.Image":
        """ConvexAdam coarse coupled-convex initializer -> robust low-resolution field on the fixed grid."""
        coarse = self._new_coarse(fixed, moving, fixed_mask, moving_mask, device)
        coarse.Update()
        field = coarse.GetOutput()
        field.DisconnectPipeline()
        return field

    def _new_fine(self, fixed, moving, fixed_mask, moving_mask, device: str):
        """The fine stage's filter, set as the engine says, but for its start and its iterations (both routes run
        it)."""
        dimension = fixed.GetImageDimension()
        fine = _fine_registration_type(dimension).New()
        fine.SetFixedImage(fixed)
        fine.SetMovingImage(moving)
        self._restrict(fine, fixed_mask, moving_mask)
        specs = self._stage_models["fine"]
        for configuration in self._model_configurations("fine", dimension):
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
        fine.SetLearningRate(self._learning_rate)
        fine.SetRegularizationWeight(self._regularization_weight)
        fine.SetGridShrinkFactor(self._grid_shrink)
        fine.SetControlGridSmoothingIterations(self._control_grid_smoothing)
        fine.SetDevice(device)
        fine.SetSeed(self._seed)
        return fine

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
        fine = self._new_fine(fixed, moving, fixed_mask, moving_mask, device)
        fine.SetInitialDisplacementField(initial_field if initial_field is not None else self._zero_field(fixed))
        # ConvexAdam returns the field its last iteration evaluates, before that iteration's step.
        fine.SetNumberOfIterations(max(self._iterations - 1, 0))

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
        dimension = reference.GetImageDimension()
        field = itk.Image[itk.Vector[itk.F, dimension], dimension].New()
        field.CopyInformation(reference)
        field.SetRegions(reference.GetLargestPossibleRegion())
        field.Allocate()
        zero = itk.Vector[itk.F, dimension]()
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
        (the composite).
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
        restrict the stages' similarity to where they are not 0, unless ``deformable_masked`` is off."""
        if not self._deformable_masked:  # no stage reads them: nothing to prepare
            fixed_mask = moving_mask = None
        if fixed_mask is not None and not sitk.GetArrayViewFromImage(fixed_mask).any():
            # A fixed mask with no voxel in it leaves nothing to register: a zero field, as the elastix engine
            # returns (in a tiled run, every tile the tissue does not reach).
            return np.zeros((fixed.GetDimension(), *fixed.GetSize()[::-1]), dtype=np.float32)
        # Only a mask that restricts (some voxels in, some out) reaches the filters, decided on the mask as given:
        # konfai-apps writes an all-ones default on the FIXED grid for both sides.
        fixed_mask = _binary_mask(fixed_mask, fixed) if is_partial_mask(fixed_mask) else None
        moving_mask = _binary_mask(moving_mask, moving) if is_partial_mask(moving_mask) else None
        if fixed_mask is not None and moving_mask is None:
            # As FireANTs gives the side without a mask a whole-image one: warped with the moving image, it drops the
            # fixed voxels the field sends out of it, as elastix does too.
            moving_mask = sitk.Image(moving.GetSize(), sitk.sitkUInt8) + 1
            moving_mask.CopyInformation(moving)
        # The starting translation, when asked for: the moving image and its mask placed by it (their origin moved,
        # nothing resampled), and the field it leaves added to the stages' field.
        shift = None
        if self._moments_init != "none":
            shift = np.asarray(
                center_of_mass_translation(fixed, moving, fixed_mask, moving_mask, kind=self._moments_init)
            )
            moving = _translated(moving, shift)
            moving_mask = None if moving_mask is None else _translated(moving_mask, shift)
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        self.check_gradients()
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
        # The stages read the images as they are, as ConvexAdam does.
        fixed_itk = _sitk_to_itk(fixed)
        moving_itk = _sitk_to_itk(moving)
        fixed_mask_itk = None if fixed_mask is None else _sitk_to_itk(fixed_mask, np.uint8)
        moving_mask_itk = None if moving_mask is None else _sitk_to_itk(moving_mask, np.uint8)
        # The coarse and fine filters bring the moving image and its mask onto the fixed grid themselves when the
        # grids differ, and their field is the answer.
        field = self._run_stages(fixed_itk, moving_itk, fixed_mask_itk, moving_mask_itk, device)
        if not reoriented:
            dvf = np.array(np.moveaxis(itk.array_view_from_image(field), -1, 0), order="C")
        else:
            aligned = _itk_field_to_sitk_transform(field, fixed)
            dvf = image_to_data(
                sitk.TransformToDisplacementField(
                    aligned,
                    sitk.sitkVectorFloat32,
                    grid.GetSize(),
                    grid.GetOrigin(),
                    grid.GetSpacing(),
                    grid.GetDirection(),
                )
            )[0]
        if shift is not None:
            dvf += shift.astype(np.float32).reshape(-1, *[1] * (dvf.ndim - 1))
        return dvf

    def region_halo(self, stage: str) -> int:
        """The voxels a region of ``stage`` is read with around it: one radius of its models' receptive field, which
        a feature reads around it, and one more for the fine stage in Jacobian mode, whose gradient goes back through
        the network. A model that sees whole images only has no receptive field, and is refused here."""
        radius = max([(model.receptive_field - 1) // 2 for model in self._feature_models[stage]], default=0)
        return 2 * radius if stage == "fine" and self._mode == "Jacobian" else radius

    def refine_by_regions(
        self,
        fixed: Path,
        moving: Path,
        fixed_mask: Path | None,
        moving_mask: Path | None,
        device_index: int,
        region_size: int,
        epochs: int,
        field: Path,
        samples: int | None = None,
    ) -> "RegionRefinement":
        """The engine's stages on the native pair, read by regions (itk-impact's ``RegionSize``): ``fixed`` and
        ``moving`` are files an ITK reader streams (the HDF5 twins of ``native.write_twin``), each mask on its image's
        grid. The coarse stage, when the engine has one, runs by regions first (one pass pools the native features to
        its coarse grid, its cost volume is built by boxes of that grid) and its field starts the fine stage's control
        grid. One pass over the boxes pools the native features to the control grid. The stages take what the device
        holds: the pooled maps on it where they fit its budget, in files beside ``field`` otherwise. Unless
        ``samples`` is given, the fine stage runs the engine's own iterations on them, every cell stepped by bricks
        when they are in files. With ``samples``, an iteration of the fine stage draws boxes at random, that many
        voxels of them, anew at each one, and steps their cells alone, as elastix steps on the points it draws: an
        iteration costs its boxes, whatever the image. A box steps ``epochs`` times on average, and its gradient is
        the whole grid's over its cells, so the field goes where the engine's own iterations take it as they grow.
        The coarse stage is no iteration: one pass, its cost volume built by boxes of a fixed size. The field on the
        control grid is written beside ``field`` (``ControlGrid.mha``); the dense field on the fixed grid is written
        to ``field`` by regions, by an ITK streaming writer. The models' receptive field sets the halo read around a
        box. An engine with the coarse stage alone writes that stage's field, interpolated from its grid region by
        region. Needs itk-impact 0.2 (``SetRegionSize``)."""
        self.check_gradients()
        if not self._deformable_masked:  # no stage reads them
            fixed_mask = moving_mask = None
        require_regions()
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
        readers = []  # ITK images hold their producer weakly: the readers live as long as the filter reads them

        def reader(path: Path, image_type):
            source = itk.ImageFileReader[image_type].New(FileName=str(path))
            source.UseStreamingOn()
            source.UpdateOutputInformation()
            readers.append(source)
            return source.GetOutput()

        from ..native import read_information

        dimension = read_information(fixed).GetDimension()  # the pair's, a plane or a volume
        fixed_image = reader(fixed, _image_f(dimension))
        moving_image = reader(moving, _image_f(dimension))
        mask_type = itk.Image[itk.UC, dimension]
        fixed_mask_image = reader(fixed_mask, mask_type) if fixed_mask is not None else None
        moving_mask_image = reader(moving_mask, mask_type) if moving_mask is not None else None
        if fixed_mask_image is not None and moving_mask_image is None:
            # As a run on the whole pair: a whole-image mask on the moving side, warped with it, drops the fixed
            # voxels the field sends out of the moving image. Ones wherever the moving image is, computed as a box
            # asks for them.
            ones = itk.BinaryThresholdImageFilter[_image_f(dimension), mask_type].New(Input=moving_image, InsideValue=1)
            ones.UpdateOutputInformation()
            readers.append(ones)
            moving_mask_image = ones.GetOutput()
        shift = None
        if self._moments_init != "none":
            # The starting translation, its centres taken by a pass over the twins: the moving image and its mask
            # placed by it (their origin moved as they stream), and the field it leaves added to the written fields.
            from ..native import hdf5_twin, mapped_centre

            twins = [None if path is None else hdf5_twin(path) for path in (fixed, moving, fixed_mask, moving_mask)]
            shift = mapped_centre(twins[1], twins[3], self._moments_init) - mapped_centre(
                twins[0], twins[2], self._moments_init
            )

            def placed(image, image_type):
                change = itk.ChangeInformationImageFilter[image_type].New(Input=image)
                change.SetOutputOrigin([float(v) for v in np.subtract(itk.origin(image), shift)])
                change.ChangeOriginOn()
                change.UpdateOutputInformation()
                readers.append(change)
                return change.GetOutput()

            moving_image = placed(moving_image, _image_f(dimension))
            moving_mask_image = None if moving_mask_image is None else placed(moving_mask_image, mask_type)

        def finish(output, grid) -> Path:
            """A stage's dense field written to ``field`` by regions, by an ITK streaming writer, its field on its own
            grid beside it, from where the filter keeps it (a mapped file past the device's budget), and the starting
            translation added to both."""
            _write_streamed(output, field, fixed_image)
            control_grid = field.with_name("ControlGrid.mha")
            itk.imwrite(grid, str(control_grid))
            if shift is not None:
                from ..native import shift_field

                for path in (field, control_grid):
                    shift_field(path, shift)
            return control_grid

        coarse_grid, coarse_stats = None, None
        if "coarse" in self._stages:
            started = time.monotonic()
            coarse = self._new_coarse(fixed_image, moving_image, fixed_mask_image, moving_mask_image, device)
            coarse.SetRegionSize(int(region_size))
            coarse.SetRegionHalo(self.region_halo("coarse"))
            coarse.SetScratchDirectory(str(field.parent))
            coarse_grid = coarse.GetCoarseGridTransform().GetDisplacementField()  # solves, allocating no dense field
            coarse_stats = (time.monotonic() - started, int(coarse.GetBytesRead()))
            if "fine" not in self._stages:
                # The coarse stage alone: its dense field, interpolated from its grid for the regions the writer asks.
                started = time.monotonic()
                control_grid = finish(coarse.GetOutput(), coarse_grid)
                return RegionRefinement(
                    control_grid=control_grid,
                    field=field,
                    metric_values=[],
                    bytes_read=coarse_stats[1],
                    streamed=(True, True),
                    regions=0,
                    iterations=0,
                    seconds=time.monotonic() - started,
                    device_peak_bytes=None,
                    resident=False,
                    coarse=coarse_stats,
                )
        geometry = region_geometry([int(v) for v in itk.size(fixed_image)], self._grid_shrink, region_size)
        drawn = geometry["regions"]
        if samples is not None:
            # An iteration steps the boxes it draws, anew at each one, as elastix steps on the points it draws; a box
            # steps as often on average as the engine's own run iterates.
            drawn = min(drawn, max(1, samples // region_size**dimension))
        if device_index >= 0:
            torch.cuda.init()  # the peak is read off the allocator, which exists once CUDA is initialised
            torch.cuda.reset_peak_memory_stats(device_index)
        started = time.monotonic()
        grid, values, bytes_read = coarse_grid, [], 0
        for number in range(self._stages.count("fine")):  # each refines the field the one before left
            scratch = field.parent / f"fine_{number}" if number else field.parent  # apart from the grid it starts from
            scratch.mkdir(exist_ok=True)
            fine = self._new_fine(fixed_image, moving_image, fixed_mask_image, moving_mask_image, device)
            readers.append(fine)  # its grid, which the next stage reads, may lie in its files
            if grid is not None:
                fine.SetInitialDisplacementField(grid)  # read as its dense field, whatever its grid
            fine.SetRegionSize(int(region_size))
            fine.SetScratchDirectory(str(scratch))
            fine.SetRegionHalo(self.region_halo("fine"))
            fine.SetRegionsPerIteration(0 if drawn == geometry["regions"] else drawn)
            fine.SetNumberOfIterations(math.ceil(int(epochs) * geometry["regions"] / drawn))
            grid = fine.GetControlGridTransform().GetDisplacementField()  # solves, allocating no dense field
            values += list(fine.GetMetricValuesPerIteration())
            bytes_read += int(fine.GetBytesRead())
        control_grid = finish(fine.GetOutput(), grid)
        if device_index >= 0:
            torch.cuda.synchronize(device_index)
        return RegionRefinement(
            control_grid,
            field,
            values,
            bytes_read,
            (bool(fine.GetFixedImageStreamed()), bool(fine.GetMovingImageStreamed())),
            geometry["regions"],
            len(values),
            time.monotonic() - started,
            int(torch.cuda.max_memory_allocated(device_index)) if device_index >= 0 else None,
            bool(fine.GetResidentFeatures()),
            coarse_stats,
        )


class RegistrationNet(network.Network):
    """Pairwise ConvexAdam registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1, fixed mask = 2,
    moving mask = 3; masks restrict every stage's similarity, whole-image = no restriction).

    Output on the fixed grid: ``DisplacementField`` (the
    DIM-component displacement field, in mm). Geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``.

    ConvexAdam registers on the fixed image's own grid. Its sizes are counted in voxels of the fixed image, the same
    number along every axis whatever its spacing, as the authors' code counts them: grid_spacing,
    displacement_half_width, grid_shrink, the step learning_rate sets, the gradient regularization_weight penalises
    and lncc_kernel. The coarse stage's cost smoothing and its NCC/LNCC window are counted in its cells,
    control_grid_smoothing in control cells. The presets' search of +/- 24 voxels is 24 mm at 1 mm, and at
    0.8 x 0.8 x 3 mm 19.2 mm in-plane and 72 mm along z.
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
            "upsampled), in voxels along every axis. Smaller = a finer coarse field and far more memory: "
            "(2 x half-width + 1)^3 / grid_spacing^3 floats per voxel.",
        ] = 6,
        displacement_half_width: Annotated[
            int,
            Range(1, 512),
            "Coarse stage: half-width of the discrete search in cells, so it captures displacement_half_width x "
            "grid_spacing voxels each way along every axis (24 with the presets' 4 x 6). Raise it for larger motion; "
            "memory grows as (2 x half-width + 1)^3.",
        ] = 4,
        balance_coarse_layers: Annotated[
            bool,
            "Coarse stage: divide each layer by its spread over the candidate displacements (the mean over the cells "
            "of its cost's range, measured on the first cost volume), times layers_weight, so that every layer moves "
            "the search alike, their total spread staying the raw one the coupling is calibrated on; a layer that "
            "does not vary weighs 0 (off = the raw costs, weighed by layers_weight alone). Experimental.",
        ] = False,
        deformable_masked: Annotated[
            bool,
            "Restrict the coarse and fine stages to the masks. False reads the whole image, the masks left unread, "
            "as ConvexAdam's authors run it (use_mask=False).",
        ] = True,
        iterations: Annotated[
            int,
            Range(0, 100000),
            "Fine stage: Adam steps; higher = more converged, slower.",
        ] = 80,
        learning_rate: Annotated[
            float,
            Range(0.0, 100.0),
            "Fine stage: Adam step size, in voxels per step along every axis (the control grid holds full-resolution "
            "displacements: ConvexAdam's step 1 on its half-resolution grid is 2 here). Higher converges faster but "
            "can oscillate.",
        ] = 2.0,
        regularization_weight: Annotated[
            float,
            Range(0.0, 1000.0),
            "Fine stage: weight of the diffusion regulariser, the squared gradient of the smoothed control grid in "
            "voxels per voxel (ConvexAdam's lambda); higher = smoother, lower = more flexible but risks folding.",
        ] = 1.25,
        grid_shrink: Annotated[
            int,
            Range(1, 128),
            "Fine stage: the spacing of the Adam control grid, in voxels along every axis (ConvexAdam's "
            "grid_sp_adam, 2); higher = a coarser, smoother, faster refinement.",
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
        moments_init: Annotated[
            Literal["cof", "com", "none"],
            "Translation the moving image starts from, as ANTs' -r [fixed,moving,N]: 'cof' aligns the frames' "
            "centres (N=0), 'com' the centres of mass of the intensities (N=1), 'none' leaves the pair as it comes, "
            "as ConvexAdam does. For a pair that does not overlap within the coarse search "
            "(displacement_half_width x grid_spacing voxels each way); nothing is optimised before the stages.",
        ] = "none",
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
            "Seed of torch's generator inside itk-impact. A GPU run is not bit-reproducible whatever the seed "
            "(grid_sample's backward accumulates in any order).",
        ] = 42,
    ) -> None:
        if "KONFAI_CONFIG_PATH" in os.environ:  # bound from a config file: the keys it sets beside the arguments
            with Config(os.environ["KONFAI_CONFIG_PATH"]) as block:
                refuse_linear_keys(block.config or {})
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
            seed,
            mode,
            normalize,
            feature_map_update_interval,
            lncc_kernel,
            voxel_sampling,
            balance_coarse_layers,
            deformable_masked,
            moments_init,
        )
        self.add_module(
            "Registration",
            EngineRegistration(engine, fuse_texpr=False),
            in_branch=[0, 1, 2, 3],
            out_branch=["registration"],
        )
        # The output module the presets name.
        self.add_module("DisplacementField", torch.nn.Identity(), in_branch=["registration"], out_branch=["dvf"])


@dataclass
class RegionRefinement:
    """What ``ConvexAdamEngine.refine_by_regions`` hands back: the file of the field on the control grid, the
    dense field file, and what the run measured."""

    control_grid: Path
    field: Path
    metric_values: list[float]
    bytes_read: int
    streamed: tuple[bool, bool]
    regions: int
    iterations: int
    seconds: float
    device_peak_bytes: int | None
    #: Whether the pooled feature maps stayed on the device (the dense iterations on them) or every iteration read
    #: its boxes.
    resident: bool
    #: The coarse stage's seconds and bytes read, when the engine has one (it runs by regions before the fine one).
    coarse: tuple[float, int] | None = None


def require_regions() -> None:
    """Refuse an itk-impact whose fine stage cannot run by regions (``SetRegionSize``, 0.2 and later)."""
    if not hasattr(_fine_registration_type(DIM).New(), "SetRegionSize"):
        raise MeasureError(
            "The fine stage by regions needs itk-impact 0.2 or later.",
            'Install it beside the torch its wheels are built against: pip install "itk-impact>=0.2"',
        )


def region_geometry(size_xyz: list[int], grid_shrink: int, region_size: int) -> dict:
    """The control grid and the boxes the fine stage lays on a fixed grid, as itk-impact counts them, in voxels
    along every axis: cells ``size // shrink``; a box ``region / shrink`` cells a side, rounded half up as its
    ``llround`` does, the last one along an axis taking what is left."""
    shrink = max(1, grid_shrink)
    cells = [max(1, extent // shrink) for extent in size_xyz]
    per_box = max(1, (2 * region_size + shrink) // (2 * shrink))
    return {"cells": cells, "regions": math.prod(math.ceil(count / per_box) for count in cells)}


def _write_streamed(output, field: Path, fixed_image) -> None:
    """Write a stage's dense field on the fixed grid to ``field`` by slabs, each computed as the writer asks for it."""
    dimension = fixed_image.GetImageDimension()
    writer = itk.ImageFileWriter[itk.Image[itk.Vector[itk.F, dimension], dimension]].New(
        Input=output, FileName=str(field)
    )
    voxels = math.prod(itk.size(fixed_image))
    writer.SetNumberOfStreamDivisions(max(1, math.ceil(voxels * dimension * 4 / _SLAB_BYTES)))
    writer.Update()
