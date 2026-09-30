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

"""Registration as a KonfAI model: the config -> elastix parameter-map mapping + the ``add_module`` graph.

``RegistrationNet`` wires ``ElastixRegistration`` (fixed = branch 0, moving = branch 1, fixed/moving masks =
2/3) and emits its ``DisplacementField`` on the fixed grid. This module owns the MAPPING: the IMPACT models
(``models``, or ``levels`` one per resolution, the schema every engine shares, see ``impact_loss``) turned into
IMPACT parameter-map lines. The elastix RUNTIME (binary install, model download, subprocess, progress) lives in
``elastix_engine.py`` and is imported only when the graph is built.

A UI reads the tuning knobs straight from the TYPES below: ``Literal`` (a fixed set),
``Annotated[.., Range]`` (numeric bounds), ``Annotated[str, Choices(...)]`` (a resolver the app owns).

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine reads runtime annotations
(``get_origin``); PEP 563 stringized annotations break arg resolution.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import torch
from konfai.metric.measure.impact import ImpactFeatureModel
from konfai.network import network
from konfai.utils.config import Range

from .impact_loss import (
    FeatureMapUpdateInterval,
    MixedPrecision,
    Mode,
    ModelSpec,
    Normalize,
    check_models,
    layer_weights,
    level_models,
    per_kept_layer,
    sorted_specs,
)

# IMPACT field docs: https://github.com/vboussot/ImpactLoss/tree/main/ParameterMaps
# A model's FIXED props (dimension / channels / receptive field) come from KonfAI's feature model (the registry,
# models.json on VBoussot/impact-torchscript-models); the config carries the model knobs (``ModelSpec``).

# elastix-IMPACT clamps ImpactSubsetFeatures to [1, channels]: "all" is written as a count no layer reaches.
_ALL_CHANNELS = 100000

# elastix's own default when a map does not say how many resolutions it runs.
_ELASTIX_DEFAULT_RESOLUTIONS = 3

_PYRAMID_SCHEDULES = tuple(
    f"{image}ImagePyramid{kind}Schedule" for image in ("Fixed", "Moving") for kind in ("", "Rescale", "Smoothing")
)


def _num(x: object) -> str:
    """Format a number the elastix way: no trailing '.0' (6.0 -> '6', 0.2 -> '0.2')."""
    return f"{float(x):g}"


@dataclass
class ElastixLevelSpec:
    """One elastix resolution: its iteration budget and the models compared there, in place of ``models``."""

    max_iterations: Annotated[int, Range(1, 100000), "Optimiser iterations spent at this resolution level."]
    models: dict[str, ModelSpec]


def _patch_size(mode: str, model: ImpactFeatureModel) -> str:
    """PatchSize, one token per model axis (2D -> 2 tokens, 3D -> 3): Static -> whole image (all zeros); Jacobian ->
    the receptive field of the model's deepest kept layer. A 2D+3D mix at a resolution concatenates, e.g.
    ``29 29 11 11 11`` (SAM 2D + TS 3D), matching IMPACT."""
    if mode.strip().strip('"').lower() != "jacobian":
        return " ".join(["0"] * model.dim)
    return " ".join([str(model.receptive_field)] * model.dim)


def _voxel_size(
    spec: ModelSpec, model: ImpactFeatureModel, mode: str, native: Sequence[float], where: str
) -> list[float]:
    """The grid (mm) ``spec``'s model sees the image on: its voxel_size, or the fixed image's own spacing without one.
    Static resamples the image on its own axes, Jacobian feeds the model patches of its own dimension."""
    axes = model.dim if mode.lower() == "jacobian" else 3
    size = [float(v) for v in spec.voxel_size] if spec.voxel_size is not None else list(native)[:axes]
    if len(size) != axes:
        raise ValueError(f"{where}: voxel_size needs {axes} values (mm) in {mode} mode, got {spec.voxel_size}.")
    return size


# The samplers that draw points at random, whose NumberOfSpatialSamples voxel_sampling sets.
_RANDOM_SAMPLERS = ("Random", "RandomCoordinate", "RandomSparseMask", "MultiInputRandomCoordinate")


def _entry(text: str, key: str) -> list[str] | None:
    """The values of a map's ``(key ...)`` entry, unquoted, or None when the map has none."""
    match = re.search(rf"^\s*\({key}\s+([^)]*)\)", text, re.MULTILINE)
    return [token.strip('"') for token in match.group(1).split()] if match else None


def sampled_spatial_samples(text: str, fraction: float, voxels: int, dim: int = 3) -> str:
    """The map's NumberOfSpatialSamples set to ``fraction`` of the fixed image's voxels at each level (``voxels`` at
    full resolution, those of the mask when there is one): voxel_sampling, until elastix reads a proportion itself.

    A level holds the voxels the map's pyramid leaves it: a recursive or shrinking pyramid divides them by its shrink
    factors, the generic one by its rescale schedule, a smoothing-only one keeps them all. The fraction replaces any
    count the map or an override wrote.
    """
    sampler = (_entry(text, "ImageSampler") or [""])[0]
    if sampler not in _RANDOM_SAMPLERS:
        raise ValueError(
            f"voxel_sampling sets how many points elastix's sampler draws, and this map samples with "
            f"'{sampler or 'its default'}': name a random one ({', '.join(_RANDOM_SAMPLERS)}) or leave voxel_sampling "
            "at 1."
        )
    levels = int((_entry(text, "NumberOfResolutions") or [_ELASTIX_DEFAULT_RESOLUTIONS])[0])
    pyramid = (_entry(text, "FixedImagePyramid") or ["FixedSmoothingImagePyramid"])[0]
    default = [2 ** (levels - 1 - level) for level in range(levels) for _ in range(dim)]
    if "Smoothing" in pyramid:
        schedule: list = [1] * (levels * dim)
    elif "Generic" in pyramid:
        schedule = (
            _entry(text, "FixedImagePyramidRescaleSchedule") or _entry(text, "FixedImagePyramidSchedule") or default
        )
    else:
        schedule = _entry(text, "FixedImagePyramidSchedule") or default
    shrink = [
        math.prod(float(factor) for factor in schedule[level * dim : (level + 1) * dim]) for level in range(levels)
    ]
    line = "(NumberOfSpatialSamples " + " ".join(str(max(1, round(fraction * voxels / s))) for s in shrink) + ")"
    if _entry(text, "NumberOfSpatialSamples") is None:
        return text + "\n" + line
    return re.sub(r"^\s*\(NumberOfSpatialSamples\s+[^)]*\).*$", line, text, count=1, flags=re.MULTILINE)


def has_impact_block(text: str) -> bool:
    """Whether a parameter map carries an IMPACT block (``(ImpactModelsPath0 ...)`` and its siblings)."""
    return re.search(r"^\s*\(Impact[A-Za-z]+\d+\s", text, re.MULTILINE) is not None


def generate_impact_parameter_map(
    template_text: str,
    models: dict[str, ModelSpec],
    levels: dict[str, ElastixLevelSpec],
    feature_models: dict[tuple[str, str], ImpactFeatureModel],
    native_voxel_size: Sequence[float],
    mode: str = "Static",
    normalize: bool = True,
    feature_map_update_interval: int = -1,
    mixed_precision: bool = False,
) -> str:
    """Rewrite the IMPACT lines of ``template_text`` from ``models`` (every resolution) or ``levels`` (one each).

    Regenerated: the whole ImpactXxxK block, ImpactMode and the loss settings (ImpactNormalizeLosses,
    ImpactFeaturesMapUpdateInterval, ImpactUseMixedPrecision), and with ``levels`` MaximumNumberOfIterations and
    NumberOfResolutions; every other line is kept verbatim, and a regenerated line the template lacks is added.
    Without ``levels`` the map's own resolutions and iterations stand. ``mode`` drives PatchSize: Static ->
    ``0 0 0``; Jacobian -> the receptive field of the deepest layer of the model's ``layers_mask``.

    IMPACT reads the original images and resamples them to each model's voxel_size itself, so the map's pyramid
    only reaches its other metrics (a Mattes MI beside IMPACT) and the samplers: it is kept, unless a schedule no
    longer holds one entry per axis and resolution, when elastix's default pyramid replaces it. A model without a
    voxel_size sees the image at ``native_voxel_size``, the fixed image's spacing.

    A map without an IMPACT block is returned as it is: the models describe the IMPACT metric's levels, and an
    intensity stage run before it (a rigid Mattes MI alignment) keeps its own pyramid and iterations.

    ``feature_models`` holds each model's KonfAI feature model by ``(ref, layers_mask)``: its file (the
    ``ImpactModelsPath``), dimension, channels and receptive field.
    """
    if not has_impact_block(template_text):
        return template_text
    if not models and not levels:
        raise ValueError(
            "The parameter map has an IMPACT block, but the preset declares no 'models' (nor 'levels') to write into "
            "it: elastix would read the map's own ImpactModelsPath entries, which name no file. Declare the models "
            "under the preset's RegistrationNet."
        )
    mode_clean = mode.strip().strip('"') or "Static"
    counted = re.search(r"^\s*\(NumberOfResolutions\s+(\d+)", template_text, re.MULTILINE)
    n = len(levels) if levels else int(counted.group(1)) if counted else _ELASTIX_DEFAULT_RESOLUTIONS

    impact: list[str] = []
    for k, specs in enumerate(level_models(models, levels, n, "elastix")):
        where = f"elastix level {k}"
        check_models(specs, where, dense=False)
        loaded = [feature_models[(m.ref, m.layers_mask)] for m in specs]
        voxels = [
            _voxel_size(m, e, mode_clean, native_voxel_size, f"{where} model '{m.ref}'")
            for m, e in zip(specs, loaded, strict=True)
        ]

        def row(stem: str, values: list[str], k: int = k) -> None:
            impact.append(f"(Impact{stem}{k} " + " ".join(values) + ")")

        # From the model ONLY its fixed props (file, Dimension, NumberOfChannels, PatchSize = its receptive field);
        # everything else is a per-model knob taken straight from the spec. SubsetFeatures, PCA, Distance and
        # LayersWeight hold one entry per kept layer, flat across the level's models. The file keeps the platform's
        # own separators: elastix's parameter parser cuts a line at '//', quotes or not.
        row("ModelsPath", [f'"{e.model_path}"' for e in loaded])
        row("Dimension", [str(e.dim) for e in loaded])
        row("NumberOfChannels", [str(e.in_channels) for e in loaded])
        row("PatchSize", [_patch_size(mode_clean, e) for e in loaded])
        row("VoxelSize", [" ".join(_num(v) for v in voxel) for voxel in voxels])
        row("LayersMask", [f'"{m.layers_mask}"' for m in specs])
        row("FeatureNormalization", [f'"{m.feature_normalization}"' for m in specs])
        row("SubsetFeatures", [str(c or _ALL_CHANNELS) for c in per_kept_layer(specs, lambda m: m.subset_features)])
        row("PCA", [str(pca) for pca in per_kept_layer(specs, lambda m: m.pca)])
        row("Distance", [f'"{d}"' for d in per_kept_layer(specs, lambda m: m.distance)])
        row("LayersWeight", [_num(w) for w in layer_weights(specs)])
        impact.append("")  # blank line between resolutions, mirroring the reference maps

    # The per-resolution block is the contiguous span from the first to the last ``Impact<name><k>`` line
    # (inner blanks fall inside it). Replace the whole span at its first line so reference blanks aren't kept.
    lines = template_text.splitlines()
    # An entry may be followed by a '// comment'.
    indexed = [(re.match(r"^\s*\((\S+?)\s+(.*?)\)\s*(?://.*)?$", ln), ln) for ln in lines]
    block_rows = [i for i, (m, _) in enumerate(indexed) if m and re.match(r"^Impact[A-Za-z]+\d+$", m.group(1))]
    block_lo, block_hi = (block_rows[0], block_rows[-1]) if block_rows else (-1, -2)
    regenerated = {
        "ImpactMode": f'(ImpactMode "{mode_clean}")',
        "ImpactNormalizeLosses": f'(ImpactNormalizeLosses "{str(normalize).lower()}")',
        "ImpactFeaturesMapUpdateInterval": f"(ImpactFeaturesMapUpdateInterval {int(feature_map_update_interval)})",
        "ImpactUseMixedPrecision": f'(ImpactUseMixedPrecision "{str(mixed_precision).lower()}")',
    }
    if levels:
        iterations = " ".join(_num(level.max_iterations) for level in sorted_specs(levels))
        regenerated["MaximumNumberOfIterations"] = f"(MaximumNumberOfIterations {iterations})"
        regenerated["NumberOfResolutions"] = f"(NumberOfResolutions {n})"

    out: list[str] = []
    for i, (m, line) in enumerate(indexed):
        key = m.group(1) if m else None
        if block_lo <= i <= block_hi:
            if i == block_lo:  # replace the whole span at its first line, drop the rest (incl. inner blanks)
                out.extend(impact[:-1])
            elif key is not None and i not in block_rows:
                out.append(line)  # another entry written inside the span is the template's, not the block's
        elif key in regenerated:
            out.append(regenerated[key])
        elif key in _PYRAMID_SCHEDULES and len(m.group(2).split()) != 3 * n:
            continue  # written for another number of resolutions: elastix's default pyramid instead
        else:
            out.append(line)
    keys = {m.group(1) for m, _ in indexed if m}
    out += [line for key, line in regenerated.items() if key not in keys]
    return "\n".join(out)


class RegistrationNet(network.Network):
    """Pairwise registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1, fixed mask = 2,
    moving mask = 3; masks restrict the metric, whole-image = no restriction).

    Output, on the fixed grid: ``DisplacementField``
    (the dim-component displacement field, mm). Output geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``.
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
        engine: Annotated[
            str, "Registration backend binary ('elastix'); selects the parameter-map dialect."
        ] = "elastix",
        parameter_maps: Annotated[
            list[str],
            "elastix parameter-map preset template(s) run in sequence (e.g. rigid then bspline); at least one "
            "is required: 'models' and 'levels' regenerate a template's IMPACT lines; they do not replace it.",
        ] = [],
        max_iterations: Annotated[
            int,
            Range(0, 100000),
            "Global override of the optimiser iterations per resolution (0 = keep each map's own value).",
        ] = 0,
        final_grid_spacing: Annotated[
            float,
            Range(0.0, 100.0),
            "Final B-spline control-point spacing (mm) of the deformable map; smaller = a more flexible "
            "deformation, 0 = keep the map's default.",
        ] = 0.0,
        spatial_samples: Annotated[
            int,
            Range(0, 100000),
            "Random spatial samples the metric draws per iteration (0 = keep the map's default); more = a "
            "smoother, slower metric.",
        ] = 0,
        parameter_overrides: Annotated[
            list[str],
            "Raw elastix parameter overrides as 'Key=value' strings, applied on top of the generated map "
            "(advanced escape hatch).",
        ] = [],
        models: dict[str, ModelSpec] = {},
        levels: Annotated[
            dict[str, ElastixLevelSpec],
            "The IMPACT map's resolutions ('0', '1', ...), each with its iterations and its models in place of "
            "'models'; empty = the map's own resolutions and iterations, 'models' at every one.",
        ] = {},
        mode: Mode = "Static",
        normalize: Normalize = True,
        feature_map_update_interval: FeatureMapUpdateInterval = -1,
        mixed_precision: MixedPrecision = False,
        voxel_sampling: Annotated[
            float,
            Range(0.00001, 1.0),
            "Share of the fixed image's voxels the IMPACT map's sampler draws at each iteration, at each level "
            "(1 = the map's own NumberOfSpatialSamples); written as NumberOfSpatialSamples from the image and its "
            "mask at run time, and preferred to spatial_samples. The other engines read the same share. Experimental.",
        ] = 1.0,
        seed: Annotated[
            int,
            "Seed of what elastix draws at random, written as RandomSeed in every parameter map: the points of the "
            "RandomCoordinate and RandomSparseMask samplers, the stochastic optimisers' perturbations, and the IMPACT "
            "metric's channels (subset_features), patches and 2D planes, which 0 leaves to the clock. The 'Random' "
            "sampler does not read it.",
        ] = 42,
    ) -> None:
        # The IMPACT metric is described by ``models`` / ``levels`` (config = source of truth); the download list
        # is derived from them. Global knobs override the generated map (final_grid_spacing ->
        # FinalGridSpacingInPhysicalUnits mm, spatial_samples -> NumberOfSpatialSamples, parameter_overrides
        # 'Key=value'). No models = an intensity-only preset (fixed maps + overrides). The elastix runtime is
        # imported here (heavy: torch/sitk/subprocess).
        from .elastix_engine import ElastixRegistration

        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=3,
        )
        self.add_module(
            "Registration",
            ElastixRegistration(
                engine,
                parameter_maps,
                max_iterations,
                final_grid_spacing,
                spatial_samples,
                parameter_overrides,
                models,
                levels,
                mode,
                normalize,
                feature_map_update_interval,
                mixed_precision,
                voxel_sampling,
                seed,
            ),
            in_branch=[0, 1, 2, 3],
            out_branch=["registration"],
        )
        # The output module the presets name.
        self.add_module("DisplacementField", torch.nn.Identity(), in_branch=["registration"], out_branch=["dvf"])
