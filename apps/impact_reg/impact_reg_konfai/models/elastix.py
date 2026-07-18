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
2/3) and splits its output into ``MovedImage`` / ``DisplacementField`` on the fixed grid. This module owns
the MAPPING — the per-resolution model matrix (``resolutions``) turned into IMPACT parameter-map lines, and
the config schema (``ModelSpec`` / ``ResolutionSpec``). The elastix RUNTIME (binary install, model download,
subprocess, progress) lives in ``elastix_engine.py`` and is imported only when the graph is built.

A UI reads the tuning knobs straight from the TYPES below: ``Literal`` (a fixed set),
``Annotated[.., Range]`` (numeric bounds), ``Annotated[str, Choices(...)]`` (a resolver the app owns).

NOTE: do NOT add ``from __future__ import annotations`` — KonfAI's config engine reads runtime annotations
(``get_origin``); PEP 563 stringized annotations break arg resolution.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import torch
from huggingface_hub import hf_hub_download
from konfai.network import network
from konfai.utils.config import Choices, Range

# IMPACT field docs: https://github.com/vboussot/ImpactLoss/tree/main/ParameterMaps
# A model's FIXED props (dimension / channels / FOV formula) come from the registry (models.json on
# VBoussot/impact-torchscript-models); the config carries the FREE knobs (models per resolution, voxel size,
# iterations, per-model weights/mask/subset/pca/distance) and the global ``mode``.
_IMPACT_MODELS_REGISTRY = "VBoussot/impact-torchscript-models:models.json"

# ``2^l+3`` plateaus: segmenter layers 7-8 share layer 6's receptive field. Deeper configs should run
# Static anyway; in Jacobian we clamp ``l`` to this plateau.
_FOV_RAMP_MAX_LAYER = 6


def registry_choices() -> list[str]:
    """The ``ref`` picker's values — model refs (``repo:path``) from the registry the engine already fetches
    (offline-first). A user may still point ``ref`` at a local model."""
    repo = _IMPACT_MODELS_REGISTRY.split(":", 1)[0]
    return [f"{repo}:{key}" for key in load_models_registry()]


def _num(x: object) -> str:
    """Format a number the elastix way: no trailing '.0' (6.0 -> '6', 0.2 -> '0.2')."""
    return "%g" % float(x)


@dataclass
class ModelSpec:
    """One feature model at one resolution (several may share a resolution). ``ref`` picks the model; the
    rest are its per-(resolution, model) knobs. Dimension / channels / FOV are intrinsic — from the registry
    (``models.json``) keyed by ``ref`` — never tuned."""

    ref: Annotated[
        str,
        Choices(registry_choices),
        "IMPACT feature model compared at this resolution (TorchScript 'repo:file' on Hugging Face); different "
        "models capture different anatomy/contrast. Suggested priors (from the IMPACT study, not forced): "
        "TotalSegmentator (TS/M730) is the general default; a model trained on the target structure (e.g. lung "
        "or vessels) sharpens local alignment there; add MIND for MR/CT to recover intra-organ detail; SAM2.1 "
        "for fast 2D exploration.",
    ]
    voxel_size: Annotated[
        list[float], "Working resolution (mm) this model is evaluated at (empty = the resolution level's default)."
    ] = field(default_factory=list)
    layers_weight: Annotated[
        list[float], "Per-layer weights of this feature model's selected layers in the metric."
    ] = field(default_factory=lambda: [1.0])
    subset_features: Annotated[
        int, Range(0, 1000), "Number of this model's feature channels to keep (0 = all); trims cost."
    ] = 0
    pca: Annotated[
        int,
        Range(0, 100),
        "Number of PCA components this model's feature channels are reduced to before matching (0 = keep all).",
    ] = 0
    distance: Annotated[
        Literal["L1", "L2", "Dice", "Cosine", "NCC"],
        "Similarity measure compared on this model's features. Suggested prior (not forced): when the task is "
        "scored on Dice, choosing 'Dice' aligns the loss with the metric.",
    ] = "L1"
    layers_mask: Annotated[
        str,
        "Per-layer on/off bitmask over the model's layers ('1' = use, '0' = skip); also sets the Jacobian FOV "
        "(the deepest selected layer's receptive field). Suggested priors (not forced): CT/CBCT favours EARLY "
        "layers (they denoise and enhance structures across modalities, robust to artifacts) with 'Jacobian' "
        "mode; MR/CT favours HIGH-LEVEL layers (contour/segmentation-driven) with 'Static' mode.",
    ] = ""


@dataclass
class ResolutionSpec:
    """One elastix resolution level: its iteration budget and the (self-configured) models compared there."""

    max_iterations: Annotated[
        int, Range(1, 100000), "Optimiser iterations spent at this resolution level."
    ]
    models: dict[str, ModelSpec]


def _sorted_specs(mapping: dict) -> list:
    """dict keyed by string indices ('0','1',...) -> values in numeric order."""
    return [mapping[k] for k in sorted(mapping, key=lambda key: int(key))]


def load_models_registry(ref: str = _IMPACT_MODELS_REGISTRY) -> dict:
    """Load models.json (the fixed params per model) from the model repo on Hugging Face.

    The registry is NOT bundled with the preset. ``KONFAI_IMPACT_MODELS_REGISTRY`` (a local path) wins for
    dev/offline; otherwise ``ref`` must be a ``repo:file`` Hugging Face reference.
    """
    local = os.environ.get("KONFAI_IMPACT_MODELS_REGISTRY", "")
    if local:
        path = Path(local)
    elif ":" in ref:
        repo, filename = ref.split(":", 1)
        path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="model"))  # nosec B615
    else:
        raise ValueError(
            f"models_registry '{ref}' must be a 'repo:file' Hugging Face reference (the registry is fetched "
            f"from HF, not bundled) — or set KONFAI_IMPACT_MODELS_REGISTRY to a local file for offline use."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _model_key(ref: str) -> str:
    """Registry key / staged relative path = the model file within the repo (strip a 'repo:' prefix)."""
    return ref.split(":", 1)[1] if ":" in ref else ref


def _deepest_active_layer(layers_mask: str) -> int:
    """Deepest (largest-FOV) layer selected by ``layers_mask``, as a 0-based index.

    A model returns its layers shallow->deep; ``layers_mask`` has one char per returned layer, position ``i``
    == ``layer_i``, ``'1'`` = selected. In Jacobian the patch must cover the DEEPEST selected layer's
    receptive field, so the FOV is governed by the rightmost ``'1'``.
    """
    mask = layers_mask.strip().strip('"')
    active = [i for i, char in enumerate(mask) if char == "1"]
    if not active:
        raise ValueError(f"LayersMask '{layers_mask}' selects no layer; cannot derive the model FOV.")
    return max(active)


def _fov_value(fov: dict, layers_mask: str) -> int:
    """Evaluate a model's field-of-view (in voxels) from its registry ``fov`` spec.

    Formulas (model repo, https://huggingface.co/VBoussot/impact-torchscript-models):
      ``2*r*d+1``  MIND, from radius ``r`` / dilation ``d`` (R1D2 -> 5);
      ``2^l+3``    TotalSegmentator / MRSegmentator, ``l`` = deepest layer picked by ``layers_mask``, clamped
                   to the receptive-field plateau ``_FOV_RAMP_MAX_LAYER`` (layers 7-8 -> layer 6);
      a bare int   a fixed FOV (SAM2.1 -> 29, DINOv2 -> 14);
      ``Global``   Anatomix — whole-image only (Static); no finite Jacobian patch -> error.
    An explicit ``value`` in the spec is honoured as a precomputed shortcut.
    """
    formula = str(fov.get("formula", "")).strip()
    key = re.sub(r"\s+", "", formula).lower()
    if key.isdigit():
        return int(key)
    if key == "2*r*d+1":
        return 2 * int(fov["r"]) * int(fov["d"]) + 1
    if key == "2^l+3":
        return 2 ** min(_deepest_active_layer(layers_mask), _FOV_RAMP_MAX_LAYER) + 3
    if "global" in key:
        raise ValueError(f"model FOV '{formula}' is whole-image only (Static); it has no Jacobian patch size.")
    if fov.get("value") is not None:
        return int(fov["value"])
    raise ValueError(f"cannot evaluate model FOV formula '{formula}'.")


def _patch_size(mode: str, entry: dict, layers_mask: str) -> str:
    """PatchSize from the model FOV, one token per model axis (2D -> 2 tokens, 3D -> 3): Static -> whole
    image (all zeros); Jacobian -> the evaluated FOV per axis. A 2D+3D mix at a resolution concatenates,
    e.g. ``29 29 11 11 11`` (SAM 2D + TS 3D), matching IMPACT."""
    dim = int(entry.get("dimension", 3))
    if mode.strip().strip('"').lower() != "jacobian":
        return " ".join(["0"] * dim)
    fov = _fov_value(entry.get("fov", {}), layers_mask)
    return " ".join([str(fov)] * dim)


def generate_impact_parameter_map(template_text: str, resolutions: dict, registry: dict, mode: str = "Static") -> str:
    """Rewrite the resolution-dependent lines of ``template_text`` from the model matrix ``resolutions``.

    Regenerated: MaximumNumberOfIterations, NumberOfResolutions, Fixed/MovingImagePyramidRescaleSchedule,
    ImpactMode, and the whole ImpactXxxK block; every other line is kept verbatim. N (number of resolutions)
    is deduced from the config. ``mode`` drives PatchSize: Static -> ``0 0 0``; Jacobian -> the per-model FOV
    from the registry formula and the cell's ``layers_mask``.
    """
    res = _sorted_specs(resolutions)
    n = len(res)
    mode_clean = mode.strip().strip('"') or "Static"

    impact: list[str] = []
    for k, r in enumerate(res):
        models = _sorted_specs(r.models)
        entries = [registry[_model_key(m.ref)] for m in models]

        def row(stem: str, values: list[str]) -> None:
            impact.append(f"(Impact{stem}{k} " + " ".join(values) + ")")

        # From the registry ONLY the 3 truly model-fixed props (Dimension, NumberOfChannels, PatchSize = the
        # model FOV); everything else is a per-model knob taken straight from the cell.
        row("ModelsPath", [f'"{_model_key(m.ref)}"' for m in models])
        row("Dimension", [e["dimension"] for e in entries])
        row("NumberOfChannels", [e["numberofchannels"] for e in entries])
        row("PatchSize", [_patch_size(mode_clean, e, m.layers_mask) for e, m in zip(entries, models)])
        row("VoxelSize", [" ".join(_num(v) for v in m.voxel_size) for m in models])
        row("LayersMask", [f'"{m.layers_mask}"' for m in models])
        row("SubsetFeatures", [str(m.subset_features) for m in models])
        row("PCA", [str(m.pca) for m in models])
        row("Distance", [f'"{m.distance}"' for m in models])
        row("LayersWeight", [" ".join(_num(w) for w in m.layers_weight) for m in models])
        impact.append("")  # blank line between resolutions, mirroring the reference maps

    # The per-resolution block is the contiguous span from the first to the last ``Impact<name><k>`` line
    # (inner blanks fall inside it). Replace the whole span at its first line so reference blanks aren't kept.
    lines = template_text.splitlines()
    indexed = [(re.match(r"^\s*\((\S+?)\s+(.*?)\)\s*$", ln), ln) for ln in lines]
    block_rows = [i for i, (m, _) in enumerate(indexed) if m and re.match(r"^Impact[A-Za-z]+\d+$", m.group(1))]
    block_lo, block_hi = (block_rows[0], block_rows[-1]) if block_rows else (-1, -2)

    out: list[str] = []
    for i, (m, line) in enumerate(indexed):
        key = m.group(1) if m else None
        if block_lo <= i <= block_hi:
            if i == block_lo:  # replace the whole span at its first line, drop the rest (incl. inner blanks)
                out.extend(impact[:-1])
        elif key == "MaximumNumberOfIterations":
            out.append("(MaximumNumberOfIterations " + " ".join(_num(r.max_iterations) for r in res) + ")")
        elif key == "NumberOfResolutions":
            out.append(f"(NumberOfResolutions {n})")
        elif key in ("FixedImagePyramidRescaleSchedule", "MovingImagePyramidRescaleSchedule"):
            out.append(f"({key} " + " ".join(["1"] * 3 * n) + ")")
        elif key == "ImpactMode":
            out.append(f'(ImpactMode "{mode_clean}")')
        else:
            out.append(line)
    return "\n".join(out)


class ChannelSelect(torch.nn.Module):
    """Select a channel slice ``[start:stop]`` (splits the registration output into moved / DVF)."""

    def __init__(self, start: int, stop: int) -> None:
        super().__init__()
        self._start = start
        self._stop = stop

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self._start : self._stop]


class RegistrationNet(network.Network):
    """Pairwise registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1, fixed mask = 2,
    moving mask = 3; masks restrict the metric, whole-image = no restriction).

    Outputs (both on the fixed grid): ``MovedImage`` (moving resampled onto fixed) and ``DisplacementField``
    (the dim-component displacement field, mm). ``ElastixRegistration`` produces both channel-stacked; two
    ``ChannelSelect`` modules split them. Output geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``.
    """

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
            "is required — 'resolutions' regenerates a template's resolution-dependent lines, it does not "
            "replace it.",
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
        subset_features: Annotated[
            int, Range(0, 1000), "Number of IMPACT feature channels to keep across models (0 = all); trims cost."
        ] = 0,
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
        resolutions: dict[str, ResolutionSpec] = {},
        mode: Annotated[
            Literal["Static", "Jacobian"],
            "IMPACT feature-extraction mode: 'Static' (whole-image features, computed once per resolution -- "
            "fast, inference-only) or 'Jacobian' (patch-wise, differentiable, precise, slower). Suggested "
            "priors (not forced): early/downsampling layers -> 'Jacobian'; high-level layers -> 'Static'. "
            "Avoid 'Static' for large-stride/transformer models (SAM, DINOv2): frozen features lose local "
            "alignment.",
        ] = "Static",
    ) -> None:
        # The registration is fully described by ``resolutions`` (config = source of truth): each resolution
        # lists its self-configured models; the download list is derived from the cells. Global knobs override
        # the generated map (final_grid_spacing -> FinalGridSpacingInPhysicalUnits mm, spatial_samples ->
        # NumberOfSpatialSamples, parameter_overrides 'Key=value'). Empty ``resolutions`` = an intensity-only
        # preset (fixed maps + overrides). The elastix runtime is imported here (heavy: torch/sitk/subprocess).
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
                subset_features,
                spatial_samples,
                parameter_overrides,
                resolutions,
                mode,
            ),
            in_branch=[0, 1, 2, 3],
            out_branch=["registration"],
        )
        self.add_module("MovedImage", ChannelSelect(0, 1), in_branch=["registration"], out_branch=["moved"])
        self.add_module("DisplacementField", ChannelSelect(1, 4), in_branch=["registration"], out_branch=["dvf"])
