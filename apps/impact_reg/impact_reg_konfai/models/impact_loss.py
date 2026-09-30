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

"""The IMPACT loss every engine shares: one model spec, the loss settings, and the per-level model lists.

The three engines (elastix, ConvexAdam, FireANTs) read these with the same names and the same meaning:

- each kept layer of each model is compared with its model's ``distance``, 0 at a perfect match but for Dice,
  which is 0 there only on binary maps (one-hot labels): two identical soft maps of 0.5 score 0.5, and raw features
  can take it below 0; each layer is weighed by ``layers_weight``;
- ``normalize`` divides every layer by its value when a level starts, so each starts at 1 (a layer that starts at
  0 or below keeps its raw value);
- ``mode`` is Static (the features extracted once, then the moving ones warped) or Jacobian (the network inside
  the loss, the warp differentiated through it);
- ``models`` applies to every level, and ``levels`` replaces it level by level.

A setting an engine cannot honour is refused with a message rather than ignored.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine reads runtime annotations.
"""

from dataclasses import dataclass
from typing import Annotated, Literal

from konfai.metric.measure.impact import MODELS_REPO, ImpactFeatureModel, models_registry
from konfai.utils.config import Choices, Range


def registry_choices() -> list[str]:
    """The ``ref`` picker's values: the registry's models as ``repo:path``. A user may still point ``ref`` at a local
    model."""
    return [f"{MODELS_REPO}:{key}" for key in models_registry()]


Distance = Literal["L1", "L2", "Dice", "Cosine", "L1Cosine", "NCC", "LNCC"]
FeatureNormalization = Literal["none", "l2", "standardized"]


@dataclass
class ModelSpec:
    """One IMPACT feature model, the same in every engine. ``ref`` picks the model; the rest are its knobs.
    Dimension, channels and field of view are the model's own, read from the registry by ``ref``."""

    ref: Annotated[
        str,
        Choices(registry_choices),
        "IMPACT feature model (TorchScript 'repo:file' on Hugging Face, or a local file); different models capture "
        "different anatomy and contrast. Suggested priors (from the IMPACT study, not forced): TotalSegmentator "
        "(TS/M730) is the general default; a model trained on the target structure sharpens the alignment there; "
        "MIND adds intra-organ detail for MR/CT.",
    ]
    layers_mask: Annotated[
        str,
        "Per-layer on/off bitmask over the model's layers ('1' = use, '0' = skip), one character per layer; in "
        "elastix's Jacobian mode the deepest kept layer's receptive field sets the patch. Suggested priors (not "
        "forced): CT/CBCT favours EARLY layers (they denoise and enhance structures across modalities, robust to "
        "artifacts); MR/CT favours HIGH-LEVEL layers (contour and segmentation driven).",
    ] = "1"
    # None, not a default_factory: the config binder writes a default it can represent.
    layers_weight: Annotated[
        float | list[float] | None,
        "Weight of each layer layers_mask keeps, relative to the other models' layers: one value for all of them, "
        "or one per kept layer (default: 1 each). With 'normalize' every layer starts at 1, so the weights are "
        "shares.",
    ] = None

    distance: Annotated[
        Distance,
        "How this model's features are compared: L1, L2, and Cosine, L1Cosine and NCC (1 - their similarity), each "
        "0 at a perfect match; Dice (1 - soft Dice), 0 at a match only on binary maps such as one-hot labels "
        "(identical soft maps keep 1 - sum(p^2)/sum(p), raw features can take it below 0); LNCC (1 - the squared "
        "local correlation over 'lncc_kernel' voxels, on dense maps only: elastix, which draws points, refuses it).",
    ] = "L2"
    pca: Annotated[
        int,
        Range(0, 100),
        "Principal components each kept layer is reduced to, fitted on the fixed image (0 = keep every channel).",
    ] = 0
    subset_features: Annotated[
        int,
        Range(0, 1000),
        "Channels of each kept layer compared, drawn at random at every iteration (0 = all): cheaper iterations.",
    ] = 0
    voxel_size: Annotated[
        list[float] | None,
        "Resolution (mm) the image is resampled to before this model sees it, by linear interpolation. Absent, the "
        "model sees the image as it is. TotalSegmentator models hold at 1 to 2 mm and collapse at 4 to 6 mm.",
    ] = None
    feature_normalization: Annotated[
        FeatureNormalization,
        "How each voxel's feature vector is scaled, per kept layer and before the PCA: 'none', 'l2' (unit length, "
        "only the direction counts), 'standardized' (zero mean and unit deviation over the channels).",
    ] = "none"

    def __post_init__(self) -> None:
        # One number weighs every kept layer, as a one-value list does.
        if isinstance(self.layers_weight, int | float):
            self.layers_weight = [float(self.layers_weight)]


@dataclass
class LevelSpec:
    """The models of one level, in place of the preset's ``models`` there."""

    models: dict[str, ModelSpec]


# The loss settings every engine takes, with the same meaning.
Mode = Annotated[
    Literal["Static", "Jacobian"],
    "How the features reach the loss. 'Static' extracts them once per image (and again every "
    "'feature_map_update_interval' iterations) and warps the moving ones: fast, and exact for a local descriptor. "
    "'Jacobian' runs the network inside the loss on the warped image and differentiates through it: slower, and "
    "exact for any model.",
]
Normalize = Annotated[
    bool,
    "Divide each layer's loss by its value when a level starts, so every layer starts at 1 and layers_weight "
    "weighs comparable quantities (MIND and a segmentation decoder answer on ranges an order of magnitude apart); "
    "a layer that starts at 0 or below keeps its raw value. ConvexAdam's coarse search keeps its raw cost, its coupling schedule being absolute: balance_coarse_layers "
    "weighs its layers there.",
]
FeatureMapUpdateInterval = Annotated[
    int,
    Range(-1, 100000),
    "Static only: extract the moving features again from the moving image warped so far every this many "
    "iterations (0 or -1 = once per level).",
]
LNCCKernel = Annotated[
    int,
    Range(1, 31),
    "Side, in voxels of the compared feature map, of the window the LNCC distance correlates each channel over (odd). "
    "ConvexAdam counts it along the map's finest axis and takes the nearest odd count of the same length in mm along "
    "the others; FireANTs, a cube of this many voxels.",
]
MixedPrecision = Annotated[
    bool,
    "Run the feature models in float16 (their weights and input; the features come back in float32): lighter and "
    "faster extraction, the optimisation stays in float32.",
]
VoxelSampling = Annotated[
    float,
    Range(0.00001, 1.0),
    "Share of the voxels the loss reads at each iteration, drawn anew at random with 'seed' (1 = every voxel). "
    "Static compares the features at those voxels only; Jacobian runs the network on a patch of its receptive "
    "field around each of them, as elastix does, instead of on the whole warped image. Point-wise distances only "
    "(not LNCC). Experimental.",
]


def sorted_specs(mapping: dict) -> list:
    """A dict keyed by string indices ('0', '1', ...) -> its values in numeric order."""
    return [mapping[key] for key in sorted(mapping, key=int)]


def feature_model(spec: ModelSpec) -> ImpactFeatureModel:
    """KonfAI's feature model of ``spec``, its layers_mask the layer weights."""
    return ImpactFeatureModel.from_ref(spec.ref, [float(bit == "1") for bit in spec.layers_mask])


def kept_layers(spec: ModelSpec) -> int:
    return spec.layers_mask.count("1")


def per_kept_layer(specs: list[ModelSpec], value) -> list:
    """One entry per layer a model's layers_mask keeps, each model's ``value(spec)`` repeated over its own layers:
    the backends index distance, weight and PCA by kept layer across all the models."""
    return [value(spec) for spec in specs for bit in spec.layers_mask if bit == "1"]


def layer_weights(specs: list[ModelSpec]) -> list[float]:
    """Every kept layer's weight, a model's single value spread over all its kept layers."""
    weights: list[float] = []
    for spec in specs:
        given = spec.layers_weight or [1.0]
        weights += [float(given[0])] * kept_layers(spec) if len(given) == 1 else [float(weight) for weight in given]
    return weights


def check_models(specs: list[ModelSpec], engine: str, dense: bool) -> None:
    """Refuse, before anything runs, what no engine can do with these models, and what this one cannot:
    ``dense`` says whether the engine compares whole feature maps (LNCC) or points drawn at random."""
    if not specs:
        raise ValueError(f"{engine}: IMPACT needs at least one feature model under 'models'.")
    for spec in specs:
        name = f"{engine}: model '{spec.ref}'"
        kept = kept_layers(spec)
        if kept == 0:
            raise ValueError(f"{name}: layers_mask '{spec.layers_mask}' keeps no layer.")
        if set(spec.layers_mask) - {"0", "1"}:
            raise ValueError(f"{name}: layers_mask '{spec.layers_mask}' holds characters other than 0 and 1.")
        weights = spec.layers_weight or [1.0]
        if len(weights) not in (1, kept):
            raise ValueError(
                f"{name}: layers_weight has {len(weights)} values for {kept} kept layers: give one for all of them, "
                "or one per kept layer."
            )
        if any(not weight >= 0 for weight in weights):
            raise ValueError(f"{name}: layers_weight must be non-negative, got {weights}.")
        if spec.voxel_size is not None and any(not size > 0 for size in spec.voxel_size):
            raise ValueError(
                f"{name}: voxel_size must be positive, got {spec.voxel_size}; leave it out for the image as it is."
            )
        if spec.distance == "LNCC" and not dense:
            raise ValueError(
                f"{name}: the LNCC distance correlates windows of a dense feature map, which {engine}, drawing "
                "random points, does not have. Choose L1, L2, Cosine, L1Cosine, Dice or NCC."
            )


def level_models(models: dict[str, ModelSpec], levels: dict, count: int, engine: str) -> list[list[ModelSpec]]:
    """The models of each of the engine's ``count`` levels: ``models`` at every level, or ``levels`` ('0', '1',
    ...: a dict, as the config binds a list of blocks) one by one."""
    if not levels:
        return [sorted_specs(models)] * count
    if len(levels) != count:
        raise ValueError(
            f"{engine}: 'levels' has {len(levels)} entries for {count} levels; give one per level, or "
            "leave 'levels' out and use 'models' at every level."
        )
    return [sorted_specs(level.models) for level in sorted_specs(levels)]
