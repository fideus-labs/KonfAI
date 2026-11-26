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


"""Prompts on a grid: the points of a fiducial file on a volume's voxels, and the channels a prompted model reads.

A prompt is a point of the case in physical space (LPS), labelled with its kind: ``positive`` or ``negative`` for a
click, ``box`` for a corner of the box, in the order the user gave them (a ``.fcsv`` group, its kinds under the
``Labels`` attribute). ``PointsOnGrid`` puts the points of a group on a volume's grid; ``DrawPrompts`` draws them as
channels of the volume it is handed; ``Crop`` with ``points`` centres its window on the last of them. Each reads the
points as the case stores them and puts them on the grid it is handed itself.

**The interactions** are the prompts in their order: each click, and the box at its first corner's place. A click
sits on its point's voxel, rounded half to even; the box spans ``[low, high)``, its corners' voxels rounded the same
way, ``high`` one past the larger, clipped to the image as nnInteractive clips it (``interactions``).
"""

import json
from collections.abc import Callable
from typing import cast

import numpy as np
import torch

from konfai.data.transform.base import LocalityKind, PatchLocality, Transform, sitk
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError
from konfai.utils.ITK import _require_simpleitk

#: The kinds of point a prompted tool reads.
POSITIVE, NEGATIVE, BOX = "positive", "negative", "box"
KINDS = (POSITIVE, NEGATIVE, BOX)

#: How a continuous index becomes a voxel: half to even (``np.rint``, Python's ``round``), towards zero (``int()``),
#: or kept continuous.
ROUNDINGS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rint": np.rint,
    "trunc": np.trunc,
    "none": lambda voxels: voxels,
}

#: What each interaction weighs against the next under ``DrawPrompts(growth=True)``: nnInteractive's decay.
DECAY = 0.98

#: The largest radius of a click's ball, in voxels: nnInteractive's is 4.
MAX_RADIUS = 16


def on_grid(points: np.ndarray, attribute: Attribute, rank: int, rounding: str = "rint") -> np.ndarray:
    """``points`` ``[N, 3]`` in LPS as indices of the grid of ``rank`` axes ``attribute`` describes, ``[N, rank]`` in
    array order (z, y, x): SimpleITK's ``TransformPhysicalPointToContinuousIndex``, then ``rounding``. A grid of rank 2
    takes the first two coordinates; a header without a geometry is the identity grid."""
    _require_simpleitk()
    if rank not in (2, 3):
        raise TransformError(f"A prompt's points lie on a grid of 2 or 3 axes, and this one has {rank}.")
    if "Spacing" in attribute and len(attribute.get_np_array("Spacing")) != rank:
        raise TransformError(f"A grid of {rank} axes carries the spacing {attribute['Spacing']}.")
    header = sitk.Image([1] * rank, sitk.sitkUInt8)
    if "Origin" in attribute:
        header.SetOrigin(attribute.get_np_array("Origin").tolist())
        header.SetSpacing(attribute.get_np_array("Spacing").tolist())
        header.SetDirection(attribute.get_np_array("Direction").tolist())
    voxels = [
        header.TransformPhysicalPointToContinuousIndex(point[:rank].tolist())[::-1] for point in np.asarray(points)
    ]
    return ROUNDINGS[rounding](np.asarray(voxels, dtype=np.float64).reshape(-1, rank))


def read_prompts(stage: Transform, group: str, name: str) -> tuple[np.ndarray, list[str]]:
    """The case's points of ``group``, ``[N, 3]`` in LPS, and the kind of each, in the file's order."""
    holder = stage.dataset_holding(group, name)
    if holder is None:
        raise TransformError(f"No dataset holds the prompts '{group}' of case '{name}'.", "Give each case its .fcsv.")
    points, attribute = holder.read_data(group, name)
    kinds = json.loads(attribute["Labels"]) if "Labels" in attribute else []
    if (unknown := sorted(set(kinds) - set(KINDS))) or len(kinds) != len(points):
        raise TransformError(
            f"The prompts '{group}' of case '{name}' hold the kinds {unknown or kinds} for {len(points)} points.",
            f"Label every point {', '.join(KINDS)}.",
        )
    return np.asarray(points, dtype=np.float64), kinds


def interactions(
    voxels: np.ndarray, kinds: list[str], shape: list[int], image: bool = True
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The interactions of the points at ``voxels`` (rounded) on a volume of ``shape``, in order, each its kind and
    the voxels it spans, ``[low, high)``: a click its own (``high`` its voxel plus one), the box its corners', clipped
    to the volume. On the ``image``, an axis the clip leaves the box no voxel of takes one as nnInteractive's session
    gives it one (``_add_validated_bbox_interaction``): the image's first where the box lies before it, and one voxel
    back where it lies past it, the image's last for a box starting at its end, none for one starting further. On a
    window cut from the image the clip stands: a box the window leaves out spans none of it. Either way it weighs as
    an interaction."""
    if not np.all(np.isfinite(voxels)) or np.any(np.abs(voxels) > np.iinfo(np.int32).max):
        raise TransformError(
            f"The prompts lie at the indices {voxels.tolist()} of a volume of {list(shape)}.",
            "Give points on the volume's grid.",
        )
    voxels = voxels.astype(np.int64)
    corners = voxels[[kind == BOX for kind in kinds]]
    found: list[tuple[str, np.ndarray, np.ndarray]] = []
    for voxel, kind in zip(voxels, kinds, strict=True):
        if kind != BOX:
            found.append((kind, voxel, voxel + 1))
        elif not any(held == BOX for held, _, _ in found):
            low = np.maximum(corners.min(axis=0), 0)
            high = np.minimum(corners.max(axis=0) + 1, np.asarray(shape))
            if image:
                empty = high <= low
                high = np.where(empty & (low == 0), np.minimum(1, shape), high)
                low = np.where(empty & (low != 0), low - 1, low)
            found.append((BOX, low, high))
    if not found:
        raise TransformError("The prompts hold no point.", "Give the case at least one click or a box.")
    return found


def centre(kind: str, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """The voxel an interaction is centred on: a click's own, the box's ``(low + high) / 2`` rounded half to even, as
    nnInteractive's session centres its window on it (the box ``[z, z + 1)`` of one slice on ``z`` or ``z + 1``,
    whichever is even)."""
    return low if kind != BOX else np.rint((low + high) / 2).astype(np.int64)


def distance_ball(radius: int) -> np.ndarray:
    """nnInteractive's click (``build_point``): scikit-image's ``ball(radius)``, its voxels within ``radius`` of the
    centre, each valued by its Euclidean distance to the nearest voxel outside it (SciPy's ``distance_transform_edt``
    on the ball's cube), over the largest: 1 at the centre; the centre alone, at 1, for a radius of 0. The nearest
    voxel outside is one that touches the ball by a face (a step from any other towards the voxel inside is nearer),
    measured a slab of the ball at a time."""
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    inside = (grid**2).sum(axis=-1) <= radius * radius
    if inside.all():
        return np.ones(inside.shape)
    padded = np.pad(inside, 1)
    touching = np.zeros_like(inside)
    for along in range(3):
        for step in (-1, 1):
            touching |= np.roll(padded, step, axis=along)[1:-1, 1:-1, 1:-1]
    held, around = np.argwhere(inside), np.argwhere(~inside & touching)
    distance = np.zeros(inside.shape)
    for start in range(0, len(held), 256):
        slab = held[start : start + 256]
        distance[tuple(slab.T)] = np.sqrt(((slab[:, None] - around[None]) ** 2).sum(axis=-1).min(axis=1))
    return distance / distance.max()


def offset_of(image: Attribute, image_shape: list[int], handed: Attribute, shape: list[int], what: str) -> np.ndarray:
    """Where the grid ``image`` describes lies on the grid handed: the index on the handed grid of its first voxel, per
    axis in array order, the handed volume being that grid or a window cut from it (a ``Crop``, a pad included): the
    same spacing and direction, a whole number of voxels apart."""
    placed = "Origin" in image and "Origin" in handed
    if len(image_shape) != len(shape):
        raise TransformError(f"{what} is a volume of {list(image_shape)} for one of {list(shape)}.")
    at = on_grid(image.get_np_array("Origin")[None], handed, len(shape), "none")[0] if placed else np.zeros(len(shape))
    same = all(
        np.allclose(image.get_np_array(key), handed.get_np_array(key), rtol=1e-6, atol=1e-6)
        for key in ("Spacing", "Direction")
        if placed
    )
    if "Origin" not in image and "Origin" not in handed and list(image_shape) != list(shape):
        raise TransformError(
            f"{what} is a volume of {list(image_shape)} and the volume handed one of {list(shape)}, and neither carries"
            " a geometry that would place one on the other.",
            "Store both with their geometry.",
        )
    lone = ("Origin" in image) != ("Origin" in handed)
    if lone or not same or np.any(np.abs(at - np.rint(at)) > 1e-3):
        raise TransformError(
            f"{what} is a volume of {list(image_shape)} that the volume of {list(shape)} handed is not cut from.",
            "Resample it onto the image.",
        )
    return np.rint(at).astype(np.int64)


class PointsOnGrid(Transform):
    """The points of a fiducial group on the voxels of the volume ``reference``: ``[N, rank]`` indices in array order
    (z, y, x), rounded by ``rounding`` (``rint``, ``trunc`` or ``none``), on the grid ``reference`` is stored on.
    ``DrawPrompts`` and ``Crop`` put the points on the grid they are handed in the same way."""

    working_multiple = 0.0

    def __init__(self, reference: str = "Volume_0", rounding: str = "rint") -> None:
        super().__init__()
        if rounding not in ROUNDINGS:
            raise TransformError(
                f"'PointsOnGrid' was given rounding: {rounding!r}.", f"Give one of {', '.join(ROUNDINGS)}."
            )
        self.reference = reference
        self.rounding = rounding

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME, reason="a set of points has no grid to cut: it is read whole")

    def _reference(self, name: str) -> tuple[list[int], Attribute]:
        holder = self.dataset_holding(self.reference, name)
        if holder is None:
            raise TransformError(f"No dataset holds the volume '{self.reference}' of case '{name}'.")
        return holder.get_infos(self.reference, name)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # Each point's coordinates become one index per axis of the reference.
        return [len(self._reference(name)[0]) - 1]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape, attribute = self._reference(name)
        voxels = on_grid(tensor.detach().cpu().numpy(), attribute, len(shape) - 1, self.rounding)
        return torch.from_numpy(voxels)


class DrawPrompts(Transform):
    """The case's prompts drawn as channels after the volume's own, on the grid it is handed, as nnInteractive's
    session writes its interactions, on a 3-D volume.

    - ``points``: the group of the prompts.
    - ``previous``: the group of the last mask, on the image's grid: one channel first, 1 where it is above 0.5. The
      volume handed may be a window cut from that grid (a ``Crop`` around the prompts before this stage): the prompts
      are drawn at its place, nothing past the image, and the last mask is 0 there. Without it the volume handed
      stands for the image, and a box holding none of its voxels draws nothing: with it, such a box takes the voxel
      nnInteractive gives it (``interactions``).
    - ``channels``: the kind of the points each channel draws, ``""`` for one that stays empty. A click is a ball of
      ``radius`` voxels whose values fall from 1 at its voxel (``distance_ball``), cut where it leaves the image;
      the box is filled. Overlapping interactions keep the larger value. A point of a kind no channel draws is
      refused.
    - ``growth``: the interactions weigh 0.98 of the next, the last 1: the k-th of n drawn at ``(1 / 0.98)^k``
      (each the last times ``1 / 0.98``, from 1) and stored in float16 (rounded through float32, as torch casts a
      double), then divided by the n-th as nnInteractive's session normalises its float16 store on CUDA (times the
      float32 reciprocal of its float32 value, rounded to float16). Without it, every interaction at 1, in float32.
    """

    def __init__(
        self,
        points: str = "Volume_1",
        channels: list[str] = [BOX, "", POSITIVE, NEGATIVE],
        previous: str | None = None,
        radius: int = 4,
        growth: bool = False,
    ) -> None:
        super().__init__()
        if unknown := [kind for kind in channels if kind and kind not in KINDS]:
            raise TransformError(
                f"'DrawPrompts' draws the kinds {unknown}.", f"A channel draws one of {', '.join(KINDS)}, or ''."
            )
        if repeated := sorted({kind for kind in channels if kind and channels.count(kind) > 1}):
            raise TransformError(f"'DrawPrompts' draws the kinds {repeated} on two channels.", "Name each kind once.")
        if not 0 <= radius <= MAX_RADIUS:
            raise TransformError(
                f"'DrawPrompts' was given radius: {radius}.", f"A ball's radius in voxels is 0 to {MAX_RADIUS}."
            )
        self.points = points
        self.channels = list(channels)
        self.previous = previous
        self.radius = int(radius)
        self.growth = bool(growth)

    def output_channels(self, channels: int) -> int:
        return channels + (self.previous is not None) + len(self.channels)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A region's header does not say where it lies on the volume once a stage before it moved the grid (a Crop
        # around the prompts): drawn region by region there, the prompts would land off their place.
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME, reason="a prompt is placed on the volume's grid, which a region does not say"
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = list(tensor.shape[1:])
        if len(shape) != 3:
            raise TransformError(f"'DrawPrompts' draws on a 3-D volume, and case '{name}' is {shape}.")
        points, kinds = read_prompts(self, self.points, name)
        if missing := sorted(set(kinds) - set(self.channels)):
            raise TransformError(
                f"The prompts '{self.points}' of case '{name}' hold {missing} points, which no channel draws.",
                "Name their kind in channels, or leave them out of the file.",
            )
        # The image: the last mask's grid, which the volume handed may be a window of, at `offset` on it.
        image, offset, holder = cache_attribute, np.zeros(3, dtype=np.int64), None
        extent = np.asarray(shape)
        if self.previous is not None:
            holder = self.dataset_holding(self.previous, name)
            if holder is None:
                raise TransformError(f"No dataset holds the last mask '{self.previous}' of case '{name}'.")
            stored_shape, stored = holder.get_infos(self.previous, name)
            image, extent = stored, np.asarray(stored_shape[1:])
            offset = offset_of(image, list(extent), cache_attribute, shape, f"The last mask '{self.previous}/{name}'")
        drawn = np.zeros((len(self.channels), *shape), dtype=np.float32 if not self.growth else np.float16)
        ball = distance_ball(self.radius)
        intensity = 1.0
        # Without the last mask the volume handed may be a window a Crop cut around the prompts, which no header tells
        # from the image: the box is clipped to it, and nnInteractive's voxel for a box off the image is not drawn.
        on_image = self.previous is not None
        for kind, low, high in interactions(on_grid(points, image, 3), kinds, list(extent), on_image):
            if self.growth:
                intensity *= 1 / DECAY
                if intensity > float(np.finfo(np.float16).max):
                    raise TransformError(
                        f"The prompts '{self.points}' of case '{name}' weigh more than float16 holds.",
                        "nnInteractive rescales its interactions past it; give fewer.",
                    )
            reach = 0 if kind == BOX else self.radius
            # In the image's indices, cut to the image, then to the volume handed at its place.
            first = np.maximum(np.maximum(low - reach, 0) + offset, 0)
            end = np.minimum(np.minimum(high + reach, extent) + offset, shape)
            if np.any(end <= first):
                continue
            if kind == BOX:
                value = np.full((end - first).tolist(), intensity)
            else:
                start = first - offset - (low - self.radius)
                value = ball[tuple(slice(int(a), int(a + b - c)) for a, b, c in zip(start, end, first, strict=True))]
                value = value * intensity
            channel = drawn[self.channels.index(kind)]
            region = tuple(slice(int(a), int(b)) for a, b in zip(first, end, strict=True))
            channel[region] = np.maximum(channel[region], value.astype(np.float32).astype(channel.dtype))
        if self.growth:
            drawn = (drawn.astype(np.float32) * (np.float32(1.0) / np.float32(intensity))).astype(np.float16)
        layers = [tensor.float()]
        if holder is not None:
            # Only the part of the last mask the volume handed covers is read.
            plane = np.zeros(shape, dtype=np.float32)
            first, end = np.maximum(offset, 0), np.minimum(offset + extent, shape)
            if np.all(end > first):
                kept = tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(first, end, offset, strict=True))
                last, _ = holder.read_data_slice(cast("str", self.previous), name, (slice(0, 1), *kept))
                plane[tuple(slice(int(a), int(b)) for a, b in zip(first, end, strict=True))] = np.asarray(last)[0] > 0.5
            layers.append(torch.from_numpy(plane[None]))
        layers.append(torch.from_numpy(drawn.astype(np.float32)))
        # Drawn on the host, beside the volume wherever it is held.
        return torch.cat([layer.to(tensor.device) for layer in layers])
