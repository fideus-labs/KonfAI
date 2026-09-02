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

"""What every built-in stage is exercised with, for the two properties that ask the same question.

A stage must behave as if it had seen the whole volume: proven on the READ side, patch by patch
(``test_transform_locality_contract``), and on the WRITE side, region by region over dtype, rank and
budget (the ``test_streamed_oracle_*`` family). Both need the same two things, so both read them here: the
enumeration of the built-ins with one representative configuration each, and a case on disk holding
one volume per input kind those configurations consume.

The case is built from a :class:`Geometry`: the extents, spacings, origins and direction cosines of
the case's own grid and of the grids its stages read beside it. ``FIXED_GEOMETRY`` is the one the
locality contract cuts its patch grid on; :func:`seeded_geometry` draws the same structure from a
seed, which is how the oracle varies rank and geometry without varying anything else.
"""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data import augmentation as augmentation_module
from konfai.data import transform as transform_module
from konfai.data.augmentation import DataAugmentation
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import DatasetManager, DatasetPatch, SweepSegment
from konfai.data.transform import (
    Argmax,
    Canonical,
    Clip,
    Crop,
    Dilate,
    FlatLabel,
    Flip,
    Foreign,
    Gradient,
    HistogramMatching,
    InferenceStack,
    LocalityKind,
    Mask,
    MergeLabels,
    OneHot,
    Padding,
    Percentage,
    Permute,
    Reduce,
    Resample,
    Save,
    SegmentationDisagreement,
    SelectLabel,
    Softmax,
    Squeeze,
    StandardDeviation,
    Sum,
    Transform,
    Variance,
    Write,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError, TransformError

CASE_NAME = "CASE_000"

#: The bone/air step of the phantom, in the units a float group is stored in. The deviation an
#: interpolation can show scales with the NEIGHBOUR GAP, so this is what the resample bound is ulps
#: of, and a smooth fixture would assert those bounds against data that cannot exercise them.
PEAK = 450.0

# Byte-identical is the norm: a streamable transform reads exactly the voxels the whole-volume pass
# reads and does the same arithmetic on them. Two kinds legitimately round differently:
#
# - the seeded statistic: streaming seeds Mean/Std from `read_data_statistics` (a numpy pass over the
#   stored volume) while the whole-volume path recomputes them with torch over the loaded tensor: the
#   same values summed in a different order, so a standardized (unit-scale) voxel may land a few
#   float32 ulps away. Data-dependent: this fixture happens to agree exactly, a smooth field showed
#   1.5e-8 (0.13 ulp), so the bound is stated rather than observed.
STAT_ATOL = 8 * float(np.finfo(np.float32).eps)

#: What two routes computing the same expression over different extents may differ by, relative to
#: each value: a few float32 ulps.
VECTORISED_RTOL = 8 * float(np.finfo(np.float32).eps)
# - a linear resample through a map that does NOT factorise (a field, a rotation), ON THE DEVICE:
#   the coordinate walk keeps global float64 coordinates and is bit-identical slab for slab, but the
#   CUDA blend goes through grid_sample, which normalises the coordinates by the extent it is handed,
#   and a region is handed a window. A weight lands ~ulp(coordinate) off and the voxel
#   `neighbour gap * ulp(coordinate)` off: the deviation scales with the local GRADIENT. The
#   fixture's gap is its 2*PEAK bone/air step, which puts the bound at ulps of PEAK; 64 of them is
#   ~8x the measured max and far below one part per million of the range. On the HOST the same case
#   goes through ITK's own resampler on a window at its true origin and is bit-identical on an
#   axis-aligned volume (measured: zero differing voxels on every Resample case, field and stored
#   map included); on OBLIQUE cosines the window's origin is one rounding the whole volume never
#   takes, and a voxel in ~1e5 lands an ulp apart. A map that DOES factorise is read one axis at a
#   time on global coordinates, so the two routes sum the same weights over different extents: equal
#   bit for bit where the compiler emits the same arithmetic for both, an ulp of the local gap apart
#   where it contracts or vectorises them differently. Nearest uses no weights and is exact either
#   way; cubic walks the corners itself and is exact.
REGRID_ATOL = 64 * float(np.spacing(np.float32(PEAK)))
# What one rounding of a value at the volume's scale is worth. `low + w * (high - low)` may be issued
# as a fused multiply-add or as a multiply then an add, and which of the two a CPU kernel picks for a
# given element depends on the extent it is walking: the vectorised body and the scalar tail need not
# agree, and the tail is where the count lands. One ulp on a third of the voxels, measured; four is
# the bound, so a deviation that is not a single rounding still fails.
FUSED_ATOL = 4 * float(np.spacing(np.float32(PEAK)))
# An integer volume truncates the interpolation, so a sub-ulp disagreement that straddles an integer
# boundary becomes a whole least-significant bit.
LSB_ATOL = 1.0
# A HALO draw is sampled by grid_sample from coordinates expressed in the halo'd read extent's frame
# rather than the whole volume's: the same disagreement, for the same reason and with the same
# gradient- and coordinate-scaling, that REGRID_ATOL bounds for the device resample. It is bitwise
# on neither, and grows with the extent: a 160^3 case at patch 64 lands at 2e-5 of its range.
AUGMENTATION_ATOL = REGRID_ATOL


# --------------------------------------------------------------------------------------
# The case on disk: one volume per input kind, on a geometry the caller chooses.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Every grid a case is built on: its own, and the ones its stages read beside it.

    ``oblique`` and ``permuting`` are the direction cosines the groups of those names are stored
    with, the case's own being axis-aligned: a reorientation is then an exact index remap on the
    first two and a resample on the third, which is what an image-dependent declaration keys on.
    """

    extents: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    oblique: np.ndarray
    permuting: np.ndarray
    reference_extents: tuple[int, ...]
    reference_spacing: tuple[float, ...]
    reference_origin: tuple[float, ...]
    field_extents: tuple[int, ...]
    field_spacing: tuple[float, ...]
    field_origin: tuple[float, ...]
    coarse_field_extents: tuple[int, ...]
    coarse_field_spacing: tuple[float, ...]
    #: The foreground box the "Boxed" group is stored with: [start, after] margins per spatial axis.
    box: np.ndarray
    #: The rigid map stored beside the case, as (centre, angles, translation) in world units.
    stored_map: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]

    @property
    def rank(self) -> int:
        return len(self.extents)


def _rotation(rank: int, angles: tuple[float, ...]) -> np.ndarray:
    """An orthonormal direction that mixes axes: one plane rotation in rank 2, three in rank 3."""
    matrix = np.eye(rank)
    planes = [(0, 1)] if rank == 2 else [(0, 1), (0, 2), (1, 2)]
    for (first, second), angle in zip(planes, angles, strict=True):
        plane = np.eye(rank)
        plane[first, first] = plane[second, second] = np.cos(angle)
        plane[first, second], plane[second, first] = -np.sin(angle), np.sin(angle)
        matrix = matrix @ plane
    return matrix


#: The geometry the locality contract cuts its patch grid on. No extent is a multiple of its patch
#: size, so the last patch of every axis is a border patch the read plan has to pad: the grid is
#: 3x3x3 and 19 of its 27 patches touch a border. The reference grid overlaps the case WITHOUT
#: NESTING (in physical x it runs to 13.4 where the case stops at 12.0), so its last columns read
#: from outside the case and take the fill: a reference contained in its case would prove the
#: sampler and never the boundary, which is the half that differs between the two paths. The field
#: is COARSER than either, which is how one is actually solved: it is in world units, so it is read
#: where it is asked rather than resampled to match anything first.
#: World units the coarse field displaces by, reversing sign between adjacent nodes.
_COARSE_FIELD_BOUND = 9.0

FIXED_GEOMETRY = Geometry(
    extents=(9, 10, 11),
    spacing=(1.5, 1.5, 2.0),
    origin=(-3.0, 5.0, 11.0),
    oblique=_rotation(3, (np.deg2rad(20.0), 0.0, 0.0)),
    permuting=np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    reference_extents=(7, 8, 9),
    reference_spacing=(1.8, 1.2, 2.5),
    reference_origin=(-1.0, 6.0, 12.0),
    field_extents=(4, 5, 5),
    field_spacing=(3.0, 2.4, 4.0),
    field_origin=(-4.0, 4.0, 10.0),
    coarse_field_extents=(3, 3, 3),
    # Two cells that COVER the case from the field origin one source voxel before it: the swing
    # stays a full amplitude per cell, and no target region falls outside the lattice.
    coarse_field_spacing=(9.0, 8.2, 9.8),
    # Both margins differ on every axis: a box symmetric anywhere is one a wrong sign or a reversed
    # axis order would still land on. (9, 10, 11) is cropped to (6, 6, 6), which still carries a
    # patch grid to disagree over.
    box=np.asarray([[2, 1], [1, 3], [3, 2]]),
    stored_map=((3.0, 5.0, 11.0), (0.05, -0.03, 0.08), (0.4, -0.6, 1.1)),
)


def seeded_geometry(seed: int, rank: int) -> Geometry:
    """The same structure, drawn: anisotropic spacings, an oblique direction, extents 16 to 40.

    The reference grid keeps FIXED_GEOMETRY's property by construction: it starts a fifth of the way
    into the case and is at most a twentieth longer, so it overlaps without nesting whatever the
    draw. Asserted where it is used, over every geometry this module ships.
    """
    rng = np.random.default_rng(seed)
    extents = tuple(int(rng.integers(16, 41)) for _ in range(rank))
    spacing = tuple(float(rng.uniform(0.5, 3.0)) for _ in range(rank))
    origin = tuple(float(rng.uniform(-20.0, 20.0)) for _ in range(rank))
    world = np.asarray(extents, dtype=np.float64)[::-1] * np.asarray(spacing)

    reference_spacing = tuple(float(value * rng.uniform(0.8, 1.6)) for value in spacing)
    reference_world = world * rng.uniform(0.85, 1.05, rank)
    reference_extents = tuple(
        int(max(6, round(reference_world[rank - 1 - axis] / reference_spacing[rank - 1 - axis])))
        for axis in range(rank)
    )
    reference_origin = tuple(float(value + 0.2 * world[index]) for index, value in enumerate(origin))

    field_spacing = tuple(float(value * rng.uniform(2.0, 3.0)) for value in spacing)
    field_extents = tuple(
        int(np.ceil(world[rank - 1 - axis] / field_spacing[rank - 1 - axis]) + 1) for axis in range(rank)
    )
    field_origin = tuple(float(value - spacing[index]) for index, value in enumerate(origin))

    # Three nodes an axis, spaced so the two cells COVER the case from the field origin one source
    # voxel before it: with the sign reversing between nodes, the interpolated displacement still
    # swings the full amplitude across one cell, and no target region falls outside the lattice.
    coarse_field_spacing = tuple(
        float((world[index] + 2.0 * spacing[index]) / 2.0 * rng.uniform(1.0, 1.1)) for index in range(rank)
    )
    coarse_field_extents = (3,) * rank

    permutation = rng.permutation(rank)
    while np.array_equal(permutation, np.arange(rank)):
        permutation = rng.permutation(rank)
    return Geometry(
        extents=extents,
        spacing=spacing,
        origin=origin,
        oblique=_rotation(rank, tuple(float(rng.uniform(0.1, 0.6)) for _ in range(1 if rank == 2 else 3))),
        permuting=np.eye(rank)[permutation],
        reference_extents=reference_extents,
        reference_spacing=reference_spacing,
        reference_origin=reference_origin,
        field_extents=field_extents,
        field_spacing=field_spacing,
        field_origin=field_origin,
        coarse_field_extents=coarse_field_extents,
        coarse_field_spacing=coarse_field_spacing,
        box=np.asarray([[int(rng.integers(1, 4)), int(rng.integers(1, 4))] for _ in range(rank)]),
        stored_map=(
            tuple(float(value + 0.5 * world[index]) for index, value in enumerate(origin)),
            tuple(float(rng.uniform(-0.1, 0.1)) for _ in range(1 if rank == 2 else 3)),
            tuple(float(rng.uniform(-1.5, 1.5)) for _ in range(rank)),
        ),
    )


def contrast(dtype: np.dtype) -> tuple[float, float]:
    """The bone/air step the phantom is built with, inside what the dtype can hold.

    An unsigned store cannot carry a negative air value, and a byte cannot carry PEAK: a phantom
    clipped into its dtype would be a constant, and a constant proves nothing about an interpolation.
    """
    if np.issubdtype(dtype, np.floating) or np.dtype(dtype).itemsize >= 4:
        return -PEAK, PEAK
    information = np.iinfo(dtype)
    if information.min < 0:
        return -PEAK, PEAK
    span = float(information.max)
    return 0.05 * span, 0.95 * span


def volumes(geometry: Geometry, dtype: np.dtype = np.dtype(np.float32)) -> dict[str, np.ndarray]:
    """One volume per input kind the built-in stages consume, all on ``geometry``.

    ``dtype`` is the intensity groups' storage dtype: the label, ensemble and reference groups keep
    the dtype their own meaning implies, since a label map stored as float64 is not a case anything
    reads that way.
    """
    rng = np.random.default_rng(0)
    axes = np.meshgrid(*[np.linspace(-1, 1, extent) for extent in geometry.extents], indexing="ij")
    # A CT-like phantom rather than a smooth field: the bone/air step is what makes an interpolation
    # disagree at all (the deviation scales with the neighbour gap), so a smooth fixture would assert
    # the resample tolerances against data that cannot exercise them.
    radius = sum(axis**2 for axis in axes)
    air, bone = contrast(dtype)
    # A thirtieth of the step: enough noise that no two neighbours agree, far short of erasing it.
    intensity = np.where(radius < 0.4, bone, air) + (bone - air) / 30.0 * rng.standard_normal(geometry.extents)
    if np.issubdtype(dtype, np.integer):
        information = np.iinfo(dtype)
        intensity = np.clip(intensity, information.min, information.max)
    return {
        "Intensity": intensity.astype(dtype)[None],
        # How a CT is actually stored, and the one dtype that quantizes an interpolation (see LSB_ATOL).
        "Int16": intensity.astype(np.int16)[None],
        # Nested structures, not uniform noise: dilating scattered foreground saturates to all-ones,
        # and an all-ones result is the same whether or not the halo was ever read. Compact labels
        # keep a border for Dilate's halo to be wrong about.
        "Labels": np.select([radius < 0.1, radius < 0.2, radius < 0.35], [1, 2, 3], 0).astype(np.uint8)[None],
        "Ensemble": rng.integers(0, 3, (3, *geometry.extents)).astype(np.float32),
        # The same intensities stored on a rotated direction: the one group whose METADATA, not whose
        # voxels, is the point (see test_a_declaration_reads_the_case_metadata).
        "Oblique": intensity.astype(dtype)[None],
        # And on a direction that permutes axes, which reorienting transposes the extents of: the group
        # whose metadata moves the patch grid the streamed patches are cut on.
        "Permuting": intensity.astype(dtype)[None],
        # Stored with the foreground box already on it, which is how a crop is a translation rather
        # than a question about the voxels.
        "Boxed": intensity.astype(dtype)[None],
        # A grid of its OWN (other extent, other spacing, other origin), for a stage that resamples
        # onto a reference rather than about the case's own extent. Only its header is ever read.
        "Reference": rng.standard_normal(geometry.reference_extents).astype(np.float32)[None],
        # A displacement field, component-first in physical (x, y, z), each component a different
        # function so a reversed axis order cannot pass unnoticed.
        "Field": _field(geometry),
        # A SECOND field, coarse against the case and violent between its nodes: the first is
        # gentle (two world units at most) and a window sized from the values read over a
        # region's own box is exact for it. Whether it stays exact where the interpolation at a
        # region's FACE blends nodes the box does not contain is a different question, and this
        # is the field that asks it. Sized after a real case: an ExaSPIM registration field is
        # four times coarser than the volume it moves.
        "CoarseField": _coarse_field(geometry),
    }


def _field(geometry: Geometry) -> np.ndarray:
    """The displacement field: physical component ``index`` varies along the array axis it maps to.

    No component reaches beyond two world units, which is what a halo read through the field is
    sized from and checked against.
    """
    components = []
    for index, amplitude in enumerate((2.0, 1.5, 1.0)[: geometry.rank]):
        axis = geometry.rank - 1 - index  # physical x is the LAST array axis
        ramp = np.arange(geometry.field_extents[axis])
        wave = np.cos(ramp) if index % 2 == 0 else np.sin(ramp)
        shape = [1] * geometry.rank
        shape[axis] = geometry.field_extents[axis]
        components.append(amplitude * wave.reshape(shape) * np.ones(geometry.field_extents))
    return np.stack(components).astype(np.float32)


def _coarse_field(geometry: Geometry) -> np.ndarray:
    """A displacement field whose sign REVERSES from one node to the next, at several world units.

    The steepest thing a lattice can carry: the interpolated displacement swings the full amplitude
    across one cell, so a region's face cuts through the middle of that swing. A field read over a
    region's own box bounds every displacement INSIDE it, which is what
    ``Resample.measured_region_source`` relies on; this asks whether the bound still holds at the
    faces, where the interpolator blends nodes from outside the box.
    """
    extents = geometry.coarse_field_extents
    parity = (-1.0) ** sum(
        np.arange(extent).reshape([-1 if axis == index else 1 for axis in range(geometry.rank)])
        for index, extent in enumerate(extents)
    )
    return np.stack([weight * _COARSE_FIELD_BOUND * parity for weight in (1.0, -0.7, 0.4)[: geometry.rank]]).astype(
        np.float32
    )


def manager(
    dataset: Dataset,
    transforms: list[Transform],
    group: str = "CT",
    name: str = CASE_NAME,
    patch: DatasetPatch | None = None,
    index: int = 0,
) -> DatasetManager:
    """One case's manager over ``dataset``: the same group in and out, no augmentation."""
    return DatasetManager(
        index=index,
        group_src=group,
        group_dest=group,
        name=name,
        dataset=dataset,
        patch=patch,
        transforms=transforms,
        data_augmentations_list=[],
    )


def geometry(
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    direction: np.ndarray | Sequence[float] | None = None,
) -> Attribute:
    """The stored geometry of a volume: ``(x, y, z)`` origin and spacing, a flattened direction
    (identity when not given)."""
    attribute = Attribute()
    attribute["Origin"] = np.asarray(origin, dtype=np.float64)
    attribute["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attribute["Direction"] = (
        (np.eye(len(origin)) if direction is None else np.asarray(direction)).astype(np.float64).reshape(-1)
    )
    return attribute


def attributes(geometry: Geometry, group: str) -> Attribute:
    """The metadata a group is stored with, and so what a declaration about it is handed."""
    attribute = Attribute()
    origins = {
        "Reference": geometry.reference_origin,
        "Field": geometry.field_origin,
        "CoarseField": geometry.field_origin,
    }
    spacings = {
        "Reference": geometry.reference_spacing,
        "Field": geometry.field_spacing,
        "CoarseField": geometry.coarse_field_spacing,
    }
    directions = {"Oblique": geometry.oblique, "Permuting": geometry.permuting}
    attribute["Origin"] = np.asarray(origins.get(group, geometry.origin))
    attribute["Spacing"] = np.asarray(spacings.get(group, geometry.spacing))
    attribute["Direction"] = directions.get(group, np.eye(geometry.rank)).reshape(-1)
    if group == "Ensemble":
        # What a `combine: Concat` reduction writes: the per-model channel counts MergeLabels and
        # Sum shift their label ranges by.
        attribute["number_of_channels_per_model"] = np.asarray([3, 3, 3])
    if group == "Boxed":
        # What Crop.transform_shape leaves on the case: [start, after] margins per spatial axis.
        attribute["box"] = geometry.box
    return attribute


def build_case(root: Path, geometry: Geometry, dtype: np.dtype = np.dtype(np.float32)) -> Dataset:
    """A real on-disk dataset, in the same format (mha) and channel-first layout a run reads."""
    import SimpleITK as sitk

    dataset = Dataset(root, "mha")
    for group, volume in volumes(geometry, dtype).items():
        dataset.write(group, CASE_NAME, volume, attributes(geometry, group))
    # A stored transform, as a group of its own: KonfAI reads one from `<case>/<group>.itk.txt`,
    # which is what SimpleITK writes. Without it the resample-through-a-stored-map case has nothing
    # to apply and drops out of the sweep, which is the one kind of hole a registry cannot show.
    centre, angles, translation = geometry.stored_map
    stored = sitk.Euler3DTransform() if geometry.rank == 3 else sitk.Euler2DTransform()
    stored.SetCenter(centre)
    stored.SetTranslation(translation)
    if geometry.rank == 3:
        stored.SetRotation(*angles)
    else:
        stored.SetAngle(angles[0])
    (root / CASE_NAME).mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(stored, str(root / CASE_NAME / "transform.itk.txt"))
    return dataset


# --------------------------------------------------------------------------------------
# The enumeration: every built-in, at one representative configuration.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCase:
    """One representative configuration of a transform, and the fixture group it consumes."""

    transform: Transform
    group: str = "Intensity"
    atol: float = 0.0
    #: A bound on the value's own scale, for a stage whose arithmetic the compiler is free to
    #: vectorise differently depending on the extent it is handed: the decomposition changes that
    #: extent, so the two routes agree to an ulp of each value rather than bit for bit.
    rtol: float = 0.0
    #: In the equivalence sweeps. False for a case whose inputs this registry cannot build (a field
    #: on disk): its streamed-equals-whole proof lives in its own test file, against real inputs.
    sweep: bool = True


def stage_cases(rank: int = 3) -> dict[str, list[StageCase]]:
    """Only the transforms whose defaults are not a meaningful streaming case: a channel reduction
    needs a multi-model input, a label op needs labels, and a required constructor argument has no
    default at all. Built per call: a transform carries state (its datasets, its statistics), so two
    properties over the registry must not share instances.

    ``rank`` cuts the per-axis arguments (a target spacing, a target shape, a padding) to the case's
    own rank, so one table describes a 2-D case and a 3-D one.
    """
    spacing, shape = [2.0, 1.0, 3.0][:rank], [12, 8, 14][:rank]
    # KonfAI's stored-map codec holds the 3-D rigid, affine and BSpline kinds only, so a 2-D case
    # has no stored map to resample through: the refusal is pinned where it is raised, not swept.
    stored_map = rank == 3
    # The axis order Flip and Permute default to, cut to the rank: "1|0|2" in 3-D, "1|0" in 2-D.
    axes = "|".join(str(axis) for axis in [1, 0, 2][:rank])
    return {
        "Argmax": [StageCase(Argmax(0), group="Ensemble")],
        # The defaults reorient the axis-aligned group, which is a mirroring. A PERMUTING direction is
        # the other exact remap, and the only one that transposes extents: its patches are cut on a
        # grid the source volume does not have, so its source region is a permuted slice tuple rather
        # than the target's own. Both must stream, and a sagittal or coronal acquisition is the second.
        "Canonical": [StageCase(Canonical()), StageCase(Canonical(), group="Permuting")],
        "Clip": [
            StageCase(Clip(-200.0, 300.0)),  # fixed bounds: POINTWISE (the default range does not clip)
            StageCase(Clip("min", "max")),  # data-dependent bounds: GLOBAL_STAT
        ],
        # Only a case whose box is already stored is a translation; without one there is nothing to
        # declare but the read that would find it.
        "Crop": [StageCase(Crop(), group="Boxed")],
        "Dilate": [StageCase(Dilate(2), group="Labels")],
        # A class from another framework, as the loader wraps one: callable on a tensor, returning it
        # transformed. torch.nn is the one such library KonfAI already depends on.
        "Foreign": [StageCase(Foreign(torch.nn.Sigmoid(), "torch.nn:Sigmoid"))],
        "FlatLabel": [StageCase(FlatLabel([1, 3]), group="Labels")],
        # Their defaults name three axes, which a 2-D case does not have.
        "Flip": [StageCase(Flip(axes))],
        "Permute": [StageCase(Permute(axes))],
        # A magnitude is a sum of squares under a square root, and the region's extent decides how
        # the compiler vectorises it: one voxel in 627 differed by 1.6e-07 of its own value between
        # a one-row decomposition and the whole volume on a CI runner.
        "Gradient": [
            StageCase(Gradient(), rtol=VECTORISED_RTOL),
            StageCase(Gradient(per_dim=True), rtol=VECTORISED_RTOL),
        ],
        "HistogramMatching": [StageCase(HistogramMatching("Intensity"))],
        "InferenceStack": [StageCase(InferenceStack("Dataset", "model"))],
        # A companion volume aligned with the case: the mask is a stored group of the same grid.
        "Mask": [StageCase(Mask(path="Labels", value_outside=-7))],
        "MergeLabels": [StageCase(MergeLabels(), group="Ensemble")],
        "OneHot": [StageCase(OneHot(4), group="Labels")],
        # Pads reaching both borders of every axis, asymmetric and wider than the patch, constant and
        # reflect: the fill and the mirror both need the source rows the clamp cut back to.
        "Padding": [
            StageCase(Padding(padding=[1, 2, 3, 0, 2, 4][: 2 * rank], mode="constant:-3")),
            StageCase(Padding(padding=[2, 1, 1, 2, 3, 2][: 2 * rank], mode="reflect")),
            StageCase(Padding(padding=[0, 0, 0, 0, 0, 0][: 2 * rank])),
            StageCase(Padding([1, 2, 3, 4, 5, 6][: 2 * rank])),
            StageCase(Padding([5, 0, 0, 5, 2, 2][: 2 * rank], mode="constant:-7")),
        ],
        "Percentage": [StageCase(Percentage(100.0))],
        # The default (the case's own grid, no map) would be a no-op resample; these are the family's
        # meaningful configurations, one per way of naming the grid and the map.
        "Resample": [
            StageCase(Resample(spacing=spacing), atol=FUSED_ATOL),  # factorises
            StageCase(Resample(spacing=spacing), group="Int16", atol=LSB_ATOL),
            # uint8 resamples by nearest neighbour: no interpolation weights, so no rounding to disagree on.
            StageCase(Resample(spacing=spacing), group="Labels"),
            StageCase(Resample(shape=shape), atol=FUSED_ATOL),  # factorises
            # Onto a grid of its own, so part of the target reads from outside the case and takes the
            # fill. Both paths run the SAME sampler over global coordinates.
            StageCase(Resample(reference=CASE_NAME, reference_group="Reference"), atol=FUSED_ATOL),
            StageCase(Resample(reference=CASE_NAME, reference_group="Reference"), group="Labels"),
            # Through a field, and through a stored map: neither factorises. Exact on the host (ITK's
            # resampler), ~ulp on the device (grid_sample's normalised coordinates): the atol says so.
            StageCase(
                Resample(reference=CASE_NAME, reference_group="Reference", field_group="Field"),
                atol=REGRID_ATOL,
            ),
            # The same stage against a field that is COARSE and STEEP. A region sizes its source
            # window from the field values inside its own box, which bounds every displacement the
            # box contains; at the box's FACES the interpolator blends nodes from outside it, and a
            # field that reverses sign between adjacent nodes is what makes the two differ.
            StageCase(
                Resample(reference=CASE_NAME, reference_group="Reference", field_group="CoarseField"),
                atol=REGRID_ATOL,
            ),
            StageCase(Resample(transforms={"transform": True}), atol=REGRID_ATOL, sweep=stored_map),
            # The same on an OBLIQUE case: a region's origin is the volume's map applied to its start,
            # one more rounding than the whole volume's index-to-world takes, and on oblique cosines
            # that rounding is not exact -- so a handful of voxels (measured: 1 in 64 in one patch,
            # ~1e-5 of a volume) land an ulp apart, on the host as on the device; the same bound holds.
            # A nearest pick on the fixture stays exact (the deviation is far from any .5 boundary).
            StageCase(Resample(spacing=spacing), group="Oblique", atol=REGRID_ATOL),
            StageCase(Resample(reference=CASE_NAME, reference_group="Reference"), group="Oblique", atol=REGRID_ATOL),
            StageCase(
                Resample(reference=CASE_NAME, reference_group="Reference", interpolation="nearest"), group="Oblique"
            ),
            # Keys' cubic: the separable axis walk and the corner walk both keep GLOBAL coordinates, so
            # streamed and whole agree exactly on floats; int16 rounds the blend once, hence the LSB.
            StageCase(Resample(spacing=spacing, interpolation="cubic")),
            StageCase(Resample(spacing=spacing, interpolation="cubic"), group="Int16", atol=LSB_ATOL),
            StageCase(Resample(transforms={"transform": True}, interpolation="cubic"), sweep=stored_map),
            # A field on disk this registry cannot build: the streamed-equals-whole proof lives in
            # test_warp.py, where a field exists: including the measured, bound-less windows.
            StageCase(Resample(field="Dataset:h5", field_group="DVF"), sweep=False),
        ],
        "Save": [StageCase(Save("Dataset"))],
        # Reduce is a cardinality marker the cohort engine splits out of the chain, never a per-case
        # stage: it declares WHOLE_VOLUME so a chain reaching the ordinary planner refuses rather than
        # streams, which is what puts it out of the equivalence sweep below.
        "Reduce": [StageCase(Reduce(output="reduced"))],
        # Write is Save with a required destination: same boundary, same WHOLE_VOLUME declaration.
        "Write": [StageCase(Write("Dataset"))],
        "SegmentationDisagreement": [StageCase(SegmentationDisagreement(), group="Ensemble")],
        "SelectLabel": [StageCase(SelectLabel(["(1,2)", "(3,1)"]), group="Labels")],
        "Softmax": [StageCase(Softmax(0), group="Ensemble")],
        "Squeeze": [StageCase(Squeeze(0))],
        "Standardize": [StageCase(transform_module.Standardize(), atol=STAT_ATOL)],
        "StandardDeviation": [StageCase(StandardDeviation(), group="Ensemble")],
        "Sum": [StageCase(Sum(0), group="Ensemble")],
        "Variance": [StageCase(Variance(), group="Ensemble")],
    }


def builtin_transforms() -> list[type[Transform]]:
    """Every concrete transform class KonfAI ships."""
    return [
        cls
        for _, cls in inspect.getmembers(transform_module, inspect.isclass)
        if issubclass(cls, Transform)
        and cls.__module__.startswith(transform_module.__name__)
        and not inspect.isabstract(cls)
    ]


def cases_of(cls: type[Transform], rank: int = 3) -> list[StageCase]:
    """The configurations to exercise for one transform: the table's, else the transform's own defaults."""
    registry = stage_cases(rank)
    if cls.__name__ in registry:
        return registry[cls.__name__]
    if any(parameter.default is parameter.empty for parameter in inspect.signature(cls).parameters.values()):
        return []
    try:
        return [StageCase(cls())]
    except TransformError:
        # A transform can carry a default for every parameter and still refuse the combination:
        # `Reduce` defaults `output` to "" and rejects it, because a reduction has no case name to
        # inherit. Such a transform cannot be exercised from its defaults, it belongs in the table
        # above, or nowhere. Enumerating it here turned that refusal into a collection error, which
        # takes the whole file down rather than skipping one transform.
        return []


# --------------------------------------------------------------------------------------
# The augmentations. Same contract, same property: asked of a draw rather than a config.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationCase:
    """One representative draw, the kind it declares, and whether the dispatcher reads it.

    Those are two questions, and keeping them apart is the contract's own split: a draw declares what
    its output depends on, and the dispatcher decides whether reading that much is worth it. A wide
    Translate declares a perfectly honest HALO and is refused all the same (see ``_affords_halo``).
    """

    augmentation: DataAugmentation
    kind: LocalityKind
    streams: bool
    group: str = "Intensity"
    atol: float = 0.0


def augmentation_cases() -> dict[str, list[AugmentationCase]]:
    """Every augmentation KonfAI ships, at a draw that is a meaningful streaming case: probabilities
    are pinned so the draw always fires (a copy no draw selected is the identity, which proves
    nothing), and a required constructor argument is given a value. The kind is what THIS draw must
    declare, so a declaration silently retreating to WHOLE_VOLUME fails rather than passes vacuously.
    Built per call, for the reason :func:`stage_cases` is.
    """
    return {
        "Brightness": [AugmentationCase(augmentation_module.Brightness(0.5), LocalityKind.POINTWISE, True)],
        "Contrast": [AugmentationCase(augmentation_module.Contrast(0.5), LocalityKind.POINTWISE, True)],
        # The box is normalised to the volume; a region keeps its part of it. A wide box so it lands
        # in more than one patch of the fixture.
        "CutOUT": [AugmentationCase(augmentation_module.CutOUT(1.0, 0.5, 0.0), LocalityKind.POINTWISE, True)],
        "Elastix": [AugmentationCase(augmentation_module.Elastix(), LocalityKind.WHOLE_VOLUME, False)],
        "Flip": [
            AugmentationCase(FlipAugmentation(f_prob=[1.0, 1.0, 1.0]), LocalityKind.ORIENTATION, True),
            # A displacement field's flipped components are negated, which is not a bijection on values.
            AugmentationCase(
                FlipAugmentation(f_prob=[1.0, 1.0, 1.0], vector_field=True), LocalityKind.WHOLE_VOLUME, False
            ),
        ],
        # A class from another framework says nothing about where its draw reads from, so no draw of it
        # streams. torch.nn is the one such library KonfAI already depends on.
        "Foreign": [
            AugmentationCase(
                augmentation_module.Foreign(torch.nn.Sigmoid(), "torch.nn:Sigmoid"), LocalityKind.WHOLE_VOLUME, False
            )
        ],
        "HUE": [AugmentationCase(augmentation_module.HUE(1.0), LocalityKind.POINTWISE, True)],
        "LumaFlip": [AugmentationCase(augmentation_module.LumaFlip(), LocalityKind.POINTWISE, True)],
        "Mask": [],  # a second on-disk volume that dictates the output grid; see its note.
        # The field at a voxel is a function of (seed, position): a region computes exactly its part.
        "Noise": [AugmentationCase(augmentation_module.Noise(1.0), LocalityKind.POINTWISE, True)],
        "Permute": [
            AugmentationCase(augmentation_module.Permute(prob_permute=[1.0, 1.0]), LocalityKind.ORIENTATION, True)
        ],
        "Rotate": [
            # A free angle resamples: a REGRID pulling the source box the region's corners map to.
            AugmentationCase(
                augmentation_module.Rotate(a_min=10.0, a_max=10.0), LocalityKind.REGRID, True, atol=AUGMENTATION_ATOL
            ),
            # A quarter draw is a signed permutation of the axes, so it is an exact remap whichever
            # multiple of 90 degrees it lands on: the declaration holds for every draw, not for the seed
            # this happens to run on. The fixture is non-cubic, so 26 of its 27 draws transpose extents
            # and cut the copy on a grid the stored volume does not have.
            AugmentationCase(augmentation_module.Rotate(is_quarter=True), LocalityKind.ORIENTATION, True),
        ],
        "Saturation": [AugmentationCase(augmentation_module.Saturation(0.5), LocalityKind.POINTWISE, True)],
        "Scale": [AugmentationCase(augmentation_module.Scale(), LocalityKind.REGRID, True, atol=AUGMENTATION_ATOL)],
        "Translate": [
            # A halo of ceil(1) + 1 = 2 on a patch of 4: half the patch, the widest _affords_halo allows.
            AugmentationCase(
                augmentation_module.Translate(t_min=-1.0, t_max=1.0), LocalityKind.HALO, True, atol=AUGMENTATION_ATOL
            ),
            # The same declaration, a 10-voxel shift: an honest halo the dispatcher will not pay for.
            AugmentationCase(augmentation_module.Translate(t_min=10.0, t_max=10.0), LocalityKind.HALO, False),
        ],
    }


#: The kinds the READ dispatcher cannot honour: WHOLE_VOLUME by definition, and SLAB because its
#: side effect needs the slab's place in the written OUTPUT, which a patch read has no notion of.
READ_REFUSED_KINDS = (LocalityKind.WHOLE_VOLUME, LocalityKind.SLAB)


def kind_of(case: StageCase, geometry: Geometry = FIXED_GEOMETRY) -> LocalityKind:
    """What a case declares about the group it consumes: asked exactly as the dispatcher asks it,
    from the metadata that group is stored with, since a declaration the image makes reads it."""
    return case.transform.patch_locality(attributes(geometry, case.group)).kind


def streamable_cases(geometry: Geometry = FIXED_GEOMETRY) -> list[StageCase]:
    """Every built-in configuration whose own declaration says a region of it can be served."""
    return [
        case
        for cls in builtin_transforms()
        for case in cases_of(cls, geometry.rank)
        if case.sweep and kind_of(case, geometry) not in READ_REFUSED_KINDS
    ]


def builtin_augmentations() -> list[type[DataAugmentation]]:
    """Every concrete augmentation class KonfAI ships."""
    return [
        cls
        for _, cls in inspect.getmembers(augmentation_module, inspect.isclass)
        if issubclass(cls, DataAugmentation)
        and cls.__module__ == augmentation_module.__name__
        and not inspect.isabstract(cls)
    ]


# --------------------------------------------------------------------------------------
# The write-side sweep vocabulary of the streamed-oracle family (test_streamed_oracle_*):
# one property, one axis per file, and here what every file shares: the same case driven
# region by region (as the budget cuts it) and whole, then compared.
# --------------------------------------------------------------------------------------

#: The geometries the oracle matrix runs on, drawn from a FIXED seed list rather than per run: a
#: property that fails only on Tuesday's seed is not a property. Extents land in 16..40, which keeps
#: the family inside its time budget while leaving every axis room for several regions.
GEOMETRIES = {
    "rank3-seed11": seeded_geometry(11, 3),
    "rank3-seed23": seeded_geometry(23, 3),
    "rank2-seed37": seeded_geometry(37, 2),
}
#: The one the tests that vary something OTHER than the geometry run on.
MAIN = "rank3-seed11"


@dataclass(frozen=True)
class Route:
    """How the sweep is made to cut the case, as the height one region spans of its first axis."""

    name: str
    #: Rows per region, as a fraction of the case's first extent. ``None`` leaves the sweep its own
    #: cap, which covers these extents whole.
    height: float | None


#: One region, a handful, and one row each: the three decompositions of the same case.
ROUTES = (Route("one-region", None), Route("few-regions", 0.25), Route("row-regions", 0.0))


def budget_for(manager: DatasetManager, route: Route) -> float | None:
    """The smallest per-rank budget under which the sweep cuts regions of ``route``'s height.

    Found by bisecting the production sizing rule rather than by restating it: the test says how
    tall a region should be, and ``_sweep_tile`` says what budget buys it. Asked with the landing
    and the pull maps the sweep itself will use, because that is what the budget is spent on.
    """
    if route.height is None:
        return None
    # Every copy the run will sweep, each with the landing and the pull maps of its own chain: a
    # draw that samples through an affine pulls more than the shared prefix, and the budget the
    # matrix asks for is the one that buys the height on all of them.
    augmented = manager._expand is not None
    copies = [0] if not augmented else list(range(1, int(manager._expand.nb) + 1))
    segments = {a: manager.sweep_segments(a, augmented) or [] for a in copies}
    rows = max(1, int(route.height * int(manager.shapes[copies[0]][0])))
    low, high = 1.0, float(2**48)
    for _ in range(64):
        middle = (low + high) / 2
        manager.set_memory_budget(middle)
        if min(_sweep_height(manager, sweeps) for sweeps in segments.values()) < rows:
            low = middle
        else:
            high = middle
    manager.set_memory_budget(None)
    return high


def _sweep_height(manager: DatasetManager, segments: Sequence[SweepSegment]) -> int:
    """The shortest region the sizing buys these segments under the budget the manager currently
    carries; zero where one of them does not fit, which is below every height the routes ask for."""
    heights = []
    for segment in segments:
        try:
            heights.append(manager._sweep_tile(segment.landing, segment.channels, segment.plans)[0])
        except DatasetManagerError:
            return 0
    return min(heights, default=0)


@dataclass(frozen=True)
class Written:
    """One materialization's result: what landed, how it landed, and in how many pieces."""

    array: np.ndarray
    attribute: Attribute
    verdict: Verdict
    regions: int


def count_regions(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count the regions a sweep reads, so a row cannot pass by never having been decomposed."""
    read = DatasetManager._read_streamed_region
    counted = [0]

    def counting(self, *args, **kwargs):
        counted[0] += 1
        return read(self, *args, **kwargs)

    monkeypatch.setattr(DatasetManager, "_read_streamed_region", counting)
    return lambda: counted[0]


def sweep(
    dataset: Dataset, group: str, stage: Transform, destination: Path, route: Route, monkeypatch: pytest.MonkeyPatch
) -> Written:
    """Write ``stage`` over the case region by region, cut as ``route`` says, and read back what landed."""
    stage.set_datasets([dataset])
    case_manager = manager(dataset, [stage, Save(f"{destination}:h5")], group=group)
    budget = budget_for(case_manager, route)
    with monkeypatch.context() as context:
        regions = count_regions(context)
        verdict = CaseMaterializer(case_manager).materialize(fallback_budget_bytes=budget)
        array, attribute = Dataset(destination, "h5").read_data(group, CASE_NAME)
        return Written(array, attribute, verdict, regions())


def whole_volume(dataset: Dataset, group: str, stage: Transform, destination: Path) -> Written:
    """The reference: the same chain over the assembled case, which is what streaming must reproduce."""
    stage.set_datasets([dataset])
    case_manager = manager(dataset, [stage, Save(f"{destination}:h5")], group=group)
    CaseMaterializer(case_manager)._assemble_and_write(0)
    case_manager.unload()
    array, attribute = Dataset(destination, "h5").read_data(group, CASE_NAME)
    return Written(array, attribute, Verdict.WHOLE_VOLUME, 0)


def assert_same(got: Written, want: Written, atol: float, rtol: float = 0.0) -> None:
    """Same voxels within the stated bound, same dtype, same geometry: streaming is invisible."""
    assert got.array.shape == want.array.shape
    assert got.array.dtype == want.array.dtype
    np.testing.assert_allclose(got.array, want.array, rtol=rtol, atol=atol)
    for key in ("Origin", "Spacing", "Direction"):
        np.testing.assert_allclose(got.attribute.get_np_array(key), want.attribute.get_np_array(key), rtol=0, atol=0)


def oracle_matrix(geometries: Sequence[str]) -> list[tuple[str, StageCase, Route]]:
    """Every (geometry, built-in, decomposition) the property is proven on, for these geometries."""
    return [
        (geometry, case, route)
        for geometry in geometries
        for case in streamable_cases(GEOMETRIES[geometry])
        for route in ROUTES
    ]


def identify(entry: tuple[str, StageCase, Route]) -> str:
    geometry, case, route = entry
    return f"{type(case.transform).__name__}-{case.group}-{geometry}-{route.name}"
