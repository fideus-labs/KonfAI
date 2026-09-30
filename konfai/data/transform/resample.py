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


"""Resampling onto a target grid: reference grids, stored maps, displacement fields, the SimpleITK host path."""

import functools
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from konfai.data.geometry import (
    _GEOMETRY_KEYS,
    AffineMap,
    AffineStage,
    DisplacementStage,
    Grid,
    SpatialStages,
    TransformBound,
    WorldBox,
    bound_of,
)
from konfai.data.sampling import (
    blend_order,
    coordinate_precision,
    default_interpolation,
    gather,
    gather_separable,
    sampling_dtype,
    scanline_index,
    scanline_map,
    separable_source_index,
    source_index,
    source_index_rows,
    source_window,
    walk_rows,
    walked_box,
    walked_window,
)
from konfai.data.transform.base import (
    LocalityKind,
    PatchLocality,
    RegionContext,
    TransformInverse,
    sitk,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import CaseReadError, DatasetManagerError, TransformError
from konfai.utils.ITK import _require_simpleitk
from konfai.utils.utils import split_path_spec

# ---------------------------------------------------------------------------------------------
# One resample. Two questions: which grid to write on, and what map to write it through.
# ---------------------------------------------------------------------------------------------


class _TargetGrid(ABC):
    """Which grid a resample writes on: the ``to`` half of the question."""

    #: The geometry keys this target cannot be built without. An extent change needs none; a
    #: density change needs the Spacing; adopting another grid needs a real physical space.
    needs: frozenset[str] = frozenset()

    @abstractmethod
    def of(self, source: Grid, name: str) -> Grid:
        """The grid a case stored on ``source`` is written on."""

    def set_datasets(self, datasets: list[Dataset]) -> None:  # noqa: B027 - only a reference has one
        """The run's roots, for a target that has an image of its own to look up."""

    @abstractmethod
    def describe(self) -> str:
        """The target named as a refusal or a plan line names it."""


class _OwnGrid(_TargetGrid):
    """No change of grid: the map moves what the voxels hold, not where they are."""

    def of(self, source: Grid, name: str) -> Grid:
        del name
        return source

    def describe(self) -> str:
        return "the case's own grid"


class _DerivedGrid(_TargetGrid):
    """The case's own grid at another density: a spacing, or a count, and where it sits."""

    def __init__(self, spacing: list[float] | None, shape: list[int] | None, align: str) -> None:
        # A value <= 0 is the KEEP-THIS-AXIS sentinel, normalised to 0 here: that axis takes the
        # source's own density/extent in `of`, so a request rescales only the axes it names.
        self.spacing = None if spacing is None else np.asarray([max(0.0, float(value)) for value in spacing])
        self.shape = None if shape is None else tuple(max(0, int(value)) for value in shape)
        self.align = align
        # A density is meaningless without the density it starts from; a count is not.
        self.needs = frozenset({"Spacing"}) if spacing is not None else frozenset()

    def of(self, source: Grid, name: str) -> Grid:
        where = f"case '{name}'" if name else "the case"
        if self.spacing is not None:
            if self.spacing.size != source.rank:
                raise TransformError(
                    f"'Resample' was given a spacing of {self.spacing.size} value(s) and {where} has"
                    f" {source.rank} spatial axis/axes."
                )
            return source.resampled(spacing_xyz=self.spacing, align=self.align)
        shape = cast("tuple[int, ...]", self.shape)
        if len(shape) != source.rank:
            raise TransformError(
                f"'Resample' was given a shape of {len(shape)} value(s) and {where} has"
                f" {source.rank} spatial axis/axes."
            )
        return source.resampled(size_zyx=shape, align=self.align)

    def describe(self) -> str:
        if self.spacing is not None:
            return f"a spacing of {[float(value) for value in self.spacing]}"
        return f"a shape of {list(cast('tuple[int, ...]', self.shape))}"


class _ReferenceGrid(_TargetGrid):
    """The grid of a STORED image: extent, spacing, origin and direction, read from its header.

    The target that makes a cohort foldable: one grid adopted whole, which is what ``Reduce``'s
    ``grid: strict`` compares against. The reference is an image, not a list of numbers: the header
    IS the declaration.

    An entry containing ``{case}`` follows the case, so ``reference: '{case}', reference_group: DVF``
    puts every moved image on its own field's grid. Headers only, either way.
    """

    needs = frozenset(_GEOMETRY_KEYS)

    def __init__(self, entry: str, group: str | None, dataset: str | None) -> None:
        self.entry = str(entry).strip()
        self.group = group
        # A root of its own, or the run's: left out, the grid to adopt is one member of the very
        # cohort being brought together.
        self.dataset: Dataset | None = None
        if dataset is not None and str(dataset).strip():
            filename, _flag, file_format = split_path_spec(str(dataset), default_format="mha")
            self.dataset = Dataset(filename, file_format)
        self.roots: list[Dataset] = []
        self._grids: dict[str, Grid] = {}

    def set_datasets(self, datasets: list[Dataset]) -> None:
        self.roots = list(datasets)

    def _roots(self) -> list[Dataset]:
        return [self.dataset] if self.dataset is not None else list(self.roots)

    def _group_in(self, dataset: Dataset) -> str:
        """Which group of ``dataset`` holds the reference: the declared one, or its only one."""
        if self.group is not None:
            return self.group
        groups = [str(group) for group in dataset.get_group()]
        if len(groups) == 1:
            return groups[0]
        raise TransformError(
            f"'Resample' cannot tell which group of '{dataset.filename}' holds reference"
            f" '{self.entry}': it has {len(groups)} ({', '.join(sorted(groups)) or 'none'}).",
            "Name it: Resample: {reference: " + self.entry + ", reference_group: <group>}.",
        )

    def _entry_for(self, name: str) -> str:
        """The entry to adopt for ``name``: literal, or the case's own when it says ``{case}``."""
        if "{case}" not in self.entry:
            return self.entry
        if not name:
            raise TransformError(
                f"'Resample' has a per-case reference ('{self.entry}') and no case to resolve it for.",
                "A per-case reference adopts, for each case, the grid of that case's own entry in"
                " reference_group; it has no single grid to answer a caseless probe with.",
            )
        return self.entry.replace("{case}", name)

    def grid(self, name: str = "") -> Grid:
        """The reference's grid, read from its header once per distinct entry, memoized by entry."""
        entry = self._entry_for(name)
        cached = self._grids.get(entry)
        if cached is not None:
            return cached
        roots = self._roots()
        if not roots:
            raise TransformError(
                f"'Resample' has no dataset to look reference '{entry}' up in.",
                "Give the stage a root of its own. Resample: {reference: "
                + entry
                + ", reference_dataset: ./Reference:omezarr}: or run it in a workflow, which hands"
                " its dataset_filenames to every stage.",
            )
        for dataset in roots:
            group = self._group_in(dataset)
            if dataset.is_dataset_exist(group, entry):
                shape, attribute = dataset.get_infos(group, entry)
                grid = Grid.of([int(extent) for extent in shape[1:]], attribute, f"reference '{entry}'")
                self._grids[entry] = grid
                return grid
        raise TransformError(
            f"'Resample' cannot find reference '{entry}'"
            + (f" in group '{self.group}'" if self.group is not None else "")
            + f" in {', '.join(str(dataset.filename) for dataset in roots)}.",
            "Check the entry name and its group. A literal reference is looked up by entry: one"
            " grid serves the whole cohort; a '{case}' reference expects every case to have its own"
            " entry in that group.",
        )

    def of(self, source: Grid, name: str) -> Grid:
        grid = self.grid(name)
        if grid.rank != source.rank:
            where = f"case '{name}'" if name else "the case"
            raise TransformError(
                f"'Resample' cannot resample {where}, which has {source.rank} spatial axis/axes,"
                f" onto reference '{self._entry_for(name)}', which has {grid.rank}."
            )
        return grid

    def describe(self) -> str:
        return f"reference '{self.entry}'" + (" (per case)" if "{case}" in self.entry else "")


def _optional_image_filler() -> Any:
    """SimpleITK's own in-place array fill (``_SetImageFromArray``), or ``None``.

    Guarded on its own: :class:`_SitkInput` allocates a fresh image instead when it is missing.
    """
    try:
        from SimpleITK.SimpleITK import _SetImageFromArray
    except ImportError:
        return None
    return _SetImageFromArray


_set_image_from_array = _optional_image_filler() if sitk is not None else None


class _SitkInput:
    """ITK's input image for :func:`_resample_with_sitk`, reused across the regions of a sweep.

    ``GetImageFromArray`` allocates and zero-fills a new image per array, several times the cost of
    filling one in place. The image is filled in place while the regions keep one shape
    and dtype and replaced when they do not; only one is ever held, and the whole-volume call drops
    it on its way out. The fill checks the byte length only, so the shape and dtype key is what
    keeps a same-length array from being reinterpreted.
    """

    def __init__(self) -> None:
        self._image: Any = None
        self._key: tuple[tuple[int, ...], str] | None = None

    def filled(self, array: np.ndarray) -> Any:
        key = (tuple(array.shape), array.dtype.str)
        if _set_image_from_array is None or key != self._key:
            self.drop()
            self._image, self._key = sitk.GetImageFromArray(array), key
        else:
            _set_image_from_array(array, self._image)
        return self._image

    def drop(self) -> None:
        self._image = self._key = None


def _warp_field_float32(stages: SpatialStages, region: Grid) -> "Any | None":
    """The one displacement this whole map is, as a float32 vector image on ``region``, or None.

    ``sitk.Warp`` is templated on the field's own type where ``DisplacementFieldTransform`` casts to
    float64, twice the field's bytes and time on a native region.

    Taken only where it changes no value. The field must ALREADY be float32, which is what
    ``precision: fast`` reads: narrowing a genuine float64 field is a different map. Warp evaluates
    the displacement on the OUTPUT grid rather than interpolating it at the target point, so it
    stands in only where the field IS the output grid and is the whole map: one order-1 stage, no
    affine beside it. Anything else keeps the composite path.
    """
    from konfai.data.geometry import DisplacementStage

    if len(stages) != 1:
        return None
    stage = stages[0]
    if not isinstance(stage, DisplacementStage) or stage.order != 1:
        return None
    if stage.values.dtype != np.float32:
        return None
    grid = stage.grid
    if (
        tuple(grid.size_zyx) != tuple(region.size_zyx)
        or not np.allclose(grid.origin_xyz, region.origin_xyz, rtol=0.0, atol=1e-9)
        or not np.allclose(grid.spacing_xyz, region.spacing_xyz, rtol=0.0, atol=1e-9)
        or not np.allclose(grid.direction_xyz, region.direction_xyz, rtol=0.0, atol=1e-9)
    ):
        return None
    components = [
        sitk.GetImageFromArray(np.ascontiguousarray(stage.values[component])) for component in range(grid.rank)
    ]
    field = sitk.Compose(components)
    field.SetOrigin(np.asarray(grid.origin_xyz, dtype=np.float64).tolist())
    field.SetSpacing(np.asarray(grid.spacing_xyz, dtype=np.float64).tolist())
    field.SetDirection(np.asarray(grid.direction_xyz, dtype=np.float64).ravel().tolist())
    return field


@functools.cache
def _itk_picks_as_the_walk() -> bool:
    """Whether this platform's ITK reads a change of grid at the index the walk reads
    (:func:`~konfai.data.sampling.scanline_index`). ITK's arithmetic is its compiler's, which fuses
    multiply-adds on some platforms and so moves an exact half-voxel tie. Where it does not pick as the
    walk picks, a whole volume is walked as its regions are, so a route or a budget never changes a voxel.
    Probed once per process on a half-spacing resample of a permuted and of an oblique grid."""
    if sitk is None:
        return False
    tilt = np.deg2rad(3.0)
    permuted = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    oblique = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]])
    origin, spacing = np.array([-11.3, 5.9, 3.3]), np.array([0.77, 1.31, 2.05])
    volume = torch.arange(7.0 * 9 * 11, dtype=torch.float64).reshape(1, 7, 9, 11)
    for direction in (permuted, oblique @ permuted):
        source = Grid((7, 9, 11), origin, spacing, direction)
        target = Grid((14, 18, 22), origin, spacing / 2, direction)
        itk = _resample_with_sitk(volume, target, source, (), [0, 0, 0], "nearest", 0.0)
        with coordinate_precision(torch.float64):
            index = scanline_index(target, source, tuple(slice(0, extent) for extent in target.size_zyx), volume.device)
            walked = gather(volume, index, [0, 0, 0], [7, 9, 11], "nearest", 0.0, True)
        if itk is None or not torch.equal(itk, walked):
            return False
    return True


def _resample_with_sitk(
    payload: torch.Tensor,
    region: Grid,
    source: Grid,
    stages: SpatialStages,
    region_starts: list[int],
    mode: str,
    fill: float,
    sitk_input: _SitkInput | None = None,
) -> torch.Tensor | None:
    """One region of a resample through ITK's own filter, on the host.

    ``payload`` is the SOURCE window read for this region (``region_starts`` says where it sits in
    the source grid); ``region`` is the target grid of the region. The stages become one composite
    transform (:func:`~konfai.utils.ITK.encode_transform_stages`), applied by ``sitk.Resample`` in
    the direction KonfAI's walk applies it (target point through the stages to the source point).
    Channels are resampled one by one. ``None`` for a payload ITK has no pixel type for (bool, f16,
    bf16).
    """
    from konfai.utils.ITK import encode_transform_stages

    if payload.dtype in (torch.bool, torch.bfloat16, torch.float16):
        return None
    interpolator = {"nearest": sitk.sitkNearestNeighbor, "linear": sitk.sitkLinear, "cubic": sitk.sitkBSpline}
    if mode == "cubic":
        return None  # ITK's BSpline is not Keys' Catmull-Rom: the walk keeps its own cubic
    rank = source.rank
    # A map that is one field ON the output grid is applied by the filter templated on the field's
    # own type, never by one that casts it to float64 to hold it (:func:`_warp_field_float32`).
    warp_field = _warp_field_float32(stages, region) if stages else None
    transform = (
        None
        if warp_field is not None
        else (encode_transform_stages(stages) if stages else sitk.Transform(rank, sitk.sitkIdentity))
    )
    # The window's own origin: the source origin moved by the window's start along each axis.
    start_index = np.asarray(list(reversed(region_starts)), dtype=np.float64)  # (x, y, z)
    window_origin = source.index_to_world.apply(start_index)
    # The target grid is given to the filter, not carried by an image standing in for it: a
    # reference image is read for its geometry and never for its pixels.
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(e) for e in reversed(region.size_zyx)])
    resampler.SetOutputOrigin(np.asarray(region.origin_xyz, dtype=np.float64).tolist())
    resampler.SetOutputSpacing(np.asarray(region.spacing_xyz, dtype=np.float64).tolist())
    resampler.SetOutputDirection(np.asarray(region.direction_xyz, dtype=np.float64).ravel().tolist())
    if transform is not None:
        resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator[mode])
    resampler.SetDefaultPixelValue(float(fill))
    # A blend interpolates in the dtype the walk accumulates in, and torch makes the final cast: ITK
    # would otherwise accumulate an integer payload in double and cast inside the filter, where one
    # ulp becomes a whole unit after the truncation. A nearest pick keeps the payload's own dtype.
    # The filter is ASKED for that dtype instead of being handed a region converted into it.
    working_dtype = payload.dtype if mode == "nearest" else sampling_dtype(payload)
    blend_pixel_id = {torch.float32: sitk.sitkFloat32, torch.float64: sitk.sitkFloat64}.get(working_dtype)
    if mode != "nearest" and blend_pixel_id is None:
        return None  # a working dtype with no ITK pixel type of its own: the walk takes it
    # One output, in the payload's dtype, written channel by channel: ITK takes one component at a
    # time, so only one is ever held in the wider dtype, and the cast that lands it is the copy out.
    result = torch.empty((int(payload.shape[0]), *(int(e) for e in region.size_zyx)), dtype=payload.dtype)
    landing = result.numpy()
    for channel in range(int(payload.shape[0])):
        array = np.ascontiguousarray(payload[channel].numpy())
        image = sitk.GetImageFromArray(array) if sitk_input is None else sitk_input.filled(array)
        image.SetOrigin(np.asarray(window_origin, dtype=np.float64).tolist())
        image.SetSpacing(np.asarray(source.spacing_xyz, dtype=np.float64).tolist())
        image.SetDirection(np.asarray(source.direction_xyz, dtype=np.float64).ravel().tolist())
        pixel_id = image.GetPixelID() if mode == "nearest" else blend_pixel_id
        # Held: a view borrows the image's buffer, and a temporary's is freed under it.
        if warp_field is not None:
            # Warp answers in its INPUT's type where the resampler is asked for an output type, so
            # the blend's dtype is carried in rather than requested (pinned in
            # test_resample_transform.py).
            resampled = sitk.Warp(
                image if image.GetPixelID() == pixel_id else sitk.Cast(image, pixel_id),
                warp_field,
                interpolator[mode],
                [int(e) for e in reversed(region.size_zyx)],
                np.asarray(region.origin_xyz, dtype=np.float64).tolist(),
                np.asarray(region.spacing_xyz, dtype=np.float64).tolist(),
                np.asarray(region.direction_xyz, dtype=np.float64).ravel().tolist(),
                float(fill),
            )
        else:
            resampler.SetOutputPixelType(pixel_id)
            resampled = resampler.Execute(image)
        np.copyto(landing[channel], sitk.GetArrayViewFromImage(resampled), casting="unsafe")
    return result


@dataclass(frozen=True)
class _StoredMap:
    """What the plan keeps of a case's decoded stored transform: its bound, and whether the map IS
    the bound's affine part (every stage affine, folded exactly as the walk folds them).

    The decoded stages are not kept: a cohort of fields would stay resident and be serialised to
    every rank. They are decoded again where a region of their case is sampled
    (:meth:`Resample._stored_stages`).
    """

    bound: TransformBound
    affine: bool
    #: Whether a dense field member was priced as the identity rather than bounded. The plan cannot
    #: bound one from headers, so the run measures its window from the values it samples anyway.
    field: bool = False


def _stages_bytes(stages: SpatialStages) -> int:
    """What decoded stages hold: a dense field's values, an affine map next to nothing."""
    return sum(stage.values.nbytes for stage in stages if isinstance(stage, DisplacementStage))


#: How many times over a region's field window is materialised at once while the region is sampled.
#: Handing the field to ITK costs three: the values the read cached, the rank component images
#: encode_transform_stages builds from them, and the vector image sitk.Compose builds beside those
#: (konfai/utils/ITK.py, all live at the DisplacementFieldTransform call). Priced rather than probed: a run cut in its FIRST region has nothing
#: measured yet. Charged on every route, though only the host one goes through ITK: over-charging
#: costs a shorter region, under-charging costs the run.
_FIELD_WINDOW_COPIES = 3.0

#: What one element of a decoded field weighs under the bit-exact walk: read as float64 whatever
#: the store holds (:meth:`_DisplacementSource.read`), against the plan's count at
#: :data:`~konfai.data.patching.budget.CASE_ELEMENT_BYTES`. Under ``precision: fast`` the field is
#: held in float32 and weighs half (:meth:`Resample._field_element_bytes`).
_FIELD_ELEMENT_BYTES = 8


class _DisplacementSource:
    """A displacement field on disk: where it is, how far it reaches, and how to read a region of it.

    :class:`Resample` is its one owner, so every refusal speaks as ``Resample`` and names
    ``field_group``, the argument the user declared.
    """

    def __init__(self, field: str | None, group: str | None) -> None:
        # A root of its own, or none: with no ``field`` path the fields are a GROUP of the run's own
        # dataset_filenames, one entry per case, beside the volumes they were solved on.
        self.dataset: Dataset | None = None
        if field is not None and str(field).strip():
            filename, _flag, file_format = split_path_spec(str(field), default_format="mha")
            self.dataset = Dataset(filename, file_format)
        elif group is None:
            raise TransformError(
                "'Resample' has neither a 'field' path nor a group to find the fields in.",
                "Name the store. Resample: {field: ./DVF:omezarr}: or, for fields stored beside"
                " the cases, the group they are in: Resample: {field_group: DVF}.",
            )
        self.group = group
        #: The run's own roots, handed over by the owner; only consulted when there is no path.
        self.roots: list[Dataset] = []
        self._scanned = False
        self._unreadable: str | None = None
        self._probed: set[str] = set()

    def unreadable_header(self) -> str | None:
        """Why a field entry's HEADER does not open, or ``None`` when every one does, memoized: the
        plan's one probe of the group.

        An unreadable entry fails both routes on whichever case reaches it, so the group is scanned
        here, one entry at a time, before any case is chosen.
        """
        if self._scanned:
            return self._unreadable
        self._scanned = True
        try:
            group = self.group_for(None)
            roots = [self.dataset] if self.dataset is not None else list(self.roots)
            for root in roots:
                for entry in root.get_names(group):
                    try:
                        root.get_infos(group, entry)
                    except CaseReadError as error:
                        # The dataset's own sentence names the entry, where it is and why.
                        self._unreadable = str(error.args[0])
                        return self._unreadable
                    except Exception as error:
                        self._unreadable = f"entry '{entry}' of '{group}': {type(error).__name__}: {error}"
                        return self._unreadable
        except Exception as error:  # an unreadable field dataset is a whole-volume answer, not a crash
            self._unreadable = f"{type(error).__name__}: {error}"
        return self._unreadable

    def group_for(self, name: str | None) -> str:
        if self.group is not None:
            return self.group
        if self.dataset is None:  # unreachable: a source with no path was given a group to use
            raise TransformError(
                "'Resample' has no field store of its own and no group to look for one in.",
                "Name the group the fields are in: Resample: {field_group: DVF}.",
            )
        groups = [str(group) for group in self.dataset.get_group()]
        if len(groups) == 1:
            return groups[0]
        where = f"the field for case '{name}'" if name is not None else "the fields"
        raise TransformError(
            f"'Resample' cannot tell which group of '{self.dataset.filename}' holds {where}: it has {len(groups)}.",
            "Name it: Resample: {field: ./DVF:omezarr, field_group: DVF}.",
        )

    def _root_for(self, name: str | None) -> Dataset:
        """The store this case's field is in: the declared one, or whichever run root holds it."""
        if self.dataset is not None:
            return self.dataset
        group = self.group_for(name)
        for root in self.roots:
            if name is None or root.is_dataset_exist(group, name):
                return root
        raise TransformError(
            f"'Resample' cannot find a field for case '{name}' in group '{group}' of"
            f" {', '.join(str(root.filename) for root in self.roots) or 'any dataset'}.",
            "A field declared by group alone is looked up beside the cases, one entry per case."
            " Give the store a path of its own instead: Resample: {field: ./DVF:omezarr}.",
        )

    def infos(self, name: str) -> tuple[list[int], Attribute]:
        """The field entry's shape and header, without reading a voxel of it."""
        return self._root_for(name).get_infos(self.group_for(name), name)

    def probe(self, name: str) -> None:
        """The case's own entry, proven present and readable: the per-case half of the scan.

        :meth:`unreadable_header` cannot know which cases the plan will ask for, so a missing entry
        would otherwise surface mid-run, after bytes are written. Memoized: one header read per case.
        """
        if name in self._probed:
            return
        try:
            self.infos(name)
        except TransformError:
            raise
        except Exception as error:
            raise TransformError(
                f"'Resample' cannot read the field header for case '{name}': {type(error).__name__}: {error}.",
                "Repair or re-write that entry, or drop the case with 'subset'.",
            ) from error
        self._probed.add(name)

    def read(
        self, name: str, region: tuple[slice, ...] | None, channels: int, dtype: type = np.float64
    ) -> torch.Tensor:
        group = self.group_for(name)
        root = self._root_for(name)
        if region is None:
            data, _attributes = root.read_data(group, name)
        else:
            data, _attributes = root.read_data_slice(group, name, (slice(None), *region))
        # ``dtype`` is a CEILING, as for a stored field (konfai.utils.ITK._displacement_stage): a
        # float32 field stays float32 under float64, which widens nothing and lets the region take
        # sitk.Warp; a float64 field narrows only under `precision: fast`.
        data = np.asarray(data)
        ceiling = np.dtype(dtype)
        held = data.dtype if data.dtype.kind == "f" and 4 <= data.dtype.itemsize <= ceiling.itemsize else ceiling
        field = torch.from_numpy(np.ascontiguousarray(data, dtype=held))
        if field.shape[0] != channels:
            raise TransformError(
                f"The field for case '{name}' has {field.shape[0]} component(s) where the case has"
                f" {channels} spatial axis/axes.",
                "A displacement field carries one component per spatial axis, component-first.",
            )
        return field


def _reached(region: Grid, stages: SpatialStages) -> WorldBox:
    """Where ``stages`` send ``region``'s voxels: walked along its faces once a displacement is among
    them (:func:`~konfai.data.sampling.walked_box`), exact through an affine alone."""
    if any(isinstance(stage, DisplacementStage) for stage in stages):
        return walked_box(region, stages)
    return bound_of(stages, region.rank).map_box(region.centres_box())


class Resample(TransformInverse):
    """Resample a case: onto another grid, through a stored map, or both, in one interpolation.

    **Which grid to write on**: at most one of:

    - nothing (the default): the case's own grid. The map moves the anatomy; the voxels stay put.
    - ``spacing``: the same field of view at another density, in physical order ``(x, y, z)``. A
      component left at ``0`` keeps its axis.
    - ``shape``: the same field of view at a given count, in array order ``(Z, Y, X)``. A component
      left at ``0`` keeps its axis. A spacing is geometry and a shape is an array extent, so each is
      written in its own convention.
    - ``reference``: the grid of a stored image, adopted whole: extent, spacing, origin, direction.
      ``'{case}'`` in the entry follows the case, so ``reference: '{case}', reference_group: DVF``
      lands every moved image on its own field's grid.

    **What map to write it through**: any of, composed in this order:

    - ``field``: a displacement field, read in world units at each TARGET voxel, on its own grid and
      spacing: a field solved at 120 um moves a volume stored at 30 um without being upsampled.
    - ``transforms``: transforms stored beside the cases (rigid, affine, BSpline, dense field, or a
      composite of them) mapping GROUP to whether to invert it. The LAST declared is applied first,
      which is SimpleITK's own composite order.

    Left out, the map is the identity and this is a change of grid and nothing else.

    ONE INTERPOLATION, ALWAYS. A grid change and a warp asked for together are composed into a single
    coordinate per target voxel and the source is read once, at the displaced point.

    ``interpolation`` is ``nearest``, ``linear`` (the default; a label dtype, ``uint8``, ``int64`` or
    ``bool``, defaults to ``nearest``) or ``cubic``: Keys' cubic convolution (Catmull-Rom, a = -1/2),
    interpolating and patch-local (four taps per axis, one extra voxel of streamed halo). It is NOT a
    prefiltered B-spline: scipy's ``order=3`` runs a global prefilter this stage does not.

    IT STREAMS, and what a region reads is known before a voxel of the SOURCE is touched. A rigid or
    affine map is an exact affine. A BSpline and a dense field are values on a grid read through a
    non-negative kernel that sums to one, so the sup-norm of those values bounds the displacement at
    every point. A field on disk is read region by region, and the sup of the values just read bounds
    that region's pull, so each slab pays exactly the halo ITS displacements require. The plan prices
    the reads as if the field were zero, and says so.

    ``align`` decides where a ``spacing`` or a ``shape`` grid SITS: ``extent`` keeps the field of
    view (the outer faces coincide), ``origin`` keeps voxel zero's centre where it is. A
    ``reference`` states its own placement and ignores this.

    WHAT IT REFUSES, rather than resample from a window it cannot size or in a space it does not have:

    - a case whose header carries no ``Origin``/``Spacing``/``Direction`` when the answer needs
      physical space (a reference, a stored transform, a field). A plain ``spacing``/``shape``
      resample does not: with no geometry a world coordinate IS an index, and the ratio is the map;
    - a transform type that decomposes into no bounded map, naming the type;
    - ``invert: true`` on anything but a rigid or affine map: inverting a spline or a field is a
      dense solve over the whole grid, and a field solved per region is not the restriction of the
      field solved once. Store the inverse instead;
    - a case that does not meet the target grid anywhere, judged THROUGH the declared map so a
      stored rigid bridging two scanner frames is not mistaken for disjointness.

    A refusal the whole-volume path can serve (a case with no geometry, an unreadable entry in the
    field group) declares ``WHOLE_VOLUME`` with its reason and the run proceeds assembled. One that
    no route can serve refuses as the plan is built, before a byte is written. A case reaching only
    PART of the target grid is legal and common (the rest takes ``fill``), and the plan prints how
    much of the grid it covers.
    """

    working_multiple = 6.5  # the sampling grid, the taps, and the widening a stored integer forces

    def __init__(
        self,
        spacing: list[float] | None = None,
        shape: list[int] | None = None,
        reference: str | None = None,
        reference_group: str | None = None,
        reference_dataset: str | None = None,
        transforms: dict[str, bool] | None = None,
        field: str | None = None,
        field_group: str | None = None,
        align: str = "extent",
        interpolation: str | None = None,
        fill: float = 0.0,
        inverse: bool = True,
        precision: str = "exact",
    ) -> None:
        super().__init__(inverse)
        if interpolation is not None and interpolation not in ("linear", "nearest", "cubic"):
            raise TransformError(
                f"'Resample' has an unknown interpolation '{interpolation}'.",
                "Use 'linear' for an image, 'nearest' for a label map, or 'cubic' (Keys/Catmull-Rom)"
                " for a sharper image blend. Left unset, uint8, int64 and bool are taken for a label map"
                " and everything else is interpolated linearly.",
            )
        if precision not in ("exact", "fast"):
            raise TransformError(
                f"'Resample' has an unknown precision '{precision}'.",
                "'exact' (the default) walks coordinates in float64, bit-identical to"
                " sitk.Resample. 'fast' lets the device walk in float32: half the bytes and about"
                " twice the rows per slab, at ~|world|/2^24 of coordinate error -- for INTENSITY"
                " resamples only. On the host it holds a stored field at float32 and applies it"
                " with sitk.Warp instead of a float64 transform, which for a field STORED in"
                " float32 is the same answer to the bit and half the field's memory."
                " A nearest pick that lands within that band of a voxel boundary picks the other"
                " voxel, so a label map must stay 'exact'.",
            )
        self.precision = precision
        self.interpolation = interpolation
        self.fill_value = float(fill)
        self._target = self._target_from(spacing, shape, reference, reference_group, reference_dataset, align)
        if transforms is not None and not transforms:
            raise TransformError(
                "'Resample' was given an empty 'transforms'.",
                "Name a group and say whether to invert it (transforms: {reg: false}) or drop the"
                " argument: without it the map is the identity and this is a change of grid alone.",
            )
        self.transforms = transforms
        declared = (field is not None and str(field).strip()) or field_group is not None
        self.displacement: _DisplacementSource | None = _DisplacementSource(field, field_group) if declared else None
        #: Per case: the grid its own header describes, recorded where that header is in hand
        #: (transform_shape, called for every case). A region read hands back the REGION's Origin, so
        #: a grid rebuilt from a streamed region would place the case by the corner of a slab.
        self._grids: dict[str, Grid] = {}
        #: Per case: the geometry keys its header did not carry (see :meth:`Grid.from_header`).
        self._assumed: dict[str, frozenset[str]] = {}
        #: Per case: the target grid and the pricing bound, each beside the objects it was built from.
        #: A sizer prices hundreds of windows per case through them; a case recorded again is a new
        #: source grid, which the identity check sees.
        self._targets: dict[str, tuple[Grid, Grid]] = {}
        self._bounds: dict[str, tuple[Grid, _StoredMap | None, TransformBound]] = {}
        #: Per case: the source grid its coverage was last found non-empty on.
        self._covering: dict[str, Grid] = {}
        #: Per case: what the plan keeps of its stored map (see :class:`_StoredMap`).
        self._maps: dict[str, _StoredMap] = {}
        #: The decoded stages of the last cases sampled, most recent last, within
        #: ``stored_stage_bytes``. Not pickled. Keyed on (case, box): a field read for one region
        #: answers for that region alone.
        self._stored: OrderedDict[tuple, SpatialStages] = OrderedDict()
        #: The last field window read, kept for the sampler: sizing a region's source window reads
        #: the very field slab the sampler needs next, so one slot makes the two one read.
        self._field_window: tuple[str, object, DisplacementStage] | None = None
        # ITK's input image on the host route, filled in place from one region to the next.
        self._sitk_input = _SitkInput()
        self._refusal: str | None = None
        self._probed = False

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_stored"] = OrderedDict()
        state["_field_window"] = None
        state["_sitk_input"] = _SitkInput()
        return state

    #: Bytes of decoded stages kept across cases: the affine maps of a whole cohort and the dense
    #: fields of the last two or three, so a fold reading N members per region does not decode one
    #: per region.
    stored_stage_bytes = 512 << 20
    #: Region-keyed entries kept at once: an all-affine map prices zero bytes and would never evict.
    stored_stage_slots = 64

    @staticmethod
    def _target_from(
        spacing: list[float] | None,
        shape: list[int] | None,
        reference: str | None,
        reference_group: str | None,
        reference_dataset: str | None,
        align: str,
    ) -> _TargetGrid:
        named = [name for name, value in (("spacing", spacing), ("shape", shape), ("reference", reference)) if value]
        if len(named) > 1:
            raise TransformError(
                f"'Resample' was given {' and '.join(named)}, each a way to name the target grid.",
                "A resample writes on one grid: give its density (spacing), its extent (shape) or the"
                " image whose grid to adopt (reference): and only one of them.",
            )
        if reference and not str(reference).strip():
            raise TransformError(
                "'Resample' was given a blank reference.",
                "Name the entry whose grid to adopt: Resample: {reference: 822174, reference_group: Volume}.",
            )
        if align not in ("extent", "origin"):
            raise TransformError(
                f"'Resample' has an unknown align '{align}'.",
                "Use align: extent to keep the field of view (the outer faces coincide, which is what"
                " KonfAI has always done) or align: origin to keep voxel zero's centre where it is.",
            )
        if reference:
            return _ReferenceGrid(reference, reference_group, reference_dataset)
        if spacing is not None or shape is not None:
            return _DerivedGrid(spacing, shape, align)
        if reference_group is not None or reference_dataset is not None:
            raise TransformError(
                "'Resample' was told where to find a reference but not which one.",
                "Name the entry whose grid to adopt: Resample: {reference: 822174, reference_group: Volume}.",
            )
        return _OwnGrid()

    def set_datasets(self, datasets: list[Dataset]) -> None:
        super().set_datasets(datasets)
        self._target.set_datasets(datasets)
        # A field declared by group alone lives beside the cases, so it looks in the same roots.
        if self.displacement is not None:
            self.displacement.roots = list(datasets)

    # ------------------------------------------------------------------ the two grids

    @property
    def _needs(self) -> frozenset[str]:
        """The geometry keys this configuration cannot be answered without.

        A stored map or a reference grid needs all three; a change of density needs the density it
        starts from; a change of extent needs nothing at all.
        """
        if self.transforms is not None or self.displacement is not None:
            return frozenset(_GEOMETRY_KEYS)
        return self._target.needs

    def _record(self, name: str, shape: list[int], cache_attribute: Attribute) -> Grid:
        """The case's own grid, remembered under its name, with what its header left unsaid."""
        where = f"case '{name}'" if name else "the case"
        grid, missing = Grid.from_header(list(shape), cache_attribute, where)
        # Every plan records the case again: the same header keeps its grid, so what was derived from
        # that grid (the target, the bound, the coverage) is not derived again. A header that left a
        # key unsaid is another header, whatever grid it is read as.
        held = self._grids.get(name)
        kept = held is not None and held.same_as(grid) and self._assumed.get(name) == missing
        self._assumed[name] = missing
        if not kept:
            self._grids[name] = grid
        return self._grids[name]

    def _source_grid(self, name: str) -> Grid:
        grid = self._grids.get(name)
        if grid is None:
            raise TransformError(
                f"'Resample' was asked for a region of case '{name}' before its grid was established.",
                "This is a bug if it was reached: transform_shape records the grid of every case as"
                " its manager is built, and a region is only ever streamed afterwards.",
            )
        return grid

    def _target_of(self, name: str) -> tuple[Grid, Grid]:
        """``(source, target)``: needs only what BUILDING the target grid needs.

        Split from :meth:`_grids_of`: the output SHAPE of a warp on the case's own grid is the case's
        own shape, knowable with no geometry, while SAMPLING it is not.
        """
        source = self._source_grid(name)
        absent = self._assumed.get(name, frozenset())
        lacking = [key for key in _GEOMETRY_KEYS if key in absent and key in self._target.needs]
        if lacking:
            raise TransformError(
                f"'Resample' cannot place {self._target.describe()} for case '{name}': its header"
                f" carries no {', '.join(lacking)}.",
                "A density is meaningless without the density it starts from, and another grid"
                " cannot be adopted without a physical space to adopt it in. Use a source whose"
                " geometry is readable (mha, nii, h5, or an OME-Zarr written by KonfAI).",
            )
        return source, self._target.of(source, name)

    def slab_height_sensitive(self, name: str) -> bool:
        """Whether this case's streamed values can depend on the slab height: only a map that does
        not factorise (a rotation, a displacement field) interpolates through per-voxel coordinates
        whose float rounding differs with where the region starts, and a change of grid with no map
        is walked as ITK walks the whole grid (``scanline_map``). True when the headers cannot settle
        it."""
        try:
            source, target = self._grids_of(name)
            if self.displacement is not None:
                return True
            stages: SpatialStages = ()
            if self.transforms is not None:
                # From the plan's record, no decode: an all-affine map IS the bound's affine part,
                # folded as the walk folds it, so the separable test answers what the run's will.
                stored = self._stored_map(name)
                if not stored.affine:
                    return True
                stages = (AffineStage(stored.bound.affine),)
            walked_as_itk = self.interpolation != "cubic" and scanline_map(target, source, stages)
            return separable_source_index(target, source, stages, torch.device("cpu")) is None and not walked_as_itk
        except Exception:  # nosec B110 - a map this cannot read is priced as the general path
            return True

    def _grids_of(self, name: str) -> tuple[Grid, Grid]:
        source = self._source_grid(name)
        held = self._targets.get(name)
        if held is not None and held[0] is source:
            return held
        absent = self._assumed.get(name, frozenset())
        lacking = [key for key in _GEOMETRY_KEYS if key in absent and key in self._needs]
        if lacking:
            raise TransformError(
                f"'Resample' needs the geometry of case '{name}' to resample it onto"
                f" {self._target.describe()}, and its header carries no {', '.join(lacking)}.",
                "Resampling onto another grid, or through a stored map, happens in physical space:"
                " without an origin, a spacing and a direction there is no space to do it in. Use a"
                " source whose geometry is readable (mha, nii, h5, or an OME-Zarr written by KonfAI).",
            )
        grids = self._targets[name] = (source, self._target.of(source, name))
        return grids

    # ------------------------------------------------------------------ the map

    def _stored_stages(self, name: str, region: Grid, before: SpatialStages = ()) -> SpatialStages:
        """This case's stored transforms, decoded and composed, in application order.

        The last cases' stages are held, most recent last, within ``stored_stage_bytes``.

        KEYED ON THE REGION: a field read for one region answers for that region and no other, so a
        second region asking with the same case name would sample outside the first one's window and
        take the border value silently. ``before`` is what runs ahead of them, the region's own.
        """
        key = (name, tuple(region.size_zyx), tuple(np.ravel(region.origin_xyz)))
        stages = self._stored.pop(key, None)
        if stages is None:
            stages = self._decode_stored(name, region, before)
        self._stored[key] = stages
        held = sum(_stages_bytes(kept) for kept in self._stored.values())
        while len(self._stored) > 1 and (held > self.stored_stage_bytes or len(self._stored) > self.stored_stage_slots):
            held -= _stages_bytes(self._stored.popitem(last=False)[1])
        return stages

    def _stored_map(self, name: str) -> _StoredMap:
        """The plan's record of this case's stored map, decoded once and kept without its stages."""
        stored = self._maps.get(name)
        if stored is None:
            rank = self._source_grid(name).rank
            # HEADERS ONLY, always: a cached region's stages are that region's, not the case's.
            # What comes back is the affine part, exact, with any dense field standing as the
            # identity, which is the price the declared route puts on its field.
            priced = self._decode_stored(name, headers_only=True)
            has_field = self._stored_has_field(name)
            stored = self._maps[name] = _StoredMap(
                bound_of(priced, rank),
                not has_field and all(isinstance(stage, AffineStage) for stage in priced),
                field=has_field,
            )
        return stored

    def _stored_has_field(self, name: str) -> bool:
        """Whether any member of this case's stored map is a dense field, from headers alone.

        Asked rather than counted: one entry can decode to several stages, so a stage count says
        nothing about how many members there were.
        """
        return bool(self._stored_field_headers(name))

    def _stored_field_headers(self, name: str) -> list[tuple[list[int], Attribute]]:
        """The shape and header of every dense-field member of this case's stored map."""
        from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE

        fields = []
        for group in cast("dict[str, bool]", self.transforms or {}):
            for dataset in self.datasets:
                if not dataset.is_dataset_exist(group, name):
                    continue
                if getattr(dataset, "read_data", None) is None:
                    break  # a transform-only store serves no field
                shape, header = dataset.get_infos(group, name)
                if DISPLACEMENT_FIELD_ATTRIBUTE in header:
                    fields.append((shape, header))
                break
        return fields

    def _decode_stored(
        self, name: str, region: Grid | None = None, before: SpatialStages = (), headers_only: bool = False
    ) -> SpatialStages:
        """This case's stored transforms read and decoded, application order, nothing kept.

        ``region`` is the target region the map will be evaluated over, sent through ``before`` and
        the members already decoded, so each one is read on the box IT sees: a field applied second
        is evaluated where the first sent the points. Without a region the whole entry is read.
        ``headers_only`` is the plan's read: a dense field member decodes to no stage, the identity,
        and its values are never touched.
        """
        from konfai.utils.ITK import invert_stages, read_transform_stages

        _require_simpleitk()
        rank = self._source_grid(name).rank
        stages: list[AffineStage | DisplacementStage] = []
        # Reversed: a CompositeTransform applies its members last-first, and decoding normalizes
        # each member to application order, so the declared list is reversed here to mean the same.
        for group in reversed(list(cast("dict[str, bool]", self.transforms))):
            invert = self.transforms[group] if self.transforms else False
            dataset = self.dataset_holding(group, name)
            if dataset is None:
                raise TransformError(
                    f"'Resample' found no transform for case '{name}' in group '{group}'.",
                    "Every case needs an entry in every group named under 'transforms:'. Check the"
                    " group name, or drop the cases that have no transform with 'subset'.",
                )
            # A dense field is read where the EFFECTIVE stages so far, after any inversion, send the
            # region; asked only of such a member, since nothing else reads by region.
            box = None if region is None else functools.partial(_reached, region, (*before, *stages))
            decoded = read_transform_stages(dataset, group, name, box, headers_only, self._field_dtype)
            if invert:
                inverted = invert_stages(decoded, rank)
                if inverted is None:
                    raise TransformError(
                        f"'Resample' cannot invert group '{group}' for case '{name}': it is not a rigid or affine map.",
                        "Inverting a spline or a displacement field is a dense solve over the whole"
                        " grid, and a field solved per region is not the restriction of the field"
                        f" solved once. Store the inverse field instead, or set '{group}: false' and"
                        " invert it where it is written.",
                    )
                decoded = inverted
            stages.extend(decoded)
        return tuple(stages)

    def _field_stage(self, name: str, region: Grid) -> DisplacementStage:
        """The declared field over ``region``, read on its own grid and no wider: once.

        The field is evaluated at the TARGET's world points, so the window it needs is that region's
        own world box, no halo whatever the displacement. What the halo sizes is the SOURCE read,
        answered from these very values (:meth:`measured_region_source`): memoized here so sizing
        and sampling share one read.
        """
        key = (tuple(int(extent) for extent in region.size_zyx), tuple(float(v) for v in np.ravel(region.origin_xyz)))
        cached = self._field_window
        if cached is not None and cached[0] == name and cached[1] == key:
            return cached[2]
        source = cast("_DisplacementSource", self.displacement)
        shape, attribute = source.infos(name)
        spatial = [int(extent) for extent in shape[1:]]
        grid = Grid.of(spatial, attribute, f"the field for case '{name}'")
        window = grid.node_window(region.centres_box())
        values = source.read(name, window, len(spatial), self._field_dtype)
        stage = DisplacementStage(grid.sub_grid(window), values.numpy(), order=1)
        self._field_window = (name, key, stage)
        return stage

    def stream_abort(self, name: str) -> None:
        for key in [key for key in self._stored if key[0] == name]:
            self._stored.pop(key, None)
        if self._field_window is not None and self._field_window[0] == name:
            self._field_window = None
        self._sitk_input.drop()

    def _stages(self, name: str, region: Grid) -> SpatialStages:
        """The whole map over one target region, in application order, each stage read on the box
        the stages before it send that region to."""
        stages: list[AffineStage | DisplacementStage] = []
        if self.displacement is not None:
            stages.append(self._field_stage(name, region))
        if self.transforms is not None:
            stages.extend(self._stored_stages(name, region, tuple(stages)))
        return tuple(stages)

    def _pricing_bound(self, name: str) -> TransformBound:
        """The map's bound as the PLAN prices it: headers and declarations, never a voxel.

        A field prices as zero displacement, declared or stored: nothing bounds one from headers.
        The run never trusts this window, since a field's regions are sized from the values it reads
        (:meth:`measured_region_source`), so the optimism costs estimate accuracy, not bytes. What is
        left is the affine part, which is exact.
        """
        source = self._source_grid(name)
        stored = self._stored_map(name) if self.transforms is not None else None
        held = self._bounds.get(name)
        if held is not None and held[0] is source and held[1] is stored:
            return held[2]
        folded = TransformBound.exact(AffineMap.identity(source.rank))
        if stored is not None:
            folded = stored.bound.after(folded)
        self._bounds[name] = (source, stored, folded)
        return folded

    # ------------------------------------------------------------------ the contract

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        del group_src
        self._record(name, [int(extent) for extent in shape], cache_attribute)
        _source, target = self._target_of(name)
        if name:
            self._require_runnable(name)
            self._refuse_if_disjoint(name)
        return [int(extent) for extent in target.size_zyx]

    def _require_runnable(self, name: str) -> None:
        """Refuse AT PLAN TIME a map neither route can apply.

        A refusal the whole-volume path can serve (a case with no geometry) stays a locality answer
        and the run proceeds assembled. A stored transform that cannot be decoded, read or inverted
        fails both paths, so declaring WHOLE_VOLUME for it would print a plan the run contradicts by
        dying per case. ``transform_shape`` runs for every case as the plan is built.
        """
        if self.displacement is not None:
            self.displacement.probe(name)
        if self.transforms is None:
            return
        try:
            self._stored_map(name)
        except TransformError:
            raise
        except Exception as error:  # a corrupt store fails both routes; name the case and the cure
            raise TransformError(
                f"'Resample' cannot read the map for case '{name}', so no route can apply it:"
                f" {type(error).__name__}: {error}.",
                "Check the group names under 'transforms:' and that every case has an entry in each.",
            ) from error

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The geometry is judged on the attribute in hand, the case's own header: one case of a
        # group may lack an Origin while the rest have one. A config-time probe hands over an empty
        # header, which reads as a case with none.
        lacking = [key for key in _GEOMETRY_KEYS if key in self._needs and key not in cache_attribute]
        if lacking:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    f"resampling onto {self._target.describe()} happens in physical space and this"
                    f" case carries no {', '.join(lacking)}. Use a source whose geometry is readable"
                    " (mha, nii, h5, or an OME-Zarr written by KonfAI)"
                ),
            )
        if not self._probed:
            self._probed = True
            self._refusal = self._probe_cohort()
        if self._refusal is not None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME, reason=self._refusal)
        return PatchLocality(LocalityKind.REGRID)

    def _probe_cohort(self) -> str | None:
        """Whether every case this stage will see is boundable, or the sentence saying which is not.

        The COHORT's answer, not one case's: a locality is declared once for the stage, so a group
        whose entries are not uniformly decodable falls back for all of them. Exceptions become a
        reason: a raise here would take the run down instead of costing it the whole-volume path.
        The GEOMETRY is per case, read by :meth:`patch_locality` off the header it is handed.
        """
        if self.transforms is not None and sitk is None:
            return (
                "SimpleITK is not installed, and a stored transform is applied in physical space by"
                " it. Install it (pip install konfai[itk]) to stream this stage"
            )
        # The field group's HEADERS are the cohort's business here: an unreadable entry anywhere
        # under it fails both routes. A field that merely records no bound streams: its windows are
        # sized from the values the run reads (measured_region_source).
        if self.displacement is not None:
            unreadable = self.displacement.unreadable_header()
            if unreadable is not None:
                return (
                    f"a field header could not be read ({unreadable.rstrip('.')}), so what any region"
                    " of the field must pull is unknown. Check the field store: one unreadable entry"
                    " anywhere under it falls the whole group back"
                )
        for name in self._grids:
            try:
                self._pricing_bound(name)
            except TransformError as error:
                # Both halves of the refusal: the first says what is wrong, the second what to
                # change. A plan line carrying only the first tells the reader nothing to do.
                return " ".join(str(part).strip() for part in error.args if part)
            except Exception as error:  # an unreadable transform is a whole-volume answer, not a crash
                return (
                    f"the map for case '{name}' could not be read ({type(error).__name__}: {error}), so"
                    " what it does to a region is unknown. Check the group names under"
                    " 'transforms:'/'field:' and that every case has an entry in each"
                )
        return None

    def stream_region_source(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        del source_spatial_shape, cache_attribute
        source, target = self._grids_of(name)
        return list(
            source_window(target.sub_grid(tuple(target_slices)), source, self._pricing_bound(name), self._tap_margin)
        )

    @property
    def measures_at_run(self) -> bool:
        """Whether the run sizes this stage's windows from the data it reads: any field at all.

        Declared or stored: both are priced as the identity by the plan, so both need the run to say
        where they actually reach. An affine-only ``transforms`` is not one of them: its bound is
        exact from the coefficients. Read off the plan's own records, which ``transform_shape`` fills
        for every case.
        """
        return self.displacement is not None or any(stored.field for stored in self._maps.values())

    def case_working_multiple(self, name: str) -> float:
        """The sampling grid, plus the field windows this case's region holds beside it.

        A region's field window is its own world box on the FIELD's grid, no halo whatever the
        displacement, so its size relative to the region is a ratio of voxel densities and both are
        in the headers. A field solved on the case's own grid costs three volumes-worth beside the
        three the sampling grid costs; one solved four times coarser per axis costs a sixteenth of
        that. A dense field stored under ``transforms`` is read on its region's box and handed to ITK
        as a declared one is, so it costs the same. Answered from headers alone; a case whose grids
        are not both known answers the class's figure.
        """
        base = float(self.working_multiple)
        # NOT the general walk. A map that does not factorise is walked coordinate by coordinate in
        # float64 and holds 21.4 to 21.6 volumes-worth where a separable one holds 0.19 to 2.85, but
        # that walk slabs ITSELF against the declared budget (konfai.data.sampling), so it is bounded
        # whatever the region is.
        if self.displacement is None and self.transforms is None:
            return base
        try:
            _source, target = self._grids_of(name)
            fields = [] if self.displacement is None else [self.displacement.infos(name)]
            if self.transforms is not None:
                fields.extend(self._stored_field_headers(name))
            target_voxel = float(np.prod(np.abs(np.asarray(target.spacing_xyz, dtype=np.float64))))
            # Components times the field's voxels per target voxel, per field.
            windows = []
            for shape, attribute in fields:
                field = Grid.of([int(extent) for extent in shape[1:]], attribute, f"the field for case '{name}'")
                field_voxel = float(np.prod(np.abs(np.asarray(field.spacing_xyz, dtype=np.float64))))
                if field_voxel <= 0.0:
                    return base
                windows.append(max(1, int(shape[0])) * target_voxel / field_voxel)
        except Exception:
            # Headers this stage has not met yet, or a field group it cannot resolve: the class's
            # figure is the honest answer, and the region sizing already treats it as a floor.
            return base
        # In the PLAN'S currency, which counts a volume at CASE_ELEMENT_BYTES: a field window is
        # charged at the ceiling its values are held at (float64 unless precision is fast), so each
        # of its components weighs two of the plan's volumes, not one.
        from konfai.data.patching.budget import CASE_ELEMENT_BYTES

        widening = self._field_element_bytes / CASE_ELEMENT_BYTES
        return base + sum(windows) * widening * _FIELD_WINDOW_COPIES

    @property
    def _field_dtype(self) -> type:
        """The CEILING a field's values are held at, never the width they are widened to.

        float64 is the bit-exact contract with SimpleITK and narrows nothing; a field stored in
        float32 stays float32 under it, losslessly (see
        :func:`~konfai.utils.ITK._displacement_stage`). ``precision: fast`` lowers the ceiling to the
        float32 its coordinate walk runs in.
        """
        return np.float32 if self.precision == "fast" else np.float64

    @property
    def _field_element_bytes(self) -> int:
        """What the PLAN charges a field value, at the ceiling rather than at the width the store
        turns out to hold: the plan reads no field header, and over-charging reserves memory a run
        does not need, which is the safe direction."""
        return int(np.dtype(self._field_dtype).itemsize)

    def measured_region_source(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        """The region's source window, walked through the map along the region's faces.

        The field a region needs is read over its own box for sampling regardless, so the walk costs
        no read. See :func:`~konfai.data.sampling.walked_window` for what the faces bound.
        """
        del source_spatial_shape, cache_attribute
        source, target = self._grids_of(name)
        region = target.sub_grid(tuple(target_slices))
        return list(walked_window(region, source, self._stages(name, region), self._tap_margin))

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        # The recorded grid, not one read off `cache_attribute`: what arrives here describes the
        # REGION, down to an Origin of its own. See _record().
        del cache_attribute
        return self._sample(name, tensor, tuple(context.target), [part.start for part in context.source])

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = [int(extent) for extent in tensor.shape[1:]]
        # Re-recorded on every call: the whole-volume path is handed the case's own evolved header,
        # and a grid recorded by an earlier walk may describe the stored volume rather than this
        # stage's true input (a Canonical upstream, a Resample before this one).
        self._record(name, shape, cache_attribute)
        source, target = self._grids_of(name)
        # The same call the streamed path makes, over one region that happens to be the whole grid:
        # equality between the two paths is then a property of the code, not a claim about it.
        whole = tuple(slice(0, extent) for extent in target.size_zyx)
        try:
            result = self._sample(name, tensor, whole, [0] * source.rank)
        finally:
            self._sitk_input.drop()  # a whole volume is never held past its call
        self.write_stream_cache_attribute(cache_attribute, shape, name)
        return result

    def _sample(
        self,
        name: str,
        sub_tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        budget_bytes: float | None = None,
    ) -> torch.Tensor:
        # 'fast' walks the coordinates in float32 for the whole sample: an opt-in the stage's
        # declaration made for its own data. The default context is the bit-exact float64 walk.
        if self.precision == "fast":
            with coordinate_precision(torch.float32):
                return self._sample_in_context(name, sub_tensor, target_slices, region_starts, budget_bytes)
        return self._sample_in_context(name, sub_tensor, target_slices, region_starts, budget_bytes)

    def _sample_in_context(
        self,
        name: str,
        sub_tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        budget_bytes: float | None = None,
    ) -> torch.Tensor:
        """One region's resample; ``budget_bytes`` bounds the walk's slabs (the machine's when None)."""
        source, target = self._grids_of(name)
        region = target.sub_grid(target_slices)
        stages = self._stages(name, region)
        shape, mode = list(source.size_zyx), self._mode(sub_tensor)
        # A map that factorises is read one axis at a time, the same arithmetic without the terms
        # that are zero and without a coordinate per voxel, and it is most maps. The general form is
        # what a rotation or a displacement needs; the two are bit-identical wherever both apply.
        axes = separable_source_index(target, source, stages, sub_tensor.device, target_slices)
        if axes is not None:
            order = blend_order(target, source)
            return gather_separable(sub_tensor, axes, region_starts, shape, mode, self.fill_value, order)
        # On the HOST, ITK's own resampler is the fastest one there is: sitk.Resample over the same
        # region, through the same stages, is 12x the torch walk per voxel (27 vs 326 ns, 12
        # threads); the walk is written for the GPU, where it is 40x faster again. Same rule, same
        # window, same fill, same working dtype, so the two agree to the ulp of the blend. On an
        # INTEGER payload that ulp can straddle a truncation boundary and become one whole unit
        # (2 voxels in 7560 through a rotation, pinned in test_sampling.py); a nearest pick copies
        # voxels and has no such seam. On the host the exact answer is also the cheapest.
        #
        # A REGION is read by the filter through its own origin and scanline, not the whole grid's,
        # which moves exact half-voxel ties. With no map between the grids, a region is walked on
        # the whole grid's terms instead, as ITK reads the whole grid (scanline_map).
        # Cubic is Keys' kernel, which ITK does not have: it keeps its own walk.
        as_itk = mode != "cubic" and scanline_map(target, source, stages)
        whole = region.size_zyx == target.size_zyx and list(sub_tensor.shape[1:]) == shape
        # The whole grid through the filter only where ITK picks the ties the regions' walk picks.
        itk_whole = whole and _itk_picks_as_the_walk()
        if sub_tensor.device.type == "cpu" and sitk is not None and (itk_whole or not as_itk):
            resampled = _resample_with_sitk(
                sub_tensor, region, source, stages, region_starts, mode, self.fill_value, self._sitk_input
            )
            if resampled is not None:
                return resampled
        if as_itk and sub_tensor.device.type == "cpu":
            # On the host this walk stands in for ITK's filter, so it keeps ITK's float64 whatever
            # the precision: 'fast' is a trade the device walk makes.
            with coordinate_precision(torch.float64):
                return self._walk(
                    sub_tensor, target, source, stages, target_slices, region_starts, mode, True, budget_bytes
                )
        return self._walk(sub_tensor, target, source, stages, target_slices, region_starts, mode, as_itk, budget_bytes)

    def _walk(
        self,
        sub_tensor: torch.Tensor,
        target: Grid,
        source: Grid,
        stages: SpatialStages,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        mode: str,
        as_itk: bool,
        budget_bytes: float | None,
    ) -> torch.Tensor:
        """One region through the torch walk: the device's route, and the host's where ITK's filter
        cannot answer as it answers for the whole grid. ``as_itk`` walks ITK's own index and blend
        (:func:`~konfai.data.sampling.scanline_index`)."""
        region = target.sub_grid(target_slices)
        shape = list(source.size_zyx)
        # The walk's coordinate tensor is float64 x rank: on a large region it dwarfs the gathered
        # payload (9 GB beside a 1 GB slab). Walking and gathering slab by slab bounds both under one
        # budget and changes no value: the row indices stay global to the region and the gather's
        # window and starts are the region's own.
        rows_total = int(region.size_zyx[0])
        rows = walk_rows(region, stages, sub_tensor.device, budget_bytes)

        def walk(start: int, stop: int) -> torch.Tensor:
            if not as_itk:
                return source_index_rows(region, source, stages, sub_tensor.device, start, stop)
            first = int(target_slices[0].start)
            return scanline_index(
                target, source, (slice(first + start, first + stop), *target_slices[1:]), sub_tensor.device
            )

        if rows >= rows_total:
            coordinates = walk(0, rows_total) if as_itk else source_index(region, source, stages, sub_tensor.device)
            return gather(sub_tensor, coordinates, region_starts, shape, mode, self.fill_value, as_itk)
        # Each slab lands in the one output as it is gathered. Slabs held for a cat were a second
        # output resident at the join, on exactly the regions that are large against the budget.
        out = torch.empty(
            (int(sub_tensor.shape[0]), *(int(extent) for extent in region.size_zyx)),
            dtype=sub_tensor.dtype,
            device=sub_tensor.device,
        )
        for start in range(0, rows_total, rows):
            stop = min(rows_total, start + rows)
            coordinates = walk(start, stop)
            out[:, start:stop] = gather(sub_tensor, coordinates, region_starts, shape, mode, self.fill_value, as_itk)
            del coordinates
        return out

    def _mode(self, tensor: torch.Tensor) -> str:
        """``nearest``, ``linear`` or ``cubic``: what a sampler asks before it blends anything.

        A dtype cannot settle this on its own, so the heuristic claims the label dtypes and nothing more
        and ``interpolation`` answers for the rest. Getting it wrong is silent: two blended labels give a
        third that was in no input.
        """
        return self.interpolation or default_interpolation(tensor)

    @property
    def _tap_margin(self) -> int:
        """Cubic reads four taps per axis (floor-1 .. floor+2): one more voxel of window each way."""
        return 2 if self.interpolation == "cubic" else 1

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        """Push the target grid over the source's, so the case now IS the grid it was written on.

        Pushed and not replaced: the source geometry stays underneath for :meth:`inverse` to pop
        back to.
        """
        shape = [int(extent) for extent in source_spatial_shape]
        source, missing = Grid.from_header(shape, cache_attribute, f"case '{name}'" if name else "the case")
        target = self._target.of(source, name)
        written = {
            "Spacing": target.spacing_xyz,
            "Origin": target.origin_xyz,
            "Direction": target.direction_xyz.ravel(),
        }
        for key in _GEOMETRY_KEYS:
            # Only over a geometry that was there: a case stored without an Origin is resampled by
            # ratio. Key by key, and not all-or-nothing, because ``inverse`` pops exactly what is
            # present.
            if key not in missing:
                cache_attribute[key] = written[key]
        # Two entries: the source's extent under the target's. A later stage reads the top one, and
        # the inverse pops both and restores the source's.
        cache_attribute["Size"] = np.asarray(shape)
        cache_attribute["Size"] = np.asarray([int(extent) for extent in target.size_zyx])

    # ------------------------------------------------------------------ the plan

    #: Below this, a case is worth a line in the plan: it reaches only part of the target grid and
    #: the rest of what it writes is fill. Above it the note would round to "100.0%".
    _WORTH_SAYING = 0.999

    #: How many probes per axis the coverage estimate uses. Coverage is a volume ratio between two
    #: boxes that a rotation makes a polytope, so it is counted rather than solved; capped because it
    #: is a plan line, not a result.
    _COVERAGE_PROBES = 24

    def coverage(self, name: str) -> float:
        """The fraction of the target grid that reads from inside the recorded case."""
        source, target = self._target_of(name)
        return self._coverage(source, target, self._map_bound(name))

    def _map_bound(self, name: str) -> TransformBound | None:
        """The declared map's bound, for a coverage judged where the samples actually land.

        ``None`` when there is no map, or when nothing bounds it: a coverage that cannot be judged
        must not refuse. A field prices as zero displacement here (:meth:`_pricing_bound`), so a
        stored affine beside it still places the samples.
        """
        if self.transforms is None and self.displacement is None:
            return None
        try:
            return self._pricing_bound(name)
        except Exception:  # an unreadable or unbounded map answers None, never a crash
            return None

    @classmethod
    def _coverage(cls, source: Grid, target: Grid, bound: TransformBound | None = None) -> float:
        """The fraction of ``target`` that reads from inside ``source``, from geometry alone.

        Judged THROUGH the declared map's affine part: a stored transform is what makes a cross-frame
        pair meet, and a coverage judged before applying it would call every such registration
        disjoint. The interval MOVES the lattice too, and only what varies widens the inside band.
        Counted on a capped lattice rather than solved: the sampled set is a box only while the grids
        are axis-aligned.
        """
        axes = [
            np.linspace(0.0, float(extent) - 1.0, min(cls._COVERAGE_PROBES, int(extent)))
            for extent in reversed(target.size_zyx)
        ]
        lattice = np.stack([axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=-1)
        to_world = target.index_to_world if bound is None else target.index_to_world.then(bound.affine)
        index = to_world.then(source.world_to_index).apply(lattice)
        low = high = np.zeros((1, source.rank))
        if bound is not None:
            # The interval, folded into index space the way a world box is: its two ends land where
            # the matrix sends them, and a negative entry swaps which end is which. As a radius, a
            # map that sent every sample past the case would reach back over it just as far.
            matrix = source.world_to_index.matrix
            rise, fall = np.maximum(matrix, 0.0), np.minimum(matrix, 0.0)
            low = (rise @ bound.low_xyz + fall @ bound.high_xyz)[None, :]
            high = (rise @ bound.high_xyz + fall @ bound.low_xyz)[None, :]
        inside = np.ones(index.shape[0], dtype=bool)
        for axis in range(source.rank):
            extent = float(source.size_zyx[source.rank - 1 - axis])
            # A probe reaches [index + low, index + high]: inside when that span meets the grid.
            inside &= (index[:, axis] + high[:, axis] >= -0.5) & (index[:, axis] + low[:, axis] < extent - 0.5)
        return float(np.count_nonzero(inside)) / float(inside.size)

    def _refuse_if_disjoint(self, name: str) -> None:
        """Refuse a case that does not meet the target grid anywhere.

        Its output would be ``fill`` from edge to edge, which no arithmetic finds and nothing
        downstream reports: a median over the cohort would simply be pulled toward the background.
        Counted from the headers, before a byte is read.

        Never with a field configured, DECLARED OR STORED: its reach is unknown before its values are
        read, the plan prices both at the identity, and bridging two frames is what a field may be
        for. A cohort registered onto a template it sits 25 mm from covers nothing until its own
        field is applied.
        """
        if self._target_is_own or self._prices_a_field(name):
            return
        source = self._source_grid(name)
        if self._covering.get(name) is source:
            return
        if self.coverage(name) > 0.0:
            self._covering[name] = source
            return
        where = f"case '{name}'" if name else "the case"
        raise TransformError(
            f"'Resample' would write {where} as nothing but 'fill': it does not overlap"
            f" {self._target.describe()} anywhere, so no voxel of the target grid reads from it.",
            "The two are in different places in physical space. Check that they share a frame (an"
            " acquisition's stage coordinates are not an anatomical one), pick a target the cohort"
            " actually surrounds, or drop this case with 'subset'.",
        )

    def _prices_a_field(self, name: str) -> bool:
        """Whether this case's map carries a field the plan prices at the identity.

        A field's reach is known only once its values are read, and the plan reads none. Every
        geometric judgement built on that price is about two grids sitting bare in world space, not
        about where the samples land, so neither the refusal nor the plan's coverage note may speak:
        a member judged bare can cover none of the target while the run reads it in full.
        """
        if self.displacement is not None:
            return True
        return self.transforms is not None and bool(self._stored_map(name).field)

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        """What this case covers of the target grid: measured on the header HANDED OVER.

        Not on the grid recorded for the case: the plan asks a stage about its own input, which the
        stages before it decide. Nothing is recorded here either: a question must not move the state
        a region read depends on.
        """
        del group_dest
        notes: list[str] = []
        if self.displacement is not None:
            # Case-independent on purpose: the plan prints identical notes once, so this is one line
            # for the stage rather than one per case.
            notes.append(
                "each region's source window is sized from the field values read at run; the read"
                " estimate prices the field as zero"
            )
        try:
            source, missing = Grid.from_header([int(extent) for extent in shape], cache_attribute, f"case '{name}'")
            if not missing & self._target.needs and not self._prices_a_field(name):
                covered = self._coverage(source, self._target.of(source, name), self._map_bound(name))
                if covered < self._WORTH_SAYING:
                    notes.append(
                        f"case '{name}' covers {covered * 100:.1f}% of {self._target.describe()};"
                        f" the rest of what it writes is fill ({self.fill_value:g})"
                    )
        except TransformError:
            pass
        return "; ".join(notes) if notes else None

    # ------------------------------------------------------------------ the inverse

    def _inverse_geometry(self, cache_attribute: Attribute) -> list[int]:
        """Pop the geometry stack the forward pushed and return the size the inverse restores."""
        cache_attribute.pop_np_array("Size")
        size = cache_attribute.pop_np_array("Size")
        for key in _GEOMETRY_KEYS:
            # Present iff the forward pushed it (see write_stream_cache_attribute): popping restores
            # the case's own, and a key the case never had is one this never wrote.
            if key in cache_attribute:
                cache_attribute.pop_np_array(key)
        return [int(extent) for extent in size]

    @staticmethod
    def _grid_from(cache_attribute: Attribute, shape: list[int]) -> Grid:
        if Grid.readable(cache_attribute):
            return Grid.of(shape, cache_attribute, "the case")
        return Grid.identity(shape)

    def _inverse_grids(self, cache_attribute: Attribute, shape: list[int]) -> tuple[Grid, Grid]:
        """``(what the accumulator is on, what to write back onto)``: both off the pushed stack.

        The forward stacked the source geometry under the target's, so the inverse reads the grid it
        is holding, pops, and reads the grid it is restoring. A copy is popped when the caller is
        only asking.
        """
        held = self._grid_from(cache_attribute, [int(extent) for extent in shape])
        restored_shape = self._inverse_geometry(cache_attribute)
        return held, self._grid_from(cache_attribute, restored_shape)

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        if self.transforms is not None or self.displacement is not None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    "resampling through a map inverts to resampling through its inverse, and that"
                    " inverse is not declared here, so a prediction finalize through this stage"
                    " assembles the volume. The forward direction streams"
                ),
            )
        try:
            self._inverse_geometry(Attribute(cache_attribute))
        except DatasetManagerError:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    "the grid this stage resampled off is not on the attribute it is being asked to"
                    " invert, so the shape it restores is unknown here. The forward direction streams"
                ),
            )
        return PatchLocality(LocalityKind.REGRID)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        try:
            return self._inverse_geometry(Attribute(cache_attribute))
        except DatasetManagerError:
            return shape

    def inverse_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        del source_spatial_shape
        self._inverse_geometry(cache_attribute)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self._target_is_own and (self.transforms is not None or self.displacement is not None):
            raise TransformError(
                "'Resample' has no inverse here: it changes no grid, so undoing it is undoing its"
                " map, which is applying a different map, not this one backwards.",
                "Set 'inverse: false' on this stage, or declare a second Resample with the inverse"
                " transforms in the chain that needs it.",
            )
        held, restored = self._inverse_grids(cache_attribute, [int(extent) for extent in tensor.shape[1:]])
        whole = tuple(slice(0, extent) for extent in restored.size_zyx)
        return self._resample_between(restored, held, tensor, whole, [0] * restored.rank)

    def stream_region_inverse(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        del name
        held, restored = self._inverse_grids(cache_attribute, [int(extent) for extent in context.source_shape])
        return self._resample_between(
            restored, held, tensor, tuple(context.target), [part.start for part in context.source]
        )

    def stream_region_target(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        del name
        held, restored = self._inverse_grids(Attribute(cache_attribute), [int(e) for e in source_spatial_shape])
        identity = TransformBound.exact(AffineMap.identity(restored.rank))
        return list(source_window(restored.sub_grid(tuple(target_slices)), held, identity, self._tap_margin))

    def _resample_between(
        self,
        target: Grid,
        source: Grid,
        tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
    ) -> torch.Tensor:
        """One region of ``target``, read off ``source`` with no map between them."""
        shape, mode = list(source.size_zyx), self._mode(tensor)
        axes = separable_source_index(target, source, (), tensor.device, target_slices)
        if axes is not None:
            order = blend_order(target, source)
            return gather_separable(tensor, axes, region_starts, shape, mode, self.fill_value, order)
        # ITK's index and blend on the host, as the forward reads a region: a slab holds what the whole
        # holds. A device keeps its fused gather, twice as fast on an oblique or permuted case. Either
        # walks slab by slab under the budget.
        as_itk = mode != "cubic" and tensor.device.type == "cpu" and scanline_map(target, source, ())
        return self._walk(tensor, target, source, (), target_slices, region_starts, mode, as_itk, None)

    @property
    def _target_is_own(self) -> bool:
        return isinstance(self._target, _OwnGrid)
