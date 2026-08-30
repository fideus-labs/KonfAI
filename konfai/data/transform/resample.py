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
    bound_of,
)
from konfai.data.sampling import (
    blend_order,
    coordinate_precision,
    gather,
    gather_separable,
    sampling_dtype,
    separable_source_index,
    source_index,
    source_index_rows,
    source_window,
    walk_rows,
)
from konfai.data.transform.base import (
    LocalityKind,
    PatchLocality,
    RegionContext,
    TransformInverse,
    sitk,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError
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

    The target that makes a cohort foldable. A spacing lines up densities and a shape lines up
    extents, but both leave each case where it was; this adopts one grid whole, which is what gives
    ``Reduce``'s ``grid: strict`` something true to compare. That is the atlas-template build.

    THE REFERENCE IS AN IMAGE, NOT A LIST OF NUMBERS. A grid is fifteen numbers in two axis orders
    at once, and transcribing them by hand is reliably wrong: silently, because a transposed grid
    resamples perfectly well onto the wrong place. Naming an
    image cannot make it: the header IS the declaration. It is also what an atlas loop needs, where
    round N+1's reference is round N's own output.

    An entry containing ``{case}`` FOLLOWS THE CASE: each case adopts the grid of its own entry --
    ``reference: '{case}', reference_group: DVF`` puts every moved image on its own field's grid,
    which is the registration idiom (a displacement field is defined ON the fixed grid). The
    literal spelling stays one lookup for the whole cohort; a per-case one is one per case,
    headers only either way.
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
        """The reference's grid, read from its header once per distinct entry.

        Headers only, and memoized by ENTRY: a literal reference is one lookup for the whole
        cohort, a per-case one is one per case, never the same answer bought again.
        """
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

    A private symbol, so it is optional on its own: guarded together with the package, a SimpleITK
    that does not export it would have read as no SimpleITK at all. Its caller
    (:class:`_SitkInput`) allocates a fresh image instead when it is missing.
    """
    try:
        from SimpleITK.SimpleITK import _SetImageFromArray
    except ImportError:
        return None
    return _SetImageFromArray


_set_image_from_array = _optional_image_filler() if sitk is not None else None


class _SitkInput:
    """ITK's input image for :func:`_resample_with_sitk`, reused across the regions of a sweep.

    ``GetImageFromArray`` allocates and zero-fills a new image per array, and the fresh mapping
    pays its page faults: 44-48 ms per 113 MiB against 6-9 ms filled in place (SimpleITK 2.5.5,
    a resample executed between two fills). The image is filled in place while the regions keep
    one shape and dtype and replaced when they do not; only one is ever held, the old dropped
    before the next is allocated. The whole-volume call drops it on its way out: what is held
    between calls is a region, never a case.

    The fill is the primitive the wrapper itself calls, ``_SetImageFromArray``
    (SimpleITK/SimpleITK#2683 asks for it as an ``outputImage`` argument). It checks the byte
    length only, so the shape and dtype key is what keeps a same-length array from being
    reinterpreted.
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

    ``payload`` is the SOURCE window read for this region (``region_starts`` says where it sits
    in the source grid); ``region`` is the target grid of the region. The window becomes an image
    with the source's geometry shifted to its start; the target region an image with its own
    grid; the stages one composite transform (:func:`~konfai.utils.ITK.encode_transform_stages`),
    applied by ``sitk.Resample`` in the direction KonfAI's walk applies it (target point through
    the stages to the source point). Channels are resampled one by one, which is what ITK does
    for a vector image anyway. ``None`` for a payload ITK has no pixel type for (bool, f16, bf16).
    """
    from konfai.utils.ITK import encode_transform_stages

    if payload.dtype in (torch.bool, torch.bfloat16, torch.float16):
        return None
    interpolator = {"nearest": sitk.sitkNearestNeighbor, "linear": sitk.sitkLinear, "cubic": sitk.sitkBSpline}
    if mode == "cubic":
        return None  # ITK's BSpline is not Keys' Catmull-Rom: the walk keeps its own cubic
    rank = source.rank
    transform = encode_transform_stages(stages) if stages else sitk.Transform(rank, sitk.sitkIdentity)
    # The window's own origin: the source origin moved by the window's start along each axis.
    start_index = np.asarray(list(reversed(region_starts)), dtype=np.float64)  # (x, y, z)
    window_origin = source.index_to_world.apply(start_index)
    # The target grid is given to the filter, not carried by an image standing in for it: a
    # reference image is read for its geometry and never for its pixels, so allocating and
    # zero-filling a region's worth of them is a copy nothing reads.
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(e) for e in reversed(region.size_zyx)])
    resampler.SetOutputOrigin(np.asarray(region.origin_xyz, dtype=np.float64).tolist())
    resampler.SetOutputSpacing(np.asarray(region.spacing_xyz, dtype=np.float64).tolist())
    resampler.SetOutputDirection(np.asarray(region.direction_xyz, dtype=np.float64).ravel().tolist())
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator[mode])
    resampler.SetDefaultPixelValue(float(fill))
    # A blend interpolates in the dtype the walk accumulates in, and torch makes the final cast:
    # ITK would otherwise accumulate an integer payload in double and cast inside the filter, where
    # one ulp of difference becomes a whole unit after the truncation both sides do. A nearest pick
    # copies voxels, so it keeps the payload's own dtype, exactly as the walk does.
    #
    # The filter is ASKED for that dtype instead of being handed a region converted into it. ITK
    # interpolates in double from either, so the values are the same for every dtype float32 carries
    # exactly (an int32 or int64 past 2^24 blends from its exact values where the cast rounded them),
    # and the conversion that was a copy of the whole region becomes the filter's own output.
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
        resampler.SetOutputPixelType(image.GetPixelID() if mode == "nearest" else blend_pixel_id)
        # Held: a view borrows the image's buffer, and a temporary's is freed under it.
        resampled = resampler.Execute(image)
        np.copyto(landing[channel], sitk.GetArrayViewFromImage(resampled), casting="unsafe")
    return result


@dataclass(frozen=True)
class _StoredMap:
    """What the plan keeps of a case's decoded stored transform: its bound, and whether the map IS
    the bound's affine part (every stage affine, folded exactly as the walk folds them).

    The decoded stages are not kept. A dense field is a values-sized array per case, and the
    launcher decodes every case as the plan is built: 50 cases of a 160x256x256 field were 11.7 GiB
    resident and serialised to every rank. The bound is three vectors, and it is all a pull map
    needs; the stages are decoded again where a region of their case is sampled, one case at a time
    (:meth:`Resample._stored_stages`).
    """

    bound: TransformBound
    affine: bool


def _stages_bytes(stages: SpatialStages) -> int:
    """What decoded stages hold: a dense field's values, an affine map next to nothing."""
    return sum(stage.values.nbytes for stage in stages if isinstance(stage, DisplacementStage))


#: How many times over a region's field window is materialised at once while the region is sampled.
#: Handing the field to ITK costs three: the values the read cached, the rank component images
#: encode_transform_stages builds from them, and the vector image sitk.Compose builds beside those
#: (konfai/utils/ITK.py, all live at the DisplacementFieldTransform call). Measured 21.2 GiB held
#: against 3.54 charged, on a fold whose plan then announced 4.65 GiB and whose run was killed at
#: 11.63 -- and killed in its FIRST region, which no run-time measurement can save, since there is
#: nothing measured yet when it is cut. That is why this is priced rather than left to the probe.
#:
#: Charged on every route, though only the host one goes through ITK: a stage is asked what it holds
#: before a device is chosen, and over-charging costs a shorter region where under-charging costs
#: the run.
_FIELD_WINDOW_COPIES = 3.0

#: What one element of a decoded field weighs. It is read as float64 whatever the store holds
#: (:meth:`_DisplacementSource.read`), while the plan counts its volumes at
#: :data:`~konfai.data.patching._SWEEP_ELEMENT_BYTES`: the ratio is what a field window costs in the
#: currency the plan is written in.
_FIELD_ELEMENT_BYTES = 8


class _DisplacementSource:
    """A displacement field on disk: where it is, how far it reaches, and how to read a region of it.

    :class:`Resample` is its one owner (the family's spellings all resolve to it), so every refusal
    speaks as ``Resample`` and names ``field_group``, the argument the user declared.
    """

    def __init__(self, field: str | None, group: str | None) -> None:
        # A root of its own, or none: with no ``field`` path the fields are a GROUP of the run's own
        # dataset_filenames, one entry per case, which is how a cohort registered in place stores
        # them, beside the volumes they were solved on.
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
        self._scan_ok: bool | None = None
        self._probed: set[str] = set()

    def headers_readable(self) -> bool:
        """Whether every field entry's HEADER opens, memoized: the plan's one probe of the group.

        An unreadable entry fails both routes on whichever case reaches it, and a directory store
        lists its entries from the filesystem alone, so a corrupt one only surfaces at its header:
        scanned here, one entry at a time, before any case is chosen.
        """
        if self._scan_ok is not None:
            return self._scan_ok
        try:
            group = self.group_for(None)
            roots = [self.dataset] if self.dataset is not None else list(self.roots)
            for root in roots:
                for entry in root.get_names(group):
                    root.get_infos(group, entry)
        except Exception:  # an unreadable field dataset is a whole-volume answer, not a crash
            self._scan_ok = False
            return False
        self._scan_ok = True
        return True

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

        :meth:`headers_readable` walks what is on disk; it cannot know which cases the plan will
        ask for, so a missing entry would otherwise surface mid-run, after bytes are written.
        Memoized: one header read per case, at plan time.
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

    def read(self, name: str, region: tuple[slice, ...] | None, channels: int) -> torch.Tensor:
        group = self.group_for(name)
        root = self._root_for(name)
        if region is None:
            data, _attributes = root.read_data(group, name)
        else:
            data, _attributes = root.read_data_slice(group, name, (slice(None), *region))
        # float64, not .float(): the walk evaluates the field in float64 (DisplacementStage's own
        # contract), and a .float() here quantized a float64-stored field before the exact
        # arithmetic ever saw it -- a silent sitk divergence for any field an external tool wrote
        # in double. A float32 store widens losslessly.
        field = torch.from_numpy(np.ascontiguousarray(data)).to(torch.float64)
        if field.shape[0] != channels:
            raise TransformError(
                f"The field for case '{name}' has {field.shape[0]} component(s) where the case has"
                f" {channels} spatial axis/axes.",
                "A displacement field carries one component per spatial axis, component-first.",
            )
        return field


class Resample(TransformInverse):
    """Resample a case: onto another grid, through a stored map, or both, in one interpolation.

    Every resample in KonfAI is these two questions, and this is the only stage that answers them.

    **Which grid to write on**: at most one of:

    - nothing (the default): the case's own grid. The map moves the anatomy; the voxels stay put.
    - ``spacing``: the same field of view at another density, in physical order ``(x, y, z)``. A
      component left at ``0`` keeps its axis.
    - ``shape``: the same field of view at a given count, in array order ``(Z, Y, X)``. A component
      left at ``0`` keeps its axis. The two orders differ on purpose: a spacing is geometry, a
      shape is an array extent, and each is written in its own convention.
    - ``reference``: the grid of a stored image, adopted whole: extent, spacing, origin, direction.
      ``'{case}'`` in the entry follows the case: each case adopts the grid of its OWN entry in
      ``reference_group``: ``reference: '{case}', reference_group: DVF`` lands every moved image
      on its own field's grid, which is where a displacement field is defined.

    **What map to write it through**: any of, composed in this order:

    - ``field``: a displacement field, read in world units at each TARGET voxel. Its own grid, its
      own spacing: a field solved at 120 um moves a volume stored at 30 um without being upsampled.
    - ``transforms``: transforms stored beside the cases (rigid, affine, BSpline, dense field, or a
      composite of them) mapping GROUP to whether to invert it. The LAST declared is applied first,
      which is SimpleITK's own composite order.

    Left out, the map is the identity and this is a change of grid and nothing else.

    ONE INTERPOLATION, ALWAYS. A grid change and a warp asked for together are composed into a single
    coordinate per target voxel and the source is read once, at the displaced point. Doing it as two
    stages resamples twice, and a volume interpolated twice has lost detail the second pass invented
    no more of, which is the whole reason an atlas's appearance is rebuilt from native volumes.

    ``interpolation`` is ``nearest``, ``linear`` (the default; ``uint8`` defaults to ``nearest``) or
    ``cubic``: Keys' cubic convolution (Catmull-Rom, a = -1/2), interpolating and patch-local (four
    taps per axis, one extra voxel of streamed halo). It is NOT a prefiltered B-spline: scipy's
    ``order=3`` (nnU-Net's resampling) runs a global prefilter this stage does not, so a spline-3
    recipe is approximated, not reproduced.

    IT STREAMS, and what a region reads is known before a voxel of the SOURCE is touched. A rigid
    or affine map is an exact affine, so the source box of a target region is that region's box
    mapped through it. A BSpline and a dense field are values on a grid read through a non-negative
    kernel that sums to one, so the sup-norm of those values bounds the displacement at EVERY point: a
    theorem, not a sample of the boundary. A field on disk is read region by region, and the
    window a region samples is its own box: the sup of the values just read bounds that region's
    pull, so each slab pays exactly the halo ITS displacements require: measured at run, from a
    read the sampler needs regardless. Nothing is declared and nothing is recorded: the plan
    prices the reads as if the field were zero, and says so.

    ``align`` decides where a ``spacing`` or a ``shape`` grid SITS, and it is the one silent choice
    in the family: a quarter of a voxel of anatomy, made differently by every library that offers
    only one of them. ``extent`` keeps the field of view (the outer faces coincide); ``origin`` keeps
    voxel zero's centre where it is. A ``reference`` states its own placement and ignores this.

    WHAT IT REFUSES, rather than resample from a window it cannot size or in a space it does not have:

    - a case whose header carries no ``Origin``/``Spacing``/``Direction`` when the answer needs
      physical space (a reference, a stored transform, a field). A plain ``spacing``/``shape``
      resample does not: with no geometry a world coordinate IS an index, and the ratio is the map;
    - a transform type that decomposes into no bounded map, naming the type;
    - ``invert: true`` on anything but a rigid or affine map: inverting a spline or a field is a
      dense solve over the whole grid, and a field solved per region is not the restriction of the
      field solved once. Store the inverse, or invert it where it is written;
    - a case that does not meet the target grid anywhere: judged THROUGH the declared map, so a
      stored rigid bridging two scanner frames is not mistaken for disjointness. The output would
      be ``fill`` from edge to edge, and an all-background member is a plausible, wrong
      contribution to a median.

    A refusal the whole-volume path can serve (a case with no geometry, an unreadable entry in
    the field group) declares ``WHOLE_VOLUME`` with its reason and the run proceeds assembled:
    the chain only stops being bounded, and says so in the plan. One that no route can serve (a
    map that cannot be decoded, read or inverted, or a disjoint case) refuses as the plan is
    built, before a byte is written. A case reaching only PART of the target grid is legal and
    common (the rest takes ``fill``), and the plan prints how much of the grid it covers.
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
                " for a sharper image blend. Left unset, uint8 is taken for a label map and"
                " everything else is interpolated linearly.",
            )
        if precision not in ("exact", "fast"):
            raise TransformError(
                f"'Resample' has an unknown precision '{precision}'.",
                "'exact' (the default) walks coordinates in float64, bit-identical to"
                " sitk.Resample. 'fast' lets the device walk in float32: half the bytes and about"
                " twice the rows per slab, at ~|world|/2^24 of coordinate error -- for INTENSITY"
                " resamples only (on the host ITK's own resampler is used either way)."
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
        #: Per case: the grid its own header describes. Recorded where that header is in hand --
        #: transform_shape, called for every case as the manager is built. A region read hands back
        #: the REGION's Origin, so a grid rebuilt from what a streamed region arrives with would
        #: place the case by the corner of whichever slab is being written, and slide it further
        #: with every slab; every voxel would still be an interpolation of real data.
        self._grids: dict[str, Grid] = {}
        #: Per case: the geometry keys its header did not carry (see :meth:`Grid.from_header`).
        self._assumed: dict[str, frozenset[str]] = {}
        #: Per case: what the plan keeps of its stored map (see :class:`_StoredMap`).
        self._maps: dict[str, _StoredMap] = {}
        #: The decoded stages of the last cases sampled, most recent last, within
        #: ``stored_stage_bytes``. Not pickled: a rank decodes what it samples, not the cohort.
        self._stored: OrderedDict[str, SpatialStages] = OrderedDict()
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

    #: Bytes of decoded stages kept across cases: the affine maps of a whole cohort (a few hundred
    #: bytes each) and the dense fields of the last two or three, so a fold reading N members per
    #: region does not decode a member per region.
    stored_stage_bytes = 512 << 20

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
                f"'Resample' was given {' and '.join(named)}, which are three ways to say the same thing.",
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

        A stored map or a reference grid is applied in physical space and needs all three; a change
        of density needs the density it starts from; a change of extent needs nothing at all. Being
        exact about this is what lets one class serve a headerless array and a real volume.
        """
        if self.transforms is not None or self.displacement is not None:
            return frozenset(_GEOMETRY_KEYS)
        return self._target.needs

    def _record(self, name: str, shape: list[int], cache_attribute: Attribute) -> Grid:
        """The case's own grid, remembered under its name, with what its header left unsaid."""
        where = f"case '{name}'" if name else "the case"
        grid, missing = Grid.from_header(list(shape), cache_attribute, where)
        self._assumed[name] = missing
        if name:
            self._grids[name] = grid
        return grid

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

        Split from :meth:`_grids_of` because the two questions have different answers: the output
        SHAPE of a warp on the case's own grid is the case's own shape, knowable with no geometry at
        all, while SAMPLING it is not. Refusing the shape too would take down the plan of a chain
        whose honest answer is to fall back to the whole volume and say so.
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
        whose float rounding differs with where the region starts; a separable map is bit-identical
        whatever the region. Answers True when the question cannot be settled from the headers."""
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
            return separable_source_index(target, source, stages, torch.device("cpu")) is None
        except Exception:  # nosec B110 - a map this cannot read is priced as the general path
            return True

    def _grids_of(self, name: str) -> tuple[Grid, Grid]:
        source = self._source_grid(name)
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
        return source, self._target.of(source, name)

    # ------------------------------------------------------------------ the map

    def _stored_stages(self, name: str) -> SpatialStages:
        """This case's stored transforms, decoded and composed, in application order.

        The last cases' stages are held, most recent last, within ``stored_stage_bytes``: a run
        samples its cases one after the other, a fold reads its members region by region, and
        a loader interleaving cases through a dense field decodes at each switch past the bound.
        """
        stages = self._stored.pop(name, None)
        if stages is None:
            stages = self._decode_stored(name)
        self._stored[name] = stages
        held = sum(_stages_bytes(kept) for kept in self._stored.values())
        while len(self._stored) > 1 and held > self.stored_stage_bytes:
            held -= _stages_bytes(self._stored.popitem(last=False)[1])
        return stages

    def _stored_map(self, name: str) -> _StoredMap:
        """The plan's record of this case's stored map, decoded once and kept without its stages."""
        stored = self._maps.get(name)
        if stored is None:
            stages = self._stored.get(name)
            if stages is None:
                stages = self._decode_stored(name)
            rank = self._source_grid(name).rank
            stored = self._maps[name] = _StoredMap(
                bound_of(stages, rank), all(isinstance(stage, AffineStage) for stage in stages)
            )
        return stored

    def _decode_stored(self, name: str) -> SpatialStages:
        """This case's stored transforms read and decoded, application order, nothing kept."""
        from konfai.utils.ITK import invert_stages, read_transform_stages

        _require_simpleitk()
        rank = self._source_grid(name).rank
        stages: list[AffineStage | DisplacementStage] = []
        # Reversed: a CompositeTransform applies its members last-first, and this stage has always
        # built one from `transforms` in declaration order. Decoding normalizes each member to
        # application order, so the declared list is reversed here to mean the same thing it did.
        for group in reversed(list(cast("dict[str, bool]", self.transforms))):
            invert = self.transforms[group] if self.transforms else False
            decoded = None
            for dataset in self.datasets:
                if dataset.is_dataset_exist(group, name):
                    decoded = read_transform_stages(dataset, group, name)
                    break
            if decoded is None:
                raise TransformError(
                    f"'Resample' found no transform for case '{name}' in group '{group}'.",
                    "Every case needs an entry in every group named under 'transforms:'. Check the"
                    " group name, or drop the cases that have no transform with 'subset'.",
                )
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
        own world box: no halo, whatever the displacement is. What the halo sizes is the SOURCE
        read, which is a different question, answered from these very values
        (:meth:`measured_region_source`): memoized here so sizing and sampling share one read.
        """
        key = (tuple(int(extent) for extent in region.size_zyx), tuple(float(v) for v in np.ravel(region.origin_xyz)))
        cached = self._field_window
        if cached is not None and cached[0] == name and cached[1] == key:
            return cached[2]
        source = cast("_DisplacementSource", self.displacement)
        shape, attribute = source.infos(name)
        spatial = [int(extent) for extent in shape[1:]]
        grid = Grid.of(spatial, attribute, f"the field for case '{name}'")
        window = grid.index_window(region.world_box(), margin=1)
        values = source.read(name, window, len(spatial))
        stage = DisplacementStage(grid.sub_grid(window), values.numpy(), order=1)
        self._field_window = (name, key, stage)
        return stage

    def stream_abort(self, name: str) -> None:
        self._stored.pop(name, None)
        if self._field_window is not None and self._field_window[0] == name:
            self._field_window = None
        self._sitk_input.drop()

    def _stages(self, name: str, region: Grid) -> SpatialStages:
        """The whole map over one target region, in application order."""
        stages: list[AffineStage | DisplacementStage] = []
        if self.displacement is not None:
            stages.append(self._field_stage(name, region))
        if self.transforms is not None:
            stages.extend(self._stored_stages(name))
        return tuple(stages)

    def _bound(self, name: str) -> TransformBound:
        """What the map is guaranteed to do, from stored coefficients alone, no voxel read."""
        rank = self._source_grid(name).rank
        folded = TransformBound.exact(AffineMap.identity(rank))
        if self.displacement is not None:
            raise TransformError(
                "a field's reach is unknown before its values are read; nothing bounds it from headers."
            )
        if self.transforms is not None:
            folded = self._stored_map(name).bound.after(folded)
        return folded

    def _pricing_bound(self, name: str) -> TransformBound:
        """The map's bound as the PLAN prices it: headers and declarations, never a voxel.

        A field prices as zero displacement. The run never trusts this window: a field's
        regions are sized from the values it reads for sampling anyway
        (:meth:`measured_region_source`), so the optimism here costs estimate accuracy, not bytes.
        """
        if self.displacement is None:
            return self._bound(name)
        rank = self._source_grid(name).rank
        folded = TransformBound.exact(AffineMap.identity(rank))
        if self.transforms is not None:
            folded = self._stored_map(name).bound.after(folded)
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

        A refusal the whole-volume path can serve (a case with no geometry) stays a locality
        answer, and the run proceeds assembled. A stored transform that cannot be decoded, read or
        inverted fails the streamed path and the whole-volume one at the same line, so declaring
        WHOLE_VOLUME for it would print a plan the run then contradicts by dying per case, after
        bytes are written. ``transform_shape`` runs for every case as the plan is built, which is
        the earliest the failure is knowable and the only place it costs nothing.
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
        # The geometry is judged on the attribute in hand: the case's own header, as the base
        # contract has it, and not on what the cohort has been seen to carry: one case of a group
        # may lack an Origin while the rest have one, and a declaration made per case is the honest
        # one. A config-time probe hands over an empty header, which reads as a case with none.
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

        The COHORT's answer, not one case's: a locality is declared once for the stage while the
        cases are many, so a group whose entries are not uniformly decodable must fall back for all
        of them rather than for the ones that happen to be planned first. Exceptions are swallowed
        into a reason: this runs inside the plan, where a raise would take the run down instead of
        costing it the whole-volume path. The GEOMETRY is not judged here; that is per case, and
        :meth:`patch_locality` reads it off the header it is handed.
        """
        if self.transforms is not None and sitk is None:
            return (
                "SimpleITK is not installed, and a stored transform is applied in physical space by"
                " it. Install it (pip install konfai[itk]) to stream this stage"
            )
        # The field group's HEADERS are the cohort's business here: an unreadable entry anywhere
        # under it fails both routes on whichever case reaches it. A field that merely records no
        # bound streams: its windows are sized from the values the run reads (measured_region_source).
        if self.displacement is not None:
            if not self.displacement.headers_readable():
                return (
                    "an entry in the field group could not be header-read, so what any region of it"
                    " must pull is unknown. Check the field store: one unreadable entry anywhere"
                    " under it falls the whole group back"
                )
        for name in self._grids:
            try:
                self._pricing_bound(name)
            except TransformError as error:
                # Both halves of the refusal: the first says what is wrong, the second what to
                # change. A plan line carrying only the first tells the reader nothing to do.
                return " ".join(str(part).strip() for part in error.args if part)
            except Exception:  # an unreadable transform is a whole-volume answer, not a crash
                return (
                    f"the map for case '{name}' could not be read, so what it does to a region is"
                    " unknown. Check the group names under 'transforms:'/'field:' and that every"
                    " case has an entry in each"
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
        """Whether the run sizes this stage's windows from the data it reads: any declared field."""
        return self.displacement is not None

    def case_working_multiple(self, name: str) -> float:
        """The sampling grid, plus the field window this case's region holds beside it.

        A region's field window is its own world box on the FIELD's grid -- no halo, whatever the
        displacement, because the field is evaluated at the target's points -- so its size relative
        to the region is a ratio of voxel densities, and both are in the headers. At a field solved
        on the case's own grid that is one component per axis, three volumes-worth beside the three
        the sampling grid already costs; at a field solved four times coarser per axis it is a
        sixteenth of that, and the plan should not charge the same for the two.

        Answered from headers alone; a case whose grids are not both known yet answers the class's
        figure rather than guessing.
        """
        base = float(self.working_multiple)
        # NOT the general walk. A map that does not factorise is walked coordinate by coordinate in
        # float64 and holds 21.4 to 21.6 volumes-worth where a separable one holds 0.19 to 2.85 --
        # but that walk slabs ITSELF against the declared budget (konfai.data.sampling), so it is
        # bounded whatever the region is. Charging the region for it as well would shrink every
        # resampling chain sevenfold for memory the walk does not take.
        if self.displacement is None:
            return base
        try:
            _source, target = self._grids_of(name)
            shape, attribute = self.displacement.infos(name)
            field = Grid.of([int(extent) for extent in shape[1:]], attribute, f"the field for case '{name}'")
            target_voxel = float(np.prod(np.abs(np.asarray(target.spacing_xyz, dtype=np.float64))))
            field_voxel = float(np.prod(np.abs(np.asarray(field.spacing_xyz, dtype=np.float64))))
        except Exception:
            # Headers this stage has not met yet, or a field group it cannot resolve: the class's
            # figure is the honest answer, and the region sizing already treats it as a floor.
            return base
        if field_voxel <= 0.0:
            return base
        # In the PLAN'S currency, which counts a volume at _SWEEP_ELEMENT_BYTES: a field window is
        # read as float64 whatever the store holds (see _DisplacementSource.read, which widens on
        # purpose so an externally written double field is not quantized before the exact
        # arithmetic), so each of its components weighs two of the plan's volumes, not one.
        # Charging it at one was counting eight bytes as four.
        from konfai.data.patching import _SWEEP_ELEMENT_BYTES

        widening = _FIELD_ELEMENT_BYTES / _SWEEP_ELEMENT_BYTES
        window = max(1, int(shape[0])) * (target_voxel / field_voxel) * widening
        return base + window * _FIELD_WINDOW_COPIES

    def measured_region_source(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        """The region's source window, sized from the field itself: the read that samples also bounds.

        The field window a region needs is its own box, read for sampling regardless; the sup of
        the values just read bounds every interpolated displacement in the region (a convex
        combination cannot exceed the lattice values it blends), so the window is exact per region: a quiet
        slab pays a quiet halo.
        """
        del source_spatial_shape, cache_attribute
        source, target = self._grids_of(name)
        region = target.sub_grid(tuple(target_slices))
        return list(source_window(region, source, bound_of(self._stages(name, region), source.rank), self._tap_margin))

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
        # never a region's, and a grid recorded by an earlier walk may describe the stored volume
        # rather than this stage's true input (a Canonical upstream, a Resample before this one).
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
        # A map that factorises is read one axis at a time, which is the same arithmetic without the
        # terms that are zero and without a coordinate per voxel, and it is most maps, because most
        # volumes are stored axis-aligned. The general form is what a rotation or a displacement
        # needs, and the two are bit-identical wherever both apply.
        axes = separable_source_index(region, source, stages, sub_tensor.device)
        if axes is not None:
            order = blend_order(target, source)
            return gather_separable(sub_tensor, axes, region_starts, shape, mode, self.fill_value, order)
        # On the HOST, ITK's own resampler is the fastest one there is: sitk.Resample over the same
        # region, through the same stages, is 12x the torch walk per voxel on this arithmetic
        # (27 vs 326 ns, measured, 12 threads) -- the walk is written for the GPU, where it is 40x
        # faster again. Same rule (ITK's), same window, same fill, same working dtype, so the two
        # agree to the ulp of the blend: as far as a CPU and a CUDA run already agree. On an INTEGER
        # payload that ulp can straddle a truncation boundary, where it becomes one whole unit
        # (measured: 2 voxels in 7560 through a rotation, pinned in test_sampling.py). A nearest
        # pick copies voxels and has no such seam, which is what a label map takes.
        # 'precision: fast' is a permission the GPU walk uses; on the host the exact answer is also
        # the cheapest.
        if sub_tensor.device.type == "cpu" and sitk is not None:
            resampled = _resample_with_sitk(
                sub_tensor, region, source, stages, region_starts, mode, self.fill_value, self._sitk_input
            )
            if resampled is not None:
                return resampled
        # The walk's coordinate tensor is float64 x rank: on a large region it dwarfs the gathered
        # payload (9 GB beside a 1 GB slab, measured on an ExaSPIM mask chain). Walking and
        # gathering slab by slab bounds both under one budget, and changes no value: the row
        # indices stay global to the region, the gather's window and starts are the region's own,
        # so every row of every slab is the very tensor the single pass produces.
        rows_total = int(region.size_zyx[0])
        rows = walk_rows(region, stages, sub_tensor.device, budget_bytes)
        if rows >= rows_total:
            coordinates = source_index(region, source, stages, sub_tensor.device)
            return gather(sub_tensor, coordinates, region_starts, shape, mode, self.fill_value)
        # Each slab lands in the one output as it is gathered. Slabs held for a cat were a second
        # output resident at the join, on exactly the regions that are large against the budget.
        out = torch.empty(
            (int(sub_tensor.shape[0]), *(int(extent) for extent in region.size_zyx)),
            dtype=sub_tensor.dtype,
            device=sub_tensor.device,
        )
        for start in range(0, rows_total, rows):
            stop = min(rows_total, start + rows)
            coordinates = source_index_rows(region, source, stages, sub_tensor.device, start, stop)
            out[:, start:stop] = gather(sub_tensor, coordinates, region_starts, shape, mode, self.fill_value)
            del coordinates
        return out

    def _mode(self, tensor: torch.Tensor) -> str:
        """``nearest``, ``linear`` or ``cubic``: what a sampler asks before it blends anything.

        A dtype cannot settle this on its own: a CT is int16 and so is nothing else about it. The
        heuristic therefore claims ``uint8`` and nothing more, and ``interpolation`` answers for
        everything it cannot know. Getting it wrong is silent: two blended labels give a third
        that was in no input, in a volume that is still a label map.
        """
        return self.interpolation or ("nearest" if tensor.dtype == torch.uint8 else "linear")

    @property
    def _tap_margin(self) -> int:
        """Cubic reads four taps per axis (floor-1 .. floor+2): one more voxel of window each way."""
        return 2 if self.interpolation == "cubic" else 1

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        """Push the target grid over the source's, so the case now IS the grid it was written on.

        Pushed and not replaced: the source geometry stays underneath for :meth:`inverse` to pop back
        to, which is the whole of the stack this and ``_inverse_geometry`` share.
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
            # ratio, and inventing one for it would be a header nobody measured. Key by key, and not
            # all-or-nothing, because ``inverse`` pops exactly what is present, so what is pushed
            # and what is popped are the same condition, read at the two ends.
            if key not in missing:
                cache_attribute[key] = written[key]
        cache_attribute["Size"] = np.asarray(shape)
        cache_attribute["Size"] = np.asarray([int(extent) for extent in target.size_zyx])

    # ------------------------------------------------------------------ the plan

    #: Below this, a case is worth a line in the plan: it reaches only part of the target grid and
    #: the rest of what it writes is fill. Above it, the note would round to "100.0%" and say
    #: nothing, and a plan that says nothing on every line is one nobody reads.
    _WORTH_SAYING = 0.999

    #: How many probes per axis the coverage estimate uses. Coverage is a volume ratio between two
    #: boxes that a rotation makes a polytope, so it is counted rather than solved; capped because
    #: it is a plan line, not a result.
    _COVERAGE_PROBES = 24

    def coverage(self, name: str) -> float:
        """The fraction of the target grid that reads from inside the recorded case."""
        source, target = self._target_of(name)
        return self._coverage(source, target, self._map_bound(name))

    def _map_bound(self, name: str) -> TransformBound | None:
        """The declared map's bound, for a coverage judged where the samples actually land.

        ``None`` when there is no map: or when nothing bounds it: a coverage that cannot be judged
        must not refuse, and the unboundable configurations carry a fallback reason of their own.
        A field prices as zero displacement here (:meth:`_pricing_bound`), so a stored affine
        beside it still places the samples instead of the grids being judged bare.
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

        Judged THROUGH the declared map's affine part: a stored transform is what makes a
        cross-frame pair meet (an MR and a CT in different scanner frames with a rigid bridging
        them), and a coverage judged before applying it would call every such registration
        disjoint. The residual (a spline's or a field's sup-norm) only ever moves a sample by a
        bounded amount, so it widens the inside band rather than moving the lattice. Counted on a
        capped lattice rather than solved, because the sampled set is a box only while the grids
        are axis-aligned and a rotation makes it a polytope.
        """
        axes = [
            np.linspace(0.0, float(extent) - 1.0, min(cls._COVERAGE_PROBES, int(extent)))
            for extent in reversed(target.size_zyx)
        ]
        lattice = np.stack([axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=-1)
        to_world = target.index_to_world if bound is None else target.index_to_world.then(bound.affine)
        index = to_world.then(source.world_to_index).apply(lattice)
        margin_xyz = (
            np.zeros(source.rank)
            if bound is None
            # A world-space residual box reaches |W2I| @ r in index space, component-wise.
            else np.abs(source.world_to_index.matrix) @ np.asarray(bound.residual_xyz, dtype=np.float64)
        )
        inside = np.ones(index.shape[0], dtype=bool)
        for axis in range(source.rank):
            extent = float(source.size_zyx[source.rank - 1 - axis])
            inside &= (index[:, axis] >= -0.5 - margin_xyz[axis]) & (index[:, axis] < extent - 0.5 + margin_xyz[axis])
        return float(np.count_nonzero(inside)) / float(inside.size)

    def _refuse_if_disjoint(self, name: str) -> None:
        """Refuse a case that does not meet the target grid anywhere.

        Its output would be ``fill`` from edge to edge. That is not an error the arithmetic can
        find (every voxel of it is exactly what was asked for), so it is one nothing downstream
        would report: a median over the cohort would simply be pulled toward the background by a
        member that contributed no anatomy. Counted from the headers, before a byte is read.

        Never with a field configured: its reach is unknown before its values are read, and
        bridging two frames is precisely what a field may be for.
        """
        if self._target_is_own or self.displacement is not None or self.coverage(name) > 0.0:
            return
        where = f"case '{name}'" if name else "the case"
        raise TransformError(
            f"'Resample' would write {where} as nothing but 'fill': it does not overlap"
            f" {self._target.describe()} anywhere, so no voxel of the target grid reads from it.",
            "The two are in different places in physical space. Check that they share a frame (an"
            " acquisition's stage coordinates are not an anatomical one), pick a target the cohort"
            " actually surrounds, or drop this case with 'subset'.",
        )

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        """What this case covers of the target grid: measured on the header HANDED OVER.

        Not on the grid recorded for the case: the plan asks a stage about its own input, which the
        stages before it decide, and a note answered from the stored header would describe a volume
        that no longer exists by the time this stage sees it. Nothing is recorded here either, for
        the mirror reason: a question must not move the state a region read depends on.
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
            if not missing & self._target.needs:
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

        The forward stacked the source geometry under the target's, so the inverse needs no memory
        of the case: it reads the grid it is holding, pops, and reads the grid it is restoring. A
        copy is popped when the caller is only asking, because a declaration never mutates.
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
        except NameError:
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
        except NameError:
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
        region = target.sub_grid(target_slices)
        shape, mode = list(source.size_zyx), self._mode(tensor)
        axes = separable_source_index(region, source, (), tensor.device)
        if axes is not None:
            order = blend_order(target, source)
            return gather_separable(tensor, axes, region_starts, shape, mode, self.fill_value, order)
        coordinates = source_index(region, source, (), tensor.device)
        return gather(tensor, coordinates, region_starts, shape, mode, self.fill_value)

    @property
    def _target_is_own(self) -> bool:
        return isinstance(self._target, _OwnGrid)
