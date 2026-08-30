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

"""SimpleITK-based helpers for geometric transforms, resampling, and masking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai.utils.errors import TransformError

if TYPE_CHECKING:
    from konfai.data.geometry import AffineMap, AffineStage, DisplacementStage, Grid, SpatialStages, WorldBox


def _require_simpleitk() -> None:
    """Raise a clear project error when an ITK-only path is used without SimpleITK."""
    if sitk is None:
        raise TransformError("SimpleITK is required for this operation. Install it with `pip install konfai[itk]`.")


def read_displacement_field(path: str | Path) -> sitk.Image:
    """A displacement field, from an ITK image file OR an NGFF RFC-5 OME-Zarr store.

    A field written as a store cannot go through ``sitk.ReadImage``, and reading it as an ordinary
    image is worse than failing: the component axis looks like any other, so indexing it yields one
    third of the field and a registration that is wrong without being obviously wrong. This is the one
    place that knows how to open either form, so no caller has to decide.

    The result is ``sitkVectorFloat64``: what ``DisplacementFieldTransform`` requires.
    """
    _require_simpleitk()
    path = Path(path)
    if not path.is_dir():
        return sitk.ReadImage(str(path), sitk.sitkVectorFloat64)

    from konfai.utils.dataset import data_to_image, ome_zarr_attributes
    from konfai.utils.ome_zarr import get_ome_zarr_info, is_displacement_field, read_ome_zarr_data_slice

    if not is_displacement_field(path):
        raise TransformError(
            f"'{path}' is an OME-Zarr image, not a displacement field: its component axis is not "
            "typed as an NGFF RFC-5 displacement.",
            "A three-component volume is a perfectly ordinary image, so the store's own declaration "
            "is the only thing that tells them apart: write the field with "
            "write_ome_zarr(displacement_field=True).",
        )
    axes = get_ome_zarr_info(path)["axes"]
    n_axes = 1 + sum(axis in axes for axis in ("z", "y", "x"))  # channel-first C[Z]YX
    data, metadata = read_ome_zarr_data_slice(path, tuple(slice(None) for _ in range(n_axes)))
    # Origin, Spacing and Direction together: NGFF scale/translation alone cannot express the
    # direction matrix, so the geometry comes from the konfai sidecar through data_to_image.
    field = data_to_image(data, ome_zarr_attributes(metadata))
    return sitk.Cast(field, sitk.sitkVectorFloat64)


def _invert_via_displacement_field(transform: sitk.Transform, image: sitk.Image) -> sitk.DisplacementFieldTransform:
    if image is None:
        raise TransformError(
            "Inverting a non-linear transform requires a reference image to sample the displacement field, "
            "but none was provided."
        )
    displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_field_filter.SetReferenceImage(image)
    displacement_field = displacement_field_filter.Execute(transform)
    iterative_inverse = sitk.IterativeInverseDisplacementFieldImageFilter()
    iterative_inverse.SetNumberOfIterations(20)
    return sitk.DisplacementFieldTransform(iterative_inverse.Execute(displacement_field))


def _copy_transform(transform_cls: type[sitk.Transform], transform: sitk.Transform, invert: bool) -> sitk.Transform:
    transform = transform_cls(transform)
    if invert:
        transform = transform_cls(transform.GetInverse())
    return transform


def _open_transform(
    transform_files: dict[str | sitk.Transform, bool], image: sitk.Image = None
) -> list[sitk.Transform]:
    _require_simpleitk()
    transforms: list[sitk.Transform] = []

    for transform_file, invert in transform_files.items():
        if isinstance(transform_file, str):
            transform = sitk.ReadTransform(transform_file + ".itk.txt")
        else:
            transform = transform_file
        if transform.GetName() == "TranslationTransform":
            transform = _copy_transform(sitk.TranslationTransform, transform, invert)
        elif transform.GetName() == "Euler3DTransform":
            transform = _copy_transform(sitk.Euler3DTransform, transform, invert)
        elif transform.GetName() == "VersorRigid3DTransform":
            transform = _copy_transform(sitk.VersorRigid3DTransform, transform, invert)
        elif transform.GetName() == "AffineTransform":
            transform = _copy_transform(sitk.AffineTransform, transform, invert)
        elif transform.GetName() == "DisplacementFieldTransform":
            if invert:
                transform = _invert_via_displacement_field(transform, image)
        else:
            transform = sitk.BSplineTransform(transform)
            if invert:
                transform = _invert_via_displacement_field(transform, image)
        transforms.append(transform)
    if len(transforms) == 0:
        transforms.append(sitk.Euler3DTransform())
    return transforms


def compose_transform(
    transform_files: dict[str | sitk.Transform, bool], image: sitk.Image = None
) -> sitk.CompositeTransform:
    transforms = _open_transform(transform_files, image)
    result = sitk.CompositeTransform(transforms)
    return result


def apply_to_data_transform(data: np.ndarray, transform_files: dict[str | sitk.Transform, bool]) -> np.ndarray:
    transforms = compose_transform(transform_files)
    result = np.copy(data)
    for i in range(data.shape[0]):
        result[i, :] = transforms.TransformPoint(np.asarray(data[i, :], dtype=np.double))
    return result


def box_with_mask(mask: sitk.Image, label: list[int], dilatations: list[int]) -> np.ndarray:
    _require_simpleitk()

    dilatations = [int(np.ceil(d / s)) for d, s in zip(dilatations, reversed(mask.GetSpacing()), strict=False)]

    data = sitk.GetArrayViewFromImage(mask)  # a view: the mask is read for its shape and its labels, not held
    border = np.where(np.isin(data, label))
    box = []
    for w, dilatation, s in zip(border, dilatations, data.shape, strict=False):
        box.append([max(np.min(w) - dilatation, 0), min(np.max(w) + dilatation, s)])
    box = np.asarray(box)
    return box


def _linear_map(transform: sitk.Transform) -> AffineMap:
    """The exact world map of a linear transform: ``T(p) = M p + T(0)``.

    ``M`` comes from ``GetMatrix`` where the type has one: the number ITK itself resamples with,
    read past the centre/translation parameterisation that differs between Euler, Similarity,
    Scale and Affine, and from ``T(e_k) - T(0)`` otherwise. The offset is ``T(0)`` directly
    rather than assembled from centre and translation: one call, no cancellation, and true for
    every parameterisation at once.

    Probing is sound HERE and nowhere else in this file: for an affine map the columns are the map,
    exactly, by linearity. For a non-linear one the same arithmetic measures a local gradient and
    extrapolates it, which under-bounds, which is why a BSpline's affine part is the identity and
    all of its reach lives in the residual.
    """
    from konfai.data.geometry import AffineMap

    rank = int(transform.GetDimension())
    offset = np.asarray(transform.TransformPoint((0.0,) * rank), dtype=np.float64)
    if hasattr(transform, "GetMatrix"):
        matrix = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(rank, rank)
    else:
        basis = np.eye(rank)
        columns = [
            np.asarray(transform.TransformPoint(tuple(basis[k])), dtype=np.float64) - offset for k in range(rank)
        ]
        matrix = np.stack(columns, axis=1)
    return AffineMap(matrix, offset)


def _grid_of_image(image: sitk.Image) -> Grid:
    from konfai.data.geometry import Grid

    rank = int(image.GetDimension())
    return Grid(
        tuple(int(extent) for extent in reversed(image.GetSize())),
        np.asarray(image.GetOrigin(), dtype=np.float64),
        np.asarray(image.GetSpacing(), dtype=np.float64),
        np.asarray(image.GetDirection(), dtype=np.float64).reshape(rank, rank),
    )


def _displacement_stage(
    grid: Grid, values: np.ndarray, order: int, what: str, dtype: np.dtype | type = np.float64
) -> DisplacementStage:
    """A stage over ``values``, component-first ``(rank, *grid)``, contiguous in ``dtype``: one copy
    where they are not that already, none where they are.

    ``dtype`` is a CEILING, not a target: the values are held at the width the STORE holds them at,
    never widened past it. A field written in float32 -- which is how ITK writes one, and how a DVF
    normally sits on disk -- carries no more information as float64, so widening it buys a copy of
    twice the bytes to say exactly the same thing, three times over once the sampler holds its own
    (``_FIELD_WINDOW_COPIES``), and quantised straight back by any walk that runs in float32.

    So float64 (the default ceiling, the bit-exact contract with SimpleITK) narrows nothing that
    was stored wide, and lets a float32 store stay float32 -- losslessly, with no flag to set and
    no precision traded, which is also what lets :func:`~konfai.data.transform._warp_field_float32`
    apply such a field without a float64 transform to hold it. A caller whose coordinate walk runs
    in float32 lowers the ceiling to float32, and there a genuinely float64 field does narrow: that
    is the trade ``precision: fast`` names.

    Anything not already float32 or float64 -- an integer field, a float16 one -- is converted to
    the ceiling: those are widths SimpleITK has no pixel type for, and the walk no kernel for.
    """
    from konfai.data.geometry import DisplacementStage

    ceiling = np.dtype(dtype)
    stored = np.asarray(values).dtype
    held = stored if stored.kind == "f" and np.float32().itemsize <= stored.itemsize <= ceiling.itemsize else ceiling
    values = np.ascontiguousarray(values, dtype=held)
    if not np.isfinite(values).all():
        raise TransformError(
            f"{what} carries a non-finite displacement value, so no bound on its reach exists.",
            "A NaN or infinite coefficient means the transform was written from a failed solve;"
            " re-export it, or drop it from 'transforms:'.",
        )
    return DisplacementStage(grid, values, order)


def decode_transform_stages(transform: sitk.Transform) -> SpatialStages:
    """A stored transform as geometry stages in APPLICATION order, or a refusal naming the type.

    ``CompositeTransform`` applies its member list in REVERSE (the last added runs first: verified
    against SimpleITK, where ``GetNthTransform(0)`` is nonetheless the first added); the reversal is
    normalized here, once, so every consumer reads stages first-applied-first.
    """
    _require_simpleitk()
    if isinstance(transform, sitk.CompositeTransform):
        stages: list[AffineStage | DisplacementStage] = []
        for index in reversed(range(transform.GetNumberOfTransforms())):
            # Downcast restores the member's concrete type where GetNthTransform hands back the
            # generic wrapper, which the isinstance dispatch below cannot read.
            stages.extend(decode_transform_stages(transform.GetNthTransform(index).Downcast()))
        return tuple(stages)
    from konfai.data.geometry import AffineStage

    if isinstance(transform, sitk.BSplineTransform):
        # One copy per component, into the stack: the images stay alive for as long as the views do.
        coefficients = transform.GetCoefficientImages()
        values = np.stack([sitk.GetArrayViewFromImage(component) for component in coefficients])
        return (
            _displacement_stage(
                _grid_of_image(coefficients[0]), values, int(transform.GetOrder()), "this BSpline transform"
            ),
        )
    if isinstance(transform, sitk.DisplacementFieldTransform):
        # One copy of the field: the view is (Z, Y, X, rank) with the components (x, y, z) fastest,
        # and the stage's component-first float64 layout is written straight from it (measured
        # 159 MiB of peak against 50 MiB of content on a 128^3 field through GetArrayFromImage,
        # a per-component ascontiguousarray and a stack).
        field = transform.GetDisplacementField()
        values = np.array(np.moveaxis(sitk.GetArrayViewFromImage(field), -1, 0), dtype=np.float64, order="C")
        return (_displacement_stage(_grid_of_image(field), values, 1, "this displacement field"),)
    if transform.IsLinear():
        return (AffineStage(_linear_map(transform)),)
    raise TransformError(
        f"A stored '{transform.GetName()}' decomposes into no bounded map: how far a target region"
        " reaches into its source is unknown, so the region it must read is unbounded.",
        "Convert it to a displacement field when it is written, or use a rigid/affine/BSpline"
        " transform, which all decompose.",
    )


def read_transform_stages(
    dataset: Any,
    group: str,
    name: str,
    box: WorldBox | None = None,
    headers_only: bool = False,
    field_dtype: np.dtype | type = np.float64,
) -> SpatialStages:
    """The stored transform ``(group, name)`` of ``dataset`` as geometry stages in APPLICATION order.

    A displacement entry becomes its stage straight from the array the store hands over, on the
    grid its attributes describe: the field never passes through a SimpleITK image (a float32 store
    widens to float64 exactly, where the image route copied it three times over on the way in and
    once more on the way out). Every other entry decodes through :func:`decode_transform_stages`,
    as does everything a store that serves transforms alone (``read_transform`` and nothing else)
    hands over.

    ``box`` is the world box the caller will evaluate the map over, and it is read from the headers
    before a voxel is fetched: a field entry then comes back as its own sub-grid over the window
    that box falls in, plus the lattice point linear interpolation reaches for. Without it the whole
    entry is read, which is what a whole-volume call and the plan's own decode still want. A field
    solved at full resolution is gigabytes -- 14.5 GiB per ExaSPIM case, and float64 on the way in
    doubles it -- so a region that read the whole one would hold, per case, more than the budget
    sizing it was ever told about.

    Only a displacement entry is windowed. An affine is a matrix and a BSpline a coarse control
    grid: both are small, and a BSpline's coefficients are not indexed by the box anyway.

    ``headers_only`` is the plan's read: a dense field comes back as NO stage at all, which is the
    identity, and its values are never touched. Nothing bounds a field from headers -- so a plan
    that read them would pay a case's worth of memory for a number it cannot use, which is exactly
    what the declared route already declines to do (:meth:`Resample._pricing_bound`).

    ``field_dtype`` is what a dense field's values are held in: float64 for the bit-exact walk,
    float32 for a caller whose walk runs in float32 and would quantise them back anyway.
    """
    from konfai.data.geometry import Grid
    from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE, data_to_transform

    read_data = getattr(dataset, "read_data", None)
    if read_data is None:
        return decode_transform_stages(dataset.read_transform(group, name))
    what = f"the displacement field '{group}' of case '{name}'"
    if headers_only:
        # The PLAN's read: a dense field answers as the identity and its values are never touched.
        # Nothing bounds a field from headers, so there is nothing to read them for -- and reading
        # them anyway is what put 29 GiB of one native case beside a budget that never saw it.
        # Everything else is coefficients, small, and bounded exactly, so it decodes as usual.
        shape, header = dataset.get_infos(group, name)
        if DISPLACEMENT_FIELD_ATTRIBUTE in header:
            return ()
        data, attribute = read_data(group, name)
        return decode_transform_stages(data_to_transform(data, attribute, name))
    window = None
    # A backend that cannot serve a slice reads whole, which is what it would do for any window it
    # was given: the transform-only stores (read_transform and nothing else) are already out above.
    if box is not None and getattr(dataset, "read_data_slice", None) is not None:
        shape, header = dataset.get_infos(group, name)
        if DISPLACEMENT_FIELD_ATTRIBUTE in header:
            grid = Grid.of([int(extent) for extent in shape[1:]], header, what)
            window = grid.index_window(box, 1)
    if window is not None:
        data, attribute = dataset.read_data_slice(group, name, (slice(None), *window))
        return (_displacement_stage(grid.sub_grid(window), data, 1, what, field_dtype),)
    data, attribute = read_data(group, name)
    if DISPLACEMENT_FIELD_ATTRIBUTE not in attribute:
        return decode_transform_stages(data_to_transform(data, attribute, name))
    return (
        _displacement_stage(
            Grid.of([int(extent) for extent in np.shape(data)[1:]], attribute, what), data, 1, what, field_dtype
        ),
    )


def encode_transform_stages(stages: SpatialStages) -> sitk.Transform:
    """Geometry stages, in APPLICATION order, as one SimpleITK transform: the decoder's inverse.

    An affine stage becomes an ``AffineTransform`` (matrix and translation, world units); a
    displacement stage of order 1 a ``DisplacementFieldTransform`` on its own grid, of order 3 a
    ``BSplineTransform`` from its coefficient images. ``CompositeTransform`` applies its members
    LAST-ADDED-FIRST, so the stages are added in reverse to run in order (the mirror of what
    :func:`decode_transform_stages` undoes). One stage is returned as itself.
    """
    _require_simpleitk()
    from konfai.data.geometry import AffineStage

    members: list[sitk.Transform] = []
    for stage in stages:
        if isinstance(stage, AffineStage):
            rank = stage.map.rank
            affine = sitk.AffineTransform(rank)
            affine.SetMatrix(np.asarray(stage.map.matrix, dtype=np.float64).ravel().tolist())
            affine.SetTranslation(np.asarray(stage.map.translation, dtype=np.float64).tolist())
            members.append(affine)
            continue
        grid = stage.grid
        rank = grid.rank
        images = []
        for component in range(rank):
            image = sitk.GetImageFromArray(np.ascontiguousarray(stage.values[component], dtype=np.float64))
            image.SetOrigin(np.asarray(grid.origin_xyz, dtype=np.float64).tolist())
            image.SetSpacing(np.asarray(grid.spacing_xyz, dtype=np.float64).tolist())
            image.SetDirection(np.asarray(grid.direction_xyz, dtype=np.float64).ravel().tolist())
            images.append(image)
        if stage.order == 1:
            field = sitk.Compose(images)
            members.append(sitk.DisplacementFieldTransform(field))
        else:
            members.append(sitk.BSplineTransform(images, int(stage.order)))
    if not members:
        raise TransformError(
            "encode_transform_stages() was given no stage to encode.",
            "Hand it the decoded stages of a transform; an identity is sitk.Transform(rank, sitk.sitkIdentity).",
        )
    if len(members) == 1:
        return members[0]
    composite = sitk.CompositeTransform(members[0].GetDimension())
    for member in reversed(members):
        composite.AddTransform(member)
    return composite


def invert_stages(stages: SpatialStages, rank: int) -> SpatialStages | None:
    """The exact inverse of an all-affine decoded map, or ``None`` when one is not algebraic.

    A BSpline or a field inverts by an iterative dense solve, not an algebraic step, and a field
    solved per region is not the restriction of the field solved once, so a non-affine inverse is
    ``None`` here and the caller refuses with the remedy, rather than resampling through a guess.
    """
    from konfai.data.geometry import AffineMap, AffineStage

    if not all(isinstance(stage, AffineStage) for stage in stages):
        return None
    folded = AffineMap.identity(rank)
    for stage in stages:
        folded = folded.then(cast("AffineStage", stage).map)
    return (AffineStage(folded.inverted()),)
