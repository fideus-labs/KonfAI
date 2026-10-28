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

"""A registration as an OME-Zarr 0.6 scene: the fixed and moving images and the transformation between them (NGFF
RFC-5), in one store a viewer or ``ngff_zarr.resample`` reads without a resampled copy.

The transform ``register`` wrote is a dense displacement field in millimetres; the scene holds it in the images' own
unit, as float32. A field that is an affine map (a rigid or affine preset) is stored as the RFC-5 ``affine`` (or the
``translation``) it is, and ``factor`` stores a deformable one that many times coarser: on idr0044's embryo, 2 kept the
registration (NCC 0.703 against 0.706) for an eighth of the bytes, 4 cost 0.02.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from konfai.utils.errors import KonfAIError
from konfai.utils.ome_zarr import _grid_is_axis_aligned, _read_konfai_attributes, millimetres_per_unit

_PATH = "coordinateTransformations/registration"
#: A field is stored as an affine when no voxel departs from the fitted map by more than this share of the finest voxel.
_AFFINE_TOLERANCE = 1e-3


def _require():
    try:
        import itk
        import ngff_zarr
        from ngff_zarr import NgffScene  # noqa: F401
    except ImportError as error:
        raise KonfAIError(
            "ImpactReg",
            f"writing a scene needs ngff-zarr with NgffScene and ITK's Python packages ({error}).",
            "Install them: pip install 'ngff-zarr[itk]' itk-io",
        ) from None
    return itk, ngff_zarr


def _oblique(path: Path):
    return KonfAIError(
        "ImpactReg",
        f"'{path}' is oblique: RFC-5 places an image by a scale and a translation along its own axes, and an anatomical "
        "orientation (RFC-4) only flips or swaps them.",
        "Resample the pair onto an axis-aligned grid before registering it, or keep the result as Transform.h5.",
    )


def _image(ngff_zarr, itk, path: Path, name: str):
    """``path``'s finest level as a 3D NgffImage, and the RFC-4 orientation the scene declares for it on top.

    An image file is read through ITK with its orientation, which a scene's transform is converted through. An
    OME-Zarr store is read lazily (a length-1 channel or time axis dropped) without its orientation, as KonfAI
    registered it: the transform is converted on its grid alone, and the orientation comes back for the scene to
    declare. An oblique grid is refused: converted as the nearest axis-aligned one, a rigid result landed 19 voxels
    off."""
    if path.is_dir():
        if not _grid_is_axis_aligned(_read_konfai_attributes(path)):
            # ponytail: a KonfAI store keeps an oriented grid's Direction in its sidecar; refused, as nothing yet
            # gives the scene's images the RFC-4 orientation it implies
            raise _oblique(path)
        image = ngff_zarr.from_ngff_zarr(str(path)).images[0]
        extra = [dim for dim in image.dims if dim not in ("z", "y", "x")]
        if any(image.data.shape[image.dims.index(dim)] != 1 for dim in extra):
            # ponytail: one channel per scene; a multichannel moving needs its channel axis in the coordinate systems
            raise KonfAIError(
                "ImpactReg",
                f"'{path}' has more than one {'/'.join(extra)}: a scene holds one channel of each image.",
                "Write each channel to a store of its own, or register the one the scene should hold.",
            )
        spatial = [dim for dim in image.dims if dim in ("z", "y", "x")]
        data = image.data[tuple(0 if dim in extra else slice(None) for dim in image.dims)]
        units, orientations = image.axes_units or {}, image.axes_orientations or {}
        return ngff_zarr.NgffImage(
            data=data,
            dims=spatial,
            scale={dim: image.scale[dim] for dim in spatial},
            translation={dim: image.translation[dim] for dim in spatial},
            axes_units={dim: units.get(dim) for dim in spatial} if units else None,
            name=name,
        ), {dim: orientations[dim] for dim in spatial if dim in orientations} or None
    itk_image = itk.imread(str(path))
    if not _signed_permutation(itk.array_from_matrix(itk_image.GetDirection())):
        raise _oblique(path)
    image = ngff_zarr.itk_image_to_ngff_image(itk_image, add_anatomical_orientation=True)
    image.name = name
    return image, None


def _signed_permutation(direction: np.ndarray) -> bool:
    """Whether ``direction`` only flips or swaps axes, as RFC-4 can: one unit entry per row and column."""
    magnitudes = np.abs(direction)
    return bool(
        np.allclose(magnitudes, magnitudes.round(), atol=1e-6)
        and np.allclose(magnitudes.sum(0), 1)
        and np.allclose(magnitudes.sum(1), 1)
    )


def _field(itk, transform: Path, per_mm: np.ndarray):
    """``transform``'s displacement field (ITK convention: fixed -> moving points) in ``per_mm`` units a millimetre
    along x, y and z rather than in mm."""
    field = itk.transformread(str(transform))[0].GetDisplacementField()
    direction = itk.array_from_matrix(field.GetDirection())
    if not _signed_permutation(direction):
        raise _oblique(transform)
    # ponytail: the field is read whole, as register held it (24 bytes a voxel), and converted in place; reading it
    # by blocks needs a chunked field source and a chunked conversion in ngff-zarr
    vectors = itk.array_view_from_image(field)  # z, y, x, (dx, dy, dz)
    vectors *= per_mm
    origin = np.asarray(field.GetOrigin()) * per_mm
    # An index axis runs along the physical axis its direction column selects.
    spacing = np.asarray(field.GetSpacing()) * per_mm[np.abs(direction).argmax(axis=0)]
    return vectors, origin, spacing, direction


def _affine_fit(vectors, origin, spacing, direction) -> np.ndarray | None:
    """The 3 x 4 map ``q = A p + b`` the field is, or ``None`` when one voxel departs from it by more than
    ``_AFFINE_TOLERANCE`` of the finest voxel. Checked a plane at a time: the points of a whole field would cost three
    times the field."""

    def points(z: slice, step: int = 1) -> np.ndarray:
        depth, height, width = vectors.shape[:3]
        grid = np.meshgrid(np.arange(depth)[z], np.arange(height)[::step], np.arange(width)[::step], indexing="ij")
        index = np.stack(grid[::-1], axis=-1)  # x, y, z
        return origin + (index * spacing) @ direction.T

    step = max(1, max(vectors.shape[:3]) // 16)
    sample = slice(None, None, step)
    p = points(sample, step).reshape(-1, 3)
    q = p + vectors[sample, sample, sample].reshape(-1, 3)
    solution, *_ = np.linalg.lstsq(np.hstack([p, np.ones((len(p), 1))]), q, rcond=None)
    tolerance = _AFFINE_TOLERANCE * spacing.min()
    for z in range(vectors.shape[0]):
        plane = points(slice(z, z + 1))
        if np.abs(plane @ solution[:3] + solution[3] - plane - vectors[z : z + 1]).max() > tolerance:
            return None
    return solution.T  # rows: x, y, z; columns: A | b


def _coarser(vectors, origin, spacing, direction, factor: int):
    """The field averaged over blocks of ``factor`` voxels a side (its edge repeated to fill the last ones), on the grid
    of the blocks' centres, plus one node of edge values each side: the blocks' centres alone stop short of the fixed
    image's first and last voxels, where RFC-5 leaves what a reader does to the reader."""
    pad = [(0, -n % factor) for n in vectors.shape[:3]] + [(0, 0)]
    padded = np.pad(vectors, pad, mode="edge")
    z, y, x = (n // factor for n in padded.shape[:3])
    blocks = padded.reshape(z, factor, y, factor, x, factor, 3).mean(axis=(1, 3, 5))
    blocks = np.pad(blocks, [(1, 1)] * 3 + [(0, 0)], mode="edge")
    return blocks, origin + direction @ (spacing * ((factor - 1) / 2 - factor)), spacing * factor


def write_scene(fixed: Path, moving: Path, transform: Path, output: Path, factor: int = 1) -> str:
    """Write the scene of one registration to ``output`` (a directory store, or a single ``.ozx`` file); return the
    RFC-5 type of the transformation it holds. The scene replaces ``output`` only once it is written whole."""
    target = output.resolve()
    for path in (fixed, moving, transform):
        source = path.resolve()
        if target == source or target in source.parents or source in target.parents:
            raise KonfAIError(
                "ImpactReg",
                f"'{output}' overlaps the input '{path}': the scene would replace it, or be written into it.",
                "Name the scene elsewhere: a new directory or a .ozx file (e.g. -o Output/P000/Scene.ozx).",
            )
    empty = output.is_dir() and not any(output.iterdir())
    if output.suffix != ".ozx" and output.exists() and not (empty or _is_scene(output)):
        # Only a scene written before is replaced: '-o Output/P000' must not take the case's results with it.
        raise KonfAIError(
            "ImpactReg",
            f"'{output}' holds something other than an OME-Zarr scene.",
            "Name the scene itself: a new directory or a .ozx file (e.g. -o Output/P000/Scene.ozx).",
        )
    itk, ngff_zarr = _require()
    from ngff_zarr import (
        CoordinateSystemIdentifier,
        NgffScene,
        itk_displacement_field_to_ngff_transform,
        itk_transform_to_ngff_transform,
    )

    fixed_image, fixed_orientations = _image(ngff_zarr, itk, fixed, "fixed")
    moving_image, moving_orientations = _image(ngff_zarr, itk, moving, "moving")
    dims = list(fixed_image.dims)
    # Per axis, as KonfAI read them.
    units = fixed_image.axes_units or {}
    per_mm = np.array([1.0 / (millimetres_per_unit(units.get(dim)) or 1.0) for dim in ("x", "y", "z")])
    vectors, origin, spacing, direction = _field(itk, transform, per_mm)
    affine = _affine_fit(vectors, origin, spacing, direction)
    field_ms = None
    if affine is not None:
        linear = itk.AffineTransform[itk.D, 3].New()
        linear.SetMatrix(itk.matrix_from_array(np.ascontiguousarray(affine[:, :3])))
        linear.SetTranslation([float(v) for v in affine[:, 3]])
        ngff_transform = itk_transform_to_ngff_transform(linear, dims, fixed=fixed_image, moving=moving_image)
    else:
        if factor > 1:
            vectors, origin, spacing = _coarser(vectors, origin, spacing, direction, factor)
        field = itk.image_from_array(np.ascontiguousarray(vectors, dtype=np.float64), is_vector=True)
        field.SetOrigin(origin.tolist())
        field.SetSpacing(spacing.tolist())
        field.SetDirection(itk.matrix_from_array(np.ascontiguousarray(direction)))
        displacement = itk.DisplacementFieldTransform[itk.D, 3].New()
        displacement.SetDisplacementField(field)
        ngff_transform, field_image = itk_displacement_field_to_ngff_transform(
            displacement, dims, path=_PATH, fixed=fixed_image, moving=moving_image
        )
        field_image.data = field_image.data.astype(np.float32)
        field_image.axes_orientations = fixed_orientations or field_image.axes_orientations
        field_ms = ngff_zarr.to_multiscales(field_image, scale_factors=[])
    fixed_image.axes_orientations = fixed_orientations or fixed_image.axes_orientations
    moving_image.axes_orientations = moving_orientations or moving_image.axes_orientations
    # ITK's Gaussian, not ngff-zarr's default itkwasm one: dask runs several itkwasm pipelines at once in its threads,
    # and wasmtime then dies of a bus error setting up their environment (ngff-zarr 7019b0b, itkwasm 1.0b201).
    fixed_ms = ngff_zarr.to_multiscales(fixed_image, method=ngff_zarr.Methods.ITK_GAUSSIAN)
    moving_ms = ngff_zarr.to_multiscales(moving_image, method=ngff_zarr.Methods.ITK_GAUSSIAN)
    ngff_transform.input = CoordinateSystemIdentifier(
        path="fixed", name=fixed_ms.metadata.intrinsic_coordinate_system.name
    )
    ngff_transform.output = CoordinateSystemIdentifier(
        path="moving", name=moving_ms.metadata.intrinsic_coordinate_system.name
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".scene_", dir=output.parent))
    store = staging / "scene.zarr"
    try:
        if field_ms is not None:
            ngff_zarr.to_ome_zarr(str(store / _PATH), field_ms, version="0.6")
        NgffScene(
            images={"fixed": fixed_ms, "moving": moving_ms}, coordinateTransformations=[ngff_transform]
        ).to_ome_zarr(str(store), overwrite=False)
        if output.suffix == ".ozx":
            ngff_zarr.write_store_to_zip(str(store), str(staging / "scene.ozx"), version="0.6")
            os.replace(staging / "scene.ozx", output)
        else:
            # The scene before is moved aside, not deleted, until the new one has taken its place.
            previous = staging / "previous"
            if output.exists():
                os.replace(output, previous)
            try:
                os.replace(store, output)
            except OSError as error:
                try:
                    if previous.exists():
                        os.replace(previous, output)
                except OSError:
                    raise KonfAIError(
                        "ImpactReg",
                        f"'{output}' could not be replaced ({error}), nor the scene before put back.",
                        f"The scene before is kept in '{previous}': move it back to '{output}'.",
                    ) from error
                raise
    finally:
        if output.exists() or not (staging / "previous").exists():  # the only copy of the scene before stays
            shutil.rmtree(staging, ignore_errors=True)
    return ngff_transform.type


def _is_scene(store: Path) -> bool:
    """Whether ``store`` is an OME-Zarr scene: its root names one (RFC-8)."""
    try:
        attributes = json.loads((store / "zarr.json").read_text(encoding="utf-8")).get("attributes") or {}
    except (OSError, ValueError):
        return False
    return isinstance(attributes.get("ome"), dict) and "scene" in attributes["ome"]
