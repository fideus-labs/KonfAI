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

"""A 2-D image through a pipeline of 3-D presets: KonfAI's patches and a preset's configuration are 3-D, the engines
are not. The image goes through as a volume one plane thick, ``EngineRegistration`` hands the engine the plane itself,
and the transform, a field or an affine, is brought back to 2-D at the end, for the moved image and every later use."""

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import as_affine, image_to_data
from konfai.utils.dataset.staging import publish
from konfai.utils.errors import KonfAIError
from konfai.utils.ITK import read_displacement_field
from konfai.utils.ome_zarr import (
    NO_UNIT,
    clear_ome_zarr_cache,
    is_affine_store,
    millimetres_per_unit,
    ome_zarr_affine_units,
    read_ome_zarr_affine,
    write_ome_zarr,
    write_ome_zarr_affine,
)
from konfai.utils.utils import path_format_token, storage_form


def is_flat(shape: list[int]) -> bool:
    """Whether a header's shape, channels first, is a 2-D image rather than a volume."""
    return len(shape) == 3


def lift_plane(path: Path, dest: Path, z_mm: float) -> Path:
    """``path``, a 2-D image, as a volume one plane thick in the directory ``dest``, its own, ``z_mm`` thick. An OME-Zarr
    store stays one, written lazily; anything else is read through SimpleITK and written as ``.mha``."""
    dest.mkdir(parents=True, exist_ok=True)
    if path_format_token(path) != "omezarr":
        target = dest / "plane.mha"
        sitk.WriteImage(sitk.JoinSeries([sitk.ReadImage(str(path))], 0.0, z_mm), str(target))
        return target
    import dask.array as da
    import ngff_zarr

    image = ngff_zarr.from_ngff_zarr(str(path)).images[0]
    spatial = [dim for dim in image.dims if dim in ("y", "x")]
    units = image.axes_units or {}
    unit = units.get(spatial[0])
    data = image.data[tuple(slice(None) if dim in spatial else 0 for dim in image.dims)]
    lifted = ngff_zarr.NgffImage(
        data=da.expand_dims(data, 0),
        dims=("z", *spatial),
        scale={"z": z_mm / (millimetres_per_unit(unit) or 1.0), **{dim: image.scale[dim] for dim in spatial}},
        translation={"z": 0.0, **{dim: image.translation[dim] for dim in spatial}},
        axes_units={"z": unit, **{dim: units.get(dim) for dim in spatial}} if units else None,
    )
    target = dest / "plane.ome.zarr"
    ngff_zarr.to_ngff_zarr(str(target), ngff_zarr.to_multiscales(lifted, scale_factors=[]))
    return target


def _read_2d_transform(path: Path) -> sitk.Transform | sitk.Image:
    """What the 2-D transform ``path`` stores: a linear transform, or a field as its vector image."""
    if path.is_dir():
        return read_ome_zarr_affine(path) if is_affine_store(path) else read_displacement_field(path)
    transform = sitk.ReadTransform(str(path))
    if transform.IsLinear():
        return transform
    if transform.GetName() != "DisplacementFieldTransform":
        raise ValueError(
            f"'{path}' is a 2-D {transform.GetName()}: the 2-D pipeline reads displacement fields and linear transforms."
        )
    return sitk.DisplacementFieldTransform(transform).GetDisplacementField()


def lift_field(path: Path, dest: Path) -> Path:
    """The 2-D transform ``path``, a transform file or an OME-Zarr store, as a transform file one plane thick under
    ``dest``: a field one plane thick, or a 3-D affine acting on the plane. A registration's later steps read it as
    they read every transform."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / (f"{path.name.removesuffix(storage_form(path))}.h5" if path.is_dir() else path.name)
    field = _read_2d_transform(path)
    if isinstance(field, sitk.Transform):
        sitk.WriteTransform(_lifted_affine(field), str(target))
        return target
    array = sitk.GetArrayFromImage(field)
    volume = sitk.GetImageFromArray(
        np.concatenate([array, np.zeros_like(array[..., :1])], axis=-1)[None], isVector=True
    )
    direction = np.eye(3)
    direction[:2, :2] = np.reshape(field.GetDirection(), (2, 2))
    volume.SetSpacing((*field.GetSpacing(), min(field.GetSpacing())))
    volume.SetOrigin((*field.GetOrigin(), 0.0))
    volume.SetDirection(direction.ravel().tolist())
    sitk.WriteTransform(sitk.DisplacementFieldTransform(volume), str(target))
    return target


def _plane_of(field: sitk.Image) -> sitk.Image:
    """The 2-D field of a field one plane thick: its plane, without its component along z."""
    plane = field[:, :, 0]
    flat = sitk.GetImageFromArray(sitk.GetArrayFromImage(plane)[..., :2].astype(np.float64), isVector=True)
    flat.CopyInformation(plane)
    return flat


def _plane_affine(transform: sitk.Transform) -> sitk.AffineTransform:
    """The 2-D affine of a one-plane volume's linear transform: its plane, without its row and column along z."""
    affine = as_affine(transform)
    plane = sitk.AffineTransform(2)
    plane.SetMatrix(np.reshape(affine.GetMatrix(), (3, 3))[:2, :2].ravel().tolist())
    plane.SetTranslation(list(affine.GetTranslation()[:2]))
    return plane


def _lifted_affine(transform: sitk.Transform) -> sitk.AffineTransform:
    """A 2-D linear transform as the 3-D affine acting on the plane, the identity along z."""
    affine = as_affine(transform)
    matrix = np.eye(3)
    matrix[:2, :2] = np.reshape(affine.GetMatrix(), (2, 2))
    lifted = sitk.AffineTransform(3)
    lifted.SetMatrix(matrix.ravel().tolist())
    lifted.SetTranslation([*affine.GetTranslation(), 0.0])
    return lifted


def _write_store(path: Path, write: Callable[[Path], None]) -> None:
    """``write`` the store ``path``, staged beside it and published once complete: ``path`` may hold the only copy of
    the plane being rewritten."""
    staging = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        write(staging)
        publish(staging, path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    clear_ome_zarr_cache(path)  # the reader memoises a store by its path


def _project_store(path: Path) -> None:
    """Rewrite the one-plane field store ``path`` as the 2-D field it is, in its dtype and its unit."""
    import ngff_zarr

    source = ngff_zarr.from_ngff_zarr(str(path)).images[0]
    # ponytail: the field is read whole, as a transform file is and as project_image reads the plane; by regions if
    # slides outgrow the RAM
    flat = _plane_of(read_displacement_field(path))
    data, attributes = image_to_data(flat)
    if source.axes_units:
        attributes["OMEUnits"] = " ".join(source.axes_units.get(dim) or NO_UNIT for dim in ("x", "y"))
    _write_store(
        path,
        lambda staging: write_ome_zarr(
            staging,
            data.astype(source.data.dtype),
            spacing=flat.GetSpacing(),
            origin=flat.GetOrigin(),
            attributes=dict(attributes),
            displacement_field=True,
        ),
    )


def _project_affine_store(path: Path) -> None:
    """Rewrite the one-plane affine store ``path`` as the 2-D affine it is, in its units."""
    units = ome_zarr_affine_units(path)
    plane = _plane_affine(read_ome_zarr_affine(path))
    attributes = {"OMEUnits": " ".join(units.get(dim) or NO_UNIT for dim in ("x", "y"))} if units else None
    _write_store(path, lambda staging: write_ome_zarr_affine(staging, plane, attributes))


def project_field(path: Path) -> None:
    """Rewrite the one-plane transform ``path``, a transform file or an OME-Zarr store, as the 2-D transform it is: a
    field's plane, or an affine without its row and column along z."""
    if path.is_dir():
        _project_affine_store(path) if is_affine_store(path) else _project_store(path)
    elif path.suffix.lower() in (".h5", ".tfm", ".txt"):
        transform = sitk.ReadTransform(str(path))
        if transform.IsLinear():
            sitk.WriteTransform(_plane_affine(transform), str(path))
        else:
            field = sitk.DisplacementFieldTransform(transform).GetDisplacementField()
            sitk.WriteTransform(sitk.DisplacementFieldTransform(_plane_of(field)), str(path))
    else:
        raise KonfAIError(
            "ImpactReg",
            f"a 2-D image needs a preset that writes its transform as an ITK transform file or an OME-Zarr store, not "
            f"'{path.name}'.",
            "Declare 'Transform:itktransform' or 'Transform:omezarr' in the preset's output dataset.",
        )


def project_image(path: Path, target: Path) -> None:
    """Rewrite the one-plane image ``path`` as the 2-D image ``target``, in the form of its name."""
    if not path.is_dir():
        sitk.WriteImage(sitk.ReadImage(str(path))[:, :, 0], str(target))
    else:
        import ngff_zarr

        image = ngff_zarr.from_ngff_zarr(str(path)).images[0]
        units = image.axes_units or {}
        per_mm = {dim: millimetres_per_unit(units.get(dim)) or 1.0 for dim in ("y", "x")}
        # ponytail: the plane is read whole, which a 2-D image affords; by blocks if slides outgrow the RAM
        data = np.squeeze(np.asarray(image.data))[None]
        _write_store(
            target,
            lambda staging: write_ome_zarr(
                staging,
                data,
                spacing=[image.scale[dim] * per_mm[dim] for dim in ("x", "y")],
                origin=[image.translation[dim] * per_mm[dim] for dim in ("x", "y")],
                attributes={"OMEUnits": " ".join(units.get(dim) or NO_UNIT for dim in ("x", "y"))} if units else None,
                scale_factors="auto",
            ),
        )
    if target != path:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
