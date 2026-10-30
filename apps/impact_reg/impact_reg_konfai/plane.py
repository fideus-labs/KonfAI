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
and the field is brought back to 2-D at the end, for the moved image and every later use of the transform."""

import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.errors import KonfAIError
from konfai.utils.ome_zarr import NO_UNIT, millimetres_per_unit, write_ome_zarr
from konfai.utils.utils import path_format_token


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


def lift_field(path: Path, dest: Path) -> Path:
    """The 2-D displacement field of the transform file ``path`` as a field one plane thick under ``dest``: a
    registration's later steps read it as they read every field."""
    transform = sitk.ReadTransform(str(path))
    if transform.GetName() != "DisplacementFieldTransform":
        raise ValueError(f"'{path}' is a 2-D {transform.GetName()}: the 2-D pipeline reads displacement fields.")
    field = sitk.DisplacementFieldTransform(transform).GetDisplacementField()
    array = sitk.GetArrayFromImage(field)
    volume = sitk.GetImageFromArray(
        np.concatenate([array, np.zeros_like(array[..., :1])], axis=-1)[None], isVector=True
    )
    direction = np.eye(3)
    direction[:2, :2] = np.reshape(field.GetDirection(), (2, 2))
    volume.SetSpacing((*field.GetSpacing(), min(field.GetSpacing())))
    volume.SetOrigin((*field.GetOrigin(), 0.0))
    volume.SetDirection(direction.ravel().tolist())
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / path.name
    sitk.WriteTransform(sitk.DisplacementFieldTransform(volume), str(target))
    return target


def project_field(path: Path) -> None:
    """Rewrite the one-plane field of the transform file ``path`` as the 2-D field it is."""
    if path.suffix.lower() not in (".h5", ".tfm", ".txt"):
        raise KonfAIError(
            "ImpactReg",
            f"a 2-D image needs a preset that writes its field as an ITK transform file, not '{path.name}'.",
            "Every published preset does: name one of them.",
        )
    field = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(path))).GetDisplacementField()
    plane = field[:, :, 0]
    flat = sitk.GetImageFromArray(sitk.GetArrayFromImage(plane)[..., :2].astype(np.float64), isVector=True)
    flat.CopyInformation(plane)
    sitk.WriteTransform(sitk.DisplacementFieldTransform(flat), str(path))


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
        write_ome_zarr(
            target,
            data,
            spacing=[image.scale[dim] * per_mm[dim] for dim in ("x", "y")],
            origin=[image.translation[dim] * per_mm[dim] for dim in ("x", "y")],
            attributes={"OMEUnits": " ".join(units.get(dim) or NO_UNIT for dim in ("x", "y"))} if units else None,
            scale_factors="auto",
        )
    if target != path:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
