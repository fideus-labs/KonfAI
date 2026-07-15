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

"""Compatibility facade for KonfAI utility helpers and lightweight array utilities."""

import importlib
import itertools
import os
import re
from types import ModuleType

import numpy as np

from konfai.utils.errors import DatasetManagerError


def get_module(classpath: str, default_classpath: str) -> tuple[ModuleType, str]:
    """Import the module a classpath names and return it with the name to take from it.

    A ``:`` separates the module from the name: everything before the last one is the module, so
    ``torch:nn:L1Loss`` and ``torch.nn:L1Loss`` name the same class. Without one, the name is taken
    from the kind's own package, and the dots between them lead there: ``Dice`` is that package's
    own, ``segmentation.UNet.UNet`` is under two of its subpackages.
    """
    if len(classpath.split(":")) > 1:
        module_name = ".".join(classpath.split(":")[:-1])
        name = classpath.split(":")[-1]
    else:
        *submodules, name = classpath.split(".")
        module_name = ".".join([default_classpath, *submodules])
    previous_mode = os.environ.get("KONFAI_CONFIG_MODE")
    os.environ["KONFAI_CONFIG_MODE"] = "Import"
    try:
        module = importlib.import_module(module_name)
    finally:
        if previous_mode is None:
            os.environ.pop("KONFAI_CONFIG_MODE", None)
        else:
            os.environ["KONFAI_CONFIG_MODE"] = previous_mode
    return module, name.split("/")[0]


def get_patch_slices_from_nb_patch_per_dim(
    patch_size_tmp: list[int],
    nb_patch_per_dim: list[tuple[int, bool]],
    overlap: int | None,
) -> list[tuple[slice, ...]]:
    patch_slices = []
    slices: list[list[slice]] = []
    if overlap is None:
        overlap = 0
    patch_size = []
    i = 0
    for nb in nb_patch_per_dim:
        if nb[1]:
            patch_size.append(1)
        else:
            patch_size.append(patch_size_tmp[i])
            i += 1

    for dim, nb in enumerate(nb_patch_per_dim):
        slices.append([])
        for index in range(nb[0]):
            start = (patch_size[dim] - overlap) * index
            end = start + patch_size[dim]
            slices[dim].append(slice(start, end))
    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    return patch_slices


def get_patch_slices_from_shape(
    patch_size: list[int], shape: list[int], overlap_tmp: int | None
) -> tuple[list[tuple[slice, ...]], list[tuple[int, bool]]]:

    if patch_size is None or all(p == 0 for p in patch_size):
        patch_size = shape
    if len(shape) != len(patch_size):
        raise DatasetManagerError(
            f"Dimension mismatch: 'patch_size' has {len(patch_size)} dimensions, but 'shape' has {len(shape)}.",
            f"patch_size: {patch_size}",
            f"shape: {shape}",
            "Both must have the same number of dimensions (e.g., 3D patch for 3D volume).",
        )
    patch_slices = []
    nb_patch_per_dim = []
    slices: list[list[slice]] = []
    if overlap_tmp is None:
        size = [np.ceil(a / b) for a, b in zip(shape, patch_size, strict=False)]
        tmp = np.zeros(len(size), dtype=np.int_)
        for i, s in enumerate(size):
            if s > 1:
                tmp[i] = np.mod(patch_size[i] - np.mod(shape[i], patch_size[i]), patch_size[i]) // (size[i] - 1)
        overlap = tmp
    else:
        overlap = [overlap_tmp if size > 1 else 0 for size in patch_size]

    for dim in range(len(shape)):
        if overlap[dim] >= patch_size[dim]:
            raise ValueError(
                f"Overlap must be less than patch size, got overlap={overlap[dim]}",
                f" ≥ patch_size={patch_size[dim]} at dim={dim}",
            )

    for dim in range(len(shape)):
        slices.append([])
        index = 0
        while True:
            start = (patch_size[dim] - overlap[dim]) * index

            end = start + patch_size[dim]
            if end >= shape[dim]:
                end = shape[dim]
                slices[dim].append(slice(start, end))
                break
            slices[dim].append(slice(start, end))
            index += 1
        nb_patch_per_dim.append((index + 1, patch_size[dim] == 1))

    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))

    return patch_slices, nb_patch_per_dim


SUPPORTED_EXTENSIONS = [
    "mha",
    "mhd",  # MetaImage
    "nii",
    "nii.gz",  # NIfTI
    "nrrd",
    "nrrd.gz",  # NRRD
    "gipl",
    "gipl.gz",  # GIPL
    "hdr",
    "img",  # Analyze
    "dcm",  # DICOM (if GDCM enabled)
    "dicom",  # DICOM series directory backend
    "omezarr",
    "ome-zarr",
    "ome_zarr",
    "zarr",  # OME-NGFF directory backend and accepted aliases
    "tif",
    "tiff",  # TIFF
    "png",
    "jpg",
    "jpeg",
    "bmp",  # 2D formats
    "h5",
    "itk.txt",
    "fcsv",
    "xml",
    "vtk",
    "npy",
]


_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def is_windows_absolute_path(path: str) -> bool:
    """Return whether *path* looks like a Windows absolute path."""
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(path))


def split_format_level(file_format: str) -> tuple[str, int]:
    """Split an optional pyramid-level suffix from a format token.

    Used by the OME-Zarr backend to pick a multiscale resolution directly in
    the dataset spec, e.g. ``omezarr@2`` selects pyramid level 2 (coarser),
    independently of any transform. Returns ``(base_format, level)`` and
    defaults to level 0 (full resolution) when no ``@<int>`` suffix is present.
    """
    base, separator, level = file_format.rpartition("@")
    if separator and level.isdigit():
        return base, int(level)
    return file_format, 0


def split_path_spec(
    value: str,
    *,
    default_format: str = "mha",
    allowed_flags: set[str] | None = None,
    supported_extensions: list[str] | None = None,
) -> tuple[str, str | None, str]:
    """Split a KonfAI ``path[:flag]:format`` spec without breaking Windows paths.

    KonfAI accepts dataset-like strings such as:

    - ``./Dataset``
    - ``./Dataset:mha``
    - ``./Dataset:a:mha``
    - ``C:\\Data\\Dataset:mha``
    - ``C:\\Data\\Dataset:a:mha``

    Parsing is performed from the right so the drive separator in Windows paths
    is preserved.
    """

    extensions = SUPPORTED_EXTENSIONS if supported_extensions is None else supported_extensions
    parts = value.rsplit(":", 2)

    if len(parts) == 1:
        return value, None, default_format

    if len(parts) == 2:
        path, maybe_format = parts
        if maybe_format in extensions:
            return path, None, maybe_format
        if is_windows_absolute_path(value):
            return value, None, default_format
        return path, None, maybe_format

    path, middle, file_format = parts
    if file_format in extensions:
        if allowed_flags is not None and middle in allowed_flags:
            return path, middle, file_format
        return f"{path}:{middle}", None, file_format

    return path, middle, file_format
