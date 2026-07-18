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


#: Default overlap on tiled axes when a free-axis patch does not say otherwise: 20 % of the patch.
DEFAULT_OVERLAP_FRACTION = 0.2

OverlapSpec = int | float | str | list["int | float | str"] | None


def concretize_patch_size(
    patch_size: list[int] | None,
    shape: list[int] | tuple[int, ...],
    multiple: list[int] | None = None,
) -> list[int]:
    """Resolve the per-axis patch convention onto a concrete shape: ``0`` = free axis -> full extent.

    ``[0,0,0]`` (or ``None``) is the whole volume; ``[1,0,0]`` is a full 2D slice; a positive entry is
    fixed by the user and passes through (clamped to the extent so a patch never exceeds the volume).

    ``multiple`` (the model's per-axis ``downsampling_factor``) rounds a free axis UP to a valid input
    size for the network — 122 -> 128 for a factor 16 — so its encoder/decoder skips align. The rounded
    size may exceed the extent; the border padding (``pad_to_patch``) fills it and the accumulator crops
    it back, exactly as it does for any patch larger than its case.
    """
    if patch_size is None:
        return list(shape)
    if len(shape) != len(patch_size):
        raise DatasetManagerError(
            f"Dimension mismatch: 'patch_size' has {len(patch_size)} dimensions, but 'shape' has {len(shape)}.",
            f"patch_size: {patch_size}",
            f"shape: {shape}",
            "Both must have the same number of dimensions (e.g., 3D patch for 3D volume).",
        )

    def resolve(d: int, p: int) -> int:
        if p != 0:
            return min(int(p), int(shape[d]))
        extent = int(shape[d])
        m = int(multiple[d]) if multiple is not None and d < len(multiple) else 1
        return ((extent + m - 1) // m) * m if m > 1 else extent

    return [resolve(d, p) for d, p in enumerate(patch_size)]


def resolve_overlap(overlap: OverlapSpec, patch_size: list[int], shape: list[int] | tuple[int, ...]) -> list[int]:
    """Resolve an overlap spec into per-axis voxels for a CONCRETE ``patch_size``.

    Accepted forms: ``None`` -> ``DEFAULT_OVERLAP_FRACTION`` of the patch per axis; an ``int`` ->
    absolute voxels on every axis; a ``float`` in [0,1[ or a ``"20%"`` string -> that fraction of the
    patch per axis; a per-axis list mixing those forms. Whatever the spec, an axis that is not tiled
    (single patch spans the extent) resolves to 0 -- overlap only exists between patches.
    """

    def one(spec: int | float | str, size: int) -> int:
        if isinstance(spec, str):
            text = spec.strip()
            if not text.endswith("%"):
                raise ValueError(f"overlap: '{spec}' is not a percentage; use e.g. '20%', a voxel int or a fraction.")
            spec = float(text[:-1]) / 100.0
        if isinstance(spec, float):
            if not 0.0 <= spec < 1.0:
                raise ValueError(f"overlap: fraction {spec} must be in [0, 1[.")
            return int(size * spec)
        if spec < 0:
            raise ValueError(f"overlap: {spec} must be >= 0 voxels.")
        return int(spec)

    if overlap is None:
        overlap = DEFAULT_OVERLAP_FRACTION
    specs: list[int | float | str] = (
        list(overlap) if isinstance(overlap, list) else [overlap] * len(patch_size)  # scalar broadcast
    )
    if len(specs) != len(patch_size):
        raise ValueError(f"overlap: {len(specs)} entries for {len(patch_size)} axes; give one per axis or a scalar.")
    resolved = []
    for spec, size, extent in zip(specs, patch_size, shape, strict=True):
        # No overlap on an axis that is not tiled (one patch spans it) nor on a length-1 patch axis
        # (2D slicing: nothing to blend along a single-voxel patch).
        voxels = one(spec, size) if 1 < size < extent else 0
        if voxels >= size:
            raise ValueError(f"overlap: {voxels} voxels must be smaller than the patch size {size}.")
        resolved.append(voxels)
    return resolved


#: Fraction of the budget the sized patch may use; the rest absorbs intermediates the estimate misses.
PATCH_BUDGET_SAFETY_FRACTION = 0.8


def resolve_patch(
    patch_size: list[int] | None,
    shape: list[int] | tuple[int, ...],
    channels: int,
    dtype_bytes: int,
    budget_bytes: float | None,
    resident_images: int = 1,
    intermediate_factor: float = 1.0,
    snap: list[int] | None = None,
) -> list[int]:
    """Size the free axes of ``patch_size`` so ONE patch fits ``budget_bytes``; fixed axes never move.

    ``0`` entries are free (their max = the volume's extent); positive entries are pinned by the user.
    A patch's footprint is ``(resident_images + intermediate_factor) * voxels * channels * dtype_bytes``
    -- ``resident_images`` is exact (counted from the config: output + targets + masks), the
    ``intermediate_factor`` covers the op's working copies. When everything fits, the free axes take
    their full extent (the whole volume when all axes are free). Otherwise the free axes shrink
    ISOTROPICALLY (the patch keeps the volume's proportions), optionally snapped down to ``snap``
    multiples (a model's valid input sizes). ``budget_bytes=None`` disables sizing (free = extent).
    Fixed axes alone exceeding the budget is an error: the user pinned more than the budget allows.
    """
    concrete = concretize_patch_size(patch_size, shape)
    free = [d for d, p in enumerate(patch_size) if p == 0] if patch_size is not None else list(range(len(concrete)))
    if budget_bytes is None or not free:
        return concrete

    def snapped(axis: int, value: int) -> int:
        if snap is None or snap[axis] <= 1:
            return max(1, value)
        return max(min(snap[axis], int(shape[axis])), (value // snap[axis]) * snap[axis])

    bytes_per_voxel = (resident_images + intermediate_factor) * channels * dtype_bytes
    fixed_voxels = int(np.prod([concrete[d] for d in range(len(concrete)) if d not in free]))
    cap = (budget_bytes * PATCH_BUDGET_SAFETY_FRACTION) / (bytes_per_voxel * fixed_voxels)
    if cap < 1.0:
        raise DatasetManagerError(
            f"The fixed patch axes alone ({fixed_voxels} voxels x {channels} channels) exceed the memory budget.",
            f"budget: {budget_bytes / 2**30:.2f} GiB | bytes/voxel: {bytes_per_voxel:.0f}",
            "Raise the budget, free an axis (0), or pin smaller sizes.",
        )
    free_extents = [int(shape[d]) for d in free]
    if float(np.prod(free_extents)) <= cap:
        return concrete
    ratio = (cap / float(np.prod(free_extents))) ** (1.0 / len(free))
    for d, extent in zip(free, free_extents, strict=True):
        concrete[d] = snapped(d, int(extent * ratio))
    return concrete


def get_patch_slices_from_shape(
    patch_size: list[int], shape: list[int], overlap_tmp: OverlapSpec
) -> tuple[list[tuple[slice, ...]], list[tuple[int, bool]]]:

    has_free_axis = patch_size is not None and any(p == 0 for p in patch_size) and not all(p == 0 for p in patch_size)
    if patch_size is None or all(p == 0 for p in patch_size):
        patch_size = shape
    else:
        patch_size = concretize_patch_size(patch_size, shape)
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
        if has_free_axis:
            # Free axes are new territory (no config predates them), so they take the modern default:
            # DEFAULT_OVERLAP_FRACTION of the patch on tiled axes, 0 on untiled ones.
            overlap = np.array(resolve_overlap(None, patch_size, shape), dtype=np.int_)
        else:
            # Fully-fixed patch with no spec: spread the last patch's remainder evenly across each axis.
            size = [np.ceil(a / b) for a, b in zip(shape, patch_size, strict=False)]
            tmp = np.zeros(len(size), dtype=np.int_)
            for i, s in enumerate(size):
                if s > 1:
                    tmp[i] = np.mod(patch_size[i] - np.mod(shape[i], patch_size[i]), patch_size[i]) // (size[i] - 1)
            overlap = tmp
    elif isinstance(overlap_tmp, int):
        # Plain int: the same voxel overlap on every axis whose patch is > 1.
        overlap = [overlap_tmp if size > 1 else 0 for size in patch_size]
    else:
        # Rich spec: a fraction, a "20%" string, or a per-axis list mixing forms.
        overlap = resolve_overlap(overlap_tmp, patch_size, shape)

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
