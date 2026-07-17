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

"""OME-Zarr (OME-NGFF) read/write backend for KonfAI, built on ``ngff-zarr``.

This module is a thin adapter: ``ngff-zarr`` owns all OME-NGFF metadata parsing,
multiscale handling, and (de)serialisation — KonfAI does not re-implement the
spec. We only

1. map between KonfAI's channel-first ``C[Z]YX`` arrays / ``(x, y, z)`` geometry
   and ngff-zarr's ``NgffImage`` (axis-named ``scale``/``translation``), and
2. round-trip KonfAI's full ``Attribute`` sidecar (including the ``Direction``
   matrix, which OME-NGFF cannot express) through a single ``konfai`` group
   attribute, read/written with ``zarr``.

Reads are lazy: ``ngff-zarr`` exposes the array as a chunked store, so slicing
only materialises the requested patch.

Optional dependencies: ``zarr`` + ``ngff-zarr`` (``pip install konfai[omezarr]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from konfai.utils.errors import DatasetManagerError

try:
    import zarr

    _ZARR_AVAILABLE = True
except ImportError:
    zarr = None  # type: ignore[assignment]
    _ZARR_AVAILABLE = False

try:
    import ngff_zarr  # type: ignore[import-untyped]

    _NGFF_ZARR_AVAILABLE = True
except ImportError:
    ngff_zarr = None  # type: ignore[assignment]
    _NGFF_ZARR_AVAILABLE = False

_KONFAI_ATTR_KEY = "konfai"
_SPATIAL = ("z", "y", "x")


def _require_zarr() -> None:
    if not _ZARR_AVAILABLE:
        raise DatasetManagerError(
            "zarr is required for OME-Zarr support.",
            "Install it with: pip install konfai[omezarr]",
        )


def _require_ngff_zarr() -> None:
    _require_zarr()
    if not _NGFF_ZARR_AVAILABLE:
        raise DatasetManagerError(
            "ngff-zarr is required for OME-Zarr support.",
            "Install it with: pip install konfai[omezarr]",
        )


def _read_konfai_attributes(store_path: str | Path) -> dict[str, Any]:
    """Read KonfAI's proprietary ``Attribute`` sidecar from the store, if present."""
    try:
        group = zarr.open_group(str(store_path), mode="r")
        return dict(dict(group.attrs).get(_KONFAI_ATTR_KEY, {}).get("attributes", {}))
    except (KeyError, OSError, ValueError, TypeError):
        return {}


def _load_image(store_path: str | Path, level: int) -> Any:
    """Return the ``NgffImage`` for ``level`` of an OME-Zarr store.

    ``@N`` selects among the levels a store offers, so a single-level store has nothing to select: its
    one level is read whatever ``N`` says (as every other backend does -- ``SitkFile`` ignores
    ``self.level`` too). Out of range on a store that IS a pyramid stays an error: asking level 3 of a
    three-level mask beside a four-level image is a real mismatch (it silently pairs 160 µm against
    320 µm), and quietly falling back to level 0 would hide it.
    """
    _require_ngff_zarr()
    try:
        multiscales = ngff_zarr.from_ngff_zarr(str(store_path))
        if len(multiscales.images) == 1:
            return multiscales.images[0]
        return multiscales.images[level]
    except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
        raise DatasetManagerError(
            f"Cannot open OME-Zarr store '{store_path}' (level {level}).",
            "Ensure the directory is a valid OME-NGFF store.",
        ) from exc


def _canonical_shape(dims: Sequence[str], shape: Sequence[int]) -> list[int]:
    """Channel-first ``[C, (Z), Y, X]`` shape derived from ngff dims."""
    axis_size = dict(zip(dims, shape, strict=True))
    return [int(axis_size.get("c", 1)), *[int(axis_size[axis]) for axis in _SPATIAL if axis in axis_size]]


def _ordered(values: dict[str, float], dims: Sequence[str]) -> list[float]:
    return [float(values.get(axis, 1.0 if axis == "c" else 0.0)) for axis in dims]


def read_ome_zarr_data_slice(
    store_path: str | Path,
    slices: tuple[slice, ...],
    *,
    level: int = 0,
    timepoint: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a KonfAI channel-first ``C[Z]YX`` patch from an OME-Zarr store (lazy)."""
    image = _load_image(store_path, level)
    dims = [str(axis).lower() for axis in image.dims]
    canonical_shape = _canonical_shape(dims, image.data.shape)
    if len(slices) != len(canonical_shape):
        raise DatasetManagerError(f"Expected {len(canonical_shape)} slices, got {len(slices)}.")

    normalized = [slice(*item.indices(size)) for item, size in zip(slices, canonical_shape, strict=True)]
    spatial_slices = dict(zip([axis for axis in _SPATIAL if axis in dims], normalized[1:], strict=True))
    index: list[int | slice] = []
    for axis in dims:
        if axis == "t":
            index.append(timepoint)
        elif axis == "c":
            index.append(normalized[0])
        elif axis in spatial_slices:
            index.append(spatial_slices[axis])
        else:
            index.append(slice(None))

    patch = np.asarray(image.data[tuple(index)])
    remaining = [axis for axis, selection in zip(dims, index, strict=True) if not isinstance(selection, int)]
    wanted = [axis for axis in ("c", *_SPATIAL) if axis in remaining]
    patch = np.transpose(patch, [remaining.index(axis) for axis in wanted])
    if "c" not in remaining:
        patch = patch[np.newaxis]

    metadata = {
        "axes": dims,
        "shape": canonical_shape,
        "chunks": list(getattr(image.data, "chunks", []) or []),
        "dtype": str(image.data.dtype),
        "scale": _ordered(dict(image.scale), dims),
        "translation": _ordered(dict(image.translation), dims),
        "attributes": _read_konfai_attributes(store_path),
    }
    return np.asarray(patch), metadata


def write_ome_zarr(
    store_path: str | Path,
    data: np.ndarray,
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: Sequence[int] | None = None,
) -> None:
    """Write one channel-first KonfAI array as a single-level OME-NGFF store."""
    _require_ngff_zarr()
    array_data = np.asarray(data)
    if array_data.ndim not in {3, 4}:
        raise DatasetManagerError(f"OME-Zarr writing expects a C-Y-X or C-Z-Y-X array, got shape {array_data.shape}.")

    spatial_axes = ["y", "x"] if array_data.ndim == 3 else ["z", "y", "x"]
    dims = ["c", *spatial_axes]
    dimension = len(spatial_axes)
    spacing_xyz = list(spacing if spacing is not None else [1.0] * dimension)
    origin_xyz = list(origin if origin is not None else [0.0] * dimension)
    if len(spacing_xyz) != dimension or len(origin_xyz) != dimension:
        raise DatasetManagerError(
            f"OME-Zarr geometry must contain {dimension} spacing and origin values for shape {array_data.shape}."
        )

    coordinate = {"x": (spacing_xyz[0], origin_xyz[0]), "y": (spacing_xyz[1], origin_xyz[1])}
    if dimension == 3:
        coordinate["z"] = (spacing_xyz[2], origin_xyz[2])
    scale = {"c": 1.0, **{axis: float(coordinate[axis][0]) for axis in spatial_axes}}
    translation = {"c": 0.0, **{axis: float(coordinate[axis][1]) for axis in spatial_axes}}

    image = ngff_zarr.to_ngff_image(array_data, dims=dims, scale=scale, translation=translation)
    multiscales = ngff_zarr.to_multiscales(
        image, scale_factors=[], chunks=tuple(chunks) if chunks is not None else None
    )
    # version 0.4 (zarr v2 layout) stays compatible with zarr-python 2.x; v0.5 needs zarr>=3
    # (unavailable on Python 3.10), so pin 0.4 for portability across the CI matrix.
    ngff_zarr.to_ngff_zarr(str(store_path), multiscales, overwrite=True, version="0.4")

    if attributes:
        group = zarr.open_group(str(store_path), mode="r+")
        group.attrs[_KONFAI_ATTR_KEY] = {"attributes": dict(attributes)}


def create_ome_zarr_store(
    store_path: str | Path,
    shape: Sequence[int],
    dtype: Any,
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: Sequence[int] | None = None,
) -> Any:
    """Create an empty single-level OME-NGFF store for region-by-region writes.

    Returns the level-0 zarr array: chunks materialise as regions are assigned, and unwritten regions
    read back as zeros. Metadata (the 0.4 multiscales entry plus the KonfAI attribute sidecar) is
    complete from the start, so the store is readable at any point during the write.
    """
    _require_zarr()
    if len(shape) not in {3, 4}:
        raise DatasetManagerError(f"OME-Zarr writing expects a C-Y-X or C-Z-Y-X shape, got {list(shape)}.")

    spatial_axes = ["y", "x"] if len(shape) == 3 else ["z", "y", "x"]
    dimension = len(spatial_axes)
    spacing_xyz = list(spacing if spacing is not None else [1.0] * dimension)
    origin_xyz = list(origin if origin is not None else [0.0] * dimension)
    if len(spacing_xyz) != dimension or len(origin_xyz) != dimension:
        raise DatasetManagerError(
            f"OME-Zarr geometry must contain {dimension} spacing and origin values for shape {list(shape)}."
        )

    coordinate = {"x": (spacing_xyz[0], origin_xyz[0]), "y": (spacing_xyz[1], origin_xyz[1])}
    if dimension == 3:
        coordinate["z"] = (spacing_xyz[2], origin_xyz[2])
    scale = [1.0, *[float(coordinate[axis][0]) for axis in spatial_axes]]
    translation = [0.0, *[float(coordinate[axis][1]) for axis in spatial_axes]]

    if chunks is None:
        spatial_chunks = [min(extent, 128) for extent in shape[1:]]
        # Keep one chunk around 32 MiB: full 128-wide spatial tiles, channels split to fit the budget.
        tile_bytes = int(np.prod(spatial_chunks, dtype=np.int64)) * np.dtype(dtype).itemsize
        chunks = [min(shape[0], max(1, (32 << 20) // max(1, tile_bytes))), *spatial_chunks]

    try:
        group = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    except TypeError:  # zarr-python 2.x: the v2 layout is its only format
        group = zarr.open_group(str(store_path), mode="w")
    create_array = getattr(group, "create_array", None) or group.create_dataset
    array = create_array("0", shape=tuple(shape), chunks=tuple(chunks), dtype=np.dtype(dtype), fill_value=0)
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "image",
            "axes": [{"name": "c", "type": "channel"}, *[{"name": axis, "type": "space"} for axis in spatial_axes]],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {"type": "translation", "translation": translation},
                    ],
                }
            ],
        }
    ]
    if attributes:
        group.attrs[_KONFAI_ATTR_KEY] = {"attributes": dict(attributes)}
    return array


def get_ome_zarr_info(store_path: str | Path, level: int = 0) -> dict[str, Any]:
    """Return OME-Zarr metadata (raw axis-order shape) without reading pixel data."""
    image = _load_image(store_path, level)
    dims = [str(axis).lower() for axis in image.dims]
    try:
        n_levels = len(ngff_zarr.from_ngff_zarr(str(store_path)).images)
    except (OSError, TypeError, ValueError):
        n_levels = 1
    return {
        "axes": dims,
        "shape": list(image.data.shape),
        "chunks": list(getattr(image.data, "chunks", []) or []),
        "dtype": str(image.data.dtype),
        "scale": _ordered(dict(image.scale), dims),
        "translation": _ordered(dict(image.translation), dims),
        "n_levels": n_levels,
        "attributes": _read_konfai_attributes(store_path),
    }
