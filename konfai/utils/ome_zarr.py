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

import shutil
from collections.abc import Sequence
from functools import lru_cache
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
    # dask sits under the same guard because it is ngff-zarr's own hard dependency: the two are
    # present or absent together, and dask.array is used only to describe a store to ngff-zarr.
    import dask.array
    import ngff_zarr  # type: ignore[import-untyped]

    _NGFF_ZARR_AVAILABLE = True
except ImportError:
    dask = None  # type: ignore[assignment]
    ngff_zarr = None  # type: ignore[assignment]
    _NGFF_ZARR_AVAILABLE = False

_KONFAI_ATTR_KEY = "konfai"
_SPATIAL = ("z", "y", "x")


def _native_byteorder(array: np.ndarray) -> np.ndarray:
    """The same samples in the machine's own byte order.

    A store may hold big-endian samples -- some acquisition software writes them, and zarr keeps the
    dtype it was given -- and a non-native array poisons everything downstream in two different ways.
    ``torch.from_numpy`` refuses it outright ("given numpy array has byte order different from the
    native byte order"), which is the loud half. The quiet half is numpy: the flag rides along
    through slicing and arithmetic, so a value read here compares and writes correctly while any
    consumer that reinterprets the buffer -- a raw ``.tobytes()``, a memory-mapped write, a C
    extension taking a pointer -- sees the bytes swapped. Normalising once at the read boundary is
    what every caller would otherwise have to remember to do by hand.
    """
    return array.astype(array.dtype.newbyteorder("="), copy=False) if array.dtype.byteorder == ">" else array


# NGFF RFC-5 types the component axis of a vector field, so a displacement field says what it is on
# disk. Those types only exist from NGFF 0.6 (zarr v3); 0.4 (zarr v2 layout) stays the default
# everywhere else, being the version portable across the whole CI matrix.
_DISPLACEMENT_AXIS_TYPE = "displacement"

#: Largest absolute displacement per component, world units, recorded by the producer of a field.
#: A consumer sizes the region it must read from this, and reading it costs a header where measuring
#: it costs a scan of the whole field.
DISPLACEMENT_BOUND_ATTRIBUTE = "MaxDisplacement"
_RFC5_VERSION = "0.6"
_DEFAULT_VERSION = "0.4"

#: How a chunk is sized, whether the store is created from a shape alone or from the region shape a
#: streamed writer declares (:func:`konfai.utils.dataset._store_chunks`). One rule, two callers: a
#: chunk is the unit a reader decompresses to reach one voxel, so an oversized one is paid by every
#: partial read forever, and by any consumer that is not KonfAI.
CHUNK_SPATIAL_TILE = 128
CHUNK_TARGET_BYTES = 32 << 20


def _zarr_v3_available() -> bool:
    """Whether the installed zarr can write a v3 store, which NGFF >= 0.5 (RFC-5) requires.

    RFC-5 axis types live only in NGFF 0.6, a zarr v3 layout, and zarr-python 3 needs Python >= 3.11 --
    so on Python 3.10 (zarr 2.x) a displacement field cannot be written. This is the capability the
    RFC-5 write actually depends on: ``coordinateSystems`` (checked in ``_type_component_axis``) tracks
    the ngff-zarr version, not the zarr one, so it alone lets a 2.x store through to an opaque failure.
    """
    if not _ZARR_AVAILABLE:
        return False
    try:
        return int(zarr.__version__.split(".")[0]) >= 3
    except (AttributeError, ValueError):
        return False


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


def _require_zarr_v3_for_rfc5() -> None:
    """Both write paths type a component axis, so both need the same capability check.

    Raised rather than silently downgraded to an untyped 0.4 store: a caller asking for a displacement
    field is asking for the one property that makes it readable as a transform, and a store that
    quietly is not one gets found out much later, by a reader that took its three channels for an
    image.
    """
    if not _zarr_v3_available():
        raise DatasetManagerError(
            "Writing an NGFF RFC-5 displacement field needs a zarr v3 store, i.e. "
            "zarr-python >= 3 (Python >= 3.11); this environment has zarr 2.",
            "Install it with: pip install 'zarr>=3' on Python >= 3.11.",
        )


def _read_konfai_attributes(store_path: str | Path) -> dict[str, Any]:
    """Read KonfAI's proprietary ``Attribute`` sidecar from the store, if present."""
    try:
        group = zarr.open_group(str(store_path), mode="r")
        return dict(dict(group.attrs).get(_KONFAI_ATTR_KEY, {}).get("attributes", {}))
    except (KeyError, OSError, ValueError, TypeError):
        return {}


@lru_cache(maxsize=8)
def _load_image(store_path: str, level: int) -> Any:
    """Return the ``NgffImage`` for ``level`` of an OME-Zarr store, memoised per (store, level).

    A streamed run reads one patch per call, and re-parsing the NGFF metadata and rebuilding the
    lazy array graph per patch is pure per-read overhead: the image object is lazy (no voxel data),
    so a handful of them is cheap to keep. The key is the path alone, so anything that puts a
    different store at a path already read must call ``clear_ome_zarr_cache()`` — see there.

    ``@N`` selects among the levels a store offers, so a single-level store has nothing to select: its
    one level is read whatever ``N`` says (as every other backend does -- ``SitkFile`` ignores
    ``self.level`` too). Out of range on a store that IS a pyramid stays an error: asking level 3 of a
    three-level mask beside a four-level image is a real mismatch (it silently pairs 160 µm against
    320 µm), and quietly falling back to level 0 would hide it.
    """
    _require_ngff_zarr()
    try:
        multiscales = ngff_zarr.from_ngff_zarr(str(store_path))
    except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
        raise DatasetManagerError(
            f"Cannot open OME-Zarr store '{store_path}' (level {level}).",
            "Ensure the directory is a valid OME-NGFF store.",
        ) from exc
    if len(multiscales.images) == 1:
        return multiscales.images[0]
    if not 0 <= level < len(multiscales.images):
        # Its own message: the store is fine, the LEVEL is not. Reporting an out-of-range level as
        # "not a valid OME-NGFF store" sends the reader to inspect a store that has nothing wrong
        # with it -- the mismatch is in what was asked of it.
        raise DatasetManagerError(
            f"OME-Zarr store '{store_path}' has {len(multiscales.images)} level(s); level {level} is out of range.",
            f"Ask for a level in 0..{len(multiscales.images) - 1} (0 is the finest).",
        )
    return multiscales.images[level]


def clear_ome_zarr_cache() -> None:
    """Forget the memoised NGFF images, so a store replaced on disk is parsed afresh.

    The write paths here call it for their own output. Anything that materialises a store by other
    means -- copying one over another, say -- has to call it too: the memo is keyed on the path, and
    a hit serves the previous store's axes and geometry against the new store's voxels. That reads as
    a shape mismatch when the two differ, and as nothing at all when they do not.
    """
    _load_image.cache_clear()


def is_displacement_field(store_path: str | Path) -> bool:
    """Whether the store declares its component axis as an NGFF RFC-5 displacement field.

    This is what lets a DVF be read back as a transform rather than as a 3-channel image, and it is
    read from the store itself: the producer does not have to be trusted, and no sidecar convention
    (a filename, an attribute) has to be agreed on separately.

    A store that predates RFC-5, or an ngff-zarr too old to model it, simply answers False -- an
    unreadable or absent store is not a displacement field either, so this never raises.
    """
    if not _NGFF_ZARR_AVAILABLE:
        return False
    try:
        metadata = ngff_zarr.from_ngff_zarr(str(store_path)).metadata
    except Exception:
        # "Not a displacement field" is the only answer this owes: it is asked purely to decide HOW to
        # read an entry, and an absent or unreadable store is not one either.
        return False
    return any(
        axis.name == "c" and axis.type == _DISPLACEMENT_AXIS_TYPE
        for system in getattr(metadata, "coordinateSystems", None) or []
        for axis in system.axes
    )


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
    image = _load_image(str(store_path), level)
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
    return _native_byteorder(np.asarray(patch)), metadata


def _spatial_geometry(
    ndim: int,
    shape_label: str,
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
) -> tuple[list[str], list[float], list[float]]:
    """The NGFF spatial axes of a channel-first array with ``ndim`` dims, and its per-axis
    scale/translation in axis order — geometry arrives ``(x, y, z)`` (SimpleITK order)."""
    if ndim not in {3, 4}:
        raise DatasetManagerError(f"OME-Zarr writing expects a C-Y-X or C-Z-Y-X array, got {shape_label}.")
    spatial_axes = ["y", "x"] if ndim == 3 else ["z", "y", "x"]
    dimension = len(spatial_axes)
    spacing_xyz = list(spacing if spacing is not None else [1.0] * dimension)
    origin_xyz = list(origin if origin is not None else [0.0] * dimension)
    if len(spacing_xyz) != dimension or len(origin_xyz) != dimension:
        raise DatasetManagerError(
            f"OME-Zarr geometry must contain {dimension} spacing and origin values for {shape_label}."
        )
    coordinate = {"x": (spacing_xyz[0], origin_xyz[0]), "y": (spacing_xyz[1], origin_xyz[1])}
    if dimension == 3:
        coordinate["z"] = (spacing_xyz[2], origin_xyz[2])
    scale_values = [float(coordinate[axis][0]) for axis in spatial_axes]
    translation_values = [float(coordinate[axis][1]) for axis in spatial_axes]
    return spatial_axes, scale_values, translation_values


def _downsample_method(downsample_method: str | None) -> Any:
    """Resolve a downsampling method name to ngff-zarr's enum, defaulting to BIN_SHRINK.

    NOT ngff-zarr's own default, which is ``ITKWASM_GAUSSIAN``: a pyramid is indexed by position and
    read as "the same image, coarser", so a level that has been smoothed is a change of pixels that
    no reader can see. Measured on a real volume, the gaussian keeps a 0.9998 correlation while
    crushing the peak intensity by 20 % -- the shape of difference that passes a sanity check and
    resurfaces months later. ``ITKWASM_BIN_SHRINK`` is a plain block mean, so a caller that already
    downsamples by averaging blocks gets the same voxels from this writer.
    """
    _require_ngff_zarr()
    if downsample_method is None:
        return ngff_zarr.Methods.ITKWASM_BIN_SHRINK
    try:
        return ngff_zarr.Methods[downsample_method]
    except KeyError:
        raise DatasetManagerError(
            f"Unknown downsample_method '{downsample_method}'.",
            f"Use one of: {', '.join(sorted(m.name for m in ngff_zarr.Methods))}.",
        ) from None


def write_ome_zarr(
    store_path: str | Path,
    data: np.ndarray,
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: Sequence[int] | None = None,
    displacement_field: bool = False,
    scale_factors: Sequence[int] | None = None,
    downsample_method: str | None = None,
) -> None:
    """Write one channel-first KonfAI array as an OME-NGFF store, single-level or a pyramid.

    ``scale_factors`` makes it a pyramid: ``[4]`` writes level 0 plus level 0 shrunk 4x per spatial
    axis, ``[4, 4]`` adds a third at 16x. Consumers index a pyramid BY POSITION, so the order is the
    contract -- 0 finest. Each level carries its OWN scale and translation, and ngff-zarr shifts the
    coarse origin by half the spacing delta, which is the centre-of-voxel convention these stores
    use; getting that wrong biases every voxel by a fraction of a coarse voxel and still looks like a
    plausible image. ``downsample_method`` selects how (see :func:`_downsample_method`).

    ``displacement_field`` writes it as a vector FIELD rather than an image: the component axis is
    typed ``displacement`` (NGFF RFC-5), which is what makes a registration DVF self-describing
    instead of an anonymous 3-channel image. A reader then no longer has to be told out of band that
    the channels are a displacement -- the mistake that path invites is silent, not loud: index the
    component axis like any other and you get one third of the field back, and a plausible-looking
    registration with it.

    The NGFF version follows from that flag and is deliberately NOT a parameter. RFC-5 axis types
    exist only from 0.6, so a caller passing both could only ever pass them consistently -- an
    invariant worth removing rather than documenting.
    """
    clear_ome_zarr_cache()
    _require_ngff_zarr()
    array_data = np.asarray(data)
    spatial_axes, scale_values, translation_values = _spatial_geometry(
        array_data.ndim, f"shape {array_data.shape}", spacing, origin
    )
    dims = ["c", *spatial_axes]
    scale = {"c": 1.0, **dict(zip(spatial_axes, scale_values, strict=True))}
    translation = {"c": 0.0, **dict(zip(spatial_axes, translation_values, strict=True))}

    image = ngff_zarr.to_ngff_image(array_data, dims=dims, scale=scale, translation=translation)
    # cache=False because the array is already resident: this function is handed one in memory, and
    # spilling it to a disk cache "to limit memory consumption" cannot lower a peak that has already
    # happened -- it only adds a full write and a full read. The estimate that would otherwise trigger
    # it is inflated besides, by itemsize once per dimension, so a 13.6 GiB field reports 868 GiB and
    # takes that path every time.
    multiscales = ngff_zarr.to_multiscales(
        image,
        scale_factors=[int(f) for f in scale_factors] if scale_factors else [],
        method=_downsample_method(downsample_method) if scale_factors else None,
        chunks=tuple(chunks) if chunks is not None else None,
        cache=False,
    )
    version = _DEFAULT_VERSION
    if displacement_field:
        _require_zarr_v3_for_rfc5()
        _type_component_axis(multiscales, _DISPLACEMENT_AXIS_TYPE)
        version = _RFC5_VERSION
    ngff_zarr.to_ngff_zarr(str(store_path), multiscales, overwrite=True, version=version)

    recorded = dict(attributes) if attributes else {}
    if displacement_field:
        # The producer holds the samples, so the bound is free here and a full scan anywhere else.
        recorded[DISPLACEMENT_BOUND_ATTRIBUTE] = displacement_bound(data)
    if recorded:
        group = zarr.open_group(str(store_path), mode="r+")
        group.attrs[_KONFAI_ATTR_KEY] = {"attributes": recorded}


def displacement_bound(data: np.ndarray) -> list[float]:
    """The largest absolute displacement per component, in the field's own world units.

    This is the number a consumer needs to size the region it must read before resampling through the
    field, and it is a property of the DATA, not of any parameter -- so the only place it is free is
    here, where the producer already holds the samples. Read back from the store's attributes it costs
    a header; recomputed by the consumer it costs a full scan of a field that can be 13.6 GiB.

    Per component, not one scalar: the component axis is (x, y, z) where the array axes are (z, y, x),
    and a consumer turns each bound into voxels by its OWN axis spacing -- these grids are anisotropic
    (40 um in z against 30.08 in x/y here), so a single collapsed maximum over-reads two axes.

    Computed in float32, the dtype a field is stored in: a bound rounded up in float64 and then
    compared against float32 samples is the one failure mode this is meant to remove.
    """
    field = np.asarray(data)
    if field.ndim < 2:
        return []
    # Cast before the max, not after: ``initial=np.float32(0.0)`` does not downcast a float64 input.
    flat = np.abs(field.reshape(field.shape[0], -1)).astype(np.float32, copy=False)
    return [float(flat[component].max(initial=np.float32(0.0))) for component in range(flat.shape[0])]


def update_konfai_attributes(store_path: str | Path, extra: dict[str, Any]) -> None:
    """Merge ``extra`` into the store's KonfAI attribute sidecar, keeping what is already there.

    For facts that are only knowable once the last region has landed -- a streamed field's own bound
    being the case that motivated it -- so they can still be recorded without holding the volume.
    """
    if not extra:
        return
    _require_zarr()
    group = zarr.open_group(str(store_path), mode="r+")
    sidecar = dict(dict(group.attrs).get(_KONFAI_ATTR_KEY, {}).get("attributes", {}))
    sidecar.update(extra)
    # Writing through an `r+` group updates the consolidated copy with it, so a sidecar landing here
    # is readable by a consolidated reader without a second consolidation pass. Measured, because the
    # opposite is the plausible assumption: a foreign `zarr.open_group(mode="r")` reads back a bound
    # written this way on both sides of this line.
    group.attrs[_KONFAI_ATTR_KEY] = {"attributes": sidecar}
    clear_ome_zarr_cache()


def _type_component_axis(multiscales: Any, axis_type: str) -> None:
    """Type the ``c`` axis of every coordinate system, in place.

    The axis type is set on the RFC-5 coordinate systems rather than on the ``NgffImage``, because
    ``to_multiscales`` derives the axes itself and hardcodes ``type="channel"`` for a ``c`` dim --
    tagging the image is a dead assignment on a non-frozen dataclass, and the store comes out an
    ordinary 3-channel image with no error raised anywhere.

    Coordinate systems are also the capability check: an ngff-zarr too old to model RFC-5 has no
    ``coordinateSystems`` on its metadata, and would otherwise write a silently untyped store.
    """
    systems = getattr(multiscales.metadata, "coordinateSystems", None)
    if not systems:
        raise DatasetManagerError(
            f"Writing a '{axis_type}' field needs NGFF RFC-5 coordinate systems, which this ngff-zarr cannot model.",
            "Upgrade it with: pip install 'ngff-zarr>=0.38'",
        )
    for system in systems:
        for axis in system.axes:
            if axis.name == "c":
                axis.type = axis_type


def create_ome_zarr_store(
    store_path: str | Path,
    shape: Sequence[int],
    dtype: Any,
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: Sequence[int] | None = None,
    displacement_field: bool = False,
) -> Any:
    """Create an empty single-level OME-NGFF store for region-by-region writes.

    Returns the level-0 zarr array: chunks materialise as regions are assigned, and unwritten regions
    read back as zeros. Metadata is complete from the start, so the store is readable at any point
    during the write.

    ngff-zarr writes that metadata, exactly as it does for the whole-array path, so both paths describe
    a store the same way -- ``displacement_field`` included, which is the point of routing it through
    ngff-zarr at all. A field too large to assemble in memory is written region by region, so this is
    the ONLY path a real one takes, and until it could type its component axis a DVF came out
    self-describing exactly when it was small enough not to need to be.

    It describes the store from a one-voxel stand-in rather than from an array of the target shape.
    With a single resolution level the metadata does not depend on the extent at all -- axes and
    coordinate transformations come from ``dims``, ``scale`` and ``translation`` -- and the two come
    out byte-identical, verified for 0.4 and 0.6. Handing ngff-zarr the real shape instead costs a
    pass over every chunk of an array that is entirely zeros (~44 ms per chunk, ~33 s for a 13.6 GiB
    field) to write no bytes at all. The real array is then created underneath that metadata, which is
    also what makes its chunking exactly the caller's: the region grid is the one thing ngff-zarr
    cannot infer, and a store whose chunks straddle it turns every region write into a
    read-modify-write.
    """
    clear_ome_zarr_cache()
    _require_ngff_zarr()
    spatial_axes, scale_values, translation_values = _spatial_geometry(
        len(shape), f"shape {list(shape)}", spacing, origin
    )
    dims = ["c", *spatial_axes]
    scale = {"c": 1.0, **dict(zip(spatial_axes, scale_values, strict=True))}
    translation = {"c": 0.0, **dict(zip(spatial_axes, translation_values, strict=True))}

    if chunks is None:
        spatial_chunks = [min(extent, CHUNK_SPATIAL_TILE) for extent in shape[1:]]
        # Keep one chunk near the target: full spatial tiles, channels split to fit the budget.
        tile_bytes = int(np.prod(spatial_chunks, dtype=np.int64)) * np.dtype(dtype).itemsize
        chunks = [min(shape[0], max(1, CHUNK_TARGET_BYTES // max(1, tile_bytes))), *spatial_chunks]
    chunks = tuple(chunks)

    stand_in = dask.array.zeros((shape[0], *(1,) * len(spatial_axes)), dtype=np.dtype(dtype))
    image = ngff_zarr.to_ngff_image(stand_in, dims=dims, scale=scale, translation=translation)
    multiscales = ngff_zarr.to_multiscales(image, scale_factors=[])
    version = _DEFAULT_VERSION
    if displacement_field:
        _require_zarr_v3_for_rfc5()
        _type_component_axis(multiscales, _DISPLACEMENT_AXIS_TYPE)
        version = _RFC5_VERSION
    # version is explicit because to_ngff_zarr defaults to 0.5, which zarr-python 2 cannot write.
    ngff_zarr.to_ngff_zarr(str(store_path), multiscales, overwrite=True, version=version)

    # The level-0 key comes from the metadata rather than a literal: ngff-zarr builds it from the
    # image name, so "scale0/image" is its convention to change, not ours to hardcode.
    group = zarr.open_group(str(store_path), mode="r+")
    create_array = getattr(group, "create_array", None) or group.create_dataset
    array = create_array(
        multiscales.metadata.datasets[0].path,
        shape=tuple(shape),
        chunks=chunks,
        dtype=np.dtype(dtype),
        fill_value=0,
        overwrite=True,
    )
    # Sidecar last: to_ngff_zarr(overwrite=True) reopens the root with mode="w" and drops every
    # attribute it finds, so writing this first loses Direction -- and losing Direction is silent,
    # the reader falls back to identity and returns a plausibly-oriented volume.
    if attributes:
        group.attrs[_KONFAI_ATTR_KEY] = {"attributes": dict(attributes)}
    # ngff-zarr leaves a consolidated index behind, and readers trust it over the arrays themselves --
    # so until it is rebuilt the store still advertises the one-voxel stand-in, whatever is on disk.
    # Last, so that the sidecar written just above is part of what gets indexed.
    zarr.consolidate_metadata(str(store_path))
    return array


def append_ome_zarr_levels(
    store_path: str | Path,
    scale_factors: Sequence[int],
    *,
    downsample_method: str | None = None,
) -> None:
    """Add coarser levels to a store that already holds its level 0.

    The companion of :func:`create_ome_zarr_store`: a store written region by region cannot be given
    ``scale_factors`` up front, because no level exists until the last region lands. This derives the
    pyramid afterwards, from what is on disk.

    It REWRITES level 0 rather than adding beside it -- ngff-zarr composes a multiscales as a whole,
    and there is no supported way to graft a level onto a published one. The cost is real and
    proportional to the store (measured ~42 s for a 2 GiB store) while the peak stays chunk-sized.
    Worth knowing before calling it in a loop; not worth hiding.

    It writes to a SIBLING store and renames over the original, rather than in place. In place is not
    slower, it is wrong: level 0 is read lazily, ``to_ngff_zarr(overwrite=True)`` truncates the store
    before dask pulls a single chunk through it, and the pyramid comes out uniformly zero -- with
    correct metadata, correct shapes, and nothing raised. The rename also makes the operation atomic,
    so an interrupted call leaves the original store intact instead of a half-written one.

    The KonfAI attribute sidecar and an RFC-5 displacement typing are carried across, because
    ``to_ngff_zarr`` writes a root of its own -- and both losses are silent, one landing as an
    identity Direction and the other as an anonymous 3-channel image.
    """
    _require_ngff_zarr()
    if not scale_factors:
        return
    store = Path(store_path)
    clear_ome_zarr_cache()
    field = is_displacement_field(store)
    sidecar = _read_konfai_attributes(store)
    staging = store.with_name(f"{store.name}.appending")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        base = ngff_zarr.from_ngff_zarr(str(store)).images[0]
        multiscales = ngff_zarr.to_multiscales(
            base,
            scale_factors=[int(f) for f in scale_factors],
            method=_downsample_method(downsample_method),
            # Level 0 is rewritten here, so the base store's own chunking has to be carried across:
            # ngff-zarr otherwise re-chunks it on its default.
            chunks=tuple(int(size) for size in base.data.chunksize),
            cache=False,
        )
        version = _DEFAULT_VERSION
        if field:
            _require_zarr_v3_for_rfc5()
            _type_component_axis(multiscales, _DISPLACEMENT_AXIS_TYPE)
            version = _RFC5_VERSION
        ngff_zarr.to_ngff_zarr(str(staging), multiscales, overwrite=True, version=version)
        if sidecar:
            group = zarr.open_group(str(staging), mode="r+")
            group.attrs[_KONFAI_ATTR_KEY] = {"attributes": dict(sidecar)}
            zarr.consolidate_metadata(str(staging))
        # Rename in both directions, so no instant exists with neither store on disk: deleting the
        # original first opens a window as long as a full tree delete, which the ``finally`` below
        # then completes by clearing the replacement too.
        replaced = store.with_name(f"{store.name}.replaced")
        shutil.rmtree(replaced, ignore_errors=True)
        store.rename(replaced)
        try:
            staging.rename(store)
        except BaseException:
            replaced.rename(store)
            raise
        shutil.rmtree(replaced, ignore_errors=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        clear_ome_zarr_cache()


def get_ome_zarr_info(store_path: str | Path, level: int = 0) -> dict[str, Any]:
    """OME-Zarr metadata, without reading pixel data.

    Three of these keys describe the same level in two different orders, and mixing them is the
    single most productive mistake this module invites. ``shape``, ``scale`` and ``translation``
    follow the STORE's own axes, listed in ``axes`` -- a scalar volume written without a channel
    axis has three of each. ``canonical_shape`` is the C[Z]YX form the reader indexes. So a caller
    that sizes its slices from ``canonical_shape`` and then reads ``scale[1:]`` for the spatial
    spacing is off by one axis, with plausible numbers and no error.

    ``geometry`` exists so that never has to be reasoned about: it maps each axis NAME to its
    ``(scale, translation)``. Prefer it.
    """
    image = _load_image(str(store_path), level)
    dims = [str(axis).lower() for axis in image.dims]
    try:
        n_levels = len(ngff_zarr.from_ngff_zarr(str(store_path)).images)
    except (OSError, TypeError, ValueError):
        n_levels = 1
    return {
        "axes": dims,
        "shape": list(image.data.shape),
        # The shape the slices of `read_ome_zarr_data_slice` are indexed against. Both are here
        # because they differ whenever the store's axes are not already C[Z]YX, and a caller sizing
        # its slices from "shape" then reads a transposed region -- with the right rank, plausible
        # values, and nothing raised. "shape" stays the store's own order; this one is the reader's.
        "canonical_shape": _canonical_shape(dims, image.data.shape),
        "chunks": list(getattr(image.data, "chunks", []) or []),
        "dtype": str(image.data.dtype),
        "scale": _ordered(dict(image.scale), dims),
        "translation": _ordered(dict(image.translation), dims),
        # Keyed by axis name, so no caller has to know which of the two orders it is holding.
        "geometry": {
            axis: {"scale": float(scale), "translation": float(translation)}
            for axis, scale, translation in zip(
                dims,
                _ordered(dict(image.scale), dims),
                _ordered(dict(image.translation), dims),
                strict=True,
            )
        },
        "n_levels": n_levels,
        "attributes": _read_konfai_attributes(store_path),
    }
