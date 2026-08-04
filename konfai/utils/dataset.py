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

"""Dataset file abstractions and image conversion utilities for KonfAI."""

from __future__ import annotations

import ast
import copy
import csv
import functools
import glob
import itertools
import math
import os
import re
import shutil
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from lxml import etree  # nosec B410

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai import current_date
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import SUPPORTED_EXTENSIONS, split_format_level

_h5_file_locks: dict[str, threading.RLock] = {}
_h5_file_locks_guard = threading.Lock()


def _get_h5_file_lock(filename: str) -> threading.RLock:
    """Return the process-wide lock guarding one HDF5 file across worker threads."""
    with _h5_file_locks_guard:
        lock = _h5_file_locks.get(filename)
        if lock is None:
            lock = threading.RLock()
            _h5_file_locks[filename] = lock
        return lock


class _PooledRead(NamedTuple):
    """An open read handle and the store it was opened on, as one thing: the two travel together through
    eviction and re-insertion, so no site can pair a handle with a view it never had."""

    file: Any
    opened_on: tuple[int, int] | None


class _H5ReadPool:
    """Pooled read handles, one per file per process, LRU-bounded.

    The HDF5 chunk cache lives on the open handle, so reusing the handle across patch reads is what
    makes the cache effective — a per-read open rebuilds it empty every time. ``get``/``drop`` must be
    called under the file's lock; a write drops the file's reader so it never serves stale metadata;
    handles inherited across ``fork`` are dropped unused (closing them would flush another process's
    state). Pooled handles open with ``locking=False``: a held HDF5 read lock would block every other
    process's write-open of the file for as long as the handle lives — the pool's whole lifetime.
    Same-process access is serialized by the per-file thread lock.

    A handle also stops answering for a store another PROCESS has written — a loader worker producing
    the group its parent reads — so one is kept only while the file it was opened on is unchanged.
    Reopening alone would not do: HDF5 shares a file's metadata state across the handles one process
    holds, so a second handle inherits the first's view. The stale one is closed before the new open."""

    _MAX = 8
    _OPEN_ATTEMPTS = 4
    _OPEN_BACKOFF = 0.05

    def __init__(self) -> None:
        self._handles: dict[str, _PooledRead] = {}
        self._guard = threading.Lock()
        self._pid = os.getpid()

    @staticmethod
    def _stamp(filename: str) -> tuple[int, int] | None:
        """What the store looked like when a handle was opened on it; ``None`` while it does not exist."""
        try:
            info = os.stat(filename)
        except OSError:
            return None
        return info.st_mtime_ns, info.st_size

    def _open(self, filename: str, **open_kwargs: Any) -> _PooledRead:
        """A handle, with the store as it was when it was opened.

        The reopen happens exactly when another process has just written, which is when that process is
        most likely to still be mid-transaction: HDF5 without SWMR then refuses the open. It is transient,
        so it is retried, and each attempt takes its own stamp — a handle is never paired with a view of
        the store taken before the write that made the previous attempt fail."""
        for remaining in reversed(range(self._OPEN_ATTEMPTS)):
            stamp = self._stamp(filename)
            try:
                return _PooledRead(h5py.File(filename, "r", locking=False, **open_kwargs), stamp)
            except OSError:
                if not remaining:
                    raise
                time.sleep(self._OPEN_BACKOFF)
        raise AssertionError("unreachable: the last attempt either returns or raises")

    def get(self, filename: str, **open_kwargs: Any) -> Any:
        # Read before the open, never after: a write landing in between then leaves a stamp older than
        # the handle, and the next call reopens. The reverse would record a view it never had.
        stamp = self._stamp(filename)
        with self._guard:
            if os.getpid() != self._pid:
                self._handles.clear()
                self._pid = os.getpid()
            pooled = self._handles.pop(filename, None)
        # Outside the pool guard: opening touches the filesystem and may sleep between attempts. The
        # caller holds this file's lock, so no other thread of ours is reading or reopening it here.
        if pooled is not None and (not pooled.file.id.valid or pooled.opened_on != stamp):
            pooled.file.close()
            pooled = None
        if pooled is None:
            pooled = self._open(filename, **open_kwargs)
        with self._guard:
            self._handles[filename] = pooled
            evicted = []
            while len(self._handles) > self._MAX:
                oldest = next(iter(self._handles))
                evicted.append((oldest, self._handles.pop(oldest)))
        for stale_name, stale in evicted:
            self._close_idle(stale_name, stale)
        return pooled.file

    def drop(self, filename: str) -> None:
        with self._guard:
            pooled = self._handles.pop(filename, None)
        if pooled is not None and pooled.file.id.valid:
            pooled.file.close()

    def _close_idle(self, filename: str, pooled: _PooledRead) -> None:
        # An evicted handle may be mid-read under its file's lock: close only when that lock is free,
        # otherwise put it back in the pool — an untracked open handle could never be dropped again.
        # It goes back with the stamp it came with: re-stamping would hand it the store as it is now,
        # and a write it never saw would stay invisible for the rest of the process.
        lock = _get_h5_file_lock(filename)
        if lock.acquire(blocking=False):
            try:
                pooled.file.close()
            finally:
                lock.release()
        else:
            with self._guard:
                self._handles.setdefault(filename, pooled)


_h5_read_pool = _H5ReadPool()


def _attribute_text(value: Any) -> str:
    """One value as an attribute holds it: its printed form, on one line and complete.

    The single place a value stops being a live object, because every consumer takes text --
    ``SetMetaData`` on a SimpleITK image, an h5 attribute, a zarr sidecar. The printed form is left
    as each type prints it: a sequence has two, :meth:`Attribute.get_np_array` reads both, and
    ``ast.literal_eval`` (:meth:`Dataset.read_transform`, on the parameter keys) needs the Python
    one, so normalising to either here breaks the other reader.

    Complete, because an attribute is a record and not a display: NumPy's own printing elides values
    past a threshold, and an elided record is one no reader can parse back.
    """
    if isinstance(value, torch.Tensor):
        # Accept a tensor from any device: attributes are host-side strings, and finalize transforms
        # (Normalize, Statistics, ...) may hand over stats computed on a CUDA-resident volume.
        value = value.detach().cpu().numpy()
    with np.printoptions(threshold=sys.maxsize):
        return str(value).replace("\n", "")


def _store_chunks(shape: list[int], region_shape: list[int] | None, dtype: Any) -> tuple[int, ...] | None:
    """Chunks a store should use, given the region shape its writer declared.

    A region write that straddles a chunk becomes a read-modify-write of the whole chunk, so the
    writer's own region is the honest starting point. Taking it verbatim is not: a slab sweep
    declares the whole trailing plane, which at 2048x2048 float32 is a gigabyte in one chunk -- past
    what zarr will hold in a single buffer at 4096x4096 -- and every later partial read pays it,
    including readers that are not KonfAI.

    So an axis the region covers END TO END is tiled, and an axis where the region is a strict
    sub-range keeps the writer's size. Tiling a fully-covered axis cannot split a region write, since
    every chunk along it falls inside one region either way; splitting a partial axis is exactly the
    read-modify-write this exists to avoid. Innermost axes go first, so the chunk stays long on the
    axis the writer actually advances along.

    ``None`` when the writer declared nothing: the store keeps its own default rather than a guess.
    """
    from konfai.utils.ome_zarr import CHUNK_SPATIAL_TILE, CHUNK_TARGET_BYTES

    if region_shape is None or len(region_shape) != len(shape):
        return None
    chunk = [max(1, min(int(region), int(extent))) for region, extent in zip(region_shape, shape, strict=True)]
    itemsize = max(1, np.dtype(dtype).itemsize)
    covered = [axis for axis, extent in enumerate(shape) if chunk[axis] >= int(extent)]
    for axis in reversed(covered):
        if int(np.prod(chunk, dtype=np.int64)) * itemsize <= CHUNK_TARGET_BYTES:
            break
        chunk[axis] = min(chunk[axis], CHUNK_SPATIAL_TILE)
    return tuple(chunk)


class Attribute(dict[str, Any]):
    """Metadata container storing repeated values with a stack-like naming scheme.

    Values are text, always. Both doors normalize -- assignment and construction -- so an attribute
    built from a store's own sidecar, which JSON hands back as live lists, is the same thing as one
    assigned in Python. Anything less and a value can be stored and not written back out.
    """

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        super().__init__()
        attributes = attributes or {}
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), _attribute_text(v))

    @staticmethod
    def _is_stack_member(stored_key: str, key: str) -> bool:
        # Values are stacked as ``{key}_{n}``; match that exact pattern (or the bare key) so a sibling that
        # merely shares a prefix -- ``SpacingOriginal`` vs ``Spacing`` -- is not miscounted as another entry.
        if stored_key == key:
            return True
        prefix = f"{key}_"
        return stored_key.startswith(prefix) and stored_key[len(prefix) :].isdigit()

    def _count_key(self, key: str) -> int:
        return sum(1 for k in super().keys() if Attribute._is_stack_member(k, key))

    def __getitem__(self, key: str) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return str(super().__getitem__(f"{key}_{i - 1}"))
        if key in super().keys():
            return str(super().__getitem__(key))
        raise NameError(f"{key} not in cache_attribute")

    def __setitem__(self, key: str, value: Any) -> None:
        result = _attribute_text(value)
        if "_" not in key:
            super().__setitem__(f"{key}_{self._count_key(key)}", result)
        else:
            super().__setitem__(key, result)

    def pop(self, key: str, default: Any = None) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return super().pop(f"{key}_{i - 1}")
        if key in super().keys():
            return super().pop(key)
        raise NameError(f"{key} not in cache_attribute")

    @staticmethod
    def _parse_array(text: str) -> np.ndarray:
        """Both printed forms of a sequence: NumPy's ``[1.5 1.5 2.]`` and Python's ``[1.5, 1.5, 2.0]``.

        Which one an attribute holds follows from what the writer handed over -- an ``ndarray``, or
        the plain list a JSON sidecar gives back -- and no reader should have to make that
        distinction. ``np.fromstring`` reads whitespace only, so the commas go first.
        """
        return np.fromstring(text[1:-1].replace(",", " "), sep=" ", dtype=np.double)

    def get_np_array(self, key: str) -> np.ndarray:
        return Attribute._parse_array(self[key])

    def get_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)

    def pop_np_array(self, key: str) -> np.ndarray:
        return Attribute._parse_array(self.pop(key))

    def pop_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(Attribute._is_stack_member(k, key) for k in super().keys())

    def is_info(self, key: str, value: str) -> bool:
        return key in self and self[key] == value


# Elements held in memory at once while accumulating statistics chunk by chunk, whatever the backend.
_STATISTICS_CHUNK_ELEMENTS = 8_000_000


def _statistics_chunk_length(shape: list[int] | tuple[int, ...], axis: int) -> int:
    """How far along ``axis`` a chunk may reach to hold about ``_STATISTICS_CHUNK_ELEMENTS``.

    A chunk spans every other axis whole (channels included), so the per-step cost is the volume
    divided by ``axis``; the length is that budget over the per-step cost, floored to one step.
    """
    per_step = int(np.prod([extent for other, extent in enumerate(shape) if other != axis], dtype=np.int64))
    return max(1, _STATISTICS_CHUNK_ELEMENTS // max(1, per_step))


def _update_running_statistics(
    state: dict[str, Any] | None,
    array: np.ndarray,
) -> dict[str, Any]:
    """Update running min/max/mean/std from a NumPy chunk, over the volume AND per channel.

    Both grains come from one pass because they come from the same samples: a chunk arrives as
    ``(C, ...)``, so the per-channel figures are the same Welford recurrence applied along axis 0,
    and the whole-volume ones are that recurrence pooled. Computing them separately would mean
    scanning the volume once per channel -- three passes over a displacement field to learn three
    numbers.
    """
    values = np.asarray(array, dtype=np.float64)
    per_channel = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(1, -1)
    flat = per_channel.reshape(-1)
    channels = per_channel.shape[0]
    if flat.size == 0:
        return state or _empty_statistics_state(channels)

    if state is None:
        state = _empty_statistics_state(channels)

    chunk_count = float(flat.size)
    chunk_mean = float(flat.mean())
    chunk_m2 = float(np.square(flat - chunk_mean).sum())

    # Per channel, the same recurrence on vectors -- one entry per channel, updated together.
    channel_count = float(per_channel.shape[1])
    channel_mean = per_channel.mean(axis=1)
    channel_m2 = np.square(per_channel - channel_mean[:, None]).sum(axis=1)
    channel_total = state["channel_count"] + channel_count
    if channel_total > 0:
        channel_delta = channel_mean - state["channel_mean"]
        state["channel_mean"] = state["channel_mean"] + channel_delta * channel_count / channel_total
        state["channel_m2"] = (
            state["channel_m2"]
            + channel_m2
            + channel_delta * channel_delta * state["channel_count"] * channel_count / channel_total
        )
        state["channel_count"] = channel_total
        state["channel_min"] = np.minimum(state["channel_min"], per_channel.min(axis=1))
        state["channel_max"] = np.maximum(state["channel_max"], per_channel.max(axis=1))

    total_count = state["count"] + chunk_count
    delta = chunk_mean - state["mean"]
    if total_count > 0:
        state["mean"] += delta * chunk_count / total_count
        state["m2"] += chunk_m2 + delta * delta * state["count"] * chunk_count / total_count
        state["count"] = total_count
        state["min"] = min(state["min"], float(flat.min()))
        state["max"] = max(state["max"], float(flat.max()))
    return state


def _empty_statistics_state(channels: int) -> dict[str, Any]:
    return {
        "count": 0.0,
        "mean": 0.0,
        "m2": 0.0,
        "min": np.inf,
        "max": -np.inf,
        "channel_count": 0.0,
        "channel_mean": np.zeros(channels, dtype=np.float64),
        "channel_m2": np.zeros(channels, dtype=np.float64),
        "channel_min": np.full(channels, np.inf),
        "channel_max": np.full(channels, -np.inf),
    }


def _finalize_running_statistics(state: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a running-statistics state into the public stats dictionary.

    The four scalars are the volume's; the four ``*_per_channel`` lists are the same figures per
    channel, which is what a vector-valued quantity needs -- the mean of a displacement field is
    three numbers, and pooling them into one describes nothing.
    """
    if state is None or state["count"] == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0} | {
            f"{key}_per_channel": [0.0] for key in ("min", "max", "mean", "std")
        }
    variance = state["m2"] / (state["count"] - 1) if state["count"] > 1 else 0.0
    channel_variance = state["channel_m2"] / max(1.0, state["channel_count"] - 1)
    return {
        "min": state["min"],
        "max": state["max"],
        "mean": state["mean"],
        "std": math.sqrt(max(variance, 0.0)),
        "min_per_channel": state["channel_min"].tolist(),
        "max_per_channel": state["channel_max"].tolist(),
        "mean_per_channel": state["channel_mean"].tolist(),
        "std_per_channel": np.sqrt(np.maximum(channel_variance, 0.0)).tolist(),
    }


# Formats already reported by _warn_unstreamed_region_read. Keyed by format, not by file: the remedy
# is dataset-wide, so every case of a dataset would otherwise repeat the same warning.
_unstreamed_formats_warned: set[str] = set()


def _warn_unstreamed_region_read(path: str) -> None:
    """Warn that `path`'s format decodes the whole volume for every patch region read from it.

    `warnings.warn` dedups per call site, which here is one line in a loop over every patch of every
    case: the seen-set is what makes this once per format rather than thousands of times.
    """
    suffix = Path(path).suffix
    if suffix in _unstreamed_formats_warned:
        return
    _unstreamed_formats_warned.add(suffix)
    warnings.warn(
        f"Patch-streaming '{suffix}' files (e.g. '{path}'): this format cannot serve a disk region "
        "(NRRD, or any compressed file), so every patch decodes the whole volume again -- many times "
        "the cost of one read. Convert the dataset to a chunked format (OME-Zarr or HDF5), which KonfAI "
        "streams natively, or to an uncompressed .mha/.nii. Warned once per format.",
        stacklevel=2,
    )


def is_an_image(attributes: Attribute) -> bool:
    """Return whether the given attribute set contains image geometry metadata."""
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes


def data_to_image(data: np.ndarray, attributes: Attribute) -> sitk.Image:
    """Convert a NumPy array and KonfAI attributes into a SimpleITK image."""
    if isinstance(data, torch.Tensor):
        # Accept a torch tensor on any device: SimpleITK works on host arrays, so a SITK-backed transform
        # fed a CUDA-resident volume converts here and naturally returns on the CPU (the pipeline then
        # continues on the CPU). This keeps every transform usable regardless of the volume's device.
        data = data.detach().cpu().numpy()
    if not is_an_image(attributes):
        raise NameError("Data is not an image")
    if data.dtype == np.float16:
        # ITK has no half-float pixel type (GetImageFromArray rejects float16), so widen to float32 --
        # exact and lossless. The streamed .mha writer widens the same way, so both write identical bytes.
        data = data.astype(np.float32)
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i + 1 for i in range(len(data.shape) - 1)] + [0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    for k, v in attributes.items():
        if v and len(v):
            image.SetMetaData(k, v)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
    return image


# Set on an entry read back from a store that types its component axis as an RFC-5 displacement
# field, so ``Dataset.read_transform`` can rebuild the transform. The underscore matters: a key
# without one is stack-renamed by ``Attribute.__setitem__`` (``Transform`` becomes ``Transform_0``).
DISPLACEMENT_FIELD_ATTRIBUTE = "konfai_displacement_field"


def displacement_field_to_data(transform: sitk.Transform, name: str) -> tuple[np.ndarray, Attribute]:
    """A displacement-field transform as a channel-first array plus its geometry.

    The counterpart of ``_encode_transform_leaves`` for the one transform kind that cannot go through
    it: a displacement field's parameters ARE the field, so serialising it as a parameter vector
    would drop the geometry that makes it meaningful. It travels as an image instead, and the store
    records what it is (see ``write_ome_zarr(displacement_field=True)``).
    """
    if not isinstance(transform, sitk.DisplacementFieldTransform):
        raise DatasetManagerError(
            f"Expected a DisplacementFieldTransform for entry '{name}', got '{type(transform).__name__}'."
        )
    return image_to_data(transform.GetDisplacementField())


def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    """Convert a SimpleITK image into a channel-first NumPy array and attributes."""
    attributes = Attribute()
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    for k in image.GetMetaDataKeys():
        attributes[k] = image.GetMetaData(k)
    data = sitk.GetArrayFromImage(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
    return data, attributes


def ome_zarr_attributes(metadata: dict[str, Any]) -> Attribute:
    """A KonfAI ``Attribute`` (Origin / Spacing / Direction) from an OME-Zarr entry's metadata.

    The store's konfai sidecar wins when present -- it carries the full Direction matrix, which NGFF
    scale/translation cannot express -- otherwise geometry falls back to the NGFF transforms, Direction
    defaulting to identity. Shared by the Dataset OME-Zarr reader and ``ITK.read_displacement_field``
    so both recover geometry the one same way.
    """
    attributes = Attribute(metadata.get("attributes", {}))
    axes = metadata["axes"]
    scale = dict(zip(axes, metadata.get("scale", []), strict=False))
    translation = dict(zip(axes, metadata.get("translation", []), strict=False))
    spatial_axes = [axis for axis in ("x", "y", "z") if axis in axes]
    if "Spacing" not in attributes:
        attributes["Spacing"] = np.asarray([scale.get(axis, 1.0) for axis in spatial_axes])
    if "Origin" not in attributes:
        attributes["Origin"] = np.asarray([translation.get(axis, 0.0) for axis in spatial_axes])
    if "Direction" not in attributes:
        attributes["Direction"] = np.eye(len(spatial_axes), dtype=np.float64).flatten()
    attributes["OMEAxes"] = np.asarray(axes)
    return attributes


def _flatten_transforms(transform: sitk.Transform) -> list[sitk.Transform]:
    """The leaf transforms of a (possibly nested) composite, in application order.

    ``CompositeTransform.GetNthTransform`` can itself return a composite, so a single-level walk
    leaves a nested composite in the list and the serializer rejects it. Recurse to the leaves.
    """
    if isinstance(transform, sitk.CompositeTransform):
        leaves: list[sitk.Transform] = []
        for i in range(transform.GetNumberOfTransforms()):
            leaves.extend(_flatten_transforms(transform.GetNthTransform(i)))
        return leaves
    return [transform]


def _transform_codec() -> list[tuple[type, str, Any]]:
    """(sitk class, serialized type tag, decode factory) for every supported transform kind.

    Built lazily because ``sitk`` is an optional import.
    """
    return [
        (sitk.Euler3DTransform, "Euler3DTransform_double_3_3", sitk.Euler3DTransform),
        (sitk.AffineTransform, "AffineTransform_double_3_3", lambda: sitk.AffineTransform(3)),
        (sitk.BSplineTransform, "BSplineTransform_double_3_3", lambda: sitk.BSplineTransform(3)),
    ]


def _encode_transform_leaves(transform: sitk.Transform, name: str, attributes: Attribute) -> list[np.ndarray]:
    """Serialize a (possibly composite) transform: record each leaf's type tag and fixed parameters
    into ``attributes`` (``{i}:Transform`` / ``{i}:FixedParameters``) and return the per-leaf
    parameter arrays, in application order."""
    datas: list[np.ndarray] = []
    for i, leaf in enumerate(_flatten_transforms(transform)):
        type_tag = next((tag for sitk_class, tag, _ in _transform_codec() if isinstance(leaf, sitk_class)), None)
        if type_tag is None:
            raise DatasetManagerError(f"Unsupported transform type '{type(leaf).__name__}' for entry '{name}'.")
        attributes[f"{i}:Transform"] = type_tag
        attributes[f"{i}:FixedParameters"] = leaf.GetFixedParameters()

        datas.append(np.asarray(leaf.GetParameters()))
    return datas


def _decode_transform(transform_type: str, name: str) -> sitk.Transform:
    """A fresh transform instance for a serialized type tag."""
    for _, type_tag, factory in _transform_codec():
        if transform_type == type_tag:
            return factory()
    raise DatasetManagerError(f"Unsupported transform type '{transform_type}' for entry '{name}'.")


def get_infos(filename: str | Path) -> tuple[list[int], Attribute]:
    """Read shape and metadata from an image file without loading its full pixel data."""
    attributes = Attribute()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(filename))
    file_reader.ReadImageInformation()
    attributes["Origin"] = np.asarray(file_reader.GetOrigin())
    attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
    attributes["Direction"] = np.asarray(file_reader.GetDirection())
    for k in file_reader.GetMetaDataKeys():
        attributes[k] = file_reader.GetMetaData(k)
    # SimpleITK GetSize() is (x, y, [z], ...); KonfAI arrays are numpy-order [C, (Z), Y, X], so the
    # spatial size must be reversed for EVERY rank -- a 3-D-only reversal transposes 2-D/4-D data.
    size = list(reversed(file_reader.GetSize()))
    size = [file_reader.GetNumberOfComponents(), *size]
    return size, attributes


def read_landmarks(filename: Path) -> np.ndarray | None:
    """Read Slicer-style fiducial landmarks from disk."""
    data = None
    with open(filename, newline="") as csvfile:
        reader = csv.reader(filter(lambda row: row[0] != "#", csvfile))
        lines = list(reader)
        data = np.zeros((len(list(lines)), 3), dtype=np.double)
        for i, row in enumerate(lines):
            data[i] = np.array(row[1:4], dtype=np.double)
        csvfile.close()
    return data


def write_landmarks(data: np.ndarray, filename: Path) -> None:
    """Write landmarks to the Slicer Markups fiducial CSV-like format."""
    with open(filename, "w") as f:
        f.write(
            "# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n#"
            " columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
        )
        for i in range(data.shape[0]):
            f.write(
                "vtkMRMLMarkupsFiducialNode_"
                + str(i + 1)
                + ","
                + str(data[i, 0])
                + ","
                + str(data[i, 1])
                + ","
                + str(data[i, 2])
                + ",0,0,0,1,1,1,0,F-"
                + str(i + 1)
                + ",,vtkMRMLScalarVolumeNode1\n"
            )
        f.close()


class DataStream(ABC):
    """One dataset entry written incrementally, region by region. Obtained from
    ``Dataset.open_data_stream``, which returns ``None`` when the write format cannot serve region writes
    (the caller then assembles the volume and uses ``Dataset.write``). Use as a context manager: a clean
    exit finalizes the entry, an exception removes the partial one so a reader never sees a half-written
    volume.

    The entry lives under a temporary name until the clean exit renames it into place: an existence
    probe (``is_dataset_exist``) or a concurrent reader never sees the entry while it is being written,
    a replaced entry stays readable until its replacement is complete, and a hard-killed writer leaves
    only temporary debris, never a plausible-looking partial volume under the final name. The
    temporary name is unique per stream (PID + sequence): two writers of the same entry (a case
    landing on two workers) each own their temporary, and whichever finalizes last publishes — a
    complete entry either way, never an interleaving of the two."""

    _sequence = itertools.count()

    @staticmethod
    def temporary_suffix() -> str:
        """The per-stream unique suffix a backend appends to its temporary name."""
        return f"{os.getpid()}-{next(DataStream._sequence)}.tmp"

    _file: Dataset.File | None = None
    _finished: bool = False

    def __enter__(self) -> DataStream:
        return self

    @abstractmethod
    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        """Write ``data`` into the region ``slices`` (channel-first indices, step 1)."""

    @abstractmethod
    def _close(self, success: bool) -> None:
        """Finalize the entry, or remove the partial one when ``success`` is False."""

    def close(self) -> None:
        """Finalize the entry under its final name."""
        self._finish(True, None, None, None)

    def abort(self, error: BaseException | None = None) -> None:
        """Remove the partial entry."""
        if error is None:
            self._finish(False, None, None, None)
        else:
            self._finish(False, type(error), error, error.__traceback__)

    def __exit__(self, exc_type, value, traceback) -> None:
        self._finish(exc_type is None, exc_type, value, traceback)

    def _finish(self, success: bool, exc_type, value, traceback) -> None:
        # Single-shot: a caller may both close() and, on the error path, abort() the same stream (or
        # exit a ``with`` that already closed). Only the first call acts, so the backing file is exited
        # once and a failed close is not overwritten by a second _close on already-released state.
        if self._finished:
            return
        self._finished = True
        try:
            self._close(success)
        finally:
            if self._file is not None:
                self._file.__exit__(exc_type, value, traceback)


class _H5DataStream(DataStream):
    def __init__(self, dataset: h5py.Dataset, final_name: str) -> None:
        self._dataset = dataset
        self._final_name = final_name

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._dataset[slices] = data

    def _close(self, success: bool) -> None:
        parent = self._dataset.parent
        temporary_name = self._dataset.name.rsplit("/", 1)[-1]
        if not success:
            del parent[temporary_name]
            return
        if self._final_name in parent:
            del parent[self._final_name]
        parent.move(temporary_name, self._final_name)


# MetaImage ElementType for each NumPy dtype a streamed .mha can hold.
_MHA_ELEMENT_TYPES = {
    "int8": "MET_CHAR",
    "uint8": "MET_UCHAR",
    "int16": "MET_SHORT",
    "uint16": "MET_USHORT",
    "int32": "MET_INT",
    "uint32": "MET_UINT",
    "int64": "MET_LONG_LONG",
    "uint64": "MET_ULONG_LONG",
    "float32": "MET_FLOAT",
    "float64": "MET_DOUBLE",
}


class _MhaDataStream(DataStream):
    """Uncompressed local-data MetaImage written region by region: a hand-written ASCII header, then a
    memmap over the flat raw block. MetaIO stores vector pixels interleaved (channel fastest), so the
    map is spatial-first ``[.., Y, X, C]`` and ``write_slice`` moves the channel axis last."""

    def __init__(self, path: str, shape: list[int], dtype: np.dtype, attributes: Attribute) -> None:
        self.path = path
        self._temporary_path = f"{path}.{self.temporary_suffix()}"
        spatial = list(shape[1:])
        # The header declares BinaryDataByteOrderMSB=False, so the map must be explicitly little-endian.
        self._dtype = np.dtype(dtype).newbyteorder("<")
        fields: list[tuple[str, str]] = [
            ("ObjectType", "Image"),
            ("NDims", str(len(spatial))),
            ("BinaryData", "True"),
            ("BinaryDataByteOrderMSB", "False"),
            ("CompressedData", "False"),
            ("TransformMatrix", " ".join(str(v) for v in attributes.get_np_array("Direction"))),
            ("Offset", " ".join(str(v) for v in attributes.get_np_array("Origin"))),
            ("ElementSpacing", " ".join(str(v) for v in attributes.get_np_array("Spacing"))),
            ("DimSize", " ".join(str(v) for v in reversed(spatial))),
        ]
        if shape[0] > 1:
            fields.append(("ElementNumberOfChannels", str(shape[0])))
        # Attribute entries ride along as MetaIO user fields, like WriteImage embeds image metadata.
        fields += [(k, str(v)) for k, v in attributes.items() if str(v) and "\n" not in str(v) and " " not in k]
        fields += [("ElementType", _MHA_ELEMENT_TYPES[self._dtype.name]), ("ElementDataFile", "LOCAL")]
        header = "".join(f"{key} = {value}\n" for key, value in fields).encode("utf-8")
        with open(self._temporary_path, "wb") as file:
            file.write(header)
            # Reserve the pixel block up front (sparse where the filesystem allows it).
            file.truncate(len(header) + int(np.prod([*spatial, shape[0]], dtype=np.int64)) * self._dtype.itemsize)
        self._memmap = np.memmap(
            self._temporary_path, dtype=self._dtype, mode="r+", offset=len(header), shape=(*spatial, shape[0])
        )

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._memmap[(*slices[1:], slices[0])] = np.moveaxis(data, 0, -1)

    def _close(self, success: bool) -> None:
        self._memmap.flush()
        del self._memmap
        if success:
            os.replace(self._temporary_path, self.path)
        else:
            os.remove(self._temporary_path)


class _OmeZarrDataStream(DataStream):
    def __init__(
        self,
        array: Any,
        store_path: Path,
        final_path: Path,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
        displacement_field: bool = False,
    ) -> None:
        self._array = array
        self._store_path = store_path
        self._final_path = final_path
        self._scale_factors = scale_factors
        self._downsample_method = downsample_method
        self._displacement_field = displacement_field
        # Running per-component bound of a streamed field. Accumulated from the regions as they are
        # written -- the only place the samples are ever all seen, since the point of this path is
        # that the field never exists whole.
        self._bound: list[float] = []

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._array[slices] = data
        if self._displacement_field:
            from konfai.utils.ome_zarr import displacement_bound

            block = displacement_bound(data)
            if len(self._bound) < len(block):
                self._bound.extend([0.0] * (len(block) - len(self._bound)))
            for component, value in enumerate(block):
                self._bound[component] = max(self._bound[component], value)

    def _close(self, success: bool) -> None:
        from konfai.utils.ome_zarr import (
            DISPLACEMENT_BOUND_ATTRIBUTE,
            append_ome_zarr_levels,
            clear_ome_zarr_cache,
            update_konfai_attributes,
        )

        if not success:
            shutil.rmtree(self._store_path, ignore_errors=True)
            return
        if self._displacement_field and self._bound:
            # Recorded before the levels are derived and before the rename, so the store is published
            # complete: a consumer that finds the entry finds its bound with it, or the entry is not
            # there at all.
            update_konfai_attributes(self._store_path, {DISPLACEMENT_BOUND_ATTRIBUTE: self._bound})
        if self._scale_factors:
            # On the temporary store, so the rename below publishes level 0 and its coarser levels in
            # one step. append_ome_zarr_levels REWRITES level 0 (ngff-zarr composes a multiscales as a
            # whole), which is why this costs a pass over what was just written rather than nothing.
            append_ome_zarr_levels(self._store_path, self._scale_factors, downsample_method=self._downsample_method)
            self._array = None
        # Move an existing store aside instead of deleting it up front, so a replaced entry stays
        # recoverable (at <name>.replaced-<pid>) until the new store is renamed into place -- a directory
        # swap is not atomic, and deleting first loses both stores on a crash in the window.
        replaced = self._final_path.exists()
        backup = self._final_path.with_name(f"{self._final_path.name}.replaced-{os.getpid()}")
        if replaced:
            shutil.rmtree(backup, ignore_errors=True)
            os.rename(self._final_path, backup)
        try:
            os.rename(self._store_path, self._final_path)
        except OSError:
            # A concurrent writer of the same entry renamed its complete, identical store into place;
            # keep it and drop ours.
            if not self._final_path.exists():
                raise
            shutil.rmtree(self._store_path, ignore_errors=True)
        if replaced:
            shutil.rmtree(backup, ignore_errors=True)
        # The reader memoises loaded stores by path, and this rename changes what that path holds.
        # A store replaced by one written through a different code path can differ down to the key
        # its level-0 array lives under, so a stale entry does not merely serve old pixels -- it
        # points at a component that is no longer there.
        clear_ome_zarr_cache()


class Dataset:
    """Filesystem or HDF5-backed dataset abstraction used across KonfAI."""

    class AbstractFile(ABC):
        @abstractmethod
        def __init__(self) -> None:
            pass

        @abstractmethod
        def __enter__(self):
            pass

        @abstractmethod
        def __exit__(self, exc_type, value, traceback):
            pass

        @abstractmethod
        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            pass

        @abstractmethod
        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            pass

        @abstractmethod
        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            pass

        @abstractmethod
        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            pass

        def open_data_stream(
            self,
            name: str,
            shape: list[int],
            dtype: np.dtype,
            attributes: Attribute,
            region_shape: list[int] | None = None,
        ) -> DataStream | None:
            """Open ``name`` for incremental region writes; ``None`` when this backend cannot."""
            return None

        @abstractmethod
        def get_names(self, group: str) -> list[str]:
            pass

        @abstractmethod
        def get_group(self) -> list[str]:
            pass

        @abstractmethod
        def is_exist(self, group: str, name: str | None = None) -> bool:
            pass

        @abstractmethod
        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            pass

    class H5File(AbstractFile):
        # Read-side HDF5 chunk cache, per opened dataset. The library default (1 MB) holds barely one
        # medical-imaging chunk, so overlapping patch reads on a chunked (compressed) store
        # re-decompress the same chunks once per patch. KonfAI writes its own h5 contiguous
        # (unaffected); this serves third-party chunked stores read through the streamed patch path.
        # nslots per the h5py guidance: a prime, well above the chunks the cache can hold.
        _READ_CHUNK_CACHE_BYTES = 128 * 1024 * 1024
        _READ_CHUNK_CACHE_SLOTS = 100003

        def __init__(self, filename: str, read: bool) -> None:
            self.h5: h5py.File | None = None
            self.filename = filename
            if not self.filename.endswith(".h5"):
                self.filename += ".h5"
            self.read = read
            self._lock: threading.RLock | None = None

        def __enter__(self):
            # A single HDF5 file cannot be opened concurrently from several threads:
            # the whole open/use/close sequence is serialised per file so that two
            # cache workers never race between the existence check and the "w"/"r+"
            # open (which would truncate each other's data).
            self._lock = _get_h5_file_lock(self.filename)
            self._lock.acquire()
            try:
                if self.read:
                    self.h5 = _h5_read_pool.get(
                        self.filename,
                        rdcc_nbytes=self._READ_CHUNK_CACHE_BYTES,
                        rdcc_nslots=self._READ_CHUNK_CACHE_SLOTS,
                    )
                else:
                    _h5_read_pool.drop(self.filename)
                    # locking=False on every KonfAI open: the HDF5 file-lock flag must agree across a file's
                    # handles, and the pooled reader (unlocked) stays open on the same file while a stream
                    # writes it -- the "invisible until finalize" read contract reads the store mid-write.
                    # Same-process races are held off by the per-file thread lock above.
                    if not os.path.exists(self.filename):
                        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
                        self.h5 = h5py.File(self.filename, "w", locking=False)
                    else:
                        self.h5 = h5py.File(self.filename, "r+", locking=False)
                    self.h5.attrs["Date"] = current_date()
            except BaseException:
                self._lock.release()
                self._lock = None
                raise
            return self.h5

        def __exit__(self, exc_type, value, traceback):
            try:
                if self.h5 is not None and not self.read:
                    self.h5.close()
            finally:
                if self._lock is not None:
                    self._lock.release()
                    self._lock = None

        def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            return data, Attribute(dict(dataset.attrs))

        def file_to_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.asarray(dataset[slices])
            return data, Attribute(dict(dataset.attrs))

        def file_to_data_statistics(
            self,
            groups: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            dataset = self._get_dataset(groups, name)
            if dataset is None:
                raise NameError(f"Dataset '{groups}/{name}' not found in '{self.filename}'.")

            axis = 1 if dataset.ndim > 1 else 0
            chunk_length = _statistics_chunk_length(dataset.shape, axis)
            state: dict[str, Any] | None = None

            for start in range(0, dataset.shape[axis], chunk_length):
                slices = [slice(None)] * dataset.ndim
                slices[axis] = slice(start, min(dataset.shape[axis], start + chunk_length))
                chunk = np.asarray(dataset[tuple(slices)])
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)

            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            if self.h5 is None:
                return
            if attributes is None:
                attributes = Attribute()
            if isinstance(data, sitk.Image):
                data, attributes_tmp = image_to_data(data)
                attributes.update(attributes_tmp)
            elif isinstance(data, sitk.Transform):
                data = np.asarray(_encode_transform_leaves(data, name, attributes))

            h5_group, name = self._resolve_group(name)
            if name in h5_group:
                del h5_group[name]

            dataset = h5_group.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
            dataset.attrs.update({k: str(v) for k, v in attributes.items()})

        def _resolve_group(self, name: str) -> tuple[h5py.Group, str]:
            """The (created) parent group a slash-qualified entry name writes into, and its leaf name."""
            h5 = cast(h5py.File, self.h5)
            h5_group: h5py.Group = h5
            if len(name.split("/")) > 1:
                group = "/".join(name.split("/")[:-1])
                if group not in h5:
                    h5.create_group(group)
                h5_group = h5[group]
            return h5_group, name.split("/")[-1]

        def open_data_stream(
            self,
            name: str,
            shape: list[int],
            dtype: np.dtype,
            attributes: Attribute,
            region_shape: list[int] | None = None,
        ) -> DataStream | None:
            if self.h5 is None:
                return None
            h5_group, name = self._resolve_group(name)
            temporary_name = f"{name}.{DataStream.temporary_suffix()}"
            dataset = h5_group.create_dataset(temporary_name, shape=tuple(shape), dtype=dtype, chunks=None)
            dataset.attrs.update({k: str(v) for k, v in attributes.items()})
            return _H5DataStream(dataset, name)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            if self.h5 is not None:
                if group in self.h5:
                    if isinstance(self.h5[group], h5py.Dataset):
                        return True
                    elif name is not None:
                        return name in self.h5[group]
                    else:
                        return False
            return False

        def get_names(self, groups: str, h5_group: h5py.Group = None) -> list[str]:
            names = []
            if h5_group is None:
                h5_group = self.h5
            group = groups.split("/")[0]
            if group == "":
                names = [
                    dataset.name.split("/")[-1]
                    for dataset in h5_group.values()
                    # ``.tmp`` keys are in-flight (or hard-kill-orphaned) DataStream writes, not entries.
                    if isinstance(dataset, h5py.Dataset) and not dataset.name.endswith(".tmp")
                ]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[k]))
            else:
                if group in h5_group:
                    names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[group]))
            return names

        def get_group(self) -> list[str]:
            return list(self.h5.keys()) if self.h5 is not None else []

        def _get_dataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset:
            if h5_group is None:
                h5_group = self.h5
            if groups != "":
                group = groups.split("/")[0]
            else:
                group = ""
            result = None
            if group == "":
                if name in h5_group:
                    result = h5_group[name]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[k])
                        if result_tmp is not None:
                            result = result_tmp
            else:
                if group in h5_group:
                    result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[group])
                    if result_tmp is not None:
                        result = result_tmp
            return result

        def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
            dataset = self._get_dataset(groups, name)
            return (
                dataset.shape,
                Attribute(dict(dataset.attrs)),
            )

    class SitkFile(AbstractFile):
        def __init__(self, filename: str, read: bool, file_format: str) -> None:
            self.filename = filename
            self.read = read
            self.file_format = file_format

        @staticmethod
        def _normalize_slices(slices: tuple[slice, ...], shape: list[int]) -> tuple[slice, ...]:
            if len(slices) != len(shape):
                raise ValueError(f"Expected {len(shape)} slices, got {len(slices)}.")

            normalized = []
            for item, size in zip(slices, shape, strict=False):
                start, stop, step = item.indices(size)
                normalized.append(slice(start, stop, step))
            return tuple(normalized)

        @staticmethod
        def _supports_direct_slice(slices: tuple[slice, ...]) -> bool:
            return all(item.step in (None, 1) for item in slices)

        @staticmethod
        @functools.cache
        def _supports_region_read(path: str) -> bool:
            """Return whether ITK can serve a region of `path` without decoding the whole volume.

            SimpleITK exposes no equivalent of ImageIOBase::CanStreamRead(), so the streaming IOs are
            mirrored here: MetaImage and NIfTI stream while their pixel data is uncompressed. A compressed
            stream is not seekable, and NrrdImageIO never streams, so both decode the whole volume for
            every region asked of them. Getting this wrong only ever costs speed, never correctness.

            Cached: the patch path asks this per read, and it opens the file to read a header.
            """
            image_io = sitk.ImageFileReader.GetImageIOFromFileName(path)
            if image_io == "MetaImageIO":
                # MetaImage announces compression in its ASCII header, ahead of ElementDataFile.
                with open(path, "rb") as file:
                    header = file.read(4096)
                return re.search(rb"CompressedData\s*=\s*True", header, re.IGNORECASE) is None
            if image_io == "NiftiImageIO":
                with open(path, "rb") as file:
                    return file.read(2) != b"\x1f\x8b"  # gzip magic: a .nii.gz stream
            return False

        def _resolve_data_path(self, name: str) -> str | None:
            base = f"{self.filename}{name}"
            for suffix in (".itk.txt", ".fcsv", ".xml", ".vtk", ".npy"):
                candidate = f"{base}{suffix}"
                if os.path.exists(candidate):
                    return candidate

            direct = f"{base}.{self.file_format}"
            if os.path.exists(direct):
                return direct

            # Skip a crashed writer's leftover temporary (``.tmp``): it is a header plus a reserved,
            # zero-filled pixel block that would read back as a plausible partial volume. Deprioritize
            # sidecar halves of paired formats: .raw/.zraw (detached MetaImage/NRRD data, unreadable
            # standalone) and .img (readable via its paired .hdr, but prefer the header half). glob order
            # is unsorted, so a bare matches[0] could hand the .raw half of a .mhd+.raw pair to the reader.
            matches = sorted(
                (candidate for candidate in glob.glob(f"{base}.*") if not candidate.endswith(".tmp")),
                key=lambda candidate: candidate.lower().endswith((".raw", ".zraw", ".img")),
            )
            return matches[0] if matches else None

        def _file_to_image_slice(self, name: str, path: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.ReadImageInformation()

            spatial_size_xyz = list(reader.GetSize())
            spatial_shape = list(reversed(spatial_size_xyz))
            data_shape = [reader.GetNumberOfComponents(), *spatial_shape]
            normalized = self._normalize_slices(slices, data_shape)

            if not self._supports_direct_slice(normalized):
                data, attributes = self.file_to_data("", name)
                return data[normalized], attributes

            if not self._supports_region_read(path):
                _warn_unstreamed_region_read(path)

            extract_index_xyz = [item.start for item in reversed(normalized[1:])]
            extract_size_xyz = [item.stop - item.start for item in reversed(normalized[1:])]
            reader.SetExtractIndex(extract_index_xyz)
            reader.SetExtractSize(extract_size_xyz)

            image = reader.Execute()
            data, attributes = image_to_data(image)
            origin = np.asarray(reader.GetOrigin(), dtype=np.float64)
            spacing = np.asarray(reader.GetSpacing(), dtype=np.float64)
            direction = np.asarray(reader.GetDirection(), dtype=np.float64).reshape(len(spacing), len(spacing))
            attributes["Origin"] = origin + direction @ (np.asarray(extract_index_xyz, dtype=np.float64) * spacing)
            return data[normalized[:1] + tuple(slice(None) for _ in normalized[1:])], attributes

        def _file_to_image_statistics(self, name: str, path: str, channels: list[int] | None) -> dict[str, float]:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.ReadImageInformation()
            data_shape = [reader.GetNumberOfComponents(), *reversed(reader.GetSize())]

            slab_length = _statistics_chunk_length(data_shape, 1)
            state: dict[str, Any] | None = None

            for start in range(0, data_shape[1], slab_length):
                slices: list[slice] = [slice(None)] * len(data_shape)
                slices[1] = slice(start, min(data_shape[1], start + slab_length))
                slab, _ = self._file_to_image_slice(name, path, tuple(slices))
                if channels is not None:
                    slab = slab[channels]
                state = _update_running_statistics(state, slab)

            return _finalize_running_statistics(state)

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            attributes = Attribute()
            if os.path.exists(f"{self.filename}{name}.itk.txt"):
                datas = _encode_transform_leaves(sitk.ReadTransform(f"{self.filename}{name}.itk.txt"), name, attributes)
                max_len = max(len(v) for v in datas)

                padded_datas = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in datas])

                data = np.asarray(padded_datas)
            elif os.path.exists(f"{self.filename}{name}.fcsv"):
                data = read_landmarks(Path(f"{self.filename}{name}.fcsv"))
            elif os.path.exists(f"{self.filename}{name}.xml"):
                with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                    root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                node = root
                while len(node):
                    node = node[-1]
                for key, value in node.attrib.items():
                    attributes[key] = value
                text = (node.text or "").strip()
                data = np.fromstring(text, sep=",", dtype=np.float64) if text else np.asarray([], dtype=np.float64)
            elif os.path.exists(f"{self.filename}{name}.vtk"):
                import vtk

                vtk_reader = vtk.vtkPolyDataReader()
                vtk_reader.SetFileName(f"{self.filename}{name}.vtk")
                vtk_reader.Update()
                data = []
                points = vtk_reader.GetOutput().GetPoints()
                num_points = points.GetNumberOfPoints()
                for i in range(num_points):
                    data.append(list(points.GetPoint(i)))
                data = np.asarray(data)
            elif os.path.exists(f"{self.filename}{name}.npy"):
                data = np.load(f"{self.filename}{name}.npy")
            else:
                # Prefer the declared format's own extension; otherwise skip a crashed writer's '.tmp' (a
                # header plus zero-reserved pixels) and deprioritize the sidecar halves of paired formats
                # (.raw/.zraw are unreadable standalone; .img reads only via its paired .hdr, so prefer the
                # header). glob order is unsorted, so without this a '.mhd'+'.raw' pair could hand the '.raw'
                # to ReadImage, or a leftover '.tmp' a zero-filled partial volume -- as in _resolve_data_path.
                direct = f"{self.filename}{name}.{self.file_format}"
                if os.path.exists(direct):
                    path = direct
                else:
                    matches = sorted(
                        (c for c in glob.glob(f"{self.filename}{name}.*") if not c.endswith(".tmp")),
                        key=lambda candidate: candidate.lower().endswith((".raw", ".zraw", ".img")),
                    )
                    if not matches:
                        raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")
                    path = matches[0]
                image = sitk.ReadImage(path)
                data, attributes_tmp = image_to_data(image)
                attributes.update(attributes_tmp)
            return data, attributes

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            path = self._resolve_data_path(name)
            if path is None:
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")

            if path.endswith(".npy"):
                data = np.load(path, mmap_mode="r")[slices]
                return np.asarray(data), Attribute()

            if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
                data, attributes = self.file_to_data(group, name)
                return data[slices], attributes

            return self._file_to_image_slice(name, path, slices)

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            path = self._resolve_data_path(name)
            if path is None:
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")

            if path.endswith(".npy"):
                data = np.load(path, mmap_mode="r")
                if channels is not None:
                    data = data[channels]
                return _finalize_running_statistics(_update_running_statistics(None, data))

            if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
                data, _ = self.file_to_data(group, name)
                if channels is not None:
                    data = data[channels]
                return _finalize_running_statistics(_update_running_statistics(None, data))

            if self._supports_region_read(path):
                return self._file_to_image_statistics(name, path, channels)

            # The whole volume lands in memory here, which the streamed path above exists to avoid. Nothing
            # better is left: a format that cannot serve a region would decode itself once per slab.
            image = sitk.ReadImage(path)
            data = sitk.GetArrayViewFromImage(image)
            if image.GetNumberOfComponentsPerPixel() == 1:
                data = np.expand_dims(data, 0)
            else:
                data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
            if channels is not None:
                data = data[channels]
            return _finalize_running_statistics(_update_running_statistics(None, data))

        def is_vtk_polydata(self, obj) -> bool:
            try:
                import vtk

                return isinstance(obj, vtk.vtkPolyData)
            except ImportError:
                return False

        def __enter__(self):
            pass

        def __exit__(self, exc_type, value, traceback):
            pass

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            if attributes is None:
                attributes = Attribute()
            os.makedirs(self.filename, exist_ok=True)
            if isinstance(data, sitk.Image):
                for k, v in attributes.items():
                    if v and len(v):
                        data.SetMetaData(k, v)
                # Publish by rename, as the streaming writer does: an existence probe answers from disk,
                # so a reader must never meet the entry while it is being written. The staging name keeps
                # the format extension (SimpleITK picks its writer from it) and is dotted so the readers'
                # own `<group>.*` glob cannot reach it; the pid keeps two writers of one entry apart.
                final = f"{self.filename}{name}.{self.file_format}"
                staging = f"{self.filename}.{name}.{os.getpid()}.tmp.{self.file_format}"
                sitk.WriteImage(data, staging)
                os.replace(staging, final)
            elif isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, f"{self.filename}{name}.itk.txt")
            elif self.is_vtk_polydata(data):
                import vtk

                vtk_writer = vtk.vtkPolyDataWriter()
                vtk_writer.SetFileName(f"{self.filename}{name}.vtk")
                vtk_writer.SetInputData(data)
                vtk_writer.Write()
            elif is_an_image(attributes):
                self.data_to_file(name, data_to_image(data, attributes), attributes)
            elif len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0:
                data = np.round(data, 4)
                write_landmarks(data, Path(f"{self.filename}{name}.fcsv"))
            elif "path" in attributes:
                if os.path.exists(f"{self.filename}{name}.xml"):
                    with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                        root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                        xml_file.close()
                else:
                    root = etree.Element(name)
                node = root
                path = attributes["path"].split(":")

                for node_name in path:
                    node_tmp = node.find(node_name)
                    if node_tmp is None:
                        node_tmp = etree.SubElement(node, node_name)
                        node.append(node_tmp)
                    node = node_tmp
                if attributes is not None:
                    for attribute_tmp in attributes.keys():
                        attribute = "_".join(attribute_tmp.split("_")[:-1])
                        if attribute != "path":
                            node.set(attribute, attributes[attribute])
                if data.size > 0:
                    node.text = ", ".join(map(str, data.flatten()))
                with open(f"{self.filename}{name}.xml", "wb") as f:
                    f.write(etree.tostring(root, pretty_print=True, encoding="utf-8"))
                    f.close()
            else:
                np.save(f"{self.filename}{name}.npy", data)

        def open_data_stream(
            self,
            name: str,
            shape: list[int],
            dtype: np.dtype,
            attributes: Attribute,
            region_shape: list[int] | None = None,
        ) -> DataStream | None:
            # Only an uncompressed local-data MetaImage is region-writable (ASCII header + flat raw
            # block); every other SimpleITK format writes the whole image in one WriteImage call.
            if self.file_format != "mha" or not is_an_image(attributes) or len(shape) < 3:
                return None
            element_dtype = np.dtype(dtype)
            if element_dtype == np.float16:
                # MetaImage has no half-float type; widen float16 to float32 (exact), as data_to_image
                # does, so streamed and whole-volume writes hold identical bytes.
                element_dtype = np.dtype(np.float32)
            if element_dtype.name not in _MHA_ELEMENT_TYPES:
                return None
            dimension = len(shape) - 1
            geometry = (("Origin", dimension), ("Spacing", dimension), ("Direction", dimension * dimension))
            if any(len(attributes.get_np_array(key)) != n for key, n in geometry):
                return None
            os.makedirs(self.filename, exist_ok=True)
            return _MhaDataStream(f"{self.filename}{name}.{self.file_format}", shape, element_dtype, attributes)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            base = f"{self.filename}{group}"
            return any(os.path.exists(base + "." + ext) for ext in SUPPORTED_EXTENSIONS)

        def get_names(self, group: str) -> list[str]:
            raise NotImplementedError()

        def get_group(self) -> list[str]:
            raise NotImplementedError()

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            attributes = Attribute()
            # Resolve the actual entry path (any image extension, not only the dataset's file_format):
            # an entry stored with a different extension must still take the header-only read below --
            # the file_to_data fallback decodes the whole volume, a hidden full load on the
            # patch-planning path.
            entry = f"{group if group is not None else ''}{name}"
            path = self._resolve_data_path(entry)
            if path is not None and not path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk", ".npy")):
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(path)
                file_reader.ReadImageInformation()
                attributes["Origin"] = np.asarray(file_reader.GetOrigin())
                attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
                attributes["Direction"] = np.asarray(file_reader.GetDirection())
                for k in file_reader.GetMetaDataKeys():
                    attributes[k] = file_reader.GetMetaData(k)
                # Reverse the spatial size for every rank (see the module-level get_infos).
                size = list(reversed(file_reader.GetSize()))
                size = [file_reader.GetNumberOfComponents(), *size]
            else:
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = list(data.shape)
            return size, attributes

    class OmeZarrFile(AbstractFile):
        """OME-NGFF backend using chunked Zarr reads for KonfAI patches.

        ``level`` selects the multiscale pyramid resolution to read (0 = full
        resolution, higher = coarser); it comes from the ``omezarr@<level>``
        dataset-spec suffix.

        ``scale_factors`` is the WRITE-side counterpart: it makes the store this backend writes a
        pyramid instead of a single level. Reading indexes a pyramid BY POSITION, so a producer that
        writes one and a consumer that asks for ``@1`` are two halves of the same contract.
        """

        def __init__(
            self,
            filename: str,
            read: bool,
            level: int = 0,
            scale_factors: list[int] | None = None,
            downsample_method: str | None = None,
        ) -> None:
            self.filename = filename if filename.endswith("/") else f"{filename}/"
            self.read = read
            self.level = level
            self.scale_factors = list(scale_factors) if scale_factors else None
            self.downsample_method = downsample_method

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback):
            return None

        def _path(self, name: str, *, writing: bool = False) -> Path:
            base = Path(self.filename) / name
            if writing:
                return Path(f"{base}.ome.zarr")
            candidates = [Path(f"{base}.ome.zarr"), Path(f"{base}.zarr"), base]
            for candidate in candidates:
                if candidate.is_dir():
                    return candidate
            raise NameError(f"OME-Zarr group '{name}' not found in '{self.filename}'.")

        @staticmethod
        def _attributes(metadata: dict[str, Any]) -> Attribute:
            return ome_zarr_attributes(metadata)

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.ome_zarr import is_displacement_field

            info_shape, _ = self.get_infos(group, name)
            data, attributes = self.file_to_data_slice(group, name, tuple(slice(None) for _ in info_shape))
            # Marked here and not in file_to_data_slice: that one is the streamed path, called once per
            # patch, and re-reading the store's metadata per patch is exactly the overhead _load_image
            # is memoised to avoid. A transform is only ever rebuilt from a whole entry.
            if is_displacement_field(self._path(name)):
                attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
            return data, attributes

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.ome_zarr import read_ome_zarr_data_slice

            path = self._path(name)
            data, metadata = read_ome_zarr_data_slice(path, slices, level=self.level)
            attributes = self._attributes(metadata)
            shape = metadata["shape"]
            normalized = tuple(slice(*item.indices(size)) for item, size in zip(slices, shape, strict=True))
            spacing = attributes.get_np_array("Spacing")
            direction = attributes.get_np_array("Direction").reshape(len(spacing), len(spacing))
            start_xyz = np.asarray([item.start for item in reversed(normalized[1:])], dtype=np.float64)
            step_xyz = np.asarray([item.step for item in reversed(normalized[1:])], dtype=np.float64)
            attributes["Origin"] = attributes.get_np_array("Origin") + direction @ (start_xyz * spacing)
            attributes["Spacing"] = spacing * step_xyz
            return data, attributes

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            shape, _ = self.get_infos(group, name)
            chunk_length = _statistics_chunk_length(shape, 1)
            state: dict[str, Any] | None = None
            for start in range(0, shape[1], chunk_length):
                slices = [slice(None)] * len(shape)
                slices[1] = slice(start, min(shape[1], start + chunk_length))
                chunk, _ = self.file_to_data_slice(group, name, tuple(slices))
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)
            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            from konfai.utils.ome_zarr import write_ome_zarr

            attributes = attributes or Attribute()
            # Two ways to say "this is a field": hand over a DisplacementFieldTransform, or mark the
            # attributes. The second exists because a producer that never builds a transform -- the
            # predictor emits arrays -- would otherwise have to wrap its output in one purely to be
            # described correctly, and a field too large to hold in memory cannot be wrapped at all.
            displacement_field = DISPLACEMENT_FIELD_ATTRIBUTE in attributes
            if sitk is not None and isinstance(data, sitk.Image):
                data, image_attributes = image_to_data(data)
                attributes.update(image_attributes)
            elif sitk is not None and isinstance(data, sitk.Transform):
                # The parametric transforms the other backends serialise (Euler, affine, B-spline) have
                # no OME-NGFF form; a displacement field does, and it is array-backed, so this backend
                # stores exactly the one kind it can store faithfully.
                data, field_attributes = displacement_field_to_data(data, name)
                attributes.update(field_attributes)
                displacement_field = True
            if not isinstance(data, np.ndarray):
                raise DatasetManagerError("OME-Zarr datasets can only store image arrays.")
            write_ome_zarr(
                self._path(name, writing=True),
                data,
                spacing=attributes.get_np_array("Spacing") if "Spacing" in attributes else None,
                origin=attributes.get_np_array("Origin") if "Origin" in attributes else None,
                attributes=dict(attributes),
                displacement_field=displacement_field,
                scale_factors=self.scale_factors,
                downsample_method=self.downsample_method,
            )

        def open_data_stream(
            self,
            name: str,
            shape: list[int],
            dtype: np.dtype,
            attributes: Attribute,
            region_shape: list[int] | None = None,
        ) -> DataStream | None:
            from konfai.utils.ome_zarr import create_ome_zarr_store

            if len(shape) not in (3, 4):
                return None
            final_path = self._path(name, writing=True)
            store_path = Path(f"{final_path}.{DataStream.temporary_suffix()}")
            array = create_ome_zarr_store(
                store_path,
                shape,
                dtype,
                spacing=attributes.get_np_array("Spacing") if "Spacing" in attributes else None,
                origin=attributes.get_np_array("Origin") if "Origin" in attributes else None,
                attributes=dict(attributes),
                displacement_field=DISPLACEMENT_FIELD_ATTRIBUTE in attributes,
                # Chunked against what the writer says it will write, capped to something a reader
                # can open. Guessing the writer's access pattern costs a read-modify-write on every
                # region whose extent straddles a chunk -- measured 1.8x on a slab sweep, paid on
                # every byte, and invisible because the bytes are correct either way.
                chunks=_store_chunks(shape, region_shape, dtype),
            )
            # The pyramid cannot be created up front -- no level exists until the last region lands --
            # so the stream derives it at finalize, on the TEMPORARY store, before the rename. That
            # order is what keeps publication atomic: a reader never sees a store whose level 0 is
            # complete but whose coarser levels are not.
            return _OmeZarrDataStream(
                array,
                store_path,
                final_path,
                self.scale_factors,
                self.downsample_method,
                DISPLACEMENT_FIELD_ATTRIBUTE in attributes,
            )

        def get_names(self, group: str) -> list[str]:
            return self.get_group()

        def get_group(self) -> list[str]:
            root = Path(self.filename)
            if not root.is_dir():
                return []
            groups = []
            for path in root.iterdir():
                if path.name.endswith(".ome.zarr"):
                    groups.append(path.name.removesuffix(".ome.zarr"))
                elif path.name.endswith(".zarr"):
                    groups.append(path.name.removesuffix(".zarr"))
            return sorted(groups)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            try:
                self._path(f"{group}/{name}" if name else group)
                return True
            except NameError:
                return False

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            from konfai.utils.ome_zarr import get_ome_zarr_info, is_displacement_field

            metadata = get_ome_zarr_info(self._path(name), level=self.level)
            axes = [str(axis).lower() for axis in metadata["axes"]]
            axis_sizes = dict(zip(axes, metadata["shape"], strict=True))
            shape = [axis_sizes.get("c", 1), *[axis_sizes[axis] for axis in ("z", "y", "x") if axis in axis_sizes]]
            metadata["shape"] = shape
            attributes = self._attributes(metadata)
            # Marked on the HEADERS path, so a field stays a field on the streamed read too --
            # file_to_data marks it only on the whole-volume read, and a store written from unmarked
            # regions is an ordinary 3-channel image. This is the once-per-case call (Dataset caches
            # it), not the per-patch one, which is why the check belongs here and not in
            # file_to_data_slice.
            if is_displacement_field(self._path(name)):
                attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
            return shape, attributes

    class DicomFile(AbstractFile):
        """DICOM series backend with header-only metadata and slice-level reads."""

        def __init__(self, filename: str, read: bool) -> None:
            self.filename = filename if filename.endswith("/") else f"{filename}/"
            self.read = read

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback):
            return None

        def _path(self, name: str) -> Path:
            return Path(self.filename) / name

        @staticmethod
        def _attributes(info: dict[str, Any]) -> Attribute:
            attributes = Attribute()
            attributes["Origin"] = np.asarray(info["origin"])
            attributes["Spacing"] = np.asarray(info["spacing"])
            attributes["Direction"] = np.asarray(info["direction"])
            attributes["SeriesInstanceUID"] = info["series_uid"]
            return attributes

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.dicom import read_dicom_series

            data, origin, spacing, direction = read_dicom_series(self._path(name))
            attributes = Attribute()
            attributes["Origin"] = origin
            attributes["Spacing"] = spacing
            attributes["Direction"] = direction
            return data, attributes

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

            path = self._path(name)
            info = dict(get_dicom_info(path))  # copy: get_dicom_info is memoised, and we update it below
            data, origin, spacing, direction = read_dicom_series_slice(
                path, slices, series_uid=info["series_uid"], info=info
            )
            info.update(origin=origin, spacing=spacing, direction=direction)
            return data, self._attributes(info)

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

            path = self._path(name)
            info = get_dicom_info(path)
            shape = info["shape"]
            state: dict[str, Any] | None = None
            for index in range(shape[1]):
                chunk, _, _, _ = read_dicom_series_slice(
                    path,
                    (slice(None), slice(index, index + 1), slice(None), slice(None)),
                    series_uid=info["series_uid"],
                    info=info,
                )
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)
            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            from konfai.utils.dicom import write_dicom_series

            attributes = attributes or Attribute()
            if sitk is not None and isinstance(data, sitk.Image):
                data, image_attributes = image_to_data(data)
                attributes.update(image_attributes)
            if not isinstance(data, np.ndarray):
                raise DatasetManagerError("DICOM datasets can only store scalar image arrays.")
            spacing = attributes.get_np_array("Spacing") if "Spacing" in attributes else np.ones(3)
            origin = attributes.get_np_array("Origin") if "Origin" in attributes else np.zeros(3)
            direction = attributes.get_np_array("Direction") if "Direction" in attributes else np.eye(3).flatten()
            metadata = {
                key: attributes[key]
                for key in ("PatientName", "PatientID", "Modality", "StudyInstanceUID", "SeriesInstanceUID")
                if key in attributes
            }
            write_dicom_series(
                self._path(name),
                data,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
            )

        def get_names(self, group: str) -> list[str]:
            return self.get_group()

        def get_group(self) -> list[str]:
            root = Path(self.filename)
            if not root.is_dir():
                return []
            return sorted(path.name for path in root.iterdir() if path.is_dir() and self.is_exist(path.name))

        def is_exist(self, group: str, name: str | None = None) -> bool:
            from konfai.utils.dicom import get_dicom_info

            try:
                get_dicom_info(self._path(f"{group}/{name}" if name else group))
                return True
            except DatasetManagerError:
                return False

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            from konfai.utils.dicom import get_dicom_info

            info = get_dicom_info(self._path(name))
            return info["shape"], self._attributes(info)

    class File:
        def __init__(
            self,
            filename: str,
            read: bool,
            file_format: str,
            level: int = 0,
            scale_factors: list[int] | None = None,
            downsample_method: str | None = None,
        ) -> None:
            self.filename = filename
            self.read = read
            self.file: Dataset.AbstractFile | None = None
            self.file_format = file_format
            self.level = level
            self.scale_factors = scale_factors
            self.downsample_method = downsample_method

        def __enter__(self) -> Dataset.AbstractFile:
            if self.file_format == "h5":
                self.file = Dataset.H5File(self.filename, self.read)
            elif self.file_format == "omezarr":
                self.file = Dataset.OmeZarrFile(
                    self.filename, self.read, self.level, self.scale_factors, self.downsample_method
                )
            elif self.file_format == "dicom":
                self.file = Dataset.DicomFile(self.filename, self.read)
            else:
                self.file = Dataset.SitkFile(self.filename + "/", self.read, self.file_format)
            self.file.__enter__()
            return self.file

        def __exit__(self, exc_type, value, traceback):
            if self.file is not None:
                self.file.__exit__(exc_type, value, traceback)

    def __init__(
        self,
        filename: str | Path,
        file_format: str,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        base_format, self.level = split_format_level(file_format)
        normalized_format = base_format.lower().removeprefix(".").replace("_", "-")
        file_format = {"ome-zarr": "omezarr", "zarr": "omezarr"}.get(normalized_format, normalized_format)
        self.filename, self.is_directory = Dataset._normalize_path(filename, file_format)
        self.file_format = file_format
        # The store backend is auto-detected from what is actually on disk (like SitkFile already probes
        # every supported extension) — an OME-Zarr / Zarr / DICOM store is a directory whose type is
        # knowable from its structure, so a ``:mha`` token never forces it to be mis-read. The token then
        # only carries the WRITE format and the OME-Zarr pyramid level (``@N``).
        detected = Dataset._detect_directory_store_format(self.filename) if self.is_directory else None
        if detected is not None:
            self.file_format = detected
        # Write-side pyramid, declared by the Save/Write that owns this destination. Refused here
        # rather than ignored: only OME-NGFF has multiple levels, so a pyramid asked of an mha or an
        # h5 is a request the format cannot serve, and silently writing one level would leave the
        # consumer's ``@1`` resolving to a level that does not exist.
        if scale_factors and self.file_format != "omezarr":
            raise DatasetManagerError(
                f"A pyramid was asked of a '{self.file_format}' destination, which has no levels.",
                "Only ':omezarr' stores levels. Drop scale_factors, or write to ':omezarr'.",
            )
        self.scale_factors = list(scale_factors) if scale_factors else None
        self.downsample_method = downsample_method
        self._names_cache: dict[str, list[str]] = {}
        self._infos_cache: dict[tuple[str, str], tuple[list[int], Attribute]] = {}

    @staticmethod
    def _normalize_path(filename: str | Path, file_format: str) -> tuple[str, bool]:
        # A single-store h5 is one file, every other backend a directory of cases: only the latter gets the
        # trailing slash that marks ``is_directory``. Keep the two in lock-step so a path never ends up a
        # directory-flagged h5 (which would write the hidden dotfile ``<dir>/.h5``). ``as_posix`` keeps the
        # separator forward on every OS, so the stored filename (and the trailing-slash marker) is the same
        # on Windows, where ``prefix / name`` would otherwise render backslashes.
        path = Path(filename).as_posix()
        if file_format != "h5" and not path.endswith("/"):
            path += "/"
        return path, path.endswith("/")

    def rebase(self, prefix: Path) -> None:
        """Prepend ``prefix`` to this dataset's path, re-deriving ``is_directory`` from the format."""
        self.filename, self.is_directory = Dataset._normalize_path(prefix / self.filename, self.file_format)

    @staticmethod
    def _detect_directory_store_format(root: str) -> str | None:
        """Detect a directory dataset's store backend from disk (``omezarr`` / ``dicom``), independent of the
        format token; ``None`` when it is plain per-file volumes (the SitkFile path, which auto-detects the
        extension itself). Probes the first case's entries only — cheap, and cases share one layout."""
        base = Path(root)
        if not base.is_dir():
            return None
        for case in sorted(base.iterdir()):
            if not case.is_dir():
                continue
            for entry in sorted(case.iterdir()):
                if entry.is_dir():
                    name = entry.name.lower()
                    if (
                        name.endswith((".ome.zarr", ".zarr"))
                        or (entry / ".zgroup").exists()
                        or (entry / "zarr.json").exists()
                    ):
                        return "omezarr"
                    files = [child for child in sorted(entry.iterdir()) if child.is_file()]
                    if any(child.suffix.lower() in (".dcm", ".dicom") for child in files):
                        return "dicom"
                    # A DICOM series is commonly exported with no extension at all, so the suffixes
                    # above miss it; the Part-10 magic at offset 128 is what identifies it then. A
                    # non-DICOM file may sort first, so probe every file, not only files[0].
                    if any(Dataset._is_dicom_file(file) for file in files):
                        return "dicom"
            return None  # first case is representative of the whole dataset's layout
        return None

    @staticmethod
    def _is_dicom_file(path: Path) -> bool:
        """Whether a file carries the DICOM Part-10 magic: ``DICM`` at offset 128."""
        try:
            with open(path, "rb") as file:
                return file.read(132)[128:132] == b"DICM"
        except OSError:
            return False

    def _exists_on_disk(self) -> bool:
        if os.path.exists(self.filename):
            return True
        return self.file_format == "h5" and os.path.exists(f"{self.filename}.h5")

    def concurrent_write_safe(self) -> bool:
        """Whether writes to different entries land in disjoint files, so a background writer may
        flush one entry while another thread writes elsewhere in the dataset.

        Mirrors the backend dispatch in ``File.__enter__``: everything that is not a single-store
        backend is a :class:`SitkFile` directory, one image file per ``(group, name)``. A single
        store (one HDF5 file, one zarr hierarchy, a DICOM series) shares handles and metadata across
        entries and must stay serial.
        """
        return self.file_format not in ("h5", "omezarr", "dicom")

    def _write_target(self, group: str, name: str) -> tuple[Dataset.File, str]:
        """The file a ``(group, name)`` write lands in and the entry name inside it, caches dropped.

        A directory dataset routes any sub-directory prefix of ``group`` into the file path (one file
        per case); a single store keeps one file and a ``group/name`` entry.
        """
        self._names_cache.clear()
        self._infos_cache.clear()
        if self.is_directory:
            os.makedirs(self.filename, exist_ok=True)
            s_group = group.split("/")
            if len(s_group) > 1:
                name = f"{'/'.join(s_group[:-1])}/{name}"
                group = s_group[-1]
            return (
                Dataset.File(
                    f"{self.filename}{name}",
                    False,
                    self.file_format,
                    self.level,
                    self.scale_factors,
                    self.downsample_method,
                ),
                group,
            )
        return (
            Dataset.File(
                self.filename, False, self.file_format, self.level, self.scale_factors, self.downsample_method
            ),
            f"{group}/{name}",
        )

    def write(
        self,
        group: str,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        target, entry = self._write_target(group, name)
        with target as file:
            file.data_to_file(entry, data, attributes if attributes is not None else Attribute())

    def can_stream_data(self, attributes: Attribute) -> bool:
        """Whether ``open_data_stream`` can serve this dataset's write format.

        H5 and OME-Zarr always can; MetaImage ``mha`` needs image geometry to write its header up
        front; every other format only writes whole volumes (use ``write``).
        """
        if self.file_format in ("h5", "omezarr"):
            return True
        return self.file_format == "mha" and is_an_image(attributes)

    def open_data_stream(
        self,
        group: str,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute | None = None,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        """Open one entry for incremental region writes.

        Returns ``None`` when the write format cannot serve region writes; the caller then assembles
        the volume and uses ``write``. The returned stream is a context manager: a clean exit
        finalizes the entry, an exception removes the partial one.

        ``region_shape`` is the extent the caller will write at a time, channels included. A store
        that chunks on it never pays a read-modify-write; a store left to guess pays one on every
        region that straddles a chunk. Declaring it is the writer's job -- it is the only party that
        knows its own access pattern.
        """
        if attributes is None:
            attributes = Attribute()
        file, entry = self._write_target(group, name)
        backend = file.__enter__()
        try:
            stream = backend.open_data_stream(entry, shape, dtype, attributes, region_shape)
        except BaseException:
            file.__exit__(None, None, None)
            raise
        if stream is None:
            file.__exit__(None, None, None)
            return None
        stream._file = file
        return stream

    def read_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        return file.file_to_data("", group)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.file_to_data(groups, name)
        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def read_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        result = file.file_to_data_slice("", group, slices)
                        return result
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.file_to_data_slice(groups, name, slices)

        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def read_data_statistics(
        self,
        groups: str,
        name: str,
        channels: list[int] | None = None,
    ) -> dict[str, Any]:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        return file.file_to_data_statistics("", group, channels)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.file_to_data_statistics(groups, name, channels)

        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def read_transform(self, group: str, name: str) -> sitk.Transform:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        transform_parameters, attribute = self.read_data(group, name)
        if DISPLACEMENT_FIELD_ATTRIBUTE in attribute:
            # A displacement field carries no parameter vector to decode: the entry IS the transform,
            # and the store said so (NGFF RFC-5). float64 is what DisplacementFieldTransform requires.
            field = sitk.Cast(data_to_image(transform_parameters, attribute), sitk.sitkVectorFloat64)
            return sitk.DisplacementFieldTransform(field)
        transforms_type = [v for k, v in attribute.items() if k.endswith(":Transform_0")]
        transforms = []
        for i, transform_type in enumerate(transforms_type):
            transform = _decode_transform(transform_type, name)
            transform.SetFixedParameters(ast.literal_eval(attribute[f"{i}:FixedParameters"]))
            transform.SetParameters(tuple(transform_parameters[i]))
            transforms.append(transform)
        return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]

    def read_image(self, group: str, name: str) -> sitk.Image:
        data, attribute = self.read_data(group, name)
        return data_to_image(data, attribute)

    def get_size(self, group: str) -> int:
        return len(self.get_names(group))

    def is_group_exist(self, group: str) -> bool:
        return self.get_size(group) > 0

    def is_dataset_exist(self, group: str, name: str) -> bool:
        """Whether ``(group, name)`` is on disk, asked of disk at the moment it is asked.

        Deliberately NOT a slice of :meth:`get_names`: that listing is a planning-time snapshot, and a
        group the run itself produces -- a ``Save`` writing into the dataset being read -- gains cases
        while it is read, through a different ``Dataset`` object and, when the loader has workers, a
        different PROCESS. No memo can be invalidated across that boundary, so membership asks the disk.
        One entry, one probe: O(1) in the number of cases, where the listing is O(N) headers, and cheaper
        than the listing it replaces.
        """
        if not self._exists_on_disk():
            # A store that is not there yet holds nothing -- the first probe of every fresh
            # destination, which a single-file backend would otherwise turn into an open error.
            return False
        if self.is_directory:
            entry_group = group.split("/")[-1]
            for sub_directory in self._get_sub_directories(group):
                base = f"{self.filename}{sub_directory}{name}"
                if not os.path.exists(f"{base}{'.h5' if self.file_format == 'h5' else ''}"):
                    continue
                with Dataset.File(base, True, self.file_format, self.level) as file:
                    if file.is_exist(entry_group):
                        return True
            return False
        with Dataset.File(self.filename, True, self.file_format, self.level) as file:
            # A wildcard group is a path pattern; only the store's own listing expands it.
            return name in file.get_names(group) if "*" in group else file.is_exist(group, name)

    def _get_sub_directories(self, groups: str, sub_directory: str = ""):
        group = groups.split("/")[0]
        sub_directories = []
        if len(groups.split("/")) == 1:
            sub_directories.append(sub_directory)
        elif group == "*":
            for k in os.listdir(f"{self.filename}{sub_directory}"):
                if not os.path.isfile(f"{self.filename}{sub_directory}{k}"):
                    sub_directories.extend(
                        self._get_sub_directories(
                            "/".join(groups.split("/")[1:]),
                            f"{sub_directory}{k}/",
                        )
                    )
        else:
            sub_directory = f"{sub_directory}{group}/"
            if os.path.exists(f"{self.filename}{sub_directory}"):
                sub_directories.extend(self._get_sub_directories("/".join(groups.split("/")[1:]), sub_directory))
        return sub_directories

    def get_names(self, groups: str, index: list[int] | None = None) -> list[str]:
        if index is None and groups in self._names_cache:
            return self._names_cache[groups]

        names = []
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}"):
                    for name in sorted(os.listdir(f"{self.filename}{sub_directory}")):
                        if os.path.isfile(f"{self.filename}{sub_directory}{name}") or self.file_format != "h5":
                            with Dataset.File(
                                f"{self.filename}{sub_directory}{name}",
                                True,
                                self.file_format,
                                self.level,
                            ) as file:
                                if file.is_exist(group):
                                    names.append(name.replace(".h5", "") if self.file_format == "h5" else name)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                names = file.get_names(groups)

        sorted_names = sorted(names)
        if index is None:
            self._names_cache[groups] = sorted_names
            return sorted_names
        return [name for i, name in enumerate(sorted_names) if i in index]

    def get_group(self) -> list[str]:
        if self.is_directory:
            if self.file_format in {"dicom", "omezarr"}:
                groups_set = set()
                root_path = Path(self.filename)
                for case_path in root_path.iterdir() if root_path.is_dir() else []:
                    if case_path.is_dir():
                        with Dataset.File(str(case_path), True, self.file_format, self.level) as dataset_file:
                            groups_set.update(dataset_file.get_group())
                return sorted(groups_set)
            groups_set = set()
            for root_dir, _, files in os.walk(self.filename):
                for file in files:
                    if file.startswith("."):  # hidden: a staging write in flight, or its crashed leftover
                        continue
                    path = Path(root_dir, file.split(".")[0]).relative_to(self.filename).as_posix()
                    parts = path.split("/")
                    if len(parts) >= 2:
                        del parts[-2]
                    groups_set.add("/".join(parts))
            groups = list(groups_set)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as dataset_file:
                groups = dataset_file.get_group()
        return list(groups)

    def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        # Memoize the header read (SITK reader + ReadImageInformation, or the HDF5/Zarr
        # metadata parse): get_infos is called once per name per group per build-pass at
        # setup, so caching it (like get_names) avoids re-parsing the same header N times.
        # Cache and hand back copies so a caller mutating the geometry cannot poison it.
        cache_key = (groups, name)
        cached = self._infos_cache.get(cache_key)
        if cached is not None:
            shape, attr = cached
            return list(shape), Attribute(attr)
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        result = file.get_infos("", group)
                        self._infos_cache[cache_key] = (list(result[0]), Attribute(result[1]))
                        return result
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                result = file.get_infos(groups, name)
                self._infos_cache[cache_key] = (list(result[0]), Attribute(result[1]))
                return result
        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def get_statistics(self, groups: str) -> dict[str, dict[str, dict[str, float | list[float]]]]:
        names = self.get_names(groups)
        stats = {}
        for name in names:
            data, attr = self.read_data(groups, name)

            min_, max_ = data.min(), data.max()
            mean_ = data.mean()
            std_ = data.std()

            p25, p50, p75 = np.percentile(data, (25, 50, 75))

            stats[name] = {
                "min": float(min_),
                "max": float(max_),
                "mean": float(mean_),
                "std": float(std_),
                "25pc": float(p25),
                "50pc": float(p50),
                "75pc": float(p75),
                "shape": list(data.shape),
                "spacing": attr.get_np_array("Spacing").tolist(),
            }

        result: dict[str, dict[str, dict[str, Any]]] = {}
        result["case"] = {}
        for name, v in stats.items():
            for metric_name, value in v.items():
                if metric_name not in result["case"]:
                    result["case"][metric_name] = {}
                result["case"][metric_name][name] = value

        result["aggregates"] = {}
        tmp: dict[str, list[float]] = {}
        for _, v in stats.items():
            for metric_name, _ in v.items():
                if metric_name not in tmp:
                    tmp[metric_name] = []
                tmp[metric_name].append(v[metric_name])
        for metric_name, values in tmp.items():
            if isinstance(values[0], float):
                result["aggregates"][metric_name] = {
                    "max": float(np.nanmax(values)) if np.any(~np.isnan(values)) else np.nan,
                    "min": float(np.nanmin(values)) if np.any(~np.isnan(values)) else np.nan,
                    "std": float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan,
                    "25pc": float(np.nanpercentile(values, 25)) if np.any(~np.isnan(values)) else np.nan,
                    "50pc": float(np.nanpercentile(values, 50)) if np.any(~np.isnan(values)) else np.nan,
                    "75pc": float(np.nanpercentile(values, 75)) if np.any(~np.isnan(values)) else np.nan,
                    "mean": float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan,
                    "count": float(np.count_nonzero(~np.isnan(values))) if np.any(~np.isnan(values)) else np.nan,
                }
            else:
                p25, p50, p75 = np.nanpercentile(values, (25, 50, 75))

                result["aggregates"][metric_name] = {
                    "max": np.nanmax(values, axis=0).tolist(),
                    "min": np.nanmin(values, axis=0).tolist(),
                    "std": np.nanstd(values, axis=0).tolist(),
                    "mean": np.nanmean(values, axis=0).tolist(),
                }
        return result
