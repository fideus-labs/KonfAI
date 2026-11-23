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

"""scipy's ``ndimage.zoom`` (order 0, 1 and 3, ``mode='nearest'``) and scikit-image's ``resize``, in numpy, to the bit.

The faces grid resamples with ``skimage.transform.resize`` (order 3, edge, no antialiasing, clipped to the input's
range), which is ``scipy.ndimage.zoom`` with ``grid_mode=True``; the zoom grid is ``scipy.ndimage.zoom`` with its
corners aligned. Neither library is a KonfAI dependency, so their arithmetic is written out here in the order their C
code runs it (``ni_splines.c``, ``NI_ZoomShift``): the same float64 operations in the same order give the same bits.

A cubic B-spline interpolates through a prefilter that is an infinite recursion along every axis: a voxel's value
depends on the whole line, so this runs on the whole volume.
"""

import itertools
import math

import numpy as np

#: The pole of the cubic B-spline prefilter, ``sqrt(3) - 2``, as scipy writes it.
_POLE = -0.267949192431122706472553658494127633
#: Voxels scipy pads a ``mode='nearest'`` prefilter with (``_prepad_for_spline_filter``).
_NPAD = 12
#: Past this ratio of an axis's spacing to the finest, that coarse axis is resampled apart, by nearest neighbour.
ANISO_THRESHOLD = 3.0
#: Output voxels per chunk of the tap sum: the sum is per voxel, so a chunk only bounds the temporaries.
_CHUNK = 1 << 22


def _prefilter_axis(c: np.ndarray) -> None:
    """``apply_filter`` along axis 0, in place, mode ``nearest`` (scipy's reflect initialisation)."""
    n = c.shape[0]
    if n < 2:
        return
    z = _POLE
    c *= (1.0 - z) * (1.0 - 1.0 / z)
    # _init_causal_reflect: c[0] is updated in place, so the last term reads the running sum.
    z_i = z
    z_n = math.pow(z, n)
    c0 = c[0].copy()
    c[0] = c[0] + z_n * c[n - 1]
    for i in range(1, n):
        c[0] += z_i * (c[i] + z_n * c[n - 1 - i])
        z_i *= z
    c[0] *= z / (1 - z_n * z_n)
    c[0] += c0
    for i in range(1, n):
        c[i] += z * c[i - 1]
    c[n - 1] *= z / (z - 1)
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])


def _prefiltered(array: np.ndarray) -> np.ndarray:
    """``spline_filter(np.pad(array, 12, 'edge'), 3, mode='nearest')``: axis by axis, in float64."""
    padded = np.pad(np.asarray(array, dtype=np.float64), _NPAD, mode="edge")
    for axis in range(padded.ndim):
        _prefilter_axis(np.moveaxis(padded, axis, 0))
    return padded


def _weights(cc: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Per output index: the first tap and the ``order + 1`` weights (``get_spline_interpolation_weights``)."""
    if order == 0:
        return np.floor(cc + 0.5).astype(np.int64), np.ones((cc.size, 1))
    floor = np.floor(cc)
    x = cc - floor
    weights = np.empty((cc.size, order + 1))
    if order == 1:
        weights[:, 0] = 1.0 - x
    else:  # 3
        y, z = x, 1.0 - x
        weights[:, 1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
        weights[:, 2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0
        weights[:, 0] = z * z * z / 6.0
    last = np.ones(cc.size)
    for tap in range(order):
        last -= weights[:, tap]
    weights[:, order] = last
    return floor.astype(np.int64) - order // 2, weights


def _sum_taps(
    source: np.ndarray,
    coordinates: list[np.ndarray],
    order: int,
    starts: list[int] | None = None,
    extents: list[int] | None = None,
) -> np.ndarray:
    """``NI_ZoomShift``'s sum: per output voxel, the taps in C order, each coefficient times its weights axis by
    axis, accumulated from 0. ``coordinates`` are continuous indices into ``source`` per axis; a tap past an edge
    reads the edge (mode ``nearest``). ``source`` may be a window ``starts`` into a volume of ``extents``: the
    coordinates and the edges stay the volume's."""
    starts = starts or [0] * source.ndim
    extents = extents or list(source.shape)
    taps = []
    for axis, cc in enumerate(coordinates):
        start, weights = _weights(cc, order)
        index = np.clip(start[:, None] + np.arange(order + 1), 0, extents[axis] - 1) - starts[axis]
        taps.append((index, weights))
    out = np.empty(tuple(len(cc) for cc in coordinates))
    rows = max(1, _CHUNK // max(1, math.prod(out.shape[1:])))
    for first in range(0, out.shape[0], rows):
        part = slice(first, first + rows)
        total = np.zeros(out[part].shape)
        for corner in itertools.product(range(order + 1), repeat=source.ndim):
            picked = [taps[0][0][part, corner[0]], *(taps[axis][0][:, corner[axis]] for axis in range(1, source.ndim))]
            coefficient = source[np.ix_(*picked)]
            if order > 0:
                for axis in range(source.ndim):
                    weight = taps[axis][1][part if axis == 0 else slice(None), corner[axis]]
                    coefficient *= weight.reshape([-1 if a == axis else 1 for a in range(source.ndim)])
            total += coefficient
        out[part] = total
    return out


def _zoom_coordinates(n_in: int, n_out: int, corners: bool, shift: float) -> np.ndarray:
    """``NI_ZoomShift``'s source coordinate of each output index: ``(k + 0.5) n / m - 0.5`` (``grid_mode``, the
    extents aligned) or ``k (n - 1) / (m - 1)`` (the corners), plus the prefilter's padding."""
    if corners:
        zoom = (n_in - 1) / (n_out - 1) if n_out > 1 else 1.0
        cc = np.arange(n_out, dtype=np.float64) * zoom
    else:
        zoom = n_in / n_out
        cc = (np.arange(n_out, dtype=np.float64) + 0.5) * zoom - 0.5
    return cc + shift


def zoom(array: np.ndarray, shape: tuple[int, ...], order: int, corners: bool = False) -> np.ndarray:
    """``scipy.ndimage.zoom(array, ..., order=order, mode='nearest', grid_mode=not corners)`` onto ``shape``, in
    float64. Order 3 runs scipy's prefilter, orders 0 and 1 read the voxels as they are."""
    if order not in (0, 1, 3):
        raise ValueError(f"spline order {order}: 0, 1 or 3")
    source = _prefiltered(array) if order == 3 else np.asarray(array, dtype=np.float64)
    shift = float(_NPAD) if order == 3 else 0.0
    coordinates = [_zoom_coordinates(n, m, corners, shift) for n, m in zip(array.shape, shape, strict=True)]
    return _sum_taps(source, coordinates, order)


def zoom_region(
    window: np.ndarray, starts: list[int], extents: list[int], shape: tuple[int, ...], region: tuple[slice, ...]
) -> np.ndarray:
    """``region`` of ``scipy.ndimage.zoom(volume, ..., order=1, mode='nearest', grid_mode=False)`` onto ``shape``, read
    off ``window``, the part of the ``extents``-long volume from ``starts`` on that holds its taps: the zoom grid's samples,
    to the bit, without the whole volume. The zoom runs on the volume as a NIfTI array holds it, (x, y, z), the
    reverse of these axes (z, y, x): the order of the sum and its rounding are that volume's."""
    coordinates = [_zoom_coordinates(n, m, True, 0.0)[part] for n, m, part in zip(extents, shape, region, strict=True)]
    flipped = np.asarray(window, dtype=np.float64).transpose(*reversed(range(window.ndim)))
    return _sum_taps(flipped, coordinates[::-1], 1, starts[::-1], extents[::-1]).transpose(
        *reversed(range(window.ndim))
    )


def resize(array: np.ndarray, shape: tuple[int, ...], order: int) -> np.ndarray:
    """``skimage.transform.resize(array, shape, order, mode='edge', anti_aliasing=False)`` on a float64 array:
    the extents aligned, then clipped to the input's range."""
    out = zoom(array, shape, order)
    if order > 0:
        low, high = (np.nanmin, np.nanmax) if np.isnan(np.min(array)) else (np.min, np.max)
        np.clip(out, np.asarray(low(array)), np.asarray(high(array)), out=out)
    return out


def separate_axis(current_zyx: list[float], new_zyx: list[float]) -> int | None:
    """The axis resampled apart, or None: on the first of the two spacings (array order) whose coarsest axis is
    over ``ANISO_THRESHOLD`` times its finest, that axis when it is the only coarsest one."""
    for spacing in (current_zyx, new_zyx):
        if max(spacing) / min(spacing) > ANISO_THRESHOLD:
            axes = np.where(max(spacing) / np.array(spacing) == 1)[0]
            return int(axes[0]) if len(axes) == 1 else None
    return None


def resample_faces(data: np.ndarray, shape: tuple[int, ...], order: int, axis: int | None) -> np.ndarray:
    """The faces grid's resampling of an image: ``data`` is ``(C, *spatial)``, each channel resized by
    scikit-image's ``resize`` at ``order``, the result float64 (the caller casts it to ``data.dtype``). With
    ``axis``, every slice across it is resized alone, then that axis is picked by nearest neighbour."""
    if tuple(data.shape[1:]) == tuple(shape):
        return data
    data = data.astype(np.float64, copy=False)
    out = np.empty((data.shape[0], *shape))
    for channel in range(data.shape[0]):
        if axis is None:
            out[channel] = resize(data[channel], tuple(shape), order)
            continue
        moved = np.moveaxis(data[channel], axis, 0)
        plane = tuple(extent for a, extent in enumerate(shape) if a != axis)
        here = np.stack([resize(moved[i], plane, order) for i in range(moved.shape[0])])
        if here.shape[0] != shape[axis]:
            cc = float(here.shape[0]) / shape[axis] * (np.arange(shape[axis]) + 0.5) - 0.5
            here = here[np.clip(np.floor(cc + 0.5).astype(np.int64), 0, here.shape[0] - 1)]
        out[channel] = np.moveaxis(here, 0, axis)
    return out
