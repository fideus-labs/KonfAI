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

A cubic B-spline interpolates through a prefilter that is an infinite recursion along every axis, but the recursion
forgets where it started as ``0.268^n``: a region of the result is read off a window of the volume ``HALO`` voxels wider
than its taps (``window_of``), the window prefiltered alone. A value so read differs from the whole volume's by at
most 16 float64 ulps of the volume's largest magnitude, or one float32 ulp of itself: only a value under about
``2^-25`` of the largest can differ by more than its ulp (an exact 0 against the last tail of a far impulse), and a NaN
reaches only the windows that hold it, where the whole volume's recursion spreads it over every voxel.
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
#: Voxels a resample through this module writes at once, when it cuts a volume into slabs to spare its float64 copies.
SLAB_VOXELS = 1 << 24
#: Voxels a window reads past a cubic B-spline's taps. The prefilter's error at a window's edge falls as ``0.268^n``,
#: under a float64's rounding of the volume's largest value (``2^-53``, times the prefilter's gain on three axes) past
#: about 30 voxels; what lies further, a NaN or a value's last tail, the window does not see.
HALO = 32


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


def zoom_window(
    window: np.ndarray,
    starts: list[int],
    extents: list[int],
    shape: tuple[int, ...],
    region: tuple[slice, ...],
    order: int,
    corners: bool = False,
) -> np.ndarray:
    """``region`` of ``scipy.ndimage.zoom(volume, ..., order=order, mode='nearest', grid_mode=not corners)`` of the
    ``extents``-long volume, in float64, read off ``window``, its part from ``starts`` on
    (:func:`window_of`). At order 3 the window is prefiltered as scipy prefilters the volume: padded by 12 of its edge
    voxels where it meets the volume's edges (``spline_filter(np.pad(volume, 12, 'edge'), 3, mode='nearest')``)."""
    if order not in (0, 1, 3):
        raise ValueError(f"spline order {order}: 0, 1 or 3")
    source = np.asarray(window, dtype=np.float64)
    pads = [(0, 0)] * source.ndim
    if order == 3:
        pads = [
            (_NPAD if start == 0 else 0, _NPAD if start + size == extent else 0)
            for start, size, extent in zip(starts, source.shape, extents, strict=True)
        ]
        source = np.pad(source, pads, mode="edge")
        for axis in range(source.ndim):
            _prefilter_axis(np.moveaxis(source, axis, 0))
    shift = float(_NPAD) if order == 3 else 0.0
    coordinates = [
        _zoom_coordinates(n, m, corners, shift)[part] for n, m, part in zip(extents, shape, region, strict=True)
    ]
    # The window's first voxel in the padded volume's indices, and the padded volume's extents.
    first = [int(start + shift) - before for start, (before, _) in zip(starts, pads, strict=True)]
    return _sum_taps(source, coordinates, order, first, [int(n + 2 * shift) for n in extents])


def window_of(
    extents: list[int],
    shape: tuple[int, ...],
    region: tuple[slice, ...],
    order: int,
    corners: bool = False,
    axis: int | None = None,
) -> list[slice]:
    """The part of a volume of ``extents`` that ``region`` of its resampling onto ``shape`` reads: the taps of the
    zoom and, at order 3, ``HALO`` voxels past them; with ``axis`` resampled apart (:func:`resample_region`), the
    slices picked along it, each whole."""
    window = []
    for dim, (n, m, part) in enumerate(zip(extents, shape, region, strict=True)):
        if axis is not None:
            picked = _picks(n, m, part)
            window.append(slice(int(picked.min()), int(picked.max()) + 1) if dim == axis else slice(0, n))
            continue
        first, stop, _ = part.indices(m)
        # The first tap of each end, as the sum reads it, clamped to the volume as mode 'nearest' clamps it.
        starts, _ = _weights(_zoom_coordinates(n, m, corners, 0.0)[[first, stop - 1]], order)
        low, high = np.clip([starts[0], starts[1] + order], 0, n - 1)
        reach = HALO if order == 3 else 0
        window.append(slice(max(0, int(low) - reach), min(n, int(high) + 1 + reach)))
    return window


def _picks(n: int, m: int, part: slice) -> np.ndarray:
    """The slice of ``n`` each of ``m`` (``part`` of them) takes along an axis resampled apart, its nearest one."""
    cc = float(n) / m * (np.arange(m)[part] + 0.5) - 0.5
    return np.clip(np.floor(cc + 0.5).astype(np.int64), 0, n - 1)


def zoom_region(
    window: np.ndarray, starts: list[int], extents: list[int], shape: tuple[int, ...], region: tuple[slice, ...]
) -> np.ndarray:
    """``region`` of ``scipy.ndimage.zoom(volume, ..., order=1, mode='nearest', grid_mode=False)`` onto ``shape``, read
    off ``window``, the part of the ``extents``-long volume from ``starts`` on that holds its taps: the zoom grid's samples,
    to the bit, without the whole volume. The zoom runs on the volume as a NIfTI array holds it, (x, y, z), the
    reverse of these axes (z, y, x): the order of the sum and its rounding are that volume's."""
    flip = tuple(reversed(range(window.ndim)))
    xyz = zoom_window(window.transpose(flip), starts[::-1], extents[::-1], shape[::-1], region[::-1], 1, corners=True)
    return xyz.transpose(flip)


def value_range(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The range scikit-image clips a resize to: the input's least and greatest values, NaN left out."""
    low, high = (np.nanmin, np.nanmax) if np.isnan(np.min(array)) else (np.min, np.max)
    return np.asarray(low(array)), np.asarray(high(array))


def separate_axis(current_zyx: list[float], new_zyx: list[float]) -> int | None:
    """The axis resampled apart, or None: on the first of the two spacings (array order) whose coarsest axis is
    over ``ANISO_THRESHOLD`` times its finest, that axis when it is the only coarsest one."""
    for spacing in (current_zyx, new_zyx):
        if max(spacing) / min(spacing) > ANISO_THRESHOLD:
            axes = np.where(max(spacing) / np.array(spacing) == 1)[0]
            return int(axes[0]) if len(axes) == 1 else None
    return None


def resample_region(
    window: np.ndarray,
    starts: list[int],
    extents: list[int],
    shape: tuple[int, ...],
    region: tuple[slice, ...],
    order: int,
    corners: bool = False,
    axis: int | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """``region`` of one channel resampled onto ``shape``, read off ``window`` (:func:`window_of`): :func:`zoom_window`,
    clipped to ``bounds`` (the whole channel's range, where scikit-image's ``resize`` clips). With ``axis`` (the
    faces grid), each slice the region picks across it is resized whole in plane, clipped to its own range."""
    if axis is None:
        out = zoom_window(window, starts, extents, shape, region, order, corners)
        if bounds is not None and order > 0:
            np.clip(out, *bounds, out=out)
        return out
    plane = tuple(extent for dim, extent in enumerate(shape) if dim != axis)
    in_plane = tuple(part for dim, part in enumerate(region) if dim != axis)
    picks = _picks(extents[axis], shape[axis], region[axis]) - starts[axis]
    slices = np.moveaxis(np.asarray(window, dtype=np.float64), axis, 0)
    origin, extent = [0] * len(plane), list(slices.shape[1:])
    resized = {
        int(i): resample_region(slices[i], origin, extent, plane, in_plane, order, bounds=value_range(slices[i]))
        for i in np.unique(picks)
    }
    return np.moveaxis(np.stack([resized[int(i)] for i in picks]), 0, axis)
