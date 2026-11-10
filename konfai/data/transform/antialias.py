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


"""The antialiasing a downsampling ``Resample`` can apply (its ``antialias`` option), each as a library does it.

- ``triangle``: a triangle filter widened by the scale, which IS the resample (it replaces the linear read):
  torchvision's ``Resize(antialias=True)`` on a float image (torch's antialiased bilinear ``interpolate``),
  PIL's bilinear ``resize`` on a uint8 one (its 22-bit fixed point, each pass rounded to 8 bits).
- ``gaussian``: scikit-image's ``resize(anti_aliasing=True)``: a Gaussian of sigma ``max(0, (s - 1) / 2)``
  per axis, ``s`` the input extent over the output's, run before the resample, as
  ``scipy.ndimage.gaussian_filter`` (``mode='mirror'``, ``truncate=4``) runs it: each axis in float64, in
  scipy's order of terms, stored back in the input's float dtype.
"""

import numpy as np
import torch
import torch.nn.functional as F

ANTIALIAS_MODES = ("triangle", "gaussian")


def triangle_resize(tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
    """``tensor`` (``[C, H, W]``) onto ``size``, antialiased bilinearly: a uint8 image as PIL's ``resize(BILINEAR)``
    computes it (:func:`_pil_resize`), any other dtype as torchvision's ``Resize(antialias=True)`` computes a float
    tensor (torch's ``interpolate``, in float32 for an integer image), cast back."""
    if tensor.dtype == torch.uint8:
        return _pil_resize(tensor, size)
    work = tensor if tensor.dtype in (torch.float32, torch.float64) else tensor.float()
    result = F.interpolate(work[None], size=tuple(size), mode="bilinear", align_corners=False, antialias=True)[0]
    return result.to(tensor.dtype)


#: PIL's fixed point for 8-bit resampling (``Resample.c``: ``PRECISION_BITS = 32 - 8 - 2``).
_PIL_BITS = 22


def _pil_weights(n_in: int, n_out: int) -> torch.Tensor:
    """PIL's ``precompute_coeffs`` for its bilinear (triangle) filter, then ``normalize_coeffs_8bpc``: the
    ``[n_out, n_in]`` integer weights, computed in double exactly as its C does."""
    scale = n_in / n_out
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale
    weights = torch.zeros(n_out, n_in, dtype=torch.int64)
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in) - xmin
        k = [max(0.0, 1.0 - abs((x + xmin - center + 0.5) * ss)) for x in range(xmax)]
        ww = 0.0
        for w in k:
            ww += w
        for x, w in enumerate(k):
            w = w / ww if ww != 0.0 else w
            weights[xx, xmin + x] = int(0.5 + w * (1 << _PIL_BITS)) if w >= 0 else int(-0.5 + w * (1 << _PIL_BITS))
    return weights


def _pil_pass(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """One of PIL's 8-bit passes along the last axis: ``(2^21 + sum x k) >> 22``, clipped to 0..255."""
    return ((values @ weights.T + (1 << (_PIL_BITS - 1))) >> _PIL_BITS).clamp_(0, 255)


def _pil_resize(tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
    """PIL's ``Image.resize(size, BILINEAR)`` of a uint8 image: two passes, the first's result held in 8 bits,
    each skipped where its extent does not change. The vertical runs first where the height shrinks by more
    than twice what the width does (``ImagingResampleInner``'s ``horizontal_first``)."""
    height, width = (int(extent) for extent in tensor.shape[1:])

    def horizontal(values: torch.Tensor) -> torch.Tensor:
        return values if width == size[1] else _pil_pass(values, _pil_weights(width, size[1]))

    def vertical(values: torch.Tensor) -> torch.Tensor:
        if height == size[0]:
            return values
        return _pil_pass(values.transpose(1, 2), _pil_weights(height, size[0])).transpose(1, 2)

    shrunk = height - size[0]
    first, second = (vertical, horizontal) if shrunk > 0 and shrunk > (width - size[1]) * 2 else (horizontal, vertical)
    return second(first(tensor.long())).to(torch.uint8).contiguous()


def gaussian_sigmas(source: list[int], target: list[int]) -> list[float]:
    """scikit-image's ``anti_aliasing_sigma``: ``max(0, (s - 1) / 2)``, ``s`` each axis's input extent over its
    output's."""
    return [max(0.0, (n / m - 1) / 2) for n, m in zip(source, target, strict=True)]


def _kernel(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """scipy's ``_gaussian_kernel1d`` (order 0), in float64 and in NumPy, as scipy evaluates it (its polynomial
    by Horner, NumPy's exp and pairwise sum): the half from the centre outwards."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1)
    phi = np.exp(x * (x * (-0.5 / (sigma * sigma))), dtype=np.double)
    phi /= phi.sum()
    return torch.from_numpy(phi[radius:].copy())


def _mirror(index: torch.Tensor, n: int) -> torch.Tensor:
    """scipy's ``mirror`` boundary (d c b | a b c d | c b a), however far past the edge."""
    if n == 1:
        return torch.zeros_like(index)
    period = 2 * (n - 1)
    index = index.remainder(period)
    return torch.where(index >= n, period - index, index)


def gaussian_filter(tensor: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """``scipy.ndimage.gaussian_filter(x, sigmas, mode='mirror')`` over the spatial axes of ``[C, ...]``, channel by
    channel: an axis of sigma 0 is skipped, as scipy skips it. Each axis sums in float64 in scipy's order (the
    centre, then the symmetric pairs from the farthest in) and is stored in the working dtype, the input's float
    dtype (float32 for an integer input), as scipy stores each axis's output in the output array."""
    dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    result = tensor.to(dtype)
    for axis, sigma in enumerate(sigmas, start=1):
        if sigma <= 1e-15:
            continue
        weights = _kernel(sigma).to(tensor.device)
        radius = len(weights) - 1
        n = result.shape[axis]
        line = result.to(torch.float64)
        index = _mirror(torch.arange(-radius, n + radius, device=tensor.device), n)
        padded = line.index_select(axis, index)
        total = line * weights[0]
        for j in range(radius, 0, -1):
            total = total + (padded.narrow(axis, radius - j, n) + padded.narrow(axis, radius + j, n)) * weights[j]
        result = total.to(dtype)
    return result
