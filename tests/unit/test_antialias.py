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

"""``Resample``'s ``antialias``: each mode bit for bit what its library computes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data.transform import Resample
from konfai.data.transform.antialias import gaussian_filter, gaussian_sigmas
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError
from oracle_support import assert_same_as_scipy, geometry


def _header(rank: int) -> Attribute:
    return geometry([0.0] * rank, [1.0] * rank)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_triangle_antialiasing_of_a_float_image_is_torchvisions_resize(dtype) -> None:
    functional = pytest.importorskip("torchvision.transforms.functional")
    image = torch.from_numpy(np.random.default_rng(0).integers(0, 256, size=(3, 1010, 1220)).astype(np.uint8))
    image = image.to(dtype)
    stage = Resample(shape=[352, 352], interpolation="linear", antialias="triangle")
    header = _header(2)
    got = stage("case", image.clone(), header)
    expected = functional.resize(image, [352, 352], antialias=True)
    assert got.dtype == dtype
    assert torch.equal(got, expected)
    # The grid is the one a plain resample onto the shape writes, which the inverse undoes.
    plain = _header(2)
    Resample(shape=[352, 352], interpolation="linear")("case", image.float(), plain)
    assert np.array_equal(header.get_np_array("Spacing"), plain.get_np_array("Spacing"))
    back = stage.inverse("case", got.float(), header)
    assert tuple(back.shape) == tuple(image.shape)


def test_a_triangle_antialiasing_of_a_uint8_image_is_pils_bilinear_resize() -> None:
    pil = pytest.importorskip("PIL.Image")
    if int(pil.__version__.split(".")[0]) < 13:  # type: ignore[attr-defined]
        # Pillow 13 runs the vertical pass first where the height shrinks more than twice what the width does; 12
        # always ran the horizontal one first, and each pass rounds to uint8. The stage follows the current Pillow.
        pytest.skip(f"Pillow {pil.__version__} orders its passes as Pillow 12 did")  # type: ignore[attr-defined]
    # Down on both axes, down on one and up on the other, one axis kept (PIL skips its pass), and the shapes
    # either pass order is taken on; then random ones.
    rng = np.random.default_rng(1)
    fixed = [((605, 700), (512, 512)), ((1010, 1220), (352, 352)), ((605, 700), (300, 900)), ((64, 80), (64, 33))]
    drawn = [(tuple(rng.integers(5, 80, 2)), tuple(rng.integers(5, 80, 2))) for _ in range(100)]
    for shape, size in fixed + drawn:
        shape, size = tuple(int(n) for n in shape), tuple(int(n) for n in size)
        array = rng.integers(0, 256, size=(*shape, 3)).astype(np.uint8)
        expected = np.asarray(pil.fromarray(array).resize(size[::-1], pil.BILINEAR))
        got = Resample(shape=list(size), interpolation="linear", antialias="triangle")(
            "case", torch.from_numpy(array).permute(2, 0, 1).contiguous(), _header(2)
        )
        assert got.dtype == torch.uint8
        assert np.array_equal(got.permute(1, 2, 0).numpy(), expected), (shape, size)


@pytest.mark.parametrize("shape", [(40, 37), (9, 50, 61)])
def test_a_gaussian_filter_is_scipys(shape) -> None:
    ndimage = pytest.importorskip("scipy.ndimage")
    volume = np.random.default_rng(2).normal(size=shape).astype(np.float32)
    sigmas = gaussian_sigmas(list(shape), [max(1, n // 3) for n in shape])
    got = gaussian_filter(torch.from_numpy(volume)[None], sigmas)[0].numpy()
    assert_same_as_scipy(got, ndimage.gaussian_filter(volume, sigmas, mode="mirror"))
    # Past the edge more than once (a sigma wider than the axis) and in float64.
    wide = np.random.default_rng(3).normal(size=(5, 30))
    got = gaussian_filter(torch.from_numpy(wide)[None], [3.0, 0.0])[0].numpy()
    assert_same_as_scipy(got, ndimage.gaussian_filter(wide, [3.0, 0.0], mode="mirror"))


def test_a_gaussian_antialiasing_filters_as_scikit_image_before_the_resample() -> None:
    transform = pytest.importorskip("skimage.transform")
    ndimage = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(4).random(size=(120, 90)).astype(np.float32)
    # scikit-image's own prefilter: its sigma, scipy's filter, then its order-1 zoom.
    sigma = np.maximum(0, (np.divide(image.shape, (40, 30)) - 1) / 2)
    assert np.array_equal(gaussian_sigmas([120, 90], [40, 30]), sigma)
    prefiltered = ndimage.gaussian_filter(image, sigma, mode="mirror")
    plain = Resample(shape=[40, 30], interpolation="linear")("case", torch.from_numpy(prefiltered)[None], _header(2))
    got = Resample(shape=[40, 30], interpolation="linear", antialias="gaussian")(
        "case", torch.from_numpy(image)[None], _header(2)
    )
    assert torch.equal(got, plain)
    # Against resize itself: inside the border its order-1 zoom reads the samples KonfAI's linear reads.
    expected = transform.resize(image, (40, 30), order=1, anti_aliasing=True)
    assert np.allclose(got[0, 1:-1, 1:-1].numpy(), expected[1:-1, 1:-1], atol=1e-6)


def test_an_antialiasing_reads_the_whole_volume_and_refuses_what_is_not_its_librarys() -> None:
    stage = Resample(shape=[8, 8], antialias="gaussian")
    assert stage.patch_locality(_header(2)).kind.name == "WHOLE_VOLUME"
    with pytest.raises(TransformError, match="unknown antialias"):
        Resample(shape=[8, 8], antialias="lanczos")
    with pytest.raises(TransformError, match="triangle"):
        Resample(spacing=[2.0, 2.0], antialias="triangle")
    with pytest.raises(TransformError, match="triangle"):
        Resample(shape=[8, 8], interpolation="nearest", antialias="triangle")
    with pytest.raises(TransformError, match="2D image"):
        Resample(shape=[4, 8, 8], antialias="triangle")("case", torch.zeros(1, 8, 16, 16), _header(3))
