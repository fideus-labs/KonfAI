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

"""Shared fixtures for the IMPACT-Reg app suite.

``unit`` tests stub the KonfAI runtime (no GPU, no network): a stubbed ``_infer_preset`` writes a fake
``Moved.mha`` / ``DVF.mha`` so the orchestration logic (single-preset reuse, ensemble averaging, mask
sentinels) can be checked in isolation. ``integration`` tests run the real FireANTs presets on a GPU and
are gated behind the ``KONFAI_IMPACTREG_REPO`` bundle + ``fireants`` + CUDA (see ``requires_fireants``)."""

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

# Make the ``impact_reg_konfai`` source package importable when running from a checkout (before install).
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))


def _sphere(shape: tuple[int, int, int], center, radius: float, value: float) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2
    return mask.astype(np.float32) * value


def _structured_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """A rich, non-symmetric intensity volume so registration has signal everywhere, not just one blob."""
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    vol = 40.0 + 25.0 * np.sin(x / 9.0) * np.cos(y / 11.0) + 0.15 * z
    for center, radius, value in [
        ((shape[0] // 3, shape[1] // 3, shape[2] // 3), shape[0] / 8, 220.0),
        ((2 * shape[0] // 3, 2 * shape[1] // 3, 2 * shape[2] // 3), shape[0] / 6, 300.0),
        ((3 * shape[0] // 4, shape[1] // 4, 2 * shape[2] // 3), shape[0] / 10, 160.0),
    ]:
        vol = np.maximum(vol, _sphere(shape, center, radius, value))
    return vol.astype(np.float32)


@pytest.fixture
def make_reg_pair(tmp_path: Path) -> Callable[..., tuple[Path, Path, float]]:
    """Build a fixed/moving pair where moving = fixed warped by a KNOWN smooth low-frequency field.

    Returns ``(fixed_path, moving_path, baseline_ncc)``; a correct registration must raise the NCC above
    ``baseline_ncc``. Structured content + a smooth (not piecewise) warp make the recovered field seamless,
    so it doubles as the patch-tiling + overlap-blending witness.
    """
    from scipy.ndimage import map_coordinates

    def build(side: int = 96, amplitude: float = 4.0) -> tuple[Path, Path, float]:
        shape = (side, side, side)
        fixed = _structured_volume(shape)
        z, y, x = np.mgrid[0:side, 0:side, 0:side].astype(np.float32)
        dz = amplitude * np.sin(x / 18.0) * np.cos(y / 20.0)
        dy = amplitude * np.sin(y / 16.0) * np.cos(z / 22.0)
        dx = amplitude * np.cos(x / 14.0) * np.sin(z / 19.0)
        moving = map_coordinates(fixed, [z + dz, y + dy, x + dx], order=1, mode="nearest").astype(np.float32)

        paths = []
        for name, arr in [("fixed", fixed), ("moving", moving)]:
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((1.0, 1.0, 1.0))
            path = tmp_path / f"{name}.mha"
            sitk.WriteImage(img, str(path))
            paths.append(path)

        def ncc(a: np.ndarray, b: np.ndarray) -> float:
            a, b = a - a.mean(), b - b.mean()
            return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))

        return paths[0], paths[1], ncc(moving, fixed)

    return build


@pytest.fixture
def write_preset_output(tmp_path: Path) -> Callable[[Path, np.ndarray | None], tuple[Path, Path]]:
    """Write a fake preset inference output (``Moved.mha`` + a 3-component ``DVF.mha``) under ``root``.

    Used by unit tests to stand in for a real (GPU) preset run so ``_find_output`` + the ensemble logic
    can be exercised without the KonfAI runtime.
    """

    def make(root: Path, dvf_value: np.ndarray | None = None, shape: tuple[int, int, int] = (8, 8, 8)):
        root.mkdir(parents=True, exist_ok=True)
        moved = sitk.GetImageFromArray(np.full(shape, 5.0, dtype=np.float32))
        sitk.WriteImage(moved, str(root / "Moved.mha"))
        field = np.zeros((*shape, 3), dtype=np.float32)
        if dvf_value is not None:
            field[...] = dvf_value
        dvf = sitk.GetImageFromArray(field, isVector=True)
        sitk.WriteImage(dvf, str(root / "DVF.mha"))
        return root / "Moved.mha", root / "DVF.mha"

    return make
