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

"""End-to-end: ``register`` with real presets, on a pair whose moving lies on a grid of its own (another
spacing, another origin), as real pairs come.

- On the GPU, FireANTs SyN in **tiles** (``max_voxels`` of 64^3 on a 96^3 pair: the global pass resampled, then
  native tiles): the registration must recover the known smooth warp (NCC up) and the blended field must show no
  seam on any plane. A pair on one grid could not show a moving tile registered against another region of the
  fixed.
- On the CPU, Generic_Rigid_BSpline on the whole pair: the contract the Slicer module reads, ``P000/Transform.h5``
  and ``P000/Moved.mha`` on the fixed grid, and ``register.json`` naming the inputs of the case.

Gated: the presets are external apps, not shipped in this repo; point ``KONFAI_IMPACTREG_REPO`` at a local
directory of preset folders. The GPU test needs CUDA and ``fireants``; the CPU test an installed elastix
(``KONFAI_ELASTIX_DIR`` or KonfAI's cache), so that it never downloads one."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai import impact_reg as reg

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO = os.environ.get("KONFAI_IMPACTREG_REPO", "")


def _skip_reasons(preset: str, gpu: bool) -> list[str]:
    reasons = []
    if not (Path(_REPO) / preset / "app.json").is_file():
        reasons.append(f"set KONFAI_IMPACTREG_REPO to a local preset directory holding {preset}")
    if gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                reasons.append("no CUDA device")
        except ImportError:
            reasons.append("torch missing")
        try:
            import fireants  # noqa: F401
        except ImportError:
            reasons.append("fireants not installed")
    elif not os.environ.get("KONFAI_ELASTIX_DIR") and not (Path.home() / ".cache/konfai/elastix-impact").is_dir():
        reasons.append("no elastix installed (KONFAI_ELASTIX_DIR)")
    return reasons


def _requires(preset: str, gpu: bool) -> pytest.MarkDecorator:
    reasons = _skip_reasons(preset, gpu)
    return pytest.mark.skipif(bool(reasons), reason="; ".join(reasons))


def _arr(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))


def _field(transform: Path, fixed: Path) -> np.ndarray:
    """The transform as a displacement field sampled on the fixed grid, ``(Z, Y, X, 3)``."""
    grid = sitk.ReadImage(str(fixed))
    field = sitk.TransformToDisplacementField(
        sitk.ReadTransform(str(transform)),
        sitk.sitkVectorFloat64,
        grid.GetSize(),
        grid.GetOrigin(),
        grid.GetSpacing(),
        grid.GetDirection(),
    )
    return sitk.GetArrayFromImage(field)


@pytest.mark.gpu
@_requires("FireANTs_SyN", gpu=True)
def test_a_patched_registration_of_a_pair_on_two_grids_is_seamless(make_reg_pair, tmp_path: Path) -> None:
    fixed, moving, baseline = make_reg_pair(side=96, amplitude=4.0, moving_spacing=1.25, moving_origin=(3.0, -2.0, 4.0))

    out = tmp_path / "Output"
    reg.ImpactRegKonfAIApp().register(
        ["FireANTs_SyN"],
        [fixed],
        [moving],
        output=out,
        gpu=[0],
        quiet=True,
        max_voxels=64**3,
        config_overrides=["affine_iterations=[100, 50, 25]", "deformable_iterations=[100, 50, 25]"],
    )

    fixed_a, moved_a = _arr(fixed), _arr(out / "P000" / "Moved.mha")
    # (1) the moved image lies on the fixed grid
    assert moved_a.shape == fixed_a.shape == (96, 96, 96)
    # (2) each patch of the moving was registered against its own region of the fixed: the NCC improves
    after = _ncc(moved_a, fixed_a)
    assert after > baseline + 0.05, f"NCC {baseline:.3f} -> {after:.3f} did not improve enough"
    # (3) no seam on any axis: the blend (a partition of unity) leaves no gradient spike at a plane where one tile
    #     ends and the next begins, wherever KonfAI cut them.
    dvf = _field(out / "P000" / "Transform.h5", fixed)
    for axis in range(3):
        grad = np.abs(np.diff(dvf, axis=axis)).sum(-1)
        global_mean = float(grad.mean())
        seam_mean = max(float(grad.take(plane, axis=axis).mean()) for plane in range(95))
        assert seam_mean < 6.0 * global_mean, f"axis {axis} patch seam: {seam_mean:.3f} vs global {global_mean:.3f}"


@_requires("Generic_Rigid_BSpline", gpu=False)
def test_register_writes_what_slicer_reads_for_a_pair_on_two_grids(make_reg_pair, tmp_path: Path) -> None:
    fixed, moving, baseline = make_reg_pair(side=64, amplitude=3.0, moving_spacing=1.25, moving_origin=(3.0, -2.0, 4.0))

    out = tmp_path / "Output"
    reg.ImpactRegKonfAIApp().register(["Generic_Rigid_BSpline"], [fixed], [moving], output=out, cpu=1, quiet=True)

    fixed_a, moved_a = _arr(fixed), _arr(out / "P000" / "Moved.mha")
    assert (out / "P000" / "Transform.h5").is_file() and moved_a.shape == fixed_a.shape
    assert _ncc(moved_a, fixed_a) > baseline
    record = json.loads((out / "register.json").read_text(encoding="utf-8"))
    assert record["cases"]["P000"]["moving"] == str(moving) and record["cases"]["P000"]["transform"] == str(
        Path("P000/Transform.h5")
    )
