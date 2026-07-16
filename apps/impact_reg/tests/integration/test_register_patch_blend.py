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

"""End-to-end: register a LARGE volume with the real FireANTs SyN preset under **patch tiling +
overlap blending** (the core lazy/patch invariant, applied to registration).

The volume (96^3) is larger than the patch (64^3), so KonfAI tiles it into overlapping patches, registers
each, and reassembles the moved image / displacement field with the ``Cosinus`` partition-of-unity window.
The test asserts the registration recovers the known smooth warp (NCC up) *and* that the blend leaves no
seam at the patch boundaries (the reassembled DVF stays smooth across the tiling planes).

Gated: needs a CUDA GPU, the ``fireants`` package, and a **local** IMPACT-Reg bundle pointed at by the
``KONFAI_IMPACTREG_REPO`` env var (the presets are external model apps, not shipped in this repo)."""

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai import impact_reg as reg

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

_REPO = os.environ.get("KONFAI_IMPACTREG_REPO", "")


def _skip_reasons() -> list[str]:
    reasons = []
    if not _REPO or not Path(_REPO).is_dir():
        reasons.append("set KONFAI_IMPACTREG_REPO to a local bundle directory")
    if not (Path(_REPO) / "FireANTs_SyN" / "app.json").is_file():
        reasons.append("bundle has no FireANTs_SyN preset")
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
    return reasons


requires_fireants_bundle = pytest.mark.skipif(bool(_skip_reasons()), reason="; ".join(_skip_reasons()))


def _arr(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))


@requires_fireants_bundle
def test_register_large_volume_with_patch_tiling_and_blending(make_reg_pair, tmp_path: Path) -> None:
    fixed, moving, baseline = make_reg_pair(side=96, amplitude=4.0)

    app = reg.ImpactRegKonfAIApp()
    out = tmp_path / "Output"
    # Force patch tiling (64^3, overlap 16) with a Cosinus partition-of-unity blend on both outputs, and
    # trim the FireANTs iterations to keep the 8-patch run quick — all forwarded to `konfai-apps infer --set`.
    app.register(
        ["FireANTs_SyN"],
        [fixed],
        [moving],
        output=out,
        gpu=[0],
        config_overrides=[
            "Predictor.Dataset.Patch.patch_size=[64, 64, 64]",
            "Predictor.Dataset.Patch.overlap=16",
            "Predictor.outputs_dataset.MovedImage.OutputDataset.patch_combine=Cosinus",
            "Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus",
            "affine_iterations=[100, 50, 25]",
            "deformable_iterations=[100, 50, 25]",
        ],
    )

    fixed_a, moved_a = _arr(fixed), _arr(out / "P000" / "Moved.mha")
    dvf = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "P000" / "DVF.mha"))).astype(np.float32)  # (Z,Y,X,3)

    # (1) the moved image stays on the fixed grid
    assert moved_a.shape == fixed_a.shape == (96, 96, 96)
    # (2) patch-tiled registration recovers the smooth warp: NCC improves clearly over the baseline
    after = _ncc(moved_a, fixed_a)
    assert after > baseline + 0.05, f"NCC {baseline:.3f} -> {after:.3f} did not improve enough"
    # (3) no seam on ANY tiled axis: the 96^3 volume is tiled on Z, Y AND X (patch 64 / overlap 16), so the
    #     Cosinus overlap-blend (a partition of unity) must leave the reassembled DVF free of a gradient spike
    #     at the patch-boundary planes on each axis (borders near {31,32,63,64}), not just Z.
    for axis in range(3):
        grad = np.abs(np.diff(dvf, axis=axis)).sum(-1)  # DVF change magnitude across this spatial axis
        global_mean = float(grad.mean())
        seam_mean = max(float(grad.take(p, axis=axis).mean()) for p in (31, 32, 63, 64) if p < grad.shape[axis])
        assert seam_mean < 6.0 * global_mean, f"axis {axis} patch seam: {seam_mean:.3f} vs global {global_mean:.3f}"
