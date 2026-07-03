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

"""Regression test for the landmark transform direction in ``ImpactRegKonfAIApp.evaluate``.

The registration transform ``T`` maps fixed-grid coordinates to moving-grid coordinates
(``sitk.Resample(moving, fixed, T)``), so a correct registration satisfies ``T(f_i) == m_i``
for corresponding fixed/moving fiducials. ``evaluate`` must therefore warp the *fixed*
fiducials forward through ``T`` and compare them against the *moving* fiducials, yielding a
near-zero TRE for a perfect transform. This test locks in that direction.
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys_path_test_dir = Path(__file__).resolve().parents[2]
if str(sys_path_test_dir) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(sys_path_test_dir))

from impact_reg_konfai import impact_reg as impact_reg_module  # noqa: E402
from konfai.utils.dataset import read_landmarks, write_landmarks  # noqa: E402


def test_evaluate_warps_fixed_fiducials_onto_moving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A known transform T: fixed -> moving (translation by +5, -3, +2 in physical space).
    translation = (5.0, -3.0, 2.0)
    transform = sitk.TranslationTransform(3, translation)
    transform_path = tmp_path / "Transform.h5"
    sitk.WriteTransform(transform, str(transform_path))

    fixed_pts = np.array([[10.0, 20.0, 30.0], [-4.0, 8.0, 1.5], [0.0, 0.0, 0.0]], dtype=np.double)
    moving_pts = fixed_pts + np.array(translation)  # m_i = T(f_i) for a perfect registration

    fixed_fid = tmp_path / "fixed.fcsv"
    moving_fid = tmp_path / "moving.fcsv"
    write_landmarks(fixed_pts, fixed_fid)
    write_landmarks(moving_pts, moving_fid)

    captured: dict[str, object] = {}

    def fake_init(self, *args, **kwargs) -> None:
        return None

    def fake_evaluate(self, *, inputs, gt, evaluation_file, **kwargs) -> None:
        if evaluation_file == "Evaluation_with_fid.yml":
            captured["reference"] = read_landmarks(Path(inputs[0][0]))
            captured["moved"] = read_landmarks(Path(gt[0][0]))

    monkeypatch.setattr(impact_reg_module.KonfAIApp, "__init__", fake_init)
    monkeypatch.setattr(impact_reg_module.KonfAIApp, "evaluate", fake_evaluate)

    app = impact_reg_module.ImpactRegKonfAIApp()
    app.evaluate(
        preset="dummy",
        transforms=[transform_path],
        gt_fixed_fid=[fixed_fid],
        gt_moving_fid=[moving_fid],
        output=tmp_path / "Output",
    )

    assert "moved" in captured, "landmark evaluation branch did not run"
    # The reference passed to the metric is the moving fiducial set; the scored points are the
    # fixed fiducials warped by T. For a perfect T they coincide (TRE ~ 0), not doubled.
    np.testing.assert_allclose(captured["reference"], moving_pts, atol=1e-6)
    np.testing.assert_allclose(captured["moved"], moving_pts, atol=1e-4)
