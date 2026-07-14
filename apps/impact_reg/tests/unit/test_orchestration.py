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

"""Unit tests for the IMPACT-Reg orchestration logic (``impact_reg_konfai.impact_reg``), with the KonfAI
runtime stubbed out: preset resolution, output discovery, the mask sentinel, displacement averaging, and
the ``register`` single-preset (reuse) vs multi-preset (ensemble-and-warp) branches."""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai import impact_reg as reg

# --------------------------------------------------------------------------- preset id / discovery


def test_app_id_uses_hf_spec_when_repo_is_not_a_local_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reg, "IMPACT_REG_KONFAI_REPO", "VBoussot/ImpactReg")
    assert reg._app_id("FireANTs_SyN") == "VBoussot/ImpactReg:FireANTs_SyN"


def test_app_id_uses_local_path_when_repo_is_a_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reg, "IMPACT_REG_KONFAI_REPO", str(tmp_path))
    assert reg._app_id("FireANTs_SyN") == str(tmp_path / "FireANTs_SyN")


def test_get_available_presets_keeps_only_registration_apps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, task in [
        ("FireANTs_SyN", "registration"),
        ("Generic_Rigid", "registration"),
        ("LegacyEval", "evaluation"),
    ]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "app.json").write_text(json.dumps({"task": task}), encoding="utf-8")
    (tmp_path / "not_an_app").mkdir()  # no app.json -> ignored
    monkeypatch.setattr(reg, "IMPACT_REG_KONFAI_REPO", str(tmp_path))
    assert reg.get_available_presets() == ["FireANTs_SyN", "Generic_Rigid"]


def test_find_output_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Moved\.mha"):
        reg._find_output(tmp_path, "Moved.mha")


# --------------------------------------------------------------------------- mask sentinel


def test_neutral_mask_writes_tiny_all_ones_sentinel(tmp_path: Path) -> None:
    path = reg._neutral_mask(tmp_path / "FixedMask.mha")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert arr.shape == (2, 2, 2) and arr.dtype == np.uint8 and (arr == 1).all()


# --------------------------------------------------------------------------- displacement averaging


def _write_dvf(path: Path, vector, reference: sitk.Image) -> Path:
    field = np.zeros((*reference.GetSize()[::-1], 3), dtype=np.float32)
    field[...] = vector
    dvf = sitk.GetImageFromArray(field, isVector=True)
    dvf.CopyInformation(reference)
    sitk.WriteImage(dvf, str(path))
    return path


def test_average_displacement_is_the_voxelwise_mean_with_reference_geometry(tmp_path: Path) -> None:
    reference = sitk.GetImageFromArray(np.zeros((6, 6, 6), dtype=np.float32))
    reference.SetSpacing((1.5, 1.5, 1.5))
    reference.SetOrigin((3.0, -2.0, 1.0))
    paths = [
        _write_dvf(tmp_path / "a.mha", (1.0, 0.0, 0.0), reference),
        _write_dvf(tmp_path / "b.mha", (3.0, 2.0, -4.0), reference),
    ]
    avg = reg.ImpactRegKonfAIApp()._average_displacement(paths)
    field = sitk.GetArrayFromImage(avg)
    np.testing.assert_allclose(field[0, 0, 0], (2.0, 1.0, -2.0), atol=1e-6)
    assert avg.GetSpacing() == pytest.approx((1.5, 1.5, 1.5))
    assert avg.GetOrigin() == pytest.approx((3.0, -2.0, 1.0))


# --------------------------------------------------------------------------- register orchestration


def _stub_infer(app: reg.ImpactRegKonfAIApp, moving_image: Path, dvf_by_preset: dict[str, tuple]):
    """Replace ``_infer_preset`` so it writes a Moved.mha + a constant DVF.mha (per preset) on the moving grid."""
    reference = sitk.ReadImage(str(moving_image))

    def fake(preset, fixed_image, mov_image, fixed_mask, moving_mask, work, *args, **kwargs):
        out = Path(work) / preset
        out.mkdir(parents=True, exist_ok=True)
        moved = sitk.Image(reference)
        sitk.WriteImage(moved, str(out / "Moved.mha"))
        _write_dvf(out / "DVF.mha", dvf_by_preset[preset], reference)
        return out / "Moved.mha", out / "DVF.mha"

    app._infer_preset = fake  # type: ignore[method-assign]


def test_register_single_preset_reuses_model_outputs(tmp_path: Path) -> None:
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    app = reg.ImpactRegKonfAIApp()
    _stub_infer(app, moving, {"FireANTs_SyN": (2.0, 0.0, 0.0)})
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out)

    case = out / "P000"
    assert (case / "Moved.mha").is_file() and (case / "DVF.mha").is_file() and (case / "Transform.h5").is_file()
    # single preset: the DVF is the model's own field, reused verbatim (no re-averaging)
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "DVF.mha")))
    np.testing.assert_allclose(field[0, 0, 0], (2.0, 0.0, 0.0), atol=1e-6)


def test_register_multi_preset_averages_and_warps_once(tmp_path: Path) -> None:
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    app = reg.ImpactRegKonfAIApp()
    _stub_infer(app, moving, {"A": (1.0, 0.0, 0.0), "B": (3.0, 0.0, 0.0)})
    out = tmp_path / "Output"
    app.register(["A", "B"], [fixed], [moving], output=out, keep_dvf=True)

    case = out / "P000"
    # ensemble DVF is the mean of the two constant fields
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "DVF.mha")))
    np.testing.assert_allclose(field[0, 0, 0], (2.0, 0.0, 0.0), atol=1e-6)
    # keep_dvf persists each preset's field for a later uncertainty pass
    assert (case / "Ensemble" / "A.mha").is_file() and (case / "Ensemble" / "B.mha").is_file()
    assert (case / "Moved.mha").is_file()
