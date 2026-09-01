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


def test_find_outputs_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Moved"):
        reg._find_outputs(tmp_path, "Moved")


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


def test_ensemble_mean_is_the_voxelwise_mean_with_reference_geometry(tmp_path: Path) -> None:
    """Reduce(Mean) over members-as-cases: the averaged DVF lands at <output>/<case>/DVF, on the
    members' shared grid, which ``grid: strict`` verified rather than assumed."""
    reference = sitk.GetImageFromArray(np.zeros((6, 6, 6), dtype=np.float32))
    reference.SetSpacing((1.5, 1.5, 1.5))
    reference.SetOrigin((3.0, -2.0, 1.0))
    paths = [
        _write_dvf(tmp_path / "a.mha", (1.0, 0.0, 0.0), reference),
        _write_dvf(tmp_path / "b.mha", (3.0, 2.0, -4.0), reference),
    ]
    output, work = tmp_path / "out", tmp_path / "work"
    work.mkdir()

    reg.ImpactRegKonfAIApp()._ensemble_mean("P000", "DVF", ["a", "b"], paths, output, work, [], 1, True)

    avg = sitk.ReadImage(str(output / "P000" / "DVF.mha"))
    field = sitk.GetArrayFromImage(avg)
    np.testing.assert_allclose(field[0, 0, 0], (2.0, 1.0, -2.0), atol=1e-6)
    assert avg.GetSpacing() == pytest.approx((1.5, 1.5, 1.5))
    assert avg.GetOrigin() == pytest.approx((3.0, -2.0, 1.0))


# --------------------------------------------------------------------------- register orchestration


def _stub_infer(app: reg.ImpactRegKonfAIApp, moving_image: Path, dvf_by_preset: dict[str, tuple]):
    """Replace ``_infer_preset`` so it writes a constant DVF.mha per preset on the moving grid,
    reported under group ``DVF`` as the real one reports the group it discovered."""
    reference = sitk.ReadImage(str(moving_image))

    def fake(preset, fixed, moving, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "DVF.mha", dvf_by_preset[preset], reference)
        return "DVF", {"P000": out / "DVF.mha"}

    app._infer_preset = fake  # type: ignore[method-assign]


def test_register_single_preset_reuses_the_field_and_derives_the_moved(tmp_path: Path) -> None:
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    app = reg.ImpactRegKonfAIApp()
    _stub_infer(app, moving, {"FireANTs_SyN": (2.0, 0.0, 0.0)})
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out)

    case = out / "P000"
    assert (case / "Moved.mha").is_file() and (case / "DVF.mha").is_file()
    # nothing else is derived: the transform exists only in the form the preset wrote it
    assert not (case / "Transform.h5").exists()
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


def test_register_derives_moved_when_the_preset_emits_only_a_field(tmp_path: Path) -> None:
    """A preset is complete with a displacement field alone: the moved image is derived here.

    The field IS the registration; the moved image is that field applied to the moving. Requiring both
    made every preset carry a second output whose content the orchestrator can produce itself: and
    for the tiled presets, blend across every patch seam only to have it thrown away.
    """
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    reference = sitk.ReadImage(str(moving))
    app = reg.ImpactRegKonfAIApp()

    def field_only(preset, fixed, moving, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "DVF.mha", (2.0, 0.0, 0.0), reference)
        return "DVF", {"P000": out / "DVF.mha"}

    app._infer_preset = field_only  # type: ignore[method-assign]
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out)

    case = out / "P000"
    assert (case / "Moved.mha").is_file(), "the orchestrator did not derive the moved image"
    # moved(p) = moving(p + d), and d is +2 along x on a unit grid: moving is z*64 + y*8 + x.
    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "Moved.mha")))
    np.testing.assert_allclose(moved[0, 0, 0], 2.0, atol=1e-6)
    np.testing.assert_allclose(moved[1, 1, 0], 74.0, atol=1e-6)


def test_register_fields_only_writes_nothing_derived(tmp_path: Path) -> None:
    """A caller that composes the field itself pays for the field, and nothing else.

    The moved image is derived FROM the field: a full-size resample of the same voxels. The tiled
    refinement reads the field, composes it with its global pass and derives its own moved, so
    producing one for it is pure waste.
    """
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    app = reg.ImpactRegKonfAIApp()
    _stub_infer(app, moving, {"FireANTs_SyN": (2.0, 0.0, 0.0)})
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out, fields_only=True)

    case = out / "P000"
    assert (case / "DVF.mha").is_file()
    assert not (case / "Moved.mha").exists()
    assert not (case / "Transform.h5").exists()


def test_register_reads_a_store_moving_against_an_itk_field(tmp_path: Path) -> None:
    """One store entry beside an ``.mha`` flips a mixed root's backend: the staging keeps one root
    per group, so a caller's OME-Zarr moving registers against the ``.mha`` field every published
    preset declares."""
    ome_zarr = pytest.importorskip("konfai.utils.ome_zarr")
    volume = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
    moving = tmp_path / "moving.ome.zarr"
    ome_zarr.write_ome_zarr(moving, volume[None], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    reference = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32))  # the moving's grid
    app = reg.ImpactRegKonfAIApp()

    def field_only(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "DVF.mha", (2.0, 0.0, 0.0), reference)
        return "DVF", {"P000": out / "DVF.mha"}

    app._infer_preset = field_only  # type: ignore[method-assign]
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out)

    # The moved image takes the MOVING's form (a store in, a store out), while the field the
    # preset wrote keeps its own. moved(p) = moving(p + d), d = +2 along x on a unit grid:
    # moving is z*64 + y*8 + x.
    store = out / "P000" / "Moved.ome.zarr"
    assert store.is_dir(), "the moved image was not written in the moving's own form"
    moved = ome_zarr.read_ome_zarr_data_slice(store, (slice(None),) * 4)[0][0]
    np.testing.assert_allclose(moved[0, 0, 0], 2.0, atol=1e-6)
    np.testing.assert_allclose(moved[1, 1, 0], 74.0, atol=1e-6)
    assert (out / "P000" / "DVF.mha").is_file()


def test_register_adopts_the_presets_output_name(tmp_path: Path) -> None:
    """A preset names its output; the pipeline follows. An official preset calls its transform
    ``Transform`` (where Slicer looks for it), and ``register`` must not rename it ``DVF``."""
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    reference = sitk.ReadImage(str(moving))
    app = reg.ImpactRegKonfAIApp()

    def named_transform(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "Transform.mha", (2.0, 0.0, 0.0), reference)
        return "Transform", {"P000": out / "Transform.mha"}

    app._infer_preset = named_transform  # type: ignore[method-assign]
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out)

    case = out / "P000"
    assert (case / "Transform.mha").is_file() and (case / "Moved.mha").is_file()
    assert not (case / "DVF.mha").exists()


@pytest.mark.parametrize(
    ("name", "token"),
    [("moving.mha", "mha"), ("patient.v2.mha", "mha"), ("Transform.h5", "itktransform"), ("Reg.tfm", "itktransform")],
)
def test_a_transform_file_is_a_transform_here(name: str, token: str) -> None:
    """The one reading this layer does not share with konfai: an ``.h5`` HERE is the registration
    every preset writes, not konfai's monolithic HDF5 dataset. The form itself is konfai's to say."""
    assert reg._format_token(reg._form(Path(name))) == token


def test_register_reads_a_dicom_series_as_the_moving(tmp_path: Path) -> None:
    """A DICOM series is a DIRECTORY carrying no extension, so its backend cannot be read off a name.
    Staged as ``mha`` (the default a formless name falls back to) konfai was handed a directory of
    slices to read as one image."""
    pytest.importorskip("pydicom")
    from konfai.utils import dicom

    series = tmp_path / "SERIES"
    series.mkdir()
    volume = np.arange(8**3, dtype=np.int16).reshape(8, 8, 8)
    dicom.write_dicom_series(series, volume[None], origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    assert reg._backend(series) == "dicom"

    reference = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32))
    app = reg.ImpactRegKonfAIApp()

    def field_only(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "DVF.mha", (2.0, 0.0, 0.0), reference)
        return "DVF", {"P000": out / "DVF.mha"}

    app._infer_preset = field_only  # type: ignore[method-assign]
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [series], output=out)

    case = out / "P000"
    assert (case / "DVF.mha").is_file()
    # The moved image takes the moving's form: a series in, a series out.
    assert (case / "Moved").is_dir() and list((case / "Moved").glob("*.dcm"))


def test_register_reads_a_moving_whose_stem_carries_a_dot(tmp_path: Path) -> None:
    """End to end on the name that used to kill the run before the first read."""
    moving = tmp_path / "patient.v2.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    reference = sitk.ReadImage(str(moving))
    app = reg.ImpactRegKonfAIApp()

    def field_only(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "DVF.mha", (2.0, 0.0, 0.0), reference)
        return "DVF", {"P000": out / "DVF.mha"}

    app._infer_preset = field_only  # type: ignore[method-assign]
    out = tmp_path / "Output"
    app.register(["FireANTs_SyN"], [fixed], [moving], output=out)

    case = out / "P000"
    assert (case / "DVF.mha").is_file()
    # The moved image takes the moving's FORM, not its whole tail of dots: moved(p) = moving(p + d),
    # d = +2 along x on a unit grid, moving = z*64 + y*8 + x.
    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "Moved.mha")))
    assert moved[0, 0, 0] == pytest.approx(2.0)
    assert moved[1, 1, 0] == pytest.approx(74.0)


def test_register_refuses_presets_that_name_their_output_differently(tmp_path: Path) -> None:
    """An ensemble folds one group: members that disagree on its name are refused, not renamed."""
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    reference = sitk.ReadImage(str(moving))
    app = reg.ImpactRegKonfAIApp()
    group_by_preset = {"A": "DVF", "B": "Transform"}

    def mixed(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        group = group_by_preset[preset]
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / f"{group}.mha", (2.0, 0.0, 0.0), reference)
        return group, {"P000": out / f"{group}.mha"}

    app._infer_preset = mixed  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="named their output differently"):
        app.register(["A", "B"], [fixed], [moving], output=tmp_path / "Output")


def test_case_order_is_numeric_past_p999() -> None:
    """`sorted` alone puts P1000 before P101 and pairs the wrong moving unit."""
    assert sorted(["P101", "P1000", "P099"], key=reg._case_key) == ["P099", "P101", "P1000"]


def test_register_refuses_a_preset_output_named_moved(tmp_path: Path) -> None:
    """The derived image is written under 'Moved'; a preset using that name would be deleted by
    its own derivation's stale-output purge."""
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))

    reference = sitk.ReadImage(str(moving))
    app = reg.ImpactRegKonfAIApp()

    def named_moved(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        _write_dvf(out / "Moved.mha", (2.0, 0.0, 0.0), reference)
        return "Moved", {"P000": out / "Moved.mha"}

    app._infer_preset = named_moved  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Moved"):
        app.register(["FireANTs_SyN"], [fixed], [moving], output=tmp_path / "Output")


def test_find_output_group_discovers_the_one_group(tmp_path: Path) -> None:
    """konfai-apps writes ``<run>/<group>/<case>/…``; the group is the one directory holding cases."""
    (tmp_path / "reg" / "Transform" / "P000").mkdir(parents=True)
    assert reg._find_output_group(tmp_path) == "Transform"


def test_find_output_group_refuses_more_than_one(tmp_path: Path) -> None:
    (tmp_path / "reg" / "DVF" / "P000").mkdir(parents=True)
    (tmp_path / "reg" / "Moved" / "P000").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="exactly one output group"):
        reg._find_output_group(tmp_path)


def test_stage_group_replaces_an_existing_link(tmp_path: Path) -> None:
    """Re-staging the same case points the link at the new source instead of raising."""
    first, second = tmp_path / "a.mha", tmp_path / "b.mha"
    for path in (first, second):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.float32)), str(path))

    reg._stage_group(tmp_path / "stage", "DVF", {"P000": first})
    spec = reg._stage_group(tmp_path / "stage", "DVF", {"P000": second})

    root = Path(spec.rpartition(":")[0])
    assert (root / "P000" / "DVF.mha").resolve() == second.resolve()
