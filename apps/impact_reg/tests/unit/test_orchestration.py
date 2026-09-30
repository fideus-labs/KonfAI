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

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai import impact_reg as reg
from konfai.utils.errors import KonfAIError
from konfai.utils.ITK import field_reach, read_displacement_field


@pytest.fixture(autouse=True)
def _local_presets(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local preset repository holding the names these tests register with: register checks the names it is
    given before it runs anything, and nothing here may reach Hugging Face."""
    repo = tmp_path_factory.mktemp("presets")
    for name in ("FireANTs_SyN", "A", "B", "P"):
        (repo / name).mkdir()
        (repo / name / "app.json").write_text(json.dumps({"task": "registration"}), encoding="utf-8")
    monkeypatch.setattr(reg, "IMPACT_REG_KONFAI_REPO", str(repo))


# --------------------------------------------------------------------------- preset id / discovery


def test_find_outputs_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Moved"):
        reg._find_outputs(tmp_path, "Moved")


# --------------------------------------------------------------------------- mask sentinel


def _mask_groups(tmp_path: Path, monkeypatch, fixed: Path, fixed_masks, moving_masks, write_preset_output):
    """The ``-i`` groups ``_infer_preset`` hands konfai-apps."""
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        write_preset_output(Path(command[command.index("-o") + 1]) / "reg" / "DVF" / "P000")

    monkeypatch.setattr(reg.subprocess, "run", fake_run)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    moving = [tmp_path / "moving.mha"]
    reg.ImpactRegKonfAIApp()._infer_preset("P", [fixed], moving, fixed_masks, moving_masks, 1, work, [], None, True)
    command, groups = captured[0], []
    for index, token in enumerate(command):
        if token == "-i":
            groups.append(list(itertools.takewhile(lambda item: not item.startswith("-"), command[index + 1 :])))
    return groups


def test_a_lone_fixed_mask_leaves_the_moving_mask_to_konfai_apps(tmp_path, monkeypatch, write_preset_output) -> None:
    """konfai-apps fills a trailing group left out with ones on the fixed grid, which patches like the fixed: the
    2 x 2 x 2 sentinel passed in its place made every patched or tiled run refuse the case."""
    fixed = tmp_path / "fixed.mha"
    groups = _mask_groups(tmp_path, monkeypatch, fixed, [tmp_path / "fm.mha"], [], write_preset_output)
    assert groups == [[str(fixed)], [str(tmp_path / "moving.mha")], [str(tmp_path / "fm.mha")]]


@pytest.mark.parametrize("form", [".mha", ".nii.gz", ".ome.zarr"])
def test_a_lone_moving_mask_gets_an_all_ones_fixed_mask_on_the_fixed_grid(
    tmp_path, monkeypatch, write_preset_output, form
) -> None:
    """The fixed-mask slot precedes the moving mask's, so it is filled: with ones on the fixed grid, in the fixed
    image's form (konfai-apps lists a staged dataset under the format of its first group)."""
    ome_zarr = pytest.importorskip("konfai.utils.ome_zarr")
    fixed = tmp_path / f"fixed{form}"
    if form == ".ome.zarr":
        ome_zarr.write_ome_zarr(fixed, np.zeros((1, 5, 6, 7), np.float32), spacing=(0.5, 1.0, 2.0), origin=(1, 2, 3))
    else:
        image = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.float32))
        image.SetSpacing((0.5, 1.0, 2.0))
        image.SetOrigin((1.0, 2.0, 3.0))
        sitk.WriteImage(image, str(fixed))
    groups = _mask_groups(tmp_path, monkeypatch, fixed, [], [tmp_path / "mm.mha"], write_preset_output)

    assert len(groups) == 4 and groups[3] == [str(tmp_path / "mm.mha")]
    (mask,) = (Path(path) for path in groups[2])
    assert mask.name == f"FixedMask{form}"
    from konfai.utils.dataset import Dataset

    ones, attributes = Dataset(str(mask.parent.parent), "omezarr" if form == ".ome.zarr" else form[1:]).read_data(
        "FixedMask", mask.parent.name
    )
    assert np.asarray(ones).shape == (1, 5, 6, 7) and (np.asarray(ones) == 1).all()
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), (0.5, 1.0, 2.0))
    np.testing.assert_allclose(attributes.get_np_array("Origin"), (1.0, 2.0, 3.0))


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


def test_a_moved_store_is_written_as_a_pyramid(tmp_path: Path) -> None:
    """Its moving came as a multi-level store; one level made a viewer load the full resolution to show it."""
    ome_zarr = pytest.importorskip("konfai.utils.ome_zarr")
    volume = np.zeros((8, 8, 600), dtype=np.float32)
    moving = tmp_path / "moving.ome.zarr"
    ome_zarr.write_ome_zarr(moving, volume[None], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(fixed))
    _stub_infer(app := reg.ImpactRegKonfAIApp(), fixed, {"A": (0.0, 0.0, 0.0)})
    app.register(["A"], [fixed], [moving], output=tmp_path / "Output", quiet=True)

    assert ome_zarr.get_ome_zarr_info(tmp_path / "Output" / "P000" / "Moved.ome.zarr")["n_levels"] == 3


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


def test_register_refuses_an_input_in_a_case_directory_it_writes(tmp_path: Path) -> None:
    """Chaining ``-m Out/P000/Moved.mha -o Out`` deleted that moved image before reading it."""
    out = tmp_path / "Output"
    (out / "P000").mkdir(parents=True)
    fixed, moving = tmp_path / "fixed.mha", out / "P000" / "Moved.mha"
    for path in (fixed, moving):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = lambda *args, **kwargs: pytest.fail("a preset ran")  # type: ignore[method-assign]

    with pytest.raises(KonfAIError, match="another -o"):
        app.register(["FireANTs_SyN"], [fixed], [moving], output=out)
    assert moving.is_file()


def test_register_checks_its_presets_and_inputs_before_any_preset_runs(tmp_path: Path) -> None:
    """A typo, a count mismatch or a mixed group failed only once the presets before it had run, for hours."""
    ome_zarr = pytest.importorskip("konfai.utils.ome_zarr")
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    images = {name: tmp_path / f"{name}.mha" for name in ("fixed", "fixed2", "moving")}
    for path in images.values():
        sitk.WriteImage(sitk.GetImageFromArray(volume), str(path))
    store = tmp_path / "moving2.ome.zarr"
    ome_zarr.write_ome_zarr(store, volume[None], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = lambda *args, **kwargs: pytest.fail("a preset ran")  # type: ignore[method-assign]
    fixed, moving = [images["fixed"]], [images["moving"]]

    for presets, pattern in ((["FireANTs_SyM"], "Did you mean 'FireANTs_SyN'"), (["A", "B", "A"], "listed twice")):
        with pytest.raises(KonfAIError, match=pattern):
            app.register(presets, fixed, moving, output=tmp_path / "Output")
    with pytest.raises(KonfAIError, match="Moving input expands to 1 volume"):
        app.register(["A"], [images["fixed"], images["fixed2"]], moving, output=tmp_path / "Output")
    with pytest.raises(KonfAIError, match=r"mix storage forms \(.mha, .ome.zarr\)"):
        app.register(["A"], [images["fixed"], images["fixed2"]], [images["moving"], store], output=tmp_path / "Out")


@pytest.mark.parametrize(
    ("name", "array", "vector"),
    [
        ("flat.png", np.zeros((8, 8), dtype=np.uint8), False),
        ("rgb.mha", np.zeros((8, 8, 8, 3), dtype=np.uint8), True),
        ("series.nii.gz", np.zeros((2, 8, 8, 8), dtype=np.float32), False),
    ],
)
def test_register_refuses_what_is_not_a_3d_scalar_volume(tmp_path: Path, name: str, array, vector: bool) -> None:
    """A 2-D image, a colour volume or a time series failed deep inside an engine, after the presets before it."""
    fixed, moving = tmp_path / "fixed.mha", tmp_path / name
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))
    sitk.WriteImage(sitk.GetImageFromArray(array, isVector=vector), str(moving))
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = lambda *args, **kwargs: pytest.fail("a preset ran")  # type: ignore[method-assign]

    with pytest.raises(KonfAIError, match="3-D single-channel"):
        app.register(["A"], [fixed], [moving], output=tmp_path / "Output", quiet=True)


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


def test_stage_group_copies_where_symlinks_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows without Developer Mode refuses symlinks (WinError 1314): the moved image, eval and uncertainty all
    stage through here, and crashed after the preset had run."""
    source = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.arange(8, dtype=np.float32).reshape(2, 2, 2)), str(source))

    def refused(*args, **kwargs):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(reg.os, "symlink", refused)
    root = Path(reg._stage_group(tmp_path / "stage", "Moving", {"P000": source}).rpartition(":")[0])

    staged = root / "P000" / "Moving.mha"
    assert not staged.is_symlink() and staged.read_bytes() == source.read_bytes()


def test_infer_preset_runs_this_interpreters_konfai_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_preset_output
) -> None:
    """Not the konfai-apps first on PATH, which an unactivated environment lacks or another environment shadows.
    The engines stage their files in the work dir, its TMPDIR."""
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured.update(command=list(command), env=kwargs["env"])
        write_preset_output(Path(command[command.index("-o") + 1]) / "reg" / "DVF" / "P000")

    monkeypatch.setattr(reg.subprocess, "run", fake_run)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "system"))
    app, arguments = reg.ImpactRegKonfAIApp(), ([tmp_path / "f.mha"], [tmp_path / "m.mha"], [], [], 1)
    app._infer_preset(
        "A", *arguments, tmp_path, [], None, True, tta=2, config_overrides=["seed=1", "cc_kernel=7"], max_voxels=1000
    )

    command, env = captured["command"], captured["env"]
    assert command[:4] == [sys.executable, "-m", "konfai_apps", "infer"]
    assert command[command.index("--tta") + 1] == "2"
    assert [command[index + 1] for index, token in enumerate(command) if token == "--set"] == ["seed=1", "cc_kernel=7"]
    # konfai-apps' own option: a --set of Patch.max_voxels was refused, the presets not declaring it.
    assert command[command.index("--max-voxels") + 1] == "1000"
    assert env["TMPDIR"] == str(tmp_path)


def test_the_preset_command_is_one_konfai_apps_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_preset_output
) -> None:
    """The command impact-reg builds, parsed by konfai-apps' real CLI into the real KonfAIApp.infer, and its --set
    applied to a preset config: a flag or a --set konfai-apps refuses fails here, not in a user's run (a --set of
    Patch.max_voxels passed every test that faked the subprocess and failed every sized run)."""
    import inspect

    from konfai_apps import app as app_module
    from konfai_apps import cli as apps_cli
    from konfai_apps.app_repository import check_overrides

    captured: dict = {}
    monkeypatch.setattr(
        reg.subprocess,
        "run",
        lambda command, **kwargs: (
            captured.update(command=list(command))
            or write_preset_output(Path(command[command.index("-o") + 1]) / "reg" / "DVF" / "P000")
        ),
    )
    app, arguments = reg.ImpactRegKonfAIApp(), ([tmp_path / "f.mha"], [tmp_path / "m.mha"], [], [], 1)
    app._infer_preset("A", *arguments, tmp_path, [0], None, True, tta=2, config_overrides=["seed=1"], max_voxels=1000)

    received: dict = {}

    class _App:
        def __init__(self, *args) -> None:
            pass

        def infer(self, **kwargs) -> None:
            inspect.signature(app_module.KonfAIApp.infer).bind(None, **kwargs)
            received.update(kwargs)

    monkeypatch.setattr(app_module, "KonfAIApp", _App)
    monkeypatch.setattr(sys, "argv", ["konfai-apps", *captured["command"][3:]])
    apps_cli.main_apps()

    assert received["max_voxels"] == 1000 and received["gpu"] == [0] and received["tta"] == 2
    config = {
        "Predictor": {
            "Model": {"classpath": "impact_reg_konfai.models.fireants:RegistrationNet", "RegistrationNet": {}},
            "Dataset": {"Patch": {"patch_size": [0, 0, 0], "mode": "resample"}},
        }
    }
    check_overrides(config, received["config_overrides"])


def test_a_failed_preset_ends_on_one_message_and_keeps_its_logs_where_it_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The preset printed its traceback, then register added a CalledProcessError of its whole argv, and deleted
    the work dir the messages pointed into. A run that succeeds leaves nothing behind."""
    fixed, moving = tmp_path / "fixed.mha", tmp_path / "moving.mha"
    for path in (fixed, moving):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))

    def failing(command, **kwargs):
        raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(reg.subprocess, "run", failing)
    with pytest.raises(KonfAIError, match=r"preset 'A' failed \(exit 3\)") as error:
        reg.ImpactRegKonfAIApp().register(
            ["A"], [fixed], [moving], output=tmp_path / "Output", tmp_dir=tmp_path / "tmp"
        )

    (work,) = (tmp_path / "tmp").iterdir()
    assert error.value.__suppress_context__ and f"kept in {work / 'A'}" in str(error.value)
    assert f"intermediates and logs kept in {work}" in capsys.readouterr().err


def test_register_stages_beside_its_output_and_hands_the_presets_its_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The system temporary directory is often a tmpfs, where the volume-sized work dir filled RAM: it is beside the
    output now, on the results' disk, and says where. A directory input reaches the presets as the volumes it held
    when the run started: walked again, it would list the work dir, here inside it."""
    fixed, moving_dir = tmp_path / "fixed.mha", tmp_path / "moving"
    moving_dir.mkdir()
    for path in (fixed, moving_dir / "m.mha"):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))
    output, seen = moving_dir / "Output", {}

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        seen.update(work=Path(work), moving=list(moving_images))
        out = Path(work) / name / "P000"
        out.mkdir(parents=True)
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (0.0, 0.0, 0.0), sitk.ReadImage(str(fixed)))}

    monkeypatch.setattr(reg.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 1000})())
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(["A"], [fixed], [moving_dir], output=output, fields_only=True)

    assert seen["work"].parent == moving_dir and seen["work"].name.startswith(".Output.")
    assert seen["moving"] == [moving_dir / "m.mha"] and not seen["work"].exists()
    console = capsys.readouterr()
    assert f"intermediates in {seen['work']}" in console.out and "may stage up to" in console.err


_CONFIG = """Predictor:
  Model:
    classpath: impact_reg_konfai.models.fireants:RegistrationNet
    RegistrationNet:
      seed: 42
      {own}
      outputs_criterions: None
  Dataset:
    batch_size: 1
"""


_TILING = {"global": "Prediction.yml", "tile": "Prediction_tile.yml"}


def _tiled_preset(name: str, config: str, tile_config: str) -> None:
    """A preset that tiles: its global config, and its tile config, the deformable stage alone."""
    folder = Path(reg.IMPACT_REG_KONFAI_REPO) / name
    (folder / "app.json").write_text(json.dumps({"task": "registration", "tiling": _TILING}), encoding="utf-8")
    (folder / "Prediction.yml").write_text(config, encoding="utf-8")
    (folder / "Prediction_tile.yml").write_text(tile_config, encoding="utf-8")


def _deformable_alone(config: str) -> str:
    """The FireANTs tile config of ``config``: no linear stage."""
    return config.replace("      seed: 42\n", "      seed: 42\n      linear_method: none\n")


def test_set_is_scoped_per_preset_and_checked_before_any_preset_runs(tmp_path: Path) -> None:
    """Engines name one knob differently, and an ensemble failed on the first member lacking the key, as a
    CalledProcessError, after the members before it had run."""
    repo = Path(reg.IMPACT_REG_KONFAI_REPO)
    for name, own in (("A", "deformable_iterations: [200, 100, 50]"), ("B", "max_iterations: 0")):
        config = _CONFIG.format(own=own)
        if name == "B":  # an elastix preset: FireANTs' knobs are not its model's
            config = config.replace("models.fireants:", "models.elastix:").replace("      seed: 42\n", "")
        (repo / name / "Prediction.yml").write_text(config, encoding="utf-8")

    common = ["Predictor.Dataset.batch_size=2"]
    shares = reg._preset_overrides(["A", "B"], ["A:deformable_iterations=[1, 1, 1]", "A:seed=3", *common])
    assert shares == {
        "A": {"global": ["deformable_iterations=[1, 1, 1]", "seed=3", *common], "tile": []},
        "B": {"global": common, "tile": []},
    }
    with pytest.raises(KonfAIError, match=r"B \(Prediction\.yml\): .*'deformable_iterations' does not exist"):
        reg._preset_overrides(["A", "B"], ["deformable_iterations=[1, 1, 1]"])
    with pytest.raises(KonfAIError, match="names the preset 'C'"):
        reg._preset_overrides(["A", "B"], ["C:max_iterations=1"])

    fixed, moving = tmp_path / "fixed.mha", tmp_path / "moving.mha"
    for path in (fixed, moving):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = lambda *args, **kwargs: pytest.fail("a preset ran")  # type: ignore[method-assign]
    with pytest.raises(KonfAIError, match="Did you mean 'max_iterations'"):
        app.register(["A", "B"], [fixed], [moving], output=tmp_path / "Output", config_overrides=["B:max_iteration=5"])


def test_set_takes_a_model_argument_the_preset_leaves_at_its_default() -> None:
    """``presets NAME`` lists every argument of the preset's model, and konfai-apps sets one the preset leaves out
    (``moments_init`` of FireANTs): register refused it before anything ran, and kept it from the tiles."""
    config = _CONFIG.format(own="deformable_iterations: [200, 100, 50]")
    _tiled_preset("A", config, config)

    assert reg._preset_overrides(["A"], ["moments_init=com"]) == {
        "A": {"global": ["moments_init=com"], "tile": ["moments_init=com"]}
    }
    with pytest.raises(KonfAIError, match="Did you mean 'moments_init'"):
        reg._preset_overrides(["A"], ["moment_init=com"])


def test_the_tiles_keep_the_stages_their_config_sets_unless_a_scope_names_them() -> None:
    """``linear_method`` is what makes the tile config the deformable stage alone: forwarded to the tiles, an override
    of it brought a linear stage back into every tile, and the blended field tore. A scope gives an override to one
    pass: ``tile:`` a tile experiment or iterations of the tiles alone, ``global:`` the global pass alone."""
    config = _CONFIG.format(own="deformable_iterations: [200, 100, 50]")
    _tiled_preset("A", config, _deformable_alone(config))

    shares = reg._preset_overrides(
        ["A"], ["seed=3", "linear_method=rigid", "A:tile:linear_method=rigid", "global:seed=4"], quiet=True
    )
    assert shares == {
        "A": {"global": ["seed=3", "linear_method=rigid", "seed=4"], "tile": ["seed=3", "linear_method=rigid"]}
    }


def test_an_override_kept_from_the_tiles_is_said_before_anything_runs_even_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It was said as the tiles started, once the global pass had run, and not at all under --quiet."""
    config = _CONFIG.format(own="deformable_iterations: [200, 100, 50]")
    _tiled_preset("A", config, _deformable_alone(config))
    fixed, moving = tmp_path / "fixed.mha", tmp_path / "moving.mha"
    for path in (fixed, moving):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))
    said: list[str] = []

    def preset(*args, **kwargs):
        said.append(capsys.readouterr().out)
        raise RuntimeError("the preset ran")

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="the preset ran"):
        app.register(
            ["A"],
            [fixed],
            [moving],
            output=tmp_path / "Output",
            config_overrides=["seed=3", "linear_method=rigid"],
            quiet=True,
        )
    assert said == [
        "[ImpactReg] A: --set linear_method=rigid -> the preset, not its tiles: Prediction_tile.yml sets it to 'none'"
        " (--set A:tile:linear_method=rigid sets it in the tiles too).\n"
    ]


def test_the_tile_share_is_checked_against_the_tile_config_before_anything_runs() -> None:
    """konfai-apps checked the tile config in the tile pass's own process, once the global pass had run; and a tile
    scope given a preset that does not tile would have set nothing."""
    config = _CONFIG.format(own="deformable_iterations: [200, 100, 50]")
    _tiled_preset("A", config, config.replace("  Dataset:\n    batch_size: 1\n", ""))

    assert reg._preset_overrides(["A"], ["Predictor.Dataset.batch_size=2"], quiet=True)["A"]["tile"] == []
    with pytest.raises(KonfAIError, match=r"A \(Prediction_tile\.yml\): .*has no key 'Dataset'"):
        reg._preset_overrides(["A"], ["tile:Predictor.Dataset.batch_size=2"])
    with pytest.raises(KonfAIError, match="'B' has no tile pass"):
        reg._preset_overrides(["A", "B"], ["B:tile:seed=1"])


def test_set_reaches_a_field_of_a_feature_model_the_preset_does_not_write() -> None:
    """A preset writes the fields of a feature model it sets: feature_normalization, which the FireANTs presets write
    and the ConvexAdam ones do not, could be swept on the first and not on the second."""
    # A feature model's fields, feature_normalization left out.
    models = (
        "models:\n        '0':\n          ref: VBoussot/impact-torchscript-models:TS/M730.pt\n"
        "          layers_mask: '1'"
    )
    (Path(reg.IMPACT_REG_KONFAI_REPO) / "A" / "Prediction.yml").write_text(_CONFIG.format(own=models), encoding="utf-8")

    name = "Predictor.Model.RegistrationNet.models.0.feature_normalization"
    assert reg._preset_overrides(["A"], [f"{name}=l2"]) == {"A": {"global": [f"{name}=l2"], "tile": []}}
    with pytest.raises(KonfAIError, match="Did you mean 'feature_normalization'"):
        reg._preset_overrides(["A"], ["Predictor.Model.RegistrationNet.models.0.feature_normalisation=l2"])
    with pytest.raises(KonfAIError, match="is not one of"):
        reg._preset_overrides(["A"], [f"{name}=L2"])


def test_register_records_the_overrides_each_pass_took(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """register.json listed every --set as applied, those the tiles had not taken included."""
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 16)
    config = _CONFIG.format(own="deformable_iterations: [200, 100, 50]")
    _tiled_preset("P", config, _deformable_alone(config))
    monkeypatch.setattr(reg, "_manifest", lambda preset: {"ram_bytes_per_voxel": 1150, "tiling": _TILING})
    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: int(1000 * 1150 / cost))

    def fake_run(command, **kwargs):
        out = Path(command[command.index("-o") + 1]) / "reg" / "DVF" / "P000"
        out.mkdir(parents=True)
        _write_dvf(out / "DVF.mha", (0.0, 0.0, 0.0), sitk.ReadImage(command[command.index("-i") + 1]))

    monkeypatch.setattr(reg.subprocess, "run", fake_run)
    overrides = ["seed=3", "linear_method=rigid"]
    reg.ImpactRegKonfAIApp().register(
        ["P"],
        [fixed],
        [moving],
        output=tmp_path / "Output",
        config_overrides=overrides,
        max_voxels=1000,
        quiet=True,
        fields_only=True,
    )

    record = json.loads((tmp_path / "Output" / "register.json").read_text(encoding="utf-8"))
    assert record["overrides"] == overrides
    assert record["applied"] == {"P_global": overrides, "P_tiles": ["seed=3"]}


def test_a_scope_comes_before_the_name_and_the_colon_of_a_value_is_none() -> None:
    name = "Predictor.Model.RegistrationNet.models.0.ref=VBoussot/impact-torchscript-models:TS/M730.pt"
    assert reg._scopes(f"A:tile:{name}", ["A"]) == ("A", "tile", name)
    assert reg._scopes("ref=repo:TS/M730.pt", ["A"]) == (None, None, "ref=repo:TS/M730.pt")
    assert reg._scopes("global:seed=3", ["A"]) == (None, "global", "seed=3")
    with pytest.raises(KonfAIError, match="names the preset 'C'"):
        reg._scopes("C:seed=3", ["A"])


def test_register_records_which_input_became_which_case_and_sums_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The cases are named by position and the run ended on KonfAI lines naming a deleted work dir: the record says
    which input each case came from, and the summary where its results are."""
    monkeypatch.chdir(tmp_path)
    fixed, moving = (
        [tmp_path / "fixed_a.mha", tmp_path / "fixed_b.mha"],
        [tmp_path / "moving_a.mha", tmp_path / "moving_b.mha"],
    )
    for path in (*fixed, *moving):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))
    reference = sitk.ReadImage(str(fixed[0]))

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        fields = {}
        for index in range(n_cases):
            case = Path(work) / name / f"P{index:03d}"
            case.mkdir(parents=True)
            fields[case.name] = _write_dvf(case / "DVF.mha", (0.0, 0.0, 0.0), reference)
        return "DVF", fields

    (Path(reg.IMPACT_REG_KONFAI_REPO) / "A" / "app.json").write_text('{"display_name": "Engine A"}', encoding="utf-8")
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(["A"], fixed[:1], moving[:1], output=tmp_path / "Output", config_overrides=None)

    summary = capsys.readouterr().out.split("[ImpactReg] Registration completed in ")[1]
    assert "Preset:    A (Engine A)" in summary and f"Moving:    {Path('moving_a.mha')}" in summary
    assert (
        f"Transform: {Path('Output/P000/DVF.mha')}" in summary
        and f"Moved:     {Path('Output/P000/Moved.mha')}" in summary
    )

    app.register(["A"], fixed, moving, output=tmp_path / "Output", quiet=True)
    assert capsys.readouterr().out == "", "-q prints nothing, KonfAI's own lines included"
    record = json.loads((tmp_path / "Output" / "register.json").read_text(encoding="utf-8"))
    assert record["presets"] == ["A"] and record["cases"]["P001"] == {
        "fixed": str(fixed[1]),
        "moving": str(moving[1]),
        "transform": str(Path("P001/DVF.mha")),
        "moved": str(Path("P001/Moved.mha")),
    }


# --------------------------------------------------------------------------- native tiles


def _ramp(path: Path, side: int) -> Path:
    """moving(z, y, x) = 256 z + 16 y + x on a 1 mm grid, so a displacement reads back as a value."""
    grid = np.mgrid[0:side, 0:side, 0:side].astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(256 * grid[0] + 16 * grid[1] + grid[2]), str(path))
    return path


def test_max_voxels_pins_the_tiles_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # A tile holds what the pair registered whole would take, whatever the device: one plan on every machine.
    manifest = {"vram_bytes_per_voxel": 1150, "tiling": {"tile": "P_tile.yml", "tile_vram_bytes_per_voxel": 575}}
    assert reg._plan(manifest, [0], 1000) == (1000, 2000)
    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: int(1150 * 575 * 10 / cost))
    assert reg._plan(manifest, [0], None) == (5750, 11500)


def test_every_preset_is_sized_from_the_costs_it_declares_for_its_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preset without a tile pass is sized too (KonfAI resamples it), from its RAM cost on the CPU; a pass that
    declares no cost is left unsized."""
    asked: list = []
    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: asked.append((cost, gpu)) or 10**6)
    manifest = {"vram_bytes_per_voxel": 100, "ram_bytes_per_voxel": 40}
    assert reg._plan(manifest, [], None) == (10**6, None) and asked == [(40.0, None)]
    assert reg._plan(manifest, [2], None) == (10**6, None) and asked[-1] == (100.0, 2)
    assert reg._plan({"vram_bytes_per_voxel": 100}, [], None) == (None, None)


def test_a_pair_too_large_runs_its_global_stages_resampled_then_native_tiles_and_composes_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 16)
    tiling = {"global": "Prediction_global.yml", "tile": "Prediction_tile.yml"}
    monkeypatch.setattr(reg, "_manifest", lambda preset: {"ram_bytes_per_voxel": 1150, "tiling": tiling})
    seen: dict = {}

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        grid = sitk.ReadImage(str(fixed_images[0]))
        out = Path(work) / kwargs["label"] / "P000"
        out.mkdir(parents=True, exist_ok=True)
        if kwargs["prediction_file"] == "Prediction_global.yml":
            seen["global"], seen["global_cap"] = grid.GetSize(), kwargs["max_voxels"]
            return "DVF", {"P000": _write_dvf(out / "DVF.mha", (2.0, 0.0, 0.0), grid)}  # +2 mm along x
        seen["tile"], seen["tile_cap"] = grid.GetSize(), kwargs["max_voxels"]
        seen["prewarped"] = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_images[0])))
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (1.0, 0.0, 0.0), grid)}  # +1 mm more, on native tiles

    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: int(1000 * 1150 / cost))
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(["P"], [fixed], [moving], output=tmp_path / "Output", max_voxels=1000, quiet=True)

    # Both passes get the native pair: KonfAI resamples the global one (mode: resample) and cuts the tiles (mode: tile)
    # to the voxels the card holds.
    assert seen["global"] == seen["tile"] == (16, 16, 16)
    assert seen["global_cap"] == seen["tile_cap"] == 1000
    plans = json.loads((tmp_path / "Output" / "register.json").read_text())["plans"]
    assert plans == {"P": {"whole_voxels": 1000, "tile_voxels": 1000}}
    # the tiles registered the moving already moved by the global stages: 2 mm further along x
    np.testing.assert_allclose(seen["prewarped"][5, 6, 7], 256 * 5 + 16 * 6 + 7 + 2, atol=1e-3)
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "Output" / "P000" / "DVF.mha")))
    np.testing.assert_allclose(field[5, 6, 7], (3.0, 0.0, 0.0), atol=1e-4)  # the two composed
    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "Output" / "P000" / "Moved.mha")))
    np.testing.assert_allclose(moved[5, 6, 7], 256 * 5 + 16 * 6 + 7 + 3, atol=1e-3)


def test_the_tiles_get_a_fixed_mask_on_the_fixed_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """KonfAI cuts every group of a case at the same voxels: a fixed mask on another grid than the fixed image reached
    the tiles as it came, and KonfAI refused them once the whole global pass had run."""
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 16)
    mask = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.uint8))
    mask.SetSpacing((2.0, 2.0, 2.0))  # the fixed extent at 2 mm
    sitk.WriteImage(mask, str(tmp_path / "mask.mha"))
    tiling = {"global": "Prediction_global.yml", "tile": "Prediction_tile.yml"}
    monkeypatch.setattr(reg, "_manifest", lambda preset: {"ram_bytes_per_voxel": 1150, "tiling": tiling})
    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: int(1000 * 1150 / cost))
    seen: dict = {}

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        if kwargs["prediction_file"] == "Prediction_tile.yml":
            seen["mask"] = sitk.ReadImage(str(fixed_masks[0]))
        out = Path(work) / kwargs["label"] / "P000"
        out.mkdir(parents=True, exist_ok=True)
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (0.0, 0.0, 0.0), sitk.ReadImage(str(fixed_images[0])))}

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(
        ["P"],
        [fixed],
        [moving],
        fixed_masks=[tmp_path / "mask.mha"],
        output=tmp_path / "Output",
        max_voxels=1000,
        quiet=True,
        fields_only=True,
    )
    assert seen["mask"].GetSize() == (16, 16, 16) and seen["mask"].GetSpacing() == (1.0, 1.0, 1.0)


def test_a_preset_that_declares_no_tiling_registers_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 16)
    monkeypatch.setattr(reg, "_manifest", lambda preset: {})
    calls: list = []

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        calls.append(kwargs.get("prediction_file"))
        out = Path(work) / name / "P000"
        out.mkdir(parents=True, exist_ok=True)
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (0.0, 0.0, 0.0), sitk.ReadImage(str(fixed)))}

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(["P"], [fixed], [moving], output=tmp_path / "Output", max_voxels=1000, quiet=True)

    assert calls == [None]  # its own Prediction.yml, whole


# --------------------------------------------------------------------------- one grid where KonfAI patches or flips


@pytest.mark.parametrize(
    ("tiling", "flags", "onto_fixed"),
    [
        (None, {}, False),  # registered whole: the moving as it came, its field of view and resolution kept
        (None, {"tta": 2}, True),  # test-time flips
        ({"tile": "Prediction_tile.yml"}, {"max_voxels": 500}, True),  # tiles with no global pass before them
        # the global pass on the pair as it came, coarse; its tiles on the moving it pre-warped onto the fixed grid
        ({"global": "Prediction.yml", "tile": "Prediction_tile.yml"}, {"max_voxels": 500}, False),
    ],
)
def test_a_pair_on_two_grids_is_patched_or_flipped_only_on_the_fixed_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiling, flags, onto_fixed
) -> None:
    """KonfAI cuts each image into patches by voxel index and hands every patch the case's geometry: a moving on
    another grid had each patch registered against another region of the fixed. It now goes onto the fixed grid
    first, through the identity, where it would be patched or flipped as it came, and nowhere else."""
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((12, 12, 12), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 10)
    image = sitk.ReadImage(str(moving))
    image.SetSpacing((1.25, 1.25, 1.25))
    image.SetOrigin((-2.0, 1.0, 0.0))
    sitk.WriteImage(image, str(moving))
    manifest = {"ram_bytes_per_voxel": 1150, "tiling": tiling} if tiling else {}
    monkeypatch.setattr(reg, "_manifest", lambda preset: manifest)
    seen: list = []

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        seen.append((kwargs.get("prediction_file"), sitk.ReadImage(str(moving_images[0]))))
        grid = sitk.ReadImage(str(fixed_images[0]))
        out = Path(work) / (kwargs.get("label") or name) / "P000"
        out.mkdir(parents=True, exist_ok=True)
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (0.0, 0.0, 0.0), grid)}

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(["P"], [fixed], [moving], output=tmp_path / "Output", quiet=True, fields_only=True, **flags)

    first = seen[0][1]  # the moving the first pass saw: whole, flipped, cut in tiles, or coarse for a global pass
    face = np.array(first.GetOrigin()) - np.array(first.GetSpacing()) / 2
    np.testing.assert_allclose(face, (-0.5, -0.5, -0.5) if onto_fixed else (-2.625, 0.375, -0.625), atol=1e-6)
    if onto_fixed:
        patched = first
        # The identity: fixed voxel (x, y, z) = 1 mm steps from the origin reads the moving at that point.
        z, y, x = 4, 5, 6
        index = [(x + 2.0) / 1.25, (y - 1.0) / 1.25, z / 1.25]
        expected = 256 * index[2] + 16 * index[1] + index[0]
        np.testing.assert_allclose(sitk.GetArrayFromImage(patched)[z, y, x], expected, atol=1e-3)


def test_a_uint8_moving_is_interpolated_linearly(tmp_path: Path) -> None:
    """KonfAI takes a uint8 image for a label map and resamples it with nearest neighbour when nothing says
    otherwise: 8-bit microscopy, ultrasound or exported MR came out blocky, half a voxel off."""
    ramp = np.broadcast_to(20 * np.arange(8, dtype=np.uint8), (8, 8, 8)).copy()  # 20 x along x
    moving = tmp_path / "moving.mha"
    sitk.WriteImage(sitk.GetImageFromArray(ramp), str(moving))
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))
    app = reg.ImpactRegKonfAIApp()
    _stub_infer(app, fixed, {"P": (0.5, 0.0, 0.0)})
    app.register(["P"], [fixed], [moving], output=tmp_path / "Output", quiet=True)

    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "Output" / "P000" / "Moved.mha")))
    np.testing.assert_allclose(moved[4, 4, 2:6], [50, 70, 90, 110])  # 20 (x + 0.5), where nearest gave 20 (x + 1)


def _x_field(path: Path, size: int, spacing: float, dx) -> Path:
    """A field ``dx(x)`` along x on a cube of ``size`` voxels of ``spacing``, its outer faces those of the 1 mm
    grid of 16 voxels at the origin (the coarse copies are extent-aligned), as an image or a transform file."""
    x = (np.arange(size) + 0.5) * spacing - 0.5
    field = np.zeros((size, size, size, 3), np.float64)
    field[..., 0] = dx(x)[None, None, :]
    image = sitk.GetImageFromArray(field, isVector=True)
    image.SetSpacing([spacing] * 3)
    image.SetOrigin([(spacing - 1) / 2] * 3)
    if path.suffix == ".h5":
        sitk.WriteTransform(sitk.DisplacementFieldTransform(image), str(path))
    else:
        sitk.WriteImage(image, str(path))
    return path


def _stored_field(path: Path) -> sitk.Image:
    """The displacement field a preset stored, from a transform file or a field image."""
    if path.suffix == ".h5":
        return sitk.DisplacementFieldTransform(sitk.ReadTransform(str(path))).GetDisplacementField()
    return read_displacement_field(path)


@pytest.mark.parametrize("form", [".mha", ".h5"])
def test_the_tiles_field_goes_first_and_the_global_one_is_read_where_it_lands(tmp_path: Path, form: str) -> None:
    """``D(x) = D_tiles(x) + D_global(x + D_tiles(x))``: with a global field that varies (0.1 x) the order shows,
    which two constant fields commute past. Past the coarse grid's face the global field is its edge value, not
    the resampling's fill of 0 that dropped the whole global displacement in a shell along the faces."""
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    coarse = _x_field(tmp_path / f"global{form}", 4, 4.0, lambda x: 0.1 * x)  # on a 4 mm grid
    tiles = _x_field(tmp_path / f"tiles{form}", 16, 1.0, np.ones_like)  # +1 mm along x
    work = tmp_path / "work"
    work.mkdir()

    composed = reg.ImpactRegKonfAIApp()._compose(
        {"P000": tiles}, {"P000": coarse}, {"P000": fixed}, "DVF", [8] * 6, work, [], 1, True
    )["P000"]

    assert composed.suffix == form
    field = sitk.GetArrayFromImage(_stored_field(composed))[8, 8, :, 0]
    np.testing.assert_allclose(field[2:13], 1 + 0.1 * (np.arange(2, 13) + 1), atol=1e-4)  # not 0.1 x + 1
    assert field[15] == pytest.approx(1 + 0.1 * 13.5, abs=1e-4)  # the global field's edge value, where 1.0 was read


def test_the_global_field_is_padded_past_the_coarse_grid_and_the_longest_tile_displacement(tmp_path: Path) -> None:
    """The global field comes back on the native grid (KonfAI's mode: resample): its edge is sized in millimetres, at
    least 8 voxels of the grid the global pass ran on, and past the tiles' longest displacement."""
    tiles = _x_field(tmp_path / "tiles.mha", 16, 1.0, lambda x: 0.5 * x)  # up to 7.5 mm along x
    assert field_reach(*reg._field(tiles, tmp_path / "work")) == pytest.approx(7.5)
    from konfai.data.transform.resample import coarse_spacing

    headers = [{"Fixed": ([1, 32, 64, 64], [0.5, 0.5, 1.0], [0.0] * 3, [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])}]
    goal = coarse_spacing([32, 64, 64], [0.5, 0.5, 1.0], 8**3)[0]  # 32 mm cube at 8^3 voxels: about 4 mm
    assert 3.5 < goal <= 4.0
    # Eight coarse voxels, in fixed voxels of 0.5 mm along x and y and 1 mm along z.
    assert reg._compose_padding(headers, 8**3, 1.0) == [math.ceil(8 * goal / 0.5)] * 4 + [math.ceil(8 * goal)] * 2
    # A tile reaching 40 mm: 40 mm and one coarse voxel.
    assert reg._compose_padding(headers, 8**3, 40.0) == [math.ceil((40 + goal) / 0.5)] * 4 + [math.ceil(40 + goal)] * 2


def test_register_derives_the_moved_image_from_a_transform_file(tmp_path: Path) -> None:
    """Every published preset writes ``Transform.h5`` (an ITK displacement-field transform), where the stubs above
    write images: the moved image must come out of the transform file as well."""
    moving = _ramp(tmp_path / "moving.mha", 8)
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(fixed))
    app = reg.ImpactRegKonfAIApp()

    def transform(preset, fixed_i, moving_i, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True, exist_ok=True)
        return "Transform", {"P000": _x_field(out / "Transform.h5", 8, 1.0, lambda x: np.full_like(x, 2.0))}

    app._infer_preset = transform  # type: ignore[method-assign]
    app.register(["P"], [fixed], [moving], output=tmp_path / "Output", quiet=True)

    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "Output" / "P000" / "Moved.mha")))
    np.testing.assert_allclose(moved[1, 2, 3], 256 * 1 + 16 * 2 + 3 + 2, atol=1e-4)  # moving(x + 2 mm)
    assert (tmp_path / "Output" / "P000" / "Transform.h5").is_file()


def test_a_pass_out_of_gpu_memory_ends_the_run_and_the_global_pass_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KonfAI resamples or cuts a pass that runs out of GPU memory and runs it again; the exit code that reaches
    ImpactReg means it could not go smaller, and the run ends there."""
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 16)
    tiling = {"global": "Prediction_global.yml", "tile": "Prediction_tile.yml"}
    monkeypatch.setattr(reg, "_manifest", lambda preset: {"ram_bytes_per_voxel": 1, "tiling": tiling})
    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: int(1000 / cost))
    calls: list = []

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        calls.append(kwargs.get("prediction_file"))
        if kwargs.get("prediction_file") == "Prediction_tile.yml":
            raise reg._OutOfMemory("ImpactReg", "out of GPU memory")
        out = Path(work) / kwargs.get("label", name) / "P000"
        out.mkdir(parents=True, exist_ok=True)
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (1.0, 0.0, 0.0), sitk.ReadImage(str(fixed_images[0])))}

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    with pytest.raises(reg._OutOfMemory):
        app.register(["P"], [fixed], [moving], output=tmp_path / "Output", quiet=True, fields_only=True)
    assert calls == ["Prediction_global.yml", "Prediction_tile.yml"]


def test_a_pair_that_fits_is_registered_whole_under_its_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = tmp_path / "fixed.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(fixed))
    moving = _ramp(tmp_path / "moving.mha", 16)
    tiling = {"global": "Prediction.yml", "tile": "Prediction_tile.yml"}
    monkeypatch.setattr(reg, "_manifest", lambda preset: {"ram_bytes_per_voxel": 1, "tiling": tiling})
    monkeypatch.setattr("konfai.utils.vram.max_voxels", lambda cost, gpu: int(10**6 / cost))
    calls: list = []

    def preset(name, fixed_images, moving_images, fixed_masks, moving_masks, n_cases, work, *args, **kwargs):
        calls.append((kwargs.get("prediction_file"), kwargs.get("max_voxels")))
        out = Path(work) / name / "P000"
        out.mkdir(parents=True, exist_ok=True)
        return "DVF", {"P000": _write_dvf(out / "DVF.mha", (1.0, 0.0, 0.0), sitk.ReadImage(str(fixed_images[0])))}

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = preset  # type: ignore[method-assign]
    app.register(["P"], [fixed], [moving], output=tmp_path / "Output", quiet=True, fields_only=True)
    # One pass, its own config, capped where an out-of-memory it could not see coming would still be resampled.
    assert calls == [(None, 10**6)]


def test_a_preset_out_of_gpu_memory_raises_its_own_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from konfai.utils.errors import EXIT_OUT_OF_MEMORY

    def out_of_memory(command, **kwargs):
        raise subprocess.CalledProcessError(EXIT_OUT_OF_MEMORY, command)

    monkeypatch.setattr(reg.subprocess, "run", out_of_memory)
    app, arguments = reg.ImpactRegKonfAIApp(), ([tmp_path / "f.mha"], [tmp_path / "m.mha"], [], [], 1)
    with pytest.raises(reg._OutOfMemory):
        app._infer_preset("A", *arguments, tmp_path, [], None, True)
