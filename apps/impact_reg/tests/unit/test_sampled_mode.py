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

"""The sampled out-of-core mode: which presets it takes, how it lays its boxes, and what it hands the engine."""

import importlib
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai import impact_reg as reg
from konfai.utils.dataset import Dataset
from konfai.utils.errors import KonfAIError

pytest.importorskip("itk")

from impact_reg_konfai.models import convexadam

CONVEXADAM = "impact_reg_konfai.models.convexadam:RegistrationNet"
OTHER = "another_package.models.dense:RegistrationNet"
ELASTIX = "impact_reg_konfai.models.elastix:RegistrationNet"


def _prediction(classpath: str, model: Path) -> dict:
    return {
        "Predictor": {
            "Model": {
                "classpath": classpath,
                "RegistrationNet": {
                    "models": {"0": {"ref": str(model), "layers_mask": "1", "distance": "L2", "pca": 0}},
                    "iterations": 12,
                    "learning_rate": 1.5,
                    "grid_shrink": 4,
                    "control_grid_smoothing": 2,
                    "stages": ["fine"],
                    "seed": 7,
                },
            }
        }
    }


@pytest.fixture
def presets(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local preset repository: a ConvexAdam preset with a tile pass, two elastix ones and one of another engine."""
    repo = tmp_path_factory.mktemp("presets")
    model = repo / "model.pt"
    model.write_bytes(b"jit")
    for name, classpath in (("Convex", CONVEXADAM), ("Other", OTHER), ("Elx", ELASTIX), ("Elx2", ELASTIX)):
        (repo / name).mkdir()
        manifest = {"task": "registration", "vram_bytes_per_voxel": 100, "ram_bytes_per_voxel": 100}
        # What konfai-apps reads of an app it resolves, as the sampled route resolves it for its requirements.
        manifest.update(description=name, short_description=name, display_name=name, tta=0, mc_dropout=0)
        if name == "Convex":
            manifest["tiling"] = {"global": "Prediction.yml", "tile": "Prediction_tile.yml"}
        (repo / name / "app.json").write_text(json.dumps(manifest), encoding="utf-8")
        for filename in ("Prediction.yml", "Prediction_tile.yml"):
            (repo / name / filename).write_text(json.dumps(_prediction(classpath, model)), encoding="utf-8")
    monkeypatch.setattr(reg, "IMPACT_REG_KONFAI_REPO", str(repo))
    return repo


def test_region_geometry_counts_cells_and_boxes_as_itk_impact_does() -> None:
    geometry = convexadam.region_geometry([115, 157, 129], 2, 48)
    assert geometry["cells"] == [57, 78, 64]
    # A box of 48 voxels is 24 cells a side, the last one along an axis taking what is left: 3 x 4 x 3 boxes.
    assert geometry["regions"] == 36
    # A box of 29 voxels is 15 cells a side, rounded half up: 3 boxes along each axis of 45 cells.
    assert convexadam.region_geometry([90, 90, 90], 2, 29)["regions"] == 27


def test_two_grids_are_one_where_their_voxels_are_a_thousandth_of_a_voxel_apart() -> None:
    """A header rounded to float32 is the image's grid; a spacing 1e-4 mm off on a 0.04 mm grid of 1000 voxels, its
    last voxel 2.5 voxels away, is not."""
    grid = ([1, 1000, 1000, 1000], [0.04] * 3, [-12.3, 4.56, 7.89], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    rounded = (grid[0], *([float(np.float32(value)) for value in values] for values in grid[1:]))
    off = (grid[0], [0.0401, 0.04, 0.04], *grid[2:])
    assert reg._same_grid(grid, rounded) and not reg._same_grid(grid, off)


def test_the_engine_of_a_config_carries_the_fine_stage_parameters(
    presets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from impact_reg_konfai.models.intensity import engine_of

    monkeypatch.setattr(convexadam, "_itk_impact_predates_its_stages", lambda: False)  # whatever is installed here
    config = json.loads((presets / "Convex" / "Prediction_tile.yml").read_text(encoding="utf-8"))
    engine = engine_of(config, convexadam.RegistrationNet, convexadam.LevelSpec)
    assert engine._iterations == 12 and engine._learning_rate == 1.5 and engine._grid_shrink == 4
    assert engine._control_grid_smoothing == 2 and engine._seed == 7 and engine._stages == ["fine"]
    assert [spec.ref for spec in engine._stage_models["fine"]] == [str(presets / "model.pt")]


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    paths = []
    for name in ("fixed", "moving"):
        path = tmp_path / f"{name}.mha"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)), str(path))
        paths.append(path)
    return paths[0], paths[1]


@pytest.mark.parametrize("classpath", [OTHER, "custom.models.fireants:RegistrationNet"])
def test_sampled_mode_refuses_a_preset_of_an_engine_it_does_not_know(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classpath: str
) -> None:
    fixed, moving = _pair(tmp_path)
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: _prediction(classpath, presets / "model.pt"))
    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = lambda *args, **kwargs: pytest.fail("a preset ran")  # type: ignore[method-assign]
    with pytest.raises(KonfAIError, match="no stage that runs on a subset"):
        app.register(["Other"], [fixed], [moving], output=tmp_path / "out", out_of_core="sampled", max_voxels=100)


def test_sampled_mode_runs_the_engine_a_set_of_the_classpath_names(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An elastix preset whose --set names ConvexAdam's class registers with ConvexAdam, its other --set checked
    # against that class.
    fixed, moving = _pair(tmp_path)
    ran = []

    def convexadam(self, run):
        ran.append(run.config["Predictor"]["Model"]["RegistrationNet"]["grid_spacing"])
        raise KonfAIError("ImpactReg", "stop", "")

    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: json.loads(json.dumps(config)))
    monkeypatch.setattr(reg.ImpactRegKonfAIApp, "_sampled_convexadam", convexadam)
    monkeypatch.setattr(reg.ImpactRegKonfAIApp, "_sampled_elastix", lambda self, run: pytest.fail("elastix ran"))
    with pytest.raises(KonfAIError, match="stop"):
        reg.ImpactRegKonfAIApp().register(
            ["Elx"],
            [fixed],
            [moving],
            output=tmp_path / "out",
            out_of_core="sampled",
            config_overrides=[f"Predictor.Model.classpath={CONVEXADAM}", "grid_spacing=5"],
        )
    assert ran == [5]


@pytest.mark.parametrize(
    ("direction", "spacing", "form"),
    [
        ((1, 0, 0, 0, 1, 0, 0, 0, 1), (1.0, 2.0, 3.0), ".mha"),
        ((0, -1, 0, 0, 0, -1, -1, 0, 0), (2.0, 3.0, 1.0), ".nii.gz"),  # and written in the form asked for
    ],
    ids=["lps", "permuted and flipped"],
)
def test_sampled_mode_runs_elastix_on_the_twins_with_its_images_mapped(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction, spacing, form
) -> None:
    """The elastix route hands the engine the twins and the workspace, the pair measured in the fixed image's finest
    voxel and its voxel axes along LPS as a run on the whole pair is, and writes the transform it returns, in
    millimetres, as the field on the fixed image's own grid and beside it."""
    from impact_reg_konfai.models import intensity

    paths = []
    for name in ("fixed", "moving"):
        image = sitk.GetImageFromArray(np.random.default_rng(0).random((8, 9, 10)).astype(np.float32))
        image.SetSpacing((2.0, 4.0, 6.0))
        image.SetDirection(direction)
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    seen = {}

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "cof"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, fixed, moving, fixed_mask, moving_mask, device_index, work, spacing, voxels):
            seen.update(fixed=fixed, masks=(fixed_mask, moving_mask), spacing=spacing, voxels=voxels)
            seen["direction"] = sitk.ReadImage(str(fixed)).GetDirection()
            return sitk.TranslationTransform(3, (1.0, -2.0, 0.5))

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    app = reg.ImpactRegKonfAIApp()
    app.register(
        ["Elx"],
        [paths[0]],
        [paths[1]],
        output=tmp_path / "out",
        out_of_core="sampled",
        fields_only=True,
        output_format=[] if form == ".mha" else [form[1:]],
    )
    assert seen["fixed"].name == "Fixed.mha" and seen["masks"] == (None, None)
    assert seen["spacing"] == spacing and seen["voxels"] == 720  # in voxels of 2 mm, LPS's axes
    assert np.allclose(seen["direction"], np.eye(3).ravel())
    field = sitk.ReadImage(str(tmp_path / "out" / "P000" / f"DVF{form}"))
    assert field.GetSize() == (10, 9, 8) and field.GetSpacing() == (2.0, 4.0, 6.0)
    assert np.allclose(field.GetDirection(), direction) and np.allclose(field.GetOrigin(), (0.0, 0.0, 0.0))
    assert np.allclose(sitk.GetArrayFromImage(field), (2.0, -4.0, 1.0))
    transform = sitk.ReadTransform(str(tmp_path / "out" / "P000" / "Elx_Transform.h5"))
    assert np.allclose(transform.TransformPoint((0.0, 0.0, 0.0)), (2.0, -4.0, 1.0))


def test_sampled_elastix_registers_a_volume_one_voxel_thick_as_its_plane(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixed volume one voxel thick is registered as its plane, as a run on the whole pair registers it: elastix
    gets the 2-D twins of the fixed plane and of the moving image's first slice, measured in the plane's finest voxel,
    and the field comes back one voxel thick, along the dropped axis the shift from one plane to the other."""
    from impact_reg_konfai.models import intensity

    rng = np.random.default_rng(0)
    arrays = {"fixed": rng.random((1, 9, 10)).astype(np.float32), "moving": rng.random((3, 9, 10)).astype(np.float32)}
    paths = []
    for name, array in arrays.items():
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((2.0, 4.0, 0.5))  # the dropped axis has the finest voxel: the plane is measured in 2 mm
        image.SetOrigin((0.0, 0.0, 1.5 if name == "moving" else 0.0))
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    seen = {}

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "none"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, fixed, moving, fixed_mask, moving_mask, device_index, work, spacing, voxels):
            fixed_twin, moving_twin = sitk.ReadImage(str(fixed)), sitk.ReadImage(str(moving))
            seen.update(size=fixed_twin.GetSize(), spacing=spacing, voxels=voxels)
            seen["moving"] = sitk.GetArrayFromImage(moving_twin)
            return sitk.TranslationTransform(2, (1.0, -2.0))

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ["Elx"], [paths[0]], [paths[1]], output=tmp_path / "out", out_of_core="sampled", fields_only=True
    )
    assert seen["size"] == (10, 9) and seen["spacing"] == (1.0, 2.0) and seen["voxels"] == 90
    assert np.allclose(seen["moving"], arrays["moving"][0])  # the moving image's first slice
    field = sitk.ReadImage(str(tmp_path / "out" / "P000" / "DVF.mha"))
    assert field.GetSize() == (10, 9, 1) and field.GetSpacing() == (2.0, 4.0, 0.5)
    assert np.allclose(sitk.GetArrayFromImage(field), (2.0, -4.0, 1.5))  # the plane's (dx, dy), then the z shift
    assert not (tmp_path / "out" / "P000" / "Elx_Transform.h5").exists()  # a plane's transform is its field


def test_sampled_convexadam_registers_a_volume_one_voxel_thick_as_its_plane(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ConvexAdam by regions takes a fixed volume one voxel thick as its plane too: chunked 2-D twins of the fixed
    plane and of the moving image's first slice, in the plane's finest voxel, and the field lifted one voxel thick."""
    from types import SimpleNamespace

    from impact_reg_konfai.models import intensity
    from impact_reg_konfai.native import field_map, open_twin

    rng = np.random.default_rng(0)
    paths = []
    for name, depth in (("fixed", 1), ("moving", 3)):
        image = sitk.GetImageFromArray(rng.random((depth, 9, 10)).astype(np.float32))
        image.SetSpacing((2.0, 4.0, 0.5))
        image.SetOrigin((0.0, 0.0, 1.5 if name == "moving" else 0.0))
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    seen = {}

    class Engine:
        _iterations, _grid_shrink, _stages, _moments_init, _deformable_masked = 3, 2, ("fine",), "none", True

        def check_gradients(self) -> None:
            pass

        def refine_by_regions(self, fixed, moving, fixed_mask, moving_mask, device, region_size, visits, field, **_):
            seen.update(
                fixed=fixed.suffix, size=open_twin(fixed).array.shape, spacing=open_twin(fixed).geometry.GetSpacing()
            )
            target = field_map(field, open_twin(fixed))
            target[:] = (1.0, -2.0)
            target.flush()
            grid = field.with_name("grid.mha")
            grid.write_bytes(b"")
            return SimpleNamespace(
                control_grid=grid, resident=True, iterations=visits, regions=1, bytes_read=0, streamed=(True, True),
                seconds=0.0, device_peak_bytes=0, metric_values=[], coarse=None,
            )  # fmt: skip

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(convexadam, "require_regions", lambda: None)
    config = json.loads((presets / "Convex" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ["Convex"], [paths[0]], [paths[1]], output=tmp_path / "out", out_of_core="sampled", fields_only=True
    )
    assert seen["fixed"] == ".h5" and seen["size"] == (9, 10) and seen["spacing"] == (1.0, 2.0)
    field = sitk.ReadImage(str(tmp_path / "out" / "P000" / "DVF.mha"))
    assert field.GetSize() == (10, 9, 1) and np.allclose(sitk.GetArrayFromImage(field), (2.0, -4.0, 1.5))
    assert not list((tmp_path / "out" / "P000").glob("*ControlGrid*"))  # a plane's transform is its field


def _masked_cohort(tmp_path: Path, empty: str) -> dict[str, list[Path]]:
    """Two cases: masks of ones on both sides for the first, the ``empty`` side's mask empty for the second."""
    paths: dict[str, list[Path]] = {"fixed": [], "moving": [], "fixed_mask": [], "moving_mask": []}
    for case in range(2):
        for group in paths:
            path = tmp_path / f"{group}{case}.mha"
            if group.endswith("mask"):
                array = np.full((8, 8, 8), int(case == 0 or not group.startswith(empty)), dtype=np.uint8)
            else:
                array = np.random.default_rng(case).random((8, 8, 8)).astype(np.float32)
            sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
            paths[group].append(path)
    return paths


@pytest.mark.parametrize("side", ["fixed", "moving"])
def test_sampled_mode_takes_an_empty_mask_as_the_engines_take_it_on_the_whole_pair(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    """An empty fixed mask leaves nothing to register: a zero field, its images never written. An empty moving mask
    restricts nothing: the case registers as one without it."""
    from impact_reg_konfai import native
    from impact_reg_konfai.models import intensity

    paths = _masked_cohort(tmp_path, side)
    registered = []

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "cof"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, fixed, moving, fixed_mask, moving_mask, *args) -> sitk.Transform:
            registered.append((fixed_mask, moving_mask))
            return sitk.TranslationTransform(3, (1.0, 0.0, 0.0))

    written = []
    write_twin = native.write_twin
    monkeypatch.setattr(
        native,
        "write_twin",
        lambda dataset, group, case, *args, **kwargs: (
            written.append((group, case)),
            write_twin(dataset, group, case, *args, **kwargs),
        )[1],
    )
    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ["Elx"],
        paths["fixed"],
        paths["moving"],
        paths["fixed_mask"],
        paths["moving_mask"],
        output=tmp_path / "out",
        out_of_core="sampled",
        max_voxels=100,
        fields_only=True,
    )
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "P001" / "DVF.mha")))
    if side == "fixed":
        assert registered == [(None, None)] and not np.any(field)
        assert ("Fixed", "P001") not in written and ("Moving", "P001") not in written
    else:
        assert registered == [(None, None), (None, None)] and np.allclose(field, (1.0, 0.0, 0.0))


def test_sampled_mode_takes_the_test_time_copies_a_run_of_the_preset_whole_takes(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--tta by boxes, as the predictor runs it: each copy the pair flipped as the preset's Flip flips it in turn, the
    headers kept, its field flipped back with its vectors reflected, and the case's field the mean of the original's
    and the copies'. The engine here gives the translation between the images' centres of mass."""
    from impact_reg_konfai.models import intensity

    def blob(centre, origin) -> sitk.Image:
        z, y, x = np.mgrid[:9, :10, :11].astype(np.float32)
        array = np.exp(-((x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2) / 4.0).astype(np.float32)
        image = sitk.GetImageFromArray(array)
        image.SetOrigin(origin)
        return image

    def centre_of_mass(image: sitk.Image, mirrored=()) -> np.ndarray:
        array = sitk.GetArrayFromImage(image)
        array = np.flip(array, [2 - c for c in mirrored]) if mirrored else array  # ITK component c: NumPy axis 2 - c
        indices = np.stack(np.meshgrid(*(np.arange(n) for n in array.shape), indexing="ij"), -1)[..., ::-1]
        mean = (indices * array[..., None]).sum((0, 1, 2)) / array.sum()
        return np.asarray(image.TransformContinuousIndexToPhysicalPoint(mean.tolist()))

    images = {"fixed": blob((4.0, 5.0, 3.5), (0.0, 0.0, 0.0)), "moving": blob((6.0, 3.5, 4.0), (1.5, -2.0, 0.5))}
    paths = {}
    for name, image in images.items():
        paths[name] = tmp_path / f"{name}.mha"
        sitk.WriteImage(image, str(paths[name]))

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "cof"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, fixed, moving, *args) -> sitk.Transform:
            shift = centre_of_mass(sitk.ReadImage(str(moving))) - centre_of_mass(sitk.ReadImage(str(fixed)))
            return sitk.TranslationTransform(3, shift.tolist())

    manifest = json.loads((presets / "Elx" / "app.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(reg, "_manifest", lambda preset: {**manifest, "tta": 2})  # an app that allows two copies
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    flip = {"Flip": {"f_prob": [0, 0.5, 0.5], "vector_field": True, "prob": 1}}
    config["Predictor"]["Dataset"] = {"augmentations": {"DataAugmentation_0": {"data_augmentations": flip, "nb": 2}}}
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF", "reduction": "Mean"}}}
    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ["Elx"], [paths["fixed"]], [paths["moving"]], output=tmp_path / "out", out_of_core="sampled", tta=2,
        fields_only=True,
    )  # fmt: skip
    # The copies flip y, then x (tensor dims 2 and 3): ITK components 1 and 0.
    shifts = []
    for mirrored in ((), (1,), (0,)):
        shift = centre_of_mass(images["moving"], mirrored) - centre_of_mass(images["fixed"], mirrored)
        shift[list(mirrored)] *= -1
        shifts.append(shift)
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "P000" / "DVF.mha")))
    assert np.allclose(field, np.mean(shifts, axis=0), atol=1e-4)
    assert not np.allclose(field, shifts[0], atol=1e-2)  # the copies count


def test_sampled_convexadam_stages_no_mask_its_stages_do_not_read(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With deformable_masked off ConvexAdam drops the masks on the whole pair, an empty one included: by regions
    too, the case registers as one without masks."""
    from types import SimpleNamespace

    from impact_reg_konfai.models import intensity
    from impact_reg_konfai.native import field_map, open_twin

    paths = _masked_cohort(tmp_path, "fixed")
    registered = []

    class Engine:
        _iterations, _grid_shrink, _stages, _moments_init, _deformable_masked = 3, 2, ("fine",), "none", False

        def check_gradients(self) -> None:
            pass

        def refine_by_regions(self, fixed, moving, fixed_mask, moving_mask, device, region_size, visits, field, **_):
            registered.append((fixed_mask, moving_mask))
            grid = field.with_name("grid.mha")
            for path in (field, grid):
                field_map(path, open_twin(fixed)).flush()
            return SimpleNamespace(
                control_grid=grid, resident=True, iterations=visits, regions=1, bytes_read=0, streamed=(True, True),
                seconds=0.0, device_peak_bytes=0, metric_values=[], coarse=None,
            )  # fmt: skip

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(convexadam, "require_regions", lambda: None)
    config = json.loads((presets / "Convex" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ["Convex"],
        paths["fixed"],
        paths["moving"],
        paths["fixed_mask"],
        paths["moving_mask"],
        output=tmp_path / "out",
        out_of_core="sampled",
        max_voxels=100,
        fields_only=True,
    )
    assert registered == [(None, None), (None, None)]


def test_sampled_elastix_starts_from_the_translation_between_the_centres_of_mass(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With moments_init com the moving twin goes there by its origin, as the engine moves a pair registered whole,
    and the transform carries the shift."""
    from impact_reg_konfai.models import intensity

    paths = []
    for name, start in (("fixed", (2, 2, 2)), ("moving", (3, 4, 5))):
        array = np.zeros((10, 10, 10), dtype=np.float32)
        array[start[2] : start[2] + 3, start[1] : start[1] + 3, start[0] : start[0] + 3] = 100.0  # (z, y, x)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((2.0, 2.0, 2.0))
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    seen = {}

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "com"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, fixed, moving, fixed_mask, moving_mask, device_index, work, spacing, voxels):
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(moving))
            reader.ReadImageInformation()
            seen["origin"] = reader.GetOrigin()
            return sitk.TranslationTransform(3, (0.0, 0.0, 0.0))

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ["Elx"],
        [paths[0]],
        [paths[1]],
        output=tmp_path / "out",
        out_of_core="sampled",
        max_voxels=100,
        fields_only=True,
    )

    assert np.allclose(seen["origin"], (-1.0, -2.0, -3.0))  # the centres 1, 2 and 3 voxels apart
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(next((tmp_path / "out" / "P000").glob("*.mha")))))
    assert np.allclose(field, (2.0, 4.0, 6.0))  # millimetres


def test_sampled_mode_refuses_the_former_affine_pre_alignment_by_name(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed, moving = _pair(tmp_path)
    config = json.loads((presets / "Convex" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["linear"] = True
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    app = reg.ImpactRegKonfAIApp()
    app._native_twins = lambda *args, **kwargs: pytest.fail("the twins were written")  # type: ignore[method-assign]
    with pytest.raises(KonfAIError, match="moments_init"):
        app.register(["Convex"], [fixed], [moving], output=tmp_path / "out", out_of_core="sampled", max_voxels=100)


def test_sampled_convexadam_probes_its_jacobian_networks_before_the_twins(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from konfai.metric.measure.impact import ImpactFeatureModel
    from konfai.utils.errors import MeasureError

    fixed, moving = _pair(tmp_path)
    config = json.loads((presets / "Convex" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["mode"] = "Jacobian"
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    monkeypatch.setattr(convexadam, "_itk_impact_predates_its_stages", lambda: False)

    def check(self, gradient: bool = False) -> None:
        if gradient:
            raise MeasureError("a weighted layer has no gradient")

    monkeypatch.setattr(ImpactFeatureModel, "check", check)
    app = reg.ImpactRegKonfAIApp()
    app._native_twins = lambda *args, **kwargs: pytest.fail("the twins were written")  # type: ignore[method-assign]
    with pytest.raises(MeasureError, match="no gradient"):
        app.register(["Convex"], [fixed], [moving], output=tmp_path / "out", out_of_core="sampled", max_voxels=100)


@pytest.mark.parametrize(
    "direction",
    [(-1, 0, 0, 0, -1, 0, 0, 0, 1), (0, 0, 1, 1, 0, 0, 0, -1, 0), (0, -1, 0, 0, 0, -1, -1, 0, 0)],
    ids=["flipped", "permuted", "permuted and flipped"],
)
@pytest.mark.parametrize("suffix", [".mha", ".h5"])
def test_a_twin_is_the_pair_a_whole_run_reorients_and_its_field_goes_back(tmp_path: Path, direction, suffix) -> None:
    """A twin written through ``AxisMap.to_lps`` holds the voxels and the geometry SimpleITK's DICOMOrient gives a run
    on the whole pair, and ``remap`` through the inverse puts an image of it back on the input's grid."""
    from impact_reg_konfai.native import AxisMap, remap, write_twin

    rng = np.random.default_rng(0)
    image = sitk.GetImageFromArray(rng.random((5, 6, 7)).astype(np.float32))
    image.SetSpacing((0.5, 0.75, 1.25))
    image.SetOrigin((3.0, -2.0, 7.0))
    image.SetDirection(direction)
    sitk.WriteImage(image, str(tmp_path / "image.mha"))
    root, _, backend = reg._stage_group(tmp_path / "stage", "Fixed", {"P000": tmp_path / "image.mha"}).rpartition(":")
    axis_map = AxisMap.to_lps(direction)
    twin = write_twin(
        Dataset(root, backend), "Fixed", "P000", tmp_path / f"twin{suffix}", dtype="float32", axis_map=axis_map
    )
    oriented, written = sitk.DICOMOrient(image, "LPS"), sitk.ReadImage(str(twin))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(written), sitk.GetArrayFromImage(oriented))
    for key in ("GetSpacing", "GetOrigin", "GetDirection"):
        np.testing.assert_allclose(getattr(written, key)(), getattr(oriented, key)(), atol=1e-6)
    if suffix == ".mha":
        remap(twin, axis_map.inverse())
        back = sitk.ReadImage(str(twin))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(image))
        for key in ("GetSpacing", "GetOrigin", "GetDirection"):
            np.testing.assert_allclose(getattr(back, key)(), getattr(image, key)(), atol=1e-6)


def test_a_mask_twin_keeps_what_its_engine_reads_of_the_mask(tmp_path: Path) -> None:
    """FireANTs reads a mask's positive voxels, the other engines its non-zero ones: so does each one's twin."""
    from impact_reg_konfai.native import write_twin

    mask = np.zeros((4, 5, 6), np.float32)
    mask[1], mask[2] = -1.0, 1.0
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "Mask.mha"))
    root, _, backend = reg._stage_group(tmp_path / "stage", "FixedMask", {"P000": tmp_path / "Mask.mha"}).rpartition(
        ":"
    )

    def twin(positive: bool) -> np.ndarray:
        target = tmp_path / f"{positive}.mha"
        write_twin(Dataset(root, backend), "FixedMask", "P000", target, dtype="uint8", positive=positive)
        return sitk.GetArrayFromImage(sitk.ReadImage(str(target)))

    assert np.array_equal(twin(False), mask != 0) and np.array_equal(twin(True), mask > 0)


def test_a_compact_transform_never_outlives_its_field(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compact transform beside a case's field is that field's: a run whose field fails to publish leaves none of
    its own beside the earlier field, and a run that publishes another field leaves no earlier transform beside it.
    A file the runs did not write stays, whatever its name."""
    from impact_reg_konfai.models import intensity

    fixed, moving = _pair(tmp_path)

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "none"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, *args) -> sitk.Transform:
            return sitk.TranslationTransform(3, (1.0, 0.0, 0.0))

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    out, compact = tmp_path / "out", tmp_path / "out" / "P000" / "Elx_Transform.h5"

    def run(app: reg.ImpactRegKonfAIApp, **options) -> None:
        app.register(["Elx"], [fixed], [moving], output=out, fields_only=True, **options)

    run(reg.ImpactRegKonfAIApp(), out_of_core="sampled", max_voxels=100)
    assert compact.exists()
    own = out / "P000" / "manual_Transform.h5"
    sitk.WriteTransform(sitk.TranslationTransform(3), str(own))
    publish = reg._copy_output

    def fails(*args, **kwargs):
        raise OSError("no space left")

    monkeypatch.setattr(reg, "_copy_output", fails)
    with pytest.raises(OSError, match="no space left"):
        run(reg.ImpactRegKonfAIApp(), out_of_core="sampled", max_voxels=100)
    assert not compact.exists() and (out / "P000" / "DVF.mha").exists()
    monkeypatch.setattr(reg, "_copy_output", publish)
    run(reg.ImpactRegKonfAIApp(), out_of_core="sampled", max_voxels=100)

    def whole(preset, fixed_images, moving_images, fixed_masks, moving_masks, work, *args, **kwargs):
        field = Path(work) / preset / "P000" / "DVF.mha"
        field.parent.mkdir(parents=True)
        shutil.copy(out / "P000" / "DVF.mha", field)
        return "DVF", {"P000": field}

    app = reg.ImpactRegKonfAIApp()
    app._infer_preset = whole  # type: ignore[method-assign]
    run(app)
    assert not compact.exists() and (out / "P000" / "DVF.mha").exists() and own.exists()


@pytest.mark.parametrize("allowed", [0, 2])
def test_tta_on_the_sampled_route_is_capped_as_konfai_apps_caps_it(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowed: int
) -> None:
    """--tta runs the copies the app allows (app.json 'tta', 0 for every shipped preset), said when it caps them, as
    konfai-apps says it on the other routes."""

    class Reached(Exception):
        pass

    copies = []

    def sampled(*args, **kwargs):
        copies.append(args[-1])
        raise Reached

    manifest = reg._manifest
    monkeypatch.setattr(reg, "_manifest", lambda preset: {**manifest(preset), "tta": allowed})
    fixed, moving = _pair(tmp_path)
    app = reg.ImpactRegKonfAIApp()
    app._register_sampled = sampled  # type: ignore[method-assign]
    with pytest.raises(Reached):
        app.register(["Convex"], [fixed], [moving], output=tmp_path / "out", out_of_core="sampled", tta=3)
    assert copies == [min(3, allowed)]


@pytest.mark.parametrize(
    ("kind", "published"),
    [("itktransform", (".h5", "itktransform")), ("mha.gz", (".mha", "mha.gz")), ("zarr", (".ome.zarr", "omezarr"))],
)
def test_an_asked_form_is_the_file_its_writer_publishes(kind, published) -> None:
    assert reg._asked_form(kind) == published


@pytest.mark.parametrize(
    ("seeds", "classpath", "share"),
    [
        ((7, 8), "custom:RegistrationNet", ["seed=3"]),
        ((None, None), "custom:RegistrationNet", []),
        ((None, None), "impact_reg_konfai.models.convexadam:RegistrationNet", ["seed=3"]),
    ],
    ids=["set", "none", "engine-default"],
)
def test_seed_seeds_each_pass_whose_engine_draws_with_one(
    monkeypatch: pytest.MonkeyPatch, seeds, classpath, share
) -> None:
    """The global and the tile pass each take --seed where their engine has a seed, whatever its default there: the
    config's own, or the one every engine of ImpactReg takes when the config leaves it out."""

    def config(preset, name):
        seed = seeds[name == "Prediction_tile.yml"]
        network = {} if seed is None else {"seed": seed}
        return {"Predictor": {"Model": {"classpath": classpath, "RegistrationNet": network}}}

    monkeypatch.setattr(reg, "_preset_config", config)
    monkeypatch.setattr(reg, "_manifest", lambda preset: {"tiling": {"tile": "Prediction_tile.yml"}})
    shares = reg._preset_overrides(["P"], [], quiet=True, seed=3)
    assert shares["P"]["global"] == share and shares["P"]["tile"] == share


@pytest.mark.parametrize("engine", reg._SAMPLED_ENGINES)
def test_every_engine_of_impact_reg_takes_a_seed(engine: str) -> None:
    # --seed seeds an engine of this package whose config leaves its seed out (_draws_with_seed).
    network = importlib.import_module(f"impact_reg_konfai.models.{engine}").RegistrationNet
    assert "seed" in inspect.signature(network).parameters


@pytest.mark.parametrize("ensemble", [["Elx"], ["Elx", "Elx2"]], ids=["preset", "ensemble"])
def test_sampled_mode_registers_a_plane_on_its_own_grid(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ensemble: list[str]
) -> None:
    """A 2-D pair is registered sampled as a volume is, on its own grid, never lifted to a plane thick volume (which
    reads it whole): the elastix route hands the engine 2-D twins along LPS and writes a 2-D field on the fixed grid,
    compressed when asked, an ensemble with its members' stack under --uncertainty."""
    from impact_reg_konfai.models import intensity

    paths = []
    for name in ("fixed", "moving"):
        image = sitk.GetImageFromArray(np.random.default_rng(0).random((60, 70)).astype(np.float32))
        image.SetSpacing((2.0, 4.0))
        image.SetDirection((-1.0, 0.0, 0.0, 1.0))
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    seen = {}

    class Engine:
        _feature_models = ()
        _elastix_bin = "elastix"
        _moments_init = "none"

        def supports_mapped_images(self) -> bool:
            return True

        def register_mapped(self, fixed, moving, fixed_mask, moving_mask, device_index, work, spacing, voxels):
            twin = sitk.ReadImage(str(fixed))
            seen.update(size=twin.GetSize(), direction=twin.GetDirection(), spacing=spacing)
            return sitk.TranslationTransform(2, (1.0, -2.0))

    monkeypatch.setattr(intensity, "engine_of", lambda *args: Engine())
    monkeypatch.setattr(reg, "app_file", lambda repo, preset, name: tmp_path / name)
    monkeypatch.setattr(reg, "lift_plane", lambda *args: pytest.fail("the plane was lifted"))
    config = json.loads((presets / "Elx" / "Prediction.yml").read_text(encoding="utf-8"))
    config["Predictor"]["Model"]["RegistrationNet"]["parameter_maps"] = ["Rigid.txt"]
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    reg.ImpactRegKonfAIApp().register(
        ensemble,
        [paths[0]],
        [paths[1]],
        output=tmp_path / "out",
        out_of_core="sampled",
        fields_only=True,
        uncertainty=len(ensemble) > 1,
        output_format=["mha.gz"],
    )
    assert (
        seen["size"] == (70, 60) and np.allclose(seen["direction"], np.eye(2).ravel()) and seen["spacing"] == (1.0, 2.0)
    )
    field = sitk.ReadImage(str(tmp_path / "out" / "P000" / "DVF.mha"))
    assert (
        field.GetDimension() == 2 and field.GetSize() == (70, 60) and np.allclose(field.GetDirection(), (-1, 0, 0, 1))
    )
    assert np.allclose(sitk.GetArrayFromImage(field), (2.0, -4.0))
    assert b"CompressedData = True" in (tmp_path / "out" / "P000" / "DVF.mha").read_bytes()[:1024]
    if len(ensemble) > 1:
        stack = sitk.ReadImage(str(tmp_path / "out" / "P000" / "InferenceStack.mha"))  # members x components
        assert stack.GetSize() == (70, 60) and stack.GetNumberOfComponentsPerPixel() == 4
        assert stack.GetMetaData("konfai_members") == "2"


def test_sampled_mode_runs_the_region_stage_on_a_pair_over_the_budget(presets: Path, tmp_path: Path) -> None:
    fixed, moving = _pair(tmp_path)
    app = reg.ImpactRegKonfAIApp()
    seen = {}

    def sampled(preset, inputs, headers, cases, work, gpu, cpu, quiet, overrides, region, samples, epochs, tta):
        seen.update(preset=preset, region=region, samples=samples, epochs=epochs, gpu=gpu)
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True)
        reference = sitk.ReadImage(str(fixed))
        field = sitk.Image(reference.GetSize(), sitk.sitkVectorFloat32, 3)
        field.CopyInformation(reference)
        sitk.WriteImage(field, str(out / "DVF.mha"))
        return "DVF", {"P000": out / "DVF.mha"}

    app._register_sampled = sampled  # type: ignore[method-assign]
    app._infer_preset = lambda *args, **kwargs: pytest.fail("the preset ran whole")  # type: ignore[method-assign]
    app.register(
        ["Convex"],
        [fixed],
        [moving],
        output=tmp_path / "out",
        out_of_core="sampled",
        max_voxels=100,
        region_size=32,
        samples=3,
        epochs=5,
        fields_only=True,
    )
    assert seen == {"preset": "Convex", "region": 32, "samples": 3, "epochs": 5, "gpu": []}
    assert (tmp_path / "out" / "P000" / "DVF.mha").exists()


def test_auto_registers_whole_a_pair_the_device_holds(presets: Path, tmp_path: Path) -> None:
    fixed, moving = _pair(tmp_path)
    app = reg.ImpactRegKonfAIApp()
    app._register_sampled = lambda *args, **kwargs: pytest.fail("the region stage ran")  # type: ignore[method-assign]

    def whole(preset, fixed_images, moving_images, fixed_masks, moving_masks, work, *args, **kwargs):
        out = Path(work) / preset / "P000"
        out.mkdir(parents=True)
        reference = sitk.ReadImage(str(fixed))
        field = sitk.Image(reference.GetSize(), sitk.sitkVectorFloat32, 3)
        field.CopyInformation(reference)
        sitk.WriteImage(field, str(out / "DVF.mha"))
        return "DVF", {"P000": out / "DVF.mha"}

    app._infer_preset = whole  # type: ignore[method-assign]
    app.register(["Convex"], [fixed], [moving], output=tmp_path / "out", out_of_core="auto", fields_only=True)
    assert (tmp_path / "out" / "P000" / "DVF.mha").exists()


def test_auto_registers_whole_the_cases_the_device_holds(presets: Path, tmp_path: Path) -> None:
    """A cohort is routed case by case: a pair over the budget registers sampled, a smaller one whole as it would
    alone (an oblique one included, which the sampled route refuses for features), each under its own case ID, and
    the moved image of each comes through its own form of transform (a dense field, a native affine)."""
    paths = {}
    for case, size, direction in (("large", 12, None), ("small", 8, (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))):
        for side in ("fixed", "moving"):
            image = sitk.GetImageFromArray(np.zeros((size,) * 3, dtype=np.float32))
            if direction is not None:
                image.SetDirection(direction)
            paths[case, side] = tmp_path / f"{case}_{side}.mha"
            sitk.WriteImage(image, str(paths[case, side]))
    seen = {}

    def sampled(preset, inputs, headers, cases, work, *args):
        seen["sampled"] = (cases, inputs["Fixed"])
        fields = {}
        for case, fixed in zip(cases, inputs["Fixed"], strict=True):
            fields[case] = Path(work) / "sampled" / case / "DVF.mha"
            fields[case].parent.mkdir(parents=True)
            image = sitk.ReadImage(str(fixed))
            vectors = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, 3)
            vectors.CopyInformation(image)
            sitk.WriteImage(vectors, str(fields[case]))
        return "DVF", fields

    def whole(preset, fixed_images, moving_images, fixed_masks, moving_masks, work, *args, **kwargs):
        seen["whole"] = fixed_images
        transforms = {}
        for index in range(len(fixed_images)):
            transforms[f"P{index:03d}"] = Path(work) / "whole" / f"P{index:03d}" / "DVF.h5"
            transforms[f"P{index:03d}"].parent.mkdir(parents=True)
            sitk.WriteTransform(sitk.AffineTransform(3), str(transforms[f"P{index:03d}"]))
        return "DVF", transforms

    app = reg.ImpactRegKonfAIApp()
    app._register_sampled = sampled  # type: ignore[method-assign]
    app._infer_preset = whole  # type: ignore[method-assign]
    app.register(
        ["Convex"],
        [paths["large", "fixed"], paths["small", "fixed"]],
        [paths["large", "moving"], paths["small", "moving"]],
        output=tmp_path / "out",
        out_of_core="auto",
        max_voxels=1000,
    )
    assert seen == {"sampled": (["P000"], [paths["large", "fixed"]]), "whole": [paths["small", "fixed"]]}
    out = tmp_path / "out"
    assert (
        sitk.ReadImage(str(out / "P000" / "DVF.mha")).GetSize() == (12, 12, 12) and (out / "P001" / "DVF.h5").exists()
    )
    sizes = [sitk.ReadImage(str(out / case / "Moved.mha")).GetSize() for case in ("P000", "P001")]
    assert sizes == [(12, 12, 12), (8, 8, 8)]


def test_the_hdf5_twin_streams_the_entry_in_itk_s_layout(tmp_path: Path) -> None:
    import itk
    from impact_reg_konfai.native import mmap_image, write_twin

    array = np.arange(40 * 36 * 44, dtype=np.float32).reshape(40, 36, 44)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 2.0, 2.5))
    image.SetOrigin((10.0, -5.0, 3.0))
    image.SetDirection((0, 1, 0, -1, 0, 0, 0, 0, 1))
    source = tmp_path / "image.mha"
    sitk.WriteImage(image, str(source))
    root, _, backend = reg._stage_group(tmp_path / "stage", "Fixed", {"P000": source}).rpartition(":")
    dataset = Dataset(root, backend)
    low, high = (float(value) for value in np.percentile(array, [40, 60]))

    raw = write_twin(dataset, "Fixed", "P000", tmp_path / "twin.mha", dtype="float32", clip=(low, high), band=16)
    twin = write_twin(dataset, "Fixed", "P000", tmp_path / "twin.h5", dtype="float32", band=16)
    mapped = mmap_image(raw)
    assert np.array_equal(mapped.array, np.clip(array, low, high))
    assert (raw.stat().st_size - mapped.array.nbytes) % 8 == 0  # the pixels are aligned: elastix maps them
    assert mapped.geometry.GetSpacing() == (1.5, 2.0, 2.5) and mapped.geometry.GetOrigin() == (10.0, -5.0, 3.0)
    assert np.allclose(sitk.ReadImage(str(raw)).GetDirection(), image.GetDirection())
    reader = itk.ImageFileReader[itk.Image[itk.F, 3]].New(FileName=str(twin))
    reader.UseStreamingOn()
    reader.UpdateOutputInformation()
    out = reader.GetOutput()
    region = itk.ImageRegion[3]()
    region.SetIndex([5, 7, 9])
    region.SetSize([20, 10, 8])
    out.SetRequestedRegion(region)
    reader.Update()
    assert list(out.GetBufferedRegion().GetSize()) == [20, 10, 8]
    got = itk.array_view_from_image(out, update=False)
    assert np.array_equal(got, array[9:17, 7:17, 5:25])
    read = sitk.ReadImage(str(twin))
    assert read.GetSpacing() == image.GetSpacing() and read.GetOrigin() == image.GetOrigin()
    assert np.allclose(read.GetDirection(), image.GetDirection())

    mask_raw, mask = (
        write_twin(dataset, "Fixed", "P000", tmp_path / f"mask{form}", dtype="uint8") for form in (".mha", ".h5")
    )
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(mask))), array != 0)
    assert np.array_equal(mmap_image(mask_raw).array, array != 0)


def test_a_twin_of_a_plane_larger_than_a_band_is_written_by_rows(tmp_path: Path, monkeypatch) -> None:
    """A plane alone over the band budget (a light-sheet section) is read and written by rows: no read takes more than
    a band, and both twins hold the image."""
    from impact_reg_konfai import native

    array = np.random.default_rng(3).random((3, 20, 30)).astype(np.float32)
    source = tmp_path / "image.mha"
    sitk.WriteImage(sitk.GetImageFromArray(array), str(source))
    root, _, backend = reg._stage_group(tmp_path / "stage", "Fixed", {"P000": source}).rpartition(":")
    dataset = Dataset(root, backend)
    budget = 30 * 4 * 7  # seven rows of the image
    monkeypatch.setattr(native, "_BAND_BYTES", budget)
    read = dataset.read_data_slice
    sizes = []

    def recorded(group, name, slices):
        values = read(group, name, slices)
        sizes.append(np.asarray(values[0]).nbytes)
        return values

    monkeypatch.setattr(dataset, "read_data_slice", recorded)
    raw, chunked = (
        native.write_twin(dataset, "Fixed", "P000", tmp_path / f"t{form}", dtype="float32") for form in (".mha", ".h5")
    )
    assert max(sizes) <= budget
    assert np.array_equal(native.mmap_image(raw).array, array)
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(chunked))), array)


def test_a_twin_gives_the_centres_a_pair_registered_whole_starts_from(tmp_path: Path) -> None:
    """The sampled route starts its linear stage where the whole run starts it: the centre of mass a twin gives, chunked
    or memory-mapped, is the one ``intensity.center_of_mass_translation`` takes on the image whole, to the bit (an
    origin off by 1e-8 voxel moves elastix's field by 0.08 mm on a lung pair)."""
    from impact_reg_konfai.models.intensity import center_of_mass_translation
    from impact_reg_konfai.native import hdf5_twin, mapped_centre, mmap_image, write_twin

    array = np.random.default_rng(0).normal(0.0, 5.0, (40, 36, 44)).astype(np.float32)
    array[8:20, 10:30, 25:40] += 100.0  # a block off the frame's centre, on noise the winsorising clamps
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 2.0, 2.5))
    image.SetOrigin((10.0, -5.0, 3.0))
    image.SetDirection((0, 1, 0, -1, 0, 0, 0, 0, 1))
    mask = sitk.Cast(image > 50, sitk.sitkUInt8)
    source, mask_source = tmp_path / "image.mha", tmp_path / "mask.mha"
    sitk.WriteImage(image, str(source))
    sitk.WriteImage(mask, str(mask_source))
    root, _, backend = reg._stage_group(tmp_path / "stage", "Fixed", {"P000": source}).rpartition(":")
    dataset = Dataset(root, backend)
    raw, chunked = (
        write_twin(dataset, "Fixed", "P000", tmp_path / f"twin{form}", dtype="float32") for form in (".mha", ".h5")
    )
    root, _, backend = reg._stage_group(tmp_path / "stage", "FixedMask", {"P000": mask_source}).rpartition(":")
    raw_mask = write_twin(Dataset(root, backend), "FixedMask", "P000", tmp_path / "m.mha", dtype="uint8")

    frame = image.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in image.GetSize()])
    origin = sitk.Image([1] * 3, sitk.sitkFloat32)  # one voxel at (0, 0, 0): the translation is minus the centre
    for mapped in (mmap_image(raw), hdf5_twin(chunked)):
        assert np.allclose(mapped_centre(mapped, None, "cof"), frame)
        for region, mapped_mask in ((None, None), (mask, mmap_image(raw_mask))):
            whole = -np.asarray(center_of_mass_translation(image, origin, region))
            assert np.array_equal(mapped_centre(mapped, mapped_mask, "com"), whole)


def test_a_field_shifted_in_place_carries_the_translation_everywhere(tmp_path: Path) -> None:
    from impact_reg_konfai.native import shift_field

    values = np.random.default_rng(0).random((12, 10, 14, 3)).astype(np.float32)
    field = sitk.GetImageFromArray(values, isVector=True)
    field.SetSpacing((0.5, 1.0, 2.0))
    path = tmp_path / "field.mha"
    sitk.WriteImage(field, str(path), False)

    shift_field(path, np.array([3.0, -2.0, 5.5]))

    shifted = sitk.ReadImage(str(path))
    assert shifted.GetSpacing() == field.GetSpacing()
    assert np.allclose(sitk.GetArrayFromImage(shifted), values + np.array([3.0, -2.0, 5.5], dtype=np.float32))


def test_a_mapped_twin_holds_its_voxels_and_its_geometry(tmp_path: Path) -> None:
    from impact_reg_konfai.native import mmap_image

    array = np.zeros((24, 20, 28), dtype=np.float32)
    array[6:14, 4:12, 10:22] = 100.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 1.0, 2.0))
    image.SetOrigin((-3.0, 4.0, 1.5))
    path = tmp_path / "twin.mha"
    sitk.WriteImage(image, str(path), False)
    mapped = mmap_image(path)
    assert np.array_equal(mapped.array, array) and mapped.geometry.GetSpacing() == (0.5, 1.0, 2.0)
    assert mapped.geometry.GetOrigin() == (-3.0, 4.0, 1.5)


def test_a_mapped_image_gives_numpy_s_quantiles(tmp_path: Path) -> None:
    from impact_reg_konfai.native import mapped_quantile, mapped_window, mmap_image

    array = np.random.default_rng(1).normal(0.0, 1.0, (30, 20, 25)).astype(np.float32)
    path = tmp_path / "image.mha"
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path), useCompression=False)
    mapped = mmap_image(path)
    for q in (0.0, 0.005, 0.5, 0.995, 1.0):
        assert mapped_quantile(mapped, q) == float(np.quantile(array, q))
    flat = np.zeros_like(array)
    flat[0, 0, 0] = 7.0  # the 0.5-99.5 percentiles meet: the whole range, as intensity.winsorized leaves it
    sitk.WriteImage(sitk.GetImageFromArray(flat), str(tmp_path / "flat.mha"), useCompression=False)
    assert mapped_window(mmap_image(tmp_path / "flat.mha")) == (0.0, 7.0)


def test_a_transform_s_field_is_written_band_by_band(tmp_path: Path) -> None:
    from impact_reg_konfai.native import mmap_image, write_field

    image = sitk.GetImageFromArray(np.zeros((7, 5, 6), dtype=np.float32))
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((4.0, -1.0, 2.0))
    sitk.WriteImage(image, str(tmp_path / "fixed.mha"), useCompression=False)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix((1.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0, 1.0))
    transform.SetTranslation((1.0, 2.0, 3.0))
    path = write_field(tmp_path / "field.mha", mmap_image(tmp_path / "fixed.mha"), transform, band=2)
    field = sitk.ReadImage(str(path))
    expected = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection()
    )
    assert field.GetSize() == image.GetSize() and field.GetOrigin() == image.GetOrigin()
    assert np.allclose(sitk.GetArrayFromImage(field), sitk.GetArrayFromImage(expected), atol=1e-5)


def test_a_plane_larger_than_a_band_is_read_and_written_by_rows(tmp_path: Path, monkeypatch) -> None:
    """A plane alone over the band budget (a light-sheet section) goes by rows: the centre of mass and the field are
    those of whole bands."""
    from impact_reg_konfai import native

    array = np.random.default_rng(2).random((3, 20, 30)).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 1.5, 2.0))
    sitk.WriteImage(image, str(tmp_path / "image.mha"), useCompression=False)
    mapped = native.mmap_image(tmp_path / "image.mha")
    transform = sitk.AffineTransform(3)  # a displacement that varies along the rows: a row block's origin shows
    transform.SetMatrix((1.1, 0.0, 0.0, 0.0, 0.9, 0.05, 0.1, 0.0, 1.0))
    transform.SetTranslation((1.0, -2.0, 0.5))
    centre = native.mapped_centre(mapped, None, "com")
    native.write_field(tmp_path / "bands.mha", mapped, transform)

    monkeypatch.setattr(native, "_BAND_BYTES", 30 * 8 * 3 * 7)  # seven rows of the field, a few of the image
    assert np.allclose(native.mapped_centre(mapped, None, "com"), centre, atol=1e-9)
    native.write_field(tmp_path / "rows.mha", mapped, transform)
    assert np.array_equal(
        *(sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / f"{name}.mha"))) for name in ("bands", "rows"))
    )


def test_with_a_host_the_server_registers_the_sampled_cases_and_their_fields_come_back(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--host with --out-of-core sampled: the server's runner (``remote.main``) registers the cases as this machine
    would, and the client gets each case's field and compact transform back as a local run leaves them."""
    import konfai_apps.app
    from impact_reg_konfai import remote
    from konfai import RemoteServer

    fixed, moving = _pair(tmp_path)

    def sampled(self, preset, inputs, headers, cases, work, *args):
        fields, compacts = {}, {}
        for number, case in enumerate(cases, 1):
            field, compact = Path(work) / case / "DVF.mha", Path(work) / case / "Transform.h5"
            field.parent.mkdir(parents=True, exist_ok=True)
            vectors = np.full((8, 8, 8, 3), float(number), dtype=np.float32)
            sitk.WriteImage(sitk.GetImageFromArray(vectors, isVector=True), str(field))
            sitk.WriteTransform(sitk.TranslationTransform(3, (float(number), 0.0, 0.0)), str(compact))
            fields[case], compacts[case] = field, [compact]
        self._compact[preset] = compacts
        return "DVF", fields

    sent: list[list[str]] = []

    class Server:
        """The server's side of ``KonfAIAppClient.run``: its runner, on the files the client sends."""

        def __init__(self, app: str, remote_server) -> None:
            self.app = app

        def run(self, runner, inputs, output, runner_args, gpu, cpu, quiet) -> None:
            assert runner == "impact_reg_sampled"
            sent.append(runner_args)
            argv = [self.app, *(arg for group in inputs for arg in ("--inputs", *group)), "--output", str(output)]
            remote.main([*argv, *(arg for value in runner_args for arg in ("--arg", value)), "--quiet"])

    monkeypatch.setattr(reg.ImpactRegKonfAIApp, "_register_sampled", sampled)
    monkeypatch.setattr(reg, "_config_output_group", lambda config: "DVF")
    monkeypatch.setattr(konfai_apps.app, "KonfAIAppClient", Server)
    import konfai.utils.vram

    monkeypatch.setattr(konfai.utils.vram, "max_voxels", lambda *args, **kwargs: 10**12)  # no server to ask
    outputs = {}
    for name, server in (("local", None), ("remote", RemoteServer("localhost", 8000, None))):
        outputs[name] = tmp_path / name
        reg.ImpactRegKonfAIApp(remote_server=server).register(
            ["Elx"], [fixed, fixed], [moving, moving], output=outputs[name], fields_only=True, out_of_core="sampled",
            samples=4096, config_overrides=["iterations=3"],
        )  # fmt: skip
    assert sent and "samples=4096" in sent[0] and "set=iterations=3" in sent[0]
    for case in ("P000", "P001"):
        local, there = (outputs[name] / case for name in ("local", "remote"))
        assert sorted(path.name for path in there.iterdir()) == sorted(path.name for path in local.iterdir())
        for name in ("DVF.mha", "Elx_Transform.h5"):
            assert (there / name).read_bytes() == (local / name).read_bytes()


def test_a_run_that_fails_while_publishing_its_compacts_leaves_none_it_did_not_record(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compact transform stands beside a field only as register.json records it: a run that fails while
    publishing them takes back those it had published."""
    fixed, moving = _pair(tmp_path)

    def sampled(self, preset, inputs, headers, cases, work, *args):
        fields, compacts = {}, {}
        for case in cases:
            field, compact = Path(work) / case / "DVF.mha", Path(work) / case / "Transform.h5"
            field.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8, 3), np.float32), isVector=True), str(field))
            sitk.WriteTransform(sitk.TranslationTransform(3, (1.0, 0.0, 0.0)), str(compact))
            fields[case], compacts[case] = field, [compact]
        self._compact[preset] = compacts
        return "DVF", fields

    published = reg._publish_output

    def fails_on_the_second_compact(src: Path, dest: Path, move: bool = False) -> Path:
        if dest.name.endswith("_Transform.h5") and any(tmp_path.glob("out/*/*_Transform.h5")):
            raise OSError("disk full")
        return published(src, dest, move)

    monkeypatch.setattr(reg.ImpactRegKonfAIApp, "_register_sampled", sampled)
    monkeypatch.setattr(reg, "_config_output_group", lambda config: "DVF")
    monkeypatch.setattr(reg, "_publish_output", fails_on_the_second_compact)
    import konfai.utils.vram

    monkeypatch.setattr(konfai.utils.vram, "max_voxels", lambda *args, **kwargs: 10**12)
    with pytest.raises(OSError, match="disk full"):
        reg.ImpactRegKonfAIApp().register(
            ["Elx"], [fixed, fixed], [moving, moving], output=tmp_path / "out", fields_only=True, out_of_core="sampled"
        )
    recorded = reg._recorded_compacts(tmp_path / "out")
    for compact in (tmp_path / "out").glob("*/*_Transform.h5"):
        assert compact.name in recorded.get(compact.parent.name, [])


def test_with_a_host_a_set_of_the_output_group_brings_the_field_back_under_it(
    presets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server applies --set before its run names its output: the field comes back under the group the run
    gave it, as a local run publishes it."""
    import konfai_apps.app
    from impact_reg_konfai import remote
    from konfai import RemoteServer

    fixed, moving = _pair(tmp_path)
    config = _prediction(ELASTIX, presets / "model.pt")
    config["Predictor"]["outputs_dataset"] = {"dvf": {"OutputDataset": {"group": "DVF"}}}
    (presets / "Elx" / "Prediction.yml").write_text(json.dumps(config), encoding="utf-8")

    def sampled(self, preset, inputs, headers, cases, work, gpu, cpu, quiet, overrides, *args):
        config = reg._preset_config(preset, "Prediction.yml")
        reg.apply_overrides(config, overrides["global"], lambda config: None)
        group, fields = reg._config_output_group(config), {}
        for case in cases:
            fields[case] = Path(work) / case / f"{group}.mha"
            fields[case].parent.mkdir(parents=True, exist_ok=True)
            vectors = np.ones((8, 8, 8, 3), dtype=np.float32)
            sitk.WriteImage(sitk.GetImageFromArray(vectors, isVector=True), str(fields[case]))
        return group, fields

    class Server:
        def __init__(self, app: str, remote_server) -> None:
            self.app = app

        def run(self, runner, inputs, output, runner_args, gpu, cpu, quiet) -> None:
            argv = [self.app, *(arg for group in inputs for arg in ("--inputs", *group)), "--output", str(output)]
            remote.main([*argv, *(arg for value in runner_args for arg in ("--arg", value)), "--quiet"])

    monkeypatch.setattr(reg.ImpactRegKonfAIApp, "_register_sampled", sampled)
    monkeypatch.setattr(konfai_apps.app, "KonfAIAppClient", Server)
    import konfai.utils.vram

    monkeypatch.setattr(konfai.utils.vram, "max_voxels", lambda *args, **kwargs: 10**12)
    reg.ImpactRegKonfAIApp(remote_server=RemoteServer("localhost", 8000, None)).register(
        ["Elx"], [fixed], [moving], output=tmp_path / "out", fields_only=True, out_of_core="sampled",
        config_overrides=["Predictor.outputs_dataset.dvf.OutputDataset.group=Field"],
    )  # fmt: skip
    assert (tmp_path / "out" / "P000" / "Field.mha").is_file()


def test_a_sampled_field_is_published_in_the_preset_s_own_form(monkeypatch: pytest.MonkeyPatch) -> None:
    # SlicerImpactReg loads <output>/P000/Transform.h5: a field the sampled route writes goes out in the form the
    # preset's config gives its output (Transform:itktransform, an .h5), as the whole run publishes it.
    config = {
        "Predictor": {"outputs_dataset": {"Field": {"OutputDataset": {"dataset_filename": "Transform:itktransform"}}}}
    }
    monkeypatch.setattr(reg, "_preset_config", lambda preset, filename: config)
    assert reg._config_output_form("Elastix#rigid") == (".h5", "itktransform")
    config["Predictor"]["outputs_dataset"]["Field"]["OutputDataset"]["dataset_filename"] = "Transform.mha"
    assert reg._config_output_form("Elastix#rigid") is None
