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

"""The ConvexAdam engine (itk-impact): what reaches the itk-impact filters, and the field it returns."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from konfai.utils.dataset import Attribute

pytest.importorskip("itk")

from impact_reg_konfai.models import convexadam
from impact_reg_konfai.models.intensity import EngineRegistration


def _net(tmp_path: Path, models: dict | None = None, **kwargs) -> convexadam.RegistrationNet:
    """A ``RegistrationNet`` over a local model file: nothing is downloaded, nothing is loaded until a run."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"jit")
    if models is None:
        models = {"0": convexadam.ModelSpec(ref=str(model))}
    return convexadam.RegistrationNet(models=models, **kwargs)


def _ct_phantom(shift_mm=(0.0, 0.0, 0.0)) -> sitk.Image:
    """A CT-like body in HU (air, soft tissue, an organ, a bone) on an oblique, anisotropic grid, the anatomy moved
    by ``shift_mm``: ``phantom(shift)(p) = phantom()(p - shift)``."""
    image = sitk.Image([48, 40, 32], sitk.sitkFloat32)
    image.SetSpacing((2.5, 2.5, 3.0))
    image.SetOrigin((-60.0, -50.0, -40.0))
    rotation = sitk.VersorTransform((0.0, 0.0, 1.0), np.deg2rad(20.0))
    image.SetDirection(rotation.GetMatrix())
    z, y, x = np.indices(image.GetSize()[::-1], dtype=float)
    index = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    origin, direction = np.array(image.GetOrigin()), np.array(image.GetDirection()).reshape(3, 3)
    p = (origin + (index * np.array(image.GetSpacing())) @ direction.T - np.array(shift_mm)).T.reshape(3, *x.shape)
    centre = np.array(image.TransformContinuousIndexToPhysicalPoint([23.5, 19.5, 15.5]))[:, None, None, None]
    r = p - centre
    hu = np.full(x.shape, -1000.0)
    hu[(r[0] / 50) ** 2 + (r[1] / 38) ** 2 + (r[2] / 30) ** 2 < 1] = 40.0
    hu[((r[0] - 15) ** 2 + (r[1] + 8) ** 2 + (r[2] - 5) ** 2) < 12**2] = 70.0
    hu[((r[0] + 5) / 7) ** 2 + ((r[1] - 20) / 7) ** 2 + (r[2] / 25) ** 2 < 1] = 800.0
    phantom = sitk.GetImageFromArray(hu.astype(np.float32))
    phantom.CopyInformation(image)
    return sitk.SmoothingRecursiveGaussian(phantom, 2.0)


def test_each_model_configuration_carries_its_voxel_size_and_feature_normalization(tmp_path: Path, monkeypatch) -> None:
    # voxel_size reaches itk-impact, which resamples the image to it before the model (absent: 0, the image as it
    # is), and the feature normalization rides on the configuration, per model.
    built: list[list] = []

    class _Configuration:
        def __init__(self, *args) -> None:
            built.append(list(args))

        def SetFeatureNormalization(self, value: str) -> None:
            built[-1].append(value)

    monkeypatch.setattr(convexadam, "itk", SimpleNamespace(ImpactModelConfiguration=_Configuration))
    model = str(tmp_path / "model.pt")
    models = {
        "0": convexadam.ModelSpec(ref=model),
        "1": convexadam.ModelSpec(ref=model, voxel_size=[2.0, 2.0, 2.0], feature_normalization="l2"),
    }
    engine = _net(tmp_path, models=models)["Registration"]._engine

    engine._model_configurations("fine")

    path = str((tmp_path / "model.pt").resolve())
    assert built == [
        [path, 3, 1, [0, 0, 0], [0.0, 0.0, 0.0], [0, 0, 0], [True], False, "none"],
        [path, 3, 1, [0, 0, 0], [2.0, 2.0, 2.0], [0, 0, 0], [True], False, "l2"],
    ]


@pytest.mark.parametrize("settings", ["", "      overlap: 2\n"])
def test_a_configuration_loads_with_or_without_the_setting_that_does_nothing(
    tmp_path: Path, monkeypatch, settings: str
) -> None:
    # overlap never had an effect (the model runs on the whole fixed image), so the presets may drop it; a preset
    # that still sets it must keep loading.
    model = tmp_path / "model.pt"
    model.write_bytes(b"jit")
    config = tmp_path / "Prediction.yml"
    config.write_text(
        "Predictor:\n  Model:\n    RegistrationNet:\n      models:\n        '0':\n"
        f"          ref: {model}\n" + settings.replace("        voxel", "          voxel"),
        encoding="utf-8",
    )
    monkeypatch.setenv("KONFAI_config_file", str(config))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    from konfai.utils.config import apply_config

    net = apply_config("Predictor.Model.RegistrationNet")(convexadam.RegistrationNet)()

    assert [model.model_path for model in net["Registration"]._engine._feature_models["fine"]] == [str(model.resolve())]


def test_a_tile_the_fixed_mask_does_not_reach_gets_a_zero_field_through_the_graph() -> None:
    # The graph module dropped the masks, so the engine's empty-mask shortcut never ran from a prediction and a
    # tiled run fitted every tile of background. The engine is unconfigured: any step past the shortcut raises.
    engine = convexadam.ConvexAdamEngine.__new__(convexadam.ConvexAdamEngine)
    geometry = Attribute()
    geometry["Origin"] = np.array([3.0, -1.0, 7.0])
    geometry["Spacing"] = np.array([0.5, 0.5, 2.0])
    geometry["Direction"] = np.eye(3).flatten()
    image = torch.rand(1, 1, 4, 5, 6)

    field = EngineRegistration(engine, fuse_texpr=False)(
        image, image, torch.zeros_like(image), torch.ones_like(image), [[geometry] for _ in range(4)]
    )

    assert field.shape == (1, 3, 4, 5, 6) and not field.any()


def _stages(tmp_path: Path, monkeypatch, masks: tuple = (None, None), **net) -> tuple["_Filter", "_Filter"]:
    """Run both stages of a ``RegistrationNet`` on filters that record what they are set to."""
    coarse, fine = _Filter(), _Filter()
    monkeypatch.setattr(convexadam, "_coarse_registration_type", lambda: SimpleNamespace(New=lambda: coarse))
    monkeypatch.setattr(convexadam, "_fine_registration_type", lambda: SimpleNamespace(New=lambda: fine))
    engine = _net(tmp_path, **net)["Registration"]._engine
    monkeypatch.setattr(engine, "_model_configurations", lambda stage: [])
    engine._fine(None, None, *masks, engine._coarse(None, None, *masks, "cpu"), "cpu")
    return coarse, fine


def test_each_model_s_settings_reach_every_layer_it_keeps_in_both_stages(tmp_path: Path, monkeypatch) -> None:
    # itk-impact indexes the distance, weight, PCA and channel subset by kept layer across the models. Passed one
    # per model, a model keeping two layers handed its second layer the next model's settings; and the coarse
    # stage took neither the weights nor the PCA nor the subset.
    model = str(tmp_path / "model.pt")
    two_layers = convexadam.ModelSpec(
        ref=model, layers_mask="0110", distance="Dice", layers_weight=[2.0], pca=4, subset_features=3
    )
    one_layer = convexadam.ModelSpec(ref=model, layers_mask="1", distance="LNCC", layers_weight=[0.5])

    coarse, fine = _stages(tmp_path, monkeypatch, models={"0": two_layers, "1": one_layer})

    for stage in (coarse, fine):
        assert stage.calls["SetLayersWeight"] == ([2.0, 2.0, 0.5],)
        assert stage.calls["SetPCA"] == ([4, 4, 0],)
        assert stage.calls["SetSubsetFeatures"] == ([3, 3, 0],)
    # The coarse cost stays raw whatever `normalize` says: its coupling coefficients are absolute.
    assert coarse.calls["SetNormalizeLosses"] == (False,) and fine.calls["SetNormalizeLosses"] == (True,)
    assert fine.calls["SetDistance"] == (["Dice", "Dice", "LNCC"],)


def test_balance_coarse_layers_reaches_the_coarse_stage_alone(tmp_path: Path, monkeypatch) -> None:
    # Off until an MR/CT measurement adopts it; either way the coarse cost is never divided by its value at zero
    # displacement, which multiplied the TotalSegmentator head by 453 and folded the MR/CT coarse stage.
    coarse, fine = _stages(tmp_path, monkeypatch)
    assert coarse.calls["SetBalanceLosses"] == (False,) and "SetBalanceLosses" not in fine.calls
    coarse, _ = _stages(tmp_path, monkeypatch, balance_coarse_layers=True)
    assert coarse.calls["SetBalanceLosses"] == (True,) and coarse.calls["SetNormalizeLosses"] == (False,)


def test_each_mask_reaches_both_stages_and_no_mask_sets_nothing(tmp_path: Path, monkeypatch) -> None:
    # Dropped at one stage, a mask still changes the field through the other: only the calls tell.
    fixed_mask, moving_mask = object(), object()
    for stage in _stages(tmp_path, monkeypatch, masks=(fixed_mask, moving_mask)):
        assert stage.calls["SetFixedMask"] == (fixed_mask,) and stage.calls["SetMovingMask"] == (moving_mask,)
    for stage in _stages(tmp_path, monkeypatch):
        assert "SetFixedMask" not in stage.calls and "SetMovingMask" not in stage.calls


def test_a_weight_per_kept_layer_reaches_each(tmp_path: Path, monkeypatch) -> None:
    model = str(tmp_path / "model.pt")
    coarse, _ = _stages(
        tmp_path,
        monkeypatch,
        models={"0": convexadam.ModelSpec(ref=model, layers_mask="011", layers_weight=[1.0, 3.0])},
    )
    assert coarse.calls["SetLayersWeight"] == ([1.0, 3.0],)


def test_the_loss_settings_reach_the_fine_stage(tmp_path: Path, monkeypatch) -> None:
    coarse, fine = _stages(
        tmp_path, monkeypatch, mode="Jacobian", normalize=False, feature_map_update_interval=20, lncc_kernel=7
    )
    assert fine.calls["SetMode"] == ("Jacobian",)
    assert coarse.calls["SetNormalizeLosses"] == fine.calls["SetNormalizeLosses"] == (False,)
    assert fine.calls["SetFeatureMapUpdateInterval"] == (20,)
    assert fine.calls["SetLNCCKernel"] == (7,)


def test_voxel_sampling_reaches_the_fine_stage(tmp_path: Path, monkeypatch) -> None:
    # Refused where it means nothing. In Jacobian mode each model runs on the patch of its receptive field around each
    # point, which a local model with no registry entry cannot size.
    _, fine = _stages(tmp_path, monkeypatch, voxel_sampling=0.1)
    assert fine.calls["SetSamplingPercentage"] == (0.1,)
    _, fine = _stages(tmp_path, monkeypatch)
    assert fine.calls["SetSamplingPercentage"] == (1.0,)
    with pytest.raises(ValueError, match="cannot be sized"):
        _net(tmp_path, voxel_sampling=0.1, mode="Jacobian")
    with pytest.raises(ValueError, match="LNCC"):
        _net(tmp_path, {"0": convexadam.ModelSpec(ref=str(tmp_path / "m.pt"), distance="LNCC")}, voxel_sampling=0.1)


def test_each_stage_takes_its_level_s_models(tmp_path: Path, monkeypatch) -> None:
    # levels replaces models stage by stage, in 'stages' order: here MIND alone in the coarse search, MIND and a
    # second model in the refinement.
    model = str(tmp_path / "model.pt")
    mind = convexadam.ModelSpec(ref=model)
    other = convexadam.ModelSpec(ref=model, layers_mask="01", distance="L1")
    levels = {"0": convexadam.LevelSpec(models={"0": mind}), "1": convexadam.LevelSpec(models={"0": mind, "1": other})}

    coarse, fine = _stages(tmp_path, monkeypatch, levels=levels)

    assert coarse.calls["SetLayersWeight"] == ([1.0],)
    assert fine.calls["SetLayersWeight"] == ([1.0, 1.0],)
    assert fine.calls["SetDistance"] == (["L2", "L1"],)
    with pytest.raises(ValueError, match="one per level"):
        _net(tmp_path, levels={"0": levels["0"]})


@pytest.mark.slow  # the first itk registration loads ITK's modules, ~10 s
def test_the_linear_stage_recovers_a_shift_whatever_the_intensity_range(tmp_path: Path) -> None:
    # The rigid was seeded from intensity moments, which divide by the total intensity: negative for a CT in HU,
    # zero for a z-scored image. A z-scored moving was seeded ~1e9 mm away and the linear stage stopped on "All
    # samples map outside moving image buffer".
    shift = np.array([7.0, -5.0, 9.0])
    fixed = _ct_phantom()
    moving = _ct_phantom(shift)
    array = sitk.GetArrayViewFromImage(moving)
    moving = sitk.Cast((moving - float(array.mean())) / float(array.std()), sitk.sitkFloat32)
    engine = _net(tmp_path, stages=[], linear=True, linear_iterations=50)["Registration"]._engine

    field = engine.register(fixed, moving, -1)

    body = field[:, 10:22, 12:28, 14:34].reshape(3, -1)
    assert np.abs(body - shift[:, None]).max() < 1.0


def test_the_linear_stage_holds_against_a_few_hot_voxels(tmp_path: Path) -> None:
    # Light-sheet tissue lies in a few tens of grey values beside lone voxels in the tens of thousands: on the raw
    # ExaSPIM brains Otsu's foreground of the fixed image was 9 hot voxels, 9 mm from the brain's centre, and the mutual
    # information binned every tissue voxel into one of 32 bins, so the affine seeded and stayed off target.
    shift = np.array([7.0, -5.0, 9.0])
    fixed, moving = ((_ct_phantom(s) + 1000.0) / 36.0 for s in (np.zeros(3), shift))  # air 0, tissue ~30, bone 50
    fixed[2, 3, 4] = 20000.0
    fixed[3, 3, 4] = 18000.0
    engine = _net(tmp_path, stages=[], linear=True, linear_iterations=50)["Registration"]._engine

    field = engine.register(fixed, moving, -1)

    body = field[:, 10:22, 12:28, 14:34].reshape(3, -1)
    assert np.abs(body - shift[:, None]).max() < 1.0


def test_a_sampled_linear_stage_still_recovers_a_shift(tmp_path: Path) -> None:
    # Reading every voxel at every iteration took 83 s of an abdominal CT pair's 106 s registration; a tenth of them,
    # drawn with the engine's seed, lands the same affine to a fraction of a voxel.
    shift = np.array([7.0, -5.0, 9.0])
    engine = _net(tmp_path, stages=[], linear=True, linear_iterations=50, linear_sampling=0.1)["Registration"]._engine

    field = engine.register(_ct_phantom(), _ct_phantom(shift), -1)

    body = field[:, 10:22, 12:28, 14:34].reshape(3, -1)
    assert np.abs(body - shift[:, None]).max() < 1.0
    for share in (0.0, 1.5):
        with pytest.raises(ValueError, match="linear_sampling"):
            _net(tmp_path, linear_sampling=share)


def test_the_linear_stage_is_not_seeded_off_target_by_a_scan_that_reaches_further(tmp_path: Path) -> None:
    # The Learn2Reg abdomen CT covers 50 mm more of the body than its MR: the foreground centres lay that far apart
    # where the anatomy did not, and the affine seeded there went off target on the coarse copy (Dice 0.42 -> 0).
    shift = np.array([3.0, -2.0, 4.0])
    moving = _ct_phantom(shift)
    array = sitk.GetArrayFromImage(moving)
    array[..., :12] = 40.0  # tissue along a whole side of the moving scan alone
    longer = sitk.GetImageFromArray(array)
    longer.CopyInformation(moving)
    engine = _net(tmp_path, stages=[], linear=True, linear_iterations=50)["Registration"]._engine

    field = engine.register(_ct_phantom(), longer, -1)

    body = field[:, 10:22, 12:28, 14:34].reshape(3, -1)
    assert np.abs(body - shift[:, None]).max() < 1.0


def test_the_linear_stage_draws_the_same_samples_every_run(tmp_path: Path) -> None:
    # Its scales estimator samples the image through ITK's global Mersenne Twister, which ITK seeds from the clock in
    # every process: the same pair got another affine each run. Whatever the generator was left at, a run draws
    # the same numbers and leaves it in the same state.
    import itk

    fixed, moving = _ct_phantom(), _ct_phantom((7.0, -5.0, 9.0))
    engine = _net(tmp_path, stages=[], linear=True, linear_iterations=5)["Registration"]._engine
    after = []
    for clock in (1917622863, 1983980151):  # two processes' seeds
        itk.MersenneTwisterRandomVariateGenerator.GetInstance().SetSeed(clock)
        engine.register(fixed, moving, -1)
        after.append(itk.MersenneTwisterRandomVariateGenerator.GetInstance().GetVariate())

    assert after[0] == after[1]


@pytest.mark.slow  # the linear stage loads ITK's modules
def test_each_mask_restricts_the_linear_stage_and_its_seed(tmp_path: Path) -> None:
    # They reached the coarse and fine stages alone: a CT's table or head rest still pulled the affine and its seed.
    import itk

    fixed, moving = _ct_phantom() + 1000.0, _ct_phantom((4.0, -3.0, 5.0)) + 1000.0
    engine = _net(tmp_path, stages=[], linear_iterations=10)["Registration"]._engine
    array = np.zeros(fixed.GetSize()[::-1], dtype=np.uint8)
    array[..., :24] = 1
    half = sitk.GetImageFromArray(array)
    half.CopyInformation(fixed)  # the phantoms share one grid
    centre = convexadam._foreground_centre(fixed, None)
    fixed_itk, moving_itk, half_itk = (
        convexadam._sitk_to_itk(fixed),
        convexadam._sitk_to_itk(moving),
        convexadam._sitk_to_itk(half, np.uint8),
    )

    def affine(fixed_mask, moving_mask) -> np.ndarray:
        found = engine._linear_align(fixed_itk, moving_itk, fixed_mask, moving_mask, centre, centre)
        return np.append(itk.array_from_matrix(found.GetMatrix()), found.GetTranslation())

    free = affine(None, None)
    assert np.allclose(affine(None, None), free)  # the threads' sums vary in their last bits, not more
    assert not np.allclose(affine(half_itk, None), free) and not np.allclose(affine(None, half_itk), free)
    assert not np.allclose(convexadam._foreground_centre(fixed, half), centre)


def test_a_mask_takes_its_image_s_grid_where_it_lies_on_it_and_is_resampled_onto_it_elsewhere() -> None:
    # It took the image's header whatever its grid: one of another size raised, one elsewhere was moved onto the image.
    image = sitk.Image([10, 10, 10], sitk.sitkFloat32)
    rounded = sitk.Image([10, 10, 10], sitk.sitkUInt8) + 3
    rounded.SetOrigin((1e-6, 0.0, 0.0))  # KonfAI's Attribute round-trip
    shifted = sitk.Image([10, 10, 10], sitk.sitkUInt8) + 1
    shifted.SetOrigin((5.0, 0.0, 0.0))
    larger = sitk.Image([12, 10, 10], sitk.sitkUInt8) + 1
    larger.SetOrigin((-7.0, 0.0, 0.0))

    assert convexadam._binary_mask(rounded, image).GetOrigin() == image.GetOrigin()
    assert (sitk.GetArrayFromImage(convexadam._binary_mask(rounded, image)) == 1).all()
    for mask, inside in ((shifted, slice(5, None)), (larger, slice(None, 5))):
        on_grid = convexadam._binary_mask(mask, image)
        expected = np.zeros((10, 10, 10), dtype=np.uint8)
        expected[..., inside] = 1
        assert on_grid.GetOrigin() == image.GetOrigin() and np.array_equal(sitk.GetArrayFromImage(on_grid), expected)


def _intensity_model(tmp_path: Path) -> str:
    """A TorchScript feature model returning the image, scaled so the data term outweighs the regulariser the way
    MIND's features do."""

    class Intensity(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
            return [10.0 * image]

    model = tmp_path / "intensity.pt"
    torch.jit.script(Intensity()).save(str(model))
    return str(model)


def _body(image: sitk.Image) -> sitk.Image:
    """The phantom's body (over 500 once 1000 is added, air at 0), as the label 3: a label map, not a 0/1 mask."""
    return sitk.Cast(image > 500, sitk.sitkUInt8) * 3


@pytest.mark.slow  # itk-impact runs the whole chain on the CPU, ITK's modules load first
def test_a_mask_restricts_both_stages_and_a_whole_image_one_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    # itk-impact had no mask API, and the engine registered the whole image whatever the mask. A whole-image mask,
    # konfai-apps' default when none is given (on the FIXED grid for both sides), must leave the run bit for bit.
    import itk

    if not hasattr(itk, "ImpactFineRegistration"):
        pytest.skip("itk-impact is not installed")
    fixed, moving = _ct_phantom() + 1000.0, _ct_phantom((4.0, -3.0, 5.0)) + 1000.0
    models = {"0": convexadam.ModelSpec(ref=_intensity_model(tmp_path), distance="L1")}
    settings = {"grid_spacing": 2, "displacement_half_width": 2, "iterations": 10, "linear": False}
    engine = _net(tmp_path, models=models, **settings)["Registration"]._engine
    whole = sitk.Cast(fixed * 0 + 1, sitk.sitkUInt8)
    array = np.zeros(fixed.GetSize()[::-1], dtype=np.uint8)
    array[..., :24] = 1
    half = sitk.GetImageFromArray(array)
    half.CopyInformation(fixed)

    free = engine.register(fixed, moving, -1)

    assert np.array_equal(engine.register(fixed, moving, -1, whole, whole), free)
    assert not np.allclose(engine.register(fixed, moving, -1, half, None), free, atol=1e-3)
    # Balanced, the coarse stage measures each layer's spread over the candidates (one layer here).
    made = []
    real = convexadam._coarse_registration_type()
    monkeypatch.setattr(
        convexadam,
        "_coarse_registration_type",
        lambda: SimpleNamespace(New=lambda: made.append(real.New()) or made[-1]),
    )
    engine = _net(tmp_path, models=models, stages=["coarse"], balance_coarse_layers=True, **settings)
    engine["Registration"]._engine.register(fixed, moving, -1, _body(fixed), None)
    assert made[0].GetBalanceLosses() and len(made[0].GetLayerSpreads()) == 1 and made[0].GetLayerSpreads()[0] > 0


@pytest.mark.slow  # the linear stage loads ITK's modules
@pytest.mark.parametrize("linear", [False, True])
def test_each_mask_reaches_the_filters_on_the_grid_of_the_image_handed_with_it(
    tmp_path: Path, monkeypatch, linear: bool
) -> None:
    # The engine hands the filters the pair with its voxel axes in LPS order, and the moving image resampled onto the
    # fixed grid after the linear stage: a mask left on the grid it came on would restrict other voxels than its
    # image's. Each must reach them on its image's grid, over the same anatomy, binarised.
    import itk

    fixed = sitk.PermuteAxes(_ct_phantom() + 1000.0, [1, 0, 2])  # axes out of LPS order
    moving = sitk.Flip(_ct_phantom((8.0, -6.0, 10.0)) + 1000.0, [True, False, True])[2:, 1:, :]  # another grid
    engine = _net(tmp_path, linear=linear, linear_iterations=5)["Registration"]._engine
    handed = {}

    def run_stages(fixed_itk, moving_itk, fixed_mask, moving_mask, device):
        handed.update(fixed=(fixed_itk, fixed_mask), moving=(moving_itk, moving_mask))
        return engine._zero_field(fixed_itk)

    monkeypatch.setattr(engine, "_run_stages", run_stages)
    linear_masks: list = []
    linear_align = engine._linear_align
    monkeypatch.setattr(engine, "_linear_align", lambda *args: linear_masks.extend(args[2:4]) or linear_align(*args))
    engine.register(fixed, moving, -1, _body(fixed), _body(moving))

    assert list(handed["fixed"][0].GetLargestPossibleRegion().GetSize()) == [48, 40, 32]  # back in LPS order
    if linear:  # the linear stage is masked too, the moving mask still on the moving grid
        assert linear_masks[0] is handed["fixed"][1] and linear_masks[1] is not None
    for side, (image, mask) in handed.items():
        for query in ("GetOrigin", "GetSpacing"):
            assert np.allclose(getattr(mask, query)(), getattr(image, query)())
        assert np.allclose(itk.array_from_matrix(mask.GetDirection()), itk.array_from_matrix(image.GetDirection()))
        body, voxels = itk.array_view_from_image(mask), itk.array_view_from_image(image)
        assert body.shape == voxels.shape and body.dtype == np.uint8 and np.unique(body).tolist() == [0, 1]
        # The same voxels as the image's body: exactly where nothing was resampled, all but the edge where the linear
        # stage resampled the moving image (linearly) and its mask (by nearest neighbour).
        agreement = np.mean(body.astype(bool) == (voxels > 500))
        assert agreement > (0.95 if linear and side == "moving" else 0.9999)


@pytest.mark.slow  # the engine loads ITK's modules
def test_a_fixed_mask_alone_gives_the_moving_side_a_whole_image_mask(tmp_path: Path, monkeypatch) -> None:
    # As FireANTs: once a fixed mask restricts the metric, the fixed voxels the field sends out of the moving image
    # stop counting, which only a moving mask tells itk-impact. A moving mask alone leaves the fixed side unmasked.
    import itk

    fixed, moving = _ct_phantom() + 1000.0, _ct_phantom((4.0, -3.0, 5.0)) + 1000.0
    engine = _net(tmp_path, linear=False)["Registration"]._engine
    handed: list = []

    def run_stages(fixed_itk, moving_itk, fixed_mask, moving_mask, device):
        handed[:] = [moving_itk, fixed_mask, moving_mask]
        return engine._zero_field(fixed_itk)

    monkeypatch.setattr(engine, "_run_stages", run_stages)
    engine.register(fixed, moving, -1, _body(fixed), None)
    moving_itk, _, moving_mask = handed
    assert moving_mask.GetLargestPossibleRegion().GetSize() == moving_itk.GetLargestPossibleRegion().GetSize()
    assert (itk.array_view_from_image(moving_mask) == 1).all()
    engine.register(fixed, moving, -1, None, _body(moving))
    assert handed[1] is None and handed[2] is not None


class _Filter:
    """An itk-impact filter that records what it is set to."""

    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def __getattr__(self, name: str):
        def call(*args):
            self.calls[name] = args
            return SimpleNamespace(DisconnectPipeline=lambda: None)

        return call


def test_a_negative_layer_weight_fails_at_build(tmp_path: Path) -> None:
    # It rewarded the dissimilarity of its layer.
    models = {"0": convexadam.ModelSpec(ref=str(tmp_path / "model.pt"), layers_weight=[-1.0])}
    with pytest.raises(ValueError, match="layers_weight must be non-negative"):
        _net(tmp_path, models=models)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"models": {}}, "at least one feature model"),
        ({"stages": ["fine", "coarse"]}, "not a ConvexAdam chain"),
        ({"stages": ["coarse", "affine"]}, "not a ConvexAdam chain"),
        ({"stages": [], "linear": False}, "nothing to register"),
        ({"grid_spacing": 0}, "grid_spacing must be at least 1"),
        ({"grid_shrink": 0}, "grid_shrink must be at least 1"),
        ({"lncc_kernel": 4}, "lncc_kernel must be odd"),
        ({"mode": "Online"}, "mode must be 'Static' or 'Jacobian'"),
    ],
)
def test_a_configuration_convexadam_cannot_run_fails_at_build(tmp_path: Path, overrides: dict, message: str) -> None:
    # Each ran silently or failed only at the first case: no model compared raw intensities, a coarse stage after
    # the fine one discarded it, an unknown stage raised after the downloads, an empty chain returned the identity,
    # a zero grid spacing divided by zero inside itk-impact.
    with pytest.raises(ValueError, match=message):
        _net(tmp_path, **overrides)


@pytest.mark.slow  # itk-impact runs the whole chain on the CPU, ITK's modules load first
@pytest.mark.parametrize("linear", [False, True])
def test_the_field_is_the_physical_fixed_to_moving_map_on_an_oblique_anisotropic_grid(
    tmp_path: Path, linear: bool
) -> None:
    # What the orchestrator and every downstream reader assume: channel-first on the fixed grid, x,y,z components in
    # millimetres, moved(p) = moving(p + d(p)), with the affine applied after the deformable field. Checked against
    # a known rigid motion on a grid whose direction, spacing and axis order all differ from the identity.
    import itk

    if not hasattr(itk, "ImpactFineRegistration"):
        pytest.skip("itk-impact is not installed")

    model = _intensity_model(tmp_path)
    rng = np.random.default_rng(0)
    content = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(rng.random((52, 56, 60)).astype(np.float32)), 2.5)
    content = sitk.Cast(sitk.RescaleIntensity(content, 0.0, 1.0), sitk.sitkFloat32)
    content.SetSpacing((1.2, 0.9, 1.5))
    content.SetDirection(sitk.VersorTransform((0.2, 0.3, 1.0), np.deg2rad(30.0)).GetMatrix())
    fixed = sitk.Resample(
        content,
        [40, 44, 36],
        sitk.Transform(),
        sitk.sitkLinear,
        content.TransformIndexToPhysicalPoint([6, 6, 8]),
        content.GetSpacing(),
        content.GetDirection(),
    )
    motion = sitk.Euler3DTransform(
        fixed.TransformContinuousIndexToPhysicalPoint([20, 22, 18]), 0.0, 0.0, np.deg2rad(4.0), (2.0, -1.5, 2.5)
    )
    moving = sitk.Resample(content, fixed, motion.GetInverse(), sitk.sitkLinear, 0.0)  # moving(motion(p)) = content(p)
    fixed = sitk.Resample(content, fixed, sitk.Transform(), sitk.sitkLinear, 0.0)
    net = convexadam.RegistrationNet(
        models={"0": convexadam.ModelSpec(ref=model, distance="L1")},
        grid_spacing=2,
        displacement_half_width=3,
        iterations=80,
        learning_rate=1.0,
        regularization_weight=1.25,
        grid_shrink=2,
        stages=["coarse", "fine"],
        linear=linear,
        linear_iterations=50,
    )

    def tensor_and_geometry(image: sitk.Image) -> tuple[torch.Tensor, Attribute]:
        array, geometry = convexadam.image_to_data(image)
        return torch.from_numpy(array)[None], geometry

    (f, fg), (m, mg) = tensor_and_geometry(fixed), tensor_and_geometry(moving)
    field = net["Registration"](f, m, torch.ones_like(f), torch.ones_like(m), [[fg], [mg], [fg], [mg]])[0].numpy()

    assert field.shape == (3, 36, 44, 40) and field.dtype == np.float32
    errors = []
    for k, j, i in np.ndindex(4, 4, 4):
        index = (int(8 + 8 * i), int(9 + 9 * j), int(7 + 7 * k))
        p = np.array(fixed.TransformIndexToPhysicalPoint(index))
        truth = np.array(motion.TransformPoint(p.tolist())) - p
        errors.append(np.linalg.norm(field[:, index[2], index[1], index[0]] - truth))
    assert np.mean(errors) < 0.6 and np.max(errors) < 2.0  # mm, over a motion of 3.6 mm on average
