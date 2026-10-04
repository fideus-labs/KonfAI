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

"""What the FireANTs engine owes a caller, checked through real (small, CPU) FireANTs runs: a field that is the
physical displacement on the fixed grid, masks that mean a region in every stage and on both sides, and the
refusals that spare a run the minutes it would otherwise spend before failing."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from impact_reg_konfai.models import fireants
from konfai.metric.measure import ImpactFeatureModel
from konfai.metric.measure.impact import DISTANCES

pytest.importorskip("fireants")


def _volume(side: int = 24) -> sitk.Image:
    z, y, x = np.mgrid[:side, :side, :side].astype(np.float32)
    image = sitk.GetImageFromArray(100.0 * np.exp(-((x - 10) ** 2 + (y - 12) ** 2 + (z - 13) ** 2) / 30.0))
    image.SetSpacing((2.0, 2.0, 2.0))
    return image


def _engine(**overrides) -> fireants.FireANTsEngine:
    settings = {
        "scales": [1],
        "affine_iterations": [1],
        "deformable_iterations": [1],
        "cc_kernel": 3,
        "affine_metric": "mse",
        "affine_lr": 0.01,
        "moments_init": "none",
        "linear_method": "rigid",
        "deformable_method": "none",
        "deformable_metric": "cc",
        "deformable_lr": 0.1,
        "smooth_warp_sigma": 0.5,
        "smooth_grad_sigma": 1.0,
        "seed": 0,
        "impact_levels": [],
    }
    return fireants.FireANTsEngine(**{**settings, **overrides})


def test_a_label_map_given_as_a_mask_reaches_fireants_as_a_region(monkeypatch: pytest.MonkeyPatch) -> None:
    # SlicerImpactReg exports segments as labels 1..N; FireANTs' masked metrics would weight voxels by them.
    seen: list[float] = []
    on_grid = fireants._mask_on_grid

    def recording(mask: sitk.Image, image: sitk.Image, device: str):
        seen.append(float(sitk.GetArrayViewFromImage(mask).max()))
        return on_grid(mask, image, device)

    monkeypatch.setattr(fireants, "_mask_on_grid", recording)
    image = _volume()
    labels = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
    labels[4:12], labels[12:20] = 3, 7
    mask = sitk.GetImageFromArray(labels)
    mask.CopyInformation(image)
    _engine().register(image, image, -1, mask, mask)
    assert seen == [1.0, 1.0]


def _masked_pair(moving_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical features except where the moving mask is off, each carrying its mask as the last channel."""
    torch.manual_seed(0)
    features = torch.rand(1, 2, 6, 6, 6)
    moved = features.clone()
    moved[..., :3] = torch.rand(1, 2, 6, 6, 3)
    fixed_mask = torch.ones(1, 1, 6, 6, 6)
    return torch.cat([moved, moving_mask], 1), torch.cat([features, fixed_mask], 1)


def _bare_loss(mode: str, core: torch.nn.Module, spec: "fireants.ModelSpec") -> "fireants.ImpactFeatureLoss":
    """An IMPACT loss over one model, without the model download of its __init__."""
    loss = fireants.ImpactFeatureLoss.__new__(fireants.ImpactFeatureLoss)
    torch.nn.Module.__init__(loss)
    loss._masked, loss._mode, loss._normalize, loss._kernel, loss._chunk = True, mode, False, 3, 0
    loss._specs, loss._levels, loss._cores = [spec], [[0]], torch.nn.ModuleList([core])
    loss._channels, loss._level, loss._factors = [[2]], -1, None
    loss._generator = torch.Generator().manual_seed(0)
    loss._sampling, loss._patch = 1.0, []
    return loss


def test_the_custom_losses_leave_out_what_the_moving_mask_leaves_out() -> None:
    # FireANTs' own masked metrics count a voxel where the fixed AND the warped moving mask hold it.
    moving_mask = torch.ones(1, 1, 6, 6, 6)
    moving_mask[..., :3] = 0.0
    moved, fixed = _masked_pair(moving_mask)

    class Recording(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kept = [0]

        def distances(self, moved, fixed, mask, terms, seed, kernel):
            self.mask = mask
            return [moved.sum() * 0.0]

    spec = fireants.ModelSpec(ref="model.pt", distance="LNCC")
    static = _bare_loss("Static", Recording(), spec)(moved, fixed)
    torch.testing.assert_close(static, fireants.distance("LNCC", moved[:, :-1], fixed[:, :-1], moving_mask, 3, 0))
    core = Recording()
    _bare_loss("Jacobian", core, spec)(moved, fixed)
    assert torch.equal(core.mask, moving_mask)


def test_each_layer_starts_every_scale_at_1_and_weighs_its_share() -> None:
    # The same normalization as the other engines: each layer divided by its value when a scale starts, then weighed.
    torch.manual_seed(0)
    moved, fixed = torch.rand(1, 2, 6, 6, 6), torch.rand(1, 2, 6, 6, 6)
    spec = fireants.ModelSpec(ref="model.pt", layers_mask="11", layers_weight=[0.25, 0.75], distance="L2")
    loss = _bare_loss("Static", torch.nn.Module(), spec)
    loss._masked, loss._normalize, loss._channels = False, True, [[1, 1]]
    loss.cores[0].kept = [0, 1]

    loss.set_current_scale_and_iterations(2, 10)
    assert loss(moved, fixed).item() == pytest.approx(1.0)
    closer = loss(fixed + 0.5 * (moved - fixed), fixed).item()  # each layer's squared distance a quarter
    assert closer == pytest.approx(0.25)
    loss.set_current_scale_and_iterations(1, 10)  # a new scale starts at 1 again
    assert loss(fixed + 0.5 * (moved - fixed), fixed).item() == pytest.approx(1.0)


def test_only_a_mutual_information_stage_sees_rescaled_intensities(monkeypatch: pytest.MonkeyPatch) -> None:
    # FireANTs' MI divides both images by their shared maximum and clamps below 0: raw CT would lose everything
    # under 0 HU. The correlation stage keeps the intensities, where FireANTs' absolute constants expect them.
    import fireants.registration.rigid as rigid_module
    import fireants.registration.syn as syn_module

    seen: dict[str, list[tuple[float, float]]] = {}

    def recording(cls: type, stage: str) -> type:
        class Recording(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, fixed_images, moving_images, **kwargs) -> None:
                seen[stage] = [
                    (float(b.batch_tensor.min()), float(b.batch_tensor.max())) for b in (fixed_images, moving_images)
                ]
                super().__init__(*args, fixed_images=fixed_images, moving_images=moving_images, **kwargs)

        return Recording

    monkeypatch.setattr(rigid_module, "RigidRegistration", recording(rigid_module.RigidRegistration, "linear"))
    monkeypatch.setattr(syn_module, "SyNRegistration", recording(syn_module.SyNRegistration, "deformable"))
    ct = _volume(32) - 1000.0
    _engine(affine_metric="mi", deformable_method="syn").register(ct, ct, -1)
    assert all(0.0 <= low and high < 1.0 for low, high in seen["linear"])
    assert all(low == -1000.0 for low, _ in seen["deformable"])


def test_the_mutual_information_images_stay_under_1_once_interpolated() -> None:
    """Winsorising leaves a plateau at the top, which the moved image's interpolation rounded past 1: FireANTs then
    divides both images by the moved image's maximum, which carries a gradient, and the fixed image's Parzen
    windowing joined the backward graph (+42 % memory, the coarse pass of a large pair out of memory)."""
    from impact_reg_konfai.models.fireants import _unit_range

    array = np.zeros((16, 16, 16), dtype=np.float32)
    array[4:12, 4:12, 4:12] = 1000.0  # a plateau well past the 99.5th percentile
    unit = _unit_range(sitk.GetImageFromArray(array))
    shift = sitk.TranslationTransform(3, (0.37, -0.21, 0.5))
    moved = sitk.Resample(unit, unit, shift, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    assert 0.99 < sitk.GetArrayViewFromImage(unit).max() < 1.0
    assert sitk.GetArrayViewFromImage(moved).max() < 1.0


class _OneLayer(torch.nn.Module):
    """A feature model with a single output, as MIND and anatomix are, taking the published models' inputs."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [x]


def test_a_layers_mask_longer_than_the_model_s_outputs_fails_at_build(tmp_path) -> None:
    # '01' on a one-layer model would weigh its only layer 0: a constant metric that SyN cannot move.
    path = tmp_path / "one_layer.pt"
    torch.jit.script(_OneLayer()).save(str(path))
    with pytest.raises(RuntimeError, match="number of weights"):
        fireants._ImpactCore(fireants.ModelSpec(ref=str(path), layers_mask="01"), False)
    fireants._ImpactCore(fireants.ModelSpec(ref=str(path)), False)


class _HeldConstant(torch.nn.Module):
    """A feature model holding a tensor that is neither a parameter nor a buffer, as a trace bakes a constant in:
    ``.to()`` leaves it where the file was mapped."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.tensor([2.0])

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [x * self.scale]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a constant left on the CPU only shows on a CUDA run")
def test_a_jacobian_model_holding_a_constant_is_scored_on_the_card(tmp_path) -> None:
    """The file is mapped onto the device the loss runs on: loaded on the CPU and moved, the network leaves its
    constant behind and its first forward on the card fails."""
    path = tmp_path / "constant.pt"
    torch.jit.script(_HeldConstant()).save(str(path))
    loss = fireants.ImpactFeatureLoss(
        [[fireants.ModelSpec(ref=str(path), distance="L1")]], "Jacobian", False, 5, 0, 0, False
    )
    moved = torch.rand(1, 1, 16, 16, 16, device="cuda", requires_grad=True)

    loss(moved, torch.rand(1, 1, 16, 16, 16, device="cuda")).sum().backward()

    assert torch.isfinite(moved.grad).all()


def test_the_feature_models_load_once_for_every_registration(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # register runs per case and per native tile, in one process: the models are fetched and loaded once.
    path = tmp_path / "one_layer.pt"
    torch.jit.script(_OneLayer()).save(str(path))
    from konfai.metric.measure import impact

    monkeypatch.delenv("KONFAI_IMPACT_MODELS_REGISTRY", raising=False)
    monkeypatch.setattr(impact, "models_registry", lambda: pytest.fail("a local model needs no registry"))
    loads: list[str] = []
    load = torch.jit.load
    monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: loads.append(args[0]) or load(*args, **kwargs))
    engine = _engine(
        linear_method="none",
        deformable_method="syn",
        deformable_metric="impact",
        impact_levels=[[fireants.ModelSpec(ref=str(path))]],
    )
    image = _volume(32)
    for _ in range(3):
        engine.register(image, image, -1)
    assert len(loads) == 2  # the build-time probe and the metric's own load


class _Oriented(torch.nn.Module):
    """Mimics a TotalSegmentator export: flips its input to the training orientation the direction implies
    (diag(-1, -1, 1) @ D^T, as the exports compute it) and leaves it as given without a direction."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if direction is not None and direction.numel() == 9:
            return [x.flip(-1).flip(-2)]  # identity direction: x and y flipped to RAS
        return [x]


def test_a_totalsegmentator_model_is_told_the_world_aligned_direction(tmp_path) -> None:
    # The engine registers world-aligned copies; a TS model reorients from the direction it is given, as
    # itk-impact passes it, and runs on the axes as given (another feature map) without one.
    path = tmp_path / "oriented.pt"
    torch.jit.script(_Oriented()).save(str(path))
    core = fireants._ImpactCore(fireants.ModelSpec(ref=str(path)), False)
    image = torch.rand(1, 1, 4, 5, 6)
    stats = {"ImageMin": 0.0, "ImageMax": 1.0, "ImageMean": 0.5, "ImageStd": 0.3}
    inputs = core.model.inputs(image, stats)
    assert torch.equal(inputs[3], torch.eye(3, dtype=torch.int16))
    core.model.model = torch.jit.load(str(path))
    torch.testing.assert_close(core.model.model(*inputs)[0], image.flip(-1).flip(-2))


@pytest.mark.parametrize(("method", "size"), [("greedy", (40, 40, 20)), ("syn", (24, 24, 24))])
def test_a_volume_fireants_cannot_warp_is_refused_before_the_optimisation(method: str, size: tuple) -> None:
    # Greedy fails on any axis under 32 voxels and SyN when all are, both after the whole optimisation.
    array = np.zeros(size[::-1], dtype=np.float32)
    array[4:-4, 4:-4, 4:-4] = 100.0
    image = sitk.GetImageFromArray(array)
    with pytest.raises(ValueError, match="at least 32"):
        _engine(linear_method="none", deformable_method=method).register(image, image, -1)


def test_an_empty_fixed_mask_says_so_outside_the_tile_pass(capsys: pytest.CaptureFixture) -> None:
    # A tile the tissue does not reach is skipped quietly; a registration with a linear stage is no such tile.
    image = _volume(32)
    empty = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(image)
    assert not _engine().register(image, image, -1, empty).any()
    assert "fixed mask is empty" in capsys.readouterr().out
    _engine(linear_method="none", deformable_method="syn").register(image, image, -1, empty)
    assert "fixed mask is empty" not in capsys.readouterr().out


_SHIFT = np.array([4.0, -3.0, 2.0])  # mm, LPS: the moving content is the fixed content moved by this


def _rotation(axis: int, degrees: float) -> np.ndarray:
    matrix = np.eye(3)
    first, second = [a for a in range(3) if a != axis]
    angle = np.deg2rad(degrees)
    matrix[first, first] = matrix[second, second] = np.cos(angle)
    matrix[first, second], matrix[second, first] = -np.sin(angle), np.sin(angle)
    return matrix


def _sampled(size, spacing, origin, direction: np.ndarray, shift: np.ndarray) -> sitk.Image:
    """One analytic function of the physical point, sampled on a grid of its own: moving(p) = fixed(p - shift)."""
    index = np.stack(np.meshgrid(*[np.arange(s) for s in size], indexing="ij"), -1).reshape(-1, 3)
    points = np.asarray(origin) + (direction @ (index * np.asarray(spacing)).T).T - shift
    values = np.zeros(len(points))
    for i, (x, y, z, width) in enumerate(
        [(0, 0, 0, 9), (12, 5, -4, 5), (-8, 10, 6, 6), (5, -12, 8, 4), (-10, -6, -8, 7)]
    ):
        values += (1.0 + 0.5 * i) * np.exp(-((points - (x, y, z)) ** 2).sum(1) / (2 * width**2))
    image = sitk.GetImageFromArray(np.transpose(100 * values.reshape(size), (2, 1, 0)).astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction.flatten().tolist())
    return image


@pytest.mark.parametrize(("metric", "mode"), [("mse", "Jacobian"), ("impact", "Jacobian"), ("impact", "Static")])
def test_the_field_is_the_physical_displacement_on_the_fixed_grid(tmp_path, metric: str, mode: str) -> None:
    # Two oblique, anisotropic, different grids: the field must read +shift (mm, LPS world axes) on the fixed
    # grid, fixed point x mapping to moving point x + d(x). 'impact' registers world-aligned copies and samples
    # the field back on the fixed image as it came; Static registers extracted feature volumes instead.
    fixed = _sampled((40, 36, 34), (1.2, 1.0, 1.25), (-22.0, -18.0, -20.0), _rotation(2, 20) @ _rotation(0, 10), 0)
    moving = _sampled((44, 40, 30), (1.0, 1.1, 1.4), (-20.0, -22.0, -18.0), _rotation(1, -15), _SHIFT)
    path = tmp_path / "one_layer.pt"
    torch.jit.script(_OneLayer()).save(str(path))
    engine = _engine(
        scales=[2, 1],
        affine_iterations=[100, 50],
        deformable_iterations=[40, 20],
        cc_kernel=5,
        affine_lr=0.1,
        linear_method="rigid_affine",
        deformable_method="syn",
        deformable_metric=metric,
        deformable_lr=0.25,
        impact_levels=[[fireants.ModelSpec(ref=str(path))]] * 2,
        mode=mode,
    )
    # No mask given: konfai-apps fills both mask branches with all ones on the FIXED grid, which world-aligned
    # copies once turned into partial masks on a grid the moving image does not share.
    ones = sitk.Image(fixed.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(fixed)
    field = engine.register(fixed, moving, -1, ones, ones)
    assert field.shape == (3, *fixed.GetSize()[::-1]) and field.dtype == np.float32
    content = sitk.GetArrayViewFromImage(fixed) > 0.2 * sitk.GetArrayViewFromImage(fixed).max()
    np.testing.assert_allclose([np.median(component[content]) for component in field], _SHIFT, atol=0.5)


def test_the_centre_of_mass_seed_is_not_dragged_by_a_bright_voxel() -> None:
    # A lone bright voxel (metal, a contrast-filled vessel) outweighed whole organs on raw intensities; the seed
    # is taken on intensities winsorised to their 0.5-99.5 percentiles, where it weighs one voxel.
    body = sitk.GetArrayFromImage(_volume(32))
    outlier = body.copy()
    outlier[1, 1, 1] = 1e6
    fixed, moving = sitk.GetImageFromArray(body), sitk.GetImageFromArray(outlier)
    for image in (fixed, moving):
        image.SetSpacing((2.0, 2.0, 2.0))
    seed = fireants.FireANTsEngine._center_of_mass_translation(fixed, moving, None, None, "cpu")
    assert float(seed.norm()) < 0.5  # mm


class _ThreeChannels(torch.nn.Module):
    """A pointwise feature model with several channels, so a distance over channels has some to reduce."""

    def forward(self, x: torch.Tensor, nb_layer: torch.Tensor, stats: torch.Tensor, direction: torch.Tensor):
        return [torch.cat([x, x * x, 1.0 - x], dim=1)]


@pytest.mark.parametrize("name", DISTANCES)
def test_every_distance_scores_inside_a_partial_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # Jacobian mode averages each layer where the mask holds; Dice, Cosine and NCC reduce over the channel axis,
    # which a flat element selection had removed (IndexError after the linear stages).
    path = tmp_path / "features.pt"
    torch.jit.script(_ThreeChannels()).save(str(path))
    monkeypatch.setattr(ImpactFeatureModel, "check", lambda self, gradient=False: None)
    core = fireants._ImpactCore(fireants.ModelSpec(ref=str(path), distance=name), False)
    torch.manual_seed(0)
    moved, fixed = torch.rand(1, 1, 6, 6, 6, requires_grad=True), torch.rand(1, 1, 6, 6, 6)
    mask = torch.zeros(1, 1, 6, 6, 6)
    mask[..., :3] = 1

    (loss,) = core.distances(moved, fixed, mask, [(name, 0)], 0, 3)
    (gradient,) = torch.autograd.grad(loss, moved)

    assert torch.isfinite(loss) and gradient[..., :3].abs().sum() > 0
    # The LNCC's window reaches one voxel past the mask; nothing further does.
    assert gradient[..., 4 if name == "LNCC" else 3 :].abs().sum() == 0
