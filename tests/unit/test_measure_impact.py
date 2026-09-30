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

"""The IMPACT feature models: a ref fetched at its pinned revision, shaped by the registry, probed once."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.metric.measure import impact
from konfai.metric.measure.impact import _EPS, ImpactFeatureModel, IMPACTReg, distance, onto_image_grid
from konfai.utils.errors import MeasureError


class _FeaturesAndHead(torch.nn.Module):
    """A feature layer, then a one-hot segmentation head in int64, as the TotalSegmentator models end."""

    def forward(self, x: torch.Tensor, nb_layers: torch.Tensor) -> list[torch.Tensor]:
        return [x * 2.0, (x > 0).long()]


def test_a_ref_is_fetched_at_the_pinned_revision_unless_it_names_one(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(impact, "_hf_hub_download", lambda criterion: lambda **kw: calls.append(kw) or str(tmp_path))
    impact.fetch_model("org/repo:TS/M730.pt")
    impact.fetch_model("org/repo@abc123:TS/M730.pt")
    assert [(c["repo_id"], c["revision"], c["filename"]) for c in calls] == [
        ("org/repo", impact.MODELS_REVISION, "TS/M730.pt"),
        ("org/repo", "abc123", "TS/M730.pt"),
    ]


def test_a_local_ref_must_exist_and_a_drive_letter_is_a_local_path(tmp_path: Path) -> None:
    with pytest.raises(MeasureError, match="does not exist"):
        impact.fetch_model(str(tmp_path / "typo.pt"))
    assert impact._is_local_ref("C:/models/m.pt") and impact._is_local_ref(r"D:\models\m.pt")
    assert not impact._is_local_ref("org/repo:MIND/R1D2.pt")
    assert impact.model_key("C:/models/m.pt") == "C:/models/m.pt"
    assert impact.model_key("org/repo:MIND/R1D2.pt") == "MIND/R1D2.pt"


def test_a_model_takes_its_shape_from_the_registry_and_a_local_one_the_default(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "model.pt"
    torch.jit.script(_FeaturesAndHead()).save(str(path))
    registry = tmp_path / "models.json"
    entry = {"dimension": "2", "numberofchannels": "3", "fov": [5, 11], "multiple": 16}
    registry.write_text(json.dumps({"VGG/VGG16.pt": entry, str(path): {**entry, "dimension": "3"}}))
    monkeypatch.setattr(impact, "fetch_model", lambda ref: path)
    monkeypatch.setenv("KONFAI_IMPACT_MODELS_REGISTRY", str(registry))

    model = ImpactFeatureModel.from_ref("org/repo:VGG/VGG16.pt", [1.0, 0.0])
    assert (model.dim, model.in_channels, model.fov, model.multiple) == (2, 3, [5, 11], 16)
    assert ImpactFeatureModel.from_ref(str(path), [1.0, 0.0]).dim == 3  # a local model the registry names
    monkeypatch.delenv("KONFAI_IMPACT_MODELS_REGISTRY")
    local = ImpactFeatureModel.from_ref(str(path), [1.0, 0.0])
    assert (local.dim, local.in_channels, local.fov, local.multiple) == (3, 1, None, 0)


def test_the_receptive_field_is_the_deepest_weighted_layer_s() -> None:
    model = ImpactFeatureModel("m.pt", 1, [1.0, 1.0, 0.0], None, 3)
    model.fov = [5, 11, 23]
    assert model.receptive_field == 11
    model.weights = [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(MeasureError, match="past the 3"):
        _ = model.receptive_field
    model.fov = None
    with pytest.raises(MeasureError, match="whole images only"):
        _ = model.receptive_field


def test_a_weighted_layer_without_a_gradient_is_refused_only_where_the_loss_needs_one(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.jit.script(_FeaturesAndHead()).save(str(path))
    ImpactFeatureModel(str(path), 1, [1.0, 0.0], None, 3).check(gradient=True)  # the features: fine
    ImpactFeatureModel(str(path), 1, [0.0, 1.0], None, 3).check()  # compared as maps: fine
    with pytest.raises(MeasureError, match="layer 2 carries no gradient"):
        ImpactFeatureModel(str(path), 1, [0.0, 1.0], None, 3).check(gradient=True)
    with pytest.raises(RuntimeError, match="number of weights"):
        ImpactFeatureModel(str(path), 1, [1.0], None, 3).check()


def test_the_registry_the_environment_names_gives_every_model_its_shape_and_receptive_fields() -> None:
    if not os.environ.get("KONFAI_IMPACT_MODELS_REGISTRY"):
        pytest.skip("KONFAI_IMPACT_MODELS_REGISTRY names no registry to check")
    for name, entry in impact.models_registry().items():
        assert int(entry["dimension"]) in (2, 3) and int(entry["numberofchannels"]) >= 1, name
        assert entry["fov"] is None or all(int(side) >= 1 for side in entry["fov"]), name
        assert int(entry.get("multiple", 1)) >= 1, name


# The itk-impact distances: positive, 0 at a perfect match, on feature maps [B, C, *spatial] (channel axis 1), plain
# values autograd differentiates.
DISTANCES = ["L1", "L2", "Dice", "Cosine", "L1Cosine", "NCC", "LNCC"]


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.rand(1, 4, 6, 6, 6, requires_grad=True), torch.rand(1, 4, 6, 6, 6)


@pytest.mark.parametrize("name", DISTANCES)
def test_every_distance_is_lowest_at_a_perfect_match_and_differentiable(name: str) -> None:
    moved, fixed = _features()
    loss = distance(name, moved, fixed, None, 3, 0)
    assert loss.item() > 0 and torch.autograd.grad(loss, moved)[0].abs().sum() > 0
    aligned = distance(name, fixed, fixed, None, 3, 0).item()
    if name == "Dice":  # a soft overlap of real-valued maps is not 1, but beats the random pair
        assert aligned < loss.item()
    else:
        assert aligned == pytest.approx(0.0, abs=1e-4)


def test_cosine_l1cosine_and_ncc_follow_itk_impact() -> None:
    moved, fixed = _features()
    x, y = moved.detach().numpy()[0], fixed.numpy()[0]
    cosine = (x * y).sum(0) / (np.sqrt((x**2).sum(0)) * np.sqrt((y**2).sum(0)) + _EPS)
    assert distance("Cosine", moved, fixed, None, 3, 0).item() == pytest.approx(1 - cosine.mean(), abs=1e-5)
    damped = (cosine[None] * np.exp(-0.1 * np.abs(y - x))).mean()
    assert distance("L1Cosine", moved, fixed, None, 3, 0).item() == pytest.approx(1 - damped, abs=1e-5)
    xf, yf = x.reshape(4, -1), y.reshape(4, -1)
    xf, yf = xf - xf.mean(1, keepdims=True), yf - yf.mean(1, keepdims=True)
    ncc = (xf * yf).sum(1) / (np.sqrt((xf**2).sum(1) * (yf**2).sum(1)) + _EPS)
    assert distance("NCC", moved, fixed, None, 3, 0).item() == pytest.approx(1 - ncc.mean(), abs=1e-5)


@pytest.mark.parametrize(("name", "formula"), [("L1", lambda d: d.abs()), ("L2", lambda d: d.pow(2))])
def test_l1_and_l2_are_the_mean_of_their_terms_for_both_maps_and_in_half_precision(name: str, formula) -> None:
    # Unmasked, they run through torch.dist and mse_loss, which keep less memory: the same value and gradients, the
    # fixed map's too (SyN warps both halves), and a half map's sum must not overflow.
    moved, fixed = _features()
    fixed.requires_grad_(True)
    fast = distance(name, moved, fixed, None, 3, 0)
    plain = formula(moved - fixed).mean()
    torch.testing.assert_close(fast, plain)
    for got, want in zip(
        torch.autograd.grad(fast, (moved, fixed)), torch.autograd.grad(plain, (moved, fixed)), strict=True
    ):
        torch.testing.assert_close(got, want)
    large = torch.full((1, 8, 64, 64, 64), 3.0, dtype=torch.float16)  # a sum of 6.3e6, past half's 65504
    expected = formula(torch.tensor(3.0)).item()
    assert distance(name, large, torch.zeros_like(large), None, 3, 0).item() == pytest.approx(expected)


def test_the_lncc_a_few_channels_a_pass_is_the_lncc() -> None:
    moved, fixed = _features()
    whole = distance("LNCC", moved, fixed, None, 3, 0)
    chunked = distance("LNCC", moved, fixed, None, 3, 3)
    torch.testing.assert_close(chunked, whole)
    torch.testing.assert_close(torch.autograd.grad(chunked, moved)[0], torch.autograd.grad(whole, moved)[0])


class _TwoLayers(torch.nn.Module):
    """An IMPACT extractor of two layers, the second at half the resolution."""

    def forward(self, x: torch.Tensor, nb_layer: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
        return [torch.cat([x, x * x], dim=1), torch.nn.functional.avg_pool3d(x, 2)]


def _impact_reg(tmp_path: Path, monkeypatch, **kwargs) -> IMPACTReg:
    """An IMPACTReg over a local extractor, built as plain torch: no download, no KonfAI configuration."""
    path = tmp_path / "two_layers.pt"
    torch.jit.script(_TwoLayers()).save(str(path))
    monkeypatch.delenv("KONFAI_CONFIG_PATH", raising=False)
    monkeypatch.setattr(
        ImpactFeatureModel,
        "download",
        classmethod(lambda cls, name, c, weights, shape: cls(str(path), c, weights, None, 3)),
    )
    return IMPACTReg(in_channels=1, weights=[1.0, 0.5], **kwargs)


def test_impact_reg_sums_each_layer_s_distance_weighed_by_the_mask(tmp_path: Path, monkeypatch) -> None:
    # Plain torch, as a VoxelMorph-like network trains with it: no attributes (each image its own statistics), the
    # mask a uint8 target weighing each layer's voxels, nearest-resampled to it, and a gradient back to the output.
    loss = _impact_reg(tmp_path, monkeypatch, distance="L2")
    torch.manual_seed(0)
    output, target = torch.rand(1, 1, 8, 8, 8, requires_grad=True), torch.rand(1, 1, 8, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, 8, dtype=torch.uint8)
    mask[..., :4] = 1

    value, _ = loss(output, target, mask)

    layers = [_TwoLayers()(image, torch.tensor([2]), torch.zeros(4)) for image in (output, target)]
    weights = [mask.float(), torch.nn.functional.interpolate(mask.float(), size=(4, 4, 4), mode="nearest")]
    expected = sum(
        w * distance("L2", o, t, m) for w, o, t, m in zip([1.0, 0.5], layers[0], layers[1], weights, strict=True)
    )
    torch.testing.assert_close(value.reshape(()), expected)
    (gradient,) = torch.autograd.grad(value.sum(), output)
    assert gradient[..., :4].abs().sum() > 0 and gradient[..., 4:].abs().sum() == 0


def test_impact_reg_keeps_its_classpath_loss_by_default(tmp_path: Path, monkeypatch) -> None:
    # Training configs name a loss by classpath: the mask selects the voxels it compares, as before.
    loss = _impact_reg(tmp_path, monkeypatch)
    assert isinstance(loss.loss, torch.nn.L1Loss)
    torch.manual_seed(0)
    output, target = torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)
    value, _ = loss(output, target)
    layers = [_TwoLayers()(image, torch.tensor([2]), torch.zeros(4)) for image in (output, target)]
    expected = sum(w * (o - t).abs().mean() for w, o, t in zip([1.0, 0.5], layers[0], layers[1], strict=True))
    torch.testing.assert_close(value.reshape(()), expected)
    with pytest.raises(MeasureError, match="Unknown IMPACT distance"):
        _impact_reg(tmp_path, monkeypatch, distance="SSD")


def test_a_model_is_told_its_images_direction_and_runs_in_float16_on_a_gpu_only() -> None:
    model = ImpactFeatureModel("m.pt", 1, [1.0], None, 3)
    stats = {"ImageMin": 0.0, "ImageMax": 1.0, "ImageMean": 0.5, "ImageStd": 0.3}
    assert len(model.inputs(torch.rand(1, 1, 4, 4, 4), stats)) == 3  # as given without a direction
    model.direction, model.half = torch.eye(3, dtype=torch.int16), True
    inputs = model.inputs(torch.rand(1, 1, 4, 4, 4), stats)
    assert torch.equal(inputs[3], torch.eye(3, dtype=torch.int16)) and inputs[0].dtype == torch.float32  # the CPU


def test_an_image_is_resampled_as_itk_impact_resamples_it() -> None:
    # ImageToTensorFilter: the size is the extent over the voxel size, rounded; new voxel i sits at old index
    # i * old / new, linearly interpolated, not smoothed. The same mapping brings the features back.
    ramp = torch.arange(10.0).expand(1, 1, 2, 2, 10).clone()
    down = impact.resampled(ramp, (2, 2, 5))
    assert torch.allclose(down[0, 0, 0, 0], torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0]))
    back = impact.resampled(down, (2, 2, 10), padding="border")
    assert torch.allclose(back[0, 0, 0, 0], torch.tensor([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 8]))
    assert impact.grid_size([100.0, 50.0, 30.0], [2.0, 2.0, 3.0], (30, 50, 100)) == (10, 25, 50)


class _TwoLayers2D(torch.nn.Module):
    """A 2D network of the IMPACT interface: its input doubled, then the same pooled to half the size."""

    def forward(
        self, x: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor, direction: torch.Tensor
    ) -> list[torch.Tensor]:
        doubled = x[:, :1] * 2
        return [doubled, torch.nn.functional.avg_pool2d(doubled, 2)]


def test_a_2d_network_is_swept_along_the_first_spatial_axis_and_comes_back_as_volumes() -> None:
    # The engine's tensors are LPS-aligned [B, C, S, P, L]: slice s of the volume must be what the network saw.
    image = torch.rand(1, 3, 5, 8, 6)
    layers = impact._SliceSweep(torch.jit.script(_TwoLayers2D()), batch=2)(
        image, torch.tensor([2]), torch.zeros(4), torch.eye(3, dtype=torch.int16)
    )

    assert [tuple(layer.shape) for layer in layers] == [(1, 1, 5, 8, 6), (1, 1, 5, 4, 3)]
    for s in range(5):
        assert torch.equal(layers[0][0, 0, s], 2 * image[0, 0, s])


def test_a_2d_models_pca_basis_comes_from_at_most_32_evenly_spread_fixed_slices() -> None:
    # itk-impact's rule: the basis is fitted on the fixed features of round(linspace(0, n-1, min(n, 32))) slices, each
    # image centred by its own mean over those slices; slices left out of the sample do not move the basis.
    torch.manual_seed(0)
    fixed, moving = torch.randn(1, 6, 40, 4, 4), torch.randn(1, 6, 40, 4, 4)
    sampled = torch.linspace(0, 39, impact.PCA_SLICES).round().long()
    left_out = sorted(set(range(40)) - set(sampled.tolist()))

    projected_moving, projected_fixed = impact.pca_project(moving, fixed, 3, 2)
    fixed[:, :, left_out] *= 100  # outside the sample: no effect on the basis or the means
    again_moving, again_fixed = impact.pca_project(moving, fixed, 3, 2)

    assert projected_fixed.shape == (1, 3, 40, 4, 4) and projected_moving.shape == (1, 3, 40, 4, 4)
    assert torch.allclose(projected_moving, again_moving, atol=1e-5)
    sample = fixed[0][:, sampled].reshape(6, -1)
    centred = sample - sample.mean(dim=1, keepdim=True)
    basis = torch.linalg.eigh(centred @ centred.t() / (centred.shape[1] - 1))[1][:, 3:]
    expected = torch.einsum("cn,ck->kn", fixed[0].reshape(6, -1) - sample.mean(dim=1, keepdim=True), basis)
    assert torch.allclose(again_fixed[0].reshape(3, -1), expected, atol=1e-4)


def test_a_dense_loss_sweeps_a_2d_network_along_an_axis_drawn_at_each_evaluation() -> None:
    # elastix draws a 2D network's plane at random at every sampled point; a dense loss draws no points, so it draws the
    # axis it sweeps: each of the three over the evaluations, the same one per seed.
    firsts = set()
    for seed in range(30):
        order = impact.swept_order(seed)
        assert order[:2] == [0, 1] and sorted(order[2:]) == [2, 3, 4] and order == impact.swept_order(seed)
        firsts.add(order[2])
    assert firsts == {2, 3, 4}


def _stub_model(network: torch.nn.Module, weights: list[float]) -> ImpactFeatureModel:
    """A 3D model around a stub network taking ``(tile, nb_layers, stats)``."""
    model = ImpactFeatureModel("stub.pt", 1, weights, None, 3)
    model.model = network
    return model


class _Constant(torch.nn.Module):
    def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
        return [torch.full((tile.shape[0], 2, *tile.shape[2:]), 3.0)]


def test_a_tiled_volume_leaves_no_seam() -> None:
    # KonfAI's cosine window sums to one over the overlap: constant features come back constant.
    (volume,), tile = _stub_model(_Constant(), [1.0]).volume(torch.rand(1, 1, 40, 24, 70), patch=32, overlap=0.25)
    assert tile == 32 and volume.shape == (1, 2, 40, 24, 70)
    assert torch.allclose(volume, torch.full_like(volume, 3.0), atol=1e-5)


class _NeedsSixteen(torch.nn.Module):
    def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
        if any(size % 16 for size in tile.shape[2:]):
            raise RuntimeError(f"sizes of tensors must match: {tuple(tile.shape[2:])}")
        return [tile.repeat(1, 4, 1, 1, 1)]


def test_a_volume_rounds_its_input_up_to_the_models_multiple() -> None:
    # anatomix's skip connections only meet on a multiple of 16; the padding must not reach the result.
    model, image = _stub_model(_NeedsSixteen(), [1.0]), torch.rand(1, 1, 40, 40, 40)
    with pytest.raises(RuntimeError, match="sizes of tensors"):
        model.volume(image)
    model.multiple = 16
    assert model.volume(image)[0][0].shape == (1, 4, 40, 40, 40)


class _TwoResolutions(torch.nn.Module):
    def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
        coarse = torch.nn.functional.avg_pool3d(tile, 2)
        return [tile.repeat(1, 3, 1, 1, 1), coarse.repeat(1, 5, 1, 1, 1)]


class _HalfConstant(torch.nn.Module):
    def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
        return [torch.full((tile.shape[0], 5, *(size // 2 for size in tile.shape[2:])), 3.0)]


def test_a_volume_blends_each_layer_on_its_own_grid() -> None:
    # A segmentation network hands back coarser deeper layers (M730: 64/32/16 voxels for a 64-voxel tile). As in
    # itk-impact, they are blended and normalised on their grid, the tiles scaled to it, and read at the image voxels.
    layers, _ = _stub_model(_TwoResolutions(), [1.0, 1.0]).volume(torch.rand(1, 1, 16, 16, 16))
    assert [tuple(layer.shape) for layer in layers] == [(1, 3, 16, 16, 16), (1, 5, 8, 8, 8)]
    (coarse,), tile = _stub_model(_HalfConstant(), [1.0]).volume(torch.rand(1, 1, 40, 24, 70), patch=32)
    assert tile == 32 and coarse.shape == (1, 5, 20, 12, 35)
    assert torch.allclose(coarse, torch.full_like(coarse, 3.0), atol=1e-5)
    assert onto_image_grid(coarse, (40, 24, 70)).shape == (1, 5, 40, 24, 70)


class _WithHead(torch.nn.Module):
    def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
        labels = (tile > 0.5).long()
        return [tile.repeat(1, 2, 1, 1, 1), torch.cat([1 - labels, labels], dim=1)]


def test_a_volume_takes_a_segmentation_head_as_float_features() -> None:
    # The TotalSegmentator models end on a one-hot head in int64, which the l2 normalisation refused.
    image = torch.rand(1, 1, 16, 16, 16)
    (volume,), _ = _stub_model(_WithHead(), [0.0, 1.0]).volume(image, "l2")
    assert volume.dtype == torch.float32 and torch.equal(volume[:, 1:], (image > 0.5).float())


def test_a_volume_that_does_not_fit_is_extracted_again_in_smaller_tiles(monkeypatch) -> None:
    # A whole image that runs out of memory is cut in FIRST_FEATURE_TILE tiles, halved below the longest axis.
    model, passes = _stub_model(_Constant(), [1.0]), []
    whole = ImpactFeatureModel._volume

    def out_of_memory_on_the_whole_image(self, image, normalization, patch, overlap):
        passes.append(patch)
        if patch == 0:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 9.10 GiB.")
        return whole(self, image, normalization, patch, overlap)

    monkeypatch.setattr(ImpactFeatureModel, "_volume", out_of_memory_on_the_whole_image)
    assert model.volume(torch.rand(1, 1, 20, 40, 100))[1] == 64 and passes == [0, 64]


def test_draw_centres_keeps_every_patch_inside_the_image_and_the_mask() -> None:
    # elastix's SampleCheck: a patch around a drawn point stays in the image, and a mask restricts the draw.
    generator, cpu = torch.Generator().manual_seed(0), torch.device("cpu")
    centres = impact.draw_centres((10, 12, 14), None, 5, 0.5, generator, cpu)
    assert len(centres) == round(0.5 * 6 * 8 * 10)
    assert centres.min() >= 2 and bool((centres <= torch.tensor([7, 9, 11])).all())
    mask = torch.zeros(1, 1, 10, 12, 14)
    mask[0, 0, 4, 5, 6] = 1
    assert impact.draw_centres((10, 12, 14), mask, 5, 0.5, generator, cpu).tolist() == [[4, 5, 6]]
    mask.zero_()
    mask[0, 0, 0, 0, 0] = 1  # a patch around it would leave the image
    assert impact.draw_centres((10, 12, 14), mask, 5, 0.5, generator, cpu) is None
    with pytest.raises(ValueError, match="larger than the image"):
        impact.draw_centres((10, 12, 14), None, 11, 0.5, generator, cpu)


def test_a_sampled_point_gets_its_own_random_plane_laid_out_in_mm() -> None:
    # elastix draws each sampled point's plane at random (itk-impact's PatchPlane): the square is centred on the point,
    # its two axes orthogonal, one step the finest voxel side in mm, whatever the voxels' shape.
    shape = (20, 30, 40)  # S, P, L voxels
    extent = [40.0, 30.0, 40.0]  # mm along L, P, S: 1 mm in plane, 2 mm along S
    centres = torch.tensor([[10, 15, 20], [5, 10, 30], [12, 20, 8]])
    grids = impact._plane_grids(centres, 5, shape, extent, seed=7)
    assert grids.shape == (3, 5, 5, 3) and torch.equal(grids, impact._plane_grids(centres, 5, shape, extent, seed=7))

    size, spacing = torch.tensor([40.0, 30.0, 20.0]), torch.tensor([1.0, 1.0, 2.0])
    mm = (grids.double() + 1) / 2 * (size - 1) * spacing  # back to mm, (L, P, S)
    assert torch.allclose(mm[:, 2, 2], centres.flip(1).double() * spacing, atol=1e-4)  # the middle is the point
    u, v = mm[:, 3, 2] - mm[:, 2, 2], mm[:, 2, 3] - mm[:, 2, 2]
    assert torch.allclose(u.norm(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(v.norm(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-4)
    assert torch.allclose((u * v).sum(dim=1), torch.zeros(3, dtype=torch.float64), atol=1e-4)
    assert not torch.allclose(u[0], u[1])  # another plane for another point

    # A ramp along L: the network is fed the image on the plane, and the gradient flows back to it.
    image = torch.arange(40.0).expand(1, 1, *shape).clone().requires_grad_(True)
    patches = impact._grid_patches(image, grids)
    assert patches.shape == (3, 1, 5, 5)
    assert torch.allclose(patches[:, 0, 2, 2], centres[:, 2].float(), atol=1e-4)
    patches.sum().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0

    rest = [torch.tensor([2]), torch.zeros(4)]
    features = impact._centre_features(
        torch.jit.script(_TwoLayers2D()), [0, 1], 5, image, image.detach(), centres, rest, rest, grids
    )
    assert [tuple(feature.shape) for feature in features] == [(3, 1)] * 4
    assert torch.allclose(features[0][:, 0], 2 * centres[:, 2].float(), atol=1e-4)


def test_a_sampled_patch_is_cut_at_the_model_resolution() -> None:
    grids = impact._cube_grids(torch.tensor([[4, 8, 8]]), 3, (8, 16, 16), [16.0, 16.0, 8.0], [2.0, 2.0, 2.0])
    index = (grids.double() + 1) / 2 * (torch.tensor([16.0, 16.0, 8.0], dtype=torch.float64) - 1)  # (L, P, S) voxels
    assert torch.allclose(index[0, 1, 1, 1], torch.tensor([8.0, 8.0, 4.0], dtype=torch.float64))
    assert torch.allclose(index[0, 1, 1, 2] - index[0, 1, 1, 1], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(index[0, 2, 1, 1] - index[0, 1, 1, 1], torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64))
