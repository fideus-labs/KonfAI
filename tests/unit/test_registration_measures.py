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

"""The measures a registration's evaluation configs compose, against independent computations: label overlaps and
surface distances, image similarities, the regularity of a field."""

import numpy as np
import pytest
import torch

pytest.importorskip("SimpleITK")

from konfai.metric.measure import ASSD, HD95, Jaccard, SurfaceDice, VolumeDifference
from konfai.utils.dataset import Attribute

SPACING = [0.8, 1.1, 2.0]  # (x, y, z)


def _labels() -> tuple[torch.Tensor, torch.Tensor]:
    """Two label maps on one [z, y, x] grid: label 1 a box shifted by two voxels along x, label 2 present in both
    with a voxel less along z, label 3 in the reference only."""
    reference = np.zeros((14, 16, 18), dtype=np.uint8)
    output = np.zeros_like(reference)
    reference[3:9, 4:10, 3:9] = 1
    output[3:9, 4:10, 5:11] = 1
    reference[10:13, 11:15, 12:16] = 2
    output[10:12, 11:15, 12:16] = 2
    reference[1:2, 1:3, 15:17] = 3
    return torch.from_numpy(output)[None, None], torch.from_numpy(reference)[None, None]


def _attributes() -> list[list[Attribute]]:
    attribute = Attribute()
    attribute["Spacing"] = np.array(SPACING)
    return [[attribute]]


def _contour(mask: np.ndarray) -> np.ndarray:
    """The voxels of ``mask`` with a face neighbour outside it (the grid's border counts as outside)."""
    padded = np.pad(mask, 1)
    inner = padded[1:-1, 1:-1, 1:-1].copy()
    for axis in range(3):
        for step in (-1, 1):
            inner &= np.roll(padded, step, axis=axis)[1:-1, 1:-1, 1:-1]
    return mask & ~inner


def _directed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Every contour voxel of ``a`` to the nearest contour voxel of ``b``, in mm, by brute force."""
    zyx_spacing = np.array(SPACING[::-1])
    pa, pb = np.argwhere(_contour(a)) * zyx_spacing, np.argwhere(_contour(b)) * zyx_spacing
    return np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1)).min(1)


def test_jaccard_and_volume_difference_count_the_voxels() -> None:
    output, reference = _labels()
    o, r = output.numpy(), reference.numpy()
    jaccard = Jaccard()(output, reference)[1]
    volume = VolumeDifference()(output, reference)[1]
    for label in (1, 2):
        a, b = o == label, r == label
        assert jaccard[label] == pytest.approx((a & b).sum() / (a | b).sum(), rel=1e-6)
        assert volume[label] == pytest.approx(abs(a.sum() - b.sum()) / b.sum())
    assert jaccard[3] == pytest.approx(0.0, abs=1e-6) and volume[3] == pytest.approx(1.0)


def test_jaccard_streams_by_patches() -> None:
    """The sums of two halves give the whole map's Jaccard: it may be read by patches."""
    output, reference = _labels()
    jaccard = Jaccard()
    halves = [
        jaccard.partial_metric(output[..., part, :, :], reference[..., part, :, :])
        for part in (slice(0, 7), slice(7, 14))
    ]
    assert jaccard.combine_metric(halves)[1] == pytest.approx(jaccard(output, reference)[1], nan_ok=True)


def test_surface_distances_are_the_brute_force_ones() -> None:
    output, reference = _labels()
    o, r = output.numpy()[0, 0], reference.numpy()[0, 0]
    hd95 = HD95()(output, reference, attributes=_attributes())[1]
    assd = ASSD()(output, reference, attributes=_attributes())[1]
    nsd = SurfaceDice(tolerance=1.0)(output, reference, attributes=_attributes())[1]
    for label in (1, 2):
        ab, ba = _directed(o == label, r == label), _directed(r == label, o == label)
        assert hd95[label] == pytest.approx(max(np.percentile(ab, 95), np.percentile(ba, 95)), rel=1e-5)
        assert assd[label] == pytest.approx((ab.sum() + ba.sum()) / (ab.size + ba.size), rel=1e-5)
        assert nsd[label] == pytest.approx(((ab <= 1.0).sum() + (ba <= 1.0).sum()) / (ab.size + ba.size))
    # a label the output lost: the grid's diagonal, and no surface within the tolerance
    diagonal = float(np.linalg.norm(np.array(o.shape[::-1]) * np.array(SPACING)))
    assert hd95[3] == pytest.approx(diagonal) and assd[3] == pytest.approx(diagonal) and nsd[3] == 0.0


def test_the_grid_s_border_is_a_surface() -> None:
    """A label touching the grid's face has its surface there too (MedPy's and DeepMind's convention): it scores as
    the same label inside the grid does, and two identical maps filling the grid score 0, not an error."""
    attributes = _attributes()
    reference = torch.zeros(1, 1, 6, 6, 6, dtype=torch.uint8)
    reference[..., 0:3, 0:3, 0:3] = 1
    output = torch.zeros_like(reference)
    output[..., 0:3, 0:3, 1:4] = 1
    inside = torch.zeros(1, 1, 8, 8, 8, dtype=torch.uint8)
    inside[..., 1:4, 1:4, 1:4] = 1
    inside_output = torch.zeros_like(inside)
    inside_output[..., 1:4, 1:4, 2:5] = 1
    for criterion in (HD95(), ASSD(), SurfaceDice(tolerance=1.0)):
        at_the_face = criterion(output, reference, attributes=attributes)[1][1]
        assert at_the_face == pytest.approx(criterion(inside_output, inside, attributes=attributes)[1][1])
    full = torch.ones(1, 1, 5, 6, 7, dtype=torch.uint8)
    assert HD95()(full, full, attributes=attributes)[1] == pytest.approx({1: 0.0}, abs=1e-5)


# ---------------------------------------------------------------- image similarity


def _images() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    z, y, x = np.mgrid[0:12, 0:14, 0:16].astype(np.float64)
    a = np.sin(x / 3) * np.cos(y / 4) * 50 + z + rng.normal(0, 2, x.shape)
    b = 0.7 * a + 10 * np.sin(z / 2) + rng.normal(0, 3, x.shape)
    mask = (x > 3) & (y < 11)
    return (torch.from_numpy(t)[None, None] for t in (a, b, mask))  # type: ignore[return-value]


def test_ncc_is_the_correlation_coefficient_and_streams() -> None:
    from konfai.metric.measure import NCC

    a, b, mask = _images()
    assert NCC()(a, b)[1] == pytest.approx(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    inside = mask.flatten()
    assert NCC()(a, b, mask)[1] == pytest.approx(np.corrcoef(a.flatten()[inside], b.flatten()[inside])[0, 1])
    halves = [NCC().partial_metric(a[..., part, :, :], b[..., part, :, :]) for part in (slice(0, 5), slice(5, 12))]
    assert NCC().combine_metric(halves)[1] == pytest.approx(NCC()(a, b)[1])


def test_nmi_is_numpy_s_and_two_for_a_one_to_one_map() -> None:
    from konfai.metric.measure import NMI

    a, b, _ = _images()
    joint, _, _ = np.histogram2d(a.flatten().numpy(), b.flatten().numpy(), bins=32)
    p = joint / joint.sum()

    def entropy(q):
        q = q[q > 0]
        return -(q * np.log(q)).sum()

    assert NMI(bins=32)(a, b)[1] == pytest.approx((entropy(p.sum(1)) + entropy(p.sum(0))) / entropy(p), rel=1e-3)
    assert NMI(bins=32)(a, 3 * a + 7)[1] == pytest.approx(2.0)
    ranges = {"output_range": [float(a.min()), float(a.max())], "target_range": [float(b.min()), float(b.max())]}
    nmi = NMI(bins=32, **ranges)
    assert nmi.reducible
    halves = [nmi.partial_metric(a[..., part, :, :], b[..., part, :, :]) for part in (slice(0, 5), slice(5, 12))]
    assert nmi.combine_metric(halves)[1] == pytest.approx(nmi(a, b)[1])


def _lncc_brute(x: np.ndarray, y: np.ndarray, radius: int) -> float:
    """The mean over the voxels of cov^2 / (var var) in the (2 radius + 1)-wide window around each, cut at the
    faces."""
    values = []
    for index in np.ndindex(x.shape):
        window = tuple(slice(max(i - radius, 0), i + radius + 1) for i in index)
        wa, wb = x[window], y[window]
        covariance = (wa * wb).mean() - wa.mean() * wb.mean()
        values.append(covariance**2 / (wa.var() * wb.var() + 1e-5))
    return float(np.mean(values))


def test_lncc_is_the_mean_of_its_windows_and_streams_with_its_halo() -> None:
    from konfai.metric.measure import LNCC

    a, b, mask = _images()
    lncc = LNCC(kernel=3)
    assert lncc(a, b)[1] == pytest.approx(_lncc_brute(a[0, 0].numpy(), b[0, 0].numpy(), 1), rel=1e-4)
    # two patches read with the halo, each scoring its own slot
    states = [
        lncc.partial_metric(a[..., 0:6, :, :], b[..., 0:6, :, :], core=(slice(0, 5), slice(None), slice(None))),
        lncc.partial_metric(a[..., 4:12, :, :], b[..., 4:12, :, :], core=(slice(1, 8), slice(None), slice(None))),
    ]
    assert lncc.combine_metric(states)[1] == pytest.approx(lncc(a, b)[1], rel=1e-5)
    assert 0 < lncc(a, b, mask)[1] <= 1


def test_lncc_reads_a_volume_thinner_than_its_kernel() -> None:
    """Four planes under the default 9-voxel window (a thin acquisition, a boundary patch): the windows are cut at
    the faces, as everywhere else, where the pool alone refused the image."""
    from konfai.metric.measure import LNCC

    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(4, 10, 10)), rng.normal(size=(4, 10, 10))
    a, b = (torch.from_numpy(t)[None, None] for t in (x, y))
    assert LNCC()(a, b)[1] == pytest.approx(_lncc_brute(x, y, 4), rel=1e-4)


def test_an_empty_mask_scores_nan_not_an_error() -> None:
    from konfai.metric.measure import LNCC, NCC, NMI

    a, b, _ = _images()
    empty = torch.zeros_like(a)
    for criterion in (NCC(), NMI(), LNCC(kernel=3)):
        assert np.isnan(criterion(a, b, empty)[1]), criterion.get_name()


def test_the_measures_are_metrics_not_losses() -> None:
    """Each returns a value without a gradient: a training config that leaves ``is_loss`` out takes it as a metric,
    and ``is_loss: true`` on it is refused."""
    from konfai.metric.measure import LNCC, NCC, NMI, BendingEnergy, Displacement, JacobianFolding, PointDistance

    for criterion in (NCC, NMI, LNCC, PointDistance, BendingEnergy, Displacement, JacobianFolding):
        assert not criterion.loss_capable, criterion.__name__
    for criterion in (Jaccard, VolumeDifference, HD95, ASSD, SurfaceDice):
        assert not criterion.loss_capable, criterion.__name__


@pytest.mark.parametrize("offset", [0.0, 1000.0, 30000.0])
def test_the_correlations_hold_far_from_zero(offset: float) -> None:
    """An MR or a CT sits far from 0: the moments' differences cancelled in float32, and LNCC scored 1672 for an image
    against itself plus noise at 1000. Both correlations are now the same wherever the intensities sit, and in
    [0, 1] for LNCC."""
    from konfai.metric.measure import LNCC, NCC

    rng = np.random.default_rng(0)
    base = torch.from_numpy(rng.normal(size=(1, 1, 12, 12, 12))).float() * 20
    noisy = base + torch.from_numpy(rng.normal(size=base.shape)).float()
    lncc, ncc = LNCC(kernel=5)(base + offset, noisy + offset)[1], NCC()(base + offset, noisy + offset)[1]
    assert 0 <= lncc <= 1
    assert lncc == pytest.approx(LNCC(kernel=5)(base, noisy)[1], rel=1e-4)
    assert ncc == pytest.approx(NCC()(base, noisy)[1], rel=1e-5)


# ---------------------------------------------------------------- the field's regularity


def _field(values: np.ndarray, spacing=(1.5, 2.0, 2.5)) -> tuple[torch.Tensor, Attribute]:
    import SimpleITK as sitk
    from konfai.utils.dataset import image_to_data

    image = sitk.GetImageFromArray(values, isVector=True)
    image.SetSpacing(spacing)
    image.SetDirection(sitk.VersorTransform((0.2, 0.3, 0.9), 0.4).GetMatrix())
    data, attributes = image_to_data(image)
    return torch.from_numpy(data)[None], attributes


def _measure(criterion, field: torch.Tensor, attributes: Attribute) -> float:
    return float(criterion(field, attributes=[[]], output_attributes=[attributes]))


def test_an_affine_map_bends_nothing_and_slabs_bend_as_the_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    from konfai.metric.measure import BendingEnergy

    z, y, x = np.mgrid[0:7, 0:6, 0:5].astype(np.float64)
    linear = np.stack([0.2 * x - 0.1 * z, 0.3 * y, 0.1 * x + 0.05 * y], -1)
    assert _measure(BendingEnergy(), *_field(linear)) == pytest.approx(0.0, abs=1e-12)
    curved = np.random.default_rng(3).normal(0.0, 1.0, (7, 6, 5, 3))
    whole = _measure(BendingEnergy(), *_field(curved))
    monkeypatch.setattr("konfai.utils.ITK.FIELD_SLAB_VOXELS", 1)
    assert _measure(BendingEnergy(), *_field(curved)) == pytest.approx(whole)
    assert whole > 0


def test_bending_energy_is_the_squared_world_hessian() -> None:
    """On an axis-aligned grid, the world second derivatives are numpy's own with the physical spacing, on every
    voxel. On an oblique grid the same values bend as much: the Hessian's norm does not see a rotation, and a
    transpose slipped into the change of axes would, with an anisotropic spacing."""
    import SimpleITK as sitk
    from konfai.metric.measure import BendingEnergy

    values = np.random.default_rng(5).normal(0.0, 1.0, (7, 6, 5, 3))
    spacing = (1.5, 2.0, 2.5)  # x, y, z
    components = np.moveaxis(values, -1, 0)  # [c, z, y, x]
    expected = 0.0
    for component in components:
        for first in np.gradient(component, spacing[2], spacing[1], spacing[0]):
            for second in np.gradient(first, spacing[2], spacing[1], spacing[0]):
                expected += (second**2).sum()
    expected /= components[0].size
    image = sitk.GetImageFromArray(values, isVector=True)
    image.SetSpacing(spacing)
    from konfai.utils.dataset import image_to_data

    data, attributes = image_to_data(image)
    assert _measure(BendingEnergy(), torch.from_numpy(data)[None], attributes) == pytest.approx(expected)
    assert _measure(BendingEnergy(), *_field(values, spacing)) == pytest.approx(expected)


def test_displacement_is_the_length_of_the_vectors() -> None:
    from konfai.metric.measure import Displacement

    values = np.random.default_rng(4).normal(0.0, 2.0, (6, 5, 4, 3))
    lengths = np.linalg.norm(values, axis=-1)
    field, attributes = _field(values)
    assert _measure(Displacement("mean"), field, attributes) == pytest.approx(lengths.mean())
    assert _measure(Displacement("p95"), field, attributes) == pytest.approx(np.percentile(lengths, 95), rel=1e-5)
    assert _measure(Displacement("max"), field, attributes) == pytest.approx(lengths.max())
    assert Displacement("p95").get_name() == "DisplacementP95"


def test_jaccard_and_volume_difference_take_a_reference_on_another_grid() -> None:
    """Both inherit Dice's ``resamples_target``: a reference of another shape is put on the output's grid."""
    output, reference = _labels()
    finer = torch.nn.functional.interpolate(reference.float(), scale_factor=2.0, mode="nearest")
    for criterion in (Jaccard(), VolumeDifference()):
        assert criterion(output, finer)[1] == pytest.approx(criterion(output, reference)[1], rel=1e-6)
