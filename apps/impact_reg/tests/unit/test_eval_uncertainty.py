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

"""``eval`` and ``uncertainty`` run for real on 24^3 toys, on the CPU, in a second or two each.

No stub stands in for KonfAI here: the evaluation configs ship with the package, so the real evaluator
runs offline, and what is asserted is what a user reads: the metrics JSON and the uncertainty map.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai import cli
from konfai.utils.dataset import write_landmarks

_SHAPE = (24, 24, 24)


def _write(array: np.ndarray, path: Path, spacing=(2.0, 2.0, 2.0), vector: bool = False) -> Path:
    image = sitk.GetImageFromArray(array, isVector=vector)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))
    return path


@pytest.fixture
def toy(tmp_path: Path) -> dict[str, Path]:
    """A fixed/moving pair (the moving shifted one voxel along x), two label maps and two landmark files."""
    fixed = np.random.default_rng(0).random(_SHAPE).astype(np.float32) * 100
    labels = np.zeros(_SHAPE, dtype=np.uint8)
    labels[4:12, 4:12, 4:12] = 1
    labels[12:20, 12:20, 12:20] = 2
    points = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    write_landmarks(points, tmp_path / "fixed.fcsv")
    write_landmarks(points + np.array([0.0, 0.0, 1.0]), tmp_path / "moving.fcsv")
    return {
        "fixed": _write(fixed, tmp_path / "fixed.mha"),
        "moving": _write(np.roll(fixed, 1, axis=2), tmp_path / "moving.mha"),
        "fixed_seg": _write(labels, tmp_path / "fixed_seg.mha"),
        "moving_seg": _write(labels, tmp_path / "moving_seg.mha"),
        "fixed_fid": tmp_path / "fixed.fcsv",
        "moving_fid": tmp_path / "moving.fcsv",
    }


def _main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv])
    try:
        cli.main()
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


def _metrics(output: Path) -> dict[str, float]:
    """Every metric the run wrote, whatever the layout under ``output``: key -> the case's value."""
    values: dict[str, float] = {}
    for path in output.rglob("Metric_TRAIN.json"):
        values.update({key: next(iter(cases.values())) for key, cases in json.loads(path.read_text())["case"].items()})
    return values


def test_eval_needs_no_preset(tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluation configs ship with the package: eval lists no preset (no network) and
    resolves none (no requirement install), and still writes the metrics."""

    def no_listing(*args, **kwargs):
        raise AssertionError("eval listed the registration presets")

    monkeypatch.setattr("impact_reg_konfai.impact_reg.app_names", no_listing)
    out = tmp_path / "Evaluation"
    code = _main(monkeypatch, ["eval", "-f", str(toy["fixed"]), "-m", str(toy["moving"]), "-o", str(out), "-q"])

    assert code == 0
    assert _metrics(out)["FixedImage:MovingImage;Mask:MAE"] > 0


def test_eval_keeps_every_modality(tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Image, segmentation and landmarks in one call: each modality keeps its metrics and its map, in a folder
    of its own. They shared one, which the evaluator clears before writing, so only the TRE survived."""
    out = tmp_path / "Evaluation"
    code = _main(
        monkeypatch,
        [
            "eval",
            *("-f", str(toy["fixed"]), "-m", str(toy["moving"])),
            *("--gt-fixed-seg", str(toy["fixed_seg"]), "--gt-moving-seg", str(toy["moving_seg"])),
            *("--gt-fixed-fid", str(toy["fixed_fid"]), "--gt-moving-fid", str(toy["moving_fid"])),
            *("-o", str(out), "-q"),
        ],
    )

    assert code == 0
    evaluation = out / "P000" / "Evaluation"
    metrics = {
        modality: json.loads((evaluation / modality / "ImpactReg" / "Metric_TRAIN.json").read_text())["case"]
        for modality in ("Image", "Segmentation", "Landmarks")
    }
    assert metrics["Image"]["FixedImage:MovingImage;Mask:MAE"]["P000"] > 0
    assert metrics["Segmentation"]["MovingSeg:FixedSeg:Dice"]["P000"] == pytest.approx(1.0)
    assert metrics["Landmarks"]["FixedFid:MovingFid:TRE"]["P000"] == pytest.approx(1.0)
    maps = {path.relative_to(evaluation).parts[0]: path.name for path in evaluation.rglob("*.mha")}
    assert maps == {"Image": "MAE_map.mha", "Segmentation": "Dice_map.mha"}


@pytest.mark.parametrize(
    "members",
    [
        [(5.0, 0.0, 0.0), (0.0, 5.0, 0.0)],  # same length, another direction: the magnitudes' std read 0
        [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0), (0.0, 3.0, 0.0)],
        [(3.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    ],
)
def test_uncertainty_is_the_spread_of_the_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, members: list[tuple[float, float, float]]
) -> None:
    """The map is the RMS distance of the members' vectors to their mean (sample, N-1), in mm, voxel by voxel."""
    fields = [
        _write(np.broadcast_to(np.float32(member), (*_SHAPE, 3)).copy(), tmp_path / f"member_{index}.mha", vector=True)
        for index, member in enumerate(members)
    ]
    out = tmp_path / "Uncertainty"
    code = _main(monkeypatch, ["uncertainty", "--dvf", *map(str, fields), "-o", str(out), "-q"])

    assert code == 0
    spread = sitk.ReadImage(str(out / "uncertainty" / "Uncertainty.mha"))
    vectors = np.array(members)
    expected = np.sqrt(((vectors - vectors.mean(axis=0)) ** 2).sum(axis=1).sum() / (len(members) - 1))
    np.testing.assert_allclose(sitk.GetArrayFromImage(spread), expected, rtol=1e-6)
    assert spread.GetSpacing() == (2.0, 2.0, 2.0)


def test_landmarks_saved_in_ras_score_a_perfect_transform_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The moving landmarks exactly where an affine sends the fixed ones, saved in RAS: TRE 0. Read as LPS they
    were the reflected points, and this TRE came out several mm with exit code 0."""
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(sitk.VersorTransform((0.0, 0.0, 1.0), 0.3).GetMatrix())
    transform.SetTranslation((5.0, -3.0, 2.0))
    sitk.WriteTransform(transform, str(tmp_path / "Transform.h5"))
    fixed = np.array([[10.0, 20.0, 30.0], [-4.0, 8.0, 1.5], [0.0, 5.0, -7.0]])
    moving_ras = np.array([transform.TransformPoint(point) for point in fixed]) * [-1.0, -1.0, 1.0]
    write_landmarks(fixed, tmp_path / "fixed.fcsv")
    rows = "".join(f"{i},{x},{y},{z},0,0,0,1,1,1,0,F-{i},,\n" for i, (x, y, z) in enumerate(moving_ras))
    (tmp_path / "moving.fcsv").write_text("# Markups fiducial file version = 5.6\n# CoordinateSystem = RAS\n" + rows)
    out = tmp_path / "Evaluation"
    code = _main(
        monkeypatch,
        [
            "eval",
            *("--transform", str(tmp_path / "Transform.h5")),
            *("--gt-fixed-fid", str(tmp_path / "fixed.fcsv"), "--gt-moving-fid", str(tmp_path / "moving.fcsv")),
            *("-o", str(out), "-q"),
        ],
    )

    assert code == 0
    assert _metrics(out)["FixedFid:MovingFid:TRE"] == pytest.approx(0.0, abs=1e-4)


def test_landmark_files_of_different_lengths_are_refused(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    write_landmarks(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), tmp_path / "three.fcsv")
    code = _main(
        monkeypatch,
        [
            "eval",
            *("--gt-fixed-fid", str(tmp_path / "three.fcsv"), "--gt-moving-fid", str(toy["moving_fid"])),
            *("-o", str(tmp_path / "Evaluation"), "-q"),
        ],
    )

    assert code == 1
    assert "3 fixed vs 2 moving landmarks" in capsys.readouterr().err


def test_dice_scores_every_fixed_label_even_above_255(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FreeSurfer-like labels, and a moving segmentation that lost one structure: the lost one scores 0.
    uint8 wrapped the labels to 234, 210 and 2, and the lost structure was dropped from a mean of 1.0."""
    fixed = np.zeros(_SHAPE, dtype=np.uint16)
    fixed[2:10, 2:10, 2:10], fixed[12:20, 2:10, 2:10], fixed[2:10, 12:20, 12:20] = 1002, 2002, 1026
    moving = np.where(fixed == 2002, 0, fixed).astype(np.uint16)
    out = tmp_path / "Evaluation"
    code = _main(
        monkeypatch,
        [
            "eval",
            *("--gt-fixed-seg", str(_write(fixed, tmp_path / "fixed_seg.mha"))),
            *("--gt-moving-seg", str(_write(moving, tmp_path / "moving_seg.mha"))),
            *("-o", str(out), "-q"),
        ],
    )

    assert code == 0
    dice = {key.rsplit(":", 1)[-1]: value for key, value in _metrics(out).items()}
    assert dice["1002"] == pytest.approx(1.0) and dice["1026"] == pytest.approx(1.0)
    assert dice["2002"] == pytest.approx(0.0, abs=1e-6)
    assert dice["Dice"] == pytest.approx(2 / 3)


def _identity_transform(path: Path) -> Path:
    sitk.WriteTransform(sitk.AffineTransform(3), str(path))
    return path


def test_one_transform_serves_every_case(tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Two pairs and one transform: the transform is used for both. The second case was scored as the
    identity, silently, beside a first case scored through the transform."""
    shift = sitk.AffineTransform(3)
    shift.SetTranslation((2.0, 0.0, 0.0))  # undoes the moving's one-voxel (2 mm) shift along x
    sitk.WriteTransform(shift, str(tmp_path / "shift.h5"))
    out = tmp_path / "Evaluation"
    pair = ["-f", str(toy["fixed"]), str(toy["fixed"]), "-m", str(toy["moving"]), str(toy["moving"])]
    code = _main(monkeypatch, ["eval", *pair, "--transform", str(tmp_path / "shift.h5"), "-o", str(out), "-q"])

    assert code == 0
    maes = [_metrics(out / case)["FixedImage:MovingImage;Mask:MAE"] for case in ("P000", "P001")]
    assert maes[0] == pytest.approx(maes[1])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            lambda t, d: ["-f", *[t["fixed"]] * 3, "-m", *[t["moving"]] * 3, "--transform", d / "a.h5", d / "b.h5"],
            "3 cases to evaluate, but --transform has 2.",
        ),
        (
            lambda t, d: ["-f", t["fixed"], "-m", t["moving"], "--gt-fixed-seg", t["fixed_seg"]],
            "--gt-fixed-seg is given without --gt-moving-seg.",
        ),
        (
            lambda t, d: ["-f", t["fixed"], "-m", t["moving"], "--transform", d / "Output"],
            "Moved.mha', register's moved image or ensemble member",
        ),
        (
            lambda t, d: ["-f", t["fixed"], "-m", t["moving"], "--transform", d / "Fields"],
            "A.h5', register's moved image or ensemble member",
        ),
    ],
)
def test_eval_refuses_groups_that_do_not_pair(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, argv, message
) -> None:
    _identity_transform(tmp_path / "a.h5")
    _identity_transform(tmp_path / "b.h5")
    (tmp_path / "Output" / "P000").mkdir(parents=True)  # what register leaves: the moved image beside the transform
    _identity_transform(tmp_path / "Output" / "P000" / "Transform.h5")
    _write(np.zeros(_SHAPE, np.float32), tmp_path / "Output" / "P000" / "Moved.mha")
    (tmp_path / "Fields" / "P000" / "Ensemble").mkdir(parents=True)  # register --fields-only --keep-fields
    for name in ("Transform.h5", "Ensemble/A.h5", "Ensemble/B.h5"):
        _identity_transform(tmp_path / "Fields" / "P000" / name)
    code = _main(monkeypatch, ["eval", *map(str, argv(toy, tmp_path)), "-o", str(tmp_path / "Evaluation"), "-q"])

    assert code == 1
    assert message in capsys.readouterr().err


def test_eval_scores_the_ensemble_members_through_their_folder(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """register's Ensemble folder named itself is a list of transforms: one case per member."""
    (tmp_path / "Ensemble").mkdir()
    for name in ("A.h5", "B.h5"):
        _identity_transform(tmp_path / "Ensemble" / name)
    argv = ["eval", "-f", str(toy["fixed"]), "-m", str(toy["moving"]), "--transform", str(tmp_path / "Ensemble")]

    assert _main(monkeypatch, [*argv, "-o", str(tmp_path / "Evaluation"), "-q"]) == 0
    assert (tmp_path / "Evaluation" / "P001" / "Evaluation" / "Image").is_dir()


def _versor_translation() -> sitk.Transform:
    versor = sitk.VersorRigid3DTransform()
    versor.SetCenter((5.0, 7.0, -3.0))
    versor.SetTranslation((2.0, 0.0, 0.0))
    return versor


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda: sitk.TranslationTransform(3, (2.0, 0.0, 0.0)), "shift.h5"),
        (_versor_translation, "shift.tfm"),
        (lambda: sitk.Similarity3DTransform(1.0, (0.0, 0.0, 1.0), 0.0, (2.0, 0.0, 0.0)), "Shift.H5"),
        (lambda: sitk.CompositeTransform([sitk.TranslationTransform(3, (2.0, 0.0, 0.0))]), "shift.itk.txt"),
    ],
)
def test_eval_warps_through_any_linear_transform_file(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch, make, name: str
) -> None:
    """The same 2 mm shift spelled as the kinds SimpleITK writes, in every transform file form: each scores the
    MAE the plain AffineTransform .h5 scores. Only Euler3D, Affine and BSpline reached the image warp; a .tfm
    was refused by the input listing and a .H5 was taken for a field image."""
    reference = sitk.AffineTransform(3)
    reference.SetTranslation((2.0, 0.0, 0.0))
    sitk.WriteTransform(reference, str(tmp_path / "affine.h5"))
    sitk.WriteTransform(make(), str(tmp_path / name.lower()))  # ITK picks its writer off a lower-case name
    (tmp_path / name.lower()).rename(tmp_path / name)
    pair = ["-f", str(toy["fixed"]), "-m", str(toy["moving"]), "-q"]
    for transform, out in [("affine.h5", "Reference"), (name, "Evaluation")]:
        code = _main(monkeypatch, ["eval", *pair, "--transform", str(tmp_path / transform), "-o", str(tmp_path / out)])
        assert code == 0

    mae = _metrics(tmp_path / "Evaluation")["FixedImage:MovingImage;Mask:MAE"]
    assert mae == pytest.approx(_metrics(tmp_path / "Reference")["FixedImage:MovingImage;Mask:MAE"], rel=1e-5)


def test_eval_warps_through_a_translation_chained_with_a_spline(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A translation then a B-spline whose support leaves out the origin scores what the same chain with the
    translation spelled as an AffineTransform scores: the spline is applied, not dropped."""
    domain = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    domain.SetOrigin((10.0, 10.0, 10.0))
    domain.SetSpacing((4.0, 4.0, 4.0))
    spline = sitk.BSplineTransformInitializer(domain, [2, 2, 2])
    spline.SetParameters(np.random.default_rng(3).normal(0.0, 2.0, len(spline.GetParameters())).tolist())
    affine = sitk.AffineTransform(3)
    affine.SetTranslation((2.0, 0.0, 0.0))
    sitk.WriteTransform(sitk.CompositeTransform([affine, spline]), str(tmp_path / "affine.h5"))
    sitk.WriteTransform(
        sitk.CompositeTransform([sitk.TranslationTransform(3, (2.0, 0.0, 0.0)), spline]), str(tmp_path / "chain.h5")
    )
    pair = ["-f", str(toy["fixed"]), "-m", str(toy["moving"]), "-q"]
    for transform, out in [("affine.h5", "Reference"), ("chain.h5", "Evaluation")]:
        code = _main(monkeypatch, ["eval", *pair, "--transform", str(tmp_path / transform), "-o", str(tmp_path / out)])
        assert code == 0

    mae = _metrics(tmp_path / "Evaluation")["FixedImage:MovingImage;Mask:MAE"]
    assert mae == pytest.approx(_metrics(tmp_path / "Reference")["FixedImage:MovingImage;Mask:MAE"], rel=1e-5)


def test_eval_summarizes_the_cohort(tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Two cases, two modalities: one summary holds each metric of each case under its real id, and the
    cohort's aggregates, readable by the evaluator's own reader (what SlicerImpactReg reads)."""
    from konfai.evaluator import Statistics

    out = tmp_path / "Evaluation"
    code = _main(
        monkeypatch,
        [
            "eval",
            *("-f", str(toy["fixed"]), "-m", str(toy["moving"]), str(toy["fixed"])),
            *("--gt-fixed-fid", str(toy["fixed_fid"]), "--gt-moving-fid", str(toy["moving_fid"])),
            *("-o", str(out), "-q"),
        ],
    )

    assert code == 0
    summary = json.loads((out / "Evaluation_summary.json").read_text())
    mae, tre = "FixedImage:MovingImage;Mask:MAE", "FixedFid:MovingFid:TRE"
    assert set(summary["case"][mae]) == set(summary["case"][tre]) == {"P000", "P001"}
    assert summary["case"][mae]["P000"] > 0 and summary["case"][mae]["P001"] == pytest.approx(0.0)
    assert summary["aggregates"][mae]["count"] == 2
    assert next(out.rglob("*.json")).name == "Evaluation_summary.json"
    assert Statistics(out / "Evaluation_summary.json").read()[tre] == pytest.approx(1.0)


def test_eval_reports_the_jacobian_of_a_field_transform(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A field transform gets its Jacobian statistics beside the modalities' metrics; an affine gets none. The
    landmarks read the field first, through konfai's HDF5 read pool: every later open must agree with it."""
    field = sitk.Image([24, 24, 24], sitk.sitkVectorFloat64, 3)
    field.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteTransform(sitk.DisplacementFieldTransform(field), str(tmp_path / "Transform.h5"))
    for transform, out in [("Transform.h5", "Field"), ("affine.h5", "Affine")]:
        sitk.WriteTransform(sitk.AffineTransform(3), str(tmp_path / "affine.h5"))
        argv = ["eval", "-f", str(toy["fixed"]), "-m", str(toy["moving"]), "--transform", str(tmp_path / transform)]
        argv += ["--gt-fixed-fid", str(toy["fixed_fid"]), "--gt-moving-fid", str(toy["moving_fid"])]
        assert _main(monkeypatch, [*argv, "-o", str(tmp_path / out), "-q"]) == 0

    metrics = _metrics(tmp_path / "Field")
    assert metrics["Transform:Transform:JacobianMin"] == pytest.approx(1.0)
    assert metrics["Transform:Transform:JacobianFolding"] == 0.0
    summary = json.loads((tmp_path / "Field" / "Evaluation_summary.json").read_text())["case"]
    assert {"Transform:Transform:JacobianLogStd", "FixedImage:MovingImage;Mask:MAE", "FixedFid:MovingFid:TRE"} <= set(
        summary
    )
    assert not any("Jacobian" in key for key in _metrics(tmp_path / "Affine"))


def test_one_field_given_for_every_case_is_measured_once(
    tmp_path: Path, toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Jacobian reads the whole field: one field broadcast to several cases is read for the first alone,
    and every case reports it."""
    from impact_reg_konfai import impact_reg

    field = sitk.Image([24, 24, 24], sitk.sitkVectorFloat64, 3)
    field.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteTransform(sitk.DisplacementFieldTransform(field), str(tmp_path / "Transform.h5"))
    measured: list[Path] = []
    original = impact_reg.KonfAIApp.evaluate

    def evaluate(self, *args, **kwargs):
        if kwargs.get("evaluation_file") == "Evaluation_with_jacobian.yml":
            measured.append(kwargs["inputs"][0][0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(impact_reg.KonfAIApp, "evaluate", evaluate)
    pair = ["-f", str(toy["fixed"]), str(toy["fixed"]), "-m", str(toy["moving"]), str(toy["moving"])]
    argv = ["eval", *pair, "--transform", str(tmp_path / "Transform.h5"), "-o", str(tmp_path / "Evaluation"), "-q"]

    assert _main(monkeypatch, argv) == 0
    assert len(measured) == 1
    summary = json.loads((tmp_path / "Evaluation" / "Evaluation_summary.json").read_text())["case"]
    assert set(summary["Transform:Transform:JacobianMin"]) == {"P000", "P001"}


def test_eval_of_a_2d_pair_is_simpleitks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2-D pair through a 2-D field: the MAE, the Dice and the TRE are SimpleITK's own, and the field's Jacobian is
    measured. KonfAI resamples a 2-D image through a 2-D map: no plane is lifted."""
    yy, xx = np.mgrid[0:40, 0:48].astype(np.float64)

    def plane(array: np.ndarray, name: str) -> Path:
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.8, 1.1))
        image.SetOrigin((12.0, -5.0))
        sitk.WriteImage(image, str(tmp_path / name))
        return tmp_path / name

    fixed = plane((np.sin(xx / 5) * np.cos(yy / 6) * 100 + 200).astype(np.float32), "fixed.mha")
    moving = plane((np.sin((xx + 2) / 5) * np.cos(yy / 6) * 100 + 200).astype(np.float32), "moving.mha")
    fixed_seg = plane(((np.sin(xx / 5) + np.cos(yy / 6)) > 0.4).astype(np.uint8) * 2, "fixed_seg.mha")
    moving_seg = plane(((np.sin((xx + 2) / 5) + np.cos(yy / 6)) > 0.4).astype(np.uint8) * 2, "moving_seg.mha")
    grid = sitk.ReadImage(str(fixed))
    field_image = sitk.GetImageFromArray(np.stack([2 * np.sin(yy / 9), 1.5 * np.cos(xx / 11)], axis=-1), isVector=True)
    field_image.CopyInformation(grid)
    field = sitk.DisplacementFieldTransform(sitk.Image(field_image))
    sitk.WriteTransform(field, str(tmp_path / "Transform.h5"))
    points = np.array([[20.0, 3.0, 0.0], [40.0, 20.0, 0.0], [30.0, 30.0, 0.0]])
    offset = np.array([0.5, -0.3, 0.0])
    write_landmarks(points, tmp_path / "fixed.fcsv")
    write_landmarks(points + offset, tmp_path / "moving.fcsv")
    out = tmp_path / "Evaluation"
    argv = ["eval", "-f", str(fixed), "-m", str(moving), "--transform", str(tmp_path / "Transform.h5")]
    argv += ["--gt-fixed-seg", str(fixed_seg), "--gt-moving-seg", str(moving_seg)]
    argv += ["--gt-fixed-fid", str(tmp_path / "fixed.fcsv"), "--gt-moving-fid", str(tmp_path / "moving.fcsv")]

    assert _main(monkeypatch, [*argv, "-o", str(out), "-q"]) == 0
    metrics = _metrics(out)
    warped = sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(str(moving)), grid, field, sitk.sitkLinear, 0.0))
    assert metrics["FixedImage:MovingImage;Mask:MAE"] == pytest.approx(
        np.abs(warped - sitk.GetArrayFromImage(grid)).mean(), rel=1e-5
    )
    moved_seg = sitk.Resample(sitk.ReadImage(str(moving_seg)), grid, field, sitk.sitkNearestNeighbor, 0)
    a, b = sitk.GetArrayFromImage(moved_seg) == 2, sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_seg))) == 2
    assert metrics["MovingSeg:FixedSeg:Dice"] == pytest.approx(2 * (a & b).sum() / (a.sum() + b.sum()), abs=1e-3)
    pushed = np.array([[*field.TransformPoint(point[:2].tolist()), 0.0] for point in points])
    tre = np.linalg.norm(pushed - (points + offset), axis=1).mean()
    assert metrics["FixedFid:MovingFid:TRE"] == pytest.approx(tre, abs=1e-6)
    assert metrics["Transform:Transform:JacobianFolding"] == 0.0
