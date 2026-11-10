# SPDX-License-Identifier: Apache-2.0
"""examples/benchmark.py on a small head-like volume: every defect planted where it says, the heuristic model run at a
depth, and the scores counted."""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _head(tmp_path: Path) -> dict[str, Path]:
    """A 64 x 64 x 40 ellipsoid head of 1.5 mm voxels: soft tissue in a bony shell, a lesion off its centre."""
    zz, yy, xx = np.indices((40, 64, 64)).astype(np.float32)
    inside = ((zz - 20) / 17) ** 2 + ((yy - 32) / 28) ** 2 + ((xx - 32) / 28) ** 2
    lesion = ((zz - 20) ** 2 + (yy - 26) ** 2 + (xx - 22) ** 2) <= 36
    ct = np.where(inside <= 1, np.where(inside >= 0.8, 900.0, 40.0), -1000.0).astype(np.float32)
    mr = np.where(inside <= 1, 300.0 + 400.0 * lesion, 0.0).astype(np.float32)
    paths = {}
    for name, array in (
        ("ct", ct),
        ("mr", mr),
        ("lesion", lesion.astype(np.uint8)),
        ("mask", (inside <= 1).astype(np.uint8)),
    ):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.5, 1.5, 1.5))
        paths[name] = tmp_path / f"{name}.nii.gz"
        sitk.WriteImage(image, str(paths[name]))
    return paths


def test_the_benchmark_plants_runs_and_scores(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import benchmark

    head = _head(tmp_path)
    manifest = {
        "segmentation": [{"name": "s", "image": head["mr"], "truth": head["lesion"], "label": "lesion"}],
        "registration": [{"name": "r", "image": head["ct"]}],
        "synthesis": [{"name": "y", "source": head["mr"], "target": head["ct"], "mask": head["mask"]}],
    }
    built = benchmark.build(manifest, tmp_path / "bench")
    names = sorted(f"{p.parent.name}/{p.name}" for p in built)
    assert names == sorted(
        [
            *(f"segmentation/s__{v}" for v in ("clean", "under_segmentation", "leak", "spurious_structure", "holes")),
            *(f"registration/r__{v}" for v in ("clean", "residual_misalignment", "folding")),
            *(f"synthesis/y__{v}" for v in ("clean", "intensity_bias", "hallucinated_structure", "missing_structure")),
        ]
    )
    truth = sitk.GetArrayFromImage(sitk.ReadImage(str(head["lesion"])))
    for folder in built:
        defects = json.loads((folder / "defects.json").read_text())
        assert bool(defects) == (not folder.name.endswith("__clean"))
        if folder.parent.name == "segmentation":  # the prediction differs from the truth inside the defect's box only
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / "prediction.nii.gz")))
            assert (prediction != truth).any() == bool(defects)
            for defect in defects.values():
                changed = np.argwhere(prediction != truth)[:, ::-1] * 1.5  # z, y, x indices to x, y, z mm
                low, high = np.array(defect["box_mm"]["min"]), np.array(defect["box_mm"]["max"])
                assert ((changed >= low) & (changed <= high)).all()

    bench = tmp_path / "run"  # the heuristic model on the registration cases: what the run and the scores record
    benchmark.build({"registration": manifest["registration"]}, bench)
    benchmark.run(bench, ["quick"], 1, 2, ["--provider", "mock"])
    scores = benchmark.score(bench)
    s = scores["quick/registration"]
    assert (s["inspections"], s["clean"], s["planted"]) == (3, 1, 2) and set(s["sensitivity"]) == {
        "folding",
        "residual_misalignment",
    }
    assert "| quick | registration |" in (bench / "scores.md").read_text()
    benchmark.run(bench, ["quick"], 1, 2, ["--provider", "mock"])  # resumes: nothing left to run
    assert len(list((bench / "runs" / "quick").iterdir())) == 3
