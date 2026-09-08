# Copyright (c) 2025 Valentin Boussot
# SPDX-License-Identifier: Apache-2.0
"""Generate tiny CT-like data, select the retained BEST model, and verify actual outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = tuple(f"CASE_{index:03d}" for index in range(4))
RUN = "CT_TWO_CLASSES"


def prepare() -> None:
    import numpy as np
    import SimpleITK as sitk

    root = Path("Dataset")
    if root.exists():
        raise ValueError("Dataset already exists. Use a fresh copy of the example for a new run.")
    rng = np.random.default_rng(32)
    zz, yy, xx = np.ogrid[:4, :32, :32]
    for index, name in enumerate(CASES):
        label = ((xx - 15 - index) ** 2 + (yy - 16) ** 2 + (zz - 1.5) ** 2 < 9**2).astype(np.uint8)
        ct = (label * 300.0 - 100.0 + rng.normal(0, 10, label.shape)).astype(np.float32)
        case = root / name
        case.mkdir(parents=True)
        for group, array in (("CT", ct), ("SEG", label)):
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((0.8, 1.2, 2.5))  # (x, y, z); arrays above are (z, y, x).
            image.SetOrigin((12.0 + index, -8.0, 3.0))
            image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
            sitk.WriteImage(image, str(case / f"{group}.mha"), useCompression=False)
    print("Prepared four CT/SEG pairs, labels 0 and 1; CASE_003 is held out during training.")


def checkpoint(directory: Path) -> Path:
    """BEST retains one dated model. Do not deserialize weights just to select its filename."""
    candidates = sorted(
        path
        for path in directory.glob("*.pt")
        if path.is_file() and path.name != "resume_latest.pt" and not path.name.startswith("crash_")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one retained BEST model in {directory}, found {len(candidates)}. "
            "Finish TRAIN in a fresh run or choose one model explicitly with --models."
        )
    return candidates[0]


def verify() -> None:
    import numpy as np
    import SimpleITK as sitk

    predictions = Path("Predictions") / RUN / "Dataset"
    found = {path.parent.name for path in predictions.glob("*/PRED.mha")}
    if found != set(CASES):
        raise ValueError(f"Expected predictions for {CASES}, found {sorted(found)}")
    metric_path = Path("Evaluations") / RUN / "Metric_TRAIN.json"
    metrics = json.loads(metric_path.read_text())
    # The label is named by the evaluator, e.g. PRED:SEG:Dice:1.
    dice_metrics = {key: value for key, value in metrics["case"].items() if "Dice" in key}
    if not dice_metrics:
        raise ValueError("Metric_TRAIN.json contains no Dice values.")
    expected = {}
    for name in CASES:
        reference = sitk.ReadImage(str(Path("Dataset") / name / "SEG.mha"))
        source = sitk.ReadImage(str(Path("Dataset") / name / "CT.mha"))
        prediction = sitk.ReadImage(str(predictions / name / "PRED.mha"))
        for other in (source, prediction):
            if other.GetSize() != reference.GetSize():
                raise ValueError(f"{name}: output/input/reference sizes differ")
            for geometry in ("GetSpacing", "GetOrigin", "GetDirection"):
                if not np.allclose(getattr(other, geometry)(), getattr(reference, geometry)(), rtol=0, atol=1e-6):
                    raise ValueError(f"{name}: {geometry} differs from the reference")
        output = sitk.GetArrayFromImage(prediction)
        target = sitk.GetArrayFromImage(reference)
        if not np.isfinite(output).all() or not np.isin(output, [0, 1]).all():
            raise ValueError(f"{name}: prediction must contain only finite labels 0 and 1")
        foreground, truth = output == 1, target == 1
        intersection = np.count_nonzero(foreground & truth)
        expected[name] = float((2 * intersection + 1e-6) / (foreground.sum() + truth.sum() + 1e-6))
    for key, values in dice_metrics.items():
        if set(values) != set(CASES):
            raise ValueError(f"{key}: metric case names do not match the four outputs")
        if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in values.values()):
            raise ValueError(f"{key}: metrics must be finite numbers")
        np.testing.assert_allclose([values[name] for name in CASES], list(expected.values()), rtol=0, atol=1e-6)
    print(json.dumps({"verified_cases": len(CASES), "geometry": "matches CT and SEG", "Dice": expected}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="write four synthetic CT/SEG pairs in a new Dataset directory")
    selection = commands.add_parser("checkpoint", help="print the single dated BEST checkpoint")
    selection.add_argument("--directory", type=Path, default=Path("Checkpoints") / RUN)
    commands.add_parser("verify", help="check predictions, geometry and Dice against the reference files")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "checkpoint":
            print(checkpoint(args.directory))
        else:
            verify()
    except (ValueError, OSError, AssertionError, KeyError) as error:
        parser.exit(1, f"Verification failed: {error}\n")


if __name__ == "__main__":
    main()
