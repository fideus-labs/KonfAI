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

"""Render the panels the examples overview shows, from real runs on real data.

Two rows. The segmentation panels take the ImpactSeg prediction the shipped example wrote, over its
input CT. The registration panels take a SynthRAD2025 MR/CT pair and the published IMPACT B-spline
transform for that case, from huggingface.co/datasets/VBoussot/synthrad2025-impact-registration, and
report the Dice the transform actually achieves over the case's own label maps.

    pixi run --environment dev python docs/scripts/generate_examples_gallery.py \\
        --mrct-case <case folder> --mrct-transform <Task_1/AB/<case>.txt>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "source" / "_static" / "gallery" / "capabilities"
SIZE = 460
BACKGROUND = "#121819"
WINDOW = (-200.0, 400.0)

# Distinct hues for a label overlay, sampled to stay legible on a dark CT.
LABEL_COLOURS = np.array(
    [
        [0, 0, 0],
        [63, 220, 195],
        [143, 201, 236],
        [183, 165, 245],
        [245, 156, 144],
        [247, 201, 106],
        [126, 231, 135],
        [236, 148, 209],
        [255, 255, 255],
        [120, 200, 255],
        [255, 180, 120],
        [180, 255, 180],
    ],
    dtype=np.uint8,
)


def middle_slice(path: Path, index: int | None = None) -> np.ndarray:
    """One axial plane of a volume, as a float array."""
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return volume[volume.shape[0] // 2 if index is None else index].astype(np.float32)


def to_rgb(plane: np.ndarray, window: tuple[float, float] = WINDOW) -> np.ndarray:
    low, high = window
    grey = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return (np.stack([grey] * 3, axis=-1) * 255).astype(np.uint8)


def overlay_labels(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Paint a label map over a greyscale plane, one hue per label."""
    out = rgb.astype(np.float32)
    for value in np.unique(labels):
        if value == 0:
            continue
        colour = LABEL_COLOURS[int(value) % len(LABEL_COLOURS)].astype(np.float32)
        mask = labels == value
        out[mask] = (1 - alpha) * out[mask] + alpha * colour
    return out.astype(np.uint8)


def save(rgb: np.ndarray, name: str) -> Path:
    """Square-pad to SIZE on the viewport background, so the panels line up."""
    image = Image.fromarray(rgb)
    image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    canvas.save(path, optimize=True)
    print(f"  {path.relative_to(ROOT)}")
    return path


def dice_scores(report: Path) -> np.ndarray:
    """Every label's Dice, as the evaluation recorded it."""
    aggregates = json.loads(report.read_text(encoding="utf-8"))["aggregates"]
    return np.array([entry["mean"] for key, entry in aggregates.items() if "Dice" in key])


def segmentation_panels(example: Path) -> None:
    """A real ImpactSeg prediction over its input CT."""
    ct = middle_slice(example / "Dataset" / "1PC006" / "CT.mha")
    prediction = middle_slice(example / "Output" / "ImpactSeg-Body" / "Output" / "P000" / "Output.mha")
    save(to_rgb(ct), "seg-input.png")
    save(overlay_labels(to_rgb(ct), prediction.astype(np.int32)), "seg-output.png")
    print(f"  labels predicted: {sorted(int(v) for v in np.unique(prediction) if v)}")


def elastix_bspline(path: Path) -> "sitk.BSplineTransform":
    """The B-spline an Elastix parameter file describes, as a SimpleITK transform.

    Elastix writes the map from FIXED points to MOVING points, which is the direction
    ``sitk.Resample`` wants, so the file is usable as it stands.
    """

    def field(key: str) -> str:
        found = re.search(r"\(" + key + r"\s+([^)]*)\)", path.read_text(encoding="utf-8"))
        if found is None:
            raise ValueError(f"{path.name} has no ({key} ...)")
        return found.group(1).strip().strip('"')

    size = [int(value) for value in field("GridSize").split()]
    spacing = [float(value) for value in field("GridSpacing").split()]
    transform = sitk.BSplineTransform(3, int(field("BSplineTransformSplineOrder")))
    transform.SetTransformDomainMeshSize([extent - 3 for extent in size])
    transform.SetTransformDomainOrigin([float(value) for value in field("GridOrigin").split()])
    transform.SetTransformDomainDirection([float(value) for value in field("GridDirection").split()])
    transform.SetTransformDomainPhysicalDimensions(
        [(extent - 1) * step for extent, step in zip(size, spacing, strict=True)]
    )
    transform.SetParameters([float(value) for value in field("TransformParameters").split()])
    return transform


def window(plane: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((plane - low) / (high - low), 0.0, 1.0)


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 38) -> np.ndarray:
    """Two planes in alternating tiles: aligned anatomy runs straight across a tile edge.

    An MR and a CT share no intensity scale, so a colour overlay says nothing about
    them. Tiles do.
    """
    rows, columns = np.indices(a.shape)
    mask = (((rows // tile) + (columns // tile)) % 2).astype(bool)
    return (np.stack([np.where(mask, a, b)] * 3, axis=-1) * 255).astype(np.uint8)


def dice_per_label(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    scores = []
    for label in (int(value) for value in np.unique(reference) if value):
        left, right = reference == label, other == label
        total = left.sum() + right.sum()
        if total:
            scores.append(2 * (left & right).sum() / total)
    return np.array(scores)


def registration_panels(case: Path, transform_file: Path) -> None:
    """One patient's MR against their CT, before and after the IMPACT registration."""
    ct_image, mr_image = sitk.ReadImage(str(case / "ct.mha")), sitk.ReadImage(str(case / "mr.mha"))
    transform = elastix_bspline(transform_file)
    moved_image = sitk.Resample(mr_image, ct_image, transform, sitk.sitkLinear, 0.0, mr_image.GetPixelID())

    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    mr = sitk.GetArrayFromImage(mr_image).astype(np.float32)
    moved = sitk.GetArrayFromImage(moved_image).astype(np.float32)

    # The plane the registration moved most, among those actually showing the patient:
    # a middle slice can sit where little happens, and the largest change of all is
    # usually where the MR's field of view ends.
    body_area = (ct > -500).sum(axis=(1, 2))
    shows_patient = body_area > 0.6 * body_area.max()
    moved_by = np.abs(mr - moved).mean(axis=(1, 2)) * shows_patient
    index = int(np.argmax(moved_by))
    rows, columns = np.where(ct[index] > -500)
    body = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))

    reference = window(ct[index][body], *WINDOW)
    high = float(np.percentile(mr[index], 99))
    save(checkerboard(reference, window(mr[index][body], 0.0, high)), "mrct-before.png")
    save(checkerboard(reference, window(moved[index][body], 0.0, high)), "mrct-after.png")

    fixed_labels = sitk.ReadImage(str(case / "fixed_labels.nii.gz"))
    moving_labels = sitk.ReadImage(str(case / "moving_labels.nii.gz"))
    warped = sitk.Resample(
        moving_labels, ct_image, transform, sitk.sitkNearestNeighbor, 0, moving_labels.GetPixelID()
    )
    truth = sitk.GetArrayFromImage(fixed_labels)
    before = dice_per_label(truth, sitk.GetArrayFromImage(moving_labels))
    after = dice_per_label(truth, sitk.GetArrayFromImage(warped))
    print(f"  slice {index}: Dice {before.mean():.3f} -> {after.mean():.3f} over {len(after)} labels")


def fold_panels(cohort: Path, folded: Path) -> None:
    """A cohort folded four ways, from a TRANSFORM run over volumes that really correspond.

    The cohort must be REGISTERED, not merely resampled onto one grid. Folding cases that
    only overlap produces a volume that looks plausible and means nothing, which is the
    artefact `config_guide/transform` warns about.
    """
    reference = next(iter(sorted(cohort.iterdir())))
    save(to_rgb(middle_slice(reference / "CT.mha")), "fold-source.png")
    median = middle_slice(folded / "ct_median" / "CT.mha")
    save(to_rgb(median), "fold-median.png")

    spread = middle_slice(folded / "ct_std" / "CT.mha")
    scaled = spread / max(float(np.percentile(spread, 99)), 1e-6)
    heat = np.clip(np.stack([scaled, scaled * 0.45, 1 - scaled], axis=-1), 0, 1)
    save((heat * 255).astype(np.uint8), "fold-std.png")

    grey = to_rgb(median)
    vote = middle_slice(folded / "seg_vote" / "SEG.mha")
    mean = middle_slice(folded / "seg_mean" / "SEG.mha")
    save(overlay_labels(grey, vote.astype(np.int32)), "fold-vote.png")
    save(overlay_labels(grey, np.rint(mean).astype(np.int32)), "fold-labelmean.png")
    print(f"  Vote keeps {len(np.unique(vote))} labels; Mean holds {len(np.unique(mean))} distinct values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=ROOT / "examples")
    parser.add_argument("--mrct-case", type=Path, help="A folder with ct.mha, mr.mha and both label maps.")
    parser.add_argument("--mrct-transform", type=Path, help="The published Elastix parameter file for it.")
    parser.add_argument("--cohort", type=Path, help="A REGISTERED cohort, one folder per case.")
    parser.add_argument("--folded", type=Path, help="Where the Reduce runs wrote ct_median, ct_std, seg_vote, seg_mean.")
    args = parser.parse_args()

    print("segmentation")
    segmentation_panels(args.examples / "ImpactSeg")
    if args.mrct_case and args.mrct_transform:
        print("registration")
        registration_panels(args.mrct_case, args.mrct_transform)
    if args.cohort and args.folded:
        print("cohort folds")
        fold_panels(args.cohort, args.folded)


if __name__ == "__main__":
    main()
