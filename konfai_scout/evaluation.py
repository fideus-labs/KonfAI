# SPDX-License-Identifier: Apache-2.0
"""The bridge to KonfAI Eval: the deterministic evaluation stays there, Scout reads its reports and asks it for more.

A prior evaluation (``Metric_*.json`` of a KonfAI EVALUATION run) gives the whole-case values Scout starts from. A
region measurement stages the case's groups cropped to a box, in millimetres, and runs ``konfai.evaluate`` on the
crop with KonfAI's own criteria, named as in its configs, so the same code scores the whole case and the region.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .case import InspectionCase
from .config import ScoutConfig
from .geometry import WorldBox
from .report import Measurement
from .warp import Derived

DEFAULT_CRITERIA: dict[str, dict[str, dict[str, Any]]] = {
    "segmentation": {"Dice": {"labels": "auto"}},
    "registration_labels": {"Dice": {"labels": "auto"}},
    "registration_images": {"MAE": {}},
    "registration_field": {
        "JacobianFolding": {},
        "JacobianMin": {},
        "JacobianLogStd": {},
        "BendingEnergy": {},
        "Displacement": {"statistic": "mean"},
        "Displacement#2": {"statistic": "p95"},
        "Displacement#3": {"statistic": "max"},
    },
    "registration_landmarks": {"TRE": {}},
    "synthesis": {"MAE": {}, "PSNR": {}, "SSIM": {}},
}


class EvaluationError(RuntimeError):
    """A measurement KonfAI Eval could not make; the message is for the model."""


def konfai_available() -> bool:
    return importlib.util.find_spec("konfai") is not None


def konfai_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("konfai")
    except Exception:
        return None


def load_prior(evaluation: list[Path] | dict[str, Any] | None, case_name: str) -> dict[str, Any]:
    """The whole-case values of a KonfAI Eval report for this case (``{metric key: value}``); a dict is taken as
    given. A case absent from a report contributes nothing."""
    if evaluation is None:
        return {}
    if isinstance(evaluation, dict):
        return dict(evaluation)
    values: dict[str, Any] = {}
    candidates = {case_name, case_name.replace("/", "__")}
    for path in evaluation:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            values[f"_unreadable:{Path(path).name}"] = str(error)
            continue
        for metric, per_case in (data.get("case") or {}).items():
            if not isinstance(per_case, dict):
                continue
            for name in candidates:
                if name in per_case:
                    values[metric] = per_case[name]
                    break
    return values


class RegionEvaluator:
    """Measures a box of the case with KonfAI Eval, one evaluation workspace per measurement, kept beside the report."""

    def __init__(self, case: InspectionCase, config: ScoutConfig, work: Path, derived: Derived | None) -> None:
        self.case = case
        self.config = config
        self.work = Path(work) / "konfai"
        self.derived = derived
        self.count = 0

    def criteria(self, kind: str) -> dict[str, dict[str, Any]]:
        return dict(self.config.criteria.get(kind) or DEFAULT_CRITERIA[kind])

    def available(self) -> str | None:
        """Why a measurement is impossible, or None."""
        if not konfai_available():
            return "KonfAI is not installed in this environment: pip install 'konfai-scout[konfai]'"
        return None

    def measure(
        self, box: WorldBox, metrics: list[str] | None = None, measurement_id: str | None = None
    ) -> Measurement:
        started = time.time()
        self.count += 1
        identifier = measurement_id or f"m{self.count:03d}"
        reason = self.available()
        if reason:
            raise EvaluationError(reason)
        grid = self.case.grid
        index_box = box.intersection(grid.bounds)
        index_box = index_box.index_box(grid) if index_box is not None else None
        if index_box is None:
            raise EvaluationError("the box holds no voxel of the case")
        lo, hi = index_box
        measured = WorldBox.of_index_box(grid, lo, hi)
        root = self.work / identifier
        dataset = root / "Dataset" / _slug(self.case.name)
        dataset.mkdir(parents=True, exist_ok=True)
        tree, reference, labels_used = self._stage(dataset, lo, hi, metrics)
        if not tree:
            raise EvaluationError("nothing to measure in this box with the roles this case has")
        from konfai import evaluate

        result = evaluate(
            "region",
            [f"{root / 'Dataset'}:mha"],
            tree,
            cpu=1,
            quiet=True,
            overwrite=True,
            evaluations_dir=root / "Evaluations",
        )
        report = result.metrics.get("TRAIN") or {}
        values: dict[str, Any] = {}
        name = _slug(self.case.name)
        for key, per_case in (report.get("case") or {}).items():
            if isinstance(per_case, dict) and name in per_case:
                values[key] = per_case[name]
        if not self.config.keep_evaluations:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
        return Measurement(
            id=identifier,
            kind="konfai_eval",
            reference=reference,
            box_requested=box.to_dict(),
            box_measured=measured.to_dict(),
            values=values,
            request={"metrics": tree, "labels": labels_used, "dataset": str(root / "Dataset")},
            workspace=str(root.relative_to(self.work.parent)) if self.config.keep_evaluations else None,
            duration_s=round(time.time() - started, 2),
        )

    # ------------------------------------------------------------------ staging

    def _write(self, dataset: Path, group: str, role_volume, lo, hi, dtype: str) -> np.ndarray:
        import SimpleITK as sitk

        array = role_volume.region(lo, hi)
        grid = self.case.grid
        if array.ndim == 4:  # a vector field: components last for SimpleITK
            image = sitk.GetImageFromArray(
                np.ascontiguousarray(np.moveaxis(array.astype(np.float32), 0, -1)), isVector=True
            )
        else:
            image = sitk.GetImageFromArray(np.ascontiguousarray(array.astype(dtype)))
        image.SetSpacing(grid.spacing_xyz)
        image.SetDirection(grid.direction)
        image.SetOrigin([float(v) for v in grid.index_to_world([lo[2], lo[1], lo[0]])])
        sitk.WriteImage(image, str(dataset / f"{group}.mha"), False)
        return array

    def _volume(self, role: str):
        if self.case.task == "registration" and self.derived is not None:
            if role == "warped":
                return self.derived.warped()
            if role == "field":
                return self.derived.field()
            if role == "warped_labels":
                return self.derived.warped_labels()
        return self.case.volume(role) if role in self.case.inputs else None

    @staticmethod
    def _with_labels(criteria: dict[str, dict[str, Any]], labels: list[int]) -> dict[str, dict[str, Any]]:
        out = {}
        for name, arguments in criteria.items():
            arguments = dict(arguments)
            if arguments.get("labels") == "auto":
                arguments["labels"] = labels if labels else None
            out[name] = arguments
        return out

    @staticmethod
    def _selected(criteria: dict[str, dict[str, Any]], metrics: list[str] | None) -> dict[str, dict[str, Any]]:
        if not metrics:
            return criteria
        wanted = {m.lower() for m in metrics}
        return {
            name: args
            for name, args in criteria.items()
            if name.lower() in wanted or name.split("#")[0].lower() in wanted
        }

    def _stage(self, dataset: Path, lo, hi, metrics: list[str] | None) -> tuple[dict[str, Any], bool, list[int]]:
        task = self.case.task
        tree: dict[str, Any] = {}
        reference = False
        labels_used: list[int] = []
        mask_role = {"segmentation": "mask", "registration": "fixed_mask", "synthesis": "mask"}[task]
        mask_group = ""
        if mask_role in self.case.inputs:
            self._write(dataset, "MASK", self.case.volume(mask_role), lo, hi, "uint8")
            mask_group = ";MASK"
        if task == "segmentation":
            pred = self._write(dataset, "PRED", self.case.volume("prediction"), lo, hi, "int32")
            if "truth" in self.case.inputs:
                truth = self._write(dataset, "TRUTH", self.case.volume("truth"), lo, hi, "int32")
                labels_used = sorted({int(v) for v in np.union1d(np.unique(pred), np.unique(truth)) if v != 0})
                criteria = self._selected(self._with_labels(self.criteria("segmentation"), labels_used), metrics)
                if criteria:
                    tree["PRED"] = {f"TRUTH{mask_group}": criteria}
                    reference = True
        elif task == "registration":
            field = self._volume("field")
            field_criteria = self._selected(self.criteria("registration_field"), metrics)
            if field is not None and field_criteria:
                self._write(dataset, "FIELD", field, lo, hi, "float32")
                tree["FIELD"] = {"None": field_criteria}
            warped_labels = self._volume("warped_labels")
            if warped_labels is not None and "fixed_labels" in self.case.inputs:
                warped = self._write(dataset, "WARPED_LABELS", warped_labels, lo, hi, "int32")
                fixed = self._write(dataset, "FIXED_LABELS", self.case.volume("fixed_labels"), lo, hi, "int32")
                labels_used = sorted({int(v) for v in np.union1d(np.unique(warped), np.unique(fixed)) if v != 0})
                criteria = self._selected(self._with_labels(self.criteria("registration_labels"), labels_used), metrics)
                if criteria:
                    tree["WARPED_LABELS"] = {f"FIXED_LABELS{mask_group}": criteria}
                    reference = True
            image_criteria = self._selected(self.criteria("registration_images"), metrics)
            if image_criteria:
                self._write(dataset, "WARPED", self._volume("warped"), lo, hi, "float32")
                self._write(dataset, "FIXED", self.case.volume("fixed"), lo, hi, "float32")
                tree["WARPED"] = {f"FIXED{mask_group}": image_criteria}
            if (
                "fixed_landmarks" in self.case.inputs
                and "moving_landmarks" in self.case.inputs
                and self.derived is not None
            ):
                landmark_criteria = self._selected(self.criteria("registration_landmarks"), metrics)
                fixed_points = self.case.landmarks("fixed_landmarks")
                moving_points = self.case.landmarks("moving_landmarks")
                moved = self.derived.moved_landmarks()
                box = WorldBox.of_index_box(self.case.grid, lo, hi)
                inside = [
                    i
                    for i, p in enumerate(fixed_points)
                    if np.all(p >= box.min_xyz) and np.all(p <= box.max_xyz) and i < len(moving_points)
                ]
                if landmark_criteria and inside and moved is not None:
                    _write_points(dataset / "MOVED_LANDMARKS.fcsv", moved[inside])
                    _write_points(dataset / "MOVING_LANDMARKS.fcsv", moving_points[inside])
                    tree["MOVED_LANDMARKS"] = {"MOVING_LANDMARKS": landmark_criteria}
                    reference = True
        else:
            if "target" in self.case.inputs:
                criteria = self._selected(self.criteria("synthesis"), metrics)
                if criteria:
                    self._write(dataset, "SYN", self.case.volume("synthetic"), lo, hi, "float32")
                    self._write(dataset, "TARGET", self.case.volume("target"), lo, hi, "float32")
                    tree["SYN"] = {f"TARGET{mask_group}": criteria}
                    reference = True
        return tree, reference, labels_used


def _slug(name: str) -> str:
    return name.replace("/", "__")


def _write_points(path: Path, points: np.ndarray) -> None:
    rows = "\n".join(f"F-{i + 1},{x:.6f},{y:.6f},{z:.6f}" for i, (x, y, z) in enumerate(points))
    path.write_text(f"# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n# columns = id,x,y,z\n{rows}\n")
