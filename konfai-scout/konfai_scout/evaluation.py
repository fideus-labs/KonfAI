# SPDX-License-Identifier: Apache-2.0
"""The bridge to KonfAI Eval: the deterministic evaluation stays there, Scout reads its reports and asks it for more.

A prior evaluation (``Metric_*.json`` of a KonfAI EVALUATION run) gives the whole-case values Scout starts from. A
region measurement reads the case's groups cropped to a box (KonfAI region reads), writes them as a one-case dataset
with their geometry and runs ``konfai.evaluate`` on it with KonfAI's own criteria, named as in its configs, so the
same code scores the whole case and the region.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from konfai.data.geometry import Grid, WorldBox
from konfai.utils.dataset import Attribute, Dataset, write_landmarks
from konfai.utils.errors import KonfAIError

from .case import MASK_ROLES, InspectionCase, Volume
from .config import ScoutConfig
from .derived import Derived
from .planes import box_to_dict, same_grid, window_of
from .report import Measurement

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
#: When a measured value is an error, by KonfAI's result name (the third part of `PRED:TRUTH:Dice:1`). A criterion
#: whose scale is the data's (MAE, PSNR, SSIM, displacement lengths) has none: it confirms nothing until
#: ``ScoutConfig.thresholds`` sets one for the data at hand.
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "Dice": {"below": 0.8},
    "SurfaceDice": {"below": 0.8},
    "HD95": {"above": 5.0},
    "ASSD": {"above": 2.0},
    "TRE": {"above": 3.0},
    "JacobianFolding": {"above": 0.0},
    "JacobianMin": {"below": 0.0},
}


class EvaluationError(RuntimeError):
    """A measurement KonfAI Eval could not make; the message is for the model."""


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


def _attribute_of(grid: Grid) -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray(grid.origin_xyz, dtype=np.float64)
    attribute["Spacing"] = np.asarray(grid.spacing_xyz, dtype=np.float64)
    attribute["Direction"] = np.asarray(grid.direction_xyz, dtype=np.float64).ravel()
    return attribute


class RegionEvaluator:
    """Measures a box of the case with KonfAI Eval, one evaluation workspace per measurement, kept beside the report."""

    def __init__(self, case: InspectionCase, config: ScoutConfig, work: Path, derived: Derived | None) -> None:
        self.case = case
        self.config = config
        self.work = Path(work) / "konfai"
        self.derived = derived
        self.count = 0
        self._dropped: list[str] = []

    def criteria(self, kind: str) -> dict[str, dict[str, Any]]:
        return dict(self.config.criteria.get(kind) or DEFAULT_CRITERIA[kind])

    def measure(
        self, box: WorldBox, metrics: list[str] | None = None, measurement_id: str | None = None
    ) -> Measurement:
        started = time.time()
        self.count += 1
        identifier = measurement_id or f"m{self.count:03d}"
        grid = self.case.grid
        window = window_of(grid, box)
        if window is None:
            raise EvaluationError("the box holds no voxel of the case")
        measured = grid.world_box(window)
        root = self.work / identifier
        dataset = Dataset(root / "Dataset", "mha")
        # SSIM needs 7 voxels along every axis: on a thinner box it is left out rather than refusing the whole call
        self._dropped = (
            ["SSIM: the box is thinner than its 7-voxel window"]
            if min(int(part.stop - part.start) for part in window) < 7
            else []
        )
        tree, reference, labels_used = self._stage(dataset, window, metrics)
        if not tree:
            raise EvaluationError(
                "nothing to measure in this box with the roles this case has"
                + (f"; metrics are criterion names, for this case: {', '.join(self.available())}" if metrics else "")
            )
        from konfai import evaluate

        try:
            result = evaluate(
                "region",
                [f"{root / 'Dataset'}:mha"],
                tree,
                cpu=1,
                quiet=True,
                overwrite=True,
                evaluations_dir=root / "Evaluations",
            )
        except KonfAIError as error:
            raise EvaluationError(f"KonfAI Eval refused the measurement: {error}") from error
        report = result.metrics.get("TRAIN") or {}
        values: dict[str, Any] = {}
        for key, per_case in (report.get("case") or {}).items():
            if isinstance(per_case, dict) and self.case.slug in per_case:
                values[key] = per_case[self.case.slug]
        if not self.config.keep_evaluations:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
        return Measurement(
            id=identifier,
            kind="konfai_eval",
            reference=reference,
            box_requested=box_to_dict(box),
            box_measured=box_to_dict(measured),
            values=values,
            request={
                "metrics": tree,
                "labels": labels_used,
                "dataset": str(root / "Dataset"),
                **({"dropped": self._dropped} if self._dropped else {}),
            },
            workspace=str(root.relative_to(self.work.parent)) if self.config.keep_evaluations else None,
            duration_s=round(time.time() - started, 2),
        )

    # ------------------------------------------------------------------ staging

    def _write(
        self, dataset: Dataset, group: str, volume: Volume, window: tuple[slice, slice, slice], dtype: str
    ) -> np.ndarray:
        """The crop of a volume as an entry of the measurement's dataset, with the crop's own geometry."""
        if not same_grid(volume.grid, self.case.grid):
            raise EvaluationError(
                f"{volume.role} is on another grid than {self.case.image_role}; Scout measures the roles of one case "
                "on one grid"
            )
        lo = tuple(int(part.start) for part in window)
        hi = tuple(int(part.stop) for part in window)
        array = volume.region(lo, hi)  # type: ignore[arg-type]
        data = array.astype(np.float32) if array.ndim == 4 else array.astype(dtype)
        dataset.write(group, self.case.slug, np.ascontiguousarray(data), _attribute_of(volume.grid.sub_grid(window)))
        return array

    def _volume(self, role: str) -> Volume | None:
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
        """The criteria asked for, by KonfAI's criterion name or by a result key a model read (`WARPED:FIXED:MAE`,
        `FIELD:None:DisplacementP95`, `PRED:TRUTH:Dice:1`)."""
        if not metrics:
            return criteria
        names = {name.lower(): name for name in criteria}
        selected: set[str] = set()
        for wanted in {_criterion_name(m) for m in metrics}:
            if wanted in names:  # the exact criterion (`Displacement#2`)
                selected.add(names[wanted])
            elif any(name.split("#")[0].lower() == wanted for name in criteria):  # its family (`Displacement`)
                selected |= {name for name in criteria if name.split("#")[0].lower() == wanted}
            else:  # a result key's last part (`DisplacementP95`)
                selected |= {name for name in criteria if wanted.startswith(name.split("#")[0].lower())}
        return {name: args for name, args in criteria.items() if name in selected}

    def _criteria_for(self, kind: str, metrics: list[str] | None) -> dict[str, dict[str, Any]]:
        selected = self._selected(self.criteria(kind), metrics)
        if self._dropped:
            selected = {name: args for name, args in selected.items() if name.split("#")[0] != "SSIM"}
        return selected

    def available(self) -> list[str]:
        """The criterion names this case can be measured with, as `measure` takes them."""
        kinds = [k for k in DEFAULT_CRITERIA if k == self.case.task or k.startswith(self.case.task + "_")]
        return sorted({name.split("#")[0] for kind in kinds for name in self.criteria(kind)})

    def _stage(
        self, dataset: Dataset, window: tuple[slice, slice, slice], metrics: list[str] | None
    ) -> tuple[dict[str, Any], bool, list[int]]:
        task = self.case.task
        tree: dict[str, Any] = {}
        reference = False
        labels_used: list[int] = []
        mask_role = MASK_ROLES[task]
        mask_group = ""
        if mask_role in self.case.inputs:
            self._write(dataset, "MASK", self.case.volume(mask_role), window, "uint8")
            mask_group = ";MASK"
        if task == "segmentation":
            pred = self._write(dataset, "PRED", self.case.volume("prediction"), window, "int32")
            if "truth" in self.case.inputs:
                truth = self._write(dataset, "TRUTH", self.case.volume("truth"), window, "int32")
                labels_used = sorted({int(v) for v in np.union1d(np.unique(pred), np.unique(truth)) if v != 0})
                criteria = self._selected(self._with_labels(self.criteria("segmentation"), labels_used), metrics)
                if criteria:
                    tree["PRED"] = {f"TRUTH{mask_group}": criteria}
                    reference = True
        elif task == "registration":
            field = self._volume("field")
            field_criteria = self._selected(self.criteria("registration_field"), metrics)
            if field is not None and field_criteria:
                self._write(dataset, "FIELD", field, window, "float32")
                tree["FIELD"] = {"None": field_criteria}
            warped_labels = self._volume("warped_labels")
            if warped_labels is not None and "fixed_labels" in self.case.inputs:
                warped = self._write(dataset, "WARPED_LABELS", warped_labels, window, "int32")
                fixed = self._write(dataset, "FIXED_LABELS", self.case.volume("fixed_labels"), window, "int32")
                labels_used = sorted({int(v) for v in np.union1d(np.unique(warped), np.unique(fixed)) if v != 0})
                criteria = self._selected(self._with_labels(self.criteria("registration_labels"), labels_used), metrics)
                if criteria:
                    tree["WARPED_LABELS"] = {f"FIXED_LABELS{mask_group}": criteria}
                    reference = True
            image_criteria = self._selected(self.criteria("registration_images"), metrics)
            warped_image = self._volume("warped")
            if image_criteria and warped_image is not None:
                self._write(dataset, "WARPED", warped_image, window, "float32")
                self._write(dataset, "FIXED", self.case.volume("fixed"), window, "float32")
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
                measured = self.case.grid.world_box(window)
                inside = [
                    i
                    for i, p in enumerate(fixed_points if fixed_points is not None else [])
                    if np.all(p >= measured.low_xyz)
                    and np.all(p <= measured.high_xyz)
                    and moving_points is not None
                    and i < len(moving_points)
                ]
                if landmark_criteria and inside and moved is not None and moving_points is not None:
                    case_dir = Path(str(dataset.filename)) / self.case.slug
                    case_dir.mkdir(parents=True, exist_ok=True)
                    write_landmarks(moved[inside], case_dir / "MOVED_LANDMARKS.fcsv")
                    write_landmarks(moving_points[inside], case_dir / "MOVING_LANDMARKS.fcsv")
                    tree["MOVED_LANDMARKS"] = {"MOVING_LANDMARKS": landmark_criteria}
                    reference = True
        else:
            if "target" in self.case.inputs:
                criteria = self._criteria_for("synthesis", metrics)
                if criteria:
                    self._write(dataset, "SYN", self.case.volume("synthetic"), window, "float32")
                    self._write(dataset, "TARGET", self.case.volume("target"), window, "float32")
                    tree["SYN"] = {f"TARGET{mask_group}": criteria}
                    reference = True
        return tree, reference, labels_used


def _criterion_name(metric: str) -> str:
    """A criterion as `measure` takes it, from its name or from a result key (`WARPED:FIXED:MAE`, `PRED:TRUTH:Dice:1`)."""
    parts = [part for part in metric.split(":") if part and not part.isdigit()]
    return (parts[-1] if parts else metric).lower()
