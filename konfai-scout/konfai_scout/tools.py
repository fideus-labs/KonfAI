# SPDX-License-Identifier: Apache-2.0
"""The tools the model drives, their schemas, and the record each call leaves.

Every call is logged with its arguments, its duration and what it produced; a view is saved as evidence before it is
handed over; a note is checked against the epistemic rules before it enters the report."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .case import InspectionCase
from .config import ScoutConfig
from .evaluation import EvaluationError, RegionEvaluator
from .measure import displacement_stats, intensity_stats, label_overlap, label_stats
from .planes import PLANES, box_from_dict, box_to_dict, window_of, world_box_of
from .providers.base import ImageContent, ToolCall, ToolResult, ToolSpec
from .report import DEFECTS, KINDS, SEVERITIES, VERDICTS, Conclusion, Measurement, Observation
from .report import ToolCall as ToolCallRecord
from .views import WINDOW_PRESETS, Renderer, ViewError

_BOX = {
    "type": "object",
    "description": "A box in millimetres (LPS): {min: [x, y, z], max: [x, y, z]} or {center: [x, y, z], size: mm}.",
    "properties": {
        "min": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "max": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "size": {"type": "number"},
    },
    "additionalProperties": False,
}
_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


class Toolbox:
    def __init__(
        self, case: InspectionCase, config: ScoutConfig, renderer: Renderer, evaluator: RegionEvaluator
    ) -> None:
        self.case = case
        self.config = config
        self.renderer = renderer
        self.evaluator = evaluator
        self.calls: list[ToolCallRecord] = []
        self.measurements: list[Measurement] = []
        self.observations: list[Observation] = []
        self.conclusion: Conclusion | None = None
        self.images_sent = 0

    # ------------------------------------------------------------------ schemas

    def specs(self, exploration: bool, final: bool = False) -> list[ToolSpec]:
        modes = self.renderer.modes()
        mode = {"type": "string", "enum": modes, "description": "What to draw over the reference image."}
        window = _nullable(
            {
                "anyOf": [
                    {"type": "string", "enum": [*sorted(WINDOW_PRESETS), "auto"]},
                    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                ],
                "description": "Intensity window: a preset name or [low, high]; auto by default.",
            }
        )
        tools: list[ToolSpec] = []
        if exploration and not final:
            tools.append(
                ToolSpec(
                    "view",
                    "Render one slice. Choose the plane and either the world position the slice passes through, "
                    "its index, "
                    "or a zoom box (the slice through its centre, cropped to it). Returns the image and its record.",
                    _schema(
                        {
                            "plane": {"type": "string", "enum": list(PLANES)},
                            "mode": mode,
                            "position_mm": _nullable(
                                {**_XYZ, "description": "A world point the slice passes through."}
                            ),
                            "index": _nullable(
                                {"type": "integer", "description": "The slice index along the plane's axis."}
                            ),
                            "zoom": _nullable(
                                {
                                    **_BOX,
                                    "description": "Crop the slice to this box (the slice through its centre).",
                                }
                            ),
                            "window": window,
                            "label": _nullable({"type": "integer", "description": "Show this label only."}),
                        },
                        ["plane", "mode", "position_mm", "index", "zoom", "window", "label"],
                    ),
                )
            )
            tools.append(
                ToolSpec(
                    "montage",
                    "Render equally spaced slices of a plane as one image, for coverage.",
                    _schema(
                        {
                            "plane": {"type": "string", "enum": list(PLANES)},
                            "mode": _nullable(mode),
                            "count": _nullable({"type": "integer", "minimum": 1, "maximum": 16}),
                            "window": window,
                        },
                        ["plane", "mode", "count", "window"],
                    ),
                )
            )
        if not final:
            tools.append(
                ToolSpec(
                    "measure",
                    "Measure a box with KonfAI Eval: the case's groups cropped to the box and scored with KonfAI's own "
                    "criteria (Dice per label, MAE, PSNR, SSIM, TRE where a reference exists; a field's Jacobian "
                    "folding, "
                    "minimum and log-std, bending energy and displacement lengths without one). Returns the values and "
                    "the measurement id to cite.",
                    _schema(
                        {
                            "box_mm": _BOX,
                            "metrics": _nullable(
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Criterion names to restrict to; all of them by default.",
                                }
                            ),
                        },
                        ["box_mm", "metrics"],
                    ),
                )
            )
            tools.append(
                ToolSpec(
                    "region_stats",
                    "Descriptive statistics of a box, no reference needed: labels present with volume, pieces and "
                    "bounding "
                    "box, intensities, displacement lengths, landmarks inside.",
                    _schema({"box_mm": _BOX}, ["box_mm"]),
                )
            )
        tools.append(
            ToolSpec(
                "note",
                "Record one observation with its evidence. kind: measured_property (a number), suspected_error "
                "(seen, not "
                "measured against a reference), confirmed_error (established by a cited measurement).",
                _schema(
                    {
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "defect": {"type": "string", "enum": list(DEFECTS[self.case.task])},
                        "severity": {"type": "string", "enum": list(SEVERITIES)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "description": {"type": "string"},
                        "box_mm": _nullable(_BOX),
                        "plane": _nullable({"type": "string", "enum": list(PLANES)}),
                        "views": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "View ids that show it.",
                        },
                        "measurements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Measurement ids that support it.",
                        },
                    },
                    [
                        "kind",
                        "defect",
                        "severity",
                        "confidence",
                        "description",
                        "box_mm",
                        "plane",
                        "views",
                        "measurements",
                    ],
                ),
            )
        )
        tools.append(
            ToolSpec(
                "conclude",
                "End the inspection with a verdict and a summary of what was found and what could not be told.",
                _schema(
                    {
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "summary": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    ["verdict", "summary", "confidence"],
                ),
            )
        )
        return tools

    # ------------------------------------------------------------------ dispatch

    def call(self, call: ToolCall) -> ToolResult:
        started = time.time()
        record = ToolCallRecord(
            n=len(self.calls) + 1,
            tool=call.name,
            args=call.arguments,
            started_at=started,
            duration_s=0.0,
            ok=True,
            summary="",
        )
        self.calls.append(record)
        images: list[ImageContent] = []
        try:
            handler = getattr(self, f"_tool_{call.name}", None)
            if handler is None:
                raise ValueError(f"unknown tool {call.name!r}")
            payload, images = handler(record, **call.arguments)
            text = json.dumps(payload, default=_default)
            record.summary = payload.get("summary", "") if isinstance(payload, dict) else ""
        except TypeError as error:
            record.ok, record.error = False, f"bad arguments: {error}"
            text = record.error
        except (ValueError, ViewError, EvaluationError) as error:
            record.ok, record.error = False, str(error)
            text = str(error)
        except Exception as error:  # a bug in a tool must reach the model as an error, not end the inspection
            record.ok, record.error = False, f"{type(error).__name__}: {error}"
            text = record.error
        record.duration_s = round(time.time() - started, 3)
        if record.ok:
            self.images_sent += len(images)
        return ToolResult(call_id=call.id, text=text, images=images if record.ok else [], is_error=not record.ok)

    # ------------------------------------------------------------------ tools

    def _tool_view(
        self,
        record: ToolCallRecord,
        plane: str,
        mode: str | None = None,
        position_mm: object = None,
        index: int | None = None,
        zoom: object = None,
        window: object = None,
        label: int | None = None,
    ) -> tuple[dict[str, Any], list[ImageContent]]:
        box = box_from_dict(zoom) if zoom else None
        png, view = self.renderer.render(
            plane, mode, index=index, position_mm=position_mm, window=window, zoom=box, label=label
        )
        record.views.append(view.id)
        record.summary = f"{view.id} {view.plane} {view.mode}"
        payload = {"view": _view_payload(view), "summary": record.summary}
        return payload, [ImageContent(png, f"[{view.id}]")]

    def _tool_montage(
        self,
        record: ToolCallRecord,
        plane: str,
        mode: str | None = None,
        count: int | None = None,
        window: object = None,
    ) -> tuple[dict[str, Any], list[ImageContent]]:
        png, view = self.renderer.montage(plane, mode, count=count, window=window)
        record.views.append(view.id)
        record.summary = f"{view.id} {view.plane} montage ({len(view.extra.get('indices', []))} slices)"
        payload = {"view": _view_payload(view), "summary": record.summary}
        return payload, [ImageContent(png, f"[{view.id}]")]

    def _tool_measure(
        self, record: ToolCallRecord, box_mm: object, metrics: list[str] | None = None
    ) -> tuple[dict[str, Any], list[ImageContent]]:
        box = box_from_dict(box_mm)
        measurement = self.evaluator.measure(box, metrics, measurement_id=f"m{len(self.measurements) + 1:03d}")
        self.measurements.append(measurement)
        record.measurements.append(measurement.id)
        values = ", ".join(f"{k} = {_fmt(v)}" for k, v in measurement.values.items())
        record.summary = f"{measurement.id}: {values}" if values else f"{measurement.id}: no value"
        return {
            "measurement": {
                "id": measurement.id,
                "reference": measurement.reference,
                "box_measured": measurement.box_measured,
                "values": measurement.values,
                "labels": measurement.request.get("labels"),
            },
            "summary": record.summary,
        }, []

    def _tool_region_stats(self, record: ToolCallRecord, box_mm: object) -> tuple[dict[str, Any], list[ImageContent]]:
        box = box_from_dict(box_mm)
        grid = self.case.grid
        window = window_of(grid, box)
        if window is None:
            raise ValueError("the box holds no voxel of the case")
        lo = tuple(int(part.start) for part in window)
        hi = tuple(int(part.stop) for part in window)
        measured = world_box_of(grid, window)
        out: dict[str, Any] = {"box_measured": box_to_dict(measured), "voxels": int(np.prod(np.subtract(hi, lo)))}
        task = self.case.task
        if task == "segmentation":
            prediction = self.case.volume("prediction").region(lo, hi)[0]
            out["prediction_labels"] = label_stats(prediction, grid, self.case.labels, lo)
            image = self.case.volume("image").region(lo, hi)[0]
            out["image"] = intensity_stats(image)
            if prediction.any():
                out["image_inside_prediction"] = intensity_stats(image, prediction != 0)
            if "truth" in self.case.inputs:
                truth = self.case.volume("truth").region(lo, hi)[0]
                out["truth_labels"] = label_stats(truth, grid, self.case.labels, lo)
                out["overlap_voxels_prediction_vs_truth"] = label_overlap(prediction, truth)
        elif task == "registration":
            out["fixed"] = intensity_stats(self.case.volume("fixed").region(lo, hi))
            out["warped"] = intensity_stats(self.renderer.volume("warped").region(lo, hi))
            out["displacement"] = displacement_stats(self.renderer.volume("field").region(lo, hi))
            if "fixed_labels" in self.case.inputs:
                out["fixed_labels"] = label_stats(
                    self.case.volume("fixed_labels").region(lo, hi)[0], grid, self.case.labels, lo
                )
            try:
                out["warped_labels"] = label_stats(
                    self.renderer.volume("warped_labels").region(lo, hi)[0], grid, self.case.labels, lo
                )
            except ViewError:
                pass
            points = self.case.landmarks("fixed_landmarks")
            if points is not None:
                out["fixed_landmarks_inside"] = int(
                    np.sum(np.all((points >= measured.low_xyz) & (points <= measured.high_xyz), axis=1))
                )
        else:
            synthetic = self.case.volume("synthetic").region(lo, hi)
            out["synthetic"] = intensity_stats(synthetic)
            out["source"] = intensity_stats(self.case.volume("source").region(lo, hi))
            if "target" in self.case.inputs:
                target = self.case.volume("target").region(lo, hi)
                out["target"] = intensity_stats(target)
                out["difference_synthetic_minus_target"] = intensity_stats(
                    synthetic.astype(np.float64) - target.astype(np.float64)
                )
        record.summary = f"stats of {out['voxels']} voxels"
        return {"region": out, "summary": record.summary}, []

    def _tool_note(
        self,
        record: ToolCallRecord,
        kind: str,
        defect: str,
        severity: str,
        confidence: float,
        description: str,
        box_mm: object = None,
        plane: str | None = None,
        views: list[str] | None = None,
        measurements: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[ImageContent]]:
        if kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}")
        if defect not in DEFECTS[self.case.task]:
            raise ValueError(f"defect is one of {DEFECTS[self.case.task]}")
        if severity not in SEVERITIES:
            raise ValueError(f"severity is one of {SEVERITIES}")
        known_views = {v.id for v in self.renderer.views}
        known_measurements = {m.id for m in self.measurements}
        views = list(views or [])
        measurements = list(measurements or [])
        unknown = [v for v in views if v not in known_views] + [m for m in measurements if m not in known_measurements]
        if unknown:
            raise ValueError(f"unknown evidence id(s) {unknown}; cite ids returned by view, montage or measure")
        notes: list[str] = []
        if kind == "confirmed_error" and not measurements:
            kind = "suspected_error"
            notes.append("downgraded from confirmed_error: no measurement was cited")
        if kind == "confirmed_error":
            cited = [m for m in self.measurements if m.id in measurements]
            if not any(m.reference for m in cited) and not any(_defect_by_definition(m) for m in cited):
                kind = "suspected_error"
                notes.append(
                    "downgraded from confirmed_error: the cited measurements have no reference and show no "
                    "error by definition"
                )
        observation = Observation(
            id=f"o{len(self.observations) + 1:03d}",
            kind=kind,
            defect=defect,
            severity=severity,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            description=description,
            box_mm=box_to_dict(box_from_dict(box_mm)) if box_mm else None,
            plane=plane,
            views=views,
            measurements=measurements,
            notes=notes,
        )
        self.observations.append(observation)
        record.observation = observation.id
        record.summary = f"{observation.id} {observation.kind} {observation.defect} ({observation.severity})"
        return {
            "observation": {"id": observation.id, "kind": observation.kind, "notes": notes},
            "summary": record.summary,
        }, []

    def _tool_conclude(
        self, record: ToolCallRecord, verdict: str, summary: str, confidence: float
    ) -> tuple[dict[str, Any], list[ImageContent]]:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict is one of {VERDICTS}")
        reason = None
        if verdict == "defective" and not any(o.kind == "confirmed_error" for o in self.observations):
            reason = (
                "the verdict is 'defective' with no confirmed error recorded: it rests on suspected errors; "
                "read it as 'likely defective'"
            )
        self.conclusion = Conclusion(
            verdict=verdict,
            summary=summary,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            by="model",
            reason=reason,
        )
        record.summary = f"{verdict} ({confidence:.2f})"
        return {"conclusion": {"verdict": verdict, "note": reason}, "summary": record.summary}, []


def _defect_by_definition(measurement: Measurement) -> bool:
    """A no-reference measurement that is an error by itself: a folded field, non-finite values."""
    for key, value in measurement.values.items():
        if "JacobianFolding" in key and isinstance(value, (int, float)) and value > 0:
            return True
        if "JacobianMin" in key and isinstance(value, (int, float)) and value <= 0:
            return True
        if value is None:
            return True
    return False


def _view_payload(view: Any) -> dict[str, Any]:
    return {
        "id": view.id,
        "kind": view.kind,
        "plane": view.plane,
        "index": view.index,
        "position_mm": view.position_mm,
        "box_mm": view.box_mm,
        "mode": view.mode,
        "window": view.window,
        "size_px": view.size_px,
        **{k: v for k, v in view.extra.items() if k in ("legend", "scale", "labels", "indices", "landmarks_in_plane")},
    }


def _fmt(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
