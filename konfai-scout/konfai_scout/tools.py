# SPDX-License-Identifier: Apache-2.0
"""The tools the model drives, their schemas, and the record each call leaves.

Every call is logged with its arguments, its duration and what it produced; a view is saved as evidence before it is
handed over; a note is checked against the epistemic rules before it enters the report."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import numpy as np
from konfai.data.geometry import Grid, WorldBox

from .case import MASK_ROLES, InspectionCase
from .config import ScoutConfig
from .evaluation import DEFAULT_THRESHOLDS, EvaluationError, RegionEvaluator
from .measure import intensity_stats, label_stats
from .planes import PLANES, box_from_dict, box_intersection, box_to_dict, window_of
from .providers.base import ImageContent, ToolCall, ToolResult, ToolSpec
from .report import (
    DEFECTS,
    KINDS,
    REFERENCE_ERROR,
    SEVERITIES,
    VERDICTS,
    Conclusion,
    Measurement,
    Observation,
    json_default,
    short_key,
    value_text,
)
from .report import ToolCall as ToolCallRecord
from .survey import Lead
from .views import WARPED_LABEL_ROLES, WINDOW_PRESETS, Renderer, ViewError

_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_BOX = {
    "type": "object",
    "description": "mm, LPS: {min, max} or {center, size}.",
    "properties": {"min": _XYZ, "max": _XYZ, "center": _XYZ, "size": {"type": "number"}},
    "additionalProperties": False,
}
_IDS = {"type": "array", "items": {"type": "string"}}


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Every property required, an optional one nullable: the form strict tool calling accepts."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


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
        self.thresholds = {**DEFAULT_THRESHOLDS, **config.thresholds}
        self.calls: list[ToolCallRecord] = []
        self.measurements: list[Measurement] = []
        self.observations: list[Observation] = []
        self.conclusion: Conclusion | None = None
        self.images_sent = 0
        #: Where the survey says to look; in active mode `conclude` waits until each was looked at.
        self.leads: list[Lead] = []
        #: Whether the budget is spent: then a conclusion is taken as it comes.
        self.closing: Callable[[], bool] = lambda: False

    # ------------------------------------------------------------------ schemas

    def specs(self, exploration: bool, final: bool = False) -> list[ToolSpec]:
        plane = {"type": "string", "enum": list(PLANES)}
        mode = {"type": "string", "enum": self.renderer.modes()}
        window = _nullable(
            {
                "anyOf": [
                    {"type": "string", "enum": [*sorted(WINDOW_PRESETS), "auto"]},
                    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                ],
                "description": "A CT preset in HU, [low, high], or auto (1st to 99.9th percentile).",
            }
        )
        tools: list[ToolSpec] = []
        if exploration and not final:
            tools += [
                ToolSpec(
                    "view",
                    "One slice: a plane and a world position, an index, or a zoom box (the slice through its centre, "
                    "cropped to the box).",
                    _schema(
                        {
                            "plane": plane,
                            "mode": mode,
                            "position_mm": _nullable(_XYZ),
                            "index": _nullable({"type": "integer"}),
                            "zoom": _nullable(_BOX),
                            "window": window,
                            "label": _nullable({"type": "integer", "description": "Show this label only."}),
                        }
                    ),
                ),
                ToolSpec(
                    "montage",
                    "Equally spaced slices of a plane in one image.",
                    _schema(
                        {
                            "plane": plane,
                            "mode": _nullable(mode),
                            "count": _nullable({"type": "integer", "minimum": 1, "maximum": 16}),
                            "window": window,
                        }
                    ),
                ),
            ]
        if not final and self.config.measure:
            tools += [
                ToolSpec(
                    "measure",
                    "KonfAI Eval on a box: the criteria the case allows (Dice, MAE, PSNR, SSIM, TRE against its "
                    "references; a field's Jacobian and displacement statistics without). Returns the values and a "
                    "measurement id.",
                    _schema({"box_mm": _BOX, "metrics": _nullable({**_IDS, "description": "Criteria to keep."})}),
                ),
                ToolSpec(
                    "region_stats",
                    "Statistics of a box, no reference needed: labels (volume, pieces, bounds), intensities, "
                    "displacement lengths, landmarks. Returns them and a measurement id.",
                    _schema({"box_mm": _BOX}),
                ),
            ]
        tools += [
            ToolSpec(
                "note",
                "Record one observation and the evidence that shows it.",
                _schema(
                    {
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "defect": {"type": "string", "enum": list(DEFECTS[self.case.task])},
                        "severity": {"type": "string", "enum": list(SEVERITIES)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "description": {"type": "string", "description": f"{MAX_WORDS['note']} words at most."},
                        "box_mm": _nullable({"anyOf": [_BOX, {"type": "string", "description": "A lead's id."}]}),
                        "plane": _nullable(plane),
                        "views": _IDS,
                        "measurements": _IDS,
                    }
                ),
            ),
            ToolSpec(
                "conclude",
                "End the inspection with the verdict.",
                _schema(
                    {
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "summary": {"type": "string", "description": f"{MAX_WORDS['conclude']} words at most."},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    }
                ),
            ),
        ]
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
            text = json.dumps(payload, default=json_default)
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
        values = ", ".join(f"{k} = {value_text(v)}" for k, v in measurement.values.items())
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
        measured = grid.world_box(window)
        out: dict[str, Any] = {"box_measured": box_to_dict(measured), "voxels": int(np.prod(np.subtract(hi, lo)))}
        task = self.case.task

        def read(role: str) -> np.ndarray:
            return self.renderer.volume(role).region(lo, hi)[0]

        mask = None  # the intensities are those of the case's mask when it has one, as KonfAI Eval scores them
        if MASK_ROLES[task] in self.case.inputs:
            mask = read(MASK_ROLES[task]) != 0
            out["mask_voxels"] = int(mask.sum())
        if task == "segmentation":
            prediction, image = read("prediction"), read("image")
            out["prediction_labels"] = label_stats(prediction, grid, self.case.labels, lo)
            out["image"] = intensity_stats(image, mask)
            if prediction.any():
                out["image_inside_prediction"] = intensity_stats(image, prediction != 0)
            if "truth" in self.case.inputs:
                truth = read("truth")
                out["truth_labels"] = label_stats(truth, grid, self.case.labels, lo)
        elif task == "registration":
            out["fixed"] = intensity_stats(read("fixed"), mask)
            out["warped"] = intensity_stats(read("warped"), mask)
            fixed_labels = read("fixed_labels") if "fixed_labels" in self.case.inputs else None
            if fixed_labels is not None:
                out["fixed_labels"] = label_stats(fixed_labels, grid, self.case.labels, lo)
            if WARPED_LABEL_ROLES & self.case.inputs.keys():
                warped_labels = read("warped_labels")
                out["warped_labels"] = label_stats(warped_labels, grid, self.case.labels, lo)
            points = self.case.landmarks("fixed_landmarks")
            if points is not None:
                out["fixed_landmarks_inside"] = int(
                    np.sum(np.all((points >= measured.low_xyz) & (points <= measured.high_xyz), axis=1))
                )
        else:
            synthetic = read("synthetic")
            out["synthetic"] = intensity_stats(synthetic, mask)
            out["source"] = intensity_stats(read("source"), mask)
            if "target" in self.case.inputs:
                out["target"] = intensity_stats(read("target"), mask)
        # the statistics are a measurement of kind `scout`: a note cites them like one of KonfAI Eval's
        measurement = Measurement(
            id=f"m{len(self.measurements) + 1:03d}",
            kind="scout",
            reference=False,  # a description of each role; comparisons are KonfAI Eval's (`measure`)
            box_requested=box_to_dict(box),
            box_measured=out["box_measured"],
            values=_flatten({k: v for k, v in out.items() if k != "box_measured"}),
            request={"tool": "region_stats"},
        )
        self.measurements.append(measurement)
        record.measurements.append(measurement.id)
        record.summary = f"{measurement.id}: stats of {out['voxels']} voxels"
        return {"region": out, "measurement": {"id": measurement.id}, "summary": record.summary}, []

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
        confidence = _confidence(confidence)
        if len(description.split()) > MAX_WORDS["note"]:
            raise ValueError(
                f"say it in {MAX_WORDS['note']} words or fewer (this is {len(description.split())}): what, where, "
                "and which view or measurement shows it"
            )
        if defect == REFERENCE_ERROR and (kind != "suspected_error" or not self.case.truth_available):
            raise ValueError(
                "a reference_error is a suspected_error of a case given with its reference: the result follows the "
                "image where the reference does not, and no measurement against that reference can confirm it"
            )
        if kind == "measured_property" and defect != "other":
            raise ValueError(
                f"a measured_property is a number, not a defect: a {defect} is a suspected_error or a "
                "confirmed_error; a property takes defect 'other'"
            )
        known_views = {v.id: v for v in self.renderer.views}
        known_measurements = {m.id: m for m in self.measurements}
        views = list(views or [])
        measurements = list(measurements or [])
        unknown = [v for v in views if v not in known_views] + [m for m in measurements if m not in known_measurements]
        if unknown:
            raise ValueError(
                f"unknown evidence id(s) {unknown}; cite ids returned by view, montage, measure or region_stats"
            )
        if isinstance(box_mm, str):  # a lead's id: its box, rather than coordinates read off a view
            leads = {lead.id: lead for lead in self.leads}
            if box_mm not in leads:
                raise ValueError(f"no lead {box_mm!r}; leads: {', '.join(leads) or 'none'}")
            box_mm = leads[box_mm].box_mm
        box = box_from_dict(box_mm) if box_mm else None
        cited_views = [known_views[v] for v in views]
        cited = [known_measurements[m] for m in measurements]
        # the hard rules: what does not enter the report at all (the model is told why and may call again)
        if kind == "measured_property" and not cited:
            raise ValueError(
                "a measured_property cites the measurement that holds its figures (measure or region_stats first); "
                "something seen but not measured is a suspected_error"
            )
        if kind != "measured_property":
            if not cited_views:
                raise ValueError("an error cites at least one view that shows it")
            grid_box = self.renderer.grid.world_box()
            if box is None or (
                defect not in GLOBAL_DEFECTS
                and np.prod(box.high_xyz - box.low_xyz) > 0.25 * np.prod(grid_box.high_xyz - grid_box.low_xyz)
            ):
                raise ValueError(
                    "an error says where it is: a box (or a lead's id) over less than a quarter of the image; only a "
                    f"defect that can be everywhere ({', '.join(GLOBAL_DEFECTS)}) may span it"
                )
            if box is not None and not any(_shows(v, box, self.renderer.grid) for v in cited_views):
                raise ValueError(
                    f"none of the cited views shows the box {box_to_dict(box)}: cite a view that covers it, or "
                    "render one first (view with zoom)"
                )
        inside, outside = list(cited), []
        if box is not None:
            inside = [
                m
                for m in cited
                if not m.box_measured or box_intersection(box_from_dict(m.box_measured), box) is not None
            ]
            outside = [m.id for m in cited if m not in inside]
            if kind == "confirmed_error" and cited and not inside:
                raise ValueError(
                    f"measurement(s) {outside} were taken outside this box: a confirmation rests on a measurement "
                    "of the region it describes; measure it"
                )
        grid = self.renderer.grid
        bounds = grid.world_box()
        facts = [float(v) for v in (*grid.spacing_xyz, *grid.size_zyx, *bounds.low_xyz, *bounds.high_xyz)]
        for view in cited_views:  # a slice's position and extent are read off the view's header
            facts += [float(x) for x in (view.position_mm or [])]
            facts += [float(x) for part in view.box_mm.values() for x in part]
        unsourced = _unsourced_figures(description, cited, box, facts)
        if kind == "measured_property" and unsourced:
            raise ValueError(
                f"figure(s) {', '.join(unsourced)} appear in no cited measurement: write the values as the tools "
                "returned them, or record a suspected_error"
            )
        notes: list[str] = []
        if unsourced:
            notes.append("figure(s) read from the views, not measured: " + ", ".join(unsourced))
        if outside:
            notes.append("measurement(s) taken outside this box, cited as context: " + ", ".join(outside))
        observation_id = f"o{len(self.observations) + 1:03d}"
        support = _support(inside, self.thresholds)
        if kind != "measured_property" and box is not None and self.config.measure:
            # Scout's own check of the box: every criterion this case allows, recorded on the observation
            try:
                check = self.evaluator.measure(box, None, measurement_id=f"m{len(self.measurements) + 1:03d}")
            except Exception as error:  # the check is Scout's own: it never costs the model its note
                notes.append(f"Scout's check of the box: {type(error).__name__}: {error}")
            else:
                check.request["check_of"] = observation_id
                self.measurements.append(check)
                record.measurements.append(check.id)
                measurements.append(check.id)
                found = _support([check], self.thresholds)
                support += found
                shown = "; ".join(found[:3]) + (f" (+{len(found) - 3})" if len(found) > 3 else "")
                notes.append(f"Scout's check {check.id}: " + (shown or "no value past a threshold"))
        if kind == "confirmed_error" and not support:
            kind = "suspected_error"
            notes.append(
                "downgraded from confirmed_error: no measured value of the box is past its criterion's threshold"
            )
        observation = Observation(
            id=observation_id,
            kind=kind,
            defect=defect,
            severity=severity,
            confidence=confidence,
            description=description,
            box_mm=box_to_dict(box) if box is not None else None,
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
        confidence = _confidence(confidence)
        if len(summary.split()) > MAX_WORDS["conclude"]:
            raise ValueError(
                f"conclude in {MAX_WORDS['conclude']} words or fewer (this is {len(summary.split())}): the verdict and "
                "why, citing the observations by id; their details are already recorded"
            )
        serious = self.serious()
        if verdict == "defective" and not serious:
            raise ValueError(
                "'defective' rests on a major or critical error noted at confidence 0.5 or more, and none is: "
                "the result holds up ('acceptable') or could not be judged ('insufficient_evidence')"
            )
        if verdict != "defective" and serious:
            raise ValueError(
                f"{', '.join(serious)} records a major error at confidence 0.5 or more: the verdict is 'defective'"
            )
        unseen = [lead.id for lead in self.leads if not self.lead_views(lead)]
        if unseen and self.config.mode == "active" and not self.closing():
            raise ValueError(
                f"lead(s) {', '.join(unseen)} not looked at: render a slice through each box (a view zoomed on it), "
                "judge what it shows, then conclude"
            )
        reason = None
        if unseen:
            reason = f"lead(s) {', '.join(unseen)} not looked at"
        if verdict == "defective" and not any(o.kind == "confirmed_error" for o in self.observations):
            reason = "; ".join(
                [
                    *([reason] if reason else []),
                    "the verdict is 'defective' with no confirmed error recorded: it "
                    "rests on suspected errors; read it as 'likely defective'",
                ]
            )
        self.conclusion = Conclusion(
            verdict=verdict,
            summary=summary,
            confidence=confidence,
            by="model",
            reason=reason,
        )
        record.summary = f"{verdict} ({confidence:.2f})"
        return {"conclusion": {"verdict": verdict, "note": reason}, "summary": record.summary}, []

    def serious(self) -> list[str]:
        """The errors that make the result defective: major or critical, noted at confidence 0.5 or more; an error of
        the reference is not one of the result."""
        return [
            o.id
            for o in self.observations
            if o.kind != "measured_property"
            and o.defect != REFERENCE_ERROR
            and o.severity in ("major", "critical")
            and o.confidence >= 0.5
        ]

    def lead_views(self, lead: Lead) -> list[str]:
        """The slice views (not montage tiles) the model was sent that show some of a lead's box."""
        box = box_from_dict(lead.box_mm)
        return [v.id for v in self.renderer.views if v.kind == "slice" and v.sent and _shows(v, box, self.case.grid)]


def _confidence(value: float) -> float:
    """A confidence in [0, 1]; NaN, which JSON parsed by Python lets through and a clamp keeps, is refused."""
    if math.isnan(value):
        raise ValueError("confidence is a number between 0 and 1")
    return float(min(max(value, 0.0), 1.0))


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _past(key: str, value: object, thresholds: dict[str, dict[str, float]]) -> str | None:
    """Why a measured value is an error: past its criterion's threshold in the bad direction, or a non-finite
    statistic of the field. ``None`` otherwise: an empty value, or one without a threshold, proves nothing."""
    number = _number(value)
    if number is None:
        return None
    parts = key.split(":")  # OUTPUT:TARGET:Criterion[:label]; a region_stats key has no criterion
    if not math.isfinite(number):
        return f"{short_key(key)} is not finite" if parts[0] == "FIELD" else None
    rule = thresholds.get(parts[2], {}) if len(parts) > 2 else {}
    if "below" in rule and number < rule["below"]:
        return f"{short_key(key)} = {value_text(number)} < {rule['below']:g}"
    if "above" in rule and number > rule["above"]:
        return f"{short_key(key)} = {value_text(number)} > {rule['above']:g}"
    return None


def _support(measurements: list[Measurement], thresholds: dict[str, dict[str, float]]) -> list[str]:
    """The values of these measurements that are errors, with why."""
    return [why for m in measurements for key, value in m.values.items() if (why := _past(key, value, thresholds))]


def _shows(view: Any, box: WorldBox, grid: Grid) -> bool:
    """Whether a view drew some of the box: a slice (or its crop) is one voxel thick, a montage is its tiles."""
    if view.kind == "montage":
        full = [slice(0, int(n)) for n in grid.size_zyx]
        for index in view.extra.get("indices") or []:
            window = list(full)
            window[view.array_axis] = slice(int(index), int(index) + 1)
            if box_intersection(grid.world_box(tuple(window)), box) is not None:
                return True
        return False
    return box_intersection(box_from_dict(view.box_mm), box) is not None


#: The longest a description and a conclusion may be: the evidence is in the views and measurements they cite.
MAX_WORDS = {"note": 70, "conclude": 50}

#: Defects that may span the whole image; any other error is local.
GLOBAL_DEFECTS = ("blur", "intensity_bias", "over_smoothing")

_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_UNIT_FIGURE = re.compile(
    rf"(?<![\w.\-])({_NUMBER})(?:\s*(?:-|to)\s*({_NUMBER}))?\s*(mm³|mm3|mm²|mm2|mm|cm|%|HU|voxels?)(?![\w])"
)
_NOT_A_FIGURE = r"(?<!label )(?<!labels )(?<!slice )(?<!slices )(?<!index )(?<!tile )(?<!tiles )(?<!box )"
_METRIC_FIGURE = re.compile(
    rf"\b(?:Dice|DSC|MAE|PSNR|SSIM|TRE|HD95|Hausdorff|NCC|det\s*J|Jacobian\w*)(?::\d+)?[^\d-]{{0,16}}?\s{_NOT_A_FIGURE}"
    rf"({_NUMBER})",
    re.IGNORECASE,
)


def _figures(description: str) -> list[tuple[str, str, bool]]:
    """The figures a description states: as written, the number alone, and whether they are percentages."""
    description = description.translate({0x2212: "-", 0x2013: "-"})  # a minus sign, an en dash: hyphens
    out: list[tuple[str, str, bool]] = []
    for match in _UNIT_FIGURE.finditer(description):
        for text in (match.group(1), match.group(2)):
            if text:
                out.append((f"{text} {match.group(3)}", text, match.group(3) == "%"))
    for match in _METRIC_FIGURE.finditer(description):
        out.append((" ".join(match.group(0).split()), match.group(1), False))
    return out


def _unsourced_figures(
    description: str, cited: list[Measurement], box: WorldBox | None, facts: list[float] | None = None
) -> list[str]:
    """The figures of a description that no cited measurement holds, to the precision written; the coordinates of
    the boxes and the facts of the case card (spacing, size, bounds) count as sourced."""
    pool: list[float] = list(facts or [])
    for m in cited:
        pool += [n for v in m.values.values() if (n := _number(v)) is not None]
        for b in (m.box_measured, m.box_requested):
            if b:
                pool += [float(x) for part in b.values() for x in part]
    if box is not None:
        pool += [float(x) for x in (*box.low_xyz, *box.high_xyz)]
    unsourced = []
    for text, number, percent in _figures(description):
        value = float(number)
        decimals = -int(Decimal(number).as_tuple().exponent)  # the digits written, the exponent counted
        candidates = [(value, decimals)] + ([(value / 100.0, decimals + 2)] if percent else [])
        if not any(_same(c, p, d) for c, d in candidates for p in pool):
            unsourced.append(text)
    return unsourced


def _same(written: float, measured: float, decimals: int) -> bool:
    return round(measured, decimals) == round(written, decimals) or abs(written - measured) <= 0.01 * abs(measured)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested statistics as one level of dotted keys; lists (boxes) are left out."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        elif value is None or isinstance(value, (int, float, str)):
            out[name] = value
    return out


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
