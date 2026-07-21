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

"""Metrics half of ``SessionService``: live-metric log parsing, evaluation metric
discovery, leaderboard/compare/curves payloads. Split out of ``server_experiments.py``
so the session service keeps only workflow/session logic; ``SessionService`` inherits
this mixin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .live_parse import parse_live_metric_line
from .server_jobs import Job
from .server_support import WorkspaceLayout, read_text_tail


class MetricsServiceMixin:
    """Metrics/leaderboard methods mixed into ``SessionService``."""

    # App evaluate/pipeline runs write their metric trees under these workspace subdirs; every trial's
    # inner run dir repeats the bundle's train_name, so a trial's IDENTITY is its top-level trial dir.
    # One constant feeds both the search roots and the identity rule — a root added to only one of them
    # would silently reintroduce the shared-inner-name ambiguity.
    _APP_TRIAL_SUBDIRS = ("AppEvaluations", "AppPipelines")

    def _metric_search_roots(self, layout: WorkspaceLayout | None = None) -> list[Path]:
        layout = layout or self.workspace_layout
        workspace = layout.workspace_dir()
        # App evaluate/pipeline runs write their KonfAI Evaluations tree (Metric_<SPLIT>.json) under these
        # dirs, so leaderboard/compare_runs rank tuned app trials alongside train-branch runs.
        return [
            layout.evaluations_dir(),
            workspace / "Evaluation",
            *(workspace / subdir for subdir in self._APP_TRIAL_SUBDIRS),
        ]

    def _metric_run_name(self, metrics_path: Path, layout: WorkspaceLayout | None = None) -> str:
        """The run identifier of a metric file: its run directory's name — except an app trial
        (``AppEvaluations/<label>-<uuid>/…/Metric_*.json``), identified by its parameter-suffixed trial
        directory, because every trial's inner run dir repeats the same bundle train_name."""
        layout = layout or self.workspace_layout
        workspace = layout.workspace_dir()
        for subdir in self._APP_TRIAL_SUBDIRS:
            try:
                return metrics_path.relative_to(workspace / subdir).parts[0]
            except ValueError:
                continue
        return metrics_path.parent.name

    def _resolve_session_layout(self, session: str | None) -> WorkspaceLayout:
        """Resolve a metrics-lookup layout: the current session, or another named one."""
        if session is None or session == self.workspace_layout.current_session:
            self.workspace_layout.ensure_session_workspace_exists()
            return self.workspace_layout
        layout = WorkspaceLayout(self.workspace_layout.root, session)
        if not layout.session_workspace_exists():
            raise ValueError(
                f"Unknown session '{session}'. Available sessions: {self.workspace_layout.available_sessions()}"
            )
        return layout

    def _parse_live_metric_line(self, line: str) -> dict[str, Any] | None:
        return parse_live_metric_line(line)  # module-level: the single source of truth (see top of file)

    def _parse_live_metrics_file(self, path: Path, max_lines: int | None = None) -> list[dict[str, Any]]:
        lines = read_text_tail(path, max_lines=max_lines or self.max_log_tail_lines).splitlines()
        return [entry for line in lines if (entry := parse_live_metric_line(line)) is not None]

    def read_live_metrics_payload(self, job: Job, max_entries: int) -> dict[str, Any]:
        runtime_log = self.job_runtime_log_path(job)
        checked_sources: list[str] = []
        parsed_entries: list[dict[str, Any]] = []
        metrics_source: str | None = None

        for path in [runtime_log, job.log_path]:
            if path is None or not path.exists():
                continue
            checked_sources.append(str(path))
            parsed_entries = self._parse_live_metrics_file(
                path, max_lines=max(self.max_log_tail_lines, max_entries * 10)
            )
            if parsed_entries:
                metrics_source = str(path)
                break

        latest_by_stage: dict[str, dict[str, Any]] = {}
        for entry in parsed_entries:
            latest_by_stage[entry["stage"].lower()] = entry
        return {
            **self.job_payload(job),
            "metrics_source": metrics_source,
            "metrics_sources_checked": checked_sources,
            "latest": parsed_entries[-1] if parsed_entries else None,
            "recent": parsed_entries[-max(max_entries, 1) :] if parsed_entries else [],
            "by_stage": latest_by_stage,
        }

    def _common_metric_prefix(self, values: list[str]) -> str | None:
        if not values:
            return None
        prefix = values[0]
        for value in values[1:]:
            while prefix and not value.startswith(prefix):
                prefix = prefix[:-1]
        return prefix.rstrip(":") or None

    def _metric_direction(self, metric_name: str, declared: str | None = None) -> tuple[Literal["min", "max"], str]:
        # A direction declared by the criterion itself (via the evaluation JSON 'directions' block,
        # sourced from each Criterion's `maximize` property) is authoritative -- no guessing.
        if declared in ("min", "max"):
            return declared, "declared"  # type: ignore[return-value]
        lowered = metric_name.lower()
        maximize_tokens = ("dice", "iou", "accuracy", "acc", "auc", "f1", "ssim", "psnr")
        minimize_tokens = ("mae", "mse", "rmse", "loss", "hausdorff", "hd")
        # 'loss' wins over any maximize token: a criterion named DiceLoss is still minimized.
        if "loss" in lowered:
            return "min", "heuristic:min"
        if any(token in lowered for token in maximize_tokens):
            return "max", "heuristic:max"
        if any(token in lowered for token in minimize_tokens):
            return "min", "heuristic:min"
        return "min", "default:min"

    @staticmethod
    def _declared_directions(payload: dict[str, Any]) -> dict[str, str]:
        directions = payload.get("directions", {})
        return directions if isinstance(directions, dict) else {}

    def _extract_metric_scoreboards(self, metrics_path: Path) -> dict[str, dict[str, Any]]:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        aggregates = payload.get("aggregates", {})
        if not isinstance(aggregates, dict):
            return {}
        declared = self._declared_directions(payload)

        result: dict[str, dict[str, Any]] = {}
        for metric_name, stats in aggregates.items():
            if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float)):
                direction, direction_source = self._metric_direction(metric_name, declared.get(metric_name))
                result[metric_name] = {
                    "metric": metric_name,
                    "value": float(stats["mean"]),
                    "direction": direction,
                    "direction_source": direction_source,
                    "stats": stats,
                }
            elif isinstance(stats, (int, float)):
                direction, direction_source = self._metric_direction(metric_name, declared.get(metric_name))
                result[metric_name] = {
                    "metric": metric_name,
                    "value": float(stats),
                    "direction": direction,
                    "direction_source": direction_source,
                    "stats": {"value": float(stats)},
                }
        return result

    def _evaluation_metric_files(self, split: str, layout: WorkspaceLayout | None = None) -> list[Path]:
        split_name = split.upper()
        candidates: list[Path] = []
        for root in self._metric_search_roots(layout):
            if not root.exists():
                continue
            candidates.extend(root.rglob(f"Metric_{split_name}.json"))
            candidates.extend(root.rglob(f"METRIC_{split_name}.json"))
        return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)

    def _available_metric_splits(self, layout: WorkspaceLayout | None = None) -> list[str]:
        splits: set[str] = set()
        for root in self._metric_search_roots(layout):
            if not root.exists():
                continue
            for pattern in ("Metric_*.json", "METRIC_*.json"):
                for path in root.rglob(pattern):
                    splits.add(path.stem.split("_", 1)[1])
        return sorted(splits)

    def run_metrics_payload(self, run_name: str, split: str = "TRAIN", session: str | None = None) -> dict[str, Any]:
        """Return the full per-case + aggregate metric JSON for ONE named run, not just the newest file."""
        layout = self._resolve_session_layout(session)
        split_name = split.upper()
        # Canonical identity first (the labels leaderboard hands out), THEN the inner-dir alias: in one
        # merged pass a newer trial's alias could shadow a plain run bearing that exact name, and a
        # leaderboard row would not round-trip through get_run_metrics.
        candidates = self._evaluation_metric_files(split_name, layout)
        metrics_path = next(
            (path for path in candidates if self._metric_run_name(path, layout) == run_name),
            None,
        ) or next((path for path in candidates if path.parent.name == run_name), None)
        if metrics_path is None:
            available_runs = sorted(
                {
                    self._metric_run_name(path, layout)
                    for root in self._metric_search_roots(layout)
                    if root.exists()
                    for pattern in ("Metric_*.json", "METRIC_*.json")
                    for path in root.rglob(pattern)
                }
            )
            raise ValueError(
                f"No metrics found for run '{run_name}' and split '{split_name}' in session "
                f"'{layout.current_session}'. Available runs: {available_runs or 'none'}. "
                f"Available splits: {self._available_metric_splits(layout) or 'none'}."
            )
        return {
            "session": layout.current_session,
            "run_name": run_name,
            "split": split_name,
            "path": str(metrics_path),
            "updated_at": self._isoformat(metrics_path.stat().st_mtime),
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "summary": self._extract_metric_scoreboards(metrics_path),
            "next_actions": ["leaderboard", "summarize_session"],
        }

    def leaderboard_payload(
        self,
        metric: str | None = None,
        split: str = "TRAIN",
        limit: int = 20,
        direction: Literal["min", "max"] | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        layout = self._resolve_session_layout(session)
        metric_files = self._evaluation_metric_files(split, layout)
        if not metric_files:
            available_splits = self._available_metric_splits(layout)
            raise ValueError(
                f"No evaluation metrics found for session '{layout.current_session}' and split '{split.upper()}'. "
                f"Available splits: {available_splits or 'none'}."
            )

        by_metric: dict[str, list[dict[str, Any]]] = {}
        available_metrics: set[str] = set()
        warnings: list[str] = []

        for metrics_path in metric_files:
            run_name = self._metric_run_name(metrics_path, layout)
            extracted = self._extract_metric_scoreboards(metrics_path)
            available_metrics.update(extracted)
            updated_at = self._isoformat(metrics_path.stat().st_mtime)
            for metric_name, payload in extracted.items():
                by_metric.setdefault(metric_name, []).append(
                    {
                        "run_name": run_name,
                        "metric": metric_name,
                        "value": payload["value"],
                        "direction": payload["direction"],
                        "direction_source": payload["direction_source"],
                        "stats": payload["stats"],
                        "metrics_path": str(metrics_path),
                        "updated_at": updated_at,
                    }
                )
                if payload["direction_source"] == "default:min":
                    warnings.append(
                        f"Metric '{metric_name}' uses the default minimize direction because its name is unknown."
                    )

        selected_metric = metric
        if selected_metric is not None and metric is not None:
            lowered_metric = metric.lower()
            matching_metrics = [
                name
                for name in available_metrics
                if name.lower() == lowered_metric or name.lower().endswith(lowered_metric)
            ]
            if len(matching_metrics) == 1:
                selected_metric = matching_metrics[0]
            elif metric not in by_metric:
                raise ValueError(f"Metric '{metric}' not found. Available metrics: {sorted(available_metrics)}")
        elif len(available_metrics) == 1:
            selected_metric = next(iter(available_metrics))

        for metric_name, rows in by_metric.items():
            row_direction = rows[0]["direction"]
            # An explicit override applies to the selected metric (or to all when none is selected).
            if direction is not None and (selected_metric is None or metric_name == selected_metric):
                row_direction = direction
            rows.sort(key=lambda row: row["value"], reverse=row_direction == "max")
            by_metric[metric_name] = rows[: max(limit, 1)]

        result: dict[str, Any] = {
            "session": layout.current_session,
            "split": split.upper(),
            "available_metrics": sorted(available_metrics),
            "available_splits": self._available_metric_splits(layout),
            "warnings": sorted(set(warnings)),
            "next_actions": ["summarize_session", "get_run_metrics", "run_train", "run_prediction", "run_evaluation"],
        }
        if direction is not None:
            result["direction_override"] = direction
        if selected_metric is not None:
            result["selected_metric"] = selected_metric
            result["leaderboard"] = by_metric[selected_metric]
            result["best"] = by_metric[selected_metric][0] if by_metric[selected_metric] else None
        else:
            result["leaderboard"] = []
            result["common_metric_prefix"] = self._common_metric_prefix(sorted(available_metrics))
            result["leaderboards"] = by_metric
        return result

    def read_metrics_payload(self) -> dict[str, Any]:
        metrics_path = self.discover_metrics_path()
        if metrics_path is None:
            return {
                "session": self.session_name(),
                "path": None,
                "metrics": None,
                "summary": None,
            }
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = self._extract_metric_scoreboards(metrics_path)
        return {
            "session": self.session_name(),
            "path": str(metrics_path),
            "metrics": payload,
            "summary": summary,
        }

    def discover_metrics_path(self) -> Path | None:
        candidates: list[Path] = []
        for root in self._metric_search_roots():
            if not root.exists():
                continue
            candidates.extend(root.rglob("Metric_*.json"))
            candidates.extend(root.rglob("METRIC_*.json"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def training_curves_payload(
        self,
        run_name: str,
        tags: list[str] | None = None,
        max_points: int = 200,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Parse the TensorBoard event files of one run into downsampled scalar series."""
        layout = self._resolve_session_layout(session)
        stats_dir = layout.workspace_dir() / "Statistics" / run_name
        if not stats_dir.exists():
            available = (
                sorted(path.name for path in (layout.workspace_dir() / "Statistics").iterdir() if path.is_dir())
                if (layout.workspace_dir() / "Statistics").exists()
                else []
            )
            raise ValueError(f"No Statistics/{run_name} directory. Available runs: {available or 'none'}.")
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError as exc:
            raise ValueError(
                "Reading training curves requires the 'tensorboard' package: pip install konfai[tensorboard]."
            ) from exc

        event_dirs = sorted({path.parent for path in stats_dir.rglob("events.out.tfevents.*")})
        if not event_dirs:
            raise ValueError(f"No TensorBoard event files found under {stats_dir}.")
        curves: dict[str, list[dict[str, float]]] = {}
        for event_dir in event_dirs:
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            prefix = event_dir.relative_to(stats_dir).as_posix()
            for tag in accumulator.Tags().get("scalars", []):
                full_tag = tag if prefix == "." else f"{prefix}/{tag}"
                if tags and not any(wanted in full_tag for wanted in tags):
                    continue
                events = accumulator.Scalars(tag)
                stride = max(1, len(events) // max(max_points, 1))
                sampled = events[::stride]
                if events and sampled[-1].step != events[-1].step:
                    sampled = [*sampled, events[-1]]
                curves[full_tag] = [{"step": int(event.step), "value": float(event.value)} for event in sampled]
        return {
            "session": layout.current_session,
            "run_name": run_name,
            "tags": sorted(curves),
            "curves": curves,
            "next_actions": ["compare_runs", "leaderboard", "summarize_session"],
        }

    def compare_runs_payload(
        self,
        run_a: str,
        run_b: str,
        split: str = "TRAIN",
        metric: str | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Per-case aligned comparison of two runs' evaluation metrics, direction-aware."""
        payload_a = self.run_metrics_payload(run_a, split=split, session=session)
        payload_b = self.run_metrics_payload(run_b, split=split, session=session)
        cases_a = payload_a["metrics"].get("case", {})
        cases_b = payload_b["metrics"].get("case", {})
        common_metrics = sorted(set(cases_a) & set(cases_b))
        if metric is not None:
            common_metrics = [name for name in common_metrics if name.lower().endswith(metric.lower())]
            if not common_metrics:
                raise ValueError(
                    f"Metric '{metric}' not found in both runs. Common metrics: {sorted(set(cases_a) & set(cases_b))}"
                )
        declared_a = self._declared_directions(payload_a["metrics"])
        declared_b = self._declared_directions(payload_b["metrics"])
        comparison: dict[str, Any] = {}
        warnings: list[str] = []
        for name in common_metrics:
            direction, direction_source = self._metric_direction(name, declared_a.get(name) or declared_b.get(name))
            if direction_source == "default:min":
                warnings.append(
                    f"No declared or recognised direction for '{name}'; assumed minimize. If it is a "
                    "higher-is-better metric, the winner is inverted -- re-evaluate so the metric declares "
                    "its direction, or read per-case deltas directly."
                )
            common_cases = sorted(set(cases_a[name]) & set(cases_b[name]))
            deltas = {case: float(cases_b[name][case]) - float(cases_a[name][case]) for case in common_cases}
            better_b = sum(1 for delta in deltas.values() if (delta > 0) == (direction == "max") and delta != 0)
            better_a = sum(1 for delta in deltas.values() if (delta < 0) == (direction == "max") and delta != 0)
            mean_a = (
                sum(float(cases_a[name][case]) for case in common_cases) / len(common_cases) if common_cases else None
            )
            mean_b = (
                sum(float(cases_b[name][case]) for case in common_cases) / len(common_cases) if common_cases else None
            )
            comparison[name] = {
                "direction": direction,
                "direction_source": direction_source,
                "cases": len(common_cases),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_delta_b_minus_a": (mean_b - mean_a) if common_cases else None,
                "cases_better_a": better_a,
                "cases_better_b": better_b,
                "winner": (run_b if better_b > better_a else run_a if better_a > better_b else "tie"),
                "per_case_delta_b_minus_a": deltas,
            }
        return {
            "session": payload_a["session"],
            "split": split.upper(),
            "run_a": run_a,
            "run_b": run_b,
            "metrics": comparison,
            "warnings": warnings,
            "next_actions": ["get_run_metrics", "leaderboard", "summarize_session"],
        }
