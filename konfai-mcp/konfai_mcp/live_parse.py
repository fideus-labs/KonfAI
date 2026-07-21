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

"""Pure parsing of KonfAI runtime-log lines — metrics and tqdm progress.

The single source of truth for reading a live training/validation log line. It carries no dependency
beyond ``re``, so any co-located consumer (the MCP ``read_live_metrics`` tool, KonfAI Studio's live feed)
imports it instead of re-implementing the format."""

from __future__ import annotations

import re
from typing import Any

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_STAGE_RE = re.compile(r"\b(Training|Validation|Prediction)\s*:")
# The evaluator/uncertainty desc — "Metric TRAIN : out:tgt:Dice: 0.91 | out:tgt:HD95: 3.4" — carries no
# Loss(...) body, so it needs its own reader (its per-case metrics would otherwise never become curves).
_EVAL_SPLIT_RE = re.compile(r"\bMetric\s+([A-Za-z]+)\s*:")
# One "output:target:Name: value" pair; the key may itself contain ':' so the value is anchored at the end
# (a trailing ':' is tqdm's desc/bar separator on the last pair).
_EVAL_PAIR_RE = re.compile(r"(.+):\s*([-+0-9.eE]+|nan|inf|-inf):?\s*$", re.IGNORECASE)
# tqdm progress, with or without the |bar| glyphs: a default bar ("45%|██| 34/75 [00:12<00:14, 2.87it/s]")
# or konfai's bar-less training tail ("… :  2% 415/21294 [01:30<1:15:29, 4.61it/s]").
_PROGRESS_RE = re.compile(
    r"(\d+)%\s*(?:\|[^|]*\|)?\s*(\d+)\s*/\s*(\d+)\s*\[([^<\]]+)<([^,\]]+),\s*([0-9.]+)\s*(it/s|s/it)\]"
)


def parse_live_progress(text: str) -> dict[str, Any] | None:
    """The tqdm progress on a live log line — percent, step/total, rate, elapsed and remaining ETA — or
    None if the line carries no progress bar. Works for both training lines and the data-caching phase."""
    match = _PROGRESS_RE.search(ANSI_ESCAPE_RE.sub("", text))
    if match is None:
        return None
    return {
        "percent": float(match.group(1)),
        "step": int(match.group(2)),
        "total": int(match.group(3)),
        "elapsed": match.group(4).strip(),
        "remaining": match.group(5).strip(),
        "rate": float(match.group(6)),
        "rate_unit": match.group(7),
    }


def progress_label(text: str) -> str:
    """The phase a progress line belongs to — the tqdm desc, e.g. 'Caching Train', 'Caching Validation',
    'Training', 'Validation'. Taken as the text before the first colon so the per-iteration loss suffix is
    dropped; best-effort and length-capped."""
    head = ANSI_ESCAPE_RE.sub("", text).strip().split(":", 1)[0].strip()
    return head[:40]


def parse_host_stats(text: str) -> dict[str, float]:
    """Host resource readouts on a konfai log line: process memory (GB and %) and CPU (%). Present on both
    the training/validation metric lines and the data-caching progress lines (so RAM can be charted while
    the dataset loads)."""
    clean = ANSI_ESCAPE_RE.sub("", text)
    stats: dict[str, float] = {}
    gpu = re.search(r"Memory GPU \(([-+0-9.]+)G \(([-+0-9.]+) %\)\)", clean)
    if gpu is not None:
        stats["memory_gpu_gb"] = float(gpu.group(1))
        stats["memory_gpu_percent"] = float(gpu.group(2))
    memory = re.search(r"(?<!GPU )Memory \(([-+0-9.]+)G \(([-+0-9.]+) %\)\)", clean)
    if memory is not None:
        stats["memory_gb"] = float(memory.group(1))
        stats["memory_percent"] = float(memory.group(2))
    cpu = re.search(r"CPU \(([-+0-9.]+) %\)", clean)
    if cpu is not None:
        stats["cpu_percent"] = float(cpu.group(1))
    return stats


def extract_parenthesized_value(text: str, prefix: str) -> str | None:
    """The balanced ``(...)`` body immediately following ``prefix`` in ``text`` (e.g. after ``Loss (``)."""
    start = text.find(prefix)
    if start < 0:
        return None
    index = start + len(prefix)
    depth = 1
    chunk: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return "".join(chunk)
        chunk.append(char)
        index += 1
    return None


def parse_model_metrics(body: str | None) -> list[dict[str, Any]]:
    """Each ``Name(lr) : Metric(weight) : value …`` model group inside a ``Loss (...)`` body."""
    if not body:
        return []
    token_pattern = re.compile(r"([A-Za-z0-9_.-]+)\(([-+0-9.eE]+)\)\s*:\s*")
    value_pattern = re.compile(r"[-+0-9.eE]+|nan|inf|-inf", flags=re.IGNORECASE)
    matches = list(token_pattern.finditer(body))
    models: list[dict[str, Any]] = []
    current_model: dict[str, Any] | None = None
    for index, match in enumerate(matches):
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        trailing = body[match.end() : segment_end].strip()
        if trailing == "" or not value_pattern.fullmatch(trailing):
            current_model = {"name": match.group(1), "lr": float(match.group(2)), "metrics": {}}
            models.append(current_model)
            continue
        if current_model is None:
            current_model = {"name": match.group(1), "lr": 0.0, "metrics": {}}
            models.append(current_model)
        current_model["metrics"][match.group(1)] = {"weight": float(match.group(2)), "value": float(trailing)}
    return models


def flatten_live_metrics(models: list[dict[str, Any]]) -> dict[str, float]:
    return {
        f"{model['name']}:{metric_name}": metric_payload["value"]
        for model in models
        for metric_name, metric_payload in model["metrics"].items()
    }


def _parse_eval_metrics(clean: str) -> dict[str, Any] | None:
    """The evaluator/uncertainty line ``Metric <SPLIT> : out:tgt:Name: value | …`` → its split and a flat
    name→value map, or None if the line is not an evaluation metric line. The pairs sit between the
    ``Metric <SPLIT> :`` prefix and the tqdm bar; each is split on its final ``: value``."""
    split_match = _EVAL_SPLIT_RE.search(clean)
    if split_match is None:
        return None
    progress = _PROGRESS_RE.search(clean)
    body = clean[split_match.end() : progress.start() if progress else len(clean)].strip()
    flat: dict[str, float] = {}
    for chunk in body.split(" | "):
        pair = _EVAL_PAIR_RE.match(chunk.strip())
        if pair is not None:
            flat[pair.group(1).strip()] = float(pair.group(2))
    if not flat:
        return None
    return {"split": split_match.group(1).upper(), "flat_metrics": flat}


def _with_host_and_progress(result: dict[str, Any], clean: str) -> dict[str, Any]:
    """Fold the host resource readouts and the tqdm progress of ``clean`` into a metric entry."""
    result.update(parse_host_stats(clean))
    progress = parse_live_progress(clean)
    if progress is not None:
        result["progress"] = progress
    return result


def parse_live_metric_line(line: str) -> dict[str, Any] | None:
    """Parse one konfai runtime-log line into a structured live-metric entry: stage, per-model metrics
    (+ EMA), a flat name→value map, host memory/CPU/GPU, and the tqdm ``progress``. Returns None for a
    line that carries no metric values. Covers Training/Validation/Prediction (``Loss (…)`` bodies) and
    the evaluator/uncertainty ``Metric <SPLIT> : …`` line (which has no Loss body)."""
    clean = ANSI_ESCAPE_RE.sub("", line).strip()
    if not clean:
        return None
    stage_match = _STAGE_RE.search(clean)
    if stage_match is not None:
        metrics = parse_model_metrics(extract_parenthesized_value(clean, "Loss ("))
        metrics_ema = parse_model_metrics(extract_parenthesized_value(clean, "Loss EMA ("))
        if metrics or metrics_ema:
            return _with_host_and_progress(
                {
                    "stage": stage_match.group(1),
                    "models": metrics,
                    "ema_models": metrics_ema,
                    "flat_metrics": flatten_live_metrics(metrics),
                    "flat_metrics_ema": flatten_live_metrics(metrics_ema),
                    "raw": clean,
                },
                clean,
            )
    evaluation = _parse_eval_metrics(clean)
    if evaluation is not None:
        return _with_host_and_progress(
            {
                "stage": "Evaluation",
                "label": f"Metric {evaluation['split']}",
                "models": [],
                "ema_models": [],
                "flat_metrics": evaluation["flat_metrics"],
                "flat_metrics_ema": {},
                "raw": clean,
            },
            clean,
        )
    return None
