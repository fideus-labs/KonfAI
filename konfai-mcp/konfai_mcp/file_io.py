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

"""Bounded text-file readers (full/range/tail/sidecar preview) and compound-suffix
helpers shared by the MCP tools. Split out of ``server_support.py``."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    """Return a full text file, or an empty string when it does not exist."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_text_range(path: Path, max_chars: int = 20000, offset: int = 0) -> dict[str, Any]:
    """Read a bounded character range of a text file for MCP read-back tools.

    Streams instead of loading the whole file: memory stays proportional to offset+max_chars,
    so a multi-GB log cannot exhaust the server. ``total_bytes`` is the on-disk size.
    """
    if not path.is_file():
        raise ValueError(f"Not a readable file: {path}")
    offset = max(offset, 0)
    max_chars = max(max_chars, 1)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if offset:
            handle.read(offset)
        content = handle.read(max_chars)
        truncated = bool(handle.read(1))
    return {
        "path": str(path),
        "content": content,
        "offset": offset,
        "returned_chars": len(content),
        "total_bytes": path.stat().st_size,
        "truncated": truncated,
    }


_BINARY_SNIFF_BYTES = 4096
_DELIMITED_SUFFIXES = {".csv": ",", ".tsv": "\t"}
# Linear ITK transforms carry at most 12 parameters (3x3 matrix + offset); anything larger is a
# grid (BSpline coefficients / displacement field) and is summarised instead of dumped.
_LINEAR_TRANSFORM_MAX_PARAMS = 16
# Whole 3-vectors per read of a dense grid (~24 MiB of float64): a displacement field can outweigh
# the server's RAM, so its summary never materializes the dataset.
_DENSE_CHUNK_ELEMENTS = 3 * 2**20


def _dense_displacement_summary(dataset: Any) -> dict[str, Any]:
    """Per-axis mean/std and magnitude stats of a dense 1-D parameter grid, in bounded slices."""
    import numpy as np

    total = int(dataset.size)
    count = total // 3
    sum_xyz = np.zeros(3)
    sumsq_xyz = np.zeros(3)
    sum_magnitude = 0.0
    max_magnitude = 0.0
    for start in range(0, count * 3, _DENSE_CHUNK_ELEMENTS):
        block = np.asarray(dataset[start : min(start + _DENSE_CHUNK_ELEMENTS, count * 3)], dtype=float)
        vectors = block.reshape(-1, 3)
        magnitudes = np.linalg.norm(vectors, axis=1)
        sum_xyz += vectors.sum(axis=0)
        sumsq_xyz += np.square(vectors).sum(axis=0)
        sum_magnitude += float(magnitudes.sum())
        max_magnitude = max(max_magnitude, float(magnitudes.max()))
    mean_xyz = sum_xyz / count
    std_xyz = np.sqrt(np.maximum(sumsq_xyz / count - np.square(mean_xyz), 0.0))
    return {
        "vectors": count,
        "mean_xyz": [round(float(v), 4) for v in mean_xyz],
        "std_xyz": [round(float(v), 4) for v in std_xyz],
        "mean_magnitude": round(sum_magnitude / count, 4),
        "max_magnitude": round(max_magnitude, 4),
    }


def _itk_transform_summary(path: Path) -> dict[str, Any] | None:
    """A structured summary of an ITK transform HDF5 (KonfAI's ``:itktransform`` backend), else None.

    Registration apps write their result as ``Transform.h5``; without this reader the number an
    agent is asked for (how far did the registration move?) sits in a file none of its tools can
    open. Linear transforms return their full parameters; dense grids return per-axis mean/std and
    displacement magnitudes, which is what a sanity check against a known shift needs.
    """
    try:
        import h5py
    except ImportError:
        raise ValueError(
            f"'{path.name}' is an HDF5 file but the 'h5py' package is not installed: "
            "install konfai[hdf5] to read ITK transforms."
        ) from None
    try:
        handle = h5py.File(path, "r")
    except OSError:
        return None
    with handle:
        if "TransformGroup" not in handle:
            return None
        transforms: list[dict[str, Any]] = []
        for index in sorted(handle["TransformGroup"], key=lambda name: int(name) if name.isdigit() else 0):
            group = handle[f"TransformGroup/{index}"]
            entry: dict[str, Any] = {"index": index}
            if "TransformType" in group:
                raw = group["TransformType"][()]
                first = raw[0] if getattr(raw, "shape", None) else raw
                entry["type"] = first.decode() if isinstance(first, bytes) else str(first)
            if "TransformFixedParameters" in group:
                entry["fixed_parameters"] = [round(float(v), 6) for v in group["TransformFixedParameters"][()]]
            if "TransformParameters" in group:
                dataset = group["TransformParameters"]
                if dataset.size <= _LINEAR_TRANSFORM_MAX_PARAMS:
                    entry["parameters"] = [round(float(v), 6) for v in dataset[()]]
                else:
                    # A dense grid is summarised without ever being loaded whole: [()] on a
                    # displacement field materializes gigabytes before the stats run.
                    entry["parameter_count"] = int(dataset.size)
                    if dataset.ndim == 1 and dataset.size % 3 == 0:
                        entry["displacement_summary"] = _dense_displacement_summary(dataset)
            transforms.append(entry)
    return {
        "ok": True,
        "path": str(path),
        "kind": "itk_transform",
        "total_bytes": path.stat().st_size,
        "transforms": transforms,
        "note": "Parameters are in the transform's own (ITK, physical-space) convention, fixed->moving.",
        "next_actions": ["inspect_dataset", "read_job_log"],
    }


def read_dataset_sidecar(path: Path, max_lines: int = 200, max_chars: int = 65536) -> dict[str, Any]:
    """Read a bounded preview of a dataset's non-image text file (CSV/TSV, JSON, YAML, headers, lists).

    Streams at most ``max_chars`` characters (memory never scales with file size) and refuses binary
    content by sniffing the first bytes for NUL: image volumes and weights belong to
    ``inspect_dataset``/``preview_volume``, not a text reader. CSV/TSV additionally get a structured
    preview (header + up to ``max_lines`` rows) so the agent can map label columns to cases without
    parsing raw text.
    """
    if not path.is_file():
        raise ValueError(f"Not a readable file: {path}")
    if path.suffix.lower() in {".h5", ".hdf5"}:
        transform = _itk_transform_summary(path)
        if transform is not None:
            return transform
    with path.open("rb") as handle:
        head = handle.read(_BINARY_SNIFF_BYTES)
    if b"\x00" in head:
        raise ValueError(
            f"'{path.name}' looks binary (NUL bytes in the first {_BINARY_SNIFF_BYTES} bytes). "
            "Use inspect_dataset for volumes/stores and preview_volume for image content."
        )
    max_lines = max(max_lines, 1)
    ranged = read_text_range(path, max_chars=max_chars)
    lines = ranged["content"].splitlines()
    line_truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    payload: dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "total_bytes": ranged["total_bytes"],
        "returned_lines": len(lines),
        "truncated": bool(ranged["truncated"] or line_truncated),
        "content": "\n".join(lines),
        "next_actions": ["inspect_dataset", "design_config_strategy"],
    }
    delimiter = _DELIMITED_SUFFIXES.get(path.suffix.lower())
    if delimiter is not None and lines:
        import csv

        rows = list(csv.reader(StringIO("\n".join(lines)), delimiter=delimiter))
        if rows:
            payload["kind"] = "delimited"
            payload["columns"] = rows[0]
            payload["rows"] = rows[1:]
            payload["returned_rows"] = len(rows) - 1
    return payload


def read_text_tail(path: Path, max_lines: int) -> str:
    """Return the last ``max_lines`` lines of a text file efficiently."""
    if not path.exists() or max_lines <= 0:
        return ""

    chunk_size = 8192
    remaining = max_lines
    chunks: list[str] = []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and remaining > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size).decode("utf-8", errors="replace")
            chunks.append(chunk)
            remaining -= chunk.count("\n")

    text = "".join(reversed(chunks))
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def full_suffix(path: Path) -> str:
    """Return all suffixes joined together, for example ``.nii.gz``."""
    return "".join(path.suffixes)


def basename_without_suffixes(path: Path) -> str:
    """Return the basename of ``path`` without compound suffixes."""
    suffix = full_suffix(path)
    return path.name[: -len(suffix)] if suffix else path.name
