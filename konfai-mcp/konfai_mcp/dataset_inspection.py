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

"""Dataset inspection half of ``SessionService``: structure scans, candidate-root
discovery, and sampled group statistics. Split out of ``server_experiments.py`` so the
session service keeps only workflow/session logic; ``SessionService`` inherits this mixin."""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import numpy as np

from .server_support import (
    DatasetGroupUnreadableError,
    aggregate_case_statistics,
    basename_without_suffixes,
    case_directories,
    full_suffix,
)


class DatasetInspectionMixin:
    """Dataset structure/statistics methods mixed into ``SessionService``."""

    def _supported_extensions(self) -> set[str]:
        from konfai.utils.utils import SUPPORTED_EXTENSIONS

        return {extension.lower() for extension in SUPPORTED_EXTENSIONS}

    def _iter_directories(
        self, root: Path, max_depth: int, max_directories: int
    ) -> tuple[list[tuple[Path, int]], bool]:
        queue: list[tuple[Path, int]] = [(root, 0)]
        visited: list[tuple[Path, int]] = []
        truncated = False

        while queue:
            current, depth = queue.pop(0)
            visited.append((current, depth))
            if len(visited) >= max_directories:
                truncated = bool(queue)
                break
            if depth >= max_depth:
                continue
            try:
                children = sorted([path for path in current.iterdir() if path.is_dir()], key=lambda path: path.name)
            except OSError:
                # An unreadable subdirectory (permissions, a broken mount) must not abort the whole scan.
                continue
            for child in children:
                queue.append((child, depth + 1))

        return visited, truncated

    def _h5_group_names(self, path: Path) -> list[str] | None:
        """List the top-level keys of one HDF5 file (KonfAI's real groups), or None without h5py."""
        try:
            import h5py
        except ImportError:
            return None
        try:
            with h5py.File(path, "r") as handle:
                return sorted(handle.keys())
        except Exception:
            return None

    # Files that mark a directory as an OME-Zarr / zarr store root: ``zarr.json`` (Zarr v3) or the
    # Zarr v2 sidecars ``.zgroup`` / ``.zarray`` / ``.zattrs``. A per-case or flat dataset root never
    # carries these at its top level (they live one or two directories down, inside ``<group>.zarr/``),
    # so sniffing them only ever fires when the root itself IS a store.
    _ZARR_STORE_MARKERS = ("zarr.json", ".zgroup", ".zarray", ".zattrs")

    def _zarr_store_extension(self, path: Path, supported_extensions: set[str]) -> str | None:
        """Return the OME-Zarr extension when ``path`` IS itself a zarr store, else None.

        Detects a store handed in as the dataset root by its ``.zarr``-family directory suffix or, for
        a store whose directory was renamed without the suffix, by its zarr store markers. Callers must
        describe such a path as one store rather than walking its multiscale levels as case directories.
        """
        for alias in (".ome_zarr", ".ome-zarr", ".omezarr", ".zarr"):
            if path.name.lower().endswith(alias):
                extension = alias.lstrip(".")
                return extension if extension in supported_extensions else "zarr"
        if any((path / marker).is_file() for marker in self._ZARR_STORE_MARKERS):
            return "zarr" if "zarr" in supported_extensions else None
        return None

    def _classify_directory_entry(self, path: Path, supported_extensions: set[str]) -> tuple[str, str] | None:
        """Classify one sub-directory as a directory-backed dataset entry ``(group, extension)``.

        KonfAI stores some formats as directories (OME-Zarr stores like ``CT.ome.zarr/``, DICOM
        series folders); a file-only scan would make them invisible to every dataset tool.
        Returns None when the directory is not a recognizable data entry (e.g. a nested root).
        """
        compound = full_suffix(path).lstrip(".").lower()
        if compound in supported_extensions:
            return basename_without_suffixes(path), compound
        last = path.suffix.lstrip(".").lower()
        if last in supported_extensions and "zarr" in last:
            return basename_without_suffixes(path), last
        try:
            for child in itertools.islice(path.iterdir(), 256):
                name = child.name.lower()
                if name == "dicomdir" or name.endswith((".dcm", ".dicom")):
                    return path.name, "dicom"
                # PACS exports routinely store DICOM slices with NO extension (e.g. ``IM0001``); a
                # suffix-only check misses them, so sniff the DICOM ``DICM`` magic at byte 128.
                if child.is_file() and self._is_dicom_file(child):
                    return path.name, "dicom"
        except OSError:
            # An unreadable sub-directory must not abort the whole dataset scan.
            return None
        return None

    @staticmethod
    def _is_dicom_file(path: Path) -> bool:
        """True if the file carries the DICOM Part-10 ``DICM`` magic at offset 128 (extensionless slices)."""
        try:
            with path.open("rb") as handle:
                handle.seek(128)
                return handle.read(4) == b"DICM"
        except OSError:
            return False

    def _scan_case_directory(
        self,
        case_dir: Path,
        supported_extensions: set[str],
    ) -> tuple[list[tuple[Path, str, str]], list[str]]:
        """Scan one case directory into ``(path, group, extension)`` entries plus ignored names."""
        entries: list[tuple[Path, str, str]] = []
        ignored: list[str] = []

        for path in sorted(case_dir.iterdir(), key=lambda child: child.name):
            if path.is_dir():
                directory_entry = self._classify_directory_entry(path, supported_extensions)
                if directory_entry is not None:
                    entries.append((path, *directory_entry))
                continue
            if not path.is_file():
                continue
            suffix = full_suffix(path).lstrip(".").lower()
            if suffix not in supported_extensions:
                ignored.append(path.name)
                continue
            if suffix == "h5":
                internal_groups = self._h5_group_names(path)
                if internal_groups:
                    entries.extend((path, group, suffix) for group in internal_groups)
                    continue
            entries.append((path, basename_without_suffixes(path), suffix))
        return entries, ignored

    def _scan_dataset_structure(self, dataset_dir: Path) -> dict[str, Any]:
        supported_extensions = self._supported_extensions()
        # The dataset root can itself BE a single OME-Zarr store (``store.zarr/`` with a root
        # ``zarr.json`` / ``.zgroup`` and multiscale ``scaleN`` levels). Walking it as a case tree
        # mis-reports the scale levels as cases and hides the store; describe it as one store.
        single_store_extension = self._zarr_store_extension(dataset_dir, supported_extensions)
        case_dirs = [dataset_dir] if single_store_extension is not None else case_directories(dataset_dir)
        # A root-level OME-Zarr store handed under a per-case root is a data entry, not a case directory:
        # treat the root as a flat dataset so dataset/CT.ome.zarr + dataset/SEG.mha is reported correctly.
        if (
            single_store_extension is None
            and case_dirs != [dataset_dir]
            and any(path.name.lower().endswith((".zarr", ".omezarr", ".ome-zarr", ".ome_zarr")) for path in case_dirs)
        ):
            case_dirs = [dataset_dir]
        per_case = single_store_extension is None and case_dirs != [dataset_dir]

        groups: dict[str, dict[str, Any]] = {}
        missing_by_case: dict[str, list[str]] = {}
        case_summaries: list[dict[str, Any]] = []
        ignored_files: list[str] = []
        case_group_names: dict[str, set[str]] = {}
        case_file_names: list[set[str]] = []
        all_groups: set[str] = set()

        for case_dir in case_dirs:
            if single_store_extension is not None:
                # The root is one store: its basename is the single group, the store dir is the entry.
                scan_entries = [(dataset_dir, basename_without_suffixes(dataset_dir), single_store_extension)]
                ignored: list[str] = []
            else:
                scan_entries, ignored = self._scan_case_directory(case_dir, supported_extensions)
            file_groups = {group for _, group, _ in scan_entries}
            case_group_names[case_dir.name] = file_groups
            case_file_names.append({path.name for path, _, _ in scan_entries})
            all_groups.update(file_groups)
            ignored_files.extend(str(case_dir / filename) for filename in ignored)
            case_summaries.append(
                {
                    "case": case_dir.name,
                    "files": sorted({path.name for path, _, _ in scan_entries}),
                    "ignored_files": ignored,
                }
            )
            for path, group, extension in scan_entries:
                entry = groups.setdefault(
                    group,
                    {
                        "count": 0,
                        "extensions": set(),
                        "sample_path": str(path),
                    },
                )
                entry["count"] += 1
                if extension:
                    entry["extensions"].add(extension)

        for case_name, file_groups in case_group_names.items():
            missing = sorted(all_groups - file_groups)
            if missing:
                missing_by_case[case_name] = missing

        for entry in groups.values():
            entry["extensions"] = sorted(entry["extensions"])

        detected_extensions = sorted({ext for info in groups.values() for ext in info["extensions"]})
        default_extension = detected_extensions[0] if detected_extensions else None
        common_filenames = sorted(set.intersection(*case_file_names)) if case_file_names else []
        suggested_groups_src = {
            group: {
                "groups_dest": {
                    group: {
                        "transforms": None,
                        "patch_transforms": None,
                        # A group's role is task-dependent and cannot be inferred from its name
                        # (a CT is the input for segmentation but the target for MR->CT synthesis),
                        # so leave it null and let the agent set it from the user's objective —
                        # see is_input_meaning. Guessing here silently mis-wires the config.
                        "is_input": None,
                    }
                }
            }
            for group in sorted(groups)
        }
        dataset_entry = f"{dataset_dir}:a:{default_extension}" if default_extension is not None else None

        structure_warnings: list[str] = []
        if single_store_extension is not None:
            # A bare store is not a KonfAI dataset directory (KonfAI expects <root>/<case>/<group>.<ext>),
            # so it cannot be loaded directly: null the entry and tell the agent how to lay it out.
            dataset_entry = None
            structure_warnings.append(
                f"The path is a single OME-Zarr store ('{dataset_dir.name}'), one image with its multiscale "
                "levels, not a KonfAI dataset root. KonfAI expects a per-case layout "
                "'<root>/<case>/<group>.zarr': move this store inside a case directory (e.g. "
                f"'<root>/<case>/{dataset_dir.name}') and point the dataset root at the directory of cases."
            )

        layout = (
            "single_store"
            if single_store_extension is not None
            else ("per_case_directories" if per_case else "flat_directory")
        )
        return {
            "layout": layout,
            **({"warnings": structure_warnings} if structure_warnings else {}),
            "total_cases": len(case_dirs),
            "groups": groups,
            "case_samples": case_summaries[: min(10, len(case_summaries))],
            "missing_by_case": missing_by_case,
            "ignored_files": ignored_files[:100],
            "detected_extensions": detected_extensions,
            "common_groups": sorted(groups),
            "common_filenames": common_filenames,
            "default_extension": default_extension,
            "dataset_entry": dataset_entry,
            "suggested_groups_src": suggested_groups_src,
            "is_input_meaning": (
                "is_input is each group's ROLE in the model graph: true = an input fed to the network "
                "(the data it sees); false = a target/supervision held out of the input (a segmentation "
                "label, or the volume a synthesis model must produce). It is left null in "
                "suggested_groups_src because it cannot be inferred from the group name — decide it from "
                "the user's task. Examples: CT->segmentation => CT is_input:true, SEG is_input:false; "
                "MR->CT synthesis => MR is_input:true, CT is_input:false. "
                "design_config_strategy(group_roles=...) resolves this with the user."
            ),
        }

    def browse_dataset_payload(
        self,
        dataset_dir: Path,
        depth: int = 2,
        max_entries: int = 200,
        max_candidate_depth: int | None = None,
    ) -> dict[str, Any]:
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        if not dataset_dir.is_dir():
            raise ValueError(f"Dataset path is not a directory: {dataset_dir}")

        depth = max(depth, 0)
        max_entries = max(max_entries, 1)
        candidate_depth = depth if max_candidate_depth is None else max(max_candidate_depth, 0)

        entries: list[dict[str, Any]] = []
        tree: list[str] = []
        truncated = False

        def visit(path: Path, current_depth: int) -> bool:
            nonlocal truncated
            children = sorted(path.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower(), child.name))
            for child in children:
                relative = child.relative_to(dataset_dir)
                display = f"{relative.as_posix()}/" if child.is_dir() else relative.as_posix()
                entries.append(
                    {
                        "path": str(relative.as_posix()),
                        "type": "directory" if child.is_dir() else "file",
                        "depth": current_depth + 1,
                    }
                )
                tree.append(display)
                if len(entries) >= max_entries:
                    truncated = True
                    return True
                # A child sits at ``current_depth + 1``; recurse only while its own children would still
                # fall within ``depth``, so ``depth`` is an inclusive cap on entry depth (depth=1 lists
                # immediate children only, not grandchildren).
                if child.is_dir() and current_depth + 1 < depth:
                    if visit(child, current_depth + 1):
                        return True
            return False

        if depth > 0:
            visit(dataset_dir, 0)

        candidate_roots = self._candidate_dataset_roots(
            dataset_dir,
            max_depth=candidate_depth,
            max_candidates=min(10, max_entries),
        )
        summary_scan = self._scan_dataset_structure(dataset_dir)
        resolved_root = dataset_dir
        if not summary_scan["groups"] and candidate_roots:
            resolved_root = Path(candidate_roots[0]["path"])
            summary_scan = self._scan_dataset_structure(resolved_root)

        return {
            "ok": True,
            "path": str(dataset_dir),
            "requested_path": str(dataset_dir),
            "root": str(resolved_root),
            "root_inferred": resolved_root != dataset_dir,
            "depth": depth,
            "max_entries": max_entries,
            "tree_format": "Relative POSIX-style paths. Directories end with '/'.",
            "tree": tree,
            "entries": entries,
            "truncated": truncated,
            "case_count": summary_scan["total_cases"],
            "common_groups": summary_scan["common_groups"],
            "common_filenames": summary_scan["common_filenames"],
            "extensions": summary_scan["detected_extensions"],
            "ignored_files": summary_scan["ignored_files"],
            "missing_by_case": summary_scan["missing_by_case"],
            **({"warnings": summary_scan["warnings"]} if summary_scan.get("warnings") else {}),
            "candidate_dataset_roots": candidate_roots,
            "next_actions": ["inspect_dataset", "design_config_strategy"],
        }

    def _candidate_dataset_roots(
        self, dataset_dir: Path, max_depth: int = 2, max_candidates: int = 10
    ) -> list[dict[str, Any]]:
        directories, truncated = self._iter_directories(dataset_dir, max_depth=max_depth, max_directories=64)
        candidates: list[dict[str, Any]] = []
        accepted_roots: list[Path] = []

        for candidate_dir, depth in directories:
            if any(parent == candidate_dir or parent in candidate_dir.parents for parent in accepted_roots):
                continue
            scan = self._scan_dataset_structure(candidate_dir)
            if not scan["groups"]:
                continue
            relative_path = "." if candidate_dir == dataset_dir else candidate_dir.relative_to(dataset_dir).as_posix()
            candidates.append(
                {
                    "path": str(candidate_dir),
                    "relative_path": relative_path,
                    "depth": depth,
                    "layout": scan["layout"],
                    "total_cases": scan["total_cases"],
                    "groups": sorted(scan["groups"]),
                    "detected_extensions": scan["detected_extensions"],
                }
            )
            accepted_roots.append(candidate_dir)

        candidates.sort(
            key=lambda item: (
                -item["total_cases"],
                -len(item["groups"]),
                item["depth"],
                item["path"],
            )
        )
        result = candidates[:max_candidates]
        if truncated and result:
            result[0] = {
                **result[0],
                "discovery_truncated": True,
            }
        return result

    def _infer_dataset_structure_payload(self, dataset_dir: Path, *, discover_candidates: bool) -> dict[str, Any]:
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        scan = self._scan_dataset_structure(dataset_dir)
        payload = {
            "ok": True,
            "path": str(dataset_dir),
            "layout": scan["layout"],
            "total_cases": scan["total_cases"],
            "groups": scan["groups"],
            "case_samples": scan["case_samples"],
            "missing_by_case": scan["missing_by_case"],
            "ignored_files": scan["ignored_files"],
            "detected_extensions": scan["detected_extensions"],
            "dataset_entry": scan["dataset_entry"],
            "suggested_groups_src": scan["suggested_groups_src"],
            "is_input_meaning": scan["is_input_meaning"],
            **({"warnings": scan["warnings"]} if scan.get("warnings") else {}),
        }
        if discover_candidates:
            candidate_roots = self._candidate_dataset_roots(dataset_dir)
            payload["candidate_dataset_roots"] = candidate_roots
            if not payload["groups"] and candidate_roots:
                payload["warnings"] = [
                    *(payload.get("warnings") or []),
                    "No supported groups were found directly under the requested path. "
                    "Inspect candidate_dataset_roots or call browse_dataset to locate the actual dataset root.",
                ]
            payload["next_actions"] = [
                "browse_dataset",
                "inspect_dataset",
                "design_config_strategy",
                "initialize_session",
            ]
        return payload

    def infer_dataset_structure_payload(self, dataset_dir: Path) -> dict[str, Any]:
        return self._infer_dataset_structure_payload(dataset_dir, discover_candidates=True)

    def _load_dataset(self, dataset_dir: Path, extension: str) -> Any:
        from konfai.utils.dataset import Dataset
        from konfai.utils.utils import SUPPORTED_EXTENSIONS

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported dataset extension '{extension}'.")
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        return Dataset(str(dataset_dir), extension)

    def _sample_dataset_names(self, names: list[str], max_cases: int | None, seed: int) -> list[str]:
        if max_cases is None or max_cases <= 0 or len(names) <= max_cases:
            return sorted(names)
        rng = random.Random(seed)
        return sorted(rng.sample(names, max_cases))

    def compute_dataset_group_statistics(
        self,
        dataset_dir: Path,
        group: str,
        extension: str = "mha",
        max_cases: int | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        dataset = self._load_dataset(dataset_dir, extension)
        names = sorted(dataset.get_names(group))
        if not names:
            raise ValueError(f"Group '{group}' not found in dataset '{dataset_dir}'.")

        sampled_names = self._sample_dataset_names(names, max_cases, seed)
        sampled_items: dict[str, dict[str, Any]] = {}
        per_case_labels: dict[str, dict[str, float]] = {}
        # Cap separating a label map (a bounded class set -- whole-body atlases like TotalSegmentator
        # top out around 117 classes) from an intensity image stored as an integer (int16 CT -> thousands
        # of distinct values, for which per-label voxel fractions are meaningless and huge).
        label_stats_max_classes = 512
        high_cardinality_labels: dict[str, int] = {}
        unreadable_cases: dict[str, str] = {}
        for name in sampled_names:
            try:
                data, attr = dataset.read_data(group, name)
            except Exception as exc:
                # A single bad file must not abort the whole group.
                # The structure scan already found this file on disk, so a read failure means the file
                # itself is unreadable (corrupt, truncated, empty, or non-image bytes) -- a different
                # problem from a missing group or a layout/token mismatch. Record it per case and keep
                # going so one bad file never hides the healthy cases' statistics.
                unreadable_cases[name] = f"{type(exc).__name__}: {str(exc).strip() or 'read failed'}"
                continue
            p25, p50, p75 = np.percentile(data, (25, 50, 75))
            sampled_items[name] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "25pc": float(p25),
                "50pc": float(p50),
                "75pc": float(p75),
                "shape": list(data.shape),
                "spacing": attr.get_np_array("Spacing").tolist(),
            }
            # Segmentation-style groups: expose label ids and per-label voxel fractions so
            # nb_class and class weights stop being guesses.
            if np.issubdtype(data.dtype, np.integer):
                unique, counts = np.unique(data, return_counts=True)
                if unique.size <= label_stats_max_classes:
                    per_case_labels[name] = {
                        str(int(label)): round(float(count) / float(data.size), 6)
                        for label, count in zip(unique, counts, strict=False)
                    }
                else:
                    # Above the cap: record the cardinality instead of silently dropping the info, so the
                    # agent learns this is an intensity-like integer group, not a many-class label map.
                    high_cardinality_labels[name] = int(unique.size)
        if unreadable_cases and not sampled_items:
            # Every sampled case exists but failed to read: a corrupt/unreadable-FILE problem, not a
            # missing group or a layout/token mismatch. Raise a distinct error so the caller's reason
            # builder tells the agent to inspect/replace the files, not restructure the dataset.
            raise DatasetGroupUnreadableError(group, extension, unreadable_cases)
        statistics = aggregate_case_statistics(sampled_items)

        payload: dict[str, Any] = {
            "group": group,
            "extension": extension,
            "dataset_path": str(dataset_dir),
            "total_cases": len(names),
            "sampled_cases": len(sampled_names),
            "readable_cases": len(sampled_items),
            "sample_names": sampled_names[: min(20, len(sampled_names))],
            "sampled": len(sampled_names) != len(names),
            "statistics": statistics,
        }
        if per_case_labels:
            all_labels = sorted({label for labels in per_case_labels.values() for label in labels}, key=int)
            payload["labels"] = {
                "unique": [int(label) for label in all_labels],
                "count": len(all_labels),
                "presence_cases": {
                    label: sum(1 for labels in per_case_labels.values() if label in labels) for label in all_labels
                },
                "mean_voxel_fraction": {
                    label: round(
                        sum(labels.get(label, 0.0) for labels in per_case_labels.values()) / len(per_case_labels), 6
                    )
                    for label in all_labels
                },
                "per_case": per_case_labels,
            }
        if high_cardinality_labels:
            # Never silently omit the label section: state that the integer group exceeds the label cap
            # (so it reads as an intensity image), with the observed distinct-value counts.
            payload["high_cardinality_integer_group"] = {
                "max_classes": label_stats_max_classes,
                "note": (
                    "integer group with more distinct values than the label cap; treated as intensity-like, "
                    "not a label map, so per-label voxel fractions are omitted"
                ),
                "distinct_values_per_case": high_cardinality_labels,
            }
        if unreadable_cases:
            # Partial corruption: some cases read, some did not. Surface the bad ones as a distinct
            # per-case signal (with the real reader error) so the agent fixes those files rather than
            # mistrusting the whole group's statistics.
            payload["unreadable_cases"] = {
                "count": len(unreadable_cases),
                "note": (
                    "these cases exist on disk but the reader raised (corrupt/truncated/empty/non-image); "
                    "statistics above are computed only from the readable cases"
                ),
                "errors": unreadable_cases,
            }
        return payload
