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


"""Entries published through a staging name: what a dead writer leaves behind and how it is recovered."""

from __future__ import annotations

import os
import re
import shutil
import warnings
from collections.abc import Iterable
from pathlib import Path

#: The suffix an entry is moved to while its replacement is published. Per pid, so two writers of
#: one entry never share a backup.
_REPLACED_MARKER = ".replaced-"


def _replaced_name(name: str) -> str:
    """Where ``name`` (an h5 key or a directory leaf) is kept while its replacement is published."""
    return f"{name}{_REPLACED_MARKER}{os.getpid()}"


def is_staging_entry(name: str) -> bool:
    """Whether ``name`` (a path or an h5 key) is a writer's staging entry, never a case: a temporary
    carrying the ``.tmp`` marker of :meth:`DataStream.temporary_suffix` or :meth:`DataStream.staging_path`,
    or the :func:`_replaced_name` an entry is moved to while its replacement is published."""
    leaf = os.path.basename(name)
    return leaf.endswith(".tmp") or ".tmp." in leaf or _REPLACED_MARKER in leaf


# A writer's staging name carries its pid: ``<entry>.<pid>[-n].tmp``, the ``.replaced`` hop it keeps
# the previous version under, or the dotted whole-file form ``.<entry>.<pid>.tmp.<ext>``.
_STAGING_PID = re.compile(r"\.(?:(?P<pid>\d+)(?:-\d+)?\.(?:tmp|replaced)|replaced-(?P<hop>\d+))(?:\.|$)")


def _writer_is_dead(pid: int) -> bool:
    """Whether the writer that staged under ``pid`` no longer runs. ``psutil``, not ``os.kill(pid, 0)``,
    which on Windows raises a generic OSError for a missing pid."""
    if pid == os.getpid():
        return False
    import psutil

    return not psutil.pid_exists(pid)


def _orphaned_backup_names(names: Iterable[str], entry: str) -> list[str]:
    """Among ``names``, the backups a DEAD writer left of ``entry``: ``<entry>.replaced-<pid>``."""
    marker = f"{entry}{_REPLACED_MARKER}"
    kept = []
    for candidate in names:
        if not candidate.startswith(marker):
            continue
        pid = candidate[len(marker) :]
        if pid.isdigit() and _writer_is_dead(int(pid)):
            kept.append(candidate)
    return kept


def _recover_orphaned_backup(final: Path) -> bool:
    """Put back the previous entry when a killed writer left it under its backup name alone.

    A replacement moves the old entry aside as ``<name>.replaced-<pid>``, publishes the new one,
    then drops the backup. Exactly one backup, from a writer that no longer runs, and no entry under
    the final name: that backup goes back. Two backups, or a writer still running, and the entry
    stays missing.
    """
    if final.exists():
        return False
    try:
        siblings = [path.name for path in final.parent.iterdir()]
    except OSError:
        return False
    backups = _orphaned_backup_names(siblings, final.name)
    if len(backups) != 1:
        return False
    backup = final.parent.joinpath(backups[0])
    try:
        # Never over a publish that landed meanwhile: the move itself refuses. os.link fails EEXIST,
        # a Windows rename fails outright, and a directory rename fails ENOTEMPTY against a
        # published store (never an empty directory).
        if backup.is_dir() or os.name == "nt":
            backup.rename(final)
        else:
            os.link(backup, final)
            backup.unlink()
    except OSError:
        return False
    warnings.warn(
        f"'{final}' was missing and its previous version was recovered from '{backups[0]}': a writer "
        "was killed between moving the entry aside and publishing its replacement. The entry is the one "
        "that was there BEFORE that write; run the write again to replace it.",
        UserWarning,
        stacklevel=2,
    )
    return True


def _retire_dead_debris(final: Path) -> None:
    """Remove what earlier, dead writers of ``final`` left beside it.

    A live writer's staging is left alone: two writers of one entry are legal, the last rename wins.
    """
    entry = final.name.split(".", 1)[0]
    try:
        siblings = list(final.parent.iterdir())
    except OSError:
        return
    for sibling in siblings:
        if sibling == final or not sibling.name.lstrip(".").startswith(f"{entry}."):
            continue
        marker = _STAGING_PID.search(sibling.name)
        if marker is None or not _writer_is_dead(int(marker.group("pid") or marker.group("hop"))):
            continue
        if sibling.is_dir():
            shutil.rmtree(sibling, ignore_errors=True)
        else:
            sibling.unlink(missing_ok=True)
