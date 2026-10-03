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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from konfai.utils.errors import DatasetManagerError, KonfAIWarning

#: The suffix an entry is moved to while its replacement is published. Per pid, so two writers of
#: one entry never share a backup.
_REPLACED_MARKER = ".replaced-"


def _replaced_name(name: str) -> str:
    """Where ``name`` (an h5 key or a directory leaf) is kept while its replacement is published."""
    return f"{name}{_REPLACED_MARKER}{os.getpid()}"


def publish(staging: Path, final: Path) -> None:
    """Put the complete ``staging`` entry, a file or a directory, at ``final``; the entry already there is replaced
    only once its successor is in place. A file goes by ``os.replace``; a directory through the :func:`_replaced_name`
    hop, which a failed rename puts back. A concurrent writer that published the same entry first keeps it."""
    if not staging.is_dir():
        os.replace(staging, final)
        return
    backup = final.with_name(_replaced_name(final.name))
    replaced = final.exists()
    if replaced:
        shutil.rmtree(backup, ignore_errors=True)
        final.rename(backup)
    try:
        staging.rename(final)
    except OSError:
        if not final.exists():
            if replaced:
                backup.rename(final)
            raise
        shutil.rmtree(staging, ignore_errors=True)
    if replaced:
        shutil.rmtree(backup, ignore_errors=True)


@contextmanager
def staged_entry(final: Path) -> Iterator[Path]:
    """``final``'s own name in a hidden staging directory beside it, for a writer that may make companion files
    (MetaImage's ``.raw``, Analyze's ``.img``). On a clean exit each file it made is published beside ``final``
    under its own name, the companions first and ``final`` last, so a reader meets the entry complete; the pixel files
    the replaced entry read go once it is replaced, when named after it. A MetaImage written under a pixel name of its
    own (``transfer_entry``) is thus replaced whole or not at all. Analyze pairs its files by name, so two renames
    could leave the old header over the new pixels: an existing pair is refused, before any work."""
    from konfai.utils.dataset.stream import DataStream  # stream builds on this module

    if final.name.lower().endswith(".hdr") and final.exists():
        raise DatasetManagerError(
            f"'{final}' already exists as an Analyze pair (.hdr/.img), which cannot be replaced whole: "
            "its two files pair by name.",
            "Write to a new destination, remove the existing pair first, or use .nii.gz, "
            "which holds its pixels in the same file.",
        )
    directory = final.with_name(f".{final.name}.{DataStream.temporary_suffix()}")
    directory.mkdir(parents=True)
    try:
        yield directory / final.name
        replaced = entry_files(final)[1:] if final.exists() else []
        made = sorted(directory.iterdir(), key=lambda path: path.name == final.name)
        for part in made:
            publish(part, final.with_name(part.name))
        kept = {part.name for part in made}
        for old in replaced:
            if old.parent == final.parent and old.name not in kept and old.name.startswith(f"{final.stem}."):
                old.unlink(missing_ok=True)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _pixel_reference(path: Path) -> Path | None:
    """Where the detached header ``path`` reads its pixels, as it names them: Analyze's ``.img`` beside it under its
    name, a MetaImage's ``ElementDataFile`` (relative to the header, or absolute); ``None`` for any other entry and
    for pixels that follow the header (``LOCAL``)."""
    name = path.name.lower()
    if name.endswith(".hdr"):
        return Path(path.with_suffix(".img").name)
    if name.endswith(".mhd"):
        from konfai.utils.dataset.raw_block import _mha_header  # raw_block builds on this module

        fields = _mha_header(str(path))
        data = fields[0].get("ElementDataFile") if fields else None
        return None if data is None or data == "LOCAL" else Path(data)
    return None


def entry_links(src: Path, dest: Path) -> list[tuple[Path, Path]]:
    """Each file of the image entry ``src`` and where it must lie for its header, left as it is, to read it as
    ``dest``: the header at ``dest``, Analyze's pixels beside it under its name, a MetaImage's where its
    ``ElementDataFile`` points from ``dest``. An absolute one is read where it lies, so a link leaves it there."""
    links = [(src, dest)]
    pixels = _pixel_reference(src)
    if pixels is not None and not pixels.is_absolute():
        read = dest.with_suffix(".img") if src.name.lower().endswith(".hdr") else dest.parent / pixels
        links.append((src.parent / pixels, read))
    return links


def entry_files(path: Path) -> list[Path]:
    """The files one image entry is made of beside its header: ``path``, then the pixel file it names."""
    return [part for part, _ in entry_links(path, path)]


def transfer_entry(src: Path, dest: Path, move: bool = False) -> None:
    """Copy (``move``: move) the image entry ``src`` to ``dest``: a file, a store, or a detached header and its pixels,
    wherever the header points, so the result stands without the source. A MetaImage's pixels take a name of their
    own, which the header written at ``dest`` gives (``staged_entry`` publishes them before it); Analyze's follow
    ``dest``'s name."""
    reference = _pixel_reference(src)
    if reference is None:
        (shutil.move if move else shutil.copytree if src.is_dir() else shutil.copy2)(src, dest)
        return
    transfer = shutil.move if move else shutil.copy2
    pixels = src.parent / reference
    if not src.name.lower().endswith(".mhd"):
        transfer(pixels, dest.with_suffix(".img"))
        transfer(src, dest)
        return
    import secrets

    own = dest.with_name(f"{dest.stem}.{secrets.token_hex(4)}{pixels.suffix}")
    transfer(pixels, own)
    header = src.read_text("latin-1")
    dest.write_text(
        re.sub(r"(?m)^(ElementDataFile\s*=\s*).*$", lambda match: match.group(1) + own.name, header), "latin-1"
    )
    if move:
        src.unlink()


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
    """Whether the writer that staged under ``pid`` no longer runs. ``psutil``, not ``os.kill(pid, 0)``:
    on Windows signal 0 is CTRL_C_EVENT, sent to the process rather than probing it."""
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
        KonfAIWarning,
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
