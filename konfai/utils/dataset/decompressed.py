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


"""Region reads of a compressed ITK file (``.nii.gz``, a zlib MetaImage, a gzip NRRD): an uncompressed
twin in a cache directory, decompressed once per run and then mapped band by band like any
uncompressed file.

A compressed stream is not seekable, so ITK decodes it from its start for every region asked of it.
The twin moves that cost to one decode per entry per run. It lives under
``$KONFAI_DECOMPRESSED_DIRECTORY`` (default ``~/.cache/konfai/decompressed``), in one directory per
run that every process of the run shares and that the run removes when it ends. A twin is filed
under the entry it is read for (the case, or an ``Expand`` copy of it), so a one-pass reader leaving
a case removes every twin of it and of its copies, from whichever root and stage it was read.
"""

from __future__ import annotations

import contextlib
import functools
import gzip
import hashlib
import multiprocessing.util
import os
import shutil
import socket
import sys
import warnings
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset.raw_block import _MHA_DTYPES, _mha_header, _nrrd_header, _sitk_component_dtypes
from konfai.utils.dataset.staging import _writer_is_dead
from konfai.utils.dataset.stream import DataStream
from konfai.utils.errors import KonfAIWarning

#: Where the runs keep their twins. Unset, ``$XDG_CACHE_HOME/konfai/decompressed`` (``~/.cache``
#: when that is unset too): a disk, where the system temporary directory is often memory.
DIRECTORY_VARIABLE = "KONFAI_DECOMPRESSED_DIRECTORY"
#: The directory of the run in progress, which :func:`run_scope` hands to every process it starts.
_RUN_VARIABLE = "KONFAI_DECOMPRESSED_RUN"
#: The share of its disk's free space a run's twins may take. An entry whose twin would pass it is
#: read from its compressed stream, as it was before twins.
FREE_SPACE_SHARE = 0.5
#: Bytes read or written per step: what a decompression holds, whatever the volume.
_CHUNK = 1 << 20
#: The compressed suffixes served, and their twin's.
_TWIN_SUFFIXES = ((".nii.gz", ".nii"), (".mha", ".mha"), (".mhd", ".mha"), (".nrrd", ".nrrd"))

_refusal_warned = False
_process_runs: set[Path] = set()


class _Layout(NamedTuple):
    """The twin a compressed file decompresses into: the block a region read maps, and how it is written."""

    shape: tuple[int, ...]  # channel-first
    interleaved: bool  # MetaIO and NRRD keep a pixel's components together; NIfTI each component's volume whole
    nbytes: int  # the twin's size on disk (an upper bound for NIfTI)
    data: str  # the file holding the compressed stream: the entry itself, or a MetaImage's detached .zraw
    offset: int  # where the stream starts in it
    header: bytes | None  # the twin's header; ``None`` for NIfTI, whose twin is its stream inflated


def _root() -> Path:
    configured = os.environ.get(DIRECTORY_VARIABLE)
    if configured:
        return Path(configured)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "konfai" / "decompressed"


def _run_name(pid: int) -> str:
    """A run directory's name: the host as well as the process, for a cache on a disk several hosts share."""
    return f"{socket.gethostname()}-{pid}"


def _reap(root: Path) -> None:
    """Remove the directories killed runs of this host left: named after a process that no longer runs."""
    with contextlib.suppress(OSError), os.scandir(root) as listing:
        for entry in listing:
            host, _, pid = entry.name.rpartition("-")
            if host == socket.gethostname() and pid.isdigit() and _writer_is_dead(int(pid)):
                shutil.rmtree(entry.path, ignore_errors=True)


def _run_directory() -> Path:
    run = os.environ.get(_RUN_VARIABLE)
    if run:
        return Path(run)
    # A read outside a workflow run (a script over a Dataset, its loader workers): this process's own
    # directory, removed when it exits. A multiprocessing finalizer and not atexit, which a forked
    # multiprocessing child never runs before Python 3.13; a forked child names a directory of its own.
    root = _root()
    process_run = root / _run_name(os.getpid())
    if process_run not in _process_runs:
        _process_runs.add(process_run)
        _reap(root)
        multiprocessing.util.Finalize(
            None, shutil.rmtree, args=(process_run,), kwargs={"ignore_errors": True}, exitpriority=0
        )
    return process_run


@contextlib.contextmanager
def run_scope() -> Iterator[None]:
    """One workflow run's twins: a directory every process of the run shares (its ranks and their loader
    workers inherit it), removed when the run ends, however it ends. A directory a killed run of this host
    left, named after a process that no longer runs, is removed on the way in."""
    if os.environ.get(_RUN_VARIABLE):
        yield  # a run started from inside a run shares its directory
        return
    root = _root()
    _reap(root)
    run = root / _run_name(os.getpid())
    os.environ[_RUN_VARIABLE] = str(run)
    try:
        yield
    finally:
        os.environ.pop(_RUN_VARIABLE, None)
        shutil.rmtree(run, ignore_errors=True)


def layout(path: str) -> _Layout | None:
    """The twin ``path`` decompresses into, or ``None`` when it is not a compressed file served this way:
    a gzipped single-file NIfTI, a zlib MetaImage whose pixels are local or in one detached file, or a
    gzip NRRD that holds its own pixels."""
    if not path.lower().endswith(tuple(compressed for compressed, _ in _TWIN_SUFFIXES)):
        return None
    try:
        info = os.stat(path)
    except OSError:
        return None
    return _layout_at(path, (info.st_mtime_ns, info.st_size))


@functools.lru_cache(maxsize=4096)
def _layout_at(path: str, stamp: tuple[int, int]) -> _Layout | None:
    del stamp  # part of the key: a rewritten file gets a layout of its own
    suffix = next(compressed for compressed, _ in _TWIN_SUFFIXES if path.lower().endswith(compressed))
    try:
        return {".nii.gz": _nifti_layout, ".nrrd": _nrrd_layout}.get(suffix, _mha_layout)(path)
    except (OSError, RuntimeError, ValueError, KeyError):
        return None  # a file ITK cannot read either: its own route says what it is


def _nifti_layout(path: str) -> _Layout | None:
    if sitk is None:
        return None  # the read that needs it says so
    with open(path, "rb") as file:
        if file.read(2) != b"\x1f\x8b":
            return None
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    dtype = _sitk_component_dtypes().get(reader.GetPixelID())
    if dtype is None:
        return None
    shape = (reader.GetNumberOfComponents(), *reversed(reader.GetSize()))
    # 544 bytes: a NIfTI-2 header and its extension flag, the larger of the two versions. A stored
    # intensity scaling makes ITK report a wider type than the one stored: the bound only grows.
    return _Layout(shape, False, 544 + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize, path, 0, None)


def _mha_layout(path: str) -> _Layout | None:
    header = _mha_header(path)
    if header is None:
        return None
    fields, offset = header
    dtype = _MHA_DTYPES.get(fields.get("ElementType", ""))
    data_file = fields["ElementDataFile"]
    if (
        dtype is None
        or fields.get("CompressedData", "").lower() != "true"
        or fields.get("BinaryData", "").lower() != "true"
        or "HeaderSize" in fields
        or data_file == "LIST"  # one file per slice
        or "%" in data_file  # a numbered file pattern
    ):
        return None
    data, start = (path, offset) if data_file == "LOCAL" else (os.path.join(os.path.dirname(path), data_file), 0)
    if not os.path.isfile(data):
        return None
    shape = (int(fields.get("ElementNumberOfChannels", "1")), *reversed([int(n) for n in fields["DimSize"].split()]))
    # The same header, uncompressed and local: MetaIO's own spelling, ElementDataFile last.
    twin_header = "".join(
        f"{key} = {'False' if key == 'CompressedData' else 'LOCAL' if key == 'ElementDataFile' else value}\n"
        for key, value in fields.items()
        if key != "CompressedDataSize"
    ).encode("latin-1")
    nbytes = len(twin_header) + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    return _Layout(shape, True, nbytes, data, start, twin_header)


def _nrrd_layout(path: str) -> _Layout | None:
    nrrd = _nrrd_header(path)
    if nrrd is None or nrrd.encoding not in ("gzip", "gz"):
        return None
    with open(path, "rb") as file:
        head = file.read(nrrd.offset)
    # The same header, its encoding spelled raw.
    start, stop = nrrd.encoding_span
    twin_header = head[:start] + b"raw" + head[stop:]
    nbytes = len(twin_header) + int(np.prod(nrrd.shape, dtype=np.int64)) * nrrd.dtype.itemsize
    return _Layout(nrrd.shape, True, nbytes, path, nrrd.offset, twin_header)


@functools.cache
def _writable(root: Path) -> bool:
    """Whether twins can be made under ``root``: its nearest existing ancestor is a directory this
    process may write. Asked once per root, and it creates nothing."""
    existing = next(part for part in (root, *root.parents) if part.exists())
    return existing.is_dir() and os.access(existing, os.W_OK | os.X_OK)


def servable(path: str) -> _Layout | None:
    """The twin a region read of ``path`` would be served from, or ``None`` when it would decode
    ``path`` instead: not a file served this way, a cache directory that cannot be written, or a
    disk that has not the room :func:`twin` will ask for, so a plan never counts on a twin it will
    not get."""
    found = layout(path)
    if found is None or not _writable(_root()):
        return None
    return found if _fits(_run_directory(), found.nbytes) else None


def _case_prefix(case: str) -> str:
    """What the twins of ``case`` are named after in the run directory: a digest, whatever the name."""
    return hashlib.sha256(case.encode()).hexdigest()[:16] + "-"


def _target(path: str, case: str) -> tuple[Path, _Layout] | None:
    """Where the twin of ``path``, read for ``case``, is in this run, keyed by the path, size and mtime
    of the file and of its detached pixels: a file changed since is decompressed again."""
    found = layout(path)
    if found is None:
        return None
    identity = [os.path.abspath(path)]
    for part in dict.fromkeys((path, found.data)):
        info = os.stat(part)
        identity += [str(info.st_size), str(info.st_mtime_ns)]
    digest = hashlib.sha256("\0".join(identity).encode()).hexdigest()[:32]
    suffix = next(twin for compressed, twin in _TWIN_SUFFIXES if path.lower().endswith(compressed))
    return _run_directory() / f"{_case_prefix(case)}{digest}{suffix}", found


def twin(path: str, case: str, make: bool = True) -> str | None:
    """The uncompressed twin of ``path``, read for ``case``, decompressed by the first reader of the run.

    ``None`` when ``path`` is not served this way, when its stream does not decode (the caller reads
    ``path`` and ITK says what is wrong with it), when the twin would take the run's twins past
    :data:`FREE_SPACE_SHARE` of their disk's free space or cannot be written, or when it is not there
    yet and ``make`` is false. Processes asking at once decompress it once: one holds the lock, the
    others wait and find it published, never half written.
    """
    try:
        target = _target(path, case)
    except OSError:
        return None
    if target is None:
        return None
    twin_path, found = target
    if twin_path.exists():
        return str(twin_path)
    if not make:
        return None
    try:
        twin_path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(twin_path.with_suffix(".lock")):
            if twin_path.exists():
                return str(twin_path)
            if not _fits(twin_path.parent, found.nbytes):
                _warn_untwinned(
                    path,
                    f"its uncompressed twin would take this run's twins past {FREE_SPACE_SHARE:.0%} of the free"
                    f" space on the disk of '{twin_path.parent.parent}'",
                )
                return None
            staging = DataStream.staging_path(str(twin_path))
            try:
                _decompress(found, staging)
            except (OSError, EOFError, zlib.error, ValueError):
                with contextlib.suppress(OSError):
                    os.unlink(staging)
                return None
            try:
                os.replace(staging, twin_path)
            except OSError:
                with contextlib.suppress(OSError):
                    os.unlink(staging)
                # Windows refuses to replace a twin a reader holds open: another process published it, whole.
                if not twin_path.exists():
                    raise
    except OSError as error:
        _warn_untwinned(path, f"its uncompressed twin cannot be written under '{twin_path.parent}' ({error})")
        return None
    return str(twin_path)


def release(case: str) -> None:
    """Remove the twins read for ``case``: a one-pass reader leaving it, whichever root and stage read them.

    A reader still mapping one keeps its pages (POSIX) and a reader asking again decompresses it again;
    a twin the system refuses to remove is left to the end of the run. A lock and a staging file stay
    for the writer that may hold them.
    """
    prefix, suffixes = _case_prefix(case), tuple(twin for _, twin in _TWIN_SUFFIXES)
    with contextlib.suppress(OSError), os.scandir(_run_directory()) as listing:
        for entry in listing:
            if entry.name.startswith(prefix) and entry.name.endswith(suffixes):
                with contextlib.suppress(OSError):
                    os.unlink(entry.path)


@contextlib.contextmanager
def _exclusive(lock: Path) -> Iterator[None]:
    """One process at a time past here for ``lock``'s twin; the lock goes with the process holding it."""
    with open(lock, "a") as handle:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        # ponytail: no lock on Windows, where two processes may both decompress one entry and either
        # publish stands (each is whole); msvcrt.locking is the upgrade.
        yield


def _fits(run: Path, needed: int) -> bool:
    """Whether ``needed`` more bytes keep the run's twins within their share of the disk: the share is of
    what is free, counting what the twins already hold as free."""
    held = 0
    if run.is_dir():
        with os.scandir(run) as listing:
            for entry in listing:
                with contextlib.suppress(OSError):
                    held += entry.stat().st_size
    disk = next(directory for directory in (run, *run.parents) if directory.exists())
    return held + needed <= FREE_SPACE_SHARE * (shutil.disk_usage(disk).free + held)


def _warn_untwinned(path: str, reason: str) -> None:
    global _refusal_warned
    if _refusal_warned:
        return
    _refusal_warned = True
    warnings.warn(
        f"'{path}' is read from its compressed stream, every region decoding it again: {reason}. Point"
        f" {DIRECTORY_VARIABLE} at a writable directory on a disk with more room. Warned once per process.",
        KonfAIWarning,
        stacklevel=4,
    )


def _decompress(found: _Layout, staging: str) -> None:
    """Write the twin under ``staging``, ``_CHUNK`` bytes at a time: the volume is never held."""
    with open(found.data, "rb") as source, open(staging, "wb") as target:
        if found.header is None:
            with gzip.GzipFile(fileobj=source) as stream:
                shutil.copyfileobj(stream, target, _CHUNK)
            return
        target.write(found.header)
        source.seek(found.offset)
        inflate = zlib.decompressobj(zlib.MAX_WBITS | 32)  # zlib or gzip framing, both of which MetaIO reads
        written = 0
        while not inflate.eof and (chunk := source.read(_CHUNK)):
            while chunk:
                out = inflate.decompress(chunk, _CHUNK)
                target.write(out)
                written += len(out)
                chunk = inflate.unconsumed_tail
        expected = found.nbytes - len(found.header)
        if not inflate.eof or written != expected:
            raise ValueError(f"'{found.data}' inflates to {written} bytes where its header announces {expected}.")
