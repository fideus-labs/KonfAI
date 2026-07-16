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

"""FastAPI server exposing KonfAI Apps as remote asynchronous jobs."""

import asyncio
import hmac
import json
import os
import shutil
import signal
import subprocess  # nosec B404
import tempfile
import time
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Annotated, TypeVar

import konfai
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from konfai.utils.errors import AppMetadataError, AppRepositoryError

from .app_repository import get_app_repository_info

MAX_ACTIVE_JOBS = 32

_APPS_CONFIG: dict = {}
if "KONFAI_APPS_CONFIG" in os.environ:
    _APPS_CONFIG = json.loads(os.environ["KONFAI_APPS_CONFIG"])

_APPS: dict[str, list] = _APPS_CONFIG.get("apps", [])

security = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """
    Enforce bearer-token authentication when server auth is enabled.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials | None, optional
        Parsed authorization header provided by FastAPI.

    Raises
    ------
    HTTPException
        If authentication is required but the request is missing a valid token.
    """
    expected = os.environ.get("KONFAI_API_TOKEN")
    if not expected:
        return

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    if not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="Invalid token")


protected = APIRouter(dependencies=[Depends(require_token)])


def initialize_gpu_semaphores() -> None:
    """Refresh the GPU semaphore registry from the currently visible devices."""
    SERVER_STATE.gpu_semaphores.clear()
    devices_index, _ = konfai.get_available_devices()
    for i in devices_index:
        SERVER_STATE.gpu_semaphores[int(i)] = asyncio.Semaphore(1)


def reset_sse_state() -> None:
    """Reset in-memory SSE accounting used for log streaming admission control."""
    SERVER_STATE.active_sse_global = 0
    SERVER_STATE.active_sse_jobs.clear()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and clear process-wide server state across FastAPI lifecycles."""
    initialize_gpu_semaphores()
    reset_sse_state()
    SERVER_STATE.loop = asyncio.get_running_loop()
    try:
        yield
    finally:
        SERVER_STATE.loop = None
        SERVER_STATE.gpu_semaphores.clear()
        reset_sse_state()


app = FastAPI(lifespan=lifespan)


MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
MAX_TOTAL_BYTES = 6 * 1024 * 1024 * 1024  # 6GB

_T = TypeVar("_T")


def split_into_groups(items: list[_T], groups: str) -> list[list[_T]]:
    """Re-split a flat, ordered list into consecutive groups from a size CSV."""
    out: list[list[_T]] = []
    idx = 0
    for size in (int(s) for s in groups.split(",")):
        out.append(list(items[idx : idx + size]))
        idx += size
    return out


def save_upload_groups(
    files: list[UploadFile],
    groups: str,
    base: Path,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> list[list[Path]]:
    """Persist uploaded files into one ``g{i}`` subdirectory per declared group."""
    saved: list[list[Path]] = []
    total = 0
    for i, group in enumerate(split_into_groups(files, groups)):
        paths = save_uploads(group, base / f"g{i}", max_file_bytes, max_total_bytes - total)
        saved.append(paths)
        # Charge the real content bytes: a directory-volume path is a directory whose inode st_size is a
        # few dozen bytes, not the extracted store -- summing st_size of the paths would let a group of
        # such volumes bypass max_total_bytes for every following group. Walk directories for their files.
        total += sum(f.stat().st_size for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())
    return saved


def save_uploads(
    files: list[UploadFile], dst: Path, max_file_bytes: int = MAX_FILE_BYTES, max_total_bytes: int = MAX_TOTAL_BYTES
) -> list[Path]:
    """
    Persist uploaded files into a job workspace while enforcing size limits.

    Parameters
    ----------
    files : list[UploadFile]
        Uploaded files received by FastAPI.
    dst : Path
        Destination directory.
    max_file_bytes : int, optional
        Maximum allowed size per file in bytes.
    max_total_bytes : int, optional
        Maximum allowed aggregate upload size in bytes.

    Returns
    -------
    list[Path]
        Absolute paths of the saved files.

    Raises
    ------
    HTTPException
        If a file or the full upload exceeds the configured limits.
    """
    dst.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    total = 0

    try:
        for f in files:
            if not f.filename:
                continue

            if f.filename.endswith(".konfaidir.zip"):
                # A directory volume (DICOM series / OME-Zarr store) uploaded whole -> re-extract it.
                volume_dir, written = _save_directory_volume(f, dst, max_file_bytes, max_total_bytes - total)
                total += written
                out.append(volume_dir)
                continue

            p = dst / Path(f.filename).name
            if p.exists():
                # Two uploads in the same group can share a basename (e.g. every case named the same);
                # disambiguate instead of silently overwriting, which would drop a case. Order is kept
                # (out is appended in upload order), so the group boundaries still map back correctly.
                base, dot, ext = Path(f.filename).name.partition(".")
                counter = 1
                while p.exists():
                    p = dst / f"{base}_{counter}{dot}{ext}"
                    counter += 1
            written = 0

            try:
                with p.open("wb") as w:
                    while True:
                        chunk = f.file.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        total += len(chunk)

                        if written > max_file_bytes:
                            raise HTTPException(413, f"File too large: {f.filename}")
                        if total > max_total_bytes:
                            raise HTTPException(413, "Total upload too large")

                        w.write(chunk)
            except Exception:
                p.unlink(missing_ok=True)
                raise

            out.append(p.resolve())
    except Exception:
        for path in out:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise

    return out


def extract_zip_safely(upload: UploadFile, dest: Path) -> Path:
    """
    Persist an uploaded zip archive and extract its contents under ``dest``.

    Every archive member is validated so its resolved destination stays inside
    ``dest``; absolute paths or ``..`` traversal entries are rejected before any
    extraction happens (zip-slip protection).

    Parameters
    ----------
    upload : UploadFile
        Uploaded zip archive.
    dest : Path
        Directory into which the archive is extracted (created if missing).

    Returns
    -------
    Path
        Resolved ``dest`` directory containing the extracted tree.

    Raises
    ------
    HTTPException
        If the payload is not a valid zip or contains an unsafe member path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()

    fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=dest.parent)
    archive_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as w:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                w.write(chunk)

        try:
            with zipfile.ZipFile(archive_path) as zf:
                for member in zf.namelist():
                    target = (dest / member).resolve()
                    if target != dest and dest not in target.parents:
                        raise HTTPException(400, f"Unsafe path in archive: {member}")
                zf.extractall(dest)
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "Uploaded dataset is not a valid zip archive") from exc
    finally:
        archive_path.unlink(missing_ok=True)

    return dest


def _extract_zip_bounded(archive_path: Path, dest: Path, max_file_bytes: int, max_total_bytes: int) -> int:
    """Extract every member of ``archive_path`` under ``dest``, streaming each member and enforcing the size
    limits on the EXTRACTED bytes.

    A zip bomb inflates far past its on-wire size, so a limit checked on the compressed upload is no limit at
    all -- the guard has to count what actually lands on disk. Each member is zip-slip checked before any
    write. Returns the total number of bytes written.
    """
    dest = dest.resolve()
    extracted = 0
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if target != dest and dest not in target.parents:
                raise HTTPException(400, f"Unsafe path in archive: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_written = 0
            with zf.open(info) as src, target.open("wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    member_written += len(chunk)
                    extracted += len(chunk)
                    if member_written > max_file_bytes:
                        raise HTTPException(413, f"File too large in archive: {info.filename}")
                    if extracted > max_total_bytes:
                        raise HTTPException(413, "Total upload too large")
                    out.write(chunk)
    return extracted


def _save_directory_volume(
    upload: UploadFile, dst: Path, max_file_bytes: int, max_total_bytes: int
) -> tuple[Path, int]:
    """Extract a ``<name>.konfaidir.zip`` upload into a single directory volume.

    The client zips a DICOM series / OME-Zarr store whole and marks the filename; here it is
    re-materialised as one directory (``dst/<name>``) so the KonfAI dataset backend reads it as a
    single volume rather than a bag of slices/chunks. Returns the directory and the extracted bytes.
    """
    dir_name = Path(upload.filename).name[: -len(".konfaidir.zip")] if upload.filename else ""
    if not dir_name or "/" in dir_name or "\\" in dir_name:
        raise HTTPException(400, f"Invalid directory-volume name: {upload.filename}")
    dst.mkdir(parents=True, exist_ok=True)
    dst_resolved = dst.resolve()
    target_dir = (dst_resolved / dir_name).resolve()
    # Require a fresh direct child of the group dir. This rejects '.' (target_dir would be dst itself, so
    # extraction overwrites sibling uploads and failure cleanup rmtrees the whole group) and '..'/traversal,
    # and refuses to reuse an existing entry (a name collision that would merge into another upload).
    if target_dir.parent != dst_resolved:
        raise HTTPException(400, f"Unsafe directory-volume name: {upload.filename}")
    if target_dir.exists():
        raise HTTPException(400, f"Directory-volume name collides with an existing upload: {upload.filename}")
    target_dir.mkdir(parents=True)

    fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=dst)
    archive_path = Path(tmp_name)
    written = 0
    try:
        compressed = 0
        with os.fdopen(fd, "wb") as w:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                compressed += len(chunk)
                # Bound the transient temp archive too: a legit store's compressed size never exceeds its
                # extracted size, so capping the download at the extracted budget rejects nothing valid.
                if compressed > max_total_bytes:
                    raise HTTPException(413, "Total upload too large")
                w.write(chunk)
        try:
            written = _extract_zip_bounded(archive_path, target_dir, max_file_bytes, max_total_bytes)
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, f"Uploaded volume is not a valid zip: {upload.filename}") from exc
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)
    return target_dir, written


@dataclass
class Job:
    """
    Represents a single KonfAI job executed on the server.

    A Job encapsulates:
    - its unique identifier (`job_id`)
    - its execution workspace (`run_dir`, `input_dir`, `output_dir`)
    - its lifecycle state (`status`, `error`, `created_at`, `finished_at`)
    - the underlying subprocess (`proc`)
    - GPU scheduling information (`requested_gpus`, `assigned_gpus`)
    - a bounded log queue used for SSE streaming (`log_q`)

    Lifecycle:
        queued   -> waiting (GPU acquisition)
        waiting  -> running
        running  -> done | error | killed

    The job owns a temporary directory which is deleted after completion,
    with a grace period allowing clients to download results.
    """

    job_id: str
    app_name: str
    run_dir: Path
    input_dir: Path
    output_dir: Path
    zip_path: Path
    log_q: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=10_000))
    status: str = "queued"  # queued|running|done|error
    error: str | None = None
    proc: subprocess.Popen | None = None

    requested_gpus: list[int] | None = None  # None => auto
    assigned_gpus: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


@dataclass
class AppServerState:
    """In-memory state container for the KonfAI app server runtime."""

    jobs: dict[str, Job] = field(default_factory=dict)
    gpu_semaphores: dict[int, asyncio.Semaphore] = field(default_factory=dict)
    active_sse_global: int = 0
    active_sse_jobs: set[str] = field(default_factory=set)
    active_sse_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    loop: asyncio.AbstractEventLoop | None = None


SERVER_STATE = AppServerState()
JOBS = SERVER_STATE.jobs
GPU_SEM = SERVER_STATE.gpu_semaphores


def active_job_count() -> int:
    """Return the number of jobs currently consuming scheduling capacity."""
    return sum(1 for job in SERVER_STATE.jobs.values() if job.status in {"queued", "waiting", "running"})


def get_job_or_404(job_id: str) -> Job:
    """Return a job from the registry or raise an HTTP 404."""
    job = SERVER_STATE.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


def cleanup_pending_job(job: Job) -> None:
    """Remove a job registry entry and its workspace before background execution starts."""
    SERVER_STATE.jobs.pop(job.job_id, None)
    shutil.rmtree(job.run_dir, ignore_errors=True)


async def acquire_gpus(job: Job, requested: list[int]) -> list[int]:
    """
    Acquire GPU resources for a job.

    Two modes are supported:

    - Auto mode (requested == []):
        The job waits until *any* GPU becomes available.
        The first free GPU semaphore is acquired.

    - Explicit mode (requested = [0, 2, ...]):
        The job waits until *all* specified GPUs are free.
        Semaphores are acquired in sorted order to avoid deadlocks.

    While waiting, the job status is set to "waiting".

    Parameters
    ----------
    job : Job
        The job requesting GPU resources.
    requested : list[int]
        List of requested GPU indices. An empty list means "auto".

    Returns
    -------
    list[int]
        The list of GPU indices that were successfully acquired.

    Raises
    ------
    HTTPException
        If a requested GPU id does not exist.
    """
    if len(requested) == 0:
        job.status = "waiting"
        while True:
            for gid, sem in SERVER_STATE.gpu_semaphores.items():
                if sem.locked() is False:
                    await sem.acquire()
                    return [gid]
            await asyncio.sleep(0.1)

    gpus = sorted({int(x) for x in requested})
    job.status = "waiting"
    for gid in gpus:
        if gid not in SERVER_STATE.gpu_semaphores:
            raise HTTPException(400, f"Unknown GPU id: {gid}")
    for gid in gpus:
        await SERVER_STATE.gpu_semaphores[gid].acquire()
    return gpus


def release_gpus(gpus: list[int]) -> None:
    """Release previously acquired GPU semaphores for a finished job."""
    for gid in gpus:
        sem = SERVER_STATE.gpu_semaphores.get(gid)
        if sem:
            try:
                sem.release()
            except ValueError:
                pass


MAX_SSE_GLOBAL = 200
SSE_HEARTBEAT_S = 15


async def sse_log_stream(job: Job):
    """
    Server-Sent Events (SSE) generator for streaming job logs.

    This stream is protected by:
    - a global limit on the number of concurrent streams
    - a per-job limit (only one active stream per job_id)
    - a heartbeat (SSE_HEARTBEAT_S) that keeps idle connections alive

    The stream yields:
        data: <log line>\n\n

    During quiet periods a ``: keepalive`` comment is emitted instead of a
    completion marker, so a long-running but silent job is never mistaken for
    a finished one.

    Termination conditions:
    - "__DONE__" log marker
    - "__ERROR__" marker
    - job reaching a terminal status (done/error/killed)
    - client disconnect

    Admission control and cleanup are guaranteed via a lock and a
    try/finally block, ensuring counters remain consistent even on
    client abort.

    Parameters
    ----------
    job : Job
        The job whose logs are streamed.
    """
    # ---- admission control ----
    async with SERVER_STATE.active_sse_lock:
        if SERVER_STATE.active_sse_global >= MAX_SSE_GLOBAL:
            raise HTTPException(429, "Too many log streams")
        if job.job_id in SERVER_STATE.active_sse_jobs:
            raise HTTPException(429, "Log stream already open for this job")
        SERVER_STATE.active_sse_global += 1
        SERVER_STATE.active_sse_jobs.add(job.job_id)

    try:
        yield f"data: [KonfAI-Apps] Remote job {job.job_id} log stream connected\n\n"

        while True:
            try:
                line = await asyncio.wait_for(job.log_q.get(), timeout=SSE_HEARTBEAT_S)
            except asyncio.TimeoutError:
                # Keep the connection alive during quiet periods without signalling
                # completion. Stop only once the job itself has terminated (guards
                # against a completion marker dropped from a saturated queue).
                yield ": keepalive\n\n"
                if job.status in {"done", "error", "killed"}:
                    break
                continue

            yield f"data: {line.strip()}\n\n"

            if line == "__DONE__" or line.startswith("__ERROR__"):
                break

    except asyncio.CancelledError:
        # client disconnected
        pass

    finally:
        # ---- release admission ----
        async with SERVER_STATE.active_sse_lock:
            SERVER_STATE.active_sse_jobs.discard(job.job_id)
            SERVER_STATE.active_sse_global -= 1


@protected.get("/health")
def health():
    """Return a lightweight health-check response for the app server."""
    return {"status": "ok"}


@protected.get("/available_devices")
def get_available_devices():
    """Return the GPU ids and names visible to the server process."""
    devices_index, devices_name = konfai.get_available_devices()
    return {"devices_index": devices_index, "devices_name": devices_name}


@protected.get("/ram")
def get_ram():
    """Return current server RAM usage in gigabytes."""
    used_gb, total_gb = konfai.get_ram()
    return {"used_gb": used_gb, "total_gb": total_gb}


@protected.get("/vram")
def get_vram(devices: list[int] = Query(...)):
    """
    Return current VRAM usage for the requested GPU ids.

    Parameters
    ----------
    devices : list[int]
        GPU ids to inspect.
    """
    used_gb, total_gb = konfai.get_vram(devices)
    return {"used_gb": used_gb, "total_gb": total_gb}


def _require_configured_app(app_id: str) -> None:
    """Reject app ids outside this server's configured allowlist.

    Without this guard the repository resolver would fetch arbitrary HuggingFace repos, read arbitrary
    local directories, or reach out to arbitrary remote servers (SSRF / exfiltration) for any token
    holder. The job endpoints already enforce the same allowlist (see the job wrapper).
    """
    if app_id not in _APPS:
        raise HTTPException(404, f"Unknown app '{app_id}'")


@protected.get("/repo_apps_list")
def get_apps():
    """Return the list of app identifiers configured for this server."""
    return {"apps": _APPS}


@protected.get("/repo_apps/{app_id:path}")
def get_app_info(app_id: str):
    """
    Return metadata and declared capabilities for a configured app.

    Parameters
    ----------
    app_id : str
        App identifier as exposed by the repository configuration.
    """
    _require_configured_app(app_id)

    try:
        app = get_app_repository_info(app_id, False)
    except (AppRepositoryError, AppMetadataError):
        return {
            "app": app_id,
            "available": False,
        }

    result = {
        "app": app_id,
        "available": True,
        "display_name": app.get_display_name(),
        "description": app.get_description(),
        "short_description": app.get_short_description(),
        "checkpoints_name": app.get_checkpoints_name(),
        "checkpoints_name_available": app.get_checkpoints_name_available(),
        "maximum_tta": app.get_maximum_tta(),
        "mc_dropout": app.get_mc_dropout(),
        "patch_size": app.get_patch_size(),
        "has_capabilities": app.has_capabilities(),
        "finetunable": app.is_finetunable(),
    }
    terminology = app.get_terminology()
    if terminology is not None:
        result["terminology"] = {str(k): asdict(v) for k, v in terminology.items()}

    result["inputs"] = {k: asdict(v) for k, v in app.get_inputs().items()}
    result["outputs"] = {k: asdict(v) for k, v in app.get_outputs().items()}

    result_tmp: dict[str, dict[str, dict]] = {}
    for key, entries in app.get_evaluations_inputs().items():
        by_file = result_tmp.setdefault(key.display_name, {})
        by_file[key.evaluation_file] = {name: asdict(entry) for name, entry in entries.items()}
    result["inputs_evaluations"] = result_tmp
    return result


@protected.get("/repo_apps_config/{app_id:path}")
def download_app_repository_configs(app_id: str, background_tasks: BackgroundTasks):
    """
    Download the configuration files of an app as a ZIP archive.
    """
    _require_configured_app(app_id)
    try:
        app = get_app_repository_info(app_id, False)
    except AppRepositoryError as exc:
        raise HTTPException(404, f"Unknown app '{app_id}'") from exc

    files = app.download_config_file()

    tmp_dir = Path(tempfile.mkdtemp(prefix="konfai_app_"))
    zip_root = tmp_dir / f"{app_id}_configs"
    zip_root.mkdir(parents=True, exist_ok=True)

    for path in files:
        shutil.copy2(path, zip_root / path.name)

    zip_path = tmp_dir / f"{app_id}_configs.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", zip_root)

    background_tasks.add_task(shutil.rmtree, tmp_dir, ignore_errors=True)

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"{app_id}_configs.zip",
    )


def q_put_drop_oldest(q: asyncio.Queue[str], item: str) -> None:
    """
    Push a log line into a bounded queue, dropping the oldest entry if needed.

    This keeps SSE log streaming responsive even when producers are temporarily
    faster than consumers.
    """
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass


def emit_log(job: Job, line: str) -> None:
    """
    Push a log line onto a job's SSE queue safely from any thread.

    ``asyncio.Queue`` is not thread-safe. Job execution runs in a worker thread
    (and ``kill_job`` in the request thread pool), so log lines are scheduled
    onto the event loop instead of touching the queue directly.
    """
    loop = SERVER_STATE.loop
    if loop is not None:
        loop.call_soon_threadsafe(q_put_drop_oldest, job.log_q, line)
    else:
        q_put_drop_oldest(job.log_q, line)


def _run_job_sync(
    job: Job,
    cmd: list[str],
):
    """
    Execute a job synchronously in a background thread.

    This function:
    - spawns the subprocess in its own process group
    - captures stdout/stderr line-by-line
    - pushes log lines into the job log queue
    - updates job status and error fields
    - packages the output directory into a zip file on success

    It is always executed inside `asyncio.to_thread(...)` to avoid
    blocking the event loop.

    Parameters
    ----------
    job : Job
        The job being executed.
    cmd : list[str]
        Fully resolved command line to execute.
    """
    job.status = "running"
    emit_log(job, f"[KonfAI-Apps] Starting job in: {job.run_dir}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(job.run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )  # nosec B603
        job.proc = proc
        if proc.stdout:
            for line in proc.stdout:
                emit_log(job, line.rstrip("\n"))

        rc = proc.wait()
        if rc != 0:
            job.status = "error"
            job.error = f"Subprocess failed (exit code {rc})"
            emit_log(job, f"__ERROR__ {job.error}")
            emit_log(job, "__DONE__")
            return

        zip_base = job.run_dir / "result"
        zip_file = shutil.make_archive(str(zip_base), "zip", root_dir=job.output_dir)
        job.zip_path = Path(zip_file)

        job.status = "done"
        emit_log(job, f"Result zip created: {job.zip_path}")
        emit_log(job, "__DONE__")

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        emit_log(job, f"__ERROR__ {job.error}")
        emit_log(job, "__DONE__")


async def start_job(job: Job, cmd: list[str], requested_gpus: list[int] | None):
    """
    Orchestrate the full lifecycle of a job:

    1. Acquire GPU resources (auto or explicit mode)
    2. Inject the assigned GPUs into the command line
    3. Run the job in a worker thread
    4. Release GPU resources (always, even on failure)
    5. Mark completion time
    6. Wait for a grace period
    7. Delete the temporary workspace
    8. Remove the job from the registry

    This function guarantees that:
    - GPUs are never leaked
    - temporary directories are cleaned
    - the job registry remains consistent

    Parameters
    ----------
    job : Job
        The job instance.
    cmd : list[str]
        Base command line (without GPU arguments).
    requested_gpus : list[int] | None
        Requested GPU ids, or None for CPU execution.
    """
    gpus = []
    try:
        cmd2 = list(cmd)
        if requested_gpus is not None:
            gpus = await acquire_gpus(job, requested_gpus)
            job.assigned_gpus = gpus

            q_put_drop_oldest(job.log_q, f"[KonfAI-Apps] Assigned GPUs: {gpus}")

            if gpus:
                cmd2 += ["--gpu"] + [str(i) for i in gpus]

        await asyncio.to_thread(_run_job_sync, job, cmd2)

    finally:
        if gpus:
            release_gpus(gpus)

        job.finished_at = time.time()

        await asyncio.sleep(120)

        shutil.rmtree(job.run_dir, ignore_errors=True)

        SERVER_STATE.jobs.pop(job.job_id, None)


def submit_job():
    """
    Decorator factory for job-submitting endpoints.

    The wrapper produced by this decorator:

    - Enforces a global limit on active jobs
    - Creates a unique temporary workspace
    - Registers a Job object in the global registry
    - Saves uploaded files with size quotas
    - Builds the final command line
    - Spawns the asynchronous job execution task
    - Returns job metadata (id, URLs)

    This provides a uniform execution model for all apps
    (infer, evaluate, pipeline, fine_tune, etc.).
    """

    def deco(fn: Callable[..., Awaitable[list[str]]]):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if active_job_count() >= MAX_ACTIVE_JOBS:
                raise HTTPException(429, "Server busy: too many active jobs")

            app_name = kwargs.get("app_name")
            if app_name not in _APPS:
                raise HTTPException(404, f"Unknown app '{app_name}'")

            job_id = uuid.uuid4().hex[:12]
            run_dir = Path(tempfile.mkdtemp(prefix=f"konfai_job_{job_id}_")).resolve()
            input_dir = run_dir / "Input"
            output_dir = run_dir / "Output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            job = Job(
                job_id=job_id,
                app_name=str(app_name),
                run_dir=run_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                zip_path=run_dir / "result.zip",
            )
            SERVER_STATE.jobs[job_id] = job

            try:
                inputs_upload_files = kwargs.get("inputs")
                inputs = None
                if inputs_upload_files:
                    inputs = save_upload_groups(
                        inputs_upload_files, kwargs.get("inputs_groups"), job.input_dir / "inputs"
                    )

                gt_upload_files = kwargs.get("gt")
                gt = None
                if gt_upload_files:
                    gt = save_upload_groups(gt_upload_files, kwargs.get("gt_groups"), job.input_dir / "gt")

                mask_upload_files = kwargs.get("mask")
                mask = None
                if mask_upload_files:
                    mask = save_upload_groups(mask_upload_files, kwargs.get("mask_groups"), job.input_dir / "mask")

                dataset_upload = kwargs.get("dataset")
                dataset_dir = None
                if dataset_upload is not None:
                    dataset_dir = extract_zip_safely(dataset_upload, job.input_dir / "dataset")

                gpu = kwargs.get("gpu")
                cpu = kwargs.get("cpu")
                quiet = kwargs.get("quiet")

                cmd = await fn(*args, **kwargs)
                cmd += ["--output", str(job.output_dir)]

                requested = [int(x) for x in (gpu or "").split(",") if x.strip()]
                if requested:
                    if not SERVER_STATE.gpu_semaphores:
                        raise HTTPException(503, "No GPU available on this server; retry with --cpu")
                    unknown = sorted({g for g in requested if g not in SERVER_STATE.gpu_semaphores})
                    if unknown:
                        raise HTTPException(400, f"Unknown GPU id(s): {unknown}")
                    gpu_list = requested
                else:
                    gpu_list = sorted(SERVER_STATE.gpu_semaphores) or None

                if gpu_list is None:
                    cmd += ["--cpu", str(cpu)]

                if inputs:
                    for group in inputs:
                        cmd += ["--inputs"] + [str(p) for p in group]

                if gt:
                    for group in gt:
                        cmd += ["--gt"] + [str(p) for p in group]

                if mask:
                    for group in mask:
                        cmd += ["--mask"] + [str(p) for p in group]

                if dataset_dir is not None:
                    cmd += ["--dataset", str(dataset_dir)]

                if quiet:
                    cmd += ["--quiet"]
            except Exception:
                cleanup_pending_job(job)
                raise

            asyncio.create_task(start_job(job, cmd, gpu_list))  # noqa: RUF006

            return {
                "job_id": job_id,
                "status_url": f"/jobs/{job_id}",
                "logs_url": f"/jobs/{job_id}/logs",
                "result_url": f"/jobs/{job_id}/result",
            }

        return wrapper

    return deco


@protected.post("/apps/{app_name:path}/infer")
@submit_job()
async def infer(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    inputs_groups: Annotated[str | None, Form()] = None,
    ensemble: Annotated[int, Form()] = 0,
    ensemble_models: Annotated[str, Form()] = "",  # CSV
    tta: Annotated[int, Form()] = 0,
    mc: Annotated[int, Form()] = 0,
    uncertainty: Annotated[bool, Form()] = False,
    prediction_file: Annotated[str, Form()] = "Prediction.yml",
    gpu: Annotated[str | None, Form()] = "",  # CSV "0,1"
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit an inference job.

    This endpoint runs `konfai-apps infer` on the server. The job is executed
    asynchronously and managed by the scheduler.

    Features:
    - Supports ensembling (by count or by explicit model list)
    - Test-time augmentation (TTA)
    - Monte-Carlo sampling (MC dropout)
    - GPU auto-allocation or explicit GPU selection
    - Isolated workspace and result packaging

    Parameters
    ----------
    app_name : str
        Name/path of the KonfAI application.
    inputs : list[UploadFile]
        Input images or volumes.
    ensemble : int
        Number of models to sample when no explicit list is provided.
    ensemble_models : str
        Comma-separated list of model identifiers.
    tta : int
        Number of test-time augmentations.
    mc : int
        Number of Monte-Carlo samples.
    prediction_file : str
        Prediction configuration file.
    gpu : str | None
        Comma-separated GPU ids, or None for auto mode.
    cpu : int
        CPU workers if GPU is not used.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata (job_id and URLs).
    """
    ensemble_models_list = [x.strip() for x in ensemble_models.split(",") if x.strip()]
    cmd = [
        "konfai-apps",
        "infer",
        app_name,
        "--tta",
        str(tta),
        "--mc",
        str(mc),
        "--prediction_file",
        prediction_file,
    ]
    if uncertainty:
        cmd += ["-uncertainty"]
    if len(ensemble_models_list) > 0:
        cmd += ["--ensemble_models", *ensemble_models_list]
    else:
        cmd += ["--ensemble", str(ensemble)]
    return cmd


@protected.post("/apps/{app_name:path}/evaluate")
@submit_job()
async def evaluate(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    gt: Annotated[list[UploadFile], File(...)],
    mask: Annotated[list[UploadFile], File(...)] = [],
    inputs_groups: Annotated[str | None, Form()] = None,
    gt_groups: Annotated[str | None, Form()] = None,
    mask_groups: Annotated[str | None, Form()] = None,
    evaluation_file: Annotated[str, Form()] = "Evaluation.yml",
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit an evaluation job.

    Runs `konfai-apps eval` to compute metrics between predictions and
    ground-truth data.

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Predicted outputs or inputs to evaluate.
    gt : list[UploadFile]
        Ground-truth data.
    mask : list[UploadFile]
        Optional masks.
    evaluation_file : str
        Evaluation configuration file.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    cmd = [
        "konfai-apps",
        "eval",
        app_name,
        "--evaluation_file",
        evaluation_file,
    ]
    return cmd


@protected.post("/apps/{app_name:path}/uncertainty")
@submit_job()
async def uncertainty(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    inputs_groups: Annotated[str | None, Form()] = None,
    uncertainty_file: Annotated[str, Form()] = "Uncertainty.yml",
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit an uncertainty estimation job.

    Executes `konfai-apps uncertainty` to compute uncertainty maps or
    statistics from model outputs.

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Input data.
    uncertainty_file : str
        Uncertainty configuration file.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    cmd = [
        "konfai-apps",
        "uncertainty",
        app_name,
        "--uncertainty_file",
        uncertainty_file,
    ]
    return cmd


@protected.post("/apps/{app_name:path}/pipeline")
@submit_job()
async def pipeline(
    app_name: str,
    inputs: Annotated[list[UploadFile], File(...)],
    gt: Annotated[list[UploadFile], File(...)],
    inputs_groups: Annotated[str | None, Form()] = None,
    gt_groups: Annotated[str | None, Form()] = None,
    mask_groups: Annotated[str | None, Form()] = None,
    ensemble: Annotated[int, Form()] = 0,
    ensemble_models: Annotated[str, Form()] = "",
    tta: Annotated[int, Form()] = 0,
    mc: Annotated[int, Form()] = 0,
    prediction_file: Annotated[str, Form()] = "Prediction.yml",
    mask: Annotated[list[UploadFile] | None, File(...)] = None,
    evaluation_file: Annotated[str, Form()] = "Evaluation.yml",
    uncertainty: Annotated[bool, Form()] = True,
    uncertainty_file: Annotated[str, Form()] = "Uncertainty.yml",
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit a full end-to-end pipeline job.

    This endpoint chains:
        inference → evaluation → (optional) uncertainty estimation

    It provides a single entry point for complete experiments with
    ensembling, TTA, MC sampling, and metric computation.

    Parameters
    ----------
    app_name : str
        Application name.
    inputs : list[UploadFile]
        Input data.
    gt : list[UploadFile]
        Ground-truth data.
    ensemble : int
        Ensemble size.
    ensemble_models : str
        Explicit ensemble model list.
    tta : int
        Test-time augmentation count.
    mc : int
        Monte-Carlo samples.
    prediction_file : str
        Prediction configuration.
    mask : list[UploadFile] | None
        Optional masks.
    evaluation_file : str
        Evaluation configuration.
    uncertainty : bool
        Enable uncertainty stage.
    uncertainty_file : str
        Uncertainty configuration.
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    ensemble_models_list = [x.strip() for x in ensemble_models.split(",") if x.strip()]
    cmd = [
        "konfai-apps",
        "pipeline",
        app_name,
        "--ensemble",
        str(ensemble),
        "--tta",
        str(tta),
        "--mc",
        str(mc),
        "--prediction_file",
        prediction_file,
        "--evaluation_file",
        evaluation_file,
        "--uncertainty_file",
        uncertainty_file,
    ]
    if uncertainty:
        cmd += ["-uncertainty"]
    if ensemble_models_list:
        cmd += ["--ensemble_models", *ensemble_models_list]
    return cmd


@protected.post("/apps/{app_name:path}/fine_tune")
@submit_job()
async def fine_tune(
    app_name: str,
    dataset: Annotated[UploadFile, File(...)],
    name: Annotated[str, Form()] = "Finetune",
    epochs: Annotated[int, Form()] = 10,
    it_validation: Annotated[int, Form()] = 1000,
    models: Annotated[str, Form()] = "",  # CSV
    config_file: Annotated[str, Form()] = "Config.yml",
    lr: Annotated[float | None, Form()] = None,
    gpu: Annotated[str | None, Form()] = "",
    cpu: Annotated[int, Form()] = 1,
    quiet: Annotated[bool, Form()] = False,
):
    """
    Submit a fine-tuning (training) job.

    Executes `konfai-apps fine-tune` to adapt a model on the provided dataset.
    The job is fully managed by the server (workspace, GPUs, logs, cleanup).

    Parameters
    ----------
    app_name : str
        Application name.
    dataset : UploadFile
        Training dataset directory packaged as a single zip archive; the server
        extracts it and passes the resulting directory to ``--dataset``.
    name : str
        Run name.
    epochs : int
        Number of training epochs.
    it_validation : int
        Validation interval.
    models : str
        Comma-separated checkpoint name(s) to fine-tune (empty = first available).
    config_file : str
        Training configuration file.
    lr : float | None
        Learning-rate override forwarded to fine-tuning (``None`` resumes the
        checkpoint learning rate).
    gpu : str | None
        GPU selection or auto mode.
    cpu : int
        CPU workers.
    quiet : bool
        Reduce verbosity.

    Returns
    -------
    dict
        Job metadata.
    """
    cmd = [
        "konfai-apps",
        "fine-tune",
        app_name,
        name,
        "--config",
        config_file,
        "--epochs",
        str(epochs),
        "--it_validation",
        str(it_validation),
    ]
    models_list = [x.strip() for x in models.split(",") if x.strip()]
    if models_list:
        cmd += ["--models", *models_list]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    return cmd


@protected.get("/jobs/{job_id}")
def job_status(job_id: str):
    """
    Query the current status of a job.

    This endpoint provides lightweight polling access to the job state
    without streaming logs.

    Parameters
    ----------
    job_id : str
        Identifier of the job.

    Returns
    -------
    dict
        Dictionary containing the job identifier, status, and optional error
        message.

    Raises
    ------
    HTTPException
        If the job_id is unknown.
    """
    job = get_job_or_404(job_id)
    return {"job_id": job.job_id, "status": job.status, "error": job.error}


@protected.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str):
    """
    Stream the logs of a job using Server-Sent Events (SSE).

    The stream is protected by:
    - a global limit on concurrent streams
    - a per-job limit (only one active stream per job)
    - a hard time-to-live (TTL)

    Log lines are emitted as SSE events of the form:

        data: <log line>

    The stream terminates when:
    - the job finishes (`__DONE__`)
    - an error is reported (`__ERROR__`)
    - the TTL expires
    - the client disconnects

    Parameters
    ----------
    job_id : str
        Identifier of the job.

    Returns
    -------
    StreamingResponse
        SSE stream of log lines.

    Raises
    ------
    HTTPException
        If the job does not exist or stream limits are exceeded.
    """
    job = get_job_or_404(job_id)
    return StreamingResponse(sse_log_stream(job), media_type="text/event-stream")


@protected.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """
    Retrieve the result archive of a completed job.

    Behavior:

    - If the job is still running or waiting, returns HTTP 202.
    - If the job failed, returns HTTP 500 with error information.
    - If the job succeeded, returns the result ZIP archive.

    Parameters
    ----------
    job_id : str
        Identifier of the job.

    Returns
    -------
    FileResponse | JSONResponse
        ZIP archive when the job is done, or a JSON status response otherwise.

    Raises
    ------
    HTTPException
        If the job_id is unknown.
    """
    job = get_job_or_404(job_id)

    if job.status == "error":
        return JSONResponse(status_code=500, content={"job_id": job.job_id, "status": job.status, "error": job.error})

    if job.status != "done" or not job.zip_path.exists():
        return JSONResponse(status_code=202, content={"job_id": job.job_id, "status": job.status})

    return FileResponse(str(job.zip_path), media_type="application/zip", filename="result.zip")


@protected.post("/jobs/{job_id}/kill")
def kill_job(job_id: str):
    """
    Terminate a running job.

    The job subprocess is started in its own process group.
    This endpoint:

    1. Sends SIGTERM to the entire process group
    2. Waits briefly for graceful shutdown
    3. Sends SIGKILL if the process is still alive
    4. Marks the job as killed
    5. Emits termination markers in the log stream

    This guarantees that:
    - All child processes are terminated
    - GPU resources are eventually released
    - Clients observing logs are notified

    Parameters
    ----------
    job_id : str
        Identifier of the job to terminate.
    """
    job = get_job_or_404(job_id)

    proc = job.proc
    if proc is None or proc.poll() is not None:
        return {"job_id": job.job_id, "status": job.status, "message": "Job not running"}

    try:
        os.killpg(proc.pid, signal.SIGTERM)

        # brief grace period, then SIGKILL
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

        job.status = "killed"
        job.error = "Killed by user"
        emit_log(job, "__ERROR__ Killed by user")
        emit_log(job, "__DONE__")

        return {"job_id": job.job_id, "status": "killed", "message": "Kill requested"}

    except Exception as e:
        raise HTTPException(500, f"Failed to kill job: {e}") from e


app.include_router(protected)
