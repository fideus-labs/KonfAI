# SPDX-License-Identifier: Apache-2.0
"""TensorBoard integration: lazy per-frame image extraction from a run's event files, and one
lazily-started ``tensorboard`` subprocess per task."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from .paths import _jail, _sane_session, _workspace_root

router = APIRouter()

_TB_IMAGE_HISTORY = 30  # steps kept per image tag (the slider's range); frames are fetched lazily
_TB_SERVERS: dict[str, dict[str, Any]] = {}  # session -> {proc, url}: one lazily-started TensorBoard per task


def _tb_image_dir(session: str, base: str = "") -> Path | None:
    """The TensorBoard image dir for a run, from its session-relative ``base`` ("<base>/tb", sibling of the
    run's log_0.txt). ``base`` is "Statistics/<run>" (session-root) or "<app_output>-<hash>/Statistics/<run>"
    (isolated app run), so an isolated run's images resolve under its own output subtree. Jailed: ``base``
    may contain '/', never '..'. Without a base, the most recently written tb dir anywhere under the session."""
    root = (_workspace_root() / "sessions" / session).resolve()
    if base:  # a provided base is authoritative — a traversal attempt is rejected, never fallen back on
        if ".." in Path(base).parts:
            return None
        one = _jail(root, f"{base}/tb")
        return one if one is not None and one.is_dir() else None
    tb_dirs = sorted(root.glob("**/tb"), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []
    return tb_dirs[0] if tb_dirs else None


def _tb_accumulator(tb_dir: Path, history: int) -> Any | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import IMAGES, EventAccumulator
    except Exception:
        return None
    try:
        acc = EventAccumulator(str(tb_dir), size_guidance={IMAGES: history})
        acc.Reload()
        return acc
    except Exception:
        return None


def _tb_previews(session: str, base: str = "", limit: int = 24) -> list[dict[str, Any]]:
    """A manifest of a run's TensorBoard image tags: [{label, steps:[…], step}] — the step history per
    output (Training/CT, Validation/MR, …), with NO image bytes (those are fetched per step, lazily, so a
    long history never bloats the payload). Empty if that run has no images yet."""
    tb_dir = _tb_image_dir(session, base)
    if tb_dir is None:
        return []
    acc = _tb_accumulator(tb_dir, _TB_IMAGE_HISTORY)
    if acc is None:
        return []
    out: list[dict[str, Any]] = []
    for tag in acc.Tags().get("images", [])[:limit]:
        steps = [image.step for image in acc.Images(tag)]
        if steps:
            out.append({"label": tag, "steps": steps, "step": steps[-1]})
    return out


def _tb_image_bytes(session: str, base: str, tag: str, step: int) -> bytes | None:
    """The encoded PNG for one (base, tag, step) TensorBoard image, or the latest for that tag if the step is
    gone from the kept window."""
    tb_dir = _tb_image_dir(session, base)
    if tb_dir is None:
        return None
    acc = _tb_accumulator(tb_dir, _TB_IMAGE_HISTORY)
    if acc is None:
        return None
    try:
        images = acc.Images(tag)
    except KeyError:
        return None
    if not images:
        return None
    for image in images:
        if image.step == step:
            return bytes(image.encoded_image_string)
    return bytes(images[-1].encoded_image_string)


@router.get("/api/previews")
async def previews(session: str = Query("default"), base: str = Query("")) -> dict[str, list[dict[str, Any]]]:
    return {"previews": await asyncio.to_thread(_tb_previews, _sane_session(session), base)}


@router.get("/api/preview_image")
async def preview_image(
    session: str = Query("default"), base: str = Query(""), tag: str = Query(...), step: int = Query(...)
) -> Response:
    """One TensorBoard image montage (PNG) for a (base, tag, step) — the lazy per-frame fetch behind the slider."""
    data = await asyncio.to_thread(_tb_image_bytes, _sane_session(session), base, tag, step)
    if data is None:
        raise HTTPException(404, "image not found")
    return Response(content=data, media_type="image/png")


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@router.get("/api/tensorboard")
async def tensorboard_link(session: str = Query("default")) -> dict[str, Any]:
    """Ensure a TensorBoard server is running for the task's Statistics dir and return its URL — the full
    TensorBoard UI (scalars, images, histograms) for every run, alongside Studio's own live feed. One
    server per task, reused while alive, bound to 127.0.0.1 (a remote deployment must proxy the port)."""
    name = _sane_session(session)
    live = _TB_SERVERS.get(name)
    if live and live["proc"].poll() is None:
        return {"ok": True, "url": live["url"]}
    # The session root, not Statistics/: TensorBoard recurses, so it surfaces every "*/tb" at any depth —
    # session-root runs AND isolated app-output runs (<app_output>-<hash>/Statistics/<run>/tb) alike.
    session_root = _workspace_root() / "sessions" / name
    if not session_root.is_dir() or not any(session_root.glob("**/tb")):
        return {"ok": False, "detail": "no TensorBoard events yet — run a training first"}
    # Look next to the running interpreter first: the server is launched as a console script, so the env's
    # bin dir may not be on PATH and shutil.which alone would miss it.
    sibling = Path(sys.executable).with_name("tensorboard")
    binary = str(sibling) if sibling.exists() else shutil.which("tensorboard")
    if not binary:
        return {"ok": False, "detail": "tensorboard is not installed (pip install konfai[tensorboard])"}
    port = _free_port()
    proc = subprocess.Popen(
        [binary, "--logdir", str(session_root), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    _TB_SERVERS[name] = {"proc": proc, "url": url}
    return {"ok": True, "url": url}
