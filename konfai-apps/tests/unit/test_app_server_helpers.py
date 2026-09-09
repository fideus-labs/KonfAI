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

import asyncio
import io
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

# Before importing anything that pulls in FastAPI (app_server, fastapi itself): a module-level import runs
# at collection, so pytestmark would skip too late and collection would error when FastAPI is absent.
pytest.importorskip("fastapi")

import konfai_apps.app_server as app_server
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


def _make_job(job_id: str, status: str = "queued") -> app_server.Job:
    job = app_server.Job(
        job_id=job_id,
        app_name="demo",
        run_dir=Path(f"/tmp/{job_id}"),
        input_dir=Path(f"/tmp/{job_id}_in"),
        output_dir=Path(f"/tmp/{job_id}_out"),
        zip_path=Path(f"/tmp/{job_id}.zip"),
    )
    job.status = status
    return job


def test_require_token_accepts_missing_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    assert app_server.require_token(None) is None


def test_require_token_rejects_invalid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KONFAI_API_TOKEN", "secret")

    with pytest.raises(HTTPException, match="Invalid token"):
        app_server.require_token(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="wrong",
            )
        )


def test_save_uploads_enforces_limits(tmp_path: Path) -> None:
    payload = io.BytesIO(b"abc")
    upload = SimpleNamespace(filename="sample.bin", file=payload)

    written = app_server.save_uploads(
        [upload],
        tmp_path,
        max_file_bytes=8,
        max_total_bytes=8,
    )
    assert written[0].read_bytes() == b"abc"

    too_large = SimpleNamespace(
        filename="large.bin",
        file=io.BytesIO(b"0123456789"),
    )
    with pytest.raises(HTTPException, match="File too large"):
        app_server.save_uploads(
            [too_large],
            tmp_path / "overflow",
            max_file_bytes=4,
            max_total_bytes=32,
        )


def test_save_uploads_cleans_previous_files_on_total_limit(tmp_path: Path) -> None:
    uploads = [
        SimpleNamespace(filename="first.bin", file=io.BytesIO(b"1234")),
        SimpleNamespace(filename="second.bin", file=io.BytesIO(b"5678")),
    ]

    with pytest.raises(HTTPException, match="Total upload too large"):
        app_server.save_uploads(
            uploads,
            tmp_path / "overflow",
            max_file_bytes=8,
            max_total_bytes=6,
        )

    assert list((tmp_path / "overflow").glob("*")) == []


def test_app_lifespan_initializes_gpu_semaphores(monkeypatch: pytest.MonkeyPatch) -> None:
    old = app_server.GPU_SEM.copy()
    app_server.GPU_SEM.clear()
    monkeypatch.setattr(app_server.konfai, "get_available_devices", lambda: ([0, 2], ["gpu0", "gpu2"]))

    async def scenario() -> None:
        async with app_server.lifespan(app_server.app):
            assert sorted(app_server.GPU_SEM) == [0, 2]

    try:
        asyncio.run(scenario())
        assert app_server.GPU_SEM == {}
    finally:
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update(old)


def test_server_state_keeps_backward_compatible_aliases() -> None:
    assert app_server.GPU_SEM is app_server.SERVER_STATE.gpu_semaphores
    assert app_server.JOBS is app_server.SERVER_STATE.jobs


def test_active_job_count_ignores_finished_jobs() -> None:
    old_jobs = dict(app_server.JOBS)
    app_server.JOBS.clear()
    app_server.JOBS.update(
        {
            "queued": _make_job("queued", "queued"),
            "running": _make_job("running", "running"),
            "done": _make_job("done", "done"),
        }
    )
    try:
        assert app_server.active_job_count() == 2
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(old_jobs)


def test_get_job_or_404_returns_registered_job() -> None:
    old_jobs = dict(app_server.JOBS)
    job = _make_job("known", "running")
    app_server.JOBS.clear()
    app_server.JOBS[job.job_id] = job
    try:
        assert app_server.get_job_or_404(job.job_id) is job
        with pytest.raises(HTTPException, match="Unknown job_id"):
            app_server.get_job_or_404("missing")
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(old_jobs)


def test_acquire_and_release_gpus_in_auto_mode() -> None:
    async def scenario() -> None:
        old = app_server.GPU_SEM.copy()
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update({0: asyncio.Semaphore(1)})
        try:
            job = app_server.Job(
                job_id="job",
                app_name="demo",
                run_dir=Path("/tmp/run"),
                input_dir=Path("/tmp/in"),
                output_dir=Path("/tmp/out"),
                zip_path=Path("/tmp/out.zip"),
            )
            acquired = await app_server.acquire_gpus(job, [])
            assert acquired == [0]
            assert job.status == "waiting"

            app_server.release_gpus(acquired)
            assert app_server.GPU_SEM[0].locked() is False
        finally:
            app_server.GPU_SEM.clear()
            app_server.GPU_SEM.update(old)

    asyncio.run(scenario())


def test_submit_job_cleans_workspace_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    @app_server.submit_job()
    async def failing_job(*args, **kwargs):
        raise HTTPException(400, "bad request")

    async def scenario() -> None:
        with pytest.raises(HTTPException, match="bad request"):
            await failing_job(
                app_name="demo",
                inputs=None,
                gt=None,
                mask=None,
                gpu=None,
                cpu=1,
                quiet=False,
            )

    asyncio.run(scenario())
    assert app_server.JOBS == {}
    assert run_dir.exists() is False


def test_submit_job_rejects_app_not_in_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(app_server, "_APPS", ["known"])

    calls: list[str] = []

    @app_server.submit_job()
    async def stub_job(*args, **kwargs):
        calls.append("ran")
        return ["konfai-apps", "infer", "evil"]

    async def scenario() -> None:
        with pytest.raises(HTTPException) as exc:
            await stub_job(
                app_name="evil",
                inputs=None,
                gt=None,
                mask=None,
                gpu=None,
                cpu=1,
                quiet=False,
            )
        assert exc.value.status_code == 404

    asyncio.run(scenario())
    # Rejected before any workspace is created or the command builder runs.
    assert calls == []
    assert run_dir.exists() is False
    assert app_server.JOBS == {}


def test_submit_job_errors_when_gpu_requested_but_none_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    old = app_server.GPU_SEM.copy()
    app_server.GPU_SEM.clear()

    @app_server.submit_job()
    async def stub_job(*args, **kwargs):
        return ["konfai-apps", "infer", "demo"]

    async def scenario() -> None:
        with pytest.raises(HTTPException) as exc:
            await stub_job(
                app_name="demo",
                inputs=None,
                gt=None,
                mask=None,
                gpu="0",
                cpu=1,
                quiet=False,
            )
        assert exc.value.status_code == 503

    try:
        asyncio.run(scenario())
    finally:
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update(old)

    assert app_server.JOBS == {}
    assert run_dir.exists() is False


def test_submit_job_rejects_unknown_gpu_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    old = app_server.GPU_SEM.copy()
    app_server.GPU_SEM.clear()
    app_server.GPU_SEM.update({0: asyncio.Semaphore(1)})

    @app_server.submit_job()
    async def stub_job(*args, **kwargs):
        return ["konfai-apps", "infer", "demo"]

    async def scenario() -> None:
        with pytest.raises(HTTPException) as exc:
            await stub_job(
                app_name="demo",
                inputs=None,
                gt=None,
                mask=None,
                gpu="5",
                cpu=1,
                quiet=False,
            )
        assert exc.value.status_code == 400

    try:
        asyncio.run(scenario())
    finally:
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update(old)

    assert app_server.JOBS == {}
    assert run_dir.exists() is False


def test_q_put_drop_oldest_drops_oldest_when_full() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
    app_server.q_put_drop_oldest(queue, "a")
    app_server.q_put_drop_oldest(queue, "b")
    app_server.q_put_drop_oldest(queue, "c")

    assert queue.get_nowait() == "b"
    assert queue.get_nowait() == "c"
    assert queue.empty()


def test_emit_log_without_loop_enqueues_directly() -> None:
    job = _make_job("nolog")
    app_server.SERVER_STATE.loop = None
    app_server.emit_log(job, "hello")
    assert job.log_q.get_nowait() == "hello"


def test_save_uploads_separates_categories(tmp_path: Path) -> None:
    same_name = "Volume.mha"
    input_upload = SimpleNamespace(filename=same_name, file=io.BytesIO(b"input"))
    gt_upload = SimpleNamespace(filename=same_name, file=io.BytesIO(b"gt"))

    inputs = app_server.save_uploads([input_upload], tmp_path / "inputs")
    gt = app_server.save_uploads([gt_upload], tmp_path / "gt")

    assert inputs[0] != gt[0]
    assert inputs[0].read_bytes() == b"input"
    assert gt[0].read_bytes() == b"gt"


def test_split_into_groups_respects_declared_sizes() -> None:
    assert app_server.split_into_groups(["f0", "f1", "f2"], "1,2") == [["f0"], ["f1", "f2"]]


def test_save_upload_groups_isolates_colliding_basenames(tmp_path: Path) -> None:
    first = SimpleNamespace(filename="ct.nii.gz", file=io.BytesIO(b"first"))
    second = SimpleNamespace(filename="ct.nii.gz", file=io.BytesIO(b"second"))

    saved = app_server.save_upload_groups([first, second], "1,1", tmp_path / "inputs")

    assert len(saved) == 2
    assert saved[0][0] != saved[1][0]
    assert saved[0][0].read_bytes() == b"first"
    assert saved[1][0].read_bytes() == b"second"
    assert saved[0][0].parent.name == "g0"
    assert saved[1][0].parent.name == "g1"


def _capture_submit_cmd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    submit_kwargs: dict,
) -> list[str]:
    run_dir = tmp_path / "job"
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    captured: dict[str, list[str]] = {}

    async def fake_start_job(job, cmd, gpus):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd

    monkeypatch.setattr(app_server, "start_job", fake_start_job)

    @app_server.submit_job()
    async def stub_job(*args, **kwargs):  # type: ignore[no-untyped-def]
        return ["konfai-apps", "infer", "demo"]

    old_jobs = dict(app_server.JOBS)

    async def scenario() -> None:
        await stub_job(app_name="demo", **submit_kwargs)
        await asyncio.sleep(0)

    try:
        asyncio.run(scenario())
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(old_jobs)

    return captured["cmd"]


def _capture_submit_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    submit_kwargs: dict,
    gpu_ids: list[int] | None = None,
) -> dict:
    run_dir = tmp_path / "job"
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    captured: dict = {}

    async def fake_start_job(job, cmd, gpus):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["gpus"] = gpus

    monkeypatch.setattr(app_server, "start_job", fake_start_job)

    @app_server.submit_job()
    async def stub_job(*args, **kwargs):  # type: ignore[no-untyped-def]
        return ["konfai-apps", "infer", "demo"]

    old_jobs = dict(app_server.JOBS)
    old_sem = app_server.GPU_SEM.copy()
    app_server.GPU_SEM.clear()
    for gid in gpu_ids or []:
        app_server.GPU_SEM[gid] = asyncio.Semaphore(1)

    async def scenario() -> None:
        await stub_job(app_name="demo", **submit_kwargs)
        await asyncio.sleep(0)

    try:
        asyncio.run(scenario())
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(old_jobs)
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update(old_sem)

    return captured


def test_submit_job_auto_mode_uses_all_available_gpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _capture_submit_gpu(
        monkeypatch,
        tmp_path,
        {"inputs": None, "gt": None, "mask": None, "gpu": "", "cpu": 1, "quiet": False},
        gpu_ids=[0, 1],
    )

    # Empty selection with GPUs present resolves to every available GPU, no 503.
    assert captured["gpus"] == [0, 1]
    assert "--cpu" not in captured["cmd"]


def test_submit_job_auto_mode_falls_back_to_cpu_without_gpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _capture_submit_gpu(
        monkeypatch,
        tmp_path,
        {"inputs": None, "gt": None, "mask": None, "gpu": "", "cpu": 3, "quiet": False},
        gpu_ids=None,
    )

    # Empty selection on a CPU-only server runs on CPU without raising 503.
    assert captured["gpus"] is None
    assert "--cpu" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--cpu") + 1] == "3"


def test_submit_job_explicit_mode_preserves_every_requested_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _capture_submit_gpu(
        monkeypatch,
        tmp_path,
        {"inputs": None, "gt": None, "mask": None, "gpu": "0,1", "cpu": 1, "quiet": False},
        gpu_ids=[0, 1],
    )

    # A CSV selection keeps every id instead of collapsing to the last one.
    assert captured["gpus"] == [0, 1]
    assert "--cpu" not in captured["cmd"]


def test_submit_job_emits_one_inputs_flag_per_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = [
        SimpleNamespace(filename="ct.nii.gz", file=io.BytesIO(b"c")),
        SimpleNamespace(filename="mr.nii.gz", file=io.BytesIO(b"m")),
    ]

    cmd = _capture_submit_cmd(
        monkeypatch,
        tmp_path,
        {
            "inputs": inputs,
            "inputs_groups": "1,1",
            "gt": None,
            "mask": None,
            "gpu": None,
            "cpu": 1,
            "quiet": False,
        },
    )

    assert cmd.count("--inputs") == 2
    input_paths = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--inputs"]
    assert Path(input_paths[0]).parent.name == "g0"
    assert Path(input_paths[1]).parent.name == "g1"
    assert [Path(p).name for p in input_paths] == ["ct.nii.gz", "mr.nii.gz"]


def test_submit_job_mono_input_emits_single_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = [SimpleNamespace(filename="ct.nii.gz", file=io.BytesIO(b"c"))]

    cmd = _capture_submit_cmd(
        monkeypatch,
        tmp_path,
        {
            "inputs": inputs,
            "inputs_groups": "1",
            "gt": None,
            "mask": None,
            "gpu": None,
            "cpu": 1,
            "quiet": False,
        },
    )

    assert cmd.count("--inputs") == 1
    input_path = Path(cmd[cmd.index("--inputs") + 1])
    assert input_path.parent.name == "g0"
    assert input_path.name == "ct.nii.gz"


def _make_dataset_upload(tmp_path: Path) -> SimpleNamespace:
    src = tmp_path / "src"
    (src / "P000").mkdir(parents=True)
    (src / "P000" / "Volume_0.mha").write_bytes(b"v0")
    (src / "P000" / "Volume_1.mha").write_bytes(b"v1")
    zip_path = shutil.make_archive(str(tmp_path / "dataset"), "zip", root_dir=str(src))
    return SimpleNamespace(filename="dataset.zip", file=io.BytesIO(Path(zip_path).read_bytes()))


def test_extract_zip_safely_reconstructs_tree(tmp_path: Path) -> None:
    upload = _make_dataset_upload(tmp_path)

    dest = app_server.extract_zip_safely(upload, tmp_path / "job" / "dataset")

    assert dest == (tmp_path / "job" / "dataset").resolve()
    assert (dest / "P000" / "Volume_0.mha").read_bytes() == b"v0"
    assert (dest / "P000" / "Volume_1.mha").read_bytes() == b"v1"
    # The temporary archive is removed once extraction completes.
    assert list(dest.parent.glob("*.zip")) == []


@pytest.mark.parametrize("evil_name", ["../evil.txt", "sub/../../evil.txt", "/abs/evil.txt"])
def test_extract_zip_safely_blocks_zip_slip(tmp_path: Path, evil_name: str) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr(evil_name, b"pwn")
    payload.seek(0)
    upload = SimpleNamespace(filename="dataset.zip", file=payload)

    with pytest.raises(HTTPException) as exc:
        app_server.extract_zip_safely(upload, tmp_path / "job" / "dataset")

    assert exc.value.status_code == 400
    assert not (tmp_path / "job" / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()
    assert not Path("/abs/evil.txt").exists()


def test_extract_zip_safely_rejects_non_zip_payload(tmp_path: Path) -> None:
    upload = SimpleNamespace(filename="dataset.zip", file=io.BytesIO(b"not a zip"))

    with pytest.raises(HTTPException) as exc:
        app_server.extract_zip_safely(upload, tmp_path / "job" / "dataset")

    assert exc.value.status_code == 400


def test_submit_job_extracts_dataset_and_appends_dataset_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_upload = _make_dataset_upload(tmp_path)

    cmd = _capture_submit_cmd(
        monkeypatch,
        tmp_path,
        {
            "inputs": None,
            "gt": None,
            "mask": None,
            "dataset": dataset_upload,
            "gpu": None,
            "cpu": 1,
            "quiet": False,
        },
    )

    assert "--inputs" not in cmd
    assert cmd.count("--dataset") == 1
    dataset_arg = Path(cmd[cmd.index("--dataset") + 1])
    assert dataset_arg.name == "dataset"
    assert (dataset_arg / "P000" / "Volume_0.mha").read_bytes() == b"v0"


def test_submit_job_without_dataset_emits_no_dataset_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = [SimpleNamespace(filename="ct.nii.gz", file=io.BytesIO(b"c"))]

    cmd = _capture_submit_cmd(
        monkeypatch,
        tmp_path,
        {
            "inputs": inputs,
            "inputs_groups": "1",
            "gt": None,
            "mask": None,
            "gpu": None,
            "cpu": 1,
            "quiet": False,
        },
    )

    assert "--dataset" not in cmd
    assert cmd.count("--inputs") == 1


def _make_directory_volume_upload(tmp_path: Path, filename: str, chunk: bytes = b"chunk") -> SimpleNamespace:
    store = tmp_path / "src_store"
    store.mkdir()
    (store / ".zgroup").write_text("{}", encoding="utf-8")
    (store / "0.0.0").write_bytes(chunk)
    zip_path = shutil.make_archive(str(tmp_path / "unit"), "zip", root_dir=str(store))
    upload = SimpleNamespace(filename=filename, file=io.BytesIO(Path(zip_path).read_bytes()))
    shutil.rmtree(store)
    return upload


def test_save_uploads_extracts_directory_volume(tmp_path: Path) -> None:
    upload = _make_directory_volume_upload(tmp_path, "unit_0.ome.zarr.konfaidir.zip")

    saved = app_server.save_uploads([upload], tmp_path / "inputs")

    assert len(saved) == 1
    volume = saved[0]
    assert volume.is_dir()
    assert volume.name == "unit_0.ome.zarr"
    assert (volume / ".zgroup").exists()
    assert (volume / "0.0.0").read_bytes() == b"chunk"


def test_save_upload_groups_maps_zip_unit_to_directory(tmp_path: Path) -> None:
    upload = _make_directory_volume_upload(tmp_path, "unit_0.konfaidir.zip")  # DICOM-style bare name

    groups = app_server.save_upload_groups([upload], "1", tmp_path / "inputs")

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert groups[0][0].is_dir()
    assert groups[0][0].name == "unit_0"


def test_save_directory_volume_rejects_dot_name_that_escapes_to_group_dir(tmp_path: Path) -> None:
    # "..konfaidir.zip" -> dir_name "." -> target_dir == the group dir itself. A containment check
    # permitting that lets extraction overwrite sibling uploads and failure cleanup rmtree the whole group.
    dst = tmp_path / "g0"
    real = SimpleNamespace(filename="case.mha", file=io.BytesIO(b"REAL"))

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("case.mha", b"CLOBBERED")
    payload.seek(0)
    evil = SimpleNamespace(filename="..konfaidir.zip", file=payload)

    with pytest.raises(HTTPException) as exc:
        app_server.save_uploads([real, evil], dst)
    assert exc.value.status_code == 400


def test_save_directory_volume_rejects_name_collision(tmp_path: Path) -> None:
    # Two directory volumes with the same target name must not merge into one another.
    first = _make_directory_volume_upload(tmp_path, "vol.konfaidir.zip")
    second = _make_directory_volume_upload(tmp_path, "vol.konfaidir.zip")

    with pytest.raises(HTTPException) as exc:
        app_server.save_uploads([first, second], tmp_path / "inputs")
    assert exc.value.status_code == 400


def _make_zip_bomb_upload(uncompressed_bytes: int) -> SimpleNamespace:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.bin", b"\0" * uncompressed_bytes)
    payload.seek(0)
    return SimpleNamespace(filename="store.konfaidir.zip", file=payload)


def test_save_directory_volume_bounds_extracted_bytes_not_compressed(tmp_path: Path) -> None:
    # A directory-volume zip that inflates far past its compressed size must be rejected on the EXTRACTED
    # bytes: a compressed-only limit lets the small upload sail under it and fill the disk.
    upload = _make_zip_bomb_upload(64 * 1024 * 1024)  # ~64MB extracted, a few KB compressed

    with pytest.raises(HTTPException) as exc:
        app_server.save_uploads([upload], tmp_path / "inputs", max_file_bytes=1024 * 1024, max_total_bytes=1024 * 1024)
    assert exc.value.status_code == 413
    # Nothing left behind: the partial extraction directory is removed on failure.
    assert not (tmp_path / "inputs" / "store").exists()


def test_save_upload_groups_charges_extracted_directory_bytes(tmp_path: Path) -> None:
    # A directory-volume path is a directory whose inode st_size is a fixed block (tens of bytes to ~4KB),
    # unrelated to the extracted store; charging that instead of the real content lets a group of such volumes
    # bypass max_total_bytes for every following group. The content is sized well above any directory inode so
    # the two behaviours diverge on every filesystem: the budget fits ONE store's real content but not two,
    # yet comfortably clears two directory inodes, so only content-charging rejects the second group.
    store_bytes = 64 * 1024
    first = _make_directory_volume_upload(tmp_path, "a.konfaidir.zip", chunk=b"\0" * store_bytes)
    second = _make_directory_volume_upload(tmp_path, "b.konfaidir.zip", chunk=b"\0" * store_bytes)

    with pytest.raises(HTTPException) as exc:
        app_server.save_upload_groups(
            [first, second], "1,1", tmp_path / "inputs", max_file_bytes=10**9, max_total_bytes=100_000
        )
    assert exc.value.status_code == 413


def test_get_app_info_reports_finetunable(monkeypatch: pytest.MonkeyPatch) -> None:
    # The server resolves the actual bundle, so it is the source of truth for remote clients:
    # the /repo_apps payload must carry the answer the remote adapter relays.
    class _FakeApp:
        def get_display_name(self) -> str:
            return "Demo"

        def get_description(self) -> str:
            return "demo"

        def get_short_description(self) -> str:
            return "demo"

        def get_checkpoints_name(self) -> list[str]:
            return ["m.pt"]

        def get_checkpoints_name_available(self) -> list[str]:
            return ["m.pt"]

        def get_maximum_tta(self) -> int:
            return 0

        def get_mc_dropout(self) -> int:
            return 0

        def get_patch_size(self) -> None:
            return None

        def has_capabilities(self) -> tuple[bool, bool, bool]:
            return (True, False, False)

        def is_finetunable(self) -> bool:
            return True

        def get_terminology(self) -> None:
            return None

        def get_inputs(self) -> dict:
            return {}

        def get_outputs(self) -> dict:
            return {}

        def get_evaluations_inputs(self) -> dict:
            return {}

    monkeypatch.setattr(app_server, "_APPS", ["demo/app"])
    monkeypatch.setattr(app_server, "get_app_repository_info", lambda *args, **kwargs: _FakeApp())

    result = app_server.get_app_info("demo/app")
    assert result["finetunable"] is True


def test_kill_job_cancels_a_job_still_waiting_for_a_gpu() -> None:
    """A waiting job has no process: kill_job answered "Job not running" and left it eligible to
    run when a GPU freed up."""
    job = _make_job("waiting-1", status="waiting")
    app_server.SERVER_STATE.jobs[job.job_id] = job
    try:
        response = app_server.kill_job(job.job_id)
    finally:
        app_server.SERVER_STATE.jobs.pop(job.job_id, None)
    assert response["status"] == "killed" and job.cancelled and job.status == "killed"


def test_a_cancel_while_waiting_on_the_second_gpu_gives_the_first_back() -> None:
    async def scenario() -> tuple[bool, bool]:
        app_server.SERVER_STATE.gpu_semaphores = {0: asyncio.Semaphore(1), 1: asyncio.Semaphore(1)}
        await app_server.SERVER_STATE.gpu_semaphores[1].acquire()  # GPU 1 is busy
        job = _make_job("explicit-1")
        task = asyncio.create_task(app_server.acquire_gpus(job, [0, 1]))
        await asyncio.sleep(0.25)  # GPU 0 acquired, waiting on GPU 1
        held_while_waiting = app_server.SERVER_STATE.gpu_semaphores[0].locked()
        job.cancelled = True
        with pytest.raises(app_server.JobCancelled):
            await task
        return held_while_waiting, app_server.SERVER_STATE.gpu_semaphores[0].locked()

    held_while_waiting, held_after = asyncio.run(scenario())
    assert held_while_waiting and not held_after
    app_server.SERVER_STATE.gpu_semaphores = {}


def test_start_job_skips_a_cancelled_job_and_keeps_the_result_for_the_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched: list[str] = []
    monkeypatch.setattr(app_server, "_run_job_sync", lambda job, cmd: launched.append(job.job_id))
    monkeypatch.setattr(app_server, "RESULT_RETENTION_S", 0.0)
    job = _make_job("cancelled-1", status="killed")
    job.cancelled = True
    job.run_dir = tmp_path / "run"
    job.run_dir.mkdir()
    app_server.SERVER_STATE.jobs[job.job_id] = job
    asyncio.run(app_server.start_job(job, ["konfai-apps"], None))
    assert launched == [] and job.status == "killed" and job.finished_at is not None
    assert not job.run_dir.exists() and job.job_id not in app_server.SERVER_STATE.jobs


def test_a_download_renews_the_retention_lease(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_server, "RESULT_RETENTION_S", 600.0)
    job = _make_job("done-1", status="done")
    job.zip_path = tmp_path / "result.zip"
    job.zip_path.write_bytes(b"zip")
    job.retain_until = 1.0  # about to expire
    app_server.SERVER_STATE.jobs[job.job_id] = job
    try:
        app_server.job_result(job.job_id)
    finally:
        app_server.SERVER_STATE.jobs.pop(job.job_id, None)
    assert job.retain_until > app_server.time.time() + 500


def test_cancellation_between_scheduling_and_worker_entry_prevents_launch(monkeypatch, tmp_path):
    job = _make_job("cancel-at-worker")
    job.run_dir = tmp_path / "run"
    job.run_dir.mkdir()
    app_server.SERVER_STATE.jobs[job.job_id] = job
    monkeypatch.setattr(app_server, "RESULT_RETENTION_S", 0)

    async def cancelled_to_thread(function, *args):
        assert app_server.kill_job(job.job_id)["status"] == "killed"
        function(*args)

    def forbidden_launch(*args, **kwargs):
        pytest.fail("cancelled work was launched")

    monkeypatch.setattr(app_server.asyncio, "to_thread", cancelled_to_thread)
    monkeypatch.setattr(app_server.subprocess, "Popen", forbidden_launch)
    asyncio.run(app_server.start_job(job, ["not-executed"], None))
    assert job.status == "killed" and job.cancelled
    assert not job.run_dir.exists()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX")
def test_cancellation_during_process_creation_reaps_the_registered_process(monkeypatch, tmp_path):
    job = _make_job("cancel-during-spawn")
    job.run_dir = tmp_path
    entered, release, terminated = Event(), Event(), Event()
    app_server.SERVER_STATE.jobs[job.job_id] = job

    def wait():
        assert terminated.wait(5)
        return -15

    proc = SimpleNamespace(pid=12345, stdout=[], wait=wait, poll=lambda: -15 if terminated.is_set() else None)

    def launch(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return proc

    monkeypatch.setattr(app_server.subprocess, "Popen", launch)
    monkeypatch.setattr(app_server.os, "killpg", lambda pid, sig: terminated.set())
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            worker = pool.submit(app_server._run_job_sync, job, ["not-executed"])
            assert entered.wait(5)
            killer = pool.submit(app_server.kill_job, job.job_id)
            release.set()
            assert killer.result(timeout=6)["status"] == "killed"
            worker.result(timeout=6)
        assert terminated.is_set() and job.status == "killed" and job.cancelled
    finally:
        release.set()
        terminated.set()
        app_server.SERVER_STATE.jobs.pop(job.job_id, None)


@pytest.mark.parametrize("disconnect", [False, True])
def test_active_download_keeps_workspace_until_response_finishes(monkeypatch, tmp_path, disconnect):
    job = _make_job("slow-download", status="done")
    job.run_dir = tmp_path / "run"
    job.run_dir.mkdir()
    job.zip_path = job.run_dir / "result.zip"
    job.zip_path.write_bytes(b"result")
    app_server.SERVER_STATE.jobs[job.job_id] = job
    monkeypatch.setattr(app_server, "RESULT_RETENTION_S", 0)

    async def scenario():
        transferring, finish = asyncio.Event(), asyncio.Event()

        async def transfer(self, scope, receive, send):
            transferring.set()
            await finish.wait()
            assert job.zip_path.exists()
            if disconnect:
                raise RuntimeError("client disconnected")

        async def completed_worker(function, *args):
            await transferring.wait()

        monkeypatch.setattr(app_server.FileResponse, "__call__", transfer)
        monkeypatch.setattr(app_server.asyncio, "to_thread", completed_worker)
        response = app_server.job_result(job.job_id)
        download = asyncio.create_task(response({}, None, None))
        cleanup = asyncio.create_task(app_server.start_job(job, ["not-executed"], None))
        await transferring.wait()
        await asyncio.sleep(0.15)
        assert not cleanup.done() and job.run_dir.exists() and job.active_downloads == 1
        finish.set()
        if disconnect:
            with pytest.raises(RuntimeError, match="client disconnected"):
                await download
        else:
            await download
        await asyncio.wait_for(cleanup, timeout=2)
        assert job.active_downloads == 0 and not job.run_dir.exists()

    asyncio.run(scenario())
