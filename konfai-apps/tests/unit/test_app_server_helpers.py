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
import importlib.util
import io
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import konfai_apps.app_server as app_server
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="fastapi is not installed",
)


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


def _make_directory_volume_upload(tmp_path: Path, filename: str) -> SimpleNamespace:
    store = tmp_path / "src_store"
    store.mkdir()
    (store / ".zgroup").write_text("{}", encoding="utf-8")
    (store / "0.0.0").write_bytes(b"chunk")
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
