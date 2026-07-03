"""``--lr`` learning-rate override plumbing for fine-tuning (local, remote client, server)."""

import asyncio
import importlib.util
import types
from pathlib import Path

import konfai_apps.app as app_module
import konfai_apps.app_server as app_server
import pytest
import torch
from ruamel.yaml import YAML


def _write_src_checkpoint(path: Path) -> None:
    torch.save({"epoch": 10, "it": 100, "loss": 0.0, "Model": {}}, path)


def _run_local_fine_tune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    lr: float | None,
    pass_lr: bool,
) -> list[float | None]:
    """Drive ``KonfAIApp.fine_tune`` locally with every heavy step stubbed, capturing the train ``lr``."""
    src_ckpt = tmp_path / "CV_0_src.pt"
    _write_src_checkpoint(src_ckpt)

    dataset_dir = tmp_path / "Dataset"
    dataset_dir.mkdir()

    output_dir = tmp_path / "Output"
    output_dir.mkdir()
    (output_dir / "Config.yml").write_text("Trainer:\n  train_name: PLACEHOLDER\n", encoding="utf-8")

    captured: list[float | None] = []

    def fake_train(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("lr", "MISSING"))  # type: ignore[arg-type]
        model_config = Path(args[7])
        with open(model_config) as file:
            data = YAML().load(file)
        produced_dir = Path(args[8]) / data["Trainer"]["train_name"]
        produced_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 0, "it": 5, "loss": 0.0, "Model": {}}, produced_dir / "out.pt")

    monkeypatch.setattr("konfai.trainer.train", fake_train)
    monkeypatch.setattr(app_module.KonfAIApp, "symlink", staticmethod(lambda *a, **k: None))

    app = app_module.KonfAIApp.__new__(app_module.KonfAIApp)
    app.app_repository = types.SimpleNamespace(  # type: ignore[attr-defined]
        install_fine_tune=lambda *a, **k: [("CV_0.pt", str(src_ckpt))]
    )

    call_kwargs: dict = {
        "dataset": dataset_dir,
        "name": "Run",
        "output": output_dir,
        "epochs": 1,
        "it_validation": 1,
        "models": ["CV_0"],
        "gpu": [],
        "cpu": 1,
        "quiet": True,
        "config_file": "Config.yml",
        "tmp_dir": output_dir,
    }
    if pass_lr:
        call_kwargs["lr"] = lr

    app.fine_tune(**call_kwargs)
    return captured


def test_local_fine_tune_forwards_lr_to_train(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _run_local_fine_tune(monkeypatch, tmp_path, lr=0.03, pass_lr=True)
    assert captured == [0.03]


def test_local_fine_tune_defaults_lr_to_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _run_local_fine_tune(monkeypatch, tmp_path, lr=None, pass_lr=False)
    assert captured == [None]


class _FakePostResponse:
    def __init__(self, captured: dict, files: list, data: dict) -> None:
        captured["files"] = files
        captured["data"] = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"job_id": "job-1"}

    def __enter__(self) -> "_FakePostResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_remote_client_fine_tune_sends_lr_in_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cohort"
    dataset_dir.mkdir()
    (dataset_dir / "case.txt").write_text("x", encoding="utf-8")

    captured: dict = {}

    def fake_post(url, files, data, headers, timeout):  # type: ignore[no-untyped-def]
        return _FakePostResponse(captured, list(files), dict(data))

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module.KonfAIAppClient, "stream_logs", lambda self, job_id: None)
    monkeypatch.setattr(app_module.KonfAIAppClient, "download_result", lambda self, job_id, output: None)

    client = app_module.KonfAIAppClient.__new__(app_module.KonfAIAppClient)
    client.app = "demo"  # type: ignore[attr-defined]
    client.remote_server = types.SimpleNamespace(  # type: ignore[attr-defined]
        get_url=lambda: "http://server",
        get_headers=lambda: {},
    )

    client.fine_tune(dataset=dataset_dir, output=tmp_path / "out", lr=0.03)

    assert captured["data"]["lr"] == 0.03


pytestmark_server = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="fastapi is not installed",
)


@pytestmark_server
def test_server_fine_tune_cmd_adds_lr_when_provided() -> None:
    cmd = asyncio.run(app_server.fine_tune.__wrapped__(app_name="demo", dataset=None, lr=0.03))
    assert "--lr" in cmd
    assert cmd[cmd.index("--lr") + 1] == "0.03"


@pytestmark_server
def test_server_fine_tune_cmd_omits_lr_by_default() -> None:
    cmd = asyncio.run(app_server.fine_tune.__wrapped__(app_name="demo", dataset=None))
    assert "--lr" not in cmd
