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

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from konfai import RemoteServer, check_server
from konfai.utils.runtime import save_weights

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[3]
KONFAI_APPS_ROOT = REPO_ROOT / "konfai-apps"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
APP_ASSETS_DIR = ASSETS_DIR / "AppClientRemote" / "TinySynthesisApp"
WORKFLOW_ASSETS_DIR = REPO_ROOT / "tests" / "assets" / "Workflows"
SimpleITK = pytest.importorskip("SimpleITK")
# The server subprocess runs under this interpreter (`sys.executable` + PYTHONPATH), so the server
# stack has to be importable here or `konfai-apps-server` exits before it can bind the port.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


@dataclass
class RunningAppServer:
    process: subprocess.Popen[str]
    port: int
    token: str
    env: dict[str, str]
    app_id: str


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_image(path: Path) -> None:
    image = SimpleITK.GetImageFromArray(torch.linspace(-1, 1, 16 * 16, dtype=torch.float32).reshape(1, 16, 16).numpy())
    image.SetSpacing((1.0, 1.0, 1.0))
    image = SimpleITK.Cast(image, SimpleITK.sitkFloat32)
    SimpleITK.WriteImage(image, str(path))


def _resolve_entrypoint(bin_dir: Path, name: str) -> Path:
    candidates = [bin_dir / name, bin_dir / f"{name}.cmd", bin_dir / f"{name}.exe"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise AssertionError(f"Missing CLI entrypoint in test bin dir: {name}")


def _entrypoint_command(bin_dir: Path, name: str) -> list[str]:
    script = bin_dir / name
    if not script.exists():
        raise AssertionError(f"Missing CLI script in test bin dir: {name}")
    return [sys.executable, str(script)]


def _write_cli_entrypoints(bin_dir: Path) -> None:
    def write_script(name: str, target: str) -> None:
        script = bin_dir / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            f"sys.path.insert(0, {str(KONFAI_APPS_ROOT)!r})\n"
            f"from konfai_apps.cli import {target} as _entry\n"
            "if __name__ == '__main__':\n"
            "    _entry()\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        (bin_dir / f"{name}.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )

    write_script("konfai-apps", "main_apps")
    write_script("konfai-apps-server", "main_apps_server")


def _write_local_synthesis_app(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    for asset in APP_ASSETS_DIR.iterdir():
        if asset.is_file():
            shutil.copy2(asset, app_dir / asset.name)
    shutil.copy2(WORKFLOW_ASSETS_DIR / "TinySynth.py", app_dir / "TinySynth.py")

    spec = importlib.util.spec_from_file_location("TinySynth", app_dir / "TinySynth.py")
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load TinySynth test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TinySynthNet()
    save_weights({"Model": model.network_states()}, app_dir / "tiny.safetensors")


def _subprocess_env(bin_dir: Path, token: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    package_paths = f"{KONFAI_APPS_ROOT}{os.pathsep}{REPO_ROOT}"
    env["PYTHONPATH"] = package_paths if not pythonpath else f"{package_paths}{os.pathsep}{pythonpath}"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["KONFAI_API_TOKEN"] = token
    return env


@pytest.fixture
def running_app_server(tmp_path: Path) -> Generator[RunningAppServer]:
    token = "secret"
    port = _free_port()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_cli_entrypoints(bin_dir)

    app_dir = tmp_path / "TinySynthesisApp"
    _write_local_synthesis_app(app_dir)

    apps_config = tmp_path / "apps.json"
    apps_config.write_text(json.dumps({"apps": [str(app_dir)]}), encoding="utf-8")

    env = _subprocess_env(bin_dir, token)
    server_command = [
        *_entrypoint_command(bin_dir, "konfai-apps-server"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--token",
        token,
        "--apps",
        str(apps_config),
    ]
    process = subprocess.Popen(
        server_command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    remote = RemoteServer("127.0.0.1", port, token)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"konfai-apps-server exited early with code {process.returncode}\n{output}")
        ok, _ = check_server(remote, timeout_s=0.2)
        if ok:
            break
        time.sleep(0.1)
    else:
        process.terminate()
        output = process.stdout.read() if process.stdout is not None else ""
        process.wait(timeout=10)
        raise AssertionError(f"konfai-apps-server did not become ready\n{output}")

    try:
        yield RunningAppServer(
            process=process,
            port=port,
            token=token,
            env=env,
            app_id=str(app_dir),
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_main_apps_remote_infer_roundtrip_against_main_apps_server(
    running_app_server: RunningAppServer,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mha"
    _write_image(input_path)
    output_dir = tmp_path / "output"

    command = [
        *_entrypoint_command(Path(running_app_server.env["PATH"].split(os.pathsep)[0]), "konfai-apps"),
        "infer",
        running_app_server.app_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(running_app_server.port),
        "--token",
        running_app_server.token,
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--cpu",
        "1",
        "--prediction-file",
        "Prediction.yml",
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=running_app_server.env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Remote CLI roundtrip failed.\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    assert "Remote job" in result.stdout
    assert "Result written to" in result.stdout
    predicted = sorted(output_dir.rglob("sCT.mha"))
    assert predicted, f"No synthesized output produced.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def _interactive(env: dict[str, str], *args: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "konfai_apps.interactive", *args],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert _answer(process) == "ready"
    except BaseException:
        process.kill()
        process.wait()
        raise
    return process


def _answer(process: subprocess.Popen[str]) -> str:
    """The session's next protocol line, its log skipped."""
    assert process.stdout is not None
    log = []
    for line in process.stdout:
        if line.startswith("konfai-interactive:"):
            return line.removeprefix("konfai-interactive:").strip()
        log.append(line)
    raise AssertionError("the session stopped:\n" + "".join(log))


def test_an_interactive_session_predicts_on_the_server_as_in_the_process(
    running_app_server: RunningAppServer, tmp_path: Path
) -> None:
    # The same requests, twice, to a session held in the process and to one held by the server: the same predictions.
    image = tmp_path / "input.mha"
    _write_image(image)
    remote = ["--host", "127.0.0.1", "--port", str(running_app_server.port), "--token", running_app_server.token]
    predictions = {}
    for mode, options in (("process", []), ("server", remote)):
        session = _interactive(running_app_server.env, running_app_server.app_id, *options)
        assert session.stdin is not None
        try:
            for step in (1, 2):
                request = {"id": step, "inputs": [[str(image)]], "output": str(tmp_path / mode / str(step))}
                session.stdin.write(json.dumps(request) + "\n")
                session.stdin.flush()
                assert _answer(session).startswith(f"done {step} "), mode
                (written,) = sorted(path for path in (tmp_path / mode / str(step)).rglob("*.mha"))
                predictions[mode, step] = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(written)))
            session.stdin.close()
            assert session.wait(timeout=60) == 0
        finally:
            if session.poll() is None:
                session.kill()
                session.wait()
    reference = predictions["process", 1]
    assert reference.size and all((prediction == reference).all() for prediction in predictions.values())


def test_a_closed_session_is_gone_from_the_server(running_app_server: RunningAppServer, tmp_path: Path) -> None:
    from konfai_apps.interactive import RemoteSession

    image = tmp_path / "input.mha"
    _write_image(image)
    server = RemoteServer("127.0.0.1", running_app_server.port, running_app_server.token)
    session = RemoteSession(server, running_app_server.app_id, [])
    session.predict([[str(image)]], tmp_path / "out")
    held = session.session_id
    session.close()
    with pytest.raises(RuntimeError, match="404"):
        session._call("DELETE", f"/sessions/{held}", 30)
    with pytest.raises(RuntimeError, match="404"):
        session._call("POST", f"/apps/{running_app_server.app_id}-not-served/sessions", 30, files={"image": b"x"})


def test_a_session_the_server_closed_is_opened_again(running_app_server: RunningAppServer, tmp_path: Path) -> None:
    # Closed idle, its worker stopped or the server restarted: the next prediction opens a session again.
    from konfai_apps.interactive import RemoteSession

    image = tmp_path / "input.mha"
    _write_image(image)
    server = RemoteServer("127.0.0.1", running_app_server.port, running_app_server.token)
    session = RemoteSession(server, running_app_server.app_id, [])
    session.predict([[str(image)]], tmp_path / "first")
    closed = session.session_id
    session._call("DELETE", f"/sessions/{closed}", 30)
    session.predict([[str(image)]], tmp_path / "second")
    assert session.session_id != closed
    first, second = (SimpleITK.ReadImage(str(tmp_path / name / "Prediction.mha")) for name in ("first", "second"))
    assert (SimpleITK.GetArrayFromImage(first) == SimpleITK.GetArrayFromImage(second)).all()
    session.close()


def test_a_session_sends_each_input_apart_and_names_its_app_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two inputs named alike in two folders reach the server as two files, and an app's #preset is part of its path.
    from konfai_apps.interactive import RemoteSession

    image = tmp_path / "input.mha"
    _write_image(image)
    for value, folder in enumerate(("a", "b")):
        (tmp_path / folder).mkdir()
        mask = SimpleITK.Image([4, 4, 4], SimpleITK.sitkUInt8) + value
        SimpleITK.WriteImage(mask, str(tmp_path / folder / "mask.nii.gz"))
    calls = []

    answer = SimpleNamespace(content=b"prediction", json=lambda: {"session_id": "s"})

    def call(method, path, timeout, **kwargs):
        calls.append(
            (
                path,
                [SimpleITK.GetArrayFromImage(_read(body)).max() for _, (_, body) in kwargs.get("files", [])]
                if path.endswith("predict")
                else None,
            )
        )
        return answer

    def _read(body: bytes):
        written = tmp_path / "sent.mha"
        written.write_bytes(body)
        return SimpleITK.ReadImage(str(written))

    session = RemoteSession(RemoteServer("127.0.0.1", 1, None), "demo#fast", [])
    monkeypatch.setattr(session, "_call", call)
    session.predict(
        [[str(image)], [str(tmp_path / "a" / "mask.nii.gz")], [str(tmp_path / "b" / "mask.nii.gz")]], tmp_path / "out"
    )
    assert calls[0][0] == "/apps/demo%23fast/sessions"
    assert calls[1] == ("/sessions/s/predict", [0, 1])
