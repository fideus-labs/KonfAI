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

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import konfai_apps.app as app_module
import konfai_apps.cli as apps_cli_module
import pytest
from konfai_apps.errors import AppRepositoryError


def test_main_apps_dispatches_local_infer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class DummyApp:
        def __init__(self, app_name: str, download: bool, force_update: bool) -> None:
            calls.append(("init", (app_name, download, force_update)))

        def infer(self, **kwargs) -> None:
            calls.append(("infer", kwargs))

    monkeypatch.setattr(app_module, "KonfAIApp", DummyApp)
    monkeypatch.setattr(app_module, "KonfAIAppClient", object)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "konfai-apps",
            "infer",
            "demo/app",
            "--inputs",
            str(tmp_path / "input.mha"),
            "--cpu",
            "2",
            "--prediction-file",
            "Prediction.custom.yml",
            "--tta",
            "2",
            "--mc",
            "1",
        ],
    )

    apps_cli_module.main_apps()

    assert calls[0] == ("init", ("demo/app", False, False))
    infer_kwargs = cast(dict[str, Any], calls[1][1])
    assert infer_kwargs["cpu"] == 2
    assert infer_kwargs["gpu"] == []
    assert infer_kwargs["prediction_file"] == "Prediction.custom.yml"
    assert infer_kwargs["tta"] == 2
    assert infer_kwargs["mc"] == 1
    assert infer_kwargs["inputs"] == [[(tmp_path / "input.mha").resolve()]]


def _run_apps_server(monkeypatch: pytest.MonkeyPatch, apps_config: Path) -> str:
    """Drive `main_apps_server` up to its `--apps` validation and return the exit message.

    `main_apps_server` opens with `import uvicorn`, so every path through it -- including the
    validation that never reaches `uvicorn.run` -- needs the server stack importable.
    """
    uvicorn = pytest.importorskip("uvicorn")

    def _refuse_to_serve(*args: object, **kwargs: object) -> None:
        # Both callers must exit during validation. Should that validation ever regress, falling
        # through to the real `uvicorn.run` would bind a port and serve forever, hanging the unit
        # suite instead of failing it -- so make reaching this point a loud failure.
        pytest.fail("main_apps_server reached uvicorn.run: --apps validation did not reject the config")

    monkeypatch.setattr(uvicorn, "run", _refuse_to_serve)
    # --auth off keeps the test on config validation instead of token resolution; delenv first so
    # the pop the CLI performs on a real inherited token is restored at teardown.
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["konfai-apps-server", "--auth", "off", "--apps", str(apps_config)])

    with pytest.raises(SystemExit) as excinfo:
        apps_cli_module.main_apps_server()
    return str(excinfo.value)


def test_the_app_server_names_its_extra_when_it_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HTTP job server is an extra: without it, konfai-apps-server says how to install it."""
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    monkeypatch.setattr(sys, "argv", ["konfai-apps-server", "--apps", "apps.json"])

    with pytest.raises(SystemExit, match=r"konfai-apps\[server\]"):
        apps_cli_module.main_apps_server()


def test_main_apps_server_rejects_a_missing_apps_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A mistyped --apps must name the path it could not find, not start a server with no apps.
    missing = tmp_path / "absent.json"

    assert str(missing) in _run_apps_server(monkeypatch, missing)


def test_main_apps_server_rejects_a_config_without_an_apps_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Valid JSON of the wrong shape must be refused up front: the server reads `apps` as a list, so
    # binding the port first would only surface the mistake on the first request.
    apps_config = tmp_path / "apps.json"
    apps_config.write_text(json.dumps({"applications": ["demo/app"]}), encoding="utf-8")

    assert "Invalid config file" in _run_apps_server(monkeypatch, apps_config)


def test_main_apps_server_check_validates_and_exits_without_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # --check is a validation command: a CI job running it must get its answer back, not a server.
    uvicorn = pytest.importorskip("uvicorn")
    served: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: served.append(kwargs))
    monkeypatch.setattr(apps_cli_module, "get_app_repository_info", lambda app_id, force_update: app_id)
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setenv("KONFAI_APPS_CONFIG", "{}")  # the CLI publishes the config; restored at teardown
    apps_config = tmp_path / "apps.json"
    apps_config.write_text(json.dumps({"apps": ["demo/app"]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["konfai-apps-server", "--auth", "off", "--apps", str(apps_config), "--check"])

    apps_cli_module.main_apps_server()

    assert "All apps validated successfully." in capsys.readouterr().out
    assert served == []


@pytest.mark.parametrize(
    "command",
    [
        ["fine-tune", "demo/app", "Tuned", "-d", "./Dataset", "-o", "."],
        ["fine-tune", "demo/app", "Tuned", "-d", "../other", "-o", "."],
        ["infer", "demo/app", "-i", "../other/in.mha", "-o", "../out", "--tmp-dir", "."],
    ],
    ids=["fine-tune-own-dataset", "fine-tune-other-dataset", "infer-tmp-dir"],
)
def test_an_app_run_in_a_project_refuses_to_delete_its_dataset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, command: list[str]
) -> None:
    """fine-tune works in its --output and infer in its --tmp-dir, and both stage their inputs as
    ./Dataset: in a project directory that is the user's own data, refused and left in place."""
    project = tmp_path / "project"
    user_file = project / "Dataset" / "P000" / "CT.mha"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("the only copy", encoding="utf-8")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "in.mha").write_bytes(b"volume")
    monkeypatch.chdir(project)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)
    monkeypatch.setattr(app_module.KonfAIApp, "__init__", lambda self, *args: None)
    monkeypatch.setattr(sys, "argv", ["konfai-apps", *command, "--cpu", "1"])

    with pytest.raises(SystemExit) as exited:
        apps_cli_module.main_apps()

    assert exited.value.code == 1
    assert "not staged by konfai-apps" in capsys.readouterr().err
    assert user_file.read_text(encoding="utf-8") == "the only copy"


def _app_cli_mains() -> dict[str, Any]:
    return {
        "konfai-apps": apps_cli_module.main_apps,
        "app-cli": apps_cli_module.build_app_cli("demo-konfai", "demo", resolve_app=lambda args: "./no_such_app"),
    }


@pytest.mark.parametrize("cli", ["konfai-apps", "app-cli"])
def test_a_refusal_prints_its_message_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, cli: str
) -> None:
    """A mistyped app directory: the refusal names the path, with no traceback, like the konfai CLI. It does
    not read "not found", which SlicerKonfAI takes for an app gone for good and drops from its list."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KONFAI_DEBUG", raising=False)
    app = ["./no_such_app"] if cli == "konfai-apps" else []
    monkeypatch.setattr(sys, "argv", [cli, "infer", *app, "-i", "in.mha", "--cpu", "1"])

    with pytest.raises(SystemExit) as exit_info:
        _app_cli_mains()[cli]()

    message = capsys.readouterr().err.strip()
    assert exit_info.value.code == 1
    assert message.startswith(f"[App repository] No app directory at '{tmp_path / 'no_such_app'}'")
    assert "not found" not in message.lower()


def test_konfai_debug_keeps_the_refusal_traceback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KONFAI_DEBUG", "1")
    monkeypatch.setattr(sys, "argv", ["konfai-apps", "infer", "./no_such_app", "-i", "in.mha", "--cpu", "1"])

    with pytest.raises(AppRepositoryError, match="No app directory at"):
        apps_cli_module.main_apps()

def test_python_m_konfai_apps_runs_the_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["konfai-apps", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("konfai_apps", run_name="__main__")
    assert exit_info.value.code == 0 and "konfai-apps" in capsys.readouterr().out


def test_list_and_show_read_an_app_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from konfai_apps.errors import AppRepositoryError

    for name, task in (("B_app", "segmentation"), ("A_app", "registration"), ("Other", "evaluation")):
        (tmp_path / name).mkdir()
        manifest = {"display_name": name.upper(), "short_description": "Does.", "description": "Does it."}
        (tmp_path / name / "app.json").write_text(json.dumps({**manifest, "task": task, "tta": 0, "mc_dropout": 0}))
    (tmp_path / "not_an_app").mkdir()
    assert apps_cli_module.app_id(str(tmp_path), "A_app") == str(tmp_path / "A_app")
    assert apps_cli_module.app_id("org/repo@v1", "A_app") == "org/repo@v1:A_app"
    assert [name for name, _ in apps_cli_module.list_apps(str(tmp_path), "registration")] == ["A_app"]

    monkeypatch.setattr(sys, "argv", ["konfai-apps", "list", str(tmp_path)])
    apps_cli_module.main_apps()
    assert capsys.readouterr().out.splitlines()[:3] == ["A_app  A_APP", "       Does.", "B_app  B_APP"]

    config = "Predictor:\n  Model:\n    classpath: torch.nn:Identity\n    Identity:\n      iterations: 3\n"
    (tmp_path / "A_app" / "Prediction.yml").write_text(config)
    manifest = json.loads((tmp_path / "A_app" / "app.json").read_text())
    manifest["model_files"] = [{"repo_id": "org/models", "revision": "0" * 40, "filename": "net.pt"}]
    (tmp_path / "A_app" / "app.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["konfai-apps", "show", str(tmp_path / "A_app")])
    apps_cli_module.main_apps()
    out = capsys.readouterr().out
    assert out.startswith("A_app: A_APP\n\nDoes it.") and "torch.nn:Identity" in out and "  iterations = 3" in out
    assert "org/models:net.pt (not cached yet" in out and "model_files" not in out
    with pytest.raises(AppRepositoryError, match="Did you mean 'A_app'"):
        apps_cli_module.describe_app(str(tmp_path), "A_ap")


def test_check_overrides_refuses_what_the_run_would_and_leaves_the_config_alone() -> None:
    from konfai_apps.app_repository import check_overrides
    from konfai_apps.errors import AppRepositoryError

    config = {"Predictor": {"Model": {"classpath": "torch.nn:Identity", "Identity": {"iterations": 3}}}}
    check_overrides(config, ["iterations=5"])
    assert config["Predictor"]["Model"]["Identity"]["iterations"] == 3
    with pytest.raises(AppRepositoryError, match="Did you mean 'iterations'"):
        check_overrides(config, ["iteration=5"])


def test_the_shared_options_refuse_a_uri_and_a_negative_gpu() -> None:
    import argparse

    from konfai_apps.options import add_device, local_path

    parser = argparse.ArgumentParser()
    add_device(parser)
    assert parser.parse_args(["--gpu", "0", "--force-update"]).force_update
    with pytest.raises(SystemExit):
        parser.parse_args(["--gpu", "-1"])
    with pytest.raises(argparse.ArgumentTypeError, match="URI"):
        local_path("s3://bucket/image.nii.gz")


def test_an_out_of_memory_run_exits_with_its_own_code() -> None:
    """IMPACT-Reg runs a preset as a child and retries smaller on EXIT_OUT_OF_MEMORY: an out-of-memory error must end
    the process with it, not with a traceback and exit code 1."""
    import torch
    from konfai.utils.errors import EXIT_OUT_OF_MEMORY

    with pytest.raises(SystemExit) as stopped, apps_cli_module._exit_on_refusal():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.35 GiB")
    assert stopped.value.code == EXIT_OUT_OF_MEMORY
