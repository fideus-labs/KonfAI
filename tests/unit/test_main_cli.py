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

"""Tests for the ``konfai`` CLI (``konfai.main``): subcommand dispatch and the
CLI-facing parameter contract of the backend entry points."""

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import konfai
import konfai.evaluator as evaluator_module
import konfai.main as main_module
import konfai.predictor as predictor_module
import konfai.trainer as trainer_module
import konfai.transformer as transformer_module
import pytest


def test_konfai_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["konfai", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 0


def test_the_version_is_looked_up_only_when_asked_for(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """``importlib.metadata.version`` scans the installed distributions; every other invocation
    builds the parser without paying for it."""
    import importlib.metadata

    looked_up: list[str] = []
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: looked_up.append(name) or real_version(name))

    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "--no-such-flag"])
    with pytest.raises(SystemExit):
        main_module.main()
    assert looked_up == []

    monkeypatch.setattr(sys, "argv", ["konfai", "--version"])
    with pytest.raises(SystemExit) as exited:
        main_module.main()
    assert exited.value.code == 0 and looked_up == ["konfai"]
    assert capsys.readouterr().out.strip() == real_version("konfai")


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        (["--help"], 0),
        (["TRANSFORM", "--help"], 0),
        (["TRAIN", "--no-such-flag"], 2),
        (["TRANSFORM", "--gpu", "0", "--cpu", "1"], 2),
    ],
)
def test_konfai_help_and_usage_errors_do_not_import_torch(argv: list[str], exit_code: int) -> None:
    """The parser is built without the runtime module: torch and the host probes (psutil, pynvml) load
    only for the command that runs."""
    script = f"""
import sys
sys.argv = ["konfai", *{argv!r}]
from konfai.main import main
try:
    main()
except SystemExit as exit:
    assert exit.code == {exit_code}, exit.code
else:
    raise AssertionError("expected an exit")
loaded = sorted(name for name in ("torch", "konfai.utils.runtime", "psutil", "pynvml") if name in sys.modules)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_gpu_ids_are_checked_against_the_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--gpu`` has no parse-time ``choices`` (resolving them imports torch); the check runs at dispatch."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(trainer_module, "train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(konfai, "cuda_visible_devices", lambda: [0, 1])

    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "--gpu", "1", "3"])
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 2
    assert not captured

    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "--gpu", "1"])
    main_module.main()
    assert captured["gpu"] == [1]


def test_a_gpu_named_by_uuid_is_a_usage_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(trainer_module, "train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-3f2a1b4c")

    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "--gpu", "0"])
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 2
    assert "CUDA_VISIBLE_DEVICES='GPU-3f2a1b4c'" in capsys.readouterr().err
    assert not captured


def test_init_reports_a_refusal_by_its_message_and_a_bug_by_its_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A user model's NameError under --init printed "[KonfAI] name 'nb_classes' is not defined": no type,
    no file, no line. A bug propagates as the run path lets it; a designed refusal stays a message."""
    from konfai.utils.errors import ConfigError

    config = tmp_path / "Config.yml"
    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "--init", "-c", str(config)])

    def buggy_build(config: Path) -> None:
        raise NameError("name 'nb_classes' is not defined")

    monkeypatch.setattr(trainer_module, "build_train", buggy_build)
    with pytest.raises(NameError, match="nb_classes"):
        main_module.main()
    assert "Wrote what resolved before the error" in capsys.readouterr().out

    def refusing_build(config: Path) -> None:
        raise ConfigError("'Trainer.Dataset' is empty.")

    monkeypatch.setattr(trainer_module, "build_train", refusing_build)
    with pytest.raises(SystemExit) as exited:
        main_module.main()
    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "Wrote what resolved before the error" in out and "[Config] 'Trainer.Dataset' is empty." in out


def test_konfai_train_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_train(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "train", fake_train)
    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "-c", "Config.yml"])

    main_module.main()

    assert captured["config"] == "Config.yml"


def test_main_prediction_dispatches_config_as_prediction_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_predict(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(predictor_module, "predict", fake_predict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "konfai",
            "PREDICTION",
            "-c",
            "Prediction.custom.yml",
            "--models",
            str(tmp_path / "checkpoint.pt"),
            "--cpu",
            "1",
        ],
    )

    main_module.main()

    assert captured["prediction_file"] == "Prediction.custom.yml"
    assert captured["cpu"] == 1
    assert captured["gpu"] == []
    assert captured["models"] == [str(tmp_path / "checkpoint.pt")]
    assert "config" not in captured


def test_konfai_eval_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(evaluator_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["konfai", "EVALUATION", "-c", "Evaluation.yml"])

    main_module.main()

    assert captured["evaluations_file"] == "Evaluation.yml"


def test_konfai_resume_dispatches_model_and_lr(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_train(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "train", fake_train)
    monkeypatch.setattr(sys, "argv", ["konfai", "RESUME", "-c", "Config.yml", "--model", "ckpt.pt", "--lr", "0.001"])

    main_module.main()

    assert captured["command"] == "RESUME"
    assert captured["config"] == "Config.yml"
    assert captured["model"] == "ckpt.pt"
    assert captured["lr"] == 0.001


def test_konfai_resume_accepts_checkpoints_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_train(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "train", fake_train)
    monkeypatch.setattr(
        sys, "argv", ["konfai", "RESUME", "-c", "Config.yml", "--model", "ckpt.pt", "--checkpoints-dir", "/elsewhere"]
    )

    main_module.main()

    assert captured["checkpoints_dir"] == "/elsewhere"


def test_konfai_transform_dispatches_config_as_transform_file(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_transform(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(transformer_module, "transform", fake_transform)
    monkeypatch.setattr(sys, "argv", ["konfai", "TRANSFORM", "-c", "Transform.yml"])

    main_module.main()

    assert captured["transform_file"] == "Transform.yml"
    assert "config" not in captured
    assert "plan" not in captured


def test_konfai_transform_plan_short_circuits_to_plan_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """--plan must dispatch to plan_transform, never transform: the distributed wrapper filters
    kwargs by the entrypoint's signature, so a 'plan' kwarg passed through would be silently
    dropped and the run would proceed as if the flag had never been given."""
    planned: dict[str, object] = {}
    plan_parameters = inspect.signature(transformer_module.plan_transform).parameters

    def fake_plan_transform(**kwargs) -> None:
        planned.update(kwargs)

    def fail_transform(**kwargs) -> None:
        raise AssertionError("--plan must not start the transform run")

    monkeypatch.setattr(transformer_module, "plan_transform", fake_plan_transform)
    monkeypatch.setattr(transformer_module, "transform", fail_transform)
    monkeypatch.setattr(sys, "argv", ["konfai", "TRANSFORM", "-c", "Transform.yml", "--plan", "-q", "--cpu", "2"])

    main_module.main()

    # Every CLI flag reaches plan_transform by name, and plan_transform declares each one: no
    # catch-all that would swallow a flag in silence.
    assert planned == {
        "transform_file": "Transform.yml",
        "transforms_dir": "./Transforms/",
        "overwrite": False,
        "gpu": [],
        "cpu": 2,
        "quiet": True,
    }
    assert set(planned) <= set(plan_parameters)
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in plan_parameters.values())


def test_plan_transform_sizes_the_plan_for_the_run_world_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """One rank per GPU, else ``cpu`` ranks: the plan shards the way ``transform`` will."""
    world_sizes: list[int] = []

    def compute_plan(world_size: int, overwrite: bool) -> SimpleNamespace:
        world_sizes.append(world_size)
        return SimpleNamespace(report=lambda: "")

    monkeypatch.setattr(
        transformer_module, "build_transform", lambda **kwargs: SimpleNamespace(compute_plan=compute_plan)
    )

    transformer_module.plan_transform(gpu=[0, 1], cpu=None, transform_file="Transform.yml")
    transformer_module.plan_transform(gpu=[], cpu=3, transform_file="Transform.yml")
    transformer_module.plan_transform(transform_file="Transform.yml")

    assert world_sizes == [2, 3, 1]


def test_konfai_cluster_refuses_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan runs where it is typed and submits nothing; the SLURM flags have no meaning for it."""
    monkeypatch.setattr(transformer_module, "plan_transform", lambda **kwargs: pytest.fail("planned"))
    monkeypatch.setattr(sys, "argv", ["konfai-cluster", "--name", "job", "TRANSFORM", "--plan"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.cluster()

    assert exc_info.value.code == 2


def test_predict_evaluate_expose_tensorboard_param():
    """#7 CLI -tb/--tensorboard (dest 'tensorboard') must reach predict(); evaluate() keeps the parameter,
    which konfai-apps passes by position."""
    for fn in (predictor_module.predict, evaluator_module.evaluate):
        params = inspect.signature(fn).parameters
        assert "tensorboard" in params, f"{fn.__name__} must accept 'tensorboard'"
        assert "tb" not in params, f"{fn.__name__} must not use the old 'tb' name"


@pytest.mark.parametrize(
    ("kind", "member"),
    [
        ("transforms", "Resample"),
        ("augmentations", "Flip"),
        ("criteria", "Dice"),
        ("reductions", "Median"),
        ("models", "default|UNet.yml"),
        ("blocks", "Conv"),
    ],
)
def test_konfai_list_prints_each_component_family(
    monkeypatch: pytest.MonkeyPatch, capsys, kind: str, member: str
) -> None:
    """`konfai list <kind>` prints one aligned `name  doc` line per component, no run machinery."""
    monkeypatch.setattr(sys, "argv", ["konfai", "list", kind])

    main_module.main()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any(line.split()[0] == member for line in lines)


def test_konfai_list_refuses_an_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["konfai", "list", "optimizers"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 2  # an argparse choices error, before anything heavy loads


@pytest.mark.parametrize("argv", [["TRAIN"], ["TRAIN", "--init"]])
def test_a_warning_the_build_raises_reads_in_konfais_format(tmp_path: Path, argv: list[str]) -> None:
    """A workflow's build warns before its run Log captures the console (an unread key): the CLI still
    spells the warning as KonfAI's, not as Python's 'file:line: KonfAIWarning:' with the source line."""
    script = f"""
import sys, warnings
import konfai.trainer as trainer
from konfai.utils.errors import KonfAIWarning

def build(**kwargs):
    warnings.warn("Unknown key 'Trainer.epoch'.", KonfAIWarning, stacklevel=1)

trainer.train = trainer.build_train = build
sys.argv = ["konfai", *{argv!r}]
from konfai.main import main
main()
"""
    run = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True, timeout=300
    )
    assert "[KonfAI] WARNING: Unknown key 'Trainer.epoch'." in run.stderr
    assert "KonfAIWarning:" not in run.stderr


def test_a_designed_refusal_exits_with_status_1_and_its_message(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The exit-code contract of the CLI reference: a refusal ends the command with 1, message only."""
    from konfai.utils.errors import ConfigError

    def refused(**kwargs):
        raise ConfigError("Nothing to run.", "Name a dataset.")

    monkeypatch.delenv("KONFAI_DEBUG", raising=False)
    monkeypatch.setattr(evaluator_module, "evaluate", refused)
    monkeypatch.setattr(sys, "argv", ["konfai", "EVALUATION"])
    with pytest.raises(SystemExit) as exited:
        main_module.main()
    assert exited.value.code == 1
    err = capsys.readouterr().err
    assert "[Config] Nothing to run." in err and "Traceback" not in err


def test_an_interrupted_run_exits_with_the_interrupt_status(tmp_path: Path) -> None:
    """Ctrl+C ends the command with status 130, so ``konfai TRAIN && konfai PREDICTION`` stops there. In a
    subprocess: a KeyboardInterrupt the CLI let through would stop the test session itself."""
    script = """
import sys
import konfai.trainer as trainer

def interrupted(**kwargs):
    raise KeyboardInterrupt

trainer.train = interrupted
sys.argv = ["konfai", "TRAIN"]
from konfai.main import main
main()
"""
    run = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert run.returncode == 130, run.stderr
    assert "Manual interruption (Ctrl+C)" in run.stdout


_REPO = Path(__file__).resolve().parents[2]
_DOCUMENTED_COMMAND_SOURCES = (".claude/skills", "docs/source", "examples", "README.md")


def _documented_konfai_commands() -> list[tuple[str, str]]:
    """Every ``konfai``/``konfai-cluster`` line of a shell block in the Markdown docs and skills."""
    commands = []
    for source in _DOCUMENTED_COMMAND_SOURCES:
        root = _REPO / source
        for path in sorted(root.rglob("*.md") if root.is_dir() else [root]):
            fence = None
            pending = ""
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    fence = None if fence is not None else stripped[3:].strip()
                    continue
                if fence not in ("", "bash", "sh", "shell", "console"):
                    continue
                if pending or stripped.startswith(("konfai ", "konfai-cluster ")):
                    pending += stripped.removesuffix("\\")
                    if stripped.endswith("\\"):
                        continue
                    commands.append((f"{path.relative_to(_REPO)}:{number}", pending))
                    pending = ""
    return commands


@pytest.mark.parametrize(("where", "command"), _documented_konfai_commands())
def test_every_documented_konfai_command_parses(monkeypatch: pytest.MonkeyPatch, where: str, command: str) -> None:
    """A command line the docs or skills show is accepted by the parser it names (nothing runs)."""
    import shlex

    argv = shlex.split(command, comments=True)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(main_module, "_dispatch", lambda parser, args: None)
    entry = main_module.cluster if argv[0] == "konfai-cluster" else main_module.main

    try:
        entry()
    except SystemExit as exit:
        assert exit.code == 0, f"{where}: {command}"
