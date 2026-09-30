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

"""Tests of the ``impact-reg-konfai`` command line: which subcommands exist, how the presets are passed, and
what a failure or a signal does. These lock the CLI contract the README and SlicerImpactReg depend on:
``register`` takes the preset(s) as a **positional** argument (or ``-p/--preset``), while ``eval`` /
``uncertainty`` take ``--preset``. No app is resolved: the app is a stub, or the invocation stops at ``--help``
or an argument error."""

import os
import subprocess
import sys

import pytest
from impact_reg_konfai import PRESETS_REVISION, cli, impact_reg
from konfai.utils.errors import KonfAIError


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    """Run ``cli.main`` on ``argv``; return the SystemExit code (0 if none)."""
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def _stub_app(calls: dict[str, dict]) -> type:
    """A drop-in for ``ImpactRegKonfAIApp`` that records the dispatched call instead of resolving any app."""

    class _StubApp:
        def __init__(self, **_: object) -> None:
            pass

        def register(self, presets: list[str], *args: object, **kwargs: object) -> None:
            calls["register"] = {"presets": presets, **kwargs}

        def evaluate(self, **kwargs: object) -> None:
            calls["evaluate"] = kwargs

        def uncertainty(self, **kwargs: object) -> None:
            calls["uncertainty"] = kwargs

    return _StubApp


@pytest.mark.parametrize("subcommand", ["list", "show", "register", "eval", "uncertainty"])
def test_subcommands_are_wired(monkeypatch: pytest.MonkeyPatch, subcommand: str) -> None:
    assert _run(monkeypatch, [subcommand, "--help"]) == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["register", "FireANTs_SyN", "Generic_Rigid", "-f", "a.mha", "-m", "b.mha"],
        ["register", "-f", "a.mha", "-m", "b.mha", "-p", "FireANTs_SyN", "Generic_Rigid"],
        ["register", "FireANTs_SyN", "-f", "a.mha", "-m", "b.mha", "--preset", "Generic_Rigid"],
    ],
)
def test_register_takes_presets_first_or_with_the_flag(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    # The positional form (the README's and SlicerImpactReg's) and -p/--preset, which may sit anywhere, name the
    # same ensemble.
    calls: dict[str, dict] = {}
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app(calls))
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv])
    cli.main()
    assert calls["register"]["presets"] == ["FireANTs_SyN", "Generic_Rigid"]


def test_register_says_where_a_trailing_preset_went(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # A name after -m is one more moving image; the error must say so, not that 'presets' is missing.
    assert _run(monkeypatch, ["register", "-f", "a.mha", "-m", "b.mha", "FireANTs_SyN"]) == 2
    assert "first or with -p/--preset" in capsys.readouterr().err


def test_eval_forwards_preset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # --preset must be parsed and forwarded to app.evaluate. Stub the app so nothing is resolved; a valid
    # dispatch (rather than an "unrecognized arguments" argparse error) proves the flag is really wired.
    calls: dict[str, dict] = {}
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app(calls))
    monkeypatch.setattr(
        sys, "argv", ["impact-reg-konfai", "eval", "--preset", "FireANTs_SyN", "-f", "a.mha", "-m", "b.mha"]
    )
    cli.main()  # no SystemExit: argparse accepted --preset and dispatch reached the stub
    assert calls["evaluate"]["preset"] == "FireANTs_SyN"


def test_uncertainty_forwards_preset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict] = {}
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app(calls))
    monkeypatch.setattr(
        sys, "argv", ["impact-reg-konfai", "uncertainty", "--preset", "FireANTs_SyN", "--dvf", "a.mha", "b.mha"]
    )
    cli.main()
    assert calls["uncertainty"]["preset"] == "FireANTs_SyN"


def test_unknown_subcommand_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, ["evaluate", "--help"]) == 2  # it is "eval", not "evaluate"


def test_list_shows_each_preset_with_its_display_name(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # A local preset directory: only folders whose app.json declares the registration task are presets.
    import json

    for name, display, task in (
        ("FireANTs_SyN", "FireANTs (SyN)", "registration"),
        ("ConvexAdam_Composite", "ConvexAdam (MIND)", "registration"),
        ("Evaluation_only", "Not a preset", "evaluation"),
    ):
        (tmp_path / name).mkdir()
        manifest = {"display_name": display, "short_description": f"What {display} does.", "task": task}
        (tmp_path / name / "app.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(impact_reg, "IMPACT_REG_KONFAI_REPO", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", "list"])

    cli.main()

    assert capsys.readouterr().out.splitlines()[:4] == [
        "ConvexAdam_Composite  ConvexAdam (MIND)",
        "                      What ConvexAdam (MIND) does.",
        "FireANTs_SyN          FireANTs (SyN)",
        "                      What FireANTs (SyN) does.",
    ]


def test_show_says_what_a_preset_will_do(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # 'show NAME' answers before a run, through konfai-apps: the manifest's own fields (tiling, memory), the feature
    # models (with their size once on disk), the installs, and the --set parameters with the range and meaning the
    # model's types declare. A real preset folder; the model class is imported for its signature only.
    import json
    import re

    model = tmp_path / "Feature.pt"
    model.write_bytes(b"0" * 2_500_000)
    preset = tmp_path / "presets" / "FireANTs_SyN"
    preset.mkdir(parents=True)
    manifest = {
        "display_name": "FireANTs (SyN)",
        "short_description": "SyN.",
        "description": "Rigid, affine, then SyN.",
        "task": "registration",
        "tta": 0,
        "mc_dropout": 0,
        "vram_bytes_per_voxel": 1000,
        "tiling": {"global": "Prediction.yml", "tile": "Prediction_tile.yml", "tile_vram_bytes_per_voxel": 600},
    }
    (preset / "app.json").write_text(json.dumps(manifest))
    (preset / "requirements.txt").write_text("# comment\nnibabel\n")
    (preset / "Prediction.yml").write_text(
        "Predictor:\n  Model:\n    classpath: impact_reg_konfai.models.fireants:RegistrationNet\n"
        f"    RegistrationNet:\n      cc_kernel: 5\n      models:\n        '0':\n          ref: {model}\n"
    )
    monkeypatch.setattr(impact_reg, "IMPACT_REG_KONFAI_REPO", str(preset.parent))
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", "show", "FireANTs_SyN"])
    monkeypatch.setenv("COLUMNS", "200")

    cli.main()

    out = capsys.readouterr().out
    assert "impact_reg_konfai.models.fireants:RegistrationNet\n" in out
    assert '"tile_vram_bytes_per_voxel": 600' in out and "vram_bytes_per_voxel 1000\n" in out
    assert f"{model} (2.5 MB)" in out
    assert "nibabel, on first use" in out
    # The shape, not FireANTs' own wording: the preset's value, its range or choices, a description under it.
    assert re.search(r"\n  cc_kernel = 5  \([^)]+\)\n      \w", out)
    assert f"  Predictor.Model.RegistrationNet.models.0.ref = {model}" in out

    assert _run(monkeypatch, ["show", "FireANTs_Syn"]) == 1
    err = capsys.readouterr().err
    assert "No app 'FireANTs_Syn'" in err and "Did you mean 'FireANTs_SyN'" in err


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, ("VBoussot/ImpactReg", PRESETS_REVISION)), ("me/Presets@v2", ("me/Presets", "v2"))],
)
def test_the_preset_repo_reaches_konfai_apps_with_its_revision(override: str | None, expected: tuple) -> None:
    # A release pins the presets by setting PRESETS_REVISION; konfai-apps must read it back as the revision of
    # every Hugging Face call, and KONFAI_IMPACTREG_REPO must still win. A fresh interpreter: the constant is
    # read from the environment at import.
    from konfai_apps.app_repository import LocalAppRepositoryFromHF

    env = {key: value for key, value in os.environ.items() if key != "KONFAI_IMPACTREG_REPO"}
    if override:
        env["KONFAI_IMPACTREG_REPO"] = override
    repo = subprocess.run(
        [sys.executable, "-c", "import impact_reg_konfai; print(impact_reg_konfai.PRESETS_REPO)"],
        env={**env, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert LocalAppRepositoryFromHF._split_repo_reference(repo) == expected


def _failing_app(error: BaseException) -> type:
    class _FailingApp:
        def __init__(self, **_: object) -> None:
            pass

        def register(self, *args: object, **kwargs: object) -> None:
            raise error

    return _FailingApp


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (
            KonfAIError("Apply", "'x.mha' is an input.", "Name another -o."),
            1,
            "[Apply] 'x.mha' is an input.\n→\tName another -o.",
        ),
        (
            FileNotFoundError("Path does not exist: 'nope.mha'"),
            1,
            "Path does not exist: 'nope.mha' (IMPACT_REG_DEBUG=1 prints the traceback)",
        ),
        (
            subprocess.CalledProcessError(3, ["konfai-apps", "infer", "repo:FireANTs_Syn", "-i", "a"]),
            3,
            "'konfai-apps infer repo:FireANTs_Syn' failed (exit 3); its error is printed above.",
        ),
    ],
)
def test_a_failure_is_one_line_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, error: BaseException, code: int, message: str
) -> None:
    # A preset failure has printed its own traceback already, and a missing file says it all in its message: the
    # CLI adds one line and the exit code, not a second traceback. IMPACT_REG_DEBUG=1 keeps the traceback, and a
    # line that might be a bug's (anything but a designed KonfAIError) says so.
    monkeypatch.delenv("IMPACT_REG_DEBUG", raising=False)
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _failing_app(error))
    assert _run(monkeypatch, ["register", "FireANTs_SyN", "-f", "a.mha", "-m", "b.mha"]) == code
    assert capsys.readouterr().err.strip() == message

    monkeypatch.setenv("IMPACT_REG_DEBUG", "1")
    with pytest.raises(type(error)):
        cli.main()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is TerminateProcess on Windows")
def test_sigterm_stops_the_preset_child_and_runs_the_cleanup(tmp_path) -> None:
    # Slicer's Stop sends SIGTERM, then SIGKILL 2 s later. The preset runs in a child process (konfai-apps infer)
    # under a `finally` that removes the work directory: both must be taken care of within those 2 s, not left
    # running and on disk, even when the child is slow to stop (this one ignores SIGTERM).
    import signal
    import time

    import psutil

    pid_file, cleaned = tmp_path / "child.pid", tmp_path / "cleaned"
    child = (
        "import os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    script = (
        "import pathlib, subprocess, sys\n"
        "from impact_reg_konfai import cli\n"
        "cli._stop_on_sigterm()\n"
        "try:\n"
        f"    subprocess.run([sys.executable, '-c', {child!r}])\n"
        "finally:\n"
        f"    pathlib.Path({str(cleaned)!r}).write_text('done')\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    )
    deadline = time.monotonic() + 60
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline and process.poll() is None
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    sent = time.monotonic()

    assert process.wait(timeout=30) == 128 + signal.SIGTERM
    assert time.monotonic() - sent < 2, "Slicer would have killed the CLI before its cleanup"
    assert cleaned.read_text() == "done"
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_uncertainty_needs_no_preset(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    # The spread of an ensemble needs no preset: omitting --preset must not list the presets (a network call that
    # fails offline), and a single field is refused before anything runs.
    def no_listing(*_: object, **__: object) -> list[str]:
        raise AssertionError("uncertainty listed the presets")

    calls: dict[str, dict] = {}
    monkeypatch.setattr(impact_reg, "app_names", no_listing)
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app(calls))
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", "uncertainty", "--dvf", "a.h5", "b.h5"])
    cli.main()
    assert [path.name for path in calls["uncertainty"]["dvfs"]] == ["a.h5", "b.h5"]

    assert _run(monkeypatch, ["uncertainty", "--dvf", "a.h5"]) == 2
    assert "two or more displacement fields" in capsys.readouterr().err
    assert _run(monkeypatch, ["uncertainty", "--help"]) == 0
    assert "--download" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "option",
    [["--max-voxels", "0"], ["--cpu", "0"], ["--tta", "-1"], ["--gpu", "-1"], ["--cpu", "x"]],
)
def test_a_bad_count_is_refused_by_the_parser(monkeypatch: pytest.MonkeyPatch, option: list[str]) -> None:
    calls: dict[str, dict] = {}
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app(calls))
    assert _run(monkeypatch, ["register", "FireANTs_SyN", "-f", "a.mha", "-m", "b.mha", *option]) == 2
    assert not calls


def test_a_remote_uri_is_refused_by_the_parser(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    # konfai-apps stages inputs from the local filesystem only; a URI used to become '<cwd>/s3:/bucket/...'.
    assert _run(monkeypatch, ["register", "FireANTs_SyN", "-f", "s3://bucket/f.ome.zarr", "-m", "b.mha"]) == 2
    assert "is a URI" in capsys.readouterr().err


def test_keep_fields_names_what_the_ensemble_holds(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # Members accumulate across runs into one --output; the stale one must be named, since `uncertainty --dvf
    # Ensemble/*` takes it too. --uncertainty stays the flag's former name.
    ensemble = tmp_path / "P000" / "Ensemble"
    ensemble.mkdir(parents=True)
    for member in ("FireANTs_SyN.h5", "Generic_Rigid.h5", "ConvexAdam_Coarse.h5"):
        (ensemble / member).touch()
    calls: dict[str, dict] = {}
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app(calls))
    argv = ["register", "FireANTs_SyN", "Generic_Rigid", "-f", "a.mha", "-m", "b.mha", "-o", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv, "--uncertainty"])
    cli.main()

    assert calls["register"]["keep_dvf"] is True
    assert capsys.readouterr().out.strip() == (
        f"[ImpactReg] {ensemble}: ConvexAdam_Coarse, FireANTs_SyN, Generic_Rigid "
        "(ConvexAdam_Coarse from an earlier run)."
    )


@pytest.mark.parametrize(("device", "noted"), [([], True), (["--cpu", "1"], False), (["--gpu", "0"], False)])
def test_register_notes_an_unused_gpu(monkeypatch: pytest.MonkeyPatch, capsys, device: list[str], noted: bool) -> None:
    # Omitting --gpu runs every preset on the CPU; on a machine with a GPU, say so once.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(impact_reg, "ImpactRegKonfAIApp", _stub_app({}))
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", "register", "FireANTs_SyN", "-f", "a", "-m", "b", *device])
    cli.main()
    assert ("No --gpu" in capsys.readouterr().out) is noted


def test_help_and_version_answer_without_torch() -> None:
    # --help and --version are what a first-time user types; they used to wait seconds for torch and KonfAI, which
    # only a command needs. --version names the preset source, the other half of what a run depends on.
    code = (
        "import sys\n"
        "from impact_reg_konfai import cli\n"
        "sys.argv = ['impact-reg-konfai', '--version']\n"
        "try:\n"
        "    cli.main()\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "KONFAI_IMPACTREG_REPO": "me/Presets@v2"},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("impact-reg-konfai ") and run.stdout.strip().endswith("presets: me/Presets@v2")


@pytest.mark.parametrize("name", ["organs.nii.gz", "organs.ome.zarr"])
def test_apply_brings_a_label_map_onto_the_fixed_grid(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, name: str
) -> None:
    # The transform maps fixed points to moving ones (+2 mm in x): a moving label cube at x 8..11 lands at x 6..9
    # on the fixed grid, nearest-neighbour, so no label value is invented at its edges. A real streamed resample,
    # written in the input's own form, as register writes Moved.
    import numpy as np
    import SimpleITK as sitk
    from konfai.utils.ome_zarr import read_ome_zarr_data_slice, write_ome_zarr

    labels = np.zeros((16, 16, 16), np.uint8)
    labels[6:10, 6:10, 8:12] = 3
    if name.endswith(".ome.zarr"):
        write_ome_zarr(tmp_path / name, labels[None], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    else:
        sitk.WriteImage(sitk.GetImageFromArray(labels), str(tmp_path / name))
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), np.float32)), str(tmp_path / "fixed.mha"))
    transform = sitk.AffineTransform(3)
    transform.SetTranslation((2.0, 0.0, 0.0))
    sitk.WriteTransform(transform, str(tmp_path / "Transform.h5"))

    argv = ["apply", "--transform", str(tmp_path / "Transform.h5"), "-f", str(tmp_path / "fixed.mha")]
    argv += ["-i", str(tmp_path / name), "--labels", "-o", str(tmp_path / "out"), "--cpu", "1", "-q"]
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv])
    cli.main()

    expected = np.zeros_like(labels)
    expected[6:10, 6:10, 6:10] = 3
    written = tmp_path / "out" / name
    if name.endswith(".ome.zarr"):
        moved = read_ome_zarr_data_slice(written, (slice(None),) * 4)[0][0]
    else:
        moved = sitk.GetArrayFromImage(sitk.ReadImage(str(written)))
    np.testing.assert_array_equal(moved, expected)
    assert sorted(entry.name for entry in (tmp_path / "out").iterdir()) == [name]
    assert "[ImpactReg]" not in capsys.readouterr().out  # -q


def test_apply_never_writes_over_an_input(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # An output takes its input's name: with -o the directory holding the input it would replace it, and two
    # inputs of one name would replace each other. Both are refused before anything runs, the inputs untouched.
    import numpy as np
    import SimpleITK as sitk

    (tmp_path / "b").mkdir()
    for path in [tmp_path / "organs.nii.gz", tmp_path / "b" / "organs.nii.gz", tmp_path / "fixed.mha"]:
        sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 4, 4), np.uint8)), str(path))
    sitk.WriteTransform(sitk.AffineTransform(3), str(tmp_path / "Transform.h5"))
    before = (tmp_path / "organs.nii.gz").read_bytes()
    apply = ["apply", "--transform", str(tmp_path / "Transform.h5"), "-f", str(tmp_path / "fixed.mha"), "--cpu", "1"]

    assert _run(monkeypatch, [*apply, "-i", str(tmp_path / "organs.nii.gz"), "--labels", "-o", str(tmp_path)]) == 1
    assert "is an input: its output would replace it" in capsys.readouterr().err
    inputs = ["-i", str(tmp_path / "organs.nii.gz"), str(tmp_path / "b" / "organs.nii.gz")]
    assert _run(monkeypatch, [*apply, *inputs, "-o", str(tmp_path / "out")]) == 1
    assert "Two images are named 'organs.nii.gz'" in capsys.readouterr().err

    assert (tmp_path / "organs.nii.gz").read_bytes() == before
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["Transform.h5", "b", "fixed.mha", "organs.nii.gz"]
