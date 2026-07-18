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

"""Contract test for the ``konfai-apps`` CLI argv and ``get_parameters()`` JSON shape driven by SlicerKonfAI.

SlicerKonfAI (a separate repository outside this CI) uses konfai-apps in exactly two ways, so a refactor that
renames a CLI flag or changes the ``{values, constraints}`` shape breaks a clinician's Slicer session instead
of failing here. This test freezes both, sourced from the Slicer repo:

- CLI argv: the ``konfai-apps infer`` / ``eval`` commands Slicer spawns as a subprocess
  (``KonfAILib/widgets/panels/inference.py`` and ``.../panels/qa.py``). Load-bearing flags/dests: the
  subcommand names ``infer``/``eval``, ``-i``/``-o``, ``--set NAME=VALUE`` (dest ``config_overrides``),
  ``--patch-size`` (dest ``patch_size``), ``--batch-size`` (dest ``batch_size``), ``--download``, ``--gpu``,
  ``--cpu``, ``--gt``, ``--mask``.
- get_parameters() JSON: the Advanced dialog renders ``app.get_parameters()`` -> ``{"values", "constraints"}``
  and reads each constraint's ``choices`` / ``min`` / ``max`` keys (``.../panels/inference.py``:
  ``_build_value_editor``).

It intentionally freezes only what Slicer actually uses. If Slicer starts using more, add a case here.
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from konfai.utils.config import Choices, Range
from konfai_apps import app as app_module
from konfai_apps import app_repository as app_repository_module
from konfai_apps.cli import main_apps


class _CapturingApp:
    """Stub standing in for ``KonfAIApp``: records how ``main_apps`` constructed and dispatched it, so the
    test asserts the parsed CLI reaches ``.infer``/``.evaluate`` without running any real inference."""

    last: "_CapturingApp | None" = None

    def __init__(self, app: str, download: bool, force_update: bool) -> None:
        self.app = app
        self.download = download
        self.force_update = force_update
        self.infer_kwargs: dict[str, Any] | None = None
        self.evaluate_kwargs: dict[str, Any] | None = None
        _CapturingApp.last = self

    def infer(self, **kwargs: Any) -> None:
        self.infer_kwargs = kwargs

    def evaluate(self, **kwargs: Any) -> None:
        self.evaluate_kwargs = kwargs


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> _CapturingApp:
    monkeypatch.setattr(app_module, "KonfAIApp", _CapturingApp)
    monkeypatch.setattr(sys, "argv", ["konfai-apps", *argv])
    main_apps()
    assert _CapturingApp.last is not None
    return _CapturingApp.last


def test_infer_argv_slicer_builds_parses_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _run_cli(
        monkeypatch,
        [
            "infer",
            "MyApp",
            "-i",
            "Volume.mha",
            "-o",
            "Output",
            "--ensemble_models",
            "CV_0",
            "--tta",
            "0",
            "--mc",
            "0",
            "--patch-size",
            "192",
            "192",
            "192",
            "--batch-size",
            "2",
            "--set",
            "iterations=300",
            "--download",
            "--gpu",
            "0",
        ],
    )

    assert app.app == "MyApp"
    assert app.download is True
    assert app.infer_kwargs is not None
    kwargs = app.infer_kwargs
    # dests Slicer's flags map to -- a rename would silently break its subprocess call.
    assert kwargs["config_overrides"] == ["iterations=300"]
    assert kwargs["patch_size"] == [192, 192, 192]
    assert kwargs["batch_size"] == 2
    assert kwargs["gpu"] == [0]
    assert kwargs["ensemble_models"] == ["CV_0"]
    assert "inputs" in kwargs and "output" in kwargs


def test_eval_argv_slicer_builds_parses_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _run_cli(
        monkeypatch,
        [
            "eval",
            "MyApp",
            "-i",
            "Volume.mha",
            "--gt",
            "Reference.mha",
            "-o",
            "Evaluation",
            "--mask",
            "Mask.mha",
            "--cpu",
            "1",
        ],
    )

    assert app.evaluate_kwargs is not None
    kwargs = app.evaluate_kwargs
    assert kwargs["cpu"] == 1
    # --gt / --mask / -i use action="append" + nargs="+", so each is a list-of-lists Slicer relies on.
    assert kwargs["gt"] and kwargs["gt"][0]
    assert kwargs["mask"] and kwargs["mask"][0]
    assert "inputs" in kwargs and "output" in kwargs


def test_get_parameters_returns_values_and_constraints_shape(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    repo = app_repository_module.get_app_repository_info(str(app_dir), False)
    params = repo.get_parameters()

    # The Advanced dialog reads params["values"] and params["constraints"] -- both must always be present.
    assert set(params) == {"values", "constraints"}
    assert isinstance(params["values"], dict)
    assert isinstance(params["constraints"], dict)


def test_constraint_vocabulary_matches_slicer_value_editor() -> None:
    class _TypedModel:
        def __init__(
            self,
            mode: Literal["a", "b"] = "a",
            iters: Annotated[int, Range(0, 10)] = 5,
            kind: Annotated[str, Choices(["x", "y"])] = "x",
        ) -> None:
            pass

    constraints = app_repository_module._constraints_of_class(_TypedModel)

    # Exactly the keys Slicer's _build_value_editor consumes: choices -> dropdown, {min,max} -> spinbox bounds.
    # No description was given, so the shape stays exactly as Slicer expects (no extra keys).
    assert constraints["mode"] == {"choices": ["a", "b"]}
    assert constraints["iters"] == {"min": 0, "max": 10}
    assert constraints["kind"] == {"choices": ["x", "y"]}


def test_constraint_surfaces_parameter_description() -> None:
    """A bare string in Annotated adds the knob's meaning to its constraint, for any base type -- so an agent
    tuning it knows what it does. It is additive: Slicer ignores the extra key, bounds/choices are unchanged."""

    class _DocumentedModel:
        def __init__(
            self,
            iterations: Annotated[int, Range(0, 1000), "Optimization steps; higher = more accurate, slower."] = 150,
            metric: Annotated[Literal["L1", "L2"], "Similarity metric on the MIND features."] = "L1",
        ) -> None:
            pass

    constraints = app_repository_module._constraints_of_class(_DocumentedModel)

    assert constraints["iterations"] == {
        "min": 0,
        "max": 1000,
        "description": "Optimization steps; higher = more accurate, slower.",
    }
    # A described Literal keeps its choices AND gains the meaning -- the case a description= field could not cover.
    assert constraints["metric"] == {
        "choices": ["L1", "L2"],
        "description": "Similarity metric on the MIND features.",
    }
