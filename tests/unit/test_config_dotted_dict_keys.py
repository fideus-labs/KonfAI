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

"""Regression tests: dict[str, Object] entries keyed by a dotted string.

A dotted dict key (e.g. a PerceptualLoss module path
``UNetBlock_0.DownConvBlock.Activation_1``) must be treated as a single flat
config key. Before the fix, ``Config.__init__`` split it on ``.`` into separate
navigation levels, so the user's value was never found (code defaults were used)
and the write-back exploded the key into a bogus nested subtree.
"""

from pathlib import Path

import pytest
import ruamel.yaml
from konfai.utils.config import apply_config


def _configure_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    return config_path


class _Child:
    def __init__(self, value: int = 1) -> None:
        self.value = value


class _Root:
    def __init__(self, children: dict[str, _Child] = {"a.b.c": _Child(1)}) -> None:
        self.children = children


def test_apply_config_honors_value_under_dotted_dict_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        "Root:\n  children:\n    a.b.c:\n      value: 99\n",
    )

    root = apply_config("Root")(_Root)()

    assert list(root.children) == ["a.b.c"]
    # Before the fix the dotted key was split and this was silently 1 (the code default).
    assert root.children["a.b.c"].value == 99


def test_apply_config_does_not_explode_dotted_dict_key_on_writeback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _configure_env(
        tmp_path,
        monkeypatch,
        "Root:\n  children:\n    a.b.c:\n      value: 99\n",
    )

    apply_config("Root")(_Root)()

    data = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    children = data["Root"]["children"]
    # Before the fix, children also contained an exploded ``a: {b: {c: {value: 1}}}`` subtree.
    assert set(children) == {"a.b.c"}
    assert "a" not in children
    assert children["a.b.c"]["value"] == 99


def test_apply_config_dotted_dict_key_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _configure_env(
        tmp_path,
        monkeypatch,
        "Root:\n  children:\n    a.b.c:\n      value: 99\n",
    )

    first = apply_config("Root")(_Root)()
    assert first.children["a.b.c"].value == 99
    after_first = config_path.read_text(encoding="utf-8")

    # Second run reads the written-back file: value preserved and write-back idempotent.
    second = apply_config("Root")(_Root)()
    assert second.children["a.b.c"].value == 99
    assert config_path.read_text(encoding="utf-8") == after_first


def test_apply_config_colon_and_plain_dict_keys_unaffected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Backward-compat guard: keys without ``.`` (``:``-separated module paths, plain
    # names) must bind exactly as before and must not be escaped/exploded.
    config_path = _configure_env(
        tmp_path,
        monkeypatch,
        "R:\n  m:\n    X:Head:Conv:\n      value: 5\n    plain:\n      value: 8\n",
    )

    class R:
        def __init__(self, m: dict[str, _Child] = {"X:Head:Conv": _Child(1), "plain": _Child(1)}) -> None:
            self.m = m

    root = apply_config("R")(R)()

    assert root.m["X:Head:Conv"].value == 5
    assert root.m["plain"].value == 8
    data = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert set(data["R"]["m"]) == {"X:Head:Conv", "plain"}
