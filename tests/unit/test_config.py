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

"""Tests for the config reflection engine (``konfai.utils.config``).

Covers ``Config`` file handling and error messages, ``apply_config`` type binding
(literals, unions, dicts, booleans), write-back round-trips (including dotted dict
keys), and the config env-var bookkeeping.
"""

import os
from pathlib import Path
from typing import Literal

import pytest
import ruamel.yaml
from konfai.utils.config import Config, apply_config, config
from konfai.utils.errors import ConfigError


def _fail_input(_: str) -> str:
    raise AssertionError("input should not be used")


# --------------------------------------------------------------------------------------
# Config file handling and error messages
# --------------------------------------------------------------------------------------


def test_config_missing_file_raises_clear_error_without_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "missing.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setattr("builtins.input", _fail_input)

    with pytest.raises(ConfigError) as exc_info:
        with Config("Trainer"):
            pass

    # The error must name the file, the mode, and hint at the fix.
    msg = str(exc_info.value)
    assert "missing.yml" in msg
    assert "does not exist" in msg
    assert "KONFAI_CONFIG_MODE=Done" in msg
    assert "konfai TRAINING" in msg


def test_config_default_mode_materializes_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "generated.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "default")
    monkeypatch.setattr("builtins.input", _fail_input)

    with Config("Trainer") as config_obj:
        value = config_obj.get_value("train_name", "default|SMOKE")

    assert config_path.exists()
    assert value == "SMOKE"
    content = config_path.read_text(encoding="utf-8")
    assert "Trainer:" in content
    assert "train_name: SMOKE" in content


def test_config_missing_env_var_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KONFAI_config_file", raising=False)

    with pytest.raises(KeyError):
        Config("Trainer")


def test_get_value_returns_default_when_key_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "empty.yml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "default")
    monkeypatch.setattr("builtins.input", _fail_input)

    with Config("Root") as cfg:
        value = cfg.get_value("missing_key", "default|FALLBACK")

    assert value == "FALLBACK"


def test_config_raises_on_invalid_yaml_syntax(write_config) -> None:
    write_config("key: {unclosed\n", name="broken.yml")

    with pytest.raises(ConfigError) as exc_info:
        with Config("Root"):
            pass

    msg = str(exc_info.value)
    assert "Invalid YAML syntax" in msg
    assert "broken.yml" in msg


def test_type_mismatch_error_names_field_and_type(write_config) -> None:
    write_config("Root:\n  count: hello\n")

    class Root:
        def __init__(self, count: int = 0) -> None:
            self.count = count

    with pytest.raises(ConfigError) as exc_info:
        apply_config("Root")(Root)()

    msg = str(exc_info.value)
    assert "count" in msg
    assert "int" in msg


# --------------------------------------------------------------------------------------
# apply_config type binding
# --------------------------------------------------------------------------------------


def test_apply_config_preserves_none_for_optional_nested_objects(write_config) -> None:
    write_config("Root:\n  child: None\n")

    @config("child")
    class Child:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    class Root:
        def __init__(self, child: Child | None = None) -> None:
            self.child = child

    root = apply_config("Root")(Root)()

    assert root.child is None


def test_apply_config_keeps_a_none_default_when_the_config_is_silent(write_config) -> None:
    """A config that never names an ``X | None = None`` object leaves it None, and records that.

    Binding it anyway would construct X and write X's OWN defaults back, so a model declaring "no
    patch" ran with a patch nobody configured -- and the resolved config, the record of the run,
    described that phantom instead of what happened.
    """
    config_path = write_config("Root: {}\n")

    @config("Child")
    class Child:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    class Root:
        def __init__(self, child: Child | None = None) -> None:
            self.child = child

    root = apply_config("Root")(Root)()

    assert root.child is None
    # "None" (the string) is how KonfAI spells an absent object in a resolved config, as every
    # `overlap: None` / `augmentations: None` in the shipped configs does.
    written = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert written["Root"]["Child"] == "None"


def test_apply_config_binds_an_optional_with_a_non_none_default(write_config) -> None:
    """``X | None = X()`` is the opposite declaration: the object exists unless the config says None.

    This is what keeps ``DatasetPatch``-style defaults instantiated by a silent config.
    """
    write_config("Root: {}\n")

    @config("Child")
    class Child:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    class Root:
        def __init__(self, child: Child | None = Child()) -> None:
            self.child = child

    root = apply_config("Root")(Root)()

    assert isinstance(root.child, Child)
    assert root.child.value == 1


def test_apply_config_accepts_literal_value(write_config) -> None:
    write_config("Root:\n  mode: eval\n")

    class Root:
        def __init__(self, mode: Literal["train", "eval"] = "train") -> None:
            self.mode = mode

    root = apply_config("Root")(Root)()

    assert root.mode == "eval"


def test_apply_config_binds_optional_literal(write_config) -> None:
    # Literal[...] | None unwraps to Literal[...] and must still bind as a literal, not fall through to
    # object instantiation.
    write_config("Root:\n  mode: eval\n")

    class Root:
        def __init__(self, mode: Literal["train", "eval"] | None = None) -> None:
            self.mode = mode

    assert apply_config("Root")(Root)().mode == "eval"


def test_apply_config_materializes_non_string_literal_default(write_config) -> None:
    # A non-string Literal default is written back as the "default|X" marker; on re-bind it must
    # recover the correctly-typed member (int/bool), not fail the membership check on a string "X".
    write_config("Root:\n  level: default|2\n  flag: default|True\n")

    class Root:
        def __init__(self, level: Literal[1, 2, 3] = 1, flag: Literal[True, False] = True) -> None:
            self.level = level
            self.flag = flag

    root = apply_config("Root")(Root)()

    assert root.level == 2 and isinstance(root.level, int)
    assert root.flag is True


def test_apply_config_rejects_invalid_literal_value(write_config) -> None:
    write_config("Root:\n  mode: invalid\n")

    class Root:
        def __init__(self, mode: Literal["train", "eval"] = "train") -> None:
            self.mode = mode

    with pytest.raises(ConfigError, match="Invalid value 'invalid'") as exc_info:
        apply_config("Root")(Root)()

    # The error must mention the valid options.
    msg = str(exc_info.value)
    assert "train" in msg or "eval" in msg


@pytest.mark.parametrize(
    ("literal", "expected"),
    [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("no", False)],
)
def test_apply_config_parses_boolean_strings(write_config, literal: str, expected: bool) -> None:
    write_config(f"Root:\n  enabled: '{literal}'\n")

    class Root:
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

    assert apply_config("Root")(Root)().enabled is expected


def test_apply_config_rejects_unknown_boolean_string(write_config) -> None:
    write_config("Root:\n  enabled: 'sometimes'\n")

    class Root:
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

    with pytest.raises(ConfigError, match="expected bool"):
        apply_config("Root")(Root)()


def test_apply_config_instantiates_dict_of_nested_objects(write_config) -> None:
    write_config("Root:\n  children:\n    left:\n      value: 3\n    right:\n      value: 7\n")

    class Child:
        def __init__(self, value: int) -> None:
            self.value = value

    class Root:
        def __init__(self, children: dict[str, Child]) -> None:
            self.children = children

    root = apply_config("Root")(Root)()

    assert sorted(root.children) == ["left", "right"]
    assert root.children["left"].value == 3
    assert root.children["right"].value == 7


def test_apply_config_preserves_dict_of_primitives(write_config) -> None:
    write_config("Root:\n  weights:\n    mae: 1\n    ssim: 2\n")

    class Root:
        def __init__(self, weights: dict[str, int]) -> None:
            self.weights = weights

    root = apply_config("Root")(Root)()

    assert root.weights == {"mae": 1, "ssim": 2}


def test_apply_config_converts_sequence_of_union_scalars(write_config) -> None:
    write_config("Root:\n  values:\n    - '1'\n    - 2\n    - '3'\n")

    class Root:
        def __init__(self, values: list[int | float]) -> None:
            self.values = values

    root = apply_config("Root")(Root)()

    assert root.values == [1, 2, 3]
    assert all(isinstance(value, int) for value in root.values)


def test_apply_config_union_keeps_the_value_type_over_lossy_coercion(write_config) -> None:
    # A value whose YAML type already matches a union member must bind unchanged: coercing in
    # declaration order turns ``overlap: 0.25`` into ``int(0.25) == 0`` (silent no overlap),
    # lets ``str`` swallow a list, and never reaches a ``list[...]`` member at all.
    write_config("Root:\n  frac: 0.25\n  voxels: 8\n  percent: '20%'\n  per_axis:\n    - 10\n    - 20\n    - 0\n")

    class Root:
        def __init__(
            self,
            frac: int | float | str | list[int] | None = None,
            voxels: int | float | str | list[int] | None = None,
            percent: int | float | str | list[int] | None = None,
            per_axis: int | float | str | list[int] | None = None,
        ) -> None:
            self.frac = frac
            self.voxels = voxels
            self.percent = percent
            self.per_axis = per_axis

    root = apply_config("Root")(Root)()

    assert root.frac == 0.25 and isinstance(root.frac, float)  # not int(0.25) == 0
    assert root.voxels == 8 and isinstance(root.voxels, int)
    assert root.percent == "20%"
    assert list(root.per_axis) == [10, 20, 0] and isinstance(root.per_axis, list)  # not the string "[10, 20, 0]"


def test_apply_config_binds_scalar_float_or_str_union(write_config) -> None:
    # Mirrors the Clip transform (``min_value``/``max_value: float | str``) which accepts numeric
    # bounds as well as string sentinels such as ``min`` / ``percentile:99.5``.
    write_config("Root:\n  low: min\n  high: 'percentile:99.5'\n  fixed: 1024\n")

    class Root:
        def __init__(
            self,
            low: float | str = 0.0,
            high: float | str = 0.0,
            fixed: float | str = 0.0,
        ) -> None:
            self.low = low
            self.high = high
            self.fixed = fixed

    root = apply_config("Root")(Root)()

    assert root.low == "min"
    assert root.high == "percentile:99.5"
    assert root.fixed == 1024.0
    assert isinstance(root.fixed, float)


def test_apply_config_honors_konfai_without_for_skipped_parameters(write_config) -> None:
    write_config("Root:\n  kept: 5\n  skipped: 42\n")

    class Root:
        def __init__(self, kept: int, skipped: int = 0) -> None:
            self.kept = kept
            self.skipped = skipped

    root = apply_config("Root")(Root)(konfai_without=["skipped"])

    assert root.kept == 5
    assert root.skipped == 0


# --------------------------------------------------------------------------------------
# Write-back round-trips
# --------------------------------------------------------------------------------------


class _RoundTripRoot:
    def __init__(self, weights: dict[str, int] = {"mae": 1, "ssim": 2}) -> None:
        self.weights = weights


def test_dict_of_primitives_default_round_trips(write_config) -> None:
    config_path = write_config("Root: {}\n")  # Root present but no 'weights'

    # Run 1: the default materialises and is written back.
    first = apply_config("Root")(_RoundTripRoot)()
    assert first.weights == {"mae": 1, "ssim": 2}

    # The write-back must persist the values, not collapse the dict to an empty mapping.
    written = config_path.read_text(encoding="utf-8")
    assert "mae" in written and "ssim" in written

    # Run 2: reading the written file must return the same dict, never {}.
    second = apply_config("Root")(_RoundTripRoot)()
    assert second.weights == {"mae": 1, "ssim": 2}


# A dotted dict key (e.g. a PerceptualLoss module path ``UNetBlock_0.DownConvBlock.Activation_1``)
# must be treated as a single flat config key. Splitting it on ``.`` into navigation levels means
# the user's value is never found (code defaults bind) and the write-back explodes the key into a
# bogus nested subtree.


class _DottedChild:
    def __init__(self, value: int = 1) -> None:
        self.value = value


class _DottedRoot:
    def __init__(self, children: dict[str, _DottedChild] = {"a.b.c": _DottedChild(1)}) -> None:
        self.children = children


def test_apply_config_honors_value_under_dotted_dict_key(write_config) -> None:
    write_config("Root:\n  children:\n    a.b.c:\n      value: 99\n")

    root = apply_config("Root")(_DottedRoot)()

    assert list(root.children) == ["a.b.c"]
    # A split dotted key would leave this silently 1 (the code default).
    assert root.children["a.b.c"].value == 99


def test_apply_config_does_not_explode_dotted_dict_key_on_writeback(write_config) -> None:
    config_path = write_config("Root:\n  children:\n    a.b.c:\n      value: 99\n")

    apply_config("Root")(_DottedRoot)()

    data = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    children = data["Root"]["children"]
    # No exploded ``a: {b: {c: {value: 1}}}`` subtree may appear beside the flat key.
    assert set(children) == {"a.b.c"}
    assert "a" not in children
    assert children["a.b.c"]["value"] == 99


def test_apply_config_dotted_dict_key_round_trips(write_config) -> None:
    config_path = write_config("Root:\n  children:\n    a.b.c:\n      value: 99\n")

    first = apply_config("Root")(_DottedRoot)()
    assert first.children["a.b.c"].value == 99
    after_first = config_path.read_text(encoding="utf-8")

    # Second run reads the written-back file: value preserved and write-back idempotent.
    second = apply_config("Root")(_DottedRoot)()
    assert second.children["a.b.c"].value == 99
    assert config_path.read_text(encoding="utf-8") == after_first


def test_apply_config_colon_and_plain_dict_keys_unaffected(write_config) -> None:
    # Backward-compat guard: keys without ``.`` (``:``-separated module paths, plain
    # names) must bind as single flat keys and must not be escaped/exploded.
    config_path = write_config("R:\n  m:\n    X:Head:Conv:\n      value: 5\n    plain:\n      value: 8\n")

    class R:
        def __init__(self, m: dict[str, _DottedChild] = {"X:Head:Conv": _DottedChild(1), "plain": _DottedChild(1)}):
            self.m = m

    root = apply_config("R")(R)()

    assert root.m["X:Head:Conv"].value == 5
    assert root.m["plain"].value == 8
    data = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert set(data["R"]["m"]) == {"X:Head:Conv", "plain"}


# --------------------------------------------------------------------------------------
# Config env-var bookkeeping
# --------------------------------------------------------------------------------------


def test_apply_config_restores_config_env(write_config, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config("Root:\n  Child:\n    value: 7\n")
    monkeypatch.setenv("KONFAI_CONFIG_PATH", "before.path")
    monkeypatch.setenv("KONFAI_CONFIG_VARIABLE", "before.variable")

    @config("Child")
    class Child:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    child = apply_config("Root")(Child)()

    assert child.value == 7
    assert os.environ["KONFAI_CONFIG_PATH"] == "before.path"
    assert os.environ["KONFAI_CONFIG_VARIABLE"] == "before.variable"


def test_apply_config_keeps_config_path_during_constructor_call(write_config) -> None:
    write_config("Root:\n  Child:\n    value: 7\n")

    @config("Child")
    class Child:
        def __init__(self, value: int = 0) -> None:
            self.value = value
            self.config_path = os.environ["KONFAI_CONFIG_PATH"]

    child = apply_config("Root")(Child)()

    assert child.value == 7
    assert child.config_path == "Root.Child"


def test_a_block_type_outside_its_two_names_is_refused(tmp_path: Path, monkeypatch) -> None:
    # A `block_type` str tested only for "Conv" builds the residual model for every other value, a
    # typo included -- another architecture, another checkpoint, and nothing says so.
    from konfai.models.python.segmentation.UNet import UNet
    from konfai.utils.config import apply_config

    config = tmp_path / "Config.yml"
    config.write_text("M:\n  block_type: Cnov\n  channels: [1, 8, 16]\n  nb_class: 2\n")
    monkeypatch.setenv("KONFAI_config_file", str(config))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")

    with pytest.raises(ConfigError) as error:
        apply_config("M")(UNet)()
    assert "'Conv', 'Res'" in str(error.value)


def test_union_with_literal_member_does_not_crash() -> None:
    # A typing-only origin (Literal) is not a class: the runtime-match fast path must skip it
    # instead of raising TypeError in isinstance, and let another member bind the value.
    from konfai.utils.config import _convert_union_sequence_value

    assert _convert_union_sequence_value("beta", (Literal["alpha", "beta"], str), "p") == "beta"
