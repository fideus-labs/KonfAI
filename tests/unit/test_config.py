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

import functools
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest
import ruamel.yaml
from konfai.utils.config import Config, _load_tree, _write_tree, apply_config, config, strict_config
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
    patch" ran with a patch nobody configured, and the resolved config, the record of the run,
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


def test_predictor_binds_check_training_transforms_from_the_file(write_config, monkeypatch) -> None:
    """``check_training_transforms: false`` under ``Predictor:`` reaches the constructor.

    The key silences the warning a prediction raises when a model input is not preprocessed the way
    its checkpoint trained on it; bound off ``Predictor``'s own signature here, so a rename of the
    parameter shows up as a key that no longer binds. The body is captured rather than run: what is
    at stake is the binding, not the workflow.
    """
    from konfai.predictor import Predictor

    write_config("Predictor:\n  check_training_transforms: false\n")
    bound: dict[str, object] = {}

    @functools.wraps(Predictor.__init__)
    def capture(_self, **kwargs) -> None:
        bound.update(kwargs)

    monkeypatch.setattr(Predictor, "__init__", capture)

    apply_config()(Predictor)()

    assert bound["check_training_transforms"] is False


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


def test_apply_config_refuses_a_nested_block_where_a_value_is_expected(write_config) -> None:
    """A mapping where the annotation says ``str`` is a parse error, not a stringified mapping.

    ``str(CommentedMap)`` never fails, so a union containing ``str`` would otherwise bind a nested
    block to its repr, and the failure would surface far from the config line that caused it ("All
    data entries were excluded": a matching symptom for a parsing cause).
    """
    write_config("Root:\n  subset:\n    CT:\n      - CASE_000\n      - CASE_001\n")

    class Root:
        def __init__(self, subset: str | list[int] | list[str] | None = None) -> None:
            self.subset = subset

    with pytest.raises(ConfigError) as error:
        apply_config("Root")(Root)()

    assert "subset" in str(error.value) and "nested block" in str(error.value)


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


def test_config_write_back_is_atomic_when_the_rename_fails(write_config, monkeypatch) -> None:
    """A failed rename leaves the on-disk config byte-identical: content reaches it only via os.replace."""
    config_path = write_config("Root:\n  count: 3\n")
    original = config_path.read_bytes()

    class Root:
        def __init__(self, count: int = 0) -> None:
            self.count = count

    replace_calls: list[tuple[str, str]] = []

    def failing_replace(src: object, dst: object) -> None:
        replace_calls.append((str(src), str(dst)))
        raise RuntimeError("crash at rename")

    # A non-OSError: OSError is the Windows in-place fallback, pinned separately below.
    monkeypatch.setattr("konfai.utils.config.os.replace", failing_replace)

    with pytest.raises(RuntimeError, match="crash at rename"):
        apply_config("Root")(Root)()

    assert replace_calls, "the write-back must land through os.replace, never a bare open(target, 'w')"
    assert config_path.read_bytes() == original  # a concurrent reader never sees a truncated config
    assert list(config_path.parent.glob("*.tmp")) == []  # the temp file is removed on failure


def test_config_write_back_retries_a_denied_rename_and_never_writes_in_place(write_config, monkeypatch) -> None:
    """A transient OSError from os.replace (Windows: an indexer holding the target) is retried and the
    file is replaced whole; a persistent one refuses with a ConfigError and leaves the file as it was.
    Never an in-place rewrite: a concurrent reader would see it truncated and bind all-defaults."""
    config_path = write_config("Root:\n  count: 3\n")

    class Root:
        def __init__(self, count: int = 0) -> None:
            self.count = count

    real_replace = os.replace
    denials: list[int] = []

    def transient_denial(src: object, dst: object) -> None:
        if len(denials) < 2:
            denials.append(1)
            raise OSError("target busy")
        real_replace(src, dst)

    monkeypatch.setattr("konfai.utils.config.os.replace", transient_denial)
    monkeypatch.setattr("konfai.utils.config.time.sleep", lambda _seconds: None)
    root = apply_config("Root")(Root)()
    assert root.count == 3 and len(denials) == 2
    data = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert data == {"Root": {"count": 3}}
    assert list(config_path.parent.glob("*.tmp")) == []

    monkeypatch.setattr("konfai.utils.config.os.replace", lambda src, dst: (_ for _ in ()).throw(OSError("held")))
    before = config_path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="atomically"):
        apply_config("Root")(Root)()
    assert config_path.read_text(encoding="utf-8") == before  # the file was left unchanged
    assert list(config_path.parent.glob("*.tmp")) == []


def test_a_reader_never_sees_the_config_a_writer_is_replacing(write_config) -> None:
    """A run rewrites the config it read while an independent launch may be reading it: that reader
    must see the whole old document or the whole new one. The write is temp + ``os.replace``, so the
    two never overlap; an in-place rewrite would hand the reader a truncated document, and every key
    it lost would silently bind its default.

    The document is padded to ~70 KB because that is what makes the failure visible: an in-place
    rewrite of a two-line file is nearly instantaneous, and the race would be missed.
    """
    config_path = write_config("Root:\n  count: 1\n")
    padding = {f"key_{index}": "v" * 256 for index in range(256)}
    trees = [{"Root": {"count": count, **padding}} for count in (1, 2)]
    _write_tree(config_path, trees[0])

    done = threading.Event()
    observed: list[str] = []
    reads = 0

    def rewrite() -> None:
        try:
            for index in range(20):
                _write_tree(config_path, trees[index % 2])
        except ConfigError:
            # Windows denies os.replace while a reader holds the target; the retry loop and its
            # refusal are pinned above. Here the reader's observation is what matters.
            pass
        finally:
            done.set()

    def reread() -> None:
        nonlocal reads
        while not done.is_set():
            try:
                root = _load_tree(config_path).get("Root", {})
            except ConfigError as error:  # a truncated document is not YAML the loader accepts
                observed.append(f"unreadable: {error}")
                return
            if root.get("count") not in (1, 2) or len(root) != len(padding) + 1:
                observed.append(f"partial: count={root.get('count')!r}, {len(root)} keys")
                return
            reads += 1

    writer, reader = threading.Thread(target=rewrite), threading.Thread(target=reread)
    reader.start()
    writer.start()
    writer.join(timeout=60)
    reader.join(timeout=60)

    assert observed == []
    assert reads > 1, "the reader read nothing while the writer ran, so it observed nothing"
    assert not writer.is_alive() and not reader.is_alive()
    assert list(config_path.parent.glob("*.tmp")) == []


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
    # typo included: another architecture, another checkpoint, and nothing says so.
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


# --------------------------------------------------------------------------------------
# strict_config: a key nothing reads is refused, whoever would have read it
# --------------------------------------------------------------------------------------


class _Leaf:
    def __init__(self, depth: int = 1) -> None:
        self.depth = depth


@config("Nested")
class _Nested:
    def __init__(self, width: int = 2) -> None:
        self.width = width


class _StrictRoot:
    def __init__(self, kept: int = 0, leaf: _Leaf = _Leaf(), nested: _Nested = _Nested()) -> None:
        self.kept, self.leaf, self.nested = kept, leaf, nested


def test_strict_config_refuses_a_key_nothing_reads_with_its_path_and_the_closest_key(write_config) -> None:
    write_config("Root:\n  kept: 1\n  kep: 2\n  Nested:\n    widht: 3\n")
    with pytest.raises(ConfigError) as raised, strict_config("Root"):
        apply_config("Root")(_StrictRoot)()
    message = str(raised.value)
    assert "'Root.kep'" in message and "Did you mean 'kept'?" in message
    assert "'Root.Nested.widht'" in message and "Did you mean 'width'?" in message
    # What the level's readers name: the parent's own parameters, the flat child's, the keyed child's key.
    assert "'Root.kep' (keys read at that level: ['Nested', 'depth', 'kept'])" in message


def test_strict_config_counts_what_a_flat_child_and_a_keyed_child_read(write_config) -> None:
    """``leaf`` has no @config key: it binds on the SAME level as its parent and reads ``depth``
    there; ``Nested`` owns a sub-level, and its key is read by the parent. Neither is unknown."""
    write_config("Root:\n  kept: 1\n  depth: 4\n  Nested:\n    width: 5\n")
    with strict_config("Root"):
        root = apply_config("Root")(_StrictRoot)()
    assert (root.kept, root.leaf.depth, root.nested.width) == (1, 4, 5)


def test_strict_config_counts_a_konfai_without_parameter_as_read(write_config) -> None:
    write_config("Root:\n  kept: 5\n  skipped: 42\n")

    class Root:
        def __init__(self, kept: int, skipped: int = 0) -> None:
            self.kept, self.skipped = kept, skipped

    with strict_config("Root"):
        root = apply_config("Root")(Root)(konfai_without=["skipped"])
    assert (root.kept, root.skipped) == (5, 0)


def test_strict_config_leaves_a_dict_of_objects_entries_free_and_checks_inside_them(write_config) -> None:
    write_config("Root:\n  children:\n    left:\n      value: 3\n    right:\n      valeu: 7\n")

    class Child:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    class Root:
        def __init__(self, children: dict[str, Child]) -> None:
            self.children = children

    with (
        pytest.raises(ConfigError, match=r"'Root\.children\.right\.valeu'.*Did you mean 'value'"),
        strict_config("Root"),
    ):
        apply_config("Root")(Root)()


def test_strict_config_refuses_a_missing_root_before_anything_binds(write_config) -> None:
    write_config("Rot:\n  kept: 2\n")
    with pytest.raises(ConfigError, match="declares no 'Root' root"), strict_config("Root"):
        raise AssertionError("refused before anything binds")


def test_strict_config_can_warn_instead_of_refusing(write_config) -> None:
    """The legacy workflows' setting: existing files carry keys older versions wrote back, so the
    reader is told and the run goes on."""
    write_config("Root:\n  kep: 2\n")
    with pytest.warns(UserWarning, match=r"'Root\.kep'.*Did you mean 'kept'"), strict_config("Root", refuse=False):
        root = apply_config("Root")(_StrictRoot)()
    assert root.kept == 0


def test_outside_strict_config_the_binder_records_nothing_and_refuses_nothing(write_config) -> None:
    from konfai.utils import config as config_module

    write_config("Root:\n  kep: 2\n")
    root = apply_config("Root")(_StrictRoot)()
    assert root.kept == 0 and config_module._ledgers == []


def test_strict_config_reports_a_yaml_syntax_error_as_a_config_error(write_config) -> None:
    write_config("Root:\n  kept: [1, 2\n")
    with pytest.raises(ConfigError, match=r"Invalid YAML syntax .* at line"), strict_config("Root"):
        raise AssertionError("refused before anything binds")


# --------------------------------------------------------------------------------------
# strict_config: one read, one write, the same bytes as a write per context
# --------------------------------------------------------------------------------------


class _Entry:
    def __init__(self, value: int = 1, tags: list[str] = ["a", "b"]) -> None:
        self.value, self.tags = value, tags


@config("Optional")
class _Optional:
    def __init__(self, depth: int = 3) -> None:
        self.depth = depth


class _WideRoot:
    """Every shape the binder writes back: primitives, a flat child, a keyed child, a dict of
    objects the file holds and one it lacks, a dict of primitives, a list, a literal, an optional
    left None."""

    def __init__(
        self,
        kept: int = 0,
        leaf: _Leaf = _Leaf(),
        present: dict[str, _Entry] = {"only": _Entry()},
        nested: _Nested = _Nested(),
        entries: dict[str, _Entry] = {"first": _Entry(), "second": _Entry(5)},
        weights: dict[str, float] = {"mae": 1.0, "ssim": 0.5},
        shape: list[int] = [1, 256, 256],
        mode: Literal["train", "eval"] = "train",
        optional: _Optional | None = None,
        name: str = "run",
    ) -> None:
        self.kept, self.leaf, self.present, self.nested, self.entries = kept, leaf, present, nested, entries
        self.weights, self.shape, self.mode, self.optional, self.name = weights, shape, mode, optional, name


_PARTIAL_WIDE_CONFIG = """Root:  # a comment on the root
  name: bound  # a comment on a key
  present:
    only:
      tags: [x, y]  # a flow-style list, a missing sibling key
  shape: [2, 2, 2]
  Nested:
    width: 7
"""


def _binder_io(monkeypatch) -> dict[str, int]:
    from konfai.utils import config as config_module

    counts = {"loads": 0, "dumps": 0}
    load, dump = config_module._load_tree, config_module.yaml.dump

    def counted_load(filename):
        counts["loads"] += 1
        return load(filename)

    def counted_dump(*args, **kwargs):
        counts["dumps"] += 1
        return dump(*args, **kwargs)

    monkeypatch.setattr(config_module, "_load_tree", counted_load)
    monkeypatch.setattr(config_module.yaml, "dump", counted_dump)
    return counts


def test_a_strict_block_reads_once_writes_once_and_the_bytes_a_write_per_context_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The block resolves every context against one in-memory tree and writes the file at its end.
    The per-context path (a context outside any block loads and writes the file itself) is the
    oracle: the resolved file must be byte-identical, key order and comments included."""
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    resolved: dict[str, bytes] = {}
    for spelling in ("per_context", "strict_block"):
        path = tmp_path / f"{spelling}.yml"
        path.write_text(_PARTIAL_WIDE_CONFIG, encoding="utf-8")
        monkeypatch.setenv("KONFAI_config_file", str(path))
        counts = _binder_io(monkeypatch)
        if spelling == "strict_block":
            with strict_config("Root"):
                root = apply_config("Root")(_WideRoot)()
            assert (counts["loads"], counts["dumps"]) == (1, 1)
        else:
            root = apply_config("Root")(_WideRoot)()
            assert counts["loads"] > 1 and counts["dumps"] > 1
        assert (root.name, root.nested.width, root.present["only"].tags, root.shape) == (
            "bound",
            7,
            ["x", "y"],
            [2, 2, 2],
        )
        assert list(root.entries) == ["first", "second"] and root.optional is None
        resolved[spelling] = path.read_bytes()
    assert resolved["strict_block"] == resolved["per_context"]
    # Comments and the flow style of what the file held are kept; what the contexts appended lands
    # where the file-backed write-back put it: a context's own keys at its exit, after the keys the
    # contexts opened inside it appended (``depth`` from the flat child, ``entries`` from the dict's
    # entries, before ``kept``, which the root set first).
    # The dump goes through text mode, so the line ending is the platform's; the layout is not.
    assert resolved["strict_block"].decode("utf-8").replace("\r\n", "\n") == (
        "Root:  # a comment on the root\n"
        "  name: bound  # a comment on a key\n"
        "  present:\n"
        "    only:\n"
        "      tags: [x, y]  # a flow-style list, a missing sibling key\n"
        "      value: 1\n"
        "  shape: [2, 2, 2]\n"
        "  Nested:\n"
        "    width: 7\n"
        "  depth: 1\n"
        "  entries:\n"
        "    first:\n"
        "      value: 1\n"
        "      tags:\n"
        "      - a\n"
        "      - b\n"
        "    second:\n"
        "      value: 1\n"
        "      tags:\n"
        "      - a\n"
        "      - b\n"
        "  kept: 0\n"
        "  weights:\n"
        "    mae: 1.0\n"
        "    ssim: 0.5\n"
        "  mode: train\n"
        "  Optional: None\n"
    )


def test_a_strict_block_that_refuses_still_leaves_what_it_bound_on_disk(write_config) -> None:
    """A context wrote its level before the block could report an unknown key; the block writes
    the same resolved file before it reports."""
    config_path = write_config("Root:\n  kep: 2\n")
    with pytest.raises(ConfigError, match="Unknown key"), strict_config("Root"):
        apply_config("Root")(_StrictRoot)()
    written = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert written["Root"] == {"kep": 2, "depth": 1, "kept": 0, "Nested": {"width": 2}}


def test_a_strict_block_does_not_create_a_file_a_context_refused(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "missing.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    with pytest.raises(ConfigError, match="does not exist"), strict_config("Root"):
        apply_config("Root")(_StrictRoot)()
    assert not config_path.exists()


def test_a_strict_block_writes_the_file_a_context_materialized(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "generated.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "default")
    monkeypatch.setattr("builtins.input", _fail_input)
    with strict_config("Root"):
        root = apply_config("Root")(_StrictRoot)()
    assert root.kept == 0
    written = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert written["Root"] == {"depth": 1, "kept": 0, "Nested": {"width": 2}}


def test_a_block_opened_inside_another_one_on_the_same_file_keeps_both_blocks_writes(write_config) -> None:
    """One file, two blocks: what the inner one bound is on disk beside what the outer one bound.

    Each block loading its own tree would have the outer's flush, of a tree read before the inner
    block existed, land last and take the inner's roots back to what the file held."""
    config_path = write_config("Root:\n  kept: 1\nOther:\n  kept: 2\n")
    with strict_config("Root"):
        outer = apply_config("Root")(_StrictRoot)()
        with strict_config("Other"):
            inner = apply_config("Other")(_StrictRoot)()
    assert (outer.kept, inner.kept) == (1, 2)
    written = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    assert written["Root"] == {"kept": 1, "depth": 1, "Nested": {"width": 2}}
    assert written["Other"] == {"kept": 2, "depth": 1, "Nested": {"width": 2}}


_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_WORKFLOWS = {
    "Trainer": ("konfai.trainer", "Trainer", "TRAIN", {"KONFAI_CHECKPOINTS_DIRECTORY", "KONFAI_STATISTICS_DIRECTORY"}),
    "Predictor": ("konfai.predictor", "Predictor", "PREDICTION", {"KONFAI_PREDICTIONS_DIRECTORY"}),
    "Evaluator": ("konfai.evaluator", "Evaluator", "EVALUATION", {"KONFAI_EVALUATIONS_DIRECTORY"}),
    "Transformer": ("konfai.transformer", "Transformer", "TRANSFORM", {"KONFAI_TRANSFORMS_DIRECTORY"}),
}


def _shipped_workflow_configs() -> list[str]:
    """Every shipped config whose first key names a workflow.

    This feeds a parametrize, so it runs at collection: an example that is empty (no mapping at
    all) or opens on a sequence is not one of these files, and must not take the module's
    collection down with it."""
    relatives = []
    for path in _EXAMPLES.glob("*/*.yml"):
        tree = ruamel.yaml.YAML().load(path.read_text(encoding="utf-8"))
        if isinstance(tree, Mapping) and next(iter(tree), None) in _WORKFLOWS:
            relatives.append(str(path.relative_to(_EXAMPLES)))
    return sorted(relatives)


def _bind_shipped_config(relative: str, workdir: Path, strict: bool, monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Bind a shipped example config on a copy under WORKDIR, over a synthetic cohort holding every
    group the config names, the way its workflow builder does (STRICT) or one context at a time."""
    import importlib

    import numpy as np
    from konfai.utils.dataset import Attribute, Dataset
    from konfai.utils.runtime import configure_workflow_environment
    from konfai.utils.utils import split_path_spec

    example = _EXAMPLES / Path(relative).parent
    workdir.mkdir(parents=True)
    for entry in example.iterdir():
        if entry.suffix in (".yml", ".py"):
            (workdir / entry.name).write_bytes(entry.read_bytes())
    config_path = workdir / Path(relative).name
    tree = ruamel.yaml.YAML().load(config_path.read_text(encoding="utf-8"))
    root = next(iter(tree))
    attribute = Attribute()
    attribute["Origin"], attribute["Spacing"], attribute["Direction"] = np.zeros(3), np.ones(3), np.eye(3).flatten()
    monkeypatch.chdir(workdir)
    for spec in tree[root]["Dataset"]["dataset_filenames"]:
        filename, _flag, file_format = split_path_spec(spec, default_format="mha", allowed_flags={"a", "i"})
        store = Dataset(filename, file_format)
        for group in tree[root]["Dataset"]["groups_src"]:
            for index in range(4):
                store.write(group, f"CASE_{index:03d}", np.zeros((1, 2, 32, 32), np.float32), attribute)
    module_name, class_name, state, path_env = _WORKFLOWS[root]
    configure_workflow_environment(
        config_path=config_path, root=root, state=state, path_env={key: workdir / key for key in path_env}
    )
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.syspath_prepend(str(workdir))
    for local in ("Model", "UnNormalize"):  # the examples' own modules, one per example
        monkeypatch.delitem(sys.modules, local, raising=False)
    workflow = getattr(importlib.import_module(module_name), class_name)
    if strict:
        with strict_config(root, refuse=False):
            apply_config()(workflow)()
    else:
        apply_config()(workflow)()
    return config_path.read_bytes()


@pytest.mark.slow
@pytest.mark.parametrize("relative", _shipped_workflow_configs())
def test_a_shipped_config_resolves_to_the_same_bytes_under_the_block_as_per_context(
    relative: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shipped workflow config, built as its workflow builds it (one strict block) and one
    context at a time: the resolved file is byte-identical. Two of them gain keys the write-back
    appends (Config_GAN.yml, Transform.yml), which is where the order of the appends shows."""
    pytest.importorskip("SimpleITK")
    if Path(relative).parent.name == "Synthesis":
        # Its Model.py imports segmentation_models_pytorch, an extra the example declares and the
        # suite does not: binding the config imports the model.
        pytest.importorskip("segmentation_models_pytorch")
    per_context = _bind_shipped_config(relative, tmp_path / "per_context", False, monkeypatch)
    strict_block = _bind_shipped_config(relative, tmp_path / "strict_block", True, monkeypatch)
    assert strict_block == per_context
