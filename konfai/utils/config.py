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

"""Configuration helpers that map YAML trees to KonfAI Python objects."""

import collections
import difflib
import functools
import inspect
import logging
import os
import sys
import time
import types
import typing
import warnings
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import ruamel.yaml

from konfai.utils.errors import ConfigError

yaml = ruamel.yaml.YAML()
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Range:
    """UI hint attached to a parameter's type: its inclusive numeric bounds.

    Use ``Annotated[int, Range(0, 100)]`` (or ``float``) in a config-bound signature: the binder ignores the
    metadata and validates the base type, while a UI reads the bounds to size a spinbox. Introspection-only.
    """

    min: float
    max: float


class Choices:
    """UI hint attached to a parameter's type: its allowed values.

    Use ``Annotated[str, Choices([...])]`` for a fixed list, or ``Annotated[str, Choices(resolver)]`` where
    ``resolver`` is a zero-arg callable the app owns (e.g. one that lists a model registry it already
    fetches). ``resolve()`` returns the list: a reader calls it lazily, so the app resolves its own values
    and no tool re-fetches. Introspection-only; the binder ignores it (a value outside the list is still
    accepted, e.g. a local path). For a small FIXED, binder-validated set, prefer ``Literal[...]``.
    """

    def __init__(self, values) -> None:
        self.values = values

    def resolve(self) -> list:
        return list(self.values() if callable(self.values) else self.values)


def _escape_key_component(component: str) -> str:
    """Percent-encode ``.`` (and ``%``) so a dict key survives dotted-path splitting."""
    return component.replace("%", "%25").replace(".", "%2E")


def _unescape_key_component(component: str) -> str:
    """Inverse of :func:`_escape_key_component`."""
    return component.replace("%2E", ".").replace("%25", "%")


def _load_tree(filename: Path | str) -> dict:
    """The file's YAML tree (an empty file is an empty tree); a syntax error is a ConfigError with its line."""
    with open(filename, encoding="utf-8") as stream:
        try:
            tree = yaml.load(stream)
        except ruamel.yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}" if mark is not None else ""
            raise ConfigError(f"Invalid YAML syntax in '{filename}'{location}.", str(exc)) from exc
    return {} if tree is None else tree


def _write_tree(target: Path, tree: dict) -> None:
    """Write TREE to TARGET atomically: a sibling temp file, then ``os.replace``, so a concurrent
    independent launch reading the file never observes it truncated and binds all-defaults."""
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as yml:
            yaml.dump(tree, yml)
        # Windows can deny the replace while the target is briefly held (a virus scanner or an
        # indexer touching the fresh file): retried a few times, then refused. Never an in-place
        # rewrite, which a concurrent reader would see truncated and bind all-defaults from.
        for attempt in range(5):
            try:
                os.replace(tmp, target)
                break
            except OSError as error:
                if attempt == 4:
                    raise ConfigError(
                        f"Could not replace the config file '{target}' atomically: {error}.",
                        "Release whatever holds the file (an editor, an indexer) and rerun; the file was left unchanged.",
                    ) from error
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp.exists():
            tmp.unlink()


def _merge_into(target: MutableMapping, source: Mapping) -> None:
    """Fold SOURCE into TARGET in place: a mapping recurses, a value replaces, a ``None`` is not
    written (a nested object's placeholder, materialized by its own context). A key TARGET lacks is
    appended, so what a context sets lands after what the contexts opened inside it appended."""
    for key, value in source.items():
        existing = target.get(key)
        if existing is value or value is None:
            continue
        if isinstance(value, Mapping):
            if not isinstance(existing, MutableMapping):
                existing = target[key] = {}
            _merge_into(existing, value)
        else:
            target[key] = deepcopy(value)


class _SharedTree:
    """The config tree a ``strict_config`` block holds in memory.

    Every ``Config`` context opened inside the block on this file reads and writes this one tree,
    and the block writes the file once, when it ends. Outside a block a context loads and writes
    the file itself.
    """

    def __init__(self, filename: Path, tree: dict) -> None:
        self.filename = filename
        self.tree = tree

    def flush(self) -> None:
        # A file no context created (refused as missing) is not created here either.
        if self.filename.exists():
            _write_tree(self.filename, self.tree)


# The trees of the open strict_config() blocks, innermost last.
_shared_trees: list[_SharedTree] = []


def _shared_tree(filename: Path) -> _SharedTree | None:
    for shared in reversed(_shared_trees):
        if shared.filename == filename:
            return shared
    return None


class _KeyLedger:
    """What the file holds against what the binder read, per level (dotted path) of the file."""

    def __init__(self) -> None:
        self.present: dict[tuple[str, ...], set[str]] = collections.defaultdict(set)
        self.consumed: dict[tuple[str, ...], set[str]] = collections.defaultdict(set)

    def opened(self, level: tuple[str, ...], keys: Iterable[object]) -> None:
        """A context opened at LEVEL over a mapping holding KEYS: they are present there, and the
        subtree's own key is read at the level above."""
        self.present[level].update(str(key) for key in keys)
        if len(level) > 1:
            self.consumed[level[:-1]].add(level[-1])

    def read(self, level: tuple[str, ...], *keys: str) -> None:
        self.consumed[level].update(keys)

    def unknown(self, root: str) -> list[str]:
        """One line per key under ROOT nothing read: its path, the keys read at that level, the closest."""
        lines = []
        for level in sorted(self.present):
            if level[0] != root:
                continue
            read = sorted(self.consumed[level])
            for key in sorted(self.present[level] - self.consumed[level]):
                close = difflib.get_close_matches(key, read, n=1)
                hint = f" Did you mean '{close[0]}'?" if close else ""
                lines.append(f"'{'.'.join(level)}.{key}' (keys read at that level: {read}).{hint}")
        return lines


# The ledgers of the open strict_config() blocks; empty outside them, where the binder records nothing.
_ledgers: list[_KeyLedger] = []


@contextmanager
def strict_config(root: str, refuse: bool = True) -> Iterator[None]:
    """Report, when the block ends, every key of the config file that nothing bound inside it read.

    The binder reads a key when a parameter names it and materializes the default when none does,
    so a typo'd key is carried along and its default used in its place. Inside this block every
    ``Config`` context records the keys its level held and the keys read there; on a clean exit
    the difference is reported by path, with the keys read at that level and the closest of them:
    a :class:`ConfigError` when ``refuse``, a warning otherwise. The block must contain everything
    the workflow binds from the file: a key a later, lazily bound callable would read is unknown to
    it. On entry the file must hold ROOT: a missing root binds an all-defaults workflow and writes
    the whole block back over the user's file.

    The file is read once here and written once when the block ends, whatever the number of
    contexts opened inside it, and whatever the number of blocks opened over the same file: they
    all resolve against the one tree held in memory, read and written by the outermost. A build of N
    nested objects otherwise parses the file 2N+1 times and writes it N times (Config_GAN.yml, 7 KB,
    76 objects: 153 parses and 76 dumps, 2.96 s of a 4.7 s build; 0.05 s held in memory).
    """
    filename = os.environ.get("KONFAI_config_file")
    tree: dict = {}
    # A block opened inside another one on the same file binds to the tree already held: reloading
    # it would hide what the outer contexts bound, and flushing it would be undone by the outer
    # flush of a tree that predates this block.
    outer = _shared_tree(Path(filename)) if filename else None
    if filename and Path(filename).exists():
        tree = outer.tree if outer is not None else _load_tree(filename)
        if root not in tree:
            _report(
                refuse,
                f"'{filename}' declares no '{root}' root (found: {sorted(str(key) for key in tree)}).",
                f"This workflow reads the '{root}:' block; anything else is ignored and a full default"
                " block appended to the file in its place.",
            )
    ledger = _KeyLedger()
    shared = _SharedTree(Path(filename), tree) if filename and outer is None else None
    _ledgers.append(ledger)
    if shared is not None:
        _shared_trees.append(shared)
    unknown: list[str] = []
    try:
        yield
    finally:
        _ledgers.remove(ledger)
        unknown = ledger.unknown(root)
        if shared is not None:
            _shared_trees.remove(shared)
            # Written whatever ended the block, EXCEPT when the block is about to refuse the config:
            # a refused run must leave the user's file exactly as it was.
            if not (refuse and unknown):
                shared.flush()
    if unknown:
        _report(
            refuse,
            f"Unknown key(s) in the {root} configuration: nothing reads them.",
            *unknown,
            "A key nothing reads is carried along and the default used in its place; fix the spelling or remove it.",
        )


def _report(refuse: bool, *messages: str) -> None:
    if refuse:
        raise ConfigError(*messages)
    warnings.warn(str(ConfigError(*messages)), UserWarning, stacklevel=4)


class Config:
    """
    Context manager for reading and updating a subtree of the active YAML
    config.

    Inside a :func:`strict_config` block the context reads the block's in-memory tree and folds
    what it set back into it on exit; outside one it loads and writes the file itself.

    Parameters
    ----------
    key : str
        Dot-separated path pointing to the configuration subtree to inspect or
        materialize.
    """

    def __init__(self, key: str) -> None:
        self.filename = Path(os.environ["KONFAI_config_file"])
        self.keys = [_unescape_key_component(part) for part in key.split(".")]
        self._shared: _SharedTree | None = None

    def __enter__(self):
        if not self.filename.exists():
            if os.environ.get("KONFAI_CONFIG_MODE") == "Import":
                self.filename.parent.mkdir(parents=True, exist_ok=True)
                self.filename.touch()
            else:
                raise ConfigError(
                    f"Config file '{self.filename.resolve()}' does not exist.",
                    "Generate a resolved default with `konfai <COMMAND> --init` "
                    "(e.g. `konfai TRAIN --init -c Config.yml`), or point -c at an existing file.",
                )

        self._shared = _shared_tree(self.filename)
        self.data = self._shared.tree if self._shared is not None else _load_tree(self.filename)
        self.config = self.data

        for index, key in enumerate(self.keys):
            if self.config is not None and not isinstance(self.config, collections.abc.Mapping):
                raise ConfigError(
                    f"'{'.'.join(self.keys[:index])}' holds the value '{self.config}' where a block is expected.",
                    f"Nest '{key}:' under it as a mapping, or remove the value.",
                )
            if self.config is None or key not in self.config:
                self.config = {key: {}}

            self.config = self.config[key]
        if self._shared is not None and isinstance(self.config, collections.abc.MutableMapping):
            # The context works on a copy of its level and folds it back on exit: what it sets is
            # not seen by the contexts opened inside it, and lands after what they appended, as
            # when every context read and wrote the file itself.
            self.config = dict(self.config)
        for ledger in _ledgers:
            ledger.opened(tuple(self.keys), self.config if isinstance(self.config, collections.abc.Mapping) else ())
        return self

    def __exit__(self, exc_type, value, traceback) -> None:
        # Only the visited subtree is folded back; the merge preserves the rest of the tree untouched.
        subtree = self.config
        for key in reversed(self.keys):
            subtree = {key: subtree}
        if self._shared is not None:
            _merge_into(self._shared.tree, subtree)
            return
        data = _load_tree(self.filename)
        _merge_into(data, subtree)
        _write_tree(self.filename, data)

    @staticmethod
    def _default_value(default):
        """Resolve the ``default|value`` marker ("materialize this when the config holds nothing")."""
        if isinstance(default, str) and default.split("|")[0] == "default":
            parts = default.split("|")
            return parts[1] if len(parts) > 1 else default
        return default

    def get_value(self, name, default) -> object:
        if not isinstance(self.config, collections.abc.MutableMapping):
            return None
        for ledger in _ledgers:
            ledger.read(tuple(self.keys), name)

        if name in self.config:
            # An explicit null (`name:` empty or `name: null`) is the disabled spelling, exactly like
            # the string "None": substituting the default here would silently reactivate the very
            # thing the line was written to suppress.
            value = self.config[name] if self.config[name] is not None else "None"
            value_config = value
        else:
            value = Config._default_value(default if default != inspect._empty else None)

            value_config = value
            if isinstance(value_config, tuple):
                value_config = list(value)

            if isinstance(value_config, list):
                value = value_config = [Config._default_value(key) for key in value_config]

            if isinstance(value, dict):
                value_config = {}
                dict_value = {}
                for key in value:
                    resolved = str(Config._default_value(key))
                    if resolved in value:
                        value_tmp = value[resolved]
                    else:
                        value_tmp = next(v for k, v in value.items() if "default" in k)

                    # dict[str, Object] entries are materialised by a later nested Config context,
                    # so a None placeholder is correct; primitive entries have no such pass, so they
                    # must be persisted here or the write-back collapses the whole dict to ``{}``
                    # (empty on the next run, silently dropping the defaults).
                    value_config[resolved] = value_tmp if isinstance(value_tmp, int | float | str | bool) else None
                    dict_value[resolved] = value_tmp
                value = dict_value
        self.config[name] = _recordable(value_config) if value_config is not None else "None"
        if value == "None":
            value = None
        return value


def config(key: str | None = None):
    """
    Attach a KonfAI configuration key to a class or callable.

    Parameters
    ----------
    key : str | None, optional
        Configuration branch handled by the decorated object.

    Returns
    -------
    Callable
        Decorator storing the key on the decorated object.
    """

    def decorator(function):
        function._key = key if key is not None else function.__name__
        return function

    return decorator


_CONFIG_PRIMITIVE_TYPES = {
    int,
    str,
    bool,
    float,
}


def _tensor_type() -> type | None:
    """``torch.Tensor`` when torch is already imported, else None.

    torch is never imported here: an annotation can only mention Tensor if its declaring module
    already paid the import, and keeping config.py torch-free keeps the light-import contract of
    ``konfai/__init__`` honest for consumers like the Slicer-facing konfai-apps helpers (measured:
    634 of this module's 676 ms import was torch).
    """
    return getattr(sys.modules.get("torch"), "Tensor", None)


_CONFIG_SUPPORTED_TYPES_MESSAGE = (
    "Config: The config only supports types : config(Object), int, str, "
    "bool, float, list[int], list[str], list[bool], list[float], "
    "dict[str, Object]"
)


def _recordable(value):
    """Normalize a default to the form the config file stores and the callable accepts back.

    An ``Enum`` is recorded as its ``.value``, any other ``type`` as its ``.__name__``: the forms
    the declaring parameter accepts (``LossReduction | str``, ``numpy.dtype | type | str``).
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, type):
        return value.__name__
    return value


def _annotation_namespace(function) -> dict[str, Any]:
    """The globals an annotation's names resolve against.

    Under ``from __future__ import annotations`` an annotation is source text resolved against its
    defining module's ``__globals__``. A class has none of its own, so fall back to its ``__init__``'s.
    """
    namespace = getattr(function, "__globals__", None)
    if namespace is None:
        namespace = getattr(getattr(function, "__init__", None), "__globals__", None)
    return dict(namespace) if namespace else {}


def _resolve_annotation(function, annotation):
    if annotation in {"int", "float", "bool", "str"}:
        return {"int": int, "float": float, "bool": bool, "str": str}[annotation]

    if not isinstance(annotation, str):
        return annotation

    try:
        return eval(  # nosec B307
            annotation,
            {
                **_annotation_namespace(function),
                "Any": Any,
                "Literal": Literal,
                "Sequence": Sequence,
                "Union": Union,
                "bool": bool,
                "dict": dict,
                "float": float,
                "int": int,
                "list": list,
                "str": str,
                "tuple": tuple,
                "typing": typing,
                **({"torch": sys.modules["torch"]} if "torch" in sys.modules else {}),
            },
        )
    except Exception:
        return annotation


def _unwrap_optional(annotation) -> tuple[Any, bool]:
    """Return ``(bound type, was Optional[X])``.

    The flag is what tells an ``X | None`` parameter from a plain ``X``: both bind on ``X``, but only
    the first may legitimately stay ``None`` (see the nested-object binding in ``apply_config``).
    """
    origin = get_origin(annotation)
    if origin not in {Union, types.UnionType}:
        return annotation, False

    args = [arg for arg in get_args(annotation) if arg not in {type(None), types.NoneType}]
    if len(args) == 1:
        return args[0], True
    # Genuine unions (e.g. ``float | str``) are kept intact so the binding can try each
    # member type; only ``Optional[X]`` (``X | None``) collapses to ``X``.
    return annotation, False


def _convert_union_sequence_value(
    value: object,
    valid_types: tuple[type | object, ...],
    param_name: str,
) -> object:
    # Keep a value whose runtime type already satisfies a union member. Coercing in declaration order is
    # lossy: int(0.25) == 0 silently beats a float member, str([1, 2]) swallows a list member, and a
    # list[...] member is never a `type`, so a list value could otherwise never bind through it.
    if value not in (None, "None"):
        for candidate_type in valid_types:
            origin = get_origin(candidate_type)
            if origin is not None:
                # A typing-only origin (Literal, Annotated, ...) is not a class: isinstance would raise.
                if isinstance(origin, type) and isinstance(value, origin):
                    return value
            elif isinstance(candidate_type, type) and candidate_type not in (type(None), types.NoneType):
                # bool subclasses int: only a bool member accepts a bool, never int/float.
                matched = candidate_type is bool if isinstance(value, bool) else isinstance(value, candidate_type)
                if matched:
                    return value

    if isinstance(value, Mapping):
        # No member of these unions can hold a mapping, and the coercion loop must not see one:
        # `str` is a member of most of them, `str(mapping)` never fails, and a block bound as its
        # own repr fails far from here, on whatever that text then selects. This key is the last
        # place the shape of the YAML is still visible.
        raise ConfigError(
            f"Parameter '{param_name}' was given a nested block, but it takes a value.",
            f"Expected one of: {valid_types}.",
            f"Write it on one line ('{param_name}: <value>'); keys nested under it select nothing.",
        )

    converted = None
    last_error: Exception | None = None
    for candidate_type in valid_types:
        try:
            if candidate_type is Any:
                return value
            if candidate_type in {type(None), types.NoneType}:
                if value in (None, "None"):
                    return None
                continue
            if not isinstance(candidate_type, type):
                continue
            if candidate_type is _tensor_type():
                converted = value if isinstance(value, candidate_type) else sys.modules["torch"].tensor(value)
            else:
                converted = candidate_type(value)
            break
        except Exception as exc:
            last_error = exc

    if converted is None and value not in (None, "None"):
        raise ConfigError(
            f"Invalid value '{value}' for parameter '{param_name}'.",
            f"Expected one of: {valid_types}.",
            f"Last conversion error: {last_error}" if last_error else "",
        )
    return converted


# Sentinel: the parameter must not be bound at all (the callable's own default applies).
_SKIP = object()


def _bind_literal(config: Config, param: inspect.Parameter, annotation, is_optional: bool = False) -> object:
    allowed_values = get_args(annotation)
    default_value = param.default if param.default != inspect._empty else allowed_values[0]
    value = config.get_value(param.name, f"default|{default_value}")
    # get_value can hand back the raw "default|X" marker or the stringified "X"; recover the
    # correctly-typed Literal member so NON-string Literals (Literal[1, 2], Literal[True, False])
    # bind and round-trip through the resolved-config write-back instead of failing the
    # membership check.
    if isinstance(value, str) and value.startswith("default|"):
        value = value.split("|", 1)[1]
    if is_optional and value in (None, "None"):
        return None
    if value not in allowed_values:
        matched = [allowed for allowed in allowed_values if str(allowed) == str(value)]
        if matched:
            value = matched[0]
    if value not in allowed_values:
        raise ConfigError(f"Invalid value '{value}' for parameter '{param.name}' expected one of: {allowed_values}.")
    return value


def _parse_bool(value: object) -> bool:
    """A YAML/CLI boolean: bool as-is, 0/1, or the usual true/false spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError("unsupported boolean literal")
    raise TypeError("unsupported boolean value")


def _bind_primitive(config: Config, param: inspect.Parameter, annotation, section_key: str) -> object:
    value = config.get_value(param.name, param.default)
    if annotation in {int, float, bool, str} and value is not None:
        if isinstance(value, Mapping | list | tuple):
            # `str` never fails to coerce, so without this a nested block or list binds as its
            # Python repr and fails far downstream, on whatever that text then selects.
            shape = "a nested block" if isinstance(value, Mapping) else "a list"
            raise ConfigError(
                f"Parameter '{section_key}.{param.name}' was given {shape}, but it takes a {annotation.__name__}.",
                f"Write it as a single value ('{param.name}: <{annotation.__name__}>').",
            )
        try:
            value = _parse_bool(value) if annotation is bool else annotation(value)
        except (ValueError, TypeError) as exc:
            raise ConfigError(
                f"Invalid value '{value}' for field '{param.name}' "
                f"(expected {annotation.__name__}, got {type(value).__name__}) "
                f"in config section '{section_key}'."
            ) from exc
    return value


def _bind_path(config: Config, param: inspect.Parameter) -> Path | None:
    raw = config.get_value(param.name, param.default)
    if raw is None:
        return None
    path = Path(str(raw))
    if not path.exists():
        _log.warning(
            "[Config] Path '%s' for field '%s' does not exist (resolved: '%s'; %s).",
            raw,
            param.name,
            path.resolve(),
            "absolute" if path.is_absolute() else "relative path",
        )
    return path


def _bind_sequence(config: Config, param: inspect.Parameter, annotation, section_key: str) -> object:
    values: Any = config.get_value(param.name, param.default)
    if values is None:
        return None
    args_annotation = get_args(annotation)
    elem_type = args_annotation[0] if args_annotation else Any
    if get_origin(elem_type) in {Union, types.UnionType}:
        return [_convert_union_sequence_value(value, get_args(elem_type), param.name) for value in values]
    if elem_type is Any or elem_type is _tensor_type():
        return values
    if isinstance(elem_type, type) and elem_type in {int, str, bool, float}:
        if not isinstance(values, list | tuple):
            raise ConfigError(
                f"Parameter '{section_key}.{param.name}' expects a list of {elem_type.__name__}, got '{values}'.",
                f"Spell it as a YAML list ('{param.name}: [a, b]' or one '- item' per line).",
            )
        converted = []
        for index, value in enumerate(values):
            if value is None or isinstance(value, Mapping | list | tuple):
                raise ConfigError(
                    f"Element {index} of '{section_key}.{param.name}' is not a {elem_type.__name__}: '{value}'."
                )
            try:
                converted.append(_parse_bool(value) if elem_type is bool else elem_type(value))
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Element {index} of '{section_key}.{param.name}' is not a {elem_type.__name__}: '{value}'."
                ) from exc
        return converted
    raise ConfigError(
        f"Parameter '{section_key}.{param.name}' is annotated {annotation}, which the config cannot bind.",
        _CONFIG_SUPPORTED_TYPES_MESSAGE,
    )


def _bind_dict(config: Config, param: inspect.Parameter, annotation, section_key: str) -> object:
    key_type, value_type = get_args(annotation)
    if key_type is not str:
        raise ConfigError(
            f"Parameter '{section_key}.{param.name}' is annotated {annotation}, which the config cannot bind"
            " (dict keys must be str).",
            _CONFIG_SUPPORTED_TYPES_MESSAGE,
        )
    values: Any = config.get_value(param.name, param.default)
    if values is None or value_type in {int, str, bool, float, Any}:
        return values
    try:
        return {
            value: apply_config(f"{section_key}.{param.name}.{_escape_key_component(value)}")(value_type)()
            for value in values
        }
    except Exception as exc:
        raise ConfigError(f"Failed to build an entry of '{section_key}.{param.name}': {exc}") from exc


def _bind_config_object(config: Config, param: inspect.Parameter, annotation, is_optional: bool, section_key: str):
    # ``X | None = None`` declares an object the config must ASK for: binding it anyway would build
    # X's defaults and write them back, turning "no patch" into a patch nobody configured. A non-None
    # default (``X | None = X()``) is the opposite declaration and still binds.
    if is_optional and param.default is None:
        annotation_key = getattr(annotation, "_key", None)
        if annotation_key is None or config.get_value(annotation_key, None) is None:
            return None
    try:
        return apply_config(section_key)(annotation)()
    except Exception as exc:
        raise ConfigError(f"Failed to instantiate {param.name} with type {annotation}, error {exc}") from exc


def _bind_parameter(function, config: Config, param: inspect.Parameter, section_key: str) -> object:
    """The config-bound value for one signature parameter, or ``_SKIP`` (dispatch by annotation kind)."""
    annotation = _resolve_annotation(function, param.annotation)
    if hasattr(annotation, "__metadata__"):  # Annotated[T, meta]: bind on T, meta is a UI hint
        annotation = get_args(annotation)[0]
    annotation, is_optional = _unwrap_optional(annotation)
    # After unwrapping, so ``Literal[X] | None`` binds as a literal (or None) instead of falling through
    # to _bind_config_object, which would try to instantiate the Literal as a class.
    if get_origin(annotation) is Literal:
        return _bind_literal(config, param, annotation, is_optional)

    if annotation == inspect._empty:
        return _SKIP if param.name == "self" else config.get_value(param.name, param.default)

    if get_origin(annotation) in {Union, types.UnionType}:
        value = config.get_value(param.name, param.default)
        return None if value is None else _convert_union_sequence_value(value, get_args(annotation), param.name)

    if annotation in _CONFIG_PRIMITIVE_TYPES or annotation is Any:
        return _bind_primitive(config, param, annotation, section_key)

    if annotation is Path:
        return _bind_path(config, param)

    origin = get_origin(annotation)
    if origin in {list, tuple, Sequence, collections.abc.Sequence}:
        return _bind_sequence(config, param, annotation, section_key)
    if origin is dict:
        return _bind_dict(config, param, annotation, section_key)

    return _bind_config_object(config, param, annotation, is_optional, section_key)


def apply_config(konfai_args: str | None = None):
    """
    Recursively instantiate callables from the active KonfAI configuration.

    Parameters
    ----------
    konfai_args : str | None, optional
        Root configuration path used to resolve nested constructor arguments.

    Returns
    -------
    Callable
        Decorator that injects configuration-backed arguments at call time.
    """

    def decorator(function):
        def new_function(*args, **kwargs):
            key = getattr(function, "_key", None)
            key_tmp = konfai_args + ("." + key if key is not None else "") if konfai_args is not None else key
            if (
                "KONFAI_config_file" in os.environ
                and "KONFAI_CONFIG_MODE" in os.environ
                and os.environ["KONFAI_CONFIG_MODE"] != "Import"
                and key_tmp is not None
            ):
                previous_path = os.environ.get("KONFAI_CONFIG_PATH")
                os.environ["KONFAI_CONFIG_PATH"] = key_tmp
                without = kwargs["konfai_without"] if "konfai_without" in kwargs else []
                try:
                    with Config(key_tmp) as config:
                        if not isinstance(config.config, collections.abc.Mapping):
                            if config.config in (None, "None"):
                                return None
                            # `optimizer: AdamW` where a block is expected would otherwise bind the
                            # whole object to None and the run would proceed without it, silently.
                            raise ConfigError(
                                f"'{key_tmp}' holds the value '{config.config}' where a block is expected.",
                                f"Nest its settings under '{key_tmp.rsplit('.', 1)[-1]}:' as a mapping"
                                " (or write 'None' to disable it).",
                            )
                        for ledger in _ledgers:  # a parameter the caller supplies itself is a read one
                            ledger.read(tuple(config.keys), *without)

                        kwargs = {}
                        params = list(inspect.signature(function).parameters.values())
                        for param in params[len(args) :]:
                            if param.name in without:
                                continue

                            # ``*args`` and ``**kwargs`` name no parameter: they stand for the ones a
                            # caller passes. There is nothing to bind them to, and binding them hands
                            # the callable a parameter called "kwargs".
                            if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                                continue

                            value = _bind_parameter(function, config, param, key_tmp)
                            if value is not _SKIP:
                                kwargs[param.name] = value
                        return function(*args, **kwargs)
                finally:
                    if previous_path is None:
                        os.environ.pop("KONFAI_CONFIG_PATH", None)
                    else:
                        os.environ["KONFAI_CONFIG_PATH"] = previous_path
            return function(*args, **kwargs)

        return new_function

    return decorator


def record_given_arguments(cls: type) -> None:
    """Make ``cls`` record, on each instance, the constructor arguments AS GIVEN: the binder's mirror.

    The binder builds an object from a config subtree; this makes the reverse spelling possible: an
    object built in Python remembers what the caller said (``_konfai_given``), so :mod:`konfai.api`
    can write a workflow tree from live objects with no second grammar: the recorded kwargs go
    back through the binder, which stays the one place that validates and resolves defaults.

    Only the OUTERMOST constructor records: a subclass delegating to ``super().__init__`` keeps its
    own spelling, which is what the caller wrote. Applied by the extension bases'
    ``__init_subclass__``, so a subclass that defines no ``__init__`` inherits a recording one. An
    ``__init__`` taking ``*args`` cannot be spelled as a config subtree; such an instance records
    nothing and :mod:`konfai.api` refuses it by name.
    """
    # A subclass with no __init__ of its own wraps the inherited one here: the extension bases are
    # never passed through this function, so their raw constructors record nothing by themselves.
    original = cls.__dict__.get("__init__") or cls.__init__  # type: ignore[misc]
    if getattr(original, "_konfai_records", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def recording(self: object, *args: object, **kwargs: object) -> None:
        if not hasattr(self, "_konfai_given"):
            arguments: dict[str, object] | None = {}
            try:
                bound = signature.bind(self, *args, **kwargs)
            except TypeError:
                arguments = None  # the original call raises its own, better error just below
            if arguments is not None:
                for name, value in list(bound.arguments.items())[1:]:
                    kind = signature.parameters[name].kind
                    if kind is inspect.Parameter.VAR_POSITIONAL:
                        arguments = None
                        break
                    if kind is inspect.Parameter.VAR_KEYWORD:
                        arguments.update(dict(value))  # type: ignore[call-overload]
                    else:
                        arguments[name] = value
            # None is recorded too: it marks the instance as spoken for, so a delegating
            # super().__init__ cannot record the INNER spelling under the outer class's name --
            # kwargs the outer constructor does not accept.
            self._konfai_given = arguments  # type: ignore[attr-defined]
        original(self, *args, **kwargs)

    recording._konfai_records = True  # type: ignore[attr-defined]
    cls.__init__ = recording  # type: ignore[method-assign, misc]
