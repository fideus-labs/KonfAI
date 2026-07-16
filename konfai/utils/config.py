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
import inspect
import logging
import os
import types
import typing
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import ruamel.yaml
import torch

from konfai import config_file
from konfai.utils.errors import ConfigError

yaml = ruamel.yaml.YAML()
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Range:
    """UI hint attached to a parameter's type — its inclusive numeric bounds.

    Use ``Annotated[int, Range(0, 100)]`` (or ``float``) in a config-bound signature: the binder ignores the
    metadata and validates the base type, while a UI reads the bounds to size a spinbox. Introspection-only.
    """

    min: float
    max: float


class Choices:
    """UI hint attached to a parameter's type — its allowed values.

    Use ``Annotated[str, Choices([...])]`` for a fixed list, or ``Annotated[str, Choices(resolver)]`` where
    ``resolver`` is a zero-arg callable the app owns (e.g. one that lists a model registry it already
    fetches). ``resolve()`` returns the list — a reader calls it lazily, so the app resolves its own values
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


class Config:
    """
    Context manager for reading and updating a subtree of the active YAML
    config.

    Parameters
    ----------
    key : str
        Dot-separated path pointing to the configuration subtree to inspect or
        materialize.
    """

    def __init__(self, key: str) -> None:
        self.filename = Path(os.environ["KONFAI_config_file"])
        self.keys = [_unescape_key_component(part) for part in key.split(".")]

    def __enter__(self):
        if not self.filename.exists():
            mode = os.environ.get("KONFAI_CONFIG_MODE", "Done")
            if mode in {"default", "interactive", "Import"}:
                self.filename.parent.mkdir(parents=True, exist_ok=True)
                self.filename.touch()
            else:
                raise ConfigError(
                    f"Config file '{self.filename.resolve()}' does not exist.",
                    f"Active config mode: KONFAI_CONFIG_MODE={mode}.",
                    "Run `konfai TRAINING -c Config.yml` to generate a default config, "
                    "or set KONFAI_CONFIG_MODE=default.",
                )

        self.yml = open(self.filename, encoding="utf-8")
        try:
            self.data = yaml.load(self.yml)
        except ruamel.yaml.YAMLError as exc:
            self.yml.close()
            location = ""
            if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
                location = f" at line {exc.problem_mark.line + 1}"
            raise ConfigError(
                f"Invalid YAML syntax in '{self.filename}'{location}.",
                str(exc),
            ) from exc
        if self.data is None:
            self.data = {}

        self.config = self.data

        for key in self.keys:
            if self.config is None or key not in self.config:
                self.config = {key: {}}

            self.config = self.config[key]
        return self

    def create_dictionary(self, data, keys, i) -> dict:
        if keys[i] not in data:
            data = {keys[i]: data}
        if i == 0:
            return data
        else:
            i -= 1
            return self.create_dictionary(data, keys, i)

    def merge(self, dict1, dict2) -> dict:
        result = deepcopy(dict1)

        for key, value in dict2.items():
            if isinstance(value, collections.abc.Mapping):
                result[key] = self.merge(result.get(key, {}), value)
            else:
                if dict2[key] is not None:
                    result[key] = deepcopy(dict2[key])
        return result

    def __exit__(self, exc_type, value, traceback) -> None:
        self.yml.close()
        if os.environ["KONFAI_CONFIG_MODE"] == "remove":
            if os.path.exists(config_file()):
                os.remove(config_file())
            return
        with open(self.filename) as yml:
            data = yaml.load(yml)
            if data is None:
                data = {}
        # Only the currently visited subtree is rewritten; the recursive merge preserves the rest of the
        # YAML file untouched. Write to a sibling temp file then os.replace (atomic on the same
        # filesystem) so a concurrent independent launch reading this file never observes a truncated or
        # empty config and silently binds all-defaults.
        merged = self.merge(
            data,
            self.create_dictionary(self.config, self.keys, len(self.keys) - 1),
        )
        target = Path(self.filename)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w") as yml:
                yaml.dump(merged, yml)
            try:
                os.replace(tmp, target)
            except OSError:
                # Windows can deny the atomic replace when the target is briefly held (a virus
                # scanner or indexer touching the fresh temp file). Fall back to the in-place rewrite
                # the pre-1.6 code always did: POSIX (the DDP path) keeps the atomic guarantee, and
                # Windows keeps its original -- non-atomic -- behaviour instead of failing outright.
                with open(target, "w") as yml:
                    yaml.dump(merged, yml)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _get_input(name: str, default: str) -> str:
        try:
            options = ",".join(default.split(":")[1:]) if ":" in default else ""
            return input(f"{name} [{options}]: ")
        except (EOFError, KeyboardInterrupt):
            # Interactive editing is optional; when stdin is unavailable we
            # degrade to default materialization instead of aborting the run.
            os.environ["KONFAI_CONFIG_MODE"] = "default"
        return default.split("|")[1] if len(default.split("|")) > 1 else default

    @staticmethod
    def _get_input_default(
        name: str,
        default: str | None,
        is_list: bool = False,
    ) -> list[str | None] | str | None:
        # ``default|value`` is KonfAI's marker for "materialize this default if
        # the user/config did not provide a concrete value".
        if isinstance(default, str) and (
            default == "default" or (len(default.split("|")) > 1 and default.split("|")[0] == "default")
        ):
            if os.environ["KONFAI_CONFIG_MODE"] == "interactive":
                if is_list:
                    list_tmp: list[str | None] = []
                    key_tmp = "OK"
                    while key_tmp != "!" and key_tmp != " " and os.environ["KONFAI_CONFIG_MODE"] == "interactive":
                        key_tmp = Config._get_input(name, default)
                        if key_tmp != "!" and key_tmp != " ":
                            if key_tmp == "":
                                key_tmp = default.split("|")[1] if len(default.split("|")) > 1 else default
                            list_tmp.append(key_tmp)
                    return list_tmp
                else:
                    value = Config._get_input(name, default)
                    if value == "":
                        return default.split("|")[1] if len(default.split("|")) > 1 else default
                    else:
                        return value
            else:
                default = default.split("|")[1] if len(default.split("|")) > 1 else default
        return [default] if is_list else default

    def get_value(self, name, default) -> object:
        if not isinstance(self.config, collections.abc.MutableMapping):
            return None

        if name in self.config and self.config[name] is not None:
            value = self.config[name]
            value_config = value
        else:
            value = Config._get_input_default(
                name,
                default if default != inspect._empty else None,
            )

            value_config = value
            if isinstance(value_config, tuple):
                value_config = list(value)

            if isinstance(value_config, list):
                list_tmp = []
                for key in value_config:
                    res = Config._get_input_default(name, key, is_list=True)
                    if isinstance(res, list):
                        list_tmp.extend(res)
                    else:
                        list_tmp.append(str(res))

                value = list_tmp
                value_config = list_tmp

            if isinstance(value, dict):
                key_tmp = []

                value_config = {}
                dict_value = {}
                for key in value:
                    res = Config._get_input_default(name, key, is_list=True)
                    if isinstance(res, list):
                        key_tmp.extend(res)
                    else:
                        key_tmp.append(str(res))
                for key in key_tmp:
                    if key in value:
                        value_tmp = value[key]
                    else:
                        value_tmp = next(v for k, v in value.items() if "default" in k)

                    # dict[str, Object] entries are materialised by a later nested Config context,
                    # so a None placeholder is correct; primitive entries have no such pass, so they
                    # must be persisted here or the write-back collapses the whole dict to ``{}``
                    # (empty on the next run, silently dropping the defaults).
                    value_config[key] = value_tmp if isinstance(value_tmp, int | float | str | bool) else None
                    dict_value[key] = value_tmp
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
    torch.Tensor,
}
_CONFIG_SUPPORTED_TYPES_MESSAGE = (
    "Config: The config only supports types : config(Object), int, str, "
    "bool, float, list[int], list[str], list[bool], list[float], "
    "dict[str, Object]"
)


def _recordable(value):
    """Normalize a default to the form the config file stores and the callable accepts back.

    An ``Enum`` is recorded as its ``.value``, any other ``type`` as its ``.__name__`` -- the forms
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
                "torch": torch,
                "tuple": tuple,
                "typing": typing,
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
            current_value = (
                torch.tensor(value) if candidate_type == torch.Tensor and not isinstance(value, torch.Tensor) else value
            )
            converted = current_value if candidate_type == torch.Tensor else candidate_type(current_value)
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
                            return None

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

                            annotation = _resolve_annotation(function, param.annotation)
                            if hasattr(annotation, "__metadata__"):  # Annotated[T, meta]: bind on T, meta is a UI hint
                                annotation = get_args(annotation)[0]
                            if get_origin(annotation) is Literal:
                                allowed_values = get_args(annotation)
                                default_value = param.default if param.default != inspect._empty else allowed_values[0]
                                value = config.get_value(
                                    param.name,
                                    f"default|{default_value}",
                                )
                                # get_value can hand back the raw "default|X" marker or the stringified
                                # "X"; recover the correctly-typed Literal member so NON-string Literals
                                # (Literal[1, 2], Literal[True, False]) bind and round-trip through the
                                # resolved-config write-back instead of failing the membership check.
                                if isinstance(value, str) and value.startswith("default|"):
                                    value = value.split("|", 1)[1]
                                if value not in allowed_values:
                                    matched = [allowed for allowed in allowed_values if str(allowed) == str(value)]
                                    if matched:
                                        value = matched[0]
                                if value not in allowed_values:
                                    raise ConfigError(
                                        f"Invalid value '{value}' for "
                                        f"parameter '{param.name}' expected "
                                        f"one of: {allowed_values}."
                                    )
                                kwargs[param.name] = value
                                continue
                            annotation, is_optional = _unwrap_optional(annotation)

                            if annotation == inspect._empty:
                                if param.name != "self":
                                    kwargs[param.name] = config.get_value(
                                        param.name,
                                        param.default,
                                    )
                                continue

                            if get_origin(annotation) in {Union, types.UnionType}:
                                value = config.get_value(param.name, param.default)
                                if value is None:
                                    kwargs[param.name] = None
                                else:
                                    kwargs[param.name] = _convert_union_sequence_value(
                                        value, get_args(annotation), param.name
                                    )
                                continue

                            if annotation in _CONFIG_PRIMITIVE_TYPES or annotation is Any:
                                value = config.get_value(param.name, param.default)
                                if annotation in {int, float, bool, str} and value is not None:
                                    try:
                                        if annotation is bool:
                                            if isinstance(value, bool):
                                                pass
                                            elif isinstance(value, int) and value in {0, 1}:
                                                value = bool(value)
                                            elif isinstance(value, str):
                                                normalized = value.strip().lower()
                                                if normalized in {"true", "1", "yes", "on"}:
                                                    value = True
                                                elif normalized in {"false", "0", "no", "off"}:
                                                    value = False
                                                else:
                                                    raise ValueError("unsupported boolean literal")
                                            else:
                                                raise TypeError("unsupported boolean value")
                                        else:
                                            value = annotation(value)
                                    except (ValueError, TypeError) as exc:
                                        raise ConfigError(
                                            f"Invalid value '{value}' for field '{param.name}' "
                                            f"(expected {annotation.__name__}, got {type(value).__name__}) "
                                            f"in config section '{key_tmp}'."
                                        ) from exc
                                kwargs[param.name] = value
                                continue

                            if annotation is Path:
                                raw = config.get_value(param.name, param.default)
                                if raw is not None:
                                    path = Path(str(raw))
                                    if not path.exists():
                                        _log.warning(
                                            "[Config] Path '%s' for field '%s' does not exist (resolved: '%s'; %s).",
                                            raw,
                                            param.name,
                                            path.resolve(),
                                            "absolute" if path.is_absolute() else "relative path",
                                        )
                                    kwargs[param.name] = path
                                else:
                                    kwargs[param.name] = None
                                continue

                            origin = get_origin(annotation)
                            if origin in {list, tuple, Sequence, collections.abc.Sequence}:
                                values = config.get_value(
                                    param.name,
                                    param.default,
                                )
                                if values is None:
                                    kwargs[param.name] = None
                                    continue

                                args_annotation = get_args(annotation)
                                elem_type = args_annotation[0] if args_annotation else Any
                                elem_origin = get_origin(elem_type)
                                if elem_origin in {Union, types.UnionType}:
                                    valid_types = get_args(elem_type)
                                    kwargs[param.name] = [
                                        _convert_union_sequence_value(value, valid_types, param.name)
                                        for value in values
                                    ]
                                elif elem_type in {int, str, bool, float, torch.Tensor, Any}:
                                    kwargs[param.name] = values
                                else:
                                    raise ConfigError(_CONFIG_SUPPORTED_TYPES_MESSAGE)
                                continue

                            if origin is dict:
                                key_type, value_type = get_args(annotation)
                                if key_type is not str:
                                    raise ConfigError(_CONFIG_SUPPORTED_TYPES_MESSAGE)

                                values = config.get_value(
                                    param.name,
                                    param.default,
                                )
                                if values is None or value_type in {
                                    int,
                                    str,
                                    bool,
                                    float,
                                    Any,
                                }:
                                    kwargs[param.name] = values
                                    continue

                                try:
                                    kwargs[param.name] = {
                                        value: apply_config(f"{key_tmp}.{param.name}.{_escape_key_component(value)}")(
                                            value_type
                                        )()
                                        for value in values
                                    }
                                except Exception as exc:
                                    raise ConfigError(f"{values} {exc}") from exc
                                continue

                            # ``X | None = None`` declares an object the config must ASK for: binding it
                            # anyway would build X's defaults and write them back, turning "no patch" into
                            # a patch nobody configured. A non-None default (``X | None = X()``) is the
                            # opposite declaration and still binds.
                            if is_optional and param.default is None:
                                annotation_key = getattr(annotation, "_key", None)
                                if annotation_key is None or config.get_value(annotation_key, None) is None:
                                    kwargs[param.name] = None
                                    continue
                            try:
                                kwargs[param.name] = apply_config(key_tmp)(annotation)()
                            except Exception as exc:
                                raise ConfigError(
                                    f"Failed to instantiate {param.name} with type {annotation}, error {exc}"
                                ) from exc
                        return function(*args, **kwargs)
                finally:
                    if previous_path is None:
                        os.environ.pop("KONFAI_CONFIG_PATH", None)
                    else:
                        os.environ["KONFAI_CONFIG_PATH"] = previous_path
            return function(*args, **kwargs)

        return new_function

    return decorator
