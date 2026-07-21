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

"""Per-operation contract for the tunables a remote KonfAI app run must honour.

The client serialises these fields into a single multipart ``options`` JSON object; the server
validates them against the same spec and turns them back into ``konfai-apps`` CLI flags. Sharing
one spec on both sides means a tunable can be rejected loudly, but never silently dropped.
"""

import json
from collections.abc import Callable
from typing import Any

# Operation -> tunables the matching local ``KonfAIApp.<op>`` honours but the plain form fields of
# the corresponding server endpoint do not carry. Every listed field must have an entry in
# ``_OPTION_FIELDS`` and be declared on both ``KonfAIApp.<op>`` and ``KonfAIAppClient.<op>``.
REMOTE_OPTION_FIELDS: dict[str, tuple[str, ...]] = {
    "infer": ("patch_size", "batch_size", "config_overrides"),
    "evaluate": (),
    "uncertainty": (),
    "pipeline": ("patch_size", "batch_size", "config_overrides"),
    "fine_tune": ("config_overrides",),
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_override(value: Any) -> bool:
    # 'NAME=VALUE' only: a leading '-' would be parsed as a flag by the job subprocess's argparse, and an
    # empty NAME ('=value') is rejected here (422) instead of failing after the job is dispatched.
    return (
        isinstance(value, str) and "=" in value and not value.startswith("-") and value.split("=", 1)[0].strip() != ""
    )


# Field -> (validator, expected shape for the 422 detail, JSON value -> CLI args).
_OPTION_FIELDS: dict[str, tuple[Callable[[Any], bool], str, Callable[[Any], list[str]]]] = {
    "patch_size": (
        lambda v: isinstance(v, list) and len(v) > 0 and all(_is_int(x) for x in v),
        "a non-empty list of int",
        lambda v: ["--patch_size", *(str(x) for x in v)],
    ),
    "batch_size": (
        _is_int,
        "an int",
        lambda v: ["--batch_size", str(v)],
    ),
    "config_overrides": (
        lambda v: isinstance(v, list) and len(v) > 0 and all(_is_override(x) for x in v),
        "a non-empty list of 'NAME=VALUE' strings",
        lambda v: [arg for override in v for arg in ("--set", override)],
    ),
}


def collect_remote_options(operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Pop ``operation``'s tunables out of ``arguments``, keeping only user-set values.

    ``None`` (the argparse default) and an empty list both mean "not set".
    """
    options: dict[str, Any] = {}
    for name in REMOTE_OPTION_FIELDS.get(operation, ()):
        value = arguments.pop(name, None)
        if value is not None and value != []:
            options[name] = value
    return options


def parse_remote_options(operation: str, options_json: str) -> dict[str, Any]:
    """Parse a submitted ``options`` JSON object and validate it against ``operation``'s spec.

    Raises ``ValueError`` naming the offending key, so an unsupported or malformed tunable is
    refused instead of being silently ignored.
    """
    try:
        options = json.loads(options_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'options' is not valid JSON: {exc}") from exc
    if not isinstance(options, dict):
        raise ValueError("'options' must be a JSON object")
    unknown = sorted(set(options) - set(REMOTE_OPTION_FIELDS.get(operation, ())))
    if unknown:
        raise ValueError(f"Unknown option(s) for '{operation}': {', '.join(unknown)}")
    for name, value in options.items():
        validate, expected, _ = _OPTION_FIELDS[name]
        if not validate(value):
            raise ValueError(f"Option '{name}' must be {expected}")
    return options


def remote_options_to_cli_args(operation: str, options: dict[str, Any]) -> list[str]:
    """Serialise validated options into ``konfai-apps`` CLI flags, in spec order."""
    args: list[str] = []
    for name in REMOTE_OPTION_FIELDS.get(operation, ()):
        if name in options:
            args += _OPTION_FIELDS[name][2](options[name])
    return args
