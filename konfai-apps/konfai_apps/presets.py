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
"""Presets: named lists of ``--set NAME=VALUE`` overrides of an app's base config.

An app is an engine; a preset is a way to run it, chosen with ``#`` (``VBoussot/ImpactReg:ConvexAdam#composite``).
A preset holds the method's parameters only, never a run's choices (output format, device, batch or patch size), and
it is flat: nothing inherits from another preset. The app's own ``app.json`` publishes presets under ``"presets"``; a
user keeps their own in a file of the same shape (:func:`user_presets_file`). Applying one is applying its overrides,
before the run's own ``--set``, through the one resolver every override goes through (``apply_overrides``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AppRepositoryError

#: A preset's name: what follows ``#`` in an app reference.
PRESET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class Preset:
    """A named way to run an app: its description, the ``NAME=VALUE`` overrides it applies, in order, and whether a
    pair too large for the device may be registered in tiles (``tiles``; when not, it is resampled instead)."""

    name: str
    description: str
    overrides: tuple[str, ...]
    tiles: bool = True

    def to_json(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"description": self.description, "set": list(self.overrides)}
        return entry if self.tiles else {**entry, "tiles": False}


def split_preset(app_id: str) -> tuple[str, str | None]:
    """``'repo:app#preset'`` -> ``('repo:app', 'preset')``; ``(app_id, None)`` when it names no preset.

    A remote reference keeps its token after the preset (``host:port:app#preset|token``), and an existing local path
    is a path, ``#`` included.
    """
    if Path(app_id).expanduser().exists():
        return app_id, None
    head, bar, token = app_id.partition("|")
    base, hashed, name = head.rpartition("#")
    if not hashed:
        return app_id, None
    if not base or not PRESET_NAME.fullmatch(name):
        raise AppRepositoryError(
            f"'{app_id}' names no preset: '{name}' is not a preset name.",
            "Name one as APP#PRESET, the preset in letters, digits, '_' and '-' (ConvexAdam#composite).",
        )
    return base + bar + token, name


def read_presets(entries: Any, source: str) -> dict[str, Preset]:
    """The presets of an ``app.json``'s ``"presets"`` (or of a user's presets file), checked for their shape.

    Each is ``{"description": str, "set": [NAME=VALUE, ...]}``, and ``"tiles": false`` for one that never runs in
    tiles; nothing else, so that a preset stays what it says.
    Whether the overrides resolve against the app's config is checked when the preset is used.
    """
    if entries is None:
        return {}
    if not isinstance(entries, dict):
        raise AppRepositoryError(f"'presets' in {source} must be an object of presets by name.")
    presets: dict[str, Preset] = {}
    for name, entry in entries.items():
        where = f"preset '{name}' of {source}"
        if not isinstance(name, str) or not PRESET_NAME.fullmatch(name):
            raise AppRepositoryError(f"{where}: a name is letters, digits, '_' and '-'.")
        if not isinstance(entry, dict) or set(entry) - {"description", "set", "tiles"}:
            raise AppRepositoryError(
                f"{where}: a preset is {{'description': str, 'set': [NAME=VALUE, ...]}}, with 'tiles': false for one "
                "that never runs in tiles."
            )
        if not isinstance(entry.get("tiles", True), bool):
            raise AppRepositoryError(f"{where}: 'tiles' is true or false.")
        overrides = entry.get("set", [])
        if not isinstance(overrides, list) or not all(isinstance(o, str) and "=" in o for o in overrides):
            raise AppRepositoryError(f"{where}: 'set' is a list of NAME=VALUE strings, as --set takes them.")
        presets[name] = Preset(name, str(entry.get("description", "")), tuple(overrides), entry.get("tiles", True))
    return presets


def user_presets_file(repo: str, app_name: str) -> Path:
    """The file holding a user's own presets of app ``app_name`` of Hugging Face repository ``repo``: under
    ``$KONFAI_PRESETS_DIR``, default ``~/.konfai/presets``, at ``<repo>/<app>.json``."""
    root = Path(os.environ.get("KONFAI_PRESETS_DIR") or Path.home() / ".konfai" / "presets").expanduser()
    return root.joinpath(*repo.split("/"), f"{app_name}.json")


def _read_json_object(path: Path) -> dict[str, Any]:
    """The JSON object at ``path``, ``{}`` when there is no file; a hand-edited file that is not one is refused."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AppRepositoryError(f"{path} is not valid JSON: {error}.") from error
    if not isinstance(data, dict):
        raise AppRepositoryError(f"{path} must hold a JSON object.")
    return data


def load_presets_file(path: Path) -> dict[str, Preset]:
    """A user's presets file, ``{}`` when there is none."""
    return read_presets(_read_json_object(path), str(path))


def merge_presets(published: dict[str, Preset], own_file: Path | None, app_name: str) -> dict[str, Preset]:
    """The ``published`` presets of app ``app_name`` with the user's own from ``own_file`` (none when None). A name in
    both is refused: a preset of one's own never hides a published one."""
    own = load_presets_file(own_file) if own_file else {}
    if clash := sorted(set(own) & set(published)):
        raise AppRepositoryError(
            f"{own_file} has preset(s) {', '.join(map(repr, clash))}, which app '{app_name}' publishes too.",
            "Rename yours: a preset of your own never hides a published one.",
        )
    return {**published, **own}


def write_preset(path: Path, preset: Preset, key: str | None = None) -> None:
    """Add ``preset`` to the JSON file at ``path``, or replace the one of that name: under ``key`` of the file's
    object (an ``app.json``'s ``"presets"``), or at its top level (a user's presets file)."""
    data = _read_json_object(path)
    target = data.setdefault(key, {}) if key else data
    target[preset.name] = preset.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
