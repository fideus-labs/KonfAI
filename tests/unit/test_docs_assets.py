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

"""The provenance manifest of the documentation figures against the files it describes."""

import hashlib
import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[2] / "docs" / "source" / "_static"
_MANIFEST = _STATIC / "apps" / "ASSET_PROVENANCE.md"


def _recorded_panels() -> dict[Path, str]:
    """Each ``| `panel` | `sha256` |`` row, its path resolved beside the manifest or under ``_static``."""
    panels = {}
    for name, digest in re.findall(r"^\| `([^`]+\.(?:png|webp))` \| `([0-9a-f]{64})` \|$", _MANIFEST.read_text(), re.M):
        beside = _MANIFEST.parent / name
        panels[beside if beside.exists() else _STATIC / name] = digest
    return panels


def test_every_recorded_panel_hash_matches_the_committed_file() -> None:
    panels = _recorded_panels()
    assert panels
    for path, digest in panels.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path


def test_every_transform_and_augmentation_panel_is_recorded() -> None:
    committed = {
        path for folder in ("transforms", "augmentations") for path in (_STATIC / "gallery" / folder).glob("*.png")
    }
    assert committed and committed <= set(_recorded_panels())
