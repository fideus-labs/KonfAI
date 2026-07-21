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

"""Live control channel for a running training job.

A training loop tails a ``control.json`` written into its run directory by an external steerer (the KonfAI
MCP server / Studio) to change tunables — learning rate, validation interval — mid-run without a restart.
Each write carries an incrementing ``revision`` so a change is consumed exactly once; the trainer applies it
at a DDP poll boundary and records an audit trail into the run's config snapshot.
"""

import json
from pathlib import Path
from typing import Any


class LiveControl:
    """Reads a run's control file and yields each new revision's tunables exactly once.

    The file shape is ``{"revision": <int>, "<key>": <value>, ...}``; ``take()`` returns the tunables (minus
    ``revision``) only when a strictly newer revision has appeared since the last call, so a persisted control
    file is not re-applied on every poll.
    """

    def __init__(self, control_path: Path):
        self.control_path = control_path
        self._revision = 0

    def take(self) -> dict[str, Any] | None:
        """The pending tunables if a newer revision was written since the last take, else ``None``. A missing,
        unreadable, or malformed file reads as 'nothing pending' — steering is best-effort, never fatal."""
        try:
            data = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        revision = data.get("revision", 0)
        if not isinstance(revision, int) or revision <= self._revision:
            return None
        self._revision = revision
        return {key: value for key, value in data.items() if key != "revision"}
