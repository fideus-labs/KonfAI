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

"""Regression test: uploads sharing a basename must not silently overwrite each other."""

import importlib.util
import io
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")

import konfai_apps.app_server as app_server  # noqa: E402


class _Upload:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self.file = io.BytesIO(content)


def test_save_uploads_disambiguates_identical_basenames(tmp_path: Path) -> None:
    uploads = [_Upload("case.nii.gz", b"AAA"), _Upload("case.nii.gz", b"BBB")]

    paths = app_server.save_uploads(uploads, tmp_path)

    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert all(p.exists() for p in paths)
    assert {p.read_bytes() for p in paths} == {b"AAA", b"BBB"}  # neither case dropped
    assert all(p.name.endswith(".nii.gz") for p in paths)  # extension preserved
