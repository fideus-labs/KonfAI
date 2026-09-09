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

"""A remote submission's body is read as it is sent, never assembled in memory first."""

import io
from pathlib import Path

import pytest

pytest.importorskip("requests_toolbelt")

from konfai_apps.app import _multipart_body


class _CountingFile(io.BytesIO):
    """A file whose reads are counted: the largest read is what the encoder holds of it at once."""

    def __init__(self, size: int) -> None:
        super().__init__(b"x" * size)
        self.largest_read = 0
        self.reads = 0

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        self.reads += 1
        chunk = super().read(size)
        self.largest_read = max(self.largest_read, len(chunk))
        return chunk


def test_the_body_streams_the_files_and_still_knows_its_length(tmp_path: Path) -> None:
    volume = _CountingFile(8 << 20)
    body = _multipart_body([("inputs", ("ct.mha", volume))], {"gpu": "0", "models": "best.pt", "skip": None})

    assert body.len > 8 << 20, "the length is known up front: a plain Content-Length, no chunked transfer"
    assert body.content_type.startswith("multipart/form-data; boundary=")
    assert volume.reads == 0, "nothing is read before the connection asks"

    sent = b""
    while chunk := body.read(64 << 10):
        sent += chunk
    assert len(sent) == body.len
    assert volume.largest_read <= 64 << 10, "the file is read in the connection's chunks, never whole"
    assert b'name="gpu"' in sent and b"best.pt" in sent and b"skip" not in sent
    assert sent.index(b'name="gpu"') < sent.index(b'name="inputs"'), "scalars first, files after, as requests sent them"


def test_a_list_value_is_repeated_under_its_name_and_scalars_are_spelled() -> None:
    body = _multipart_body([], {"gpu": [0, 1], "ensemble": True})
    sent = body.read()
    assert sent.count(b'name="gpu"') == 2 and b"True" in sent
