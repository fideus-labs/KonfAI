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

"""What a remote submission sends, read back from its streamed body: the ``files`` and ``data`` a
``requests.post(files=..., data=...)`` spy used to see, in the order they were sent."""

import io
import re
from typing import Any

_DISPOSITION = re.compile(rb'name="([^"]*)"(?:; filename="([^"]*)")?')


def decode_multipart(body: Any) -> tuple[list[tuple[str, tuple[str, io.BytesIO]]], dict[str, str]]:
    """``(files, data)`` of a ``MultipartEncoder`` body: files as ``(field, (filename, handle))``, the
    scalar fields as strings (a repeated field keeps its last value, as a dict would)."""
    from requests_toolbelt.multipart.decoder import MultipartDecoder

    files: list[tuple[str, tuple[str, io.BytesIO]]] = []
    data: dict[str, str] = {}
    for part in MultipartDecoder(body.read(), body.content_type).parts:
        match = _DISPOSITION.search(part.headers.get(b"Content-Disposition", b""))
        assert match is not None, part.headers
        field, filename = match.group(1).decode(), match.group(2)
        if filename is None:
            data[field] = part.content.decode()
        else:
            handle = io.BytesIO(part.content)
            handle.name = filename.decode()  # type: ignore[attr-defined]
            files.append((field, (filename.decode(), handle)))
    return files, data
