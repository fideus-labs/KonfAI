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

"""The body of a GitHub Release: the committed CHANGELOG section for a tag, verbatim.

Commitizen drafts that section from the commits, but the file is edited afterwards -- a squash merge
collapses to one line, a subject with no conventional prefix is dropped entirely, and a subject
written for a reviewer says nothing to a user. Rendering the commits again at release time would
publish text nobody reviewed, and the file and the release page would then describe one version
differently.

A file rather than a heredoc in the workflow because this decides what gets published, and a thing
that decides that should be testable. See ``tests/unit/test_release_notes.py``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def section_for(changelog: str, tag: str) -> str:
    """The body under ``## <tag>``, stripped, without its heading.

    The heading has to END where the tag does. A ``\\b`` would match at the dot too, so a ``v1.8``
    tag would take ``v1.8.0``'s notes and publish them under its own release -- silently, since both
    are real versions and the text reads fine.
    """
    found = re.search(rf"^## {re.escape(tag)}(?=[ \t]|$)[^\n]*\n(.*?)(?=^## |\Z)", changelog, re.M | re.S)
    if found is None or not found.group(1).strip():
        raise SystemExit(f"CHANGELOG.md carries no section for {tag}. Write it before tagging.")
    return found.group(1).strip() + "\n"


#: PEP 440, as the spec itself writes it. Only the groups this module decides on are named.
_VERSION = re.compile(
    r"^v?(?:\d+!)?\d+(?:\.\d+)*"
    r"(?P<pre>[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?\d*)?"
    r"(?:[-_.]?(?:post|rev|r)[-_.]?\d*|-\d+)?"
    r"(?P<dev>[-_.]?dev[-_.]?\d*)?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?$",
    re.IGNORECASE,
)


def is_prerelease(tag: str) -> bool:
    """Whether ``tag`` names a pre-release — the versions ``latest`` must not follow.

    PEP 440 and not a numeric shape: a POST-release (``v1.8.0.post1``) is stable and must take
    ``latest``, while ``v1.8.0rc1`` and ``v1.8.0.dev1`` must not. A tag that does not parse counts as
    a pre-release, because the alternative is handing ``latest`` to something nobody can classify.
    """
    matched = _VERSION.match(tag.strip())
    return matched is None or bool(matched.group("pre") or matched.group("dev"))


def main(argv: list[str]) -> None:
    if argv[1] == "--prerelease":
        print("true" if is_prerelease(argv[2]) else "false")
        return
    tag, source, destination = argv[1], Path(argv[2]), Path(argv[3])
    destination.write_text(section_for(source.read_text(encoding="utf-8"), tag), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
