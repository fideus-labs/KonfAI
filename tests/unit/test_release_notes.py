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

"""What a tag publishes as its release body.

This runs once per release, on a tag, in the job that holds ``contents: write``: so its failures
are expensive and rare, which is exactly the shape of code that goes unchecked. The `v1.8` case
below is not hypothetical: the first version of this extraction used ``\\b`` and would have taken
v1.8.0's notes for it, silently, because both are real versions and the text reads fine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "release_notes.py"
_CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"


def _section_for():
    """The shipped script, loaded from where the workflow runs it."""
    assert _SCRIPT.is_file(), f"the workflow runs a script that is not there: {_SCRIPT}"
    spec = importlib.util.spec_from_file_location("release_notes", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.section_for


_CHANGELOG_SAMPLE = """# Changelog

## v1.8.0 (2026-08-04)

### Features

- something a user gets

## v1.7.0 (2026-07-29)

### Features

- an older thing
"""


def test_it_takes_the_section_the_tag_names() -> None:
    body = _section_for()(_CHANGELOG_SAMPLE, "v1.8.0")
    assert "something a user gets" in body
    assert "an older thing" not in body, "the section stops at the next heading"
    assert not body.startswith("## "), "the heading belongs to the release title, not its body"


@pytest.mark.parametrize("tag", ["v1.8", "v1", "v1.8.0rc1", "v9.9.9", ""])
def test_a_tag_with_no_section_of_its_own_fails_the_job(tag: str) -> None:
    """Refusing is the point: the alternative is an empty release, or (for a prefix like ``v1.8``) a release carrying someone else's notes under its own name."""
    with pytest.raises(SystemExit, match="carries no section"):
        _section_for()(_CHANGELOG_SAMPLE, tag)


def test_the_real_changelog_answers_for_the_version_being_cut() -> None:
    """The file in the tree, not a fixture: this is what the next tag will actually publish."""
    body = _section_for()(_CHANGELOG.read_text(encoding="utf-8"), "v1.8.0")
    assert body.strip(), "v1.8.0 has no notes to publish"
    assert body.endswith("\n")
    assert "## v1.7.0" not in body, "the section must not run into the previous release"


def _is_prerelease():
    """The shipped classifier, loaded from where both workflow jobs run it."""
    spec = importlib.util.spec_from_file_location("release_notes", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_prerelease


@pytest.mark.parametrize(
    "tag,prerelease",
    [
        ("v1.8.0", False),
        ("v1.8.0.post1", False),  # a POST-release is stable and must take `latest`
        ("v1.8.0-1", False),  # PEP 440's implicit post-release spelling
        ("v1.8.0rc1", True),
        ("v1.8.0a1", True),
        ("v1.8.0b2", True),
        ("v1.8.0.dev1", True),
        ("v1.8.0rc1.post1", True),  # a post of a pre is still a pre
        ("v1.9.0-backport", True),  # unparseable: never hand `latest` to what nobody can classify
        ("nawak", True),
    ],
)
def test_the_tag_classifier_follows_pep_440(tag: str, prerelease: bool) -> None:
    """What `latest` follows, on Docker Hub and on the releases page.

    Neither a substring search nor a numeric shape: `contains(tag, 'a')` calls `v1.9.0-backport` a
    pre-release, and `^[0-9.]+$` calls the stable `v1.8.0.post1` one, which would then be published
    as a pre-release and lose `latest` to nothing.
    """
    assert _is_prerelease()(tag) is prerelease
