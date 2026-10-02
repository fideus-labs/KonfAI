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

"""The README's links into the published docs, and the site's redirects, land on an anchor that exists.

Sphinx checks the links between its own pages; these two point at a rendered URL, so a renamed heading
leaves them dangling without a warning.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "source"


def _anchors(page: Path) -> set[str]:
    """The ids Sphinx gives a page's sections, its includes' and its labels: docutils' ``make_id`` of the
    rendered text, which is what a URL fragment must name (MyST's own slug keeps an underscore it drops)."""
    from docutils.nodes import make_id

    text = page.read_text(encoding="utf-8")
    for include in re.findall(r"^```\{include\} (\S+)", text, re.M):
        text += "\n" + (page.parent / include).read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)  # a comment in a code block is no heading
    titles = [re.sub(r"\[([^\]]*)\]\([^)]*\)|[`*]", r"\1", title) for title in re.findall(r"^#+ +(.+?) *$", text, re.M)]
    return {make_id(name) for name in [*titles, *re.findall(r"^\(([^)\s]+)\)=$", text, re.M)]}


def _links() -> list[tuple[str, str, str]]:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    return [
        *(
            ("README.md", page, anchor)
            for page, anchor in re.findall(r"konfai\.readthedocs\.io/en/latest/(\S+?)\.html#([\w-]+)", readme)
        ),
        *(("conf.py", page, anchor) for page, anchor in re.findall(r'"([\w/-]+)\.html#([\w-]+)"', conf)),
    ]


@pytest.mark.parametrize(("source", "page", "anchor"), _links())
def test_a_link_into_the_docs_names_an_anchor_the_page_has(source: str, page: str, anchor: str) -> None:
    pytest.importorskip("docutils")
    target = DOCS / f"{page}.md"
    if not target.is_file():
        pytest.skip(f"{page} is not a MyST page")
    assert anchor in _anchors(target), f"{source} links to {page}.html#{anchor}, which the page does not have"
