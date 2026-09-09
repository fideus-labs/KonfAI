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

from email import message_from_string
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parents[1]


def _release_version() -> str:
    pkg_info = Path(__file__).with_name("PKG-INFO")
    if pkg_info.exists():
        return message_from_string(pkg_info.read_text())["Version"]
    from setuptools_scm import get_version

    return get_version(root=str(_ROOT), tag_regex=r"^v(?P<version>.*)$", local_scheme="no-local-version")


def _sibling(name: str, version: str) -> str:
    """The family ships in lockstep: at a release tag the sibling is pinned to the exact version.
    From a working tree (a ``.dev`` version) the pin is the closest release or newer, so
    ``pip install -e`` resolves against the core installed beside it; the publish workflow
    builds at the tag, where the pin is exact."""
    if ".dev" not in version:
        return f"{name}=={version}"
    try:
        from setuptools_scm import get_version

        floor = get_version(
            root=str(_ROOT),
            tag_regex=r"^v(?P<version>.*)$",
            version_scheme=lambda scm_version: str(scm_version.tag),
            local_scheme="no-local-version",
        )
    except Exception:  # no git history to read a tag from: the dev version's own floor
        floor = version
    return f"{name}>={floor}"


# konfai and konfai-apps are pinned to the exact release version at a tag: runner.py imports core
# internals (konfai.transformer, konfai.network.network) whose layout moves between minor releases, so
# the family ships in lockstep, like the apps/* bundles pin konfai and studio pins konfai-mcp.
_version = _release_version()

setup(
    install_requires=[
        _sibling("konfai", _version),
        _sibling("konfai-apps", _version),
        # >= 2.10.2 for stateless streamable HTTP (stateless_http/json_response run
        # kwargs and the FASTMCP_STATELESS_HTTP setting) per MCP spec rev 2025-03-26.
        "fastmcp>=2.10.2",
        "ruamel.yaml",
        "numpy",
    ]
)
