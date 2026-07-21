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


# konfai-mcp is pinned to the exact release version -- Studio drives it over stdio and imports its
# internals (konfai_mcp.live_parse), so the two ship in lockstep, like the apps/* bundles pin konfai.
_version = _release_version()

setup(
    install_requires=[
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",  # [standard] pulls websockets -- the integrated terminal socket
        "fastmcp>=2.4",
        f"konfai-mcp=={_version}",
        "claude-agent-sdk>=0.2",  # default brain: the Claude Code subscription
        "pywinpty>=2.0; sys_platform == 'win32'",  # Windows ConPTY for the terminal (POSIX uses pty)
    ]
)
