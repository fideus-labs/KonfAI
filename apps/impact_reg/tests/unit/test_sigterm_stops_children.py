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

"""SIGTERM on impact-reg-konfai (SlicerImpactReg's Stop) leaves no preset running as an orphan (GAP6-07)."""

import signal
import subprocess
import sys
import time

import psutil
import pytest

PARENT = """
import subprocess, sys
from impact_reg_konfai import cli
cli._stop_on_sigterm()
subprocess.run([sys.executable, "-c", "import time; time.sleep(60)"], check=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX")
def test_sigterm_on_the_parent_leaves_no_child_alive() -> None:
    parent = subprocess.Popen([sys.executable, "-c", PARENT])
    try:
        deadline = time.monotonic() + 20
        children: list[psutil.Process] = []
        while not children and time.monotonic() < deadline:
            children = psutil.Process(parent.pid).children(recursive=True)
            time.sleep(0.1)
        assert children, "the parent never started its child"
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=5)
        time.sleep(2)
        assert not [child for child in children if child.is_running() and child.status() != psutil.STATUS_ZOMBIE]
    finally:
        parent.kill()
