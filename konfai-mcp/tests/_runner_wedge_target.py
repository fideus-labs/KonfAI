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

"""Lightweight spawn target for the run_api_in_subprocess bounded-join test.

Kept out of the test module so the spawn child imports only threading/time, not the whole test file.
"""

import threading
import time
from typing import Any


def wedge_after_result(value: Any) -> dict[str, Any]:
    """Return a result, then keep the process alive so it wedges during teardown.

    A non-daemon thread that sleeps far longer than the test blocks interpreter shutdown after the
    target returns, standing in for a child that produced its result but will not exit (e.g. a native
    or CUDA context). run_api_in_subprocess already has the result, so a bounded join + escalation must
    reap it instead of hanging the caller forever.
    """
    threading.Thread(target=time.sleep, args=(3600,), daemon=False).start()
    return {"echoed": value}
