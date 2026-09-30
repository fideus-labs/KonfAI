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

"""``python -m konfai_apps``: the ``konfai-apps`` command, run by the interpreter that imports this package.

A caller that spawns konfai-apps from Python runs it this way rather than through the console script on PATH,
which an unactivated environment, cron or an embedded interpreter lacks, or finds in another environment.
"""

from konfai_apps import main_apps

if __name__ == "__main__":
    main_apps()
