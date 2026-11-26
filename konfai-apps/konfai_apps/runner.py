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

"""``python -m konfai_apps.runner MODULE:FUNCTION ARGS...``: a server job running a runner an installed package
registers (entry point group ``konfai_apps.runners``, see ``app_server.run``): FUNCTION is called with ARGS, its argv.
The server resolves MODULE:FUNCTION from its own environment; a client only names the runner."""

import importlib
import sys


def main() -> None:
    target, *argv = sys.argv[1:]
    module, _, function = target.partition(":")
    getattr(importlib.import_module(module), function)(argv)


if __name__ == "__main__":
    main()
