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

"""A two-rank workflow whose last rank refuses or is killed, spawned by the unit tests of the launcher.
A module of its own: a spawned rank imports it to unpickle the workflow, and nothing else."""

import os
import signal

from konfai.utils.errors import ConfigError
from konfai.utils.runtime.distributed import DistributedObject


class FailingLastRank(DistributedObject):
    uses_collectives = False

    def __init__(self, how: str) -> None:
        super().__init__("failing_last_rank")
        self.how = how

    def setup(self, world_size: int) -> None:
        self.dataloader = [[] for _ in range(world_size)]

    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list) -> None:
        if global_rank != world_size - 1:
            return
        if self.how == "refuses":
            raise ConfigError("Rank refuses.", "A designed refusal raised on a rank.")
        os.kill(os.getpid(), signal.SIGKILL)
