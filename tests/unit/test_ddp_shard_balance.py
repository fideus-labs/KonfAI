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

"""Regression test: TRAIN/RESUME shards must be equal length to avoid a DDP hang."""

import pytest
from konfai.data.data_manager import Data
from konfai.utils.runtime import State


@pytest.mark.parametrize("state", [State.TRAIN, State.RESUME])
def test_train_split_equalises_indivisible_shards(monkeypatch: pytest.MonkeyPatch, state: State) -> None:
    # DDP(static_graph=True) needs every rank to run the same number of backward all-reduces per
    # epoch. A contiguous split of 7 patches over 3 ranks gives [2, 2, 3] and hangs NCCL on the
    # extra step; drop_last equalises to [2, 2, 2].
    monkeypatch.setenv("KONFAI_STATE", str(state))

    shards = Data._split([(index, 0, 0) for index in range(7)], 3)

    assert [len(shard) for shard in shards] == [2, 2, 2]
    flattened = [item for shard in shards for item in shard]
    assert len(flattened) == len(set(flattened))  # never duplicated across ranks


def test_train_split_two_ranks_indivisible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))
    assert [len(shard) for shard in Data._split([(i, 0, 0) for i in range(5)], 2)] == [2, 2]


def test_train_split_single_process_keeps_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))
    mapping = [(i, 0, 0) for i in range(5)]
    assert Data._split(mapping, 1) == [mapping]  # world_size == 1 is a no-op
