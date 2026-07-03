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

"""Regression test: a nested sibling's output must not be dropped by a stale inner-match set."""

import torch
from konfai.network.network import ModuleArgsDict


class _Add(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.value


def _nested(value: float, inner_out: int | str) -> ModuleArgsDict:
    sub = ModuleArgsDict()
    sub.add_module("L", _Add(value), in_branch=[0], out_branch=[inner_out])
    return sub


def test_later_nested_sibling_output_reaches_downstream() -> None:
    # M1 writes branch 0 via inner-match; M2 shares out_branch=[0] but its inner module writes a
    # different branch, so it relies on the fallback. A ``tmp`` kept across siblings made the fallback
    # see branch 0 as already filled (by M1) and silently drop M2, leaving M1's value downstream.
    graph = ModuleArgsDict()
    graph.add_module("M1", _nested(1.0, 0), in_branch=[0], out_branch=[0])
    graph.add_module("M2", _nested(10.0, "zz"), in_branch=[0], out_branch=[0])
    graph.add_module("Id", torch.nn.Identity(), in_branch=[0], out_branch=[0])

    outputs = list(graph.named_forward(torch.zeros(1)))
    downstream = [tensor for name, tensor in outputs if name.startswith("Id")][-1]

    # Branch 0: input 0 -> M1 (+1) = 1 -> M2 reads branch 0 (+10) = 11 -> Id. Not M1's stale 1.
    assert downstream.item() == 11.0
