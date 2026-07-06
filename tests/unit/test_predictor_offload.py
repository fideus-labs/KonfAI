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

import pytest
import torch
from konfai.predictor import OutSameAsGroupDataset


def _dataset(monkeypatch: pytest.MonkeyPatch) -> OutSameAsGroupDataset:
    monkeypatch.setenv("KONFAI_config_file", "unused.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    return OutSameAsGroupDataset(same_as_group="default:default", dataset_filename="default|./Dataset:mha")


def test_offload_cpu_tensor_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(monkeypatch)
    x = torch.randn(3, 4, 4)
    out = ds._offload_to_cpu(x)
    assert torch.equal(out, x.detach().cpu())
    assert ds._pin_buffer is None  # no page-locked buffer allocated for a CPU patch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging only applies to CUDA patches")
def test_offload_large_cuda_patch_is_bit_identical_and_pageable(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(monkeypatch)
    # above _PINNED_OFFLOAD_MIN_BYTES (64 MiB): 40M fp16 elements = 80 MiB
    x = torch.empty(40 * 1024 * 1024, dtype=torch.float16, device="cuda").normal_()

    out = ds._offload_to_cpu(x)

    assert torch.equal(out, x.detach().cpu())  # staging through pinned memory changes nothing numerically
    assert out.device.type == "cpu"
    assert not out.is_pinned()  # the stored patch must be pageable, not page-locked
    assert ds._pin_buffer is not None and ds._pin_buffer.is_pinned()
    # the one-patch pinned buffer is reused, not reallocated, for the next same-shape patch
    reused = ds._pin_buffer
    ds._offload_to_cpu(x)
    assert ds._pin_buffer is reused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging only applies to CUDA patches")
def test_offload_small_cuda_patch_takes_plain_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _dataset(monkeypatch)
    x = torch.randn(8, 8, 8, device="cuda")  # well under the staging threshold
    out = ds._offload_to_cpu(x)
    assert torch.equal(out, x.detach().cpu())
    assert ds._pin_buffer is None  # small patches never allocate the pinned buffer
