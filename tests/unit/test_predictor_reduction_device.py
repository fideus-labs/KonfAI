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


def _dataset(device: torch.device | int) -> OutSameAsGroupDataset:
    ds = OutSameAsGroupDataset.__new__(OutSameAsGroupDataset)
    ds.device = device  # NeedDevice stores a torch.device on CPU and a CUDA ordinal (int) on GPU
    return ds


def test_output_dataset_device_defaults_to_cpu_without_to(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: ``Dataset.__init__`` does not forward ``super().__init__()``, so ``NeedDevice.__init__``
    # is skipped through the MRO. A real (not ``__new__``) construction must still leave ``self.device``
    # set, otherwise a CPU-only PREDICTION run (device propagation is CUDA-gated) raises AttributeError.
    monkeypatch.setenv("KONFAI_config_file", "unused.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    ds = OutSameAsGroupDataset(same_as_group="default:default", dataset_filename="default|./Dataset:mha")
    assert ds.device == torch.device("cpu")
    assert ds._reduction_device(torch.zeros(4, 8, dtype=torch.float16)).type == "cpu"


def test_reduction_device_is_cpu_for_a_cpu_dataset() -> None:
    ds = _dataset(torch.device("cpu"))
    assert ds._reduction_device(torch.zeros(4, 8, dtype=torch.float16)).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU reduction only applies on CUDA")
def test_reduction_device_uses_gpu_when_it_fits_and_falls_back_when_it_does_not() -> None:
    ds = _dataset(0)  # the on-GPU convention: a CUDA ordinal int, not a torch.device
    # a tiny chunk comfortably fits free VRAM -> reduction runs on the dataset's CUDA device
    assert ds._reduction_device(torch.zeros(4, 8, dtype=torch.float16)).type == "cuda"

    # a chunk claiming more than free VRAM -> memory-safe CPU fallback
    class _Oversized:
        def numel(self) -> int:
            return 10**15

        def element_size(self) -> int:
            return 2

    assert ds._reduction_device(_Oversized()).type == "cpu"
