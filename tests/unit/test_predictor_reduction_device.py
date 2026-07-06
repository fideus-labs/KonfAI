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


def _dataset(device: torch.device | int, nb_data_augmentation: int = 1) -> OutSameAsGroupDataset:
    ds = OutSameAsGroupDataset.__new__(OutSameAsGroupDataset)
    ds.device = device  # NeedDevice stores a torch.device on CPU and a CUDA ordinal (int) on GPU
    ds.nb_data_augmentation = nb_data_augmentation
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


class _FakeAccumulator:
    def __init__(self, shape: list[int]) -> None:
        self.shape = shape


def test_accumulate_device_is_cpu_for_a_cpu_dataset() -> None:
    ds = _dataset(torch.device("cpu"))
    # A CPU dataset (or a CPU patch) always blends on the CPU — no GPU accumulation.
    assert ds._accumulate_device(torch.zeros(4, 8, dtype=torch.float16), _FakeAccumulator([8])).type == "cpu"


def test_accumulate_device_is_cpu_when_the_patch_is_on_cpu() -> None:
    ds = _dataset(0)  # the on-GPU convention: a CUDA ordinal int
    # The accumulator's device follows the first patch's; a CPU patch can only blend on the CPU.
    assert ds._accumulate_device(torch.zeros(4, 8, dtype=torch.float16), _FakeAccumulator([8])).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU accumulation only applies on CUDA")
def test_accumulate_device_uses_gpu_when_the_volume_fits_and_falls_back_when_it_does_not() -> None:
    ds = _dataset(0)
    patch = torch.zeros(4, 8, dtype=torch.float16, device="cuda")
    # a small combined volume comfortably fits free VRAM -> blend on the dataset's CUDA device
    assert ds._accumulate_device(patch, _FakeAccumulator([8, 8, 8])).type == "cuda"
    # a volume larger than free VRAM -> memory-safe CPU fallback (never risk an OOM mid-case)
    assert ds._accumulate_device(patch, _FakeAccumulator([10**6, 10**6])).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU accumulation only applies on CUDA")
def test_accumulate_device_falls_back_when_free_vram_is_low(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``free`` is read after the first batch's forward, so a larger batch (which leaves little free
    # VRAM) is rejected even for a tiny accumulator -- exactly when the resident accumulator plus the
    # remaining forwards would otherwise OOM mid-case.
    ds = _dataset(0)
    patch = torch.zeros(4, 8, dtype=torch.float16, device="cuda")
    accumulator = _FakeAccumulator([8, 8, 8])  # a negligible combined volume
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (1024, 25 * 1024**3))  # ~1 KiB free
    assert ds._accumulate_device(patch, accumulator).type == "cpu"
# with plenty of free VRAM the very same accumulator blends on the GPU
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (25 * 1024**3, 25 * 1024**3))
        assert ds._accumulate_device(patch, accumulator).type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU accumulation only applies on CUDA")
def test_accumulate_device_budgets_every_tta_augmentation(monkeypatch: pytest.MonkeyPatch) -> None:
    # All of a case's TTA accumulators are resident simultaneously (``is_done`` requires every
    # augmentation complete before ``get_output``), so the per-case decision budgets accumulator x T:
    # a volume where one copy fits but T copies do not must take the CPU path for the WHOLE case —
    # a per-augmentation flip would hand the reduction a mixed CPU/CUDA list and crash it.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (12 * 1024**3, 25 * 1024**3))
    patch = torch.zeros(1, 8, dtype=torch.float16, device="cuda")
    voxels = 512 * 1024**2  # (C+1)=2 x 2 bytes x voxels = 2 GiB per augmentation
    # T=1: 2 GiB x 2 < 12 GiB x 0.56 -> GPU; T=3: 6 GiB x 2 >= 6.72 GiB -> whole case on CPU
    assert _dataset(0, nb_data_augmentation=1)._accumulate_device(patch, _FakeAccumulator([voxels])).type == "cuda"
    assert _dataset(0, nb_data_augmentation=3)._accumulate_device(patch, _FakeAccumulator([voxels])).type == "cpu"
