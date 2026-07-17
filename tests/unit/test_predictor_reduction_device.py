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

from typing import cast

import pytest
import torch
from konfai.predictor import OutSameAsGroupDataset
from konfai.utils.utils import get_patch_slices_from_shape


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

    @property
    def footprint_shape(self) -> list[int]:
        # A whole-volume accumulator's resident footprint IS its shape (StreamingAccumulator overrides
        # this with its window); _accumulate_device budgets against the footprint.
        return self.shape


def test_accumulate_device_is_cpu_for_a_cpu_dataset() -> None:
    ds = _dataset(torch.device("cpu"))
    # A CPU dataset (or a CPU patch) always blends on the CPU — no GPU accumulation.
    assert ds._accumulate_device(torch.zeros(4, 8, dtype=torch.float16), _FakeAccumulator([8])).type == "cpu"


def test_accumulate_device_is_cpu_when_the_patch_is_on_cpu() -> None:
    ds = _dataset(0)  # the on-GPU convention: a CUDA ordinal int
    # The accumulator's device follows the first patch's; a CPU patch can only blend on the CPU.
    assert ds._accumulate_device(torch.zeros(4, 8, dtype=torch.float16), _FakeAccumulator([8])).type == "cpu"


def _no_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the MEASURED forward transient to 0 (memory_allocated == max_memory_allocated) so a test
    # exercises the accumulator budget alone, independent of any stale process-wide peak.
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU accumulation only applies on CUDA")
def test_accumulate_device_uses_gpu_when_the_volume_fits_and_falls_back_when_it_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_transient(monkeypatch)
    ds = _dataset(0)
    patch = torch.zeros(4, 8, dtype=torch.float16, device="cuda")
    # a small combined volume comfortably fits free VRAM -> blend on the dataset's CUDA device
    assert ds._accumulate_device(patch, _FakeAccumulator([8, 8, 8])).type == "cuda"
    # a volume larger than free VRAM -> memory-safe CPU fallback (never risk an OOM mid-case)
    assert ds._accumulate_device(patch, _FakeAccumulator([10**6, 10**6])).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU accumulation only applies on CUDA")
def test_accumulate_device_falls_back_when_free_vram_is_low(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``free`` is read after the first batch's forward, so little free VRAM is rejected even for a tiny
    # accumulator -- exactly when the resident accumulator plus the remaining forwards would OOM mid-case.
    _no_transient(monkeypatch)
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
    _no_transient(monkeypatch)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (5 * 1024**3, 25 * 1024**3))
    patch = torch.zeros(1, 8, dtype=torch.float16, device="cuda")
    voxels = 512 * 1024**2  # (C+1)=2 x 2 bytes x voxels = 2 GiB per augmentation
    # T=1: 2 GiB < 5 x 0.9 -> GPU ; T=3: 6 GiB >= 4.5 GiB -> whole case on CPU. The accumulation gate now
    # budgets accumulator + measured forward (0 here); the reduction temp fit-checks itself at get_output.
    assert _dataset(0, nb_data_augmentation=1)._accumulate_device(patch, _FakeAccumulator([voxels])).type == "cuda"
    assert _dataset(0, nb_data_augmentation=3)._accumulate_device(patch, _FakeAccumulator([voxels])).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU accumulation only applies on CUDA")
def test_accumulate_device_budgets_the_streaming_window_not_the_whole_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    # A StreamingAccumulator only ever holds its window, so _accumulate_device budgets that, not the
    # full volume: a volume too big to assemble on the GPU still streams on the GPU when its window
    # fits. This is what keeps a huge case both GPU-fast and VRAM-bounded.
    from konfai.data.patching import Cosinus, StreamingAccumulator

    _no_transient(monkeypatch)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (3 * 1024**3, 25 * 1024**3))
    ds = _dataset(0)
    patch = torch.zeros(64, 8, dtype=torch.float16, device="cuda")
    # The whole volume's accumulator (~3.4 GiB) overflows 3 x 0.9; the window (one patch extent,
    # ~0.13 GiB) fits easily.
    patch_slices, _ = get_patch_slices_from_shape([16, 256, 256], [400, 256, 256], 0)
    combine = Cosinus()
    combine.set_patch_config([16, 256, 256], 0)
    whole = _FakeAccumulator([400, 256, 256])
    streaming = StreamingAccumulator(patch_slices, [16, 256, 256], patch_combine=combine, batch=False)
    assert ds._accumulate_device(patch, whole).type == "cpu"
    assert ds._accumulate_device(patch, streaming).type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU reduction only applies on CUDA")
def test_reduction_device_budgets_every_parked_chunk() -> None:
    # A combine:Concat ensemble parks M transformed chunks on the reduce device before the stack: the
    # budget must scale with the chunk count, not assume a single resident chunk.
    ds = _dataset(0)
    free, _ = torch.cuda.mem_get_info(0)

    class _Chunk:
        def __init__(self, numel: int) -> None:
            self._numel = numel

        def numel(self) -> int:
            return self._numel

        def element_size(self) -> int:
            return 2

    fits_alone = int(free / 8)  # (2*1+1) x chunk fits free VRAM...
    assert ds._reduction_device(cast(torch.Tensor, _Chunk(fits_alone)), 1).type == "cuda"
    # ...but (2*8+1) x chunk does not: eight parked chunks must fall back to the CPU reduction.
    assert ds._reduction_device(cast(torch.Tensor, _Chunk(fits_alone)), 8).type == "cpu"


def test_reduction_declares_slab_locality_by_type() -> None:
    # The streamed-write gate asks the reduction whether it is voxel-local (per-voxel over the model/TTA
    # axis). Mean/Median/Concat all reduce orthogonally to the spatial slab axis; a bare custom reduction
    # is unknown and must default to not-streamable, the way a transform defaults to WHOLE_VOLUME.
    from konfai.predictor import Concat, Mean, Median, Reduction

    assert Mean().voxel_local
    assert Median().voxel_local
    assert Concat().voxel_local

    class _CustomReduction(Reduction):
        def __call__(self, tensors):
            return tensors[0]

    assert _CustomReduction().voxel_local is False
