# SPDX-License-Identifier: Apache-2.0
"""A pinned offload buffer is lent only while an accumulator consumes its patch."""

from contextlib import contextmanager

import pytest
import torch
from konfai.data.patching import Accumulator, Cosinus, Gaussian, Mean, StreamingAccumulator, Trim
from konfai.predictor.output import OutputDataset
from konfai.utils.utils import get_patch_slices_from_shape

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA offload needs a CUDA device")


def _owner(force_pinned=False):
    owner = OutputDataset.__new__(OutputDataset)
    owner._pin_buffer = None
    owner._accum_device = {0: torch.device("cpu")}
    if force_pinned:
        owner._PINNED_OFFLOAD_MIN_BYTES = 0
    return owner


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("combine_type", [None, Mean, Cosinus, Gaussian, Trim])
@pytest.mark.parametrize("streaming", [False, True])
def test_offload_storage_reuse_cannot_change_accumulated_voxels(device, combine_type, streaming):
    shape, patch_size, overlap = [13, 9, 11], [4, 4, 4], 2
    slices = get_patch_slices_from_shape(patch_size, shape, overlap)

    def combine():
        value = combine_type() if combine_type else None
        if value is not None:
            value.set_patch_config(patch_size, overlap)
        return value

    owner = _owner(force_pinned=True)
    reference = Accumulator(slices, patch_size, combine(), batch=False)
    accumulator_type = StreamingAccumulator if streaming else Accumulator
    actual = accumulator_type(slices, patch_size, combine(), batch=False)
    source = torch.empty(2, *patch_size, device=device, dtype=torch.float16)
    slabs = []
    pinned = None
    for index in range(len(slices)):
        expected_patch = torch.arange(source.numel(), dtype=torch.float16).reshape(source.shape) + index
        reference.add_layer(index, expected_patch)
        source.copy_(expected_patch)
        slabs.extend(owner._blend_patch(0, index, source, actual))
        source.fill_(float("nan"))
        if device == "cuda":
            assert owner._pin_buffer is not None and owner._pin_buffer.is_pinned()
            if pinned is not None:
                assert owner._pin_buffer is pinned
            pinned = owner._pin_buffer
            pinned.fill_(float("nan"))
    if streaming:
        slabs.extend(actual.finalize())
        result = torch.cat([slab for _, slab in slabs], dim=1)
        cursor = 0
        for span, slab in slabs:
            assert span.start == cursor and slab.shape[1] == span.stop - span.start
            cursor = span.stop
        assert cursor == shape[0]
    else:
        result = actual.assemble()
    assert list(result.shape) == [2, *shape]
    assert torch.equal(result, reference.assemble())


@CUDA
def test_cpu_blend_consumes_the_borrowed_buffer_inside_its_context(monkeypatch):
    owner = _owner(force_pinned=True)
    real_borrow = owner._borrow_cpu_patch
    active = False

    @contextmanager
    def tracked_borrow(layer):
        nonlocal active
        with real_borrow(layer) as cpu_patch:
            active = True
            try:
                yield cpu_patch
            finally:
                active = False

    class InspectAccumulator(Accumulator):
        def add_layer(self, index, layer):
            assert active
            assert layer is owner._pin_buffer
            return super().add_layer(index, layer)

    monkeypatch.setattr(owner, "_borrow_cpu_patch", tracked_borrow)
    accumulator = InspectAccumulator([(slice(0, 3), slice(0, 3))], [3, 3], batch=False)
    source = torch.arange(18, dtype=torch.float32, device="cuda").reshape(2, 3, 3)
    owner._blend_patch(0, 0, source, accumulator)
    assert not active
    owner._pin_buffer.fill_(float("nan"))
    assert torch.equal(accumulator.assemble(), source.cpu())


@CUDA
def test_owned_cpu_results_survive_shape_dtype_and_buffer_reuse():
    owner = _owner(force_pinned=True)
    retained = []
    previous_buffer = None
    for shape, dtype, value in [
        ((2, 3, 3), torch.float16, 1),
        ((2, 3, 3), torch.float16, 2),
        ((3, 3, 3), torch.float16, 3),
        ((3, 3, 3), torch.float32, 4),
    ]:
        source = torch.full(shape, value, dtype=dtype, device="cuda")
        owned = owner._offload_to_cpu(source)
        assert not owned.is_pinned()
        assert owned.data_ptr() != owner._pin_buffer.data_ptr()
        if value == 2:
            assert owner._pin_buffer is previous_buffer
        previous_buffer = owner._pin_buffer
        retained.append((owned, value))
        owner._pin_buffer.fill_(-99)
    for result, value in retained:
        assert torch.equal(result, torch.full_like(result, value))


@CUDA
def test_small_cuda_patches_keep_the_pageable_fallback():
    owner = _owner()
    source = torch.arange(18, dtype=torch.float32, device="cuda").reshape(2, 3, 3)
    accumulator = Accumulator([(slice(0, 3), slice(0, 3))], [3, 3], batch=False)
    owner._blend_patch(0, 0, source, accumulator)
    assert owner._pin_buffer is None
    assert torch.equal(accumulator.assemble(), source.cpu())


@CUDA
def test_failed_pinned_allocation_keeps_the_pageable_fallback(monkeypatch):
    owner = _owner(force_pinned=True)
    source = torch.arange(18, dtype=torch.float32, device="cuda").reshape(2, 3, 3)
    real_empty = torch.empty
    attempts = []

    def empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            attempts.append(True)
            raise RuntimeError("pinned allocation refused")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    accumulator = Accumulator([(slice(0, 3), slice(0, 3))], [3, 3], batch=False)
    owner._blend_patch(0, 0, source, accumulator)
    assert attempts == [True] and owner._pin_buffer is None
    assert torch.equal(accumulator.assemble(), source.cpu())
    assert torch.equal(owner._offload_to_cpu(source), source.cpu())


@CUDA
def test_initial_gpu_accumulation_oom_falls_back_to_synchronous_cpu_borrow():
    owner = _owner(force_pinned=True)
    owner._accum_device[0] = torch.device("cuda")

    class InitiallyFullGPU(Accumulator):
        def add_layer(self, index, layer):
            if layer.device.type == "cuda":
                raise torch.cuda.OutOfMemoryError("simulated first allocation failure")
            assert layer is owner._pin_buffer
            return super().add_layer(index, layer)

    accumulator = InitiallyFullGPU([(slice(0, 3), slice(0, 3))], [3, 3], batch=False)
    source = torch.ones(2, 3, 3, device="cuda")
    owner._blend_patch(0, 0, source, accumulator)
    assert owner._accum_device[0].type == "cpu"
    owner._pin_buffer.fill_(float("nan"))
    assert torch.equal(accumulator.assemble(), source.cpu())


@CUDA
def test_interleaved_output_sinks_keep_independent_reusable_buffers():
    owners = [_owner(force_pinned=True), _owner(force_pinned=True)]
    spans = [(slice(0, 3), slice(0, 3)), (slice(3, 6), slice(0, 3))]
    sinks = [Accumulator(spans, [3, 3], batch=False) for _ in owners]
    inputs = [
        torch.full((channels, 3, 3), value, device="cuda", dtype=torch.float16) for channels, value in [(2, 5), (3, 9)]
    ]
    pinned = [None, None]
    for index in range(2):
        for output_index, (owner, sink, source) in enumerate(zip(owners, sinks, inputs, strict=True)):
            owner._blend_patch(0, index, source, sink)
            if index:
                assert owner._pin_buffer is pinned[output_index]
            pinned[output_index] = owner._pin_buffer
            owner._pin_buffer.fill_(float("nan"))
    assert pinned[0].data_ptr() != pinned[1].data_ptr()
    for sink, source in zip(sinks, inputs, strict=True):
        assert torch.equal(sink.assemble(), source.cpu().repeat(1, 2, 1))
