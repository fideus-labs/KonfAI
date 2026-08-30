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

"""A sweep reads one block ahead and writes one behind, and that changes nothing but the clock.

Pinned here: the order, the bytes and the failures are the sequential loop's, and a rank with one
core keeps that loop, so ``OMP_NUM_THREADS=1`` means a serial run.
"""

import itertools
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import (
    SWEEP_CLOCK,
    DatasetManager,
    DatasetPatch,
    _ReadAhead,
    _stage_failure,
    _sweep_pipeline_depth,
    _sweep_resident_regions,
    _WriteBehind,
)
from konfai.data.transform import Clip, LocalityKind, PatchLocality, RegionContext, Resample, Save, Transform
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError
from oracle_support import geometry, manager

pytest.importorskip("SimpleITK")


# ---------------------------------------------------------------- reading one block ahead


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_read_ahead_yields_every_block_in_order(depth: int) -> None:
    with _ReadAhead(iter(range(5)), depth) as blocks:
        assert list(blocks) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_read_ahead_raises_what_the_read_raised(depth: int) -> None:
    def failing():
        yield 0
        raise ValueError("unreadable region")

    with pytest.raises(ValueError, match="unreadable region"):
        with _ReadAhead(failing(), depth) as blocks:
            list(blocks)


def test_read_ahead_holds_no_more_than_its_depth() -> None:
    """The bound is the memory the plan sized: unbounded, a producer reads the whole stream into RAM
    while the first block is still being transformed."""
    started = 0

    def counted():
        nonlocal started
        for index in range(100):
            started = index + 1
            yield index

    with _ReadAhead(counted(), 2) as blocks:
        first = next(blocks)
        time.sleep(0.2)  # far longer than an unbounded producer needs to reach block 100
        reached = started
        assert first == 0
        # Two slots, the block handed over, and the one the producer is blocked trying to put.
        assert reached <= 4, f"the producer read {reached} blocks ahead of a two-slot queue"
        assert list(blocks) == list(range(1, 100))


def test_read_ahead_stops_the_producer_when_the_consumer_leaves_early() -> None:
    """A queue nobody drains holds its producer forever, and a daemon thread hides that until exit."""
    reached = 0

    def counted():
        nonlocal reached
        for index in range(1000):
            reached = index
            yield index

    with _ReadAhead(counted(), 2) as blocks:
        assert list(itertools.islice(blocks, 3)) == [0, 1, 2]
    assert reached < 100, "the producer ran to the end of a stream nobody was reading"


def test_a_single_core_rank_keeps_the_sequential_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert _sweep_pipeline_depth() == 0
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert _sweep_pipeline_depth() == 1


# ---------------------------------------------------------------- writing one block behind


class _RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.opened: set = set()

    def write(self, key, region, block, header) -> None:
        del header
        self.opened.add(key)
        self.calls.append((key, region, int(block)))

    def close(self) -> None:
        self.calls.append(("close", None, -1))


@pytest.mark.parametrize("depth", [0, 1])
def test_write_behind_writes_every_block_in_order(depth: int) -> None:
    writer = _RecordingWriter()
    write = _WriteBehind(writer, depth)
    try:
        for index in range(4):
            write.write("k", (slice(index, index + 1),), np.int64(index), Attribute())
        write.close()
    finally:
        write.shutdown()
    assert [call[2] for call in writer.calls] == [0, 1, 2, 3, -1], "written in order, then published"


def test_write_behind_opens_and_closes_on_the_thread_that_writes() -> None:
    """The h5 backend holds a per-file lock across the stream's life: a release from a thread that
    did not take it raises and leaves the file locked for the rest of the process."""
    threads: set[int] = set()

    class _Recording:
        opened: ClassVar[set] = set()

        def write(self, *_args) -> None:
            threads.add(threading.get_ident())

        def close(self) -> None:
            threads.add(threading.get_ident())

        def abort(self, _error) -> None:
            threads.add(threading.get_ident())

    write = _WriteBehind(_Recording(), 1)
    try:
        write.write("k", (slice(0, 1),), np.int64(0), Attribute())
        write.close()
    finally:
        write.shutdown()
    assert len(threads) == 1 and threading.get_ident() not in threads


def test_write_behind_raises_what_the_write_raised() -> None:
    class _Failing:
        def write(self, *_args) -> None:
            raise OSError("no space left on device")

    write = _WriteBehind(_Failing(), 1)
    try:
        with pytest.raises(OSError, match="no space left"):
            for index in range(4):
                write.write("k", (slice(0, 1),), np.int64(index), Attribute())
            write.flush()
    finally:
        write.shutdown()


# ---------------------------------------------------------------- the bytes are the same bytes


def _attributes() -> Attribute:
    return geometry((10.0, 20.0, 30.0), (0.5, 1.5, 2.0))


def _manager(source: Dataset, transforms: list, out: Path) -> DatasetManager:
    del out
    return manager(source, transforms, name="CASE_000", patch=DatasetPatch([4, 5, 4]))


def _sweep_into(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch, depth: int) -> np.ndarray:
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
    monkeypatch.setattr("konfai.data.patching.sweep._sweep_pipeline_depth", lambda: depth)
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / f"src_{name}", "mha")
    source.write("CT", "CASE_000", (rng.random((1, 14, 10, 8)) * 100).astype(np.float32), _attributes())
    manager = _manager(source, [Clip(min_value=10.0, max_value=90.0), Save(f"{tmp_path / name}:h5")], tmp_path)
    assert CaseMaterializer(manager).materialize() is Verdict.STREAM
    return Dataset(tmp_path / name, "h5").read_data("CT", "CASE_000")[0]


def test_a_pipelined_sweep_writes_the_bytes_the_sequential_one_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequential = _sweep_into(tmp_path, "serial", monkeypatch, depth=0)
    pipelined = _sweep_into(tmp_path, "pipelined", monkeypatch, depth=1)
    np.testing.assert_array_equal(pipelined, sequential)


class _DeclaredThenHanded(Transform):
    """A per-voxel stage logging the regions declared to it and the ones it is then handed."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, RegionContext]] = []

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def plan_region_reads(self, name: str, contexts) -> None:
        self.events.extend(("declared", context) for context in contexts)

    def stream_region(self, name: str, tensor, context: RegionContext, cache_attribute: Attribute):
        self.events.append(("handed", context))
        return tensor

    def __call__(self, name: str, tensor, cache_attribute: Attribute):
        return tensor


def test_a_sweep_declares_to_each_stage_the_regions_it_will_hand_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage reading a companion volume beside its region (a mask) declares those reads from what
    the sweep tells it ahead: the contexts ``stream_region`` will be handed, all of them before the
    first, in the order they come. And in the stage's own space: through a resample after it, the
    regions it sees are the source hulls the landing's blocks pull, not the blocks."""
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
    monkeypatch.setattr("konfai.data.patching.sweep._sweep_pipeline_depth", lambda: 0)
    source = Dataset(tmp_path / "src", "mha")
    source.write("CT", "CASE_000", np.ones((1, 14, 10, 8), np.float32), _attributes())
    stage = _DeclaredThenHanded()
    chain = [stage, Resample(shape=[7, 5, 4]), Save(f"{tmp_path / 'out'}:h5")]
    assert CaseMaterializer(_manager(source, chain, tmp_path)).materialize() is Verdict.STREAM

    declared = [context for kind, context in stage.events if kind == "declared"]
    handed = [context for kind, context in stage.events if kind == "handed"]
    assert len(handed) > 1 and declared == handed
    assert [kind for kind, _context in stage.events[: len(declared)]] == ["declared"] * len(declared)
    assert {context.source_shape for context in handed} == {(14, 10, 8)}, "its own input, not the landing"


# ---------------------------------------------------------------- and the clock closes on it


@pytest.mark.parametrize("depth", [0, 1])
def test_a_sweep_accounts_for_every_second_of_its_own_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    """Optimising a stage while a third of the run is attributed to nothing is guesswork. The
    sweep's own thread is fully decomposed, so what its phases do not name is a residual the
    report prints rather than a gap it hides."""
    SWEEP_CLOCK.reset()
    _sweep_into(tmp_path, f"clocked_{depth}", monkeypatch, depth=depth)

    wall = SWEEP_CLOCK.spent("sweep")
    named = sum(SWEEP_CLOCK.spent(phase) for phase in ("chain", "fetch", "wait(read)", "wait(write)"))
    assert wall > 0.0
    assert 0.0 <= wall - named <= wall, "a phase is counted outside the sweep it belongs to"

    report = SWEEP_CLOCK.report(min_seconds=0.0)
    assert report is not None
    for phase in ("chain", "fetch", "wait(read)", "wait(write)", "other", "read", "write"):
        assert phase in report


def test_a_serial_sweep_waits_out_the_read_it_does_not_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no pipeline the read happens on the sweep's own thread, so the sweep waits exactly as
    long as the read takes. Overlapping it is what the pipeline buys, and what the two numbers
    diverging measures."""
    SWEEP_CLOCK.reset()
    _sweep_into(tmp_path, "serial_clock", monkeypatch, depth=0)

    assert SWEEP_CLOCK.spent("wait(read)") >= SWEEP_CLOCK.spent("read") > 0.0


def test_the_clock_sums_the_sweeps_a_rank_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    SWEEP_CLOCK.reset()
    _sweep_into(tmp_path, "first", monkeypatch, depth=0)
    one = SWEEP_CLOCK.spent("sweep")
    _sweep_into(tmp_path, "second", monkeypatch, depth=0)
    assert SWEEP_CLOCK.spent("sweep") > one


# ---------------------------------------------------------------- and what it says when it cannot


def test_a_dtype_torch_has_no_kernel_for_names_itself_and_its_remedy() -> None:
    """torch ships no kernel for several dtypes a store legitimately holds: uint16 is what
    microscopy writes, and torch implements for it neither comparison, nor flip, nor arithmetic,
    nor scalar fill. The stage then raises with the missing operator's name and nothing else, where
    the reader needs the dtype and the one config line that fixes it."""
    reason = _stage_failure(NotImplementedError("\"index_put\" not implemented for 'UInt16'"))

    assert "index_put" in reason, "what torch said is kept"
    assert "'UInt16'" in reason and "TensorCast" in reason, "the dtype, and what to do about it"


@pytest.mark.parametrize(
    "error",
    [
        OSError("no space left on device"),
        NotImplementedError("this backend serves no region writes"),  # no dtype to name
    ],
)
def test_every_other_failure_is_reported_as_it_was_raised(error: Exception) -> None:
    reason = _stage_failure(error)
    assert reason == f"{type(error).__name__}: {error}"


def test_a_uint16_case_through_a_chain_torch_cannot_run_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end, on the dtype a microscopy store actually hands the sweep."""
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
    source = Dataset(tmp_path / "src", "mha")
    source.write(
        "CT", "CASE_000", (np.arange(14 * 10 * 8).reshape(1, 14, 10, 8) % 900).astype(np.uint16), _attributes()
    )
    manager = _manager(source, [Clip(min_value=10.0, max_value=90.0), Save(f"{tmp_path / 'out'}:h5")], tmp_path)

    with pytest.warns(UserWarning, match="TensorCast"), pytest.raises(TransformError, match="TensorCast"):
        CaseMaterializer(manager).materialize()


class _OutOfMemoryOnRegions(Transform):
    """Fails on a region the way a read under a tight cap fails, and succeeds on the whole volume."""

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def stream_region(self, name: str, tensor, context: RegionContext, cache_attribute: Attribute):
        raise MemoryError("Unable to allocate 30.5 MiB for an array with shape (78, 320, 320)")

    def __call__(self, name: str, tensor, cache_attribute: Attribute):
        return tensor


def test_a_sweep_that_runs_out_of_memory_does_not_answer_with_the_whole_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-volume fallback holds the WHOLE case, a larger allocation than the region that just
    failed and on the same device: it repairs a stage that cannot serve a region, never a shortage
    of memory. So an out-of-memory propagates where every other failure falls back."""
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
    source = Dataset(tmp_path / "src", "mha")
    source.write("CT", "CASE_000", np.ones((1, 14, 10, 8), np.float32), _attributes())
    chain = [_OutOfMemoryOnRegions(), Save(f"{tmp_path / 'out'}:h5")]

    with pytest.raises(MemoryError):
        CaseMaterializer(_manager(source, chain, tmp_path)).materialize()


def test_a_serial_sweep_charges_its_writes_where_a_pipelined_one_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figure after the bar is the write itself, on one thread or another."""
    SWEEP_CLOCK.reset()
    _sweep_into(tmp_path, "serial_write", monkeypatch, depth=0)

    assert SWEEP_CLOCK.spent("wait(write)") >= SWEEP_CLOCK.spent("write") > 0.0


def test_a_save_into_the_h5_file_the_sweep_reads_still_lands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The h5 backend holds a per-file lock for a stream's whole life, on the thread that opened it.
    A sweep reading that file on another thread would wait for a close its own read holds up."""
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
    monkeypatch.setattr("konfai.data.patching.sweep._sweep_pipeline_depth", lambda: 1)
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / "store", "h5")
    # Twenty blocks: more than the pipeline reads before its first write opens the stream.
    volume = (rng.random((1, 60, 10, 8)) * 100).astype(np.float32)
    source.write("CT", "CASE_000", volume, _attributes())
    manager = _manager(
        source, [Clip(min_value=10.0, max_value=90.0), Save(f"{tmp_path / 'store'}:h5", group="CACHE")], tmp_path
    )

    verdicts: list = []
    worker = threading.Thread(target=lambda: verdicts.append(CaseMaterializer(manager).materialize()), daemon=True)
    worker.start()
    worker.join(60)

    assert not worker.is_alive(), "the sweep never finished: its read waits on the lock its own stream holds"
    assert verdicts == [Verdict.STREAM]
    cached, _ = Dataset(tmp_path / "store", "h5").read_data("CACHE", "CASE_000")
    np.testing.assert_array_equal(cached, np.clip(volume, 10.0, 90.0))


# ---------------------------------------------------------------- what the height rule prices is what is held


def test_a_pipelined_sweep_holds_no_more_blocks_than_the_height_rule_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget bounds the blocks a sweep holds at once, so the height rule must count them
    all: the one in the chain (its pulled region and its landed block), the ``depth`` queued
    ahead, the one the reader holds while the queue is full, and the one being written behind.
    Counted here as blocks between their read and their write, with a writer slower than the
    reader so the queue fills and the reader blocks on it."""
    from konfai.data.patching import RegionWriter, _sweep_resident_regions

    depth = 1
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
    monkeypatch.setattr("konfai.data.patching.sweep._sweep_pipeline_depth", lambda: depth)
    monkeypatch.setattr("konfai.data.patching.budget._SWEEP_MAX_DEPTH", depth)  # the budget-bound case: no free blocks
    lock = threading.Lock()
    counts = {"handed": 0, "written": 0, "peak": 0}
    read = DatasetManager._read_streamed_region
    write = RegionWriter.write

    def counted_read(self, *args, **kwargs):
        block = read(self, *args, **kwargs)
        with lock:
            counts["handed"] += 1
            counts["peak"] = max(counts["peak"], counts["handed"] - counts["written"])
        return block

    def slow_write(self, *args, **kwargs):
        time.sleep(0.02)
        write(self, *args, **kwargs)
        with lock:
            counts["written"] += 1

    monkeypatch.setattr(DatasetManager, "_read_streamed_region", counted_read)
    monkeypatch.setattr(RegionWriter, "write", slow_write)
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / "src", "mha")
    source.write("CT", "CASE_000", (rng.random((1, 30, 10, 8)) * 100).astype(np.float32), _attributes())
    manager = _manager(source, [Clip(min_value=10.0, max_value=90.0), Save(f"{tmp_path / 'out'}:h5")], tmp_path)
    assert CaseMaterializer(manager).materialize() is Verdict.STREAM

    assert counts["peak"] >= depth + 2, "the pipeline did not overlap: nothing was measured"
    # The block in the chain is two slabs (pulled and landed); every other block in flight is one.
    assert counts["peak"] + 1 <= sum(_sweep_resident_regions(depth)), f"{counts['peak']} blocks in flight"


def test_the_sizing_prices_the_regions_and_blocks_each_depth_holds() -> None:
    """Pulled regions: the one in the chain, ``depth`` queued ahead, one in the reader's hand while
    the queue is full. Landed blocks: the one being landed and the one written behind. None past
    depth 0, and no more than that."""
    assert [_sweep_resident_regions(depth) for depth in range(4)] == [(1, 1), (3, 2), (4, 2), (5, 2)]


# ---------------------------------------------------------------- the publish is a write, and is waited for


@pytest.mark.parametrize("depth", [0, 1])
def test_the_publish_is_charged_to_the_write_and_to_the_wait_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    """Closing the streams publishes them, and an OME-Zarr store derives its pyramid there: time
    the sweep waited for while a phase named none of it was the residual the report calls
    ``other``, the part nothing has explained."""
    from konfai.data.patching import RegionWriter

    close = RegionWriter.close

    def slow_close(self) -> None:
        time.sleep(0.2)
        close(self)

    monkeypatch.setattr(RegionWriter, "close", slow_close)
    SWEEP_CLOCK.reset()
    _sweep_into(tmp_path, f"published_{depth}", monkeypatch, depth=depth)

    assert SWEEP_CLOCK.spent("write") >= 0.2
    assert SWEEP_CLOCK.spent("wait(write)") >= 0.2
    wall = SWEEP_CLOCK.spent("sweep")
    named = sum(SWEEP_CLOCK.spent(phase) for phase in ("chain", "fetch", "wait(read)", "wait(write)"))
    assert wall - named < 0.2, "the publish was charged to 'other'"
