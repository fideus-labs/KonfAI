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
from konfai.data import patching as patching_module
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import (
    SWEEP_CLOCK,
    DatasetManager,
    DatasetPatch,
    _ReadAhead,
    _stage_failure,
    _sweep_pipeline_depth,
    _WriteBehind,
)
from konfai.data.transform import Clip, Save
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError

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
    attribute = Attribute()
    attribute["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attribute["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


def _manager(source: Dataset, transforms: list, out: Path) -> DatasetManager:
    del out
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch([4, 5, 4]),
        transforms=transforms,
        data_augmentations_list=[],
    )


def _sweep_into(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch, depth: int) -> np.ndarray:
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", 3)
    monkeypatch.setattr(patching_module, "_sweep_pipeline_depth", lambda: depth)
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
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", 3)
    source = Dataset(tmp_path / "src", "mha")
    source.write(
        "CT", "CASE_000", (np.arange(14 * 10 * 8).reshape(1, 14, 10, 8) % 900).astype(np.uint16), _attributes()
    )
    manager = _manager(source, [Clip(min_value=10.0, max_value=90.0), Save(f"{tmp_path / 'out'}:h5")], tmp_path)

    with pytest.warns(UserWarning, match="TensorCast"), pytest.raises(TransformError, match="TensorCast"):
        CaseMaterializer(manager).materialize()


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
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", 3)
    monkeypatch.setattr(patching_module, "_sweep_pipeline_depth", lambda: 1)
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
