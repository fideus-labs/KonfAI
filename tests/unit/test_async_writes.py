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

"""The background writer: disk writes overlap the prediction loop, byte-identically.

Writes are submitted to one worker per output dataset — in order, bounded queue, failures kept and
re-raised — but only when the destination serves disjoint files per entry
(``Dataset.concurrent_write_safe``): a single-store backend (h5, zarr) stays inline, so no store is
ever written from two threads. Pure ``threading``/``queue``, no fork and no signals, so the behaviour
is the same on Linux, macOS and Windows."""

import pytest
import torch
from konfai.data.augmentation import Flip
from konfai.predictor import _AsyncWriter
from konfai.utils.dataset import Dataset
from konfai.utils.errors import PredictorError
from test_streamed_tta import _drive_tta


def test_async_writer_runs_in_order_and_surfaces_failures() -> None:
    writer = _AsyncWriter()
    done: list[int] = []
    writer.submit(lambda: done.append(1))
    writer.submit(lambda: (_ for _ in ()).throw(PredictorError("destination died")))
    # Operations submitted after a failure drain unexecuted; the failure surfaces at the LATEST at
    # close(), possibly earlier at this submit if the worker already recorded it -- accept it wherever
    # it lands, so a run can never end with a write silently missing.
    with pytest.raises(PredictorError, match="destination died"):
        writer.submit(lambda: done.append(2))
        writer.close()
    assert done == [1]


def test_concurrent_write_safety_is_declared_per_backend(tmp_path) -> None:
    assert Dataset(f"{tmp_path}/a", "mha").concurrent_write_safe()
    assert not Dataset(f"{tmp_path}/a.h5", "h5").concurrent_write_safe()
    assert not Dataset(f"{tmp_path}/a", "omezarr").concurrent_write_safe()


def test_async_streamed_writes_match_the_inline_reference(tmp_path, monkeypatch) -> None:
    # A per-file destination goes through the background writer (forced here — the automatic gate
    # also requires a GPU-placed output); the kill-switch runs the same store inline. Same
    # operations, same order — the files must match bit for bit.
    used: list[int] = []
    submit = _AsyncWriter.submit
    monkeypatch.setattr(_AsyncWriter, "submit", lambda self, op: (used.append(1), submit(self, op))[1])
    monkeypatch.setenv("KONFAI_ASYNC_WRITES", "1")
    asynchronous, whole_volume = _drive_tta(
        tmp_path / "async", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="mha"
    )
    assert not whole_volume and used, "the mha destination should have taken the background writer"
    monkeypatch.setenv("KONFAI_ASYNC_WRITES", "0")
    inline, _ = _drive_tta(
        tmp_path / "inline", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="mha"
    )
    assert torch.equal(asynchronous, inline)


def test_async_gate_stays_inline_for_single_stores_and_cpu_outputs(tmp_path, monkeypatch) -> None:
    used: list[int] = []
    submit = _AsyncWriter.submit
    monkeypatch.setattr(_AsyncWriter, "submit", lambda self, op: (used.append(1), submit(self, op))[1])
    # Even forced, a single-store destination never crosses threads.
    monkeypatch.setenv("KONFAI_ASYNC_WRITES", "1")
    _drive_tta(tmp_path / "h5", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="h5")
    assert not used, "an h5 store must never be written from the background thread"
    # Automatic mode on a CPU-placed output stays inline: the blend already saturates the memory
    # bandwidth the writer would consume.
    monkeypatch.delenv("KONFAI_ASYNC_WRITES", raising=False)
    _drive_tta(tmp_path / "cpu", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, file_format="mha")
    assert not used, "a CPU-only output must stay inline in automatic mode"
