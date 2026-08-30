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

"""Where a run's wall clock went, phase by phase: one accounting vocabulary for every workflow."""

import contextlib
import time
from collections.abc import Iterable, Iterator
from typing import Any


class SweepClock:
    """Wall clock spent per named phase, reported as one line that closes on the total.

    A phase is accumulated by exactly one thread (a sweep's read by its producer, its write by the
    writer, the rest by the loop's own), so the additions need no lock; the report is read once
    every helper thread has been joined. Two ``perf_counter`` calls per phase per block.
    """

    _END = object()

    def __init__(self) -> None:
        self._spent: dict[str, float] = {}

    def reset(self) -> None:
        self._spent = {}

    def spent(self, name: str) -> float:
        return self._spent.get(name, 0.0)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._spent[name] = self._spent.get(name, 0.0) + time.perf_counter() - start

    def waiting(self, name: str, blocks: Iterable[Any]) -> Iterator[Any]:
        """``blocks``, charging to ``name`` the time spent waiting for each one."""
        blocks = iter(blocks)
        while True:
            with self.phase(name):
                block = next(blocks, SweepClock._END)
            if block is SweepClock._END:
                return
            yield block

    def report(self, min_seconds: float = 1.0) -> str | None:
        """One line accounting for the sweeps' wall clock, or ``None`` below ``min_seconds``.

        The sum before the bar is the sweep's own thread and closes exactly: what its phases do
        not name is ``other``. After the bar are the read and the write themselves, which run
        beside that thread when the sweep pipelines and inside its ``wait`` when it does not.
        """
        wall = self.spent("sweep")
        if wall < min_seconds:
            return None
        named = {phase: self.spent(phase) for phase in ("chain", "fetch", "wait(read)", "wait(write)")}
        parts = " + ".join(f"{phase} {value:.1f}" for phase, value in named.items())
        return (
            f"[KonfAI] sweep {wall:.1f} s = {parts} + other {wall - sum(named.values()):.1f}"
            f" | stages read {self.spent('read'):.1f} s, write {self.spent('write'):.1f} s"
        )


class StartupClock:
    """Where a run's time went before its ranks start, phase by phase.

    Built at the launcher's entry, carried to the ranks on the workflow object, reported once by
    rank 0 as it starts. The build's phases (``cases``: the cohort listed, ``grids``: the managers
    built, ``model``: the network built) and the setup's (``checkpoint``: the weights loaded) are
    taken out of ``build`` and ``setup``, which then read as their remainders; ``launch`` is spawn
    to the rank's start. The two stamps are ``time.time``: a rank reads them on another node under
    a cluster launch, where a process counter means nothing.
    """

    def __init__(self) -> None:
        self._phases = SweepClock()
        self.started = time.time()
        self.launched: float | None = None

    def phase(self, name: str) -> contextlib.AbstractContextManager[None]:
        return self._phases.phase(name)

    def spent(self, name: str) -> float:
        return self._phases.spent(name)

    def launch(self) -> None:
        self.launched = time.time()

    def report(self, min_seconds: float = 1.0) -> str | None:
        """One line accounting for the wall clock since the launcher's entry, or ``None`` below
        ``min_seconds``; ``other`` is what no phase names."""
        now = time.time()
        wall = now - self.started
        if wall < min_seconds:
            return None
        nested = {name: self.spent(name) for name in ("cases", "grids", "model")}
        named = {
            "build": max(0.0, self.spent("build") - sum(nested.values())),
            **nested,
            "checkpoint": self.spent("checkpoint"),
            "setup": max(0.0, self.spent("setup") - self.spent("checkpoint")),
            "launch": 0.0 if self.launched is None else now - self.launched,
        }
        parts = " + ".join(f"{name} {value:.1f}" for name, value in named.items())
        return f"[KonfAI] startup {wall:.1f} s = {parts} + other {wall - sum(named.values()):.1f}"


_startup_clock: StartupClock | None = None


def startup_clock() -> StartupClock:
    """This process's startup clock, built on first use."""
    global _startup_clock
    if _startup_clock is None:
        _startup_clock = StartupClock()
    return _startup_clock


def restart_startup_clock() -> StartupClock:
    """A fresh startup clock: the Python API runs several workflows in one process, each its own."""
    global _startup_clock
    _startup_clock = StartupClock()
    return _startup_clock
