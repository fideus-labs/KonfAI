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


"""Logs and TensorBoard: the console capture, the run log, the data-log strategies."""

import os
import re
import shutil
import socket
import subprocess  # nosec B404
import sys
import time
from enum import Enum
from pathlib import Path
from typing import TextIO, cast

import numpy as np
import torch

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
from konfai import (
    evaluations_directory,
    konfai_state,
    predictions_directory,
    statistics_directory,
    transforms_directory,
)
from konfai.utils.errors import ConfigError


class NullSummaryWriter:
    """Stands in for TensorBoard's ``SummaryWriter`` when the extra is absent: every ``add_*`` and
    ``close`` call is absorbed, so the workflow still produces its outputs; only the curves are lost.
    """

    def __getattr__(self, name: str):
        def _absorb(*args, **kwargs) -> None:
            return None

        return _absorb


def _log_signal_format(array: np.ndarray) -> dict[str, np.ndarray]:
    return {str(i): channel for i, channel in enumerate(array)}


def _log_image_format(array: np.ndarray) -> np.ndarray:
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)

    if len(array.shape) == 3 and array.shape[0] != 1:
        array = np.expand_dims(array, axis=0)
    if len(array.shape) == 4:
        array = array[:, array.shape[1] // 2]

    array = array.astype(float)
    b = -np.min(array)
    if (np.max(array) + b) > 0:
        return (array + b) / (np.max(array) + b)
    else:
        return 0 * array


def _log_images_format(array: np.ndarray) -> np.ndarray:
    result = []
    for n in range(array.shape[0]):
        result.append(_log_image_format(array[n]))
    result = np.stack(result, axis=0)
    return result


def _log_video_format(array: np.ndarray) -> np.ndarray:
    result_list = []
    for t in range(array.shape[1]):
        result_list.append(_log_images_format(array[:, t, ...]))
    result = np.stack(result_list, axis=1)

    nb_channel = result.shape[2]
    if nb_channel < 3:
        channel_split = [result[:, :, 0, ...] for i in range(3)]
    else:
        channel_split = np.split(result, 3, axis=0)
    array = np.zeros((result.shape[0], result.shape[1], 3, *list(result.shape[3:])))
    for i, channels in enumerate(channel_split):
        array[:, :, i] = np.mean(channels, axis=0)
    return array


class DataLog(Enum):
    """TensorBoard logging strategy selected in YAML runtime configs."""

    SIGNAL = "SIGNAL"
    IMAGE = "IMAGE"
    IMAGES = "IMAGES"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"

    @classmethod
    def parse(cls, entries: list[str] | None) -> dict[str, tuple["DataLog", int]]:
        """``{target: (strategy, count)}`` from ``"group_or_module/STRATEGY/N"`` entries; a ``:``-spelled
        module path is keyed by its dotted name."""
        parsed: dict[str, tuple[DataLog, int]] = {}
        for entry in entries or []:
            target, strategy, count = entry.split("/")
            parsed[target.replace(":", ".")] = (cls[strategy], int(count))
        return parsed

    def __call__(self, tb: SummaryWriter, name: str, layer: torch.Tensor, it: int):
        if self == DataLog.SIGNAL:
            return [
                tb.add_scalars(name, _log_signal_format(layer[b, :, 0]), layer.shape[0] * it + b)
                for b in range(layer.shape[0])
            ]
        elif self == DataLog.IMAGE:
            return tb.add_image(name, _log_image_format(layer[0]), it)
        elif self == DataLog.IMAGES:
            return tb.add_images(name, _log_images_format(layer), it)
        elif self == DataLog.VIDEO:
            return tb.add_video(name, _log_video_format(layer), it)
        elif self == DataLog.AUDIO:
            return tb.add_audio(name, _log_image_format(layer), it)
        else:
            raise ValueError(f"Unsupported DataLog type: {self}")


def _bar_key(line: str) -> str:
    """What identifies a progress bar across its redraws: its text up to the first digit."""
    return re.split(r"\d", line, maxsplit=1)[0]


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class MinimalLog:
    """Capture stdout/stderr while keeping a one-line rolling status buffer."""

    # Off a terminal, at most one redrawn progress-bar frame is mirrored per this many seconds.
    _MIRROR_REDRAW_EVERY = 2.0

    def __init__(self, rank: int = 0) -> None:
        self._stdout_bak = sys.stdout
        self._stderr_bak = sys.stderr
        self._buffered_line = ""
        self.verbose = os.environ.get("KONFAI_VERBOSE", "True") == "True"
        self.rank = rank
        try:
            self._mirror_is_tty = bool(self._stdout_bak.isatty())
        except (AttributeError, ValueError, OSError):
            self._mirror_is_tty = False
        self._mirror_last_redraw = 0.0
        # Folded frames withheld by the throttle, one slot per bar (keyed by the text before the first
        # digit): interleaved bars (train + validation) would overwrite a single slot, and one of the
        # two would end the run without its final state ever mirrored.
        self._mirror_pending: dict[str, str] = {}

    def __enter__(self):
        sys.stdout = cast(TextIO, self)
        sys.stderr = cast(TextIO, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # A run's last writes are often throttled frames: emitted here, so the sink ends on the bar's
        # final state. flush() cannot carry this: the bars flush after every frame, which would empty
        # the throttle each time.
        self._mirror_emit_pending()
        sys.stdout = self._stdout_bak
        sys.stderr = self._stderr_bak

    def write(self, msg: str):
        if not msg:
            return
        msg_clean = ANSI_ESCAPE_RE.sub("", msg)
        # A CRLF line ending is not a redraw: "warning\r\n" folds to the text after its last \r,
        # which is nothing, and the message would vanish from the mirror and the log file both.
        redraw = "\r" in msg_clean.replace("\r\n", "\n") or "[A" in msg
        if redraw:
            self._buffered_line = msg_clean.split("\r")[-1].strip()
        else:
            self._buffered_line = msg_clean.strip()

        if self.verbose and (self.rank == 0 or "KONFAI_CLUSTER" in os.environ):
            self._mirror(msg, redraw)

    def _mirror(self, msg: str, redraw: bool) -> None:
        # A terminal overwrites a redrawn bar; a file appends every frame, so a mirrored animation is
        # megabytes per run in an MCP job log or a slurm-*.out. Off a terminal the mirror sends the
        # folded line instead, throttled (a log tail still shows live progress), and a skipped frame
        # is kept pending so a bar's final state lands before whatever message follows it.
        if not self._mirror_is_tty:
            if redraw:
                now = time.monotonic()
                if now - self._mirror_last_redraw < self._MIRROR_REDRAW_EVERY:
                    self._mirror_pending[_bar_key(self._buffered_line)] = self._buffered_line
                    return
                self._mirror_last_redraw = now
                held = self._mirror_take_pending(exclude=_bar_key(self._buffered_line))
                msg = f"{held}{self._buffered_line}\n"
            else:
                msg = f"{self._mirror_take_pending()}{msg}"
        # Best-effort: if the mirror's reader is gone (an interactive launcher exited, a server
        # restarted), the pipe is broken: keep running and keep writing to the log file rather than
        # crashing the job.
        try:
            self._stdout_bak.write(msg)
            self._stdout_bak.flush()
        except (BrokenPipeError, ValueError):
            pass

    def flush(self):
        try:
            self._stdout_bak.flush()
        except (BrokenPipeError, ValueError):
            pass

    def _mirror_take_pending(self, exclude: str | None = None) -> str:
        """The withheld frames as mirror-ready lines, cleared; ``exclude`` skips the bar being emitted."""
        if not self._mirror_pending:
            return ""
        lines = [line for key, line in self._mirror_pending.items() if key != exclude]
        self._mirror_pending.clear()
        return "".join(f"{line}\n" for line in lines)

    def _mirror_emit_pending(self) -> None:
        held = self._mirror_take_pending()
        if held:
            try:
                self._stdout_bak.write(held)
                self._stdout_bak.flush()
            except (BrokenPipeError, ValueError):
                pass

    def fileno(self):
        if sys.__stdout__ is None:
            raise RuntimeError("sys.__stdout__ is None, cannot get fileno")
        return sys.__stdout__.fileno()


class Log(MinimalLog):
    """Mirror console output to a rank-specific log file."""

    def __init__(self, name: str, rank: int) -> None:
        super().__init__(rank)
        if konfai_state() == "PREDICTION":
            path = predictions_directory()
        elif konfai_state() == "EVALUATION":
            path = evaluations_directory()
        elif konfai_state() == "TRANSFORM":
            path = transforms_directory()
        else:
            path = statistics_directory()
        # ``name`` is train_name from the config; an absolute path or '..' segments would place the logs
        # outside the run's output directory. A nested run name stays inside and is fine.
        self.log_path = path / name
        if not self.log_path.resolve().is_relative_to(path.resolve()):
            raise ConfigError(
                f"train_name '{name}' resolves outside the output directory.",
                "Use a plain run name, without an absolute path or '..' segments.",
            )
        self.log_path.mkdir(parents=True, exist_ok=True)
        # Append, never truncate: this file is opened BEFORE the overwrite prompt runs, so a "w" mode
        # destroyed the previous run's log even when the user declined the overwrite.
        self.file = open(self.log_path / f"log_{rank}.txt", "a", buffering=1)
        self._last_logged: str | None = None

    def __enter__(self):
        super().__enter__()
        self.file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.file.__exit__(exc_type, exc_val, exc_tb)

    def write(self, msg: str):
        super().write(msg)
        # Consecutive identical lines are one fact said twice: a progress bar arrives as several
        # write() calls per frame and a case line rides beside its own counter frame, which would
        # multiply the file by ~4x against the console. Only CONSECUTIVE repeats fold: a fact that
        # genuinely recurs later still lands.
        if self._buffered_line and self._buffered_line != self._last_logged:
            self._last_logged = self._buffered_line
            self.file.write(self._buffered_line + "\n")
            self.file.flush()

    def flush(self):
        super().flush()
        self.file.flush()


def record(message: str) -> Path | None:
    """Keep ``message`` in the run's log without printing it, and answer where it went.

    For detail that belongs beside the run but not on a console every other workflow keeps to a
    progress bar (the TRANSFORM plan). Answers None when no ``Log`` is installed, which is every
    context that has no run directory to keep it in.
    """
    sink = sys.stdout
    if not isinstance(sink, Log):
        return None
    sink.file.write(message if message.endswith("\n") else message + "\n")
    sink.file.flush()
    return Path(sink.file.name)


class TensorBoard:
    """Lifecycle helper that optionally starts a TensorBoard side process."""

    def __init__(self, name: str) -> None:
        self.process: subprocess.Popen | None = None
        self.name = name

    def __enter__(self):
        if "KONFAI_TENSORBOARD_PORT" in os.environ:
            tensorboard_exe = shutil.which("tensorboard")
            if tensorboard_exe is None:
                raise RuntimeError("TensorBoard executable not found in PATH.")

            logdir = predictions_directory() if konfai_state() == "PREDICTION" else statistics_directory() / self.name

            port = os.environ.get("KONFAI_TENSORBOARD_PORT")
            if not port or not port.isdigit():
                raise ValueError("Invalid or missing KONFAI_TENSORBOARD_PORT.")

            command = [
                tensorboard_exe,
                "--logdir",
                str(logdir),
                "--port",
                port,
                "--bind_all",
            ]
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            print(f"[KonfAI] Tensorboard : http://{ip}:{os.environ['KONFAI_TENSORBOARD_PORT']}/")
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
