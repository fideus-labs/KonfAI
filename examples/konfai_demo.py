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

"""Helpers shared by the example notebooks, so each one shows the task and not the plumbing.

Nothing here is part of KonfAI's API, it is notebook scaffolding: install the missing packages,
run a CLI command with readable output, and draw a row of slices.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

_WORKDIR = Path.cwd()

# Emitted by huggingface_hub's tqdm when ipywidgets is absent; harmless, and alarming in red.
warnings.filterwarnings("ignore", message="IProgress not found.*")


def setup(repo_dir: Path, example: str, *packages: str | tuple[str, str]) -> tuple[Path, Path, list[str]]:
    """Install what is missing and return the example directory, its dataset directory, and the device flags.

    `packages` are pip requirements, installed only when already absent. The import name is derived
    from the requirement (`konfai[imaging]` -> `konfai`, `huggingface-hub` -> `huggingface_hub`); pass a
    `(import_name, requirement)` pair when it cannot be, as for a local path.
    """
    global _WORKDIR

    wanted = [item if isinstance(item, tuple) else (_import_name(item), item) for item in packages]
    missing = [package for module, package in wanted if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing", ", ".join(missing), "...", flush=True)
        report = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing], capture_output=True, text=True
        )
        if report.returncode:
            raise RuntimeError(report.stdout + report.stderr)

    import torch

    _WORKDIR = repo_dir / "examples" / example
    # The first VISIBLE device, not device 0: `--gpu` is validated against cuda_visible_devices(),
    # which reports the raw CUDA_VISIBLE_DEVICES values, so on a workstation exporting
    # CUDA_VISIBLE_DEVICES=1 argparse rejects `--gpu 0` outright.
    if torch.cuda.is_available():
        from konfai import cuda_visible_devices

        device = ["--gpu", str(cuda_visible_devices()[0])]
    else:
        device = ["--cpu", "1"]
    print("KonfAI :", repo_dir)
    print("Example:", _WORKDIR)
    print("Device :", " ".join(device))
    return _WORKDIR, _WORKDIR / "Dataset", device


def _import_name(requirement: str) -> str:
    return re.split(r"[\[<>=;]", requirement, maxsplit=1)[0].strip().replace("-", "_")


def run(*command: str) -> None:
    """Run a command in the example directory, echoing its progress every so often.

    KonfAI reports progress with a carriage-returned bar; printing it verbatim would bury the notebook
    under thousands of lines, so only a snapshot is shown, at a rate that backs off from two seconds
    to thirty: frequent enough to see a short command move, sparse enough that a six-minute training
    stays a dozen lines. On failure the tail is raised instead.
    """
    print("$", " ".join(command), flush=True)
    started = time.time()
    process = subprocess.Popen(
        command, cwd=_WORKDIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    tail: list[str] = []
    pending, next_print, interval = "", 0.0, 2.0
    for chunk in iter(lambda: process.stdout.read(256), ""):  # type: ignore[union-attr]
        *complete, pending = re.split(r"[\r\n]", pending + chunk)
        tail = (tail + [line for line in complete if line.strip()])[-40:]
        if tail and time.time() - started >= next_print:
            print("   ", tail[-1][:140], flush=True)
            next_print = time.time() - started + interval
            interval = min(interval * 1.6, 30.0)
    if pending.strip():
        # The final line carries no terminator, and on a crash that line is the exception.
        tail = [*tail, pending][-40:]
    if process.wait():
        raise RuntimeError("\n".join(tail[-25:]))
    print(f"    done in {time.time() - started:.0f} s\n", flush=True)


def latest_checkpoint(train_name: str) -> str:
    """Path of the newest checkpoint of a run, relative to the example directory."""
    checkpoints = sorted((_WORKDIR / "Checkpoints" / train_name).glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under Checkpoints/{train_name}: did training run?")
    return str(checkpoints[-1].relative_to(_WORKDIR))


def read(path: Path) -> Any:
    """Read a volume as a (Z, Y, X) numpy array."""
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def show(panels: list[tuple], label_max: int | None = None) -> None:
    """Draw a row of slices.

    A panel is `(title, image, cmap)`, optionally followed by a label map to overlay and/or a
    `(vmin, vmax)` tuple. Give two panels the same `(vmin, vmax)` to make them comparable by eye: matplotlib otherwise rescales each one to its own range, which hides the difference.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), constrained_layout=True)
    for axis, (title, image, cmap, *extra) in zip(np.atleast_1d(axes), panels, strict=True):
        vmin, vmax = next((limits for limits in extra if isinstance(limits, tuple)), (None, None))
        axis.imshow(image, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        for labels in (item for item in extra if not isinstance(item, tuple)):
            axis.imshow(
                np.ma.masked_where(labels == 0, labels),
                cmap="nipy_spectral",
                interpolation="nearest",
                alpha=0.65,
                vmin=0,
                vmax=label_max or labels.max(),
            )
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    plt.show()
