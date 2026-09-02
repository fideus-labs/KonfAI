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

"""One-command proof of the bounded-memory claim.

Synthesizes a volume of ``--gib`` GiB, runs a real TRANSFORM chain over it under a declared
``--budget`` GiB, and reports the whole-process-tree peak RSS beside both figures. The claim this
evidences: peak RAM tracks the BUDGET, not the volume.

    python benchmarks/bench_streaming.py --gib 16 --budget 1

Needs the ``imaging`` extra (SimpleITK + h5py) and free disk for the synthetic volume.
"""

import argparse
import json
import platform
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil


def _tree_rss(process: psutil.Process) -> int:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


class PeakSampler(threading.Thread):
    """Whole-tree peak RSS, sampled at 50 ms: children (DataLoader workers, spawn ranks) count."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._process = psutil.Process()
        # Not `_stop`: threading.Thread.join() calls its own internal `_stop()` method.
        self._stop_event = threading.Event()
        self.peak = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, _tree_rss(self._process))
            time.sleep(0.05)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak


def synthesize(root: Path, gib: float) -> tuple[Path, list[int]]:
    """A synthetic volume of ~``gib`` GiB in an h5 store, written slab by slab so the synthesis
    itself never holds the volume (the bench must not be the thing that spends the RAM)."""
    import h5py

    voxels = int(gib * 2**30 / 4)  # float32
    side = max(64, int(round((voxels / 4) ** (1 / 3))))  # anisotropic: 4x taller than wide
    shape = [4 * side, side, side]
    store = root / "Dataset.h5"
    rng = np.random.default_rng(0)
    with h5py.File(store, "w") as file:
        dataset = file.create_dataset("CT/CASE_000", shape=(1, *shape), dtype=np.float32, chunks=(1, 64, side, side))
        dataset.attrs["Origin"] = "[0. 0. 0.]"
        dataset.attrs["Spacing"] = "[1. 1. 1.]"
        dataset.attrs["Direction"] = "[1. 0. 0. 0. 1. 0. 0. 0. 1.]"
        for start in range(0, shape[0], 64):
            stop = min(start + 64, shape[0])
            dataset[0, start:stop] = rng.normal(0.0, 100.0, size=(stop - start, *shape[1:])).astype(np.float32)
    return store, shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=float, default=4.0, help="synthetic volume size in GiB (default 4)")
    parser.add_argument("--budget", type=float, default=1.0, help="declared memory_budget in GiB (default 1)")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = parser.parse_args()

    import konfai

    scratch = Path(tempfile.mkdtemp(prefix="konfai_bench_"))
    print(f"[bench] scratch: {scratch}")
    sampler = None
    try:
        print(f"[bench] synthesizing ~{args.gib:g} GiB volume ...", flush=True)
        store, shape = synthesize(scratch, args.gib)

        from konfai.data.transform import Normalize, Write

        sampler = PeakSampler()
        sampler.start()
        start = time.perf_counter()
        result = konfai.transform(
            "BENCH",
            f"{store}:h5",
            {"CT": {"CT": [Normalize(min_value=-1, max_value=1), Write(dataset=f"{scratch / 'Out'}:h5")]}},
            memory_budget=f"{args.budget}gib",
            transforms_dir=scratch / "Transforms",
            quiet=True,
        )
        elapsed = time.perf_counter() - start
        peak = sampler.stop()

        del result
        report = {
            "volume_gib": round(args.gib, 2),
            "declared_budget_gib": round(args.budget, 2),
            "peak_tree_rss_gib": round(peak / 2**30, 2),
            "wall_s": round(elapsed, 1),
            "shape_zyx": shape,
            "versions": {
                "konfai": getattr(konfai, "__version__", "dev"),
                "numpy": np.__version__,
                "python": platform.python_version(),
            },
            "host": platform.node(),
        }
        print(json.dumps(report, indent=2))
        print(
            f"| {args.gib:g} GiB volume | budget {args.budget:g} GiB | peak {peak / 2**30:.2f} GiB | {elapsed:.1f} s |"
        )
    finally:
        if sampler is not None and sampler.is_alive():
            sampler.stop()
        # A failed large-volume run must not leave its synthetic input and output on disk.
        if not args.keep:
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
