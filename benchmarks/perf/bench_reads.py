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

"""What a region read costs through ``Dataset``, per layout an HDF5 store comes in.

    python benchmarks/perf/bench_reads.py [--repeats 3] [--force] [--quick]

One volume, written five ways into one h5 file as a third-party tool would: contiguous (what
``Dataset.write`` stores), chunked with no filter in small and in large chunks, chunked with gzip,
chunked with byte shuffle. Each layout serves, through ``Dataset.read_data_slice``, the shuffled
one-row windows of a patch loader and one pass in whole chunks, as a sweep or a statistics scan reads.

Every read must return h5py's own bytes. The shuffled reads of a chunked layout are also timed
against the contiguous layout's. HDF5's chunk cache lives in the open dataset, so a dataset reopened
on every read decodes its chunk again each time, and a large unfiltered chunk loaded through it is
read whole for one row.

The same volume as an uncompressed MetaImage, NIfTI and NRRD is then read whole, ``Dataset.read_data``
against the one window covering it: each comes off the file's raw block. The NRRD's shuffled one-row
reads are timed against the MetaImage's: ITK reads a NRRD whole for any region of it, where the raw
block serves the row.

The volume is last written whole, ``Dataset.write``, with one channel and with four: a channel of
the four is compared with the one-channel write. SimpleITK takes a vector volume pixel by pixel,
which the region writer handed the volume whole does not. Times and ratios remain measurements,
not machine-independent facts: compare the times to the same machine's baseline with compare.py.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import fact
from harness import fingerprint, machine_gate, write_result
from konfai.utils.dataset import Attribute, Dataset

SHAPE = (1, 128, 256, 256)
LAYOUTS: dict[str, dict[str, object]] = {
    "contiguous": {},
    "small_chunks": {"chunks": (1, 4, 64, 64)},
    "large_chunks": {"chunks": SHAPE},
    "gzip": {"chunks": (1, 16, 256, 256), "compression": "gzip", "compression_opts": 1},
    "shuffle": {"chunks": (1, 16, 256, 256), "shuffle": True},
}
_CHANNELS = 4


def _row(z: int) -> tuple[slice, ...]:
    return (slice(None), slice(z, z + 1), slice(None), slice(None))


def _median_ms(read: Callable[[], object], repeats: int) -> float:
    read()
    walls = []
    for _ in range(repeats):
        start = time.perf_counter()
        read()
        walls.append((time.perf_counter() - start) * 1e3)
    return round(statistics.median(walls), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()
    repeats = 1 if args.quick else args.repeats
    random.seed(0)
    rows = [random.randrange(SHAPE[1]) for _ in range(64 if args.quick else 256)]
    volume = np.random.default_rng(0).integers(-1000, 3000, size=SHAPE, dtype=np.int16)

    metrics: dict[str, float] = {}
    facts = []
    with tempfile.TemporaryDirectory(prefix="konfai_perf_reads_") as scratch:
        store = Path(scratch) / "Layouts.h5"
        with h5py.File(store, "w") as file:
            for name, options in LAYOUTS.items():
                file.create_dataset(f"CT/{name}", data=volume, **options)
        dataset = Dataset(str(store), "h5")
        for name, options in LAYOUTS.items():
            same = all(np.array_equal(dataset.read_data_slice("CT", name, _row(z))[0], volume[_row(z)]) for z in rows)
            same = same and np.array_equal(dataset.read_data("CT", name)[0], volume)
            facts.append(fact(f"{name}_same_bytes_as_h5py", same, 1))
            step = int(options.get("chunks", SHAPE)[1])  # type: ignore[index]
            reads, passes = [], []
            for _ in range(repeats):
                start = time.perf_counter()
                for z in rows:
                    dataset.read_data_slice("CT", name, _row(z))
                reads.append((time.perf_counter() - start) * 1e3)
                start = time.perf_counter()
                for z in range(0, SHAPE[1], step):
                    dataset.read_data_slice("CT", name, (slice(None), slice(z, z + step), slice(None), slice(None)))
                passes.append((time.perf_counter() - start) * 1e3)
            metrics[f"{name}_reads_ms"] = round(statistics.median(reads), 2)
            metrics[f"{name}_pass_ms"] = round(statistics.median(passes), 2)
            print(
                f"[perf] {name}: {len(rows)} one-row reads {metrics[f'{name}_reads_ms']} ms, one pass in whole chunks {metrics[f'{name}_pass_ms']} ms"
            )
        geometry = Attribute()
        geometry["Origin"], geometry["Spacing"], geometry["Direction"] = np.zeros(3), np.ones(3), np.eye(3).reshape(-1)
        whole_window = tuple(slice(0, extent) for extent in SHAPE)
        for fmt in ("mha", "nii", "nrrd"):
            Dataset(f"{scratch}/{fmt}", fmt).write("CT", "P000", volume, Attribute(geometry))
            raw = Dataset(f"{scratch}/{fmt}", fmt)
            facts.append(fact(f"{fmt}_whole_same_voxels", np.array_equal(raw.read_data("CT", "P000")[0], volume), 1))
            # More repeats than the layouts: one read is a few milliseconds.
            metrics[f"{fmt}_whole_ms"] = _median_ms(lambda raw=raw: raw.read_data("CT", "P000"), 5 * repeats)
            metrics[f"{fmt}_window_ms"] = _median_ms(
                lambda raw=raw: raw.read_data_slice("CT", "P000", whole_window), 5 * repeats
            )
            metrics[f"{fmt}_reads_ms"] = _median_ms(
                lambda raw=raw: [raw.read_data_slice("CT", "P000", _row(z)) for z in rows], repeats
            )
            ratio = round(metrics[f"{fmt}_whole_ms"] / metrics[f"{fmt}_window_ms"], 3)
            metrics[f"{fmt}_whole_over_window"] = ratio
            print(
                f"[perf] {fmt}: whole read {metrics[f'{fmt}_whole_ms']} ms, the window covering it {metrics[f'{fmt}_window_ms']} ms,"
                f" {len(rows)} one-row reads {metrics[f'{fmt}_reads_ms']} ms"
            )
        ratio = round(metrics["nrrd_reads_ms"] / metrics["mha_reads_ms"], 3)
        metrics["nrrd_reads_over_mha"] = ratio
        vector = np.concatenate([volume + channel for channel in range(_CHANNELS)])
        for fmt in ("mha", "nrrd"):
            written = Dataset(f"{scratch}/written_{fmt}", fmt)
            for name, data in (("scalar", volume), ("vector", vector)):
                metrics[f"{fmt}_{name}_write_ms"] = _median_ms(
                    lambda written=written, name=name, data=data: written.write("CT", name, data, Attribute(geometry)),
                    5 * repeats,
                )
            same = np.array_equal(written.read_data("CT", "vector")[0], vector)
            facts.append(fact(f"{fmt}_vector_write_same_voxels", same, 1))
            ratio = round(metrics[f"{fmt}_vector_write_ms"] / (_CHANNELS * metrics[f"{fmt}_scalar_write_ms"]), 3)
            metrics[f"{fmt}_vector_write_over_scalar"] = ratio
            print(
                f"[perf] {fmt}: whole write {metrics[f'{fmt}_scalar_write_ms']} ms, with {_CHANNELS} channels {metrics[f'{fmt}_vector_write_ms']} ms"
            )
    for name in LAYOUTS:
        if name != "contiguous":
            ratio = round(metrics[f"{name}_reads_ms"] / metrics["contiguous_reads_ms"], 3)
            metrics[f"{name}_reads_over_contiguous"] = ratio

    result = {
        "gate_warnings": gate.warnings,
        "quick": args.quick,
        "volume_shape": list(SHAPE),
        "reads": len(rows),
        "repeats": repeats,
        "metrics": metrics,
        "facts": facts,
        "headline": f"{len(rows)} shuffled one-row reads: "
        + ", ".join(f"{name} {metrics[f'{name}_reads_ms']:g} ms" for name in LAYOUTS),
    }
    path = write_result("reads", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()
