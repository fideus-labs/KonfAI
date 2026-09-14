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

"""What a training epoch costs when the patches are read from the store instead of held in RAM.

    python benchmarks/perf/bench_train_stream.py [--cases 6] [--shape 64,256,256] [--epochs 3] [--force]

One dataset, written once per storage backend, and one model, so the only thing that moves between
runs is where a patch comes from. Each cell is a fresh ``konfai TRAIN``:

- **loaded**: a memory budget above the dataset's size, so the source holds every case in RAM.
- **streamed**: a budget below it, so each patch is read from the store as a region.

The trainer's own clocks separate the two halves of an epoch: ``wait(data)`` is the loop waiting on
the loader, the rest is compute. A backend that serves regions badly shows up as ``wait(data)``
growing while the compute stays put.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    CliRun,
    copy_example,
    fingerprint,
    konfai_executable,
    machine_gate,
    run_cli,
    write_result,
)

#: A compressed MetaImage cannot serve a region, so a patch read off one decodes the whole volume.
#: The pair is here on purpose: it is the axis the framework's own guard
#: (``_patch_read_decodes_the_volume``) turns on.
COMPRESSED = {"mha-gz"}


def synthesize(root: Path, cases: int, shape: tuple[int, int, int], *, compress: bool = False) -> None:
    """A CT and a label map per case, as MetaImage: the source every other backend is converted from."""
    import SimpleITK as sitk

    rng = np.random.default_rng(0)
    for index in range(cases):
        case = root / f"case_{index:03d}"
        case.mkdir(parents=True, exist_ok=True)
        volume = rng.normal(0.0, 300.0, shape).astype(np.float32) - 200.0
        labels = np.zeros(shape, dtype=np.uint8)
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
        centre = [extent / 2 for extent in shape]
        radius = min(shape) / 3
        inside = (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2 < radius**2
        labels[inside] = 1
        for name, array, dtype in (("CT", volume, sitk.sitkFloat32), ("SEG", labels, sitk.sitkUInt8)):
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(sitk.Cast(image, dtype), str(case / f"{name}.mha"), compress)


def convert(source: Path, target: Path, fmt: str, scratch: Path) -> None:
    """The same cases under another backend, written by TRANSFORM so the layout is the framework's."""
    script = f"""
import konfai
from konfai.data.transform import Write
konfai.transform(
    name="convert_{fmt.replace(".", "_")}",
    datasets="{source}:a:mha",
    chains={{
        "CT": {{"CT": [Write(dataset="{target}:{fmt}")]}},
        "SEG": {{"SEG": [Write(dataset="{target}:{fmt}")]}},
    }},
    quiet=True,
    overwrite=True,
    transforms_dir="{scratch / "Transforms"}",
)
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=scratch, capture_output=True, text=True)


def write_config(
    example: Path,
    dataset: str,
    budget: str,
    epochs: int,
    patch: tuple[int, int, int],
    workers: int,
    chain: str = "plain",
    extend_slice: int = 0,
    shuffle_window: int | None = None,
) -> None:
    """Point the example's config at one dataset spec and one memory budget; nothing else moves.

    ``num_workers`` is pinned in both regimes: left to itself the framework gives a cached source none
    and a streaming one four, which would move the read path and the worker count in the same step."""
    config = example / "Config.yml"
    text = config.read_text()
    text = text.replace("    - ./Dataset:a:mha", f"    - {dataset}")
    text = text.replace("    memory_budget: None", f"    memory_budget: {budget}", 1)
    text = text.replace("    num_workers: None", f"    num_workers: {workers}", 1)
    if shuffle_window:
        text = text.replace("    shuffle_window: None", f"    shuffle_window: {shuffle_window}", 1)
    if extend_slice:
        # 2.5D: the patch stacks `extend_slice + 1` adjacent slices as channels, so the network's first
        # level takes that many.
        text = text.replace("      extend_slice: 0", f"      extend_slice: {extend_slice}", 1)
        text = text.replace(
            "        channels:  # Feature channels per level; channels[0] = 1 = a single-channel CT input\n        - 1\n",
            f"        channels:\n        - {extend_slice + 1}\n",
            1,
        )
    if chain.startswith("resample") or chain == "standardize-first":
        # A regrid stage ahead of the per-case statistics: cached, it runs once per case; streamed, it
        # runs on the window every patch read pulls.
        regrid = (
            "            transforms:\n"
            "              Resample:\n"
            "                spacing:\n"
            "                - 3\n"
            "                - 3\n"
            "                - 3\n"
            "                shape: None\n"
            "                reference: None\n"
            "                inverse: false\n"
        )
        # Both groups, or the image and its labels land on different grids and the run refuses.
        if chain == "standardize-first":
            # The regrid goes AFTER the statistic, so the statistic's input is the stored volume and
            # the chain streams while still scanning each case.
            text = text.replace(
                "                mask: None\n                inverse: false\n            patch_transforms: None\n            is_input: true",
                "                mask: None\n                inverse: false\n"
                + regrid.split("transforms:\n", 1)[1]
                + "            patch_transforms: None\n            is_input: true",
                1,
            )
        else:
            text = text.replace(
                "            transforms:\n              Standardize:", regrid + "              Standardize:", 1
            )
        text = text.replace(
            "            transforms:\n              TensorCast:", regrid + "              TensorCast:", 1
        )
    if chain == "resample-fixed":
        # The one change that makes the same chain streamable: a Standardize told its two numbers is
        # pointwise, where one that must find them needs a volume nothing upstream has touched.
        text = text.replace(
            "                mean: None\n                std: None",
            "                mean:\n                - 0.0\n                std:\n                - 1.0",
            1,
        )
    text = text.replace("  epochs: 5", f"  epochs: {epochs}")
    text = text.replace(
        "      patch_size:  # [slice, height, width]; one 256x256 tile of a single slice\n      - 1\n      - 256\n      - 256",
        f"      patch_size:\n      - {patch[0]}\n      - {patch[1]}\n      - {patch[2]}",
    )
    config.write_text(text)


def cell(
    scratch: Path,
    dataset: str,
    budget: str,
    epochs: int,
    patch: tuple[int, int, int],
    label: str,
    workers: int,
    chain: str,
    extend_slice: int,
    gpu: bool,
    window: int | None = None,
) -> CliRun:
    example = copy_example("Segmentation", scratch / label)
    shutil.rmtree(example / "Dataset", ignore_errors=True)
    write_config(example, dataset, budget, epochs, patch, workers, chain, extend_slice, window)
    # On the GPU the model is not what the clock measures, which is the point: the data path is.
    argv = konfai_executable() + ["TRAIN", "-y"] + (["--gpu", "0"] if gpu else [])
    return run_cli(argv, cwd=example, timeout=3600.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--shape", default="64,256,256")
    parser.add_argument("--patch", default="1,256,256")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--backends", default="mha,mha-gz,h5,omezarr")
    parser.add_argument("--workers", type=int, default=4, help="pinned in both regimes, so only the read path moves")
    parser.add_argument(
        "--chain",
        default="plain",
        choices=("plain", "resample", "resample-fixed", "standardize-first"),
        help="'resample' puts a Resample before the Standardize, the shape an ImpactSeg training chain has",
    )
    parser.add_argument(
        "--extend-slice", type=int, default=0, help="2.5D: stack this many neighbours beside each slice"
    )
    parser.add_argument(
        "--window", type=int, default=None, help="shuffle_window: keep this many cases under the reader"
    )
    parser.add_argument(
        "--epochs-pair",
        default=None,
        help="two epoch counts, 'a,b': each cell is run twice and the marginal per-epoch cost (Tb-Ta)/(b-a)"
        " separates the one-off measurement pass from the regime that lasts",
    )
    parser.add_argument("--regimes", default="loaded,streamed")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="train on the CPU, where the model drowns the data path")
    args = parser.parse_args()

    gate = machine_gate(force=args.force, require_performance=False)
    if not gate.quiet and not args.force:
        print(f"[perf] the machine is not quiet; refusing to time (pass --force):\n{gate.detail}")
        return 2

    shape = tuple(int(v) for v in args.shape.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))
    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]

    import torch

    gpu = torch.cuda.is_available() and not args.cpu
    scratch = Path(tempfile.mkdtemp(prefix="konfai-train-stream-"))
    print(f"device: {'gpu 0' if gpu else 'cpu'}")
    print(f"scratch {scratch}")
    source = scratch / "Dataset"
    print(f"synthesising {args.cases} cases of {shape} ...")
    synthesize(source, args.cases, shape)
    bytes_per_case = int(np.prod(shape)) * 4
    print(f"one case is {bytes_per_case / 2**20:.0f} MiB, the set {args.cases * bytes_per_case / 2**30:.2f} GiB")

    specs = {"mha": f"{source}:a:mha"}
    for fmt in wanted:
        if fmt == "mha":
            continue
        target = scratch / f"Dataset_{fmt.replace('.', '_').replace('-', '_')}"
        if fmt in COMPRESSED:
            print(f"writing {fmt} (compressed MetaImage) ...")
            synthesize(target, args.cases, shape, compress=True)
            specs[fmt] = f"{target}:a:mha"
            continue
        print(f"converting to {fmt} ...")
        convert(source, target, fmt, scratch)
        specs[fmt] = f"{target}:a:{fmt}"

    budgets = {"loaded": "64G", "streamed": "128M"}
    regimes = [(r.strip(), budgets[r.strip()]) for r in args.regimes.split(",") if r.strip()]
    epoch_counts = [int(v) for v in args.epochs_pair.split(",")] if args.epochs_pair else [args.epochs]
    if args.epochs_pair and (len(epoch_counts) != 2 or epoch_counts[0] == epoch_counts[1]):
        parser.error("--epochs-pair takes two distinct epoch counts, as 'a,b'")

    rows = []
    for fmt in wanted:
        for label, budget in regimes:
            for epochs in epoch_counts:
                name = f"{fmt}-{label}-e{epochs}" if len(epoch_counts) > 1 else f"{fmt}-{label}"
                print(f"running {name} ...", flush=True)
                run = cell(
                    scratch,
                    specs[fmt],
                    budget,
                    epochs,
                    patch,
                    name,
                    args.workers,
                    args.chain,
                    args.extend_slice,
                    gpu,
                    args.window,
                )
                epoch_clocks = run.clocks.get("epoch", {})
                # The run says which regime it chose; read it rather than assume the budget got its way.
                chosen = next(
                    (line.split("|")[-1].strip() for line in run.output.splitlines() if "memory_budget:" in line),
                    "unreported",
                )
                decided = chosen.split("->")[-1]
                decision = "loaded" if "CACHE" in decided else "streamed" if "STREAM" in decided else "unreported"
                returncode = run.returncode
                if returncode == 0 and decision != label:
                    # A cell measured in the other regime would be compared as this one.
                    print(f"  REJECTED: asked for {label}, the run chose {decision}")
                    returncode = 3
                row = {
                    "backend": fmt,
                    "regime": label,
                    "epochs": epochs,
                    "window": args.window,
                    "returncode": returncode,
                    "wall_s": round(run.wall_s, 1),
                    "peak_rss_gib": run.peak_rss_gib,
                    "clocks": epoch_clocks,
                    "regime_reported": chosen,
                }
                rows.append(row)
                if run.returncode != 0:
                    print(f"  FAILED rc={run.returncode}\n{run.output[-1500:]}")
                else:
                    wait = epoch_clocks.get("wait(data)")
                    print(
                        f"  wall {run.wall_s:7.1f} s   peak {run.peak_rss_gib:5.2f} GiB"
                        f"   wait(data) {wait}   -> {chosen[:60]}"
                    )

    if len(epoch_counts) > 1:
        lo, hi = min(epoch_counts), max(epoch_counts)
        print("\nmarginal cost of one epoch, the one-off measurement pass removed:")
        for fmt in wanted:
            for label, _ in regimes:
                pair = {r["epochs"]: r for r in rows if r["backend"] == fmt and r["regime"] == label}
                if lo in pair and hi in pair and pair[lo]["returncode"] == 0 and pair[hi]["returncode"] == 0:
                    marginal = (pair[hi]["wall_s"] - pair[lo]["wall_s"]) / (hi - lo)
                    print(
                        f"  {fmt:8s} {label:9s} {marginal:6.2f} s/epoch   (first pass {pair[lo]['wall_s']:.1f} s over {lo})"
                    )

    payload = {
        "shape": list(shape),
        "patch": list(patch),
        "cases": args.cases,
        "chain": args.chain,
        "extend_slice": args.extend_slice,
        "window": args.window,
        "rows": rows,
    }
    path = write_result("train_stream", payload, fp=fingerprint())
    print(f"\nwritten {path}")
    print(json.dumps(rows, indent=2)[:2000])
    return 2 if any(row["returncode"] != 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
