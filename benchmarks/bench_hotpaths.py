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

"""Micro-benchmarks of framework-side hot paths, each printed beside the alternative it replaced.

    python benchmarks/bench_hotpaths.py [--device cuda]

These are the measurements behind three audit-driven changes: the residual ``Add`` fold (one fused
elementwise kernel against stack+reduce and its N-tensor transient), the one-pass collate view
(no volume copy at batch size 1), and the deferred criterion readout (no ``.item()`` host sync
inside forward). CPU numbers show the copies; the sync cost needs ``--device cuda``.
"""

import argparse
import time

import torch


def _timed(fn, repeats: int = 20, device: torch.device | None = None) -> float:
    fn()  # warmup
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / repeats


def bench_add(device: torch.device) -> None:
    a = torch.randn(2, 32, 64, 64, 64, device=device)
    b = torch.randn_like(a)
    stacked = _timed(lambda: torch.sum(torch.stack([a, b]), dim=0), device=device)
    folded = _timed(lambda: a + b, device=device)
    transient = 2 * a.numel() * a.element_size()
    print(
        f"Add (residual sum, {list(a.shape)}): stack+sum {stacked * 1e3:.2f} ms"
        f" (+{transient / 2**20:.0f} MiB transient) vs fold {folded * 1e3:.2f} ms"
    )


def bench_collate(device: torch.device) -> None:
    volume = torch.randn(1, 256, 256, 256, device=device)
    copy = _timed(lambda: torch.stack([volume], dim=0), device=device)
    view = _timed(lambda: volume.unsqueeze(0), device=device)
    print(
        f"collate at batch=1 ({list(volume.shape)}): stack {copy * 1e3:.2f} ms"
        f" ({volume.numel() * 4 / 2**20:.0f} MiB copied) vs view {view * 1e6:.1f} us"
    )


def bench_criterion_sync(device: torch.device) -> None:
    if device.type != "cuda":
        print("criterion sync: needs --device cuda (a CPU tensor has no queue to drain)")
        return
    output = torch.randn(2, 1, 64, 64, 64, device=device)
    target = torch.randn_like(output)

    def synced() -> float:
        return (output - target).abs().mean().item()

    def deferred() -> torch.Tensor:
        return (output - target).abs().mean().detach()

    eager = _timed(synced, device=device)
    lazy = _timed(deferred, device=device)
    print(f"criterion readout: .item() in forward {eager * 1e3:.3f} ms vs deferred 0-d tensor {lazy * 1e3:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"torch {torch.__version__} on {device}")
    bench_add(device)
    bench_collate(device)
    bench_criterion_sync(device)


if __name__ == "__main__":
    main()
