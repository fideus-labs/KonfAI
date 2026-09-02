# Benchmarks and protocol

Every performance number the documentation carries is meant to be re-runnable.
The tracked
[`benchmarks/`](https://github.com/fideus-labs/KonfAI/tree/main/benchmarks)
directory holds the harness; this page says what each script evidences and
under which protocol, so a published figure and your own re-run are compared on
the same footing.

## Protocol

- Wall time is the median of 3 runs after 1 warmup, on an otherwise idle
  machine.
- Host memory is the peak resident set of the whole process tree (`psutil`),
  sampled at 50 ms, so DataLoader workers and spawned ranks count.
- Device memory is `torch.cuda.max_memory_allocated()` plus the NVML process
  figure when available.
- Every report line carries the konfai/torch/SimpleITK versions, CPU model,
  GPU model, and the input's shape, dtype and checksum.

## The streaming claim

{doc}`../concepts/streaming` states that a case never has to fit in RAM: a
16 GiB uncompressed volume is transformed with peak host memory bounded by the
declared `memory_budget`, not by the volume. Reproduce it with one command from
a checkout (needs the `imaging` extra and free disk for the synthetic volume):

```bash
python benchmarks/bench_streaming.py --gib 16 --budget 1
```

The script synthesizes a volume of `--gib` GiB, runs a real TRANSFORM chain
over it under a declared `--budget` GiB, and reports the whole-process-tree
peak RSS beside both figures. Smaller sizes (`--gib 4`, the default) tell the
same story faster.

`benchmarks/bench_hotpaths.py` covers the framework-side hot paths (the
residual `Add` fold, the one-pass collate view, the deferred criterion
readout): micro-costs the docs assert are held rather than headline claims.

## The app comparison tables

The published tables (KonfAI-MRSegmentator and KonfAI-TotalSegmentator against
the original tools, same weights, same card) live with the apps:
[MRSegmentator](https://github.com/fideus-labs/KonfAI/tree/main/apps/mrsegmentator)
and
[TotalSegmentator](https://github.com/fideus-labs/KonfAI/tree/main/apps/totalsegmentator).
Each bundle README states its case sizes, ensemble and hardware conditions and
is produced under the protocol above; re-running them needs the published
weights and a licensed case, which is why they are app-level entries rather
than a synthetic one-command script.

Those tables are per-app measurements, not a claim of universal speedups:
compare on your own cases before drawing conclusions for your workload.

## See also

- {doc}`../concepts/streaming`: the mechanism behind the memory claim
- {doc}`large-images`: putting bounded-memory runs to work
- [`benchmarks/README.md`](https://github.com/fideus-labs/KonfAI/tree/main/benchmarks):
  the harness's own documentation
