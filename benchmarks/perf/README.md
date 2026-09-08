# benchmarks/perf: the numbers KonfAI publishes, from scripts

Every performance figure a page of this repository carries should come from one of these scripts,
run on a quiet machine, with the commit and the machine state written beside it. This folder is the
harness for that; `benchmarks/bench_streaming.py` and `bench_hotpaths.py` stay as they are and the
transform bench here reuses the former.

## Protocol (enforced, not promised)

1. **A quiet machine, or nothing.** `harness.machine_gate()` refuses to time when the 1-minute load
   is above 3, the power profile is not `performance`, the GPU is in use by another process, or a
   process outside the run holds more than half a core. `--force` records the numbers anyway and
   stamps every warning into the result. The audit of 2026-09-06 measured a 2x spread on every
   absolute number from the profile and the load alone; ratios measured back to back survive, absolute
   numbers do not.
2. **A fingerprint beside every number.** Commit (and dirty flag), versions, CPU, GPU and driver,
   power profile, load average, thread pin. `harness.fingerprint()`; written by `write_result`.
3. **Warmup, then the median of three**, every run kept in the JSON. Device timing is synchronized
   (`cuda_timer`, CUDA events on both ends).
4. **Memory is the whole process tree** (`PeakSampler`, RSS at 50 ms; children count) and the GPU's
   used memory (`GpuSampler`, 250 ms, baseline recorded): a working budget is not a cap on a process.
5. **Correctness is part of the timing**: a bench that produces an output compares it (voxels and
   geometry through SimpleITK, never bytes: the MHA header spelling differs between writers) against
   the reference before it reports a time.
6. **The framework's own clocks are parsed, not re-implemented**: `[KonfAI] startup ...`,
   `[KonfAI] epoch ...`, `[KonfAI] prediction ...`, `[KonfAI] sweep ...` (`parse_clock_lines`).
7. **Profilers are real ones**: `torch.profiler` for kernel time by family (`kernel_families`),
   `cProfile` for host time with the share of a named family (`cprofile_summary`), `tracemalloc`
   where a host allocation is the question, `nvidia-smi`/NVML for the device. `perf`, `nsys` and
   `ncu` are on this machine and documented per bench; `py-spy`, `memray`, `scalene` and
   `pyinstrument` are not installed and are named where they would add something.

## What to profile, with which tool (the map)

| Question | Bench | Tool | What the number is |
|---|---|---|---|
| What does a process pay before the first byte? | `bench_startup.py` | `python -X importtime`, RSS after import | import wall by top module (konfai, torch, dask, ngff_zarr), `konfai --help` wall, RSS |
| Where does one training step go, and what does autocast / channels_last do? | `bench_train_step.py` | CUDA events per phase, `torch.profiler` kernel families | ms per phase (h2d, forward, loss, backward, optimizer) fp32 / autocast / channels_last / both; kernel time by family |
| Where does an epoch go, host side? | `bench_train_epoch.py` | the trainer's `[KonfAI] epoch` clock, tree RSS, GPU peak, optional `cProfile` of the child | wall per phase (wait(data), forward, criteria, backward, validation), peaks; first vs second epoch |
| What does prediction cost, and does streaming change the result? | `bench_predict.py` | the loop's `[KonfAI] prediction` clock, tree RSS, GPU peak, SimpleITK voxel identity | whole-volume vs streamed (`KONFAI_STREAM_WORTH_THRESHOLD=0`): wall, phases, peaks, 0 differing voxels |
| What does evaluation cost outside its loop? | `bench_evaluate.py` | `[KonfAI] startup`, wall | wall vs loop |
| What does the framework add over a plain loop, and how does the region height matter? | `bench_transform.py` | `bench_streaming.py` reused, a 25-line numpy+h5py baseline, a budget sweep, `cProfile` share of the statistics scan | s/GiB over the naive loop, peak RSS, wall vs budget curve, the scan's share |
| How long is the test suite, and what does the thread pin do? | `bench_tests.py` | `pytest` wall and CPU-seconds, `OMP_NUM_THREADS` 1 vs unset | test-fast pinned/unpinned, CPU-s, failures |
| Chunked store, region height (2026-09-07, quiet: 1G 18.8 s, 2G 11.2 s, 4G 8.1 s, 8G 5.5 s, 16G 5.5 s; before the growth 24.5 / 17.5 / 8.7 / 5.9 / 7 s) | `bench_exaspim_budget.sh` | wall, peak RSS, the sweep clock, three slices against a reference | a 513x1331x1775 uint16 OME-Zarr chunked at 256^3 resampled through an ITK transform, one run per budget (`KONFAI_EXASPIM_BENCH` names the bench directory, which is not in the tree) |
| Not covered yet (next) | | | multi-GPU DDP (needs two cards), the apps with real weights (each app's own bench entry), the data path in isolation (`__getitem__` per patch), backend read/write high-water marks |

## Traps the harness knows about

- `getrusage(RUSAGE_SELF).ru_maxrss` is inherited across `exec` on Linux: a child of a parent that
  imported torch reports the parent's 650 MB peak. A child measures its own peak from
  `/proc/self/status` (`VmHWM`); the parent measures the child's tree with `PeakSampler`.
- zsh does not word-split an unquoted `$VAR`: a flag list in a shell variable reaches Python as one
  argument. Use arrays or `${=VAR}` in shell wrappers.
- The MHA header is spelled differently by the streamed writer and by SimpleITK: compare voxels and
  geometry, never bytes.
- The framework's clock lines (`[KonfAI] startup|epoch|prediction|sweep`) print only above one second: a
  0.0 in a result means under a second, not zero.
- The framework pins its own threads (`rank_cpu_share`) but the test task does not; the tests bench
  measures both states on purpose.

## Running

```bash
# one bench, on a quiet machine (the gate refuses otherwise)
pixi run --environment dev python benchmarks/perf/bench_train_step.py
# everything, sequentially, into results/<host>/<stamp>-<sha>.json + .md
pixi run --environment dev python benchmarks/perf/run_all.py            # add --quick for a smoke pass
# compare a run to a baseline (exit 1 on a regression beyond the thresholds)
pixi run --environment dev python benchmarks/perf/compare.py results/<host>/<baseline>.json results/<host>/<run>.json
# the release gate: the series, then the comparison to baselines/<host>.json (exit 1 on a regression)
pixi run perf-check
```

The comparison exits **0** only for a complete comparable series, **1** for a measured regression,
and **2** for invalid evidence: missing/failed scenarios or metrics, non-finite values, mismatched
machine settings, gate warnings, or a quick run. Unit tokens also cover scenario-qualified names
such as `wall_s_whole`, `peak_rss_gib_whole` and `konfai_wall_s_b1`. A failed scenario makes
`run_all.py` exit nonzero after writing its diagnostic series; it cannot certify a release by
leaving the failed scenario out. Run `run_all.py --quick` directly when only checking the scripts.

Every bench accepts `--force` (time on a busy machine, warnings recorded), `--repeats N`, `--quick`
(smaller inputs, one repeat: a smoke test of the bench itself, never a number to publish) and
`--out DIR`. Results are git-ignored; a curated baseline per machine is what gets committed, by hand,
with its fingerprint.

## Reading a result

A result is `{"bench", "fingerprint", "result"}`. Compare two results only when their fingerprints
agree on the commit's dirty flag, the GPU, the power profile and the thread pin, and both load
averages were under the gate. A `--force` result carries `"gate_warnings"`: it is a diagnostic, not a
baseline.

## The numbers this folder was built to pin (audit of 2026-09-06, quiet machine, performance profile)

| Bench | Figure | Where it is quoted |
|---|---|---|
| train step, 2D example | 39.5 ms fp32, 22.4 ms autocast (1.76x); backward 65 %, forward 30 %, criteria 4 %. This folder, 2026-09-07: 38.2 / 20.4 (1.88x) / channels_last 35.1 / both 17.0 ms (2.25x) | `.audit-local/AUDIT-2026-09-06.md` 3.3.2 |
| train loop, 3D 64^3 toy | fp32 2.60 s, autocast 3.88 s, autocast + channels_last 2.23 s | `.audit-local/RECONCILIATION-CODEX-2026-09-07.md` 6 |
| train step, SynthRAD 2025 Task 1 UNet++ (2.5D, batch 32 of 5x320x320, 26 M) | fp32 852 ms, autocast 521 (1.63x), channels_last 741 (1.15x), both 387 ms (2.20x) | this folder, 2026-09-07, `bench_train_step.py --model-classpath UNetpp:UNetpp5 --sys-path <clone>/KonfAI --shape 32,5,320,320` |
| train step, CURVAS ResidualEncoderUNet (3D, batch 2 of 128x160x160, 102 M) | fp32 1244 ms / 17.6 GB, autocast 617 (2.02x) / 9.6 GB, channels_last 1532 (0.81x), both 779 ms (1.60x): channels_last hurts this model | this folder, 2026-09-07, `--model-classpath Model:ResidualEncoderUNet --sys-path <clone>:<clone>/dynamic-network-architectures --shape 2,1,128,160,160 --steps 10 --profile-steps 0` (at 1x1x112x224x288 the channels_last variant ran 40 min at the VRAM limit without finishing) |
| prediction, 2D example | loop 4.1 s whole, 4.4 s streamed, voxel-identical | 3.3.5 |
| transform, 2 GiB h5 | 7.8 s / 1.14 GiB vs naive 4.0 s / 0.24 GiB; flat past 4-8 chunk rows. This folder, fresh processes, 2026-09-07: 7.4-7.9 s / 0.9-1.0 GiB vs 4.1-4.6 s / 0.23 GiB; sweep 15.6 / 10.9 / 9.1 / 7.6 / 7.7 / 7.8 / 7.6 s; statistics scan 2.4 s cumulative, 0.6 s own (not 72 %) | 3.2.3, 3.2.6, 3.2.7 |
| tests | test-fast 181 s unpinned, 21.6 s pinned (quiet). This folder under load 4-5: 116.5 vs 26.4 s, the three known failures listed | 3.4.0 |
| startup | `import konfai.trainer` 2.0 s, 604 MB; OME-Zarr chain +0.28 s idle | 3.3.4 |
