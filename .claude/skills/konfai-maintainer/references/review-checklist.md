# Review checklist — per-subsystem invariant + confirmed-trap watchlist

Map the diff to the subsystem it touches, then check the invariant that a change there most easily breaks.
The **watchlist** entries are traps confirmed by the 2026-07-19 deep audit — verify the diff does not
introduce or leave them (some are open at time of writing; keep them from spreading).

## Per-subsystem invariant

| If the diff touches… | The invariant to protect | How to check |
|---|---|---|
| `utils/config.py` (binder) | Reading a config **mutates the file**; the write-back is **atomic** (temp + `os.replace`); resolved defaults are the reproducibility record | Snapshot config bytes before any load; a reader must never see a truncated file. Add a `test_config.py` case for any new type shape. |
| `utils/config.py` union types | Coercion is **declaration-order** and lossy | Test the new key with a value that matches a *later* union member (e.g. a float where `int` is first, a `list`). |
| `data/patching.py`, `data/data_manager.py` | **Patch order out (disassemble) == order in (Accumulator)**; all patches of a case stay on **one DDP rank** for predict/eval | Reassemble an identity through the pipe and assert byte-identical, incl. `overlap>0` and non-divisible shapes. |
| `data/transform.py` | `transform_shape()` is **exact**; `patch_locality` never over-claims (a wrong locality silently corrupts the streamed path) | Cross-check predicted vs actual output shape; run the streamed vs whole-volume A/B for the transform. |
| geometry anywhere | Arrays are **channel-first `[C,(Z),Y,X]`**; spacing/origin/direction are **`(x,y,z)`** (sitk) | A displacement/resample must map physical mm through spacing+direction, then reverse to array order. Test a known 1-axis translation lands on the expected axis/voxel. |
| `network/network.py` | `state_dict` does **not** recurse into nested Networks; `outputs_criterions` keys equal a module's **dotted path** (`:`/`.` load-bearing); alias lists are **positional** | Build + save + load + RESUME a *nested* model; assert optimizer/scheduler/`_it` continue. |
| `metric/measure.py` | A `Criterion.forward` returns a `Tensor` (loss) or `(value, dict)` (metric); metric state must reset per split | Instantiate and run the criterion on a tiny CPU tensor; a metric must not accumulate across train+val. |
| `predictor.py` | Streamed output is **byte-identical** to whole-volume; the mode is stated when a case is not window-bounded | A/B with `KONFAI_STREAMED_WRITES=0`; check the writer fingerprint (streamed sink omits `CenterOfRotation`). |
| `utils/dataset.py` | Streaming paths **never materialize a full volume**; `Attribute` round-trips only flat scalars/1-D arrays; streamed entries appear only once finalized (temp+rename) | Confirm no hidden full read on the streaming path; a replaced entry stays readable until its replacement is ready. |
| `utils/model_builder.py` | The trusted/untrusted **boundary** — only registry types; module names contain no `.`; `default|` = bare filename | No `eval`/import injection; a path separator in a `default|` name is refused. |
| `konfai-mcp` job/path code | Validation/smoke-tests run **only in a spawn subprocess** (never the server process); `read/write_session_file` are **path-jailed**; `cancel_job` reaps the whole process group | Any write target that composes a path rejects separators; dataset *reads* may be arbitrary host paths by design — writes must never widen. |
| a workflow kind | Adding one touches ~12 registries + ~8 `Literal`s | Prefer one descriptor table; a drift-guard test asserts each registry is derived from it. |

## Confirmed-trap watchlist (2026-07-19 audit — do not introduce or spread)

- **Config union coercion**: `overlap: 0.25` binds `0` (no blend); `list`-typed union members never bind. Any
  new union-typed config key must be tested against a non-first member. *(open P0)*
- **Nested-Network save/load key mismatch**: `checkpoint_save` writes dotted paths, `Network.load` reads bare
  class names — composite models lose optimizer/scheduler/`_it` on RESUME, and dedup-by-class-name loses a
  duplicate sibling. Keep the two coordinate systems in agreement.
- **Criterion coordinates**: `Measure.init` validates per-owning-network but runtime matches in root
  coordinates — a GAN criterion key fails at startup or trains at silent zero.
- **Geometry axis order**: `ResampleTransform` / `Elastix` mixing `(z,y,x)` grids with `(x,y,z)` spacing and
  mm-as-voxel — silently wrong warps.
- **Augmentation determinism**: per-epoch redraws don't reach `persistent_workers`; validation reuses train
  draws (index-keyed state); foreign-augment CUDA RNG not restored.
- **Split reproducibility**: the train/val split is drawn from the unseeded global RNG before per-rank
  seeding, so `manual_seed` doesn't cover it and RESUME re-splits (validation leakage).
- **DDP prediction with measures**: per-batch `all_gather` on unequal shards deadlocks — prediction-time
  measure logging must be rank-local.
- **h5 output path**: string-mutating an output path (append `"/"`) without re-deriving `is_directory` hides
  predictions in `Dataset/.h5`. Rebuild through `Dataset` normalization.
- **Metrics that can't run**: `FID` used `torch.nn.functional.resize` (nonexistent); `Accuracy` accumulates
  for the process lifetime; `PerceptualLoss` zips `strict=False` and drops extra losses. A metric smoke test
  catches all three.
- **CI/wheel**: no CI installs the built wheel — a PEP 420 / package-data regression ships green. Use the
  release script.

Full evidence for each trap lives in the maintainer's local audit notes under `.audit-local/` (git-ignored);
this watchlist is the portable summary.
