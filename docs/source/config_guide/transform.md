# Transform configuration

Transform configuration lives under the `Transformer` root object. It is the one
workflow with **no model**: it reads a dataset, applies a chain of transforms,
and writes the result. If you never train anything, this is the only page you
need.

```yaml
Transformer:
  name: RESAMPLE_TO_ISO
  on_fallback: warn
  Dataset:
    dataset_filenames:
      - ./Raw:mha
    memory_budget: 8G
    groups_src:
      CT:
        groups_dest:
          CT_iso:
            transforms:
              ResampleToResolution:
                spacing: [1.0, 1.0, 1.0]
              Write:
                dataset: ./Out:omezarr
```

That writes `./Out/<case>/CT_iso.ome.zarr/` for every case in `./Raw`, one slab
at a time — the whole volume is never held in memory.

## Running it

```bash
konfai TRANSFORM --config Transform.yml          # transform
konfai TRANSFORM --config Transform.yml --plan   # print the plan and stop
konfai TRANSFORM --config Transform.yml --cpu 4  # shard the cases over 4 processes
```

A case whose output already exists is **skipped** — rerunning after an
interruption resumes where it stopped. Pass `-y/--overwrite` to recompute
everything.

Read transforms run on CPU; `--gpu` matters only when a chain embeds a
`KonfAIInference` stage, whose nested inference does use the device. That stage
is whole-volume and its GPU/RAM usage lives **outside** `memory_budget` — the
plan prints a note saying it cannot bound those chains. There is no `-tb`: the
workflow emits no scalars.

## Read the plan before you read anything else

Every run prints its plan first, and writes it to
`./Transforms/<name>/plan.txt`. Nothing is written until the plan is accepted.

```text
[Transformer] plan over 1 rank(s) | per-rank budget 7.45 GiB ('8G') | fallback working set
= case x 4 B x 2 (in-flight copy), headers-only estimate | output dtype/channels assumed
float32 / source channels until the first slab
  CT -> CT_iso: 120 case(s) -- 118 STREAM, 2 WHOLE-VOLUME, 0 SKIP (output already written)
    (2 case(s)) WHOLE-VOLUME: stage 1 'Standardize' needs whole-volume statistics, but an
    earlier stage changes the values -- the stored volume's statistic is not this stage's input.
    worst fallback case ~= 3.10 GiB vs per-rank budget 7.45 GiB
```

The verdicts, and each one is a fact about *your* run:

- **STREAM** — the case is read and written region by region. Memory is one
  slab, whatever the volume's size.
- **WHOLE-VOLUME** — the case is assembled in memory, then written. Always
  correct, never bounded. The line says which stage refused and why.
- **SKIP** — the output already exists; nothing is recomputed.
- **REDUCE** / **REFUSED** — a chain that folds the cohort (`Reduce`, below)
  prints one line for the whole cohort rather than one per case: `REDUCE` when
  the fold streams, `REFUSED` when it cannot. A reduction has no whole-volume
  path to fall back to, so `REFUSED` refuses the run whatever `on_fallback`
  says.

The plan is a measurement, not a prediction: it opens a **real** region-write
stream on each destination and removes it immediately, so the verdict it prints
is the one the run will act on. That is why `--plan` still touches the output
directories.

Two numbers it prints are estimates, and it says so rather than hiding it. The
case sizes come from headers alone (`prod(shape) x 4 bytes`), and the output
dtype is a hypothesis until the first slab is computed. A case sitting within a
few percent of the budget can still exceed it.

## When a chain cannot stream

`on_fallback` decides what a WHOLE-VOLUME verdict means to you:

| Value | Effect |
| --- | --- |
| `allow` | Take the whole-volume path silently — but the plan still names it. |
| `warn` (default) | Same, plus a warning line after the plan. |
| `error` | Refuse the run. Nothing is written. |

Independently of `on_fallback`, a case that **cannot stream and does not fit
`memory_budget`** always refuses the whole run, before the first byte. Writing
40 cases and dying on the 41st is worse than writing nothing: every case's size
is readable from headers, so there is no reason to find out late.

The usual reasons a chain refuses to stream, and what to do:

| Reason in the plan | Fix |
| --- | --- |
| a stage declares `WHOLE_VOLUME` | Some transforms genuinely need the volume (`Squeeze`, `Norm` change the tensor's rank). Nothing to fix — check it fits the budget. |
| a statistic after a value-changing stage | Insert a `Save:` before the statistic; the cache becomes the source the statistic reads. See below. |
| the destination cannot serve region writes | Write to `:h5` or `:omezarr` (or `:mha` for an image with geometry). |
| a halo too wide for the grid | The transform's neighbourhood is over half the slab extent; it is cheaper to load the volume. |

### `Save:` unlocks chains that would otherwise refuse

`[Clip, Standardize]` cannot stream — after `Clip`, the stored volume's
statistics are no longer `Standardize`'s input. Cutting the chain in two fixes
it:

```yaml
transforms:
  Clip: {min_value: 0.0, max_value: 400.0}
  Save: {dataset: ./Work:h5}        # milestone: the statistic seeds from THIS
  Standardize: {inverse: false}
  Write: {dataset: ./Out:omezarr}   # deliverable
```

`Save` and `Write` are the same mechanism with different intent. **`Write` is
the deliverable**: required, planned, resumed, reported. **`Save` is an
opportunistic milestone**: it is never written when the `Write` after it already
exists (the boundary skips the whole prefix). Both need a `dataset` of their own
in this workflow: a `Save:` without one would write next to your source, and
that is refused at parse time.

## What the config refuses, and when

Everything decidable from the config alone is refused **at parse time**, before
a byte is read:

- a chain that does not end with a `Write` (it would read everything and write
  nothing), or a transform placed *after* the terminal `Write` (its result goes
  nowhere);
- two chains writing the same `(dataset, group)` — the second would find the
  first one's output and report the case as already done;
- a `Write` whose destination is inside a source dataset — reading is lazy and
  streaming re-reads the source while writing, so an in-place transform would
  read its own half-written output;
- a `Save` with no `dataset` of its own, which would write next to the source;
- **any key this page does not document.** A typo'd `memory_budge:` would
  otherwise be ignored and its default used silently; here it is an error naming
  the exact path.

## Fields

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `name` | string | `TRANSFORM_01` | Names the run folder under `--transforms-dir`. |
| `on_fallback` | `allow` \| `warn` \| `error` | `warn` | What a whole-volume case means. |
| `manual_seed` | int | `0` | The seed every `Expand` in the run draws from. Same seed, same copies — which is what makes a resumed run redraw what it already wrote. |
| `Dataset` | mapping | `DataTransform()` | Sources, chains, budget. |

Under `Dataset:`:

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `dataset_filenames` | list of `path[:format]` | `["./Dataset:mha"]` | Where cases are read. |
| `memory_budget` | size string or number | `auto` | Per-rank ceiling. A bare number is GiB; `"8G"` is decimal (8 x 10^9 = 7.45 GiB), `"8GiB"` binary; `"512MB"` also works. `auto` is 80% of the node's memory, split across ranks. |
| `subset` | string / list / null | `null` | Restricts which cases run: a flat selector — a case name, a case-list file, `~file` to exclude, a `start:end` slice, or a list of those. **Not** a nested mapping; a block written under it is refused. |
| `groups_src` | mapping | — | The chains, keyed by source group then destination group. |

The grammar is deliberately **smaller** than `Train`'s `Dataset:`. There is no
`patch:` (the planner cuts slabs; declaring a patch size would be guessing on
KonfAI's behalf, and would make the output depend on it), no `batch_size`, no
`validation`, no `shuffle`, no `is_input` (every group is an input when there is
no network).

```{note}
Several `groups_src` keep only the cases present in **all** of them: a case with
an image but no label disappears from the run. The plan prints how many were
dropped, before writing anything.
```

## Three cardinalities, and where a chain changes its own

A chain is 1-to-1 by default: one case in, one entry out. Two markers change
that, at a **declared position** — everything before the marker runs at the old
cardinality, everything after it at the new one.

| Marker | Cardinality | Writes |
| --- | --- | --- |
| *(none)* | 1-to-1 per case | one entry per case |
| `Reduce` | N-to-1 over the cohort | one entry, named by `output` |
| `Expand` | 1-to-N per case | one entry per copy, named by `pattern` |

One chain changes its cardinality at most once. Composing the two — augment a
cohort, then fold it — is two invocations, the second reading the first one's
output back.

### `Expand`: one case, N copies

`Expand` multiplies, and nothing else. The draws are **ordinary stages of the
chain**, declared where they apply — so transforms and augmentations interleave
freely after the marker, and `T, draw, T, draw` means exactly what it reads
like.

```yaml
Transformer:
  name: AUGMENT
  manual_seed: 7
  Dataset:
    dataset_filenames:
      - ./Raw:mha
    groups_src:
      CT:
        groups_dest:
          CT_aug:
            transforms:
              Clip:                          # once per case, shared by the copies
                min_value: 0.0
                max_value: 400.0
              Expand:
                nb: 8
                pattern: "{name}_r{a:02d}"
              Rotate:                        # a draw, per copy
                is_quarter: true
              ResampleToResolution:          # a transform, per copy
                spacing: [2.0, 2.0, 2.0]
              Brightness:                    # another draw, per copy
                b_std: 0.2
              Write:                         # once per copy
                dataset: ./Augmented:omezarr
```

That writes `./Augmented/<case>_r01/CT_aug.ome.zarr/` … `_r08/`, eight entries
per case.

Each stage is parameterised on the grid and the case state the stages before it
leave: a draw that permutes axes hands the next stage its own extent, and a
resample between two draws is seen by the second. That is the same contract a
transform has — a draw is a stage, not a separate phase.

`pattern` is a `str.format` template and **both** tokens are required: `{name}`
keeps cases apart, `{a}` (1-based) keeps a case's copies apart. A pattern missing
either is refused at parse time, because every copy would otherwise overwrite the
previous one. Quote it, or YAML reads `{name}` as a mapping.

Two more parse-time refusals, both because the config would otherwise silently do
nothing: a draw **before** the marker (it would be a random transform applied once
per case, not a copy), and an `Expand` with **no draw after it** (every copy would
be identical).

### Augmenting an image and its mask together

The copies are drawn from `manual_seed`, not from a shared random generator: each
draw is parameterised from `(seed, case, which draw this is)`. Two chains never
meet and cannot agree on the order they consume a generator in — but they can
derive from one number they both hold. So declaring the same `nb` in both chains
is enough, and copy `k` of the mask carries copy `k` of the image's rotation:

```yaml
Transformer:
  name: AUGMENT_PAIR
  manual_seed: 7
  Dataset:
    dataset_filenames:
      - ./Raw:mha
    groups_src:
      CT:
        groups_dest:
          CT_aug:
            transforms:
              Expand: {nb: 8, pattern: "{name}_r{a:02d}"}
              Rotate: {is_quarter: true}
              Brightness: {b_std: 0.2}      # image only: it does not shift the rotation
              Write: {dataset: ./Augmented:omezarr}
      SEG:
        groups_dest:
          SEG_aug:
            transforms:
              Expand: {nb: 8, pattern: "{name}_r{a:02d}"}
              Rotate: {is_quarter: true}
              Write: {dataset: ./AugmentedSeg:omezarr}
```

A draw is keyed on its own class and its rank among draws of that class, not on
its position in the chain — so `Brightness`, declared in one chain and not the
other, does not desynchronise the `Rotate` they share.

`Expand` also takes a `seed` of its own. Leave it out and the chain inherits
`manual_seed`, which is what makes the two chains above agree; set it when you
want two chains to draw **different** copies of the same cases on purpose.

### What it costs, and what the plan tells you

Reads are decompression-bound, so N copies must not cost N reads. The engine
picks a regime per copy and the plan prints which:

```text
  CT -> CT_aug: EXPAND 40 case(s) -> 320 cop(ies) -- 280 STREAM (shared read pass),
  40 STREAM (own pass), 0 WHOLE-VOLUME, 0 SKIP (copy already written)
    (40 cop(ies)) own pass: stage 'Rotate' of this copy declares ORIENTATION, so its
    read geometry is the draw's own and it sweeps its own pass.
```

- **shared read pass** — every copy whose stages after the marker are per-voxel
  rides *one* read of the case: each slab is decompressed once, each copy applies
  its draw to it and writes into its own stream. The marginal cost of a copy is
  its draw and its write, no reads at all.
- **own pass** — a draw that reads somewhere other than its target slab (a
  rotation, a halo) has its own read geometry, so it sweeps its own pass. The
  plan names the stage that decided it.
- **WHOLE-VOLUME** — the copy's chain cannot stream at all; the shared part is
  still assembled only once for the case.

When the shared prefix is expensive (a `Warp`, a resample), put a `Save` before
the `Expand`: it is materialized once and every copy reads the cache.

Resume is **per copy**: a copy whose entry exists is skipped, so an interrupted
run picks up at the copy it stopped on — and because the draws come from
`manual_seed` rather than from the order a generator was consumed in, the copies
it keeps are the copies it would have written.

## Statistics a stage can ask for

A stage that needs a whole-volume figure declares it rather than computing it:
`GLOBAL_STAT` with the keys it reads. The planner obtains them once, by scanning
the stored entry without materialising it, and the stage is then an ordinary
value map — so it streams. That is how `Normalize` and `Standardize` work.

Two grains are available, and they come from the **same single pass**:

| Keys | What they are |
| --- | --- |
| `Min` `Max` `Mean` `Std` | the figure over the whole volume, every channel pooled |
| `MinPerChannel` `MaxPerChannel` `MeanPerChannel` `StdPerChannel` | one figure per channel |

The per-channel form exists because some quantities have no single mean. The
spatial mean of a displacement field is a **translation**: it has one part per
component, and pooling them into one number describes nothing.

```{warning}
A statistic is the STORED volume's. It is still the stage's own input only when
every stage before it preserves statistics — a reorientation does, a `Clip` does
not. `Clip` then `Normalize` therefore takes the whole-volume path, and the plan
says so. Reorder the chain, or cut it with a `Save`.
```

### `ShapeUpdate`: the shape residual of a displacement field

The shape update of an atlas build: `output = -step * (field - t)`, where `t` is
the field's per-component spatial mean in world units. Resampling a template
through the result moves it along the cohort's shape residual, at ANTs'
gradient step.

```yaml
transforms:
  ShapeUpdate:
    step: 0.25
  Write:
    dataset: ./Update:omezarr
```

`t` is **stripped, not applied**, and that is the whole point of the stage. A
total field maps template coordinates into each specimen's OWN world frame, so
its spatial mean is dominated by the frame-to-frame offset, not by any pose error
of the template. Applying it in full translates the template out of its own grid
and clips the anatomy; stripping it is what keeps the template anchored — the
same thing ANTs' `AverageAffineTransformNoRigid` is for.

Because the statistic is declared rather than recomputed per region, a field of
any size runs region by region: the volume is never assembled. Handed the whole
volume anyway — a chain that fell back for another reason — the stage takes the
statistic from the tensor in hand, so both paths leave the same state behind.

## Writing your own transform

A transform that streams is not a special kind of transform. It is three
methods, and the third is the one that matters:

```python
# BoxFilter.py -- importable from the directory you launch konfai from
import torch
import torch.nn.functional as F

from konfai.data.transform import LocalityKind, PatchLocality, Transform
from konfai.utils.dataset import Attribute


class BoxFilter(Transform):
    """Cubic moving average of radius `radius`."""

    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        self.radius = radius

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Each output voxel depends on a BOUNDED neighbourhood: declare the halo and the
        # dispatcher reads the enlarged region, crops after, and streams the whole chain.
        if self.radius == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.HALO, halo=(self.radius,))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.radius == 0:
            return tensor
        k = 2 * self.radius + 1
        return F.avg_pool3d(tensor.to(torch.float32), k, stride=1, padding=self.radius).to(tensor.dtype)
```

```yaml
transforms:
  BoxFilter:BoxFilter: {radius: 2}
  Write: {dataset: ./Out:omezarr}
```

That is the whole contract. The dispatcher builds the region map from your
declaration; you never touch it.

**Declare honestly — that is the one rule.** `patch_locality` is a promise about
what your `__call__` reads, and nothing verifies it against the code. Declaring
`POINTWISE` while reading your neighbours produces seams at every slab boundary:
a plausible, wrong image, with no error. When unsure, declare nothing — the
default is `WHOLE_VOLUME`, which is always correct and merely slower. The plan
will tell you that is what you got.

Two details the example carries on purpose. The internal padding must be the one
the whole-volume path would apply (here `avg_pool3d`'s zero padding, which the
halo never reaches except at the real volume border) — a `reflect` pad computed
on the received extent would reflect the *patch*'s edge, not the volume's. And
the halo must stay small relative to the slab: a radius over half the extent is
refused (cheaper to load the volume), and the plan says so.

### Which kind to declare

| Your `__call__` reads… | Declare | You must also write |
| --- | --- | --- |
| the same voxel only | `POINTWISE` | nothing |
| a bounded neighbourhood | `HALO`, with `halo=(r,)` | nothing |
| the volume flipped/permuted | `ORIENTATION` | `stream_region_source()` |
| a translated sub-box | `CROP` | `stream_region_source()` |
| a resampled grid | `RESCALE` | inherit from `Resample` |
| whole-volume Min/Max/Mean/Std | `GLOBAL_STAT`, with `stat_keys` | nothing |
| the same, per component | `GLOBAL_STAT`, with `MinPerChannel`/`MaxPerChannel`/`MeanPerChannel`/`StdPerChannel` | nothing |
| genuinely the whole volume | nothing (the default) | nothing |

A transform that does not subclass `Transform` still works — it is wrapped
automatically — but it is treated as `WHOLE_VOLUME`.

## Python API

```python
from konfai.transformer import build_transform

workflow = build_transform(transform_file="Transform.yml", transforms_dir="./Transforms")
plan = workflow.compute_plan(world_size=1)   # dry run: the same verdicts, as objects
print(plan.report())
print([(e.case, e.verdict, e.reason) for e in plan.fallback_entries])
```

`build_transform` returns the configured workflow **without running it**, which
is how the plan is available programmatically. `konfai.transformer.transform()`
is the execute-everything entrypoint used by the CLI.
