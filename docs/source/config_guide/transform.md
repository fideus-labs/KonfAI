# Transform configuration

Transform configuration lives under the `Transformer` root object. `TRANSFORM`
is the dataset-preparation workflow: it reads a dataset, applies a chain of
transforms, and writes a dataset. `EVALUATION` measures, this one **makes**. If
you never train anything, this is the only page you need.

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
              Resample:
                spacing: [1.0, 1.0, 1.0]
              Write:
                dataset: ./Out:omezarr
```

That writes `./Out/<case>/CT_iso.ome.zarr/` for every case in `./Raw`, one slab
at a time: the whole volume is never held in memory.

A streamed OME-Zarr write chunks the store the way it was written: one chunk is
`[C, slab_rows, Y, X]`, tiled down to at most `128 x 128` in-plane once that
exceeds 32 MiB, so a large plane lands as `[C, slab_rows, 128, 128]`.
`slab_rows` follows the budget (64 without one, fewer under a tight
`memory_budget`), so the layout of the store depends on the machine that wrote
it. A training reader cutting `32^3` patches from that store decompresses those
chunks, `slab_rows x 128 x 128` voxels for a 32-row patch. A case written whole
(`LOAD`, `WHOLE-VOLUME`) declares no region shape and keeps `ngff-zarr`'s
default chunking.

## Running it

```bash
konfai TRANSFORM --config Transform.yml          # transform
konfai TRANSFORM --config Transform.yml --plan   # print the plan and stop
konfai TRANSFORM --config Transform.yml --cpu 4  # shard the cases over 4 processes
```

A case whose output already exists is **skipped**: rerunning after an
interruption resumes where it stopped. Pass `-y/--overwrite` to recompute
everything. A case that fails (an unreadable file, a stage that raises) does
not stop the others: the rank finishes its shard, prints the failed cases,
and exits non-zero; a rerun resumes at exactly those cases.

`--plan` reads the config the way a run does, and reading a config resolves
its defaults back into the file: after `--plan`, `Transform.yml` carries every
key the grammar knows, at the value the plan used. Copy the file first if you
want to keep the text you wrote. `konfai-mcp`'s `plan_transform` snapshots the
session's `Transform.yml` and restores it once the plan is back, and
`konfai.plan_transform` writes the tree it is handed to a scratch file, so
neither leaves that write behind.

`--cpu N` shards the **work items** over N processes, not the cases: an
ordinary chain has one item per case, so a case read by two chains is two items
and may land on two ranks; a reduction is one item for the whole cohort. Items
are dealt heaviest first onto the least-loaded rank, weighed by bytes, so one
large case does not hold a rank alone while the others finish. Every
`Save`/`Write` must then point at a directory destination (`mha`, `nii.gz`,
`omezarr`), because a single-file store (`h5`) would have every rank writing
into the same file, and the run refuses before any byte. A reduction being one
item, `--cpu 8` on a chain that only reduces leaves seven ranks idle. The ranks
share nothing but the work list: no process group is initialized, and each rank
reports its own shard.

With `--gpu`, each rank runs its chain on its device: the slabs are pulled to
the GPU, transformed there, and land back on the host to be written, in taller
slabs than on a CPU (as much as a quarter of the free device memory allows).
The bytes are the ones a CPU run writes. The gain is modest for a light chain
(the reads and the writes are the cost either way) and grows with the volume
and the chain. `--gpu` also matters when a chain embeds a `KonfAIInference`
stage. That stage is a bridge, not a second prediction
engine: it is whole-volume (the case is assembled, written to a temporary
`.mha`, and read back), and each case spawns a process that resolves the app
(the HuggingFace files are cached after the first case, its code is imported
every time) and loads the model again, so a 100-case cohort loads the model 100
times. Its GPU/RAM usage lives
**outside** `memory_budget`: the plan prints a note saying it cannot bound those
chains. Inference over a cohort is what `PREDICTION` is for: it loads the model
once, tiles patches, and streams the output. Use `KonfAIInference` in a chain
when the inference is one stage of a larger transform and a later stage
consumes its result. There is no `-tb`: the workflow emits no scalars.

## Read the plan before you read anything else

Every run plans first, and a run that proceeds opens its log with that plan,
in `./Transforms/<name>/`, next to an `outputs.json` declaring each chain's
configured destination. No *data* is written until the plan is accepted.

The console gets one line, because the plan's detail grows with the cohort:
what it will do, and where to read the rest. Nothing is dropped in silence,
what it folds away it counts.

```text
[KonfAI] plan over 1 rank(s) | 120 entr(ies): 18 LOAD, 100 STREAM, 2 WHOLE-VOLUME | per-rank
budget 7.45 GiB ('8G') | 2 note(s) -> full plan in ./Transforms/CT_ISO/log_0.txt
```

The log holds the same plan in full, one line per chain and per reason.
`--plan` prints that full form and stops, without running anything:

```text
[KonfAI] plan over 1 rank(s) | per-rank budget 7.45 GiB ('8G') | fallback working set
= case x 4 B x (2 + the widest stage's own buffers), headers-only estimate | output dtype/channels assumed
float32 / source channels until the first slab
  CT -> CT_iso: 120 case(s) -- 100 STREAM, 18 LOAD, 2 WHOLE-VOLUME, 0 SKIP (output already written)
    (18 case(s)) LOAD: fits the per-rank budget (~0.42 GiB vs 7.45 GiB); streaming would read
    ~2.0x the source
    (2 case(s)) WHOLE-VOLUME: stage 1 'Standardize' needs whole-volume statistics, but an
    earlier stage changes the values -- the stored volume's statistic is not this stage's input.
    worst fallback case ~= 3.10 GiB vs per-rank budget 7.45 GiB
```

A reduction prints one line for the cohort instead, `REDUCE` when the fold
streams and `REFUSED` when it cannot:

```text
  CT -> CT_template: REDUCE 120 case(s) -> 1 output 'template': REFUSED
    case 'CASE_001' lands on extent [44, 60, 52] where 'CASE_000' lands on [48, 56, 56]
```

The verdicts, and each one is a fact about *your* run:

- **STREAM**: the case is read and written region by region. Memory is one
  slab, whatever the volume's size.
- **LOAD**: the case *could* stream, fits the budget, and streaming would
  re-read the source (a halo re-reads its overlap, a regrid pulls each slab's
  window through its map, a compressed store decodes the whole volume per
  slab). Loading reads it once; the line prints the predicted factor. A choice,
  not a fallback: `on_fallback` has nothing to say about it. Streaming is a
  memory strategy: it is chosen when the case does not fit, or when it costs
  nothing.
- **WHOLE-VOLUME**: the case cannot stream and is assembled in memory, then
  written. Always correct, never bounded. The line says which stage refused and
  why.
- **SKIP**: the output already exists; nothing is recomputed.
- **REDUCE** / **REFUSED**: a chain that folds the cohort into one entry
  ({doc}`Reduce <../reference/components/transforms>`)
  prints one line for the whole cohort rather than one per case: `REDUCE` when
  the fold streams, `REFUSED` when it cannot. A reduction has no whole-volume
  path to fall back to, so `REFUSED` refuses the run whatever `on_fallback`
  says.

The plan is a measurement, not a prediction: it opens a **real** region-write
stream on each destination and removes it immediately, so the verdict it prints
is the one the run will act on. A store the probe had to create is removed with
it, so `--plan` leaves no output behind.

Two numbers it prints are estimates, and it says so rather than hiding it. The
case sizes come from headers alone (`prod(shape) x 4 bytes`), and the output
dtype is a hypothesis until the first slab is computed. A case sitting within a
few percent of the budget can still exceed it.

## When a chain cannot stream

`on_fallback` decides what a WHOLE-VOLUME verdict means to you:

| Value | Effect |
| --- | --- |
| `allow` | Take the whole-volume path silently, but the plan still names it. |
| `warn` (default) | Same, plus a warning line after the plan. |
| `error` | Refuse the run. Nothing is written. A fallback only discovered mid-run (a failed sweep) stops at that case: earlier cases stay written, and the per-case resume covers the rerun. |

Independently of `on_fallback`, a case that **cannot stream and does not fit
`memory_budget`** always refuses the whole run, before the first byte. Writing
40 cases and dying on the 41st is worse than writing nothing: every case's size
is readable from headers, so there is no reason to find out late.

The usual reasons a chain refuses to stream, and what to do:

| Reason in the plan | Fix |
| --- | --- |
| a stage declares `WHOLE_VOLUME` | Some transforms genuinely need the volume (`Squeeze`, `Norm` change the tensor's rank). Nothing to fix: check it fits the budget. |
| a statistic after a value-changing stage | Insert a `Save:` before the statistic; the cache becomes the source the statistic reads. See below. |
| the destination cannot serve region writes | Write to `:h5` or `:omezarr` (or `:mha`/`:nii` for an image with geometry). |
| a halo too wide for the grid | The transform's neighbourhood is over half the slab extent; it is cheaper to load the volume. |

### `Save:` unlocks chains that would otherwise refuse

`[Clip, Standardize]` cannot stream: after `Clip`, the stored volume's
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
- two chains writing the same `(dataset, group)`: the second would find the
  first one's output and report the case as already done;
- a `Write` whose destination is inside a source dataset: reading is lazy and
  streaming re-reads the source while writing, so an in-place transform would
  read its own half-written output;
- a `Save` with no `dataset` of its own, which would write next to the source;
- **any structural key the grammar does not list, and any stage argument its
  constructor does not take.** A typo'd `memory_budge:` or
  `Clip: {min_val: …}` would otherwise be ignored and its default used
  silently; here it is an error naming the exact path and the legal keys. (A
  stage that takes `**kwargs` or resolves nowhere is left to the loader's own
  error, and the contents of `subset:` are not walked.)

## Fields

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `name` | string | `TRANSFORM_01` | Names the run folder under `--transforms-dir`. |
| `on_fallback` | `allow` \| `warn` \| `error` | `warn` | What a whole-volume case means. |
| `manual_seed` | int | `0` | The seed every `Expand` in the run draws from. Same seed, same copies, which is what makes a resumed run redraw what it already wrote. |
| `Dataset` | mapping | `DataTransform()` | Sources, chains, budget. |

Under `Dataset:`:

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `dataset_filenames` | list of `path[:format]` | `["./Dataset:mha"]` | Where cases are read. |
| `memory_budget` | size string or number | `auto` | Per-rank ceiling. A bare number is GiB; `"8G"` is decimal (8 x 10^9 = 7.45 GiB), `"8GiB"` binary; `"512MB"` also works. `auto` is 80% of the node's memory, split across ranks. |
| `subset` | string / list / null | `null` | Restricts which cases run: a flat selector: a case name, a case-list file, `~file` to exclude, a `start:end` slice, or a list of those. **Not** a nested mapping; a block written under it is refused. |
| `groups_src` | mapping |: | The chains, keyed by source group then destination group. |

The grammar is deliberately **smaller** than `Train`'s `Dataset:`. There is no
`patch:` (the planner cuts slabs; declaring a patch size would be guessing on
KonfAI's behalf, and would make the output depend on it), no `batch_size`, no
`validation`, no `shuffle`, no `is_input` (every group is an input here: nothing
is a target).

Several `groups_src` keep only the cases present in **all** of them: a case with
an image but no label disappears from the run. The plan prints how many were
dropped, before writing anything.

## Three cardinalities, and where a chain changes its own

A chain is 1-to-1 by default: one case in, one entry out. Two markers change
that, at a **declared position**: everything before the marker runs at the old
cardinality, everything after it at the new one.

| Marker | Cardinality | Writes |
| --- | --- | --- |
| *(none)* | 1-to-1 per case | one entry per case |
| `Reduce` | N-to-1 over the cohort | one entry, named by `output` |
| `Expand` | 1-to-N per case | one entry per copy, named by `pattern` |

One chain changes its cardinality at most once. Composing the two (augment a
cohort, then fold it) is two invocations, the second reading the first one's
output back.

### `Reduce`: N cases, one volume

Everything before the marker runs once per case, `Reduce` folds the cohort at
fixed voxel, and everything after it runs once on the result. The chain is
driven by the reduction engine (it walks the *output's* regions and, within a
region, the cases), so peak memory is a few regions, never N volumes:

```yaml
transforms:
  Clip: {min_value: 0.0, max_value: 400.0}
  Reduce:
    operator: Mean
    output: template
  Write: {dataset: ./Atlas:h5}
```

That writes **one** entry named `template`, whatever the cohort's size.

| Field | Default | Effect |
| --- | --- | --- |
| `operator` | `Median` | A classpath resolved against `konfai.data.reduction`: `Mean`, `Median`, `Vote`, `Concat`, or your own `Reduction` subclass. An operator's own parameters go in the same mapping, next to `operator`. |
| `output` |: | **Required**: the entry name the single result is written under. |
| `grid` | `strict` | How much agreement between members is demanded before a byte is read (below). |
| `grid_tolerance` | `1e-6` | The tolerance `strict` compares geometry within. |
| `provenance` | `true` | Record the operator and the folded case list in the output's header: a cohort that silently changed between two runs writes a different volume under the same name, and nothing about the output would look wrong. |

**Operators.** `Mean` folds one case at a time, so its working set is two
regions whatever N is. `Median` needs every case per region, and stacks and
sorts them on top: the plan says how many regions that is, and `memory_budget`
sizes and refuses against it. `Concat` puts the cases side by side: the output
carries `N × C` channels. A custom operator must declare `voxel_local = True`: one that reads across space cannot stream and is refused outright. It should
also declare `working_multiple` if it allocates over the buffer it is handed,
or the plan promises a working set the run exceeds.

```{warning}
`Mean` and `Median` are for intensities. Both answer with values that were in no
input (the median of labels `1` and `5` is `3`, a different structure) and
both widen an integer input to float32. Over exactly two cases `Median` *is*
`Mean`, so the robustness the name promises is not there. Fold segmentations
with **`Vote`**, which takes the label the most cases agree on, keeps the input
dtype, and breaks a tie toward the smallest label so the fold is reproducible.
```

**`grid` decides what counts as "the same space"**, compared on the grid each
case's chain *lands* on (a `Resample` before the `Reduce` counts):

- `strict` (default): equal extents **and** equal `Spacing`/`Origin`/
  `Direction` within `grid_tolerance`.
- `shape_only`: equal extents alone: the escape hatch for volumes already
  resampled together but carrying approximate headers.
- `reference:<case>`: equal extents, and the output adopts that member's
  geometry: how a cohort says its members disagree on their headers and which
  one to believe.

```{warning}
Nothing can verify that the members truly live in a common space: only that
they claim to. `shape_only` and `reference:` will happily average misaligned
volumes; the result still looks like a volume and is an artefact. Put the
cohort on one grid first: `Resample: {reference: …}`, below.
```

**A reduction has no whole-volume fallback.** Folding every case in memory is
what it exists to avoid, so a `Reduce` that cannot stream refuses the run
whatever `on_fallback` says: the plan prints `REFUSED` and the reason. Only
voxel-local stages (and statistics, seeded by an extra pass of the engine's
own) may follow the `Reduce` in the same chain: anything reading across space
belongs in a second chain that reads the written output back.

**What it reads.** The fold walks the output's regions and reads each region
from every member. A member on a store that serves bounded region reads (`h5`,
`omezarr`, an uncompressed `mha` or `nii`) is read once. A member on a store
that cannot (`nii.gz`, a compressed `mha`, NRRD) decodes its whole volume
behind every region read: once per region, twice that when a statistic of the
result is seeded by a first pass, and a `memory_budget` that lowers the slab
raises the count. The plan prices it (`reads: ... decodes per member`) and
prints the remedy: put a `Save: {dataset: ./Cache:h5}` before the `Reduce`, so
each member is materialized on a bounded store first and the fold reads the
cache. A reduction is also one work item: `--cpu N` cannot split it over ranks.

### `Resample`: one stage, two questions

Every resample in KonfAI is one stage, `Resample`, and it asks two independent
questions:

| | key | meaning |
| --- | --- | --- |
| **which grid to write on** | *(nothing)* | the case's own: the map moves the anatomy, the voxels stay put |
| | `spacing` | the same field of view at another density |
| | `shape` | the same field of view at a given count |
| | `reference` | the grid of a stored image, adopted whole |
| **what map to write it through** | *(nothing)* | the identity: a change of grid and nothing else |
| | `field` | a displacement field, on its own grid, in world units |
| | `transforms` | transforms stored beside the cases (rigid, affine, BSpline, field, composite) |

Any combination is legal, and asked for together they compose into **one
interpolation**: the source is read once, at the displaced point. Doing it as
two stages resamples twice, and a volume interpolated twice has lost detail the
second pass invented no more of.

`align` places a `spacing` or `shape` grid, and it is the one choice the family
used to make silently: `extent` (the default) keeps the field of view (the
outer faces coincide), while `origin` keeps voxel zero's centre where it is. A
quarter of a voxel of anatomy separates them, and a `reference` states its own
placement and ignores this.

Under `extent` the voxel count is rounded, so the spacing written is the one
that fits the field of view, not exactly the one asked: `spacing: [1, 1, 1]`
over 204.8 mm lands 204 voxels of 1.0039 mm. Two cases of different extents
resampled to the same `spacing` therefore land on two slightly different
spacings, which `Reduce` under `grid: strict` refuses. To fold a cohort onto one
grid, resample every member onto a `reference` (a stored image, or the grid a
`Reduce` names) rather than to a bare spacing.

### `Resample: {reference: …}`: making `strict` true rather than waived

A reference can also **follow the case**: `reference: '{case}'` adopts, for each case,
the grid of that case's *own* entry in `reference_group`. That is the registration idiom: `reference: '{case}', reference_group: DVF` lands every moved image on its own field's grid,
which is where a displacement field is defined. A literal reference stays one header lookup
for the whole cohort; a per-case one is one per case, headers only either way.


A cohort as acquired rarely passes `strict`: extents differ, and origins can
differ by more than the volumes are wide, because an acquisition's stage
coordinates are not an anatomical frame. A `reference` grid is what makes
`strict` true rather than waived, it resamples each case onto the grid of a
**declared reference**, adopting its extent, spacing, origin and direction:

```yaml
transforms:
  Resample: {reference: case_0, reference_group: CT, fill: 0.0}
  Reduce: {operator: Median, output: template, grid: strict}
  Write: {dataset: ./Template:mha}
```

The reference is a stored image, named by `reference`, and by
`reference_group` when the store holds more than one. It is looked up by entry, not by the case being
processed, because one grid serves the whole cohort: in the run's own
`dataset_filenames`, or in a store of its own.

```yaml
transforms:
  Resample: {reference: case_1, reference_group: CT, reference_dataset: ./Raw:mha}
  Write: {dataset: ./OnTemplate:mha}
```

`reference_dataset:` takes the same `path[:format]` spec as everywhere else, so the
reference can live anywhere, which is the atlas loop: point round N+1 at the
store round N wrote its template into.

#### Through a displacement field, in one interpolation

Add `field:` and the stage becomes the whole of a registration's apply step: `sitk.Resample(image, reference_grid, DisplacementFieldTransform(field))`. For
each voxel of the **target** grid the field is read at that voxel's world
position, added, and the source sampled **once** at the displaced point:

```yaml
transforms:
  Resample:
    reference: case_0
    reference_group: CT
    field: ./Fields:mha
    field_group: DVF
  Write: {dataset: ./Registered:mha}
```

Fields stored *beside* the cases (one entry per case, in the same roots) need
no path at all: `field_group: DVF` on its own finds them.

Doing this as two stages instead (resample onto the grid, then warp) costs
**two** interpolations, and the second cannot restore the detail the first
smoothed away. That is not a small effect: on a high-frequency volume the
second pass moves voxels by a large fraction of the range, which is exactly why
an atlas rebuilds its appearance from native volumes rather than from resampled
ones.

The field lives on **its own grid**, usually coarser than either the source or
the target, and is read where it is asked, it is defined in world units, so a
field solved at 120 µm moves a volume stored at 30 µm without being upsampled
first. Outside its own extent the displacement is zero: the transform is the
identity where the field says nothing, as SimpleITK has it.

Nothing is declared about how far the field reaches, and nothing is recorded.
The field window a region samples is its own box, read for sampling regardless, and the sup of the values just read bounds that region's source pull, so
each slab pays exactly the halo *its* displacements require, measured at run
from a read the sampler needed anyway. The one thing the plan cannot know from
headers is the cost of those reads: it prices them as if the field were zero,
and says so.

Naming no target grid is the shape update of an atlas build (the field applied
on the case's *own* grid), and is the same stage with `reference` left out:

```yaml
transforms:
  Resample: {field: ./Fields:mha, field_group: DVF}
  Write: {dataset: ./Warped:mha}
```

The field and the case need not share a grid: the field is read at each target
voxel's world position on the field's own grid, so a field solved at 120 µm
moves a volume stored at 30 µm without being upsampled first.

Naming an image rather than fifteen numbers is deliberate. A grid is an extent
in array order `(Z, Y, X)` plus an origin, a spacing and a direction in physical
`(x, y, z)`: transcribing those by hand is the mistake that actually gets made,
and a transposed grid resamples perfectly well onto the wrong place. A header
cannot make that mistake.

It streams: a slab of the output reads only the part of the input under it, so a
case never has to fit in memory. The sampler is `sitk.Resample`'s: linear with
taps clamped to the buffer, nearest by round-half-up, and `fill` wherever the
reference grid reaches past the case.

**Label maps.** Left unset, `interpolation` is read off the dtype: `uint8` takes
the nearest voxel, everything else is interpolated. A dtype cannot decide this
on its own (a CT is `int16` and so is nothing else about it), so a label map
stored as anything but `uint8` must say so:

```yaml
Resample: {reference: case_0, reference_group: Labels, interpolation: nearest}
```

Getting it wrong is silent. Two labels blended give a third that was never in
the source, the dtype is unchanged, and the result is still a label map.

**What it refuses**, rather than write something plausible and wrong:

- a case or a reference carrying no `Origin` / `Spacing` / `Direction`: without
  geometry there is no physical space to resample in, and a size ratio must not
  quietly stand in for one;
- a reference whose `Direction` differs from the case's: the map between them
  is then a rotation, not a scale and a shift per axis. Reorient first
  (`Canonical`);
- a case that does not meet the reference grid **anywhere**: its output would
  be `fill` from edge to edge, and a median would take that as anatomy.

Partial overlap is legal and ordinary: the rest of the output is `fill`, and the
plan prints the fraction of the grid each case covers. "Most of this template is
background" is then something read before the run rather than after it.

### `Resample: {transforms: …}`: applying a registration that was already solved

With no target grid named, `Resample` changes nothing about the grid and moves
the anatomy through transforms **stored beside the cases**: the apply step of a
registration solved elsewhere:

```yaml
transforms:
  Resample: {transforms: {reg: false}}
  Write: {dataset: ./Registered:mha}
```

Each key of `transforms:` is a group of the run's own datasets holding one
transform per case; the value says whether to invert it. Rigid, affine, BSpline
and displacement-field entries are all read the same way, and several groups
compose: the **last declared is applied first**, which is SimpleITK's own
composite order.

**It streams**, and what makes that possible is that a stored transform can say
how far it reaches. A rigid or affine map is an exact affine, so the source box
of a target region is that region's box mapped through it. A BSpline and a dense
field are values on a grid read through a kernel that is non-negative and sums
to one, so the largest of those values bounds the displacement at *every* point, not at the points someone sampled. The region a slab must read is therefore
known before a voxel is touched.

Bounded is not the same as cheap. A map **oblique to the storage axes** has an
axis-aligned source box that covers most of the volume on two axes, and it gets
worse the thinner the slabs: the same case that reads 1.0× its bytes in one
piece reads several times that in slabs. The bound is exact either way (this is
a property of the decomposition, not a defect), but it is why streaming such a
map is not automatically worth it. Bring the case onto an axis-aligned grid
first (`Canonical`) when the geometry allows it.

**What it refuses**, declaring `WHOLE_VOLUME` with the reason so the run
proceeds on the whole-volume path rather than breaking:

- a case whose header carries no `Origin` / `Spacing` / `Direction`: a stored
  transform is applied in physical space, and without a geometry there is none;
- a transform type that decomposes into no bounded map, naming the type;
- `invert: true` on a spline or a displacement field. Inverting one is a dense
  solve over the whole grid, and a field solved per region is not the
  restriction of the field solved once, so store the inverse where the
  transform is written, or set the group to `false`.

`interpolation` and `fill` work the same wherever they appear: unset, `uint8`
takes the nearest voxel and everything else is interpolated, and a label map
stored as anything else must say `interpolation: nearest`. Nearest here is ITK's
round-half-up on the physical index: the same coordinate the linear sampler
reads, so a mask and the image beside it land on the same voxels.

### `Expand`: one case, N copies

`Expand` multiplies, and nothing else. The draws are **ordinary stages of the
chain**, declared where they apply, so transforms and augmentations interleave
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
              Resample:                       # a transform, per copy
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
transform has: a draw is a stage, not a separate phase.

```{note}
`Flip`, `Permute`, `Mask` and `Foreign` exist as a transform and as a draw. A
bare name resolves against `konfai.data.transform` first **before** the
`Expand` marker and against `konfai.data.augmentation` first **after** it, so
`Flip: {f_prob: [0.33, 0.33, 0.33]}` past the marker is the draw. To force the
other one, spell it out: `konfai.data.transform:Flip`.
```

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
draw is parameterised from `(seed, the case's name, which draw this is)`. Two
chains never meet and cannot agree on the order they consume a generator in, but
they can derive from one number they both hold. So declaring the same `nb` in
both chains is enough, and copy `k` of the mask carries copy `k` of the image's
rotation:

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
its position in the chain, so `Brightness`, declared in one chain and not the
other, does not desynchronise the `Rotate` they share. And it is keyed on the
case's **name**, not on its position in the run's case list, so a `subset`, a
case that fails, or an image and a mask run over different subsets hand a case
the same copies.

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

- **shared read pass**: every copy whose stages after the marker are per-voxel
  rides *one* read of the case: each slab is decompressed once, each copy applies
  its draw to it and writes into its own stream. The marginal cost of a copy is
  its draw and its write, no reads at all.
- **own pass**: a draw that reads somewhere other than its target slab (a
  rotation, a halo) has its own read geometry, so it sweeps its own pass. The
  plan names the stage that decided it.
- **WHOLE-VOLUME**: the copy's chain cannot stream at all; the shared part is
  still assembled only once for the case.

Which regime a copy takes is decided by the draws after the marker, and the
draws are not equal:

| Draw | Reads | Regime |
|---|---|---|
| `Brightness`, `Contrast`, `LumaFlip`, `HUE`, `Saturation` | its own voxel | shared read pass |
| `Noise`, `CutOUT` | its own voxel (the field and the box are functions of the voxel's position) | shared read pass |
| `Translate` | a halo around the slab | own pass |
| `Flip` (not a vector field), `Permute`, `Rotate` with `is_quarter: true` | a permutation of the volume | own pass |
| `Rotate` (free angle), `Scale` | the source box each region maps to (a slab of a rotated volume pulls a wide band, which the plan prices) | own pass |
| `Elastix`, `Mask`, `Foreign`, `Flip` of a vector field | the whole volume | WHOLE-VOLUME |

Eight copies of `Brightness` or `Noise` are one read; eight copies of a free
`Rotate` are eight bounded passes, each re-reading the band its slabs pull.
When the shared prefix is expensive (a resample, a warp), put a `Save` before
the `Expand`: it is materialized once and every copy reads the cache.

Resume is **per copy**: a copy whose entry exists is skipped, so an interrupted
run picks up at the copy it stopped on, and because the draws come from
`manual_seed` rather than from the order a generator was consumed in, the copies
it keeps are the copies it would have written.

## Reproducibility across machines

What the run writes is a function of the config and the data, with one
machine-dependent quantity, and it is worth knowing where it can and cannot
show.

- **The draws.** A case's `Expand` copies are keyed by `manual_seed`, the
  case's name and the draw's rank in the chain: nothing else. The same case
  gets the same copies on another machine, under `--cpu 1` or `--cpu 8`, in a
  `subset` or in the full cohort, and in a rerun that resumes. `Noise` draws
  its field's generator seed with the copy, so it holds too.
- **The slab height.** A streamed case is swept in slabs of full planes, 64
  rows by default; a `memory_budget` can only lower that, and an `auto`
  budget is read from the machine (its RAM, or the cgroup it runs in). So the
  slab height depends on the machine, and with it the OME-Zarr chunk layout
  above and the number of regions the log counts.
- **The bytes.** For a chain of pointwise, halo, orientation and crop stages,
  and for an axis-aligned `Resample` (a change of spacing, a change of shape),
  the written values are identical whatever the slab height: streamed at 8
  rows or at 64, and equal to the case assembled whole. Only a `Resample`
  through a map that does not factorise into axes, a rotation or a stored
  displacement field, computes its interpolation weights per slab, and there a
  shorter slab differs from a taller one by about `1e-5` of the data's range.
  That is what the plan's note says when the budget lowers the slab height
  below the default; it is printed for the run, so read it against your chain:
  it names a possibility, and a chain with no such resample is exact. How a
  streamed result compares to a whole-volume one stage by stage is in
  {doc}`../concepts/streaming`.

## Statistics a stage can ask for

A stage that needs a whole-volume figure declares it rather than computing it:
`GLOBAL_STAT` with the keys it reads. The plan only checks the source can serve
them (every store can); the rank reads them at the case's first data access, by
scanning the stored entry without materialising it (a store without bounded
reads, a gzipped NIfTI, is decoded once for it), and the stage is then an
ordinary value map, so it streams. A case the plan LOADs computes them on the
volume it holds. That is how `Normalize` and `Standardize` work.

Two grains are available, and they come from the **same single pass**:

| Keys | What they are |
| --- | --- |
| `Min` `Max` `Mean` `Std` | the figure over the whole volume, every channel pooled |
| `MinPerChannel` `MaxPerChannel` `MeanPerChannel` `StdPerChannel` | one figure per channel |

The per-channel form exists because some quantities have no single mean. The
spatial mean of a displacement field is a **translation**: it has one part per
component, and pooling them into one number describes nothing.

A statistic is the STORED volume's. It is still the stage's own input only when
every stage before it preserves statistics: a reorientation does, a `Clip` does
not. `Clip` then `Normalize` therefore takes the whole-volume path, and the plan
says so. Reorder the chain, or cut it with a `Save`.

Handed the whole volume anyway (a chain that fell back for another reason) a
stage should take the statistic from the tensor in hand and record it, so both
paths leave the same state behind.

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

**Declare honestly, that is the one rule.** `patch_locality` is a promise about
what your `__call__` reads, and nothing verifies it against the code. Declaring
`POINTWISE` while reading your neighbours produces seams at every slab boundary:
a plausible, wrong image, with no error. When unsure, declare nothing: the
default is `WHOLE_VOLUME`, which is always correct and merely slower. The plan
will tell you that is what you got.

Two details the example carries on purpose. The internal padding must be the one
the whole-volume path would apply (here `avg_pool3d`'s zero padding, which the
halo never reaches except at the real volume border): a `reflect` pad computed
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
| another grid, or the same one at another density | `REGRID` | `stream_region_source()` and `stream_region()` |
| whole-volume Min/Max/Mean/Std | `GLOBAL_STAT`, with `stat_keys` | nothing |
| the same, per component | `GLOBAL_STAT`, with `MinPerChannel`/`MaxPerChannel`/`MeanPerChannel`/`StdPerChannel` | nothing |
| genuinely the whole volume | nothing (the default) | nothing |

A transform that does not subclass `Transform` still works (it is wrapped
automatically), but it is treated as `WHOLE_VOLUME`.

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
