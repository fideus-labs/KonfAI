# Working out of core

A case larger than your RAM is a normal case. This page is how you set one up and
tune it; [Patch streaming](#patch-streaming), below, is what the engine does with it.

<figure class="kf-visual kf-visual--wide">
  <a class="kf-visual-frame" href="../_static/gallery/scale-omezarr.webp" aria-label="Open the OME-Zarr regional-read figure at full resolution">
    <picture>
      <source media="(max-width: 640px)" srcset="../_static/gallery/scale-omezarr-mobile.webp" width="500" height="2147">
      <img src="../_static/gallery/scale-omezarr.webp" alt="A real ExaSPIM OME-Zarr pyramid showing a coarse overview, the selected native-resolution source region with chunk boundaries, and its matching mask." width="1530" height="900" fetchpriority="high" decoding="async">
    </picture>
  </a>
  <figcaption>
    <span class="kf-visual-copy">
      <strong>One bounded request through KonfAI's OME-Zarr backend.</strong>
      <span class="kf-visual-meta">1.98 GiB source volume · 0.50 MiB native image window · matching mask region</span>
    </span>
    <a class="kf-visual-inspect" href="../_static/gallery/scale-omezarr.webp">Inspect 1530 × 900 <span aria-hidden="true">↗</span></a>
  </figcaption>
</figure>

A real read from AIND ExaSPIM specimen `822175` (CC BY 4.0), not a synthetic
volume. Level 0 holds `449 × 1331 × 1775` uint16 voxels, 1.98 GiB uncompressed,
in `256³` chunks. The figure reads one coarse level-1 plane to locate the field
of view, then asks for one 512² region at native resolution and the identical
region of the mask: **0.50 MiB materialised** out of 1.98 GiB.

## A dataset that streams

```yaml
Dataset:
  dataset_filenames:
    - ./Dataset:omezarr
  memory_budget: auto
  batch_size: 2
  num_workers: 4
  Patch:
    patch_size: [64, 128, 128]
    overlap: 16
```

Prediction, evaluation and transform stream by default. Training caches unless
the dataset exceeds `memory_budget`, so setting a budget below your dataset's
size is how you force the streaming path there.

The format decides how cheap it is. HDF5 and OME-Zarr serve regions natively,
DICOM reads per slice, and SimpleITK serves them for uncompressed MetaImage and
non-gzipped NIfTI. Anything compressed still returns correct patches, but decodes
the whole volume for each one. The token for a DICOM series is `./Dataset:dicom`;
`dcm` means a single file through SimpleITK, which is a different backend.

## OME-Zarr layout

One store per case and group:

```text
Dataset/
├── CASE_001/
│   └── CT.ome.zarr/
└── CASE_002/
    └── CT.ome.zarr/
```

The selector can name a pyramid level, `omezarr@1`. KonfAI reads metadata through
`get_infos()` and touches only the chunks that intersect the window. Chunk shape
matters: chunks much larger than your patches cost I/O you throw away, very small
ones cost metadata and decompression overhead.

Reproduce the figure above with a local ExaSPIM store:

```bash
pixi run --environment dev python docs/scripts/generate_scale_gallery.py \
  --root /path/to/ExaSPIM_Template/Data/Dataset_prepared \
  --case 822175
```

The generator only calls `get_infos()` and `read_data_slice()`, never
`read_data()`.

The same specimen family is public, without a download: the fused volume of
specimen `822174` is
`s3://aind-open-data/exaSPIM_822174_2026-04-28_12-29-55_processed_2026-07-09_03-49-09/fusion2halves/SPIM.ome.zarr`
(513 × 1331 × 1775 uint16, 256³ chunks, four levels, CC BY 4.0). With
`pip install konfai[s3]` and `FSSPEC_S3_ANON=true`, a `dataset_filenames` entry
naming the asset (the parent of `fusion2halves/`) with the `:omezarr` suffix
streams it region by region as case `fusion2halves`, group `SPIM`, and the
large-images notebook on {doc}`../examples/index` reads one coarse level and one native
window of it.

## Tune in this order

1. `memory_budget` first: `auto` decides from the dataset's size, an explicit
   value below it forces streaming. Give it as much as the machine can spare:
   the budget buys read locality, and a region that shrinks re-reads and
   re-decodes chunks a taller one would have read once. On a 513x1331x1776
   uint16 OME-Zarr resampled on one GPU, the whole run costs 5.0 s at 4 GiB,
   10.9 s at 1 GiB, 22.8 s at 512 MiB and 49.1 s at 256 MiB, and the whole
   difference is the wait on reads (0.2 s to 42.6 s); the chain itself stays
   between 3.1 and 4.6 s. A budget pays where reading costs, so it pays most on
   a compressed, remote or cache-cold store, and least on a small dataset the
   operating system already holds in its page cache. The sweep spends it by
   measurement: the first region is priced against half of the budget, and
   each region that holds under a third of it doubles the next, up to eight chunk
   rows, so the height settles on what the run actually holds rather than on a
   prediction of it.
2. `patch_size`: leave an axis at `0` and KonfAI sizes it, taking the whole
   volume when it fits and shrinking on OOM. Otherwise pin the largest size your
   model and context need.
3. `batch_size: 1` to start, raise it while watching throughput and VRAM.
4. Overlap only when the borders need it: more overlap is more reads and more
   forward passes.
5. `num_workers` up until storage or CPU preprocessing saturates.
6. `pin_memory: true` then measure; it locks host memory and is not always
   faster. It buys the upload a real DMA: a 32 MiB `int64` target reaches the
   device in 1.1 ms instead of 2.9 ms of compute-stream time. The epoch rarely
   follows, because the step is bound by the device, not by the copy. On eight
   `256³` cases, `128³` patches, batch 2, streamed over four workers, the median
   epoch went 5.2 s to 5.1 s, inside a 4.9 to 5.5 s spread, for 430 MiB more
   resident and page-locked; with the dataset cached in RAM the loader pins
   inline and the epoch did not move at all (4.2 s either way).
7. `prefetch_factor` only with worker processes, and count the extra batches in
   RAM.

## Two kinds of patching

They solve different problems and stack:

- `Dataset.Patch` decides what the dataloader hands the model. It bounds
  source, preprocessing, batch and forward memory. Its keys: `patch_size`
  (default `[128, 128, 128]`), `overlap` (`null` picks a 20 % default; a voxel
  count, a fraction, a percent string or a per-axis list), `pad_value` (`null`
  pads with the data's minimum) and `extend_slice` (2.5-D context, only when
  `patch_size[0] == 1`). A `0` in `patch_size` is a free axis the framework
  sizes by measurement: see {doc}`../config_guide/prediction` and
  {doc}`../config_guide/training`.
- `Model.ModelPatch` splits again inside the network, for a heavy subgraph, a
  2D or 2.5D model inside a 3D workflow, or patch-level supervision. Its
  `patch_combine` blender reassembles the overlaps: `Mean`, `Cosinus`, `Trim`
  or `Gaussian` (nnU-Net-style importance weighting).

Start with dataset patching. Add `ModelPatch` only when a stage of the network
has its own memory or dimensionality requirement; the `examples/Synthesis` GAN
variant is the clearest case, a 3D chunk for the whole GAN and 2D slices inside
the generator. How the graph names its outputs under either is on
{doc}`../reference/components/models`.

Whichever level cuts the patches, the `Accumulator` reassembles them with
overlap blending and corrects the border voxels fewer patches covered. Patch
read order must match write order. For PREDICTION and EVALUATION every patch
of a case stays on the same DDP rank, where the volume is reassembled; for
TRAIN the shards are padded to the same length so every rank runs the same
number of backward passes.

## When the output is the peak

Prediction keeps its reassembly accumulator on the GPU when it fits and falls
back to host memory when it does not. Streaming handles the rest on its own:
slabs are written as they complete, and geometry inverses stream with them, so a
full-resolution multi-class output never exists whole. That is the case where
streaming pays most, and [Patch streaming](#patch-streaming) covers what it can and
cannot honour.

## When it does not do what you expected

- **RSS still grows with the case**: the planner took the whole-volume path for
  that case or that augmentation draw. Check the locality rules, or run the
  unsupported stage once through `Save` and stream from the materialised
  dataset. The `Save` itself streams: its cache is written slab by slab on first
  access.
- **OME-Zarr is slow**: look at chunk shape, compression, pyramid level, worker
  count and overlap. More workers do not always help on remote storage.
- **Seams in the output**: add overlap and pick a compatible `patch_combine`. If
  you have custom code in the loop, check it has not changed patch ordering.
- **CUDA OOM after the forwards finish**: the volume-sized output or the
  reduction is the peak. Cut output channels, TTA or ensemble size, or let the
  predictor accumulate on the host.

A successful run does not prove streaming happened, since the fallback is
designed to stay correct. Measure peak RSS on a representative case, and hold the
model, patch size, overlap, TTA, batch size, workers and hardware fixed when you
compare.

## Patch streaming

**KonfAI works out of core.** A case never has to fit in RAM: each patch's source
region is read straight from the file, and the result is written slab by slab as
it completes. Neither the input nor the output is ever held whole. A 16 GiB
uncompressed volume trains at a peak of **0.46 GiB of host RAM**, stable across
epochs, with VRAM equal to one batch. The bounded-memory claim is reproducible
with one command, `python benchmarks/bench_streaming.py --gib 16 --budget 1`:
see [Reproducing the numbers](#reproducing-the-numbers).

That is not only a memory story. Running published models through KonfAI, on the
same weights and the same card, against their reference implementations:

| Model, large case (512 × 512 × 531) | Time | Peak host RAM | Peak VRAM |
| --- | --- | --- | --- |
| MRSegmentator, KonfAI | **120 s** | **6.2 GB** | 16.7 GB |
| MRSegmentator, original | 192 s | 37.5 GB | 14.6 GB |
| TotalSegmentator, KonfAI | **314 s** | **19.3 GB** | **10.4 GB** |
| TotalSegmentator, original | 459 s | 51.8 GB | 23.3 GB |

Across sizes that is 1.5 to 3.6× faster with 1.4 to 6× less host RAM, and on the
large case KonfAI bounds VRAM where the original nears the card limit. The full
tables, including small and medium cases, are in the
[MRSegmentator](https://github.com/fideus-labs/KonfAI/tree/main/apps/mrsegmentator)
and
[TotalSegmentator](https://github.com/fideus-labs/KonfAI/tree/main/apps/totalsegmentator)
app pages; the measurement protocol behind every number is
[below](#reproducing-the-numbers).

Nothing in YAML asks for any of it. KonfAI reads your preprocessing chain, works
out whether a patch's answer can be computed from a bounded region of the file,
and streams when it can. When it cannot it loads the volume, and the patches are
the same either way. Only memory and speed change.

```{mermaid}
flowchart TB
    F[(the file on disk)]:::disk
    subgraph one[" per patch "]
        direction TB
        RR[read one source region]:::step
        CH[run the chain on it]:::step
        M[[model]]:::model
        RR --> CH --> M
    end
    F -- "only the region<br/>the patch needs" --> RR
    M --> AC[accumulate,<br/>overlap blended]:::step
    AC --> WS[write slab by slab]:::step
    WS --> OUT[(the result on disk)]:::disk

```

Neither end is ever whole in memory: the read is bounded by what the patch needs,
and the write lands as each slab completes.

### The three regimes

| Regime | When | Memory held |
| --- | --- | --- |
| **Cache** | the training default | every case, resident for the whole run |
| **Stream** | the predict/eval/transform default, or a budget the dataset exceeds, chain streamable | one patch |
| **Buffer** | same triggers, chain not streamable | a FIFO of `batch_size + 1` cases, or `shuffle_window`, whichever is larger |

The decision is per case **and** per augmented copy, so stream and buffer coexist
in one run: a chain that streams for one draw may load the volume for the next.
A cached case is always cut from the resident volume, even when its chain would
stream.

Two keys under `Dataset:` move the needle:

| Key | Default | Effect |
| --- | --- | --- |
| `memory_budget` | `auto` | Derives the regime from the dataset's size. Under `Transformer:` it is a per-rank ceiling that can refuse a run. |
| `shuffle_window` | `null` | Bounds how many cases stay resident on the buffer path. `Trainer:` only. |

A bare number means GiB (`24`), a string may carry its unit (`"24GB"`,
`"512mb"`), and `auto` offers 80% of the detected node memory, cgroup limit
included, divided by the ranks sharing it. Declaring a budget **below** your
dataset's size is how you force the streaming path in training. Prediction,
evaluation and transform read each case once and always stream; there the budget
sizes the disjoint patches a too-large case is cut into instead.

The figure is an estimate, computed from file headers (`prod(shape) × 4 bytes`
per group), so it ignores the dtype you stored, size-changing transforms and
augmented copies. It is a switch, not an allocator limit.

It also bounds only the buffers the pipeline holds for your data, the OME-Zarr
decoded-chunk cache among them: a third of the budget, on every rank of every
workflow. Under a 768 MiB budget that third is below the 256 MiB the cache is
worth, where a region touching more chunks than it holds decodes them again; the
TRANSFORM plan says so in its header. The interpreter, torch and its CUDA
context, the model and each worker's own working set sit outside it, so the peak
RSS a run reports is the budget plus a floor that does not move when the budget
does: lowering the budget lowers the peak by roughly what you took off, never to
the budget itself.

### What decides whether a chain streams

Every transform declares how its output at one voxel depends on its input. That
declaration, its **patch locality**, is what the dispatcher reads to work out
which region of the file a patch needs.

| Kind | Meaning | What KonfAI reads |
| --- | --- | --- |
| `POINTWISE` | the voxel depends only on itself | the exact patch |
| `HALO` | a bounded neighbourhood of radius `halo` | the patch enlarged by `halo`, cropped after |
| `ORIENTATION` | flip or permute | the index-remapped region |
| `CROP` | the source region is the target translated | the region, and reading it *is* the answer |
| `REGRID` | a change of grid | the mapped region, plus the interpolation taps |
| `GLOBAL_STAT` | needs whole-volume `Min`/`Max`/`Mean`/`Std` | the statistic once from disk, then the exact patch |
| `SLAB` | a value map plus a side effect that needs the slab's place in the volume | nothing: read-side it falls back to `WHOLE_VOLUME`, it streams on the write side (`InferenceStack`) |
| `WHOLE_VOLUME` | genuinely needs everything | the volume, the fallback |

A chain streams when every stage is pointwise or a region kind (`HALO`,
`ORIENTATION`, `CROP`, `REGRID`), with `GLOBAL_STAT` counting as pointwise.
Region stages **compose**, in any number: each stage's region pulls through the
one before it, down to a single bounded read. `[Dilate(1), Gradient()]` is two
halos that add; `[Canonical(), Permute('2|1|0')]` is two remaps that pull through
each other. The planned chain is the group's `transforms` followed by the copy's
augmentation draw, one list, so a region transform and a region augmentation
compose exactly like two transforms.

Six things send a chain back to the whole volume:

1. any `WHOLE_VOLUME` or `SLAB` declaration;
2. a halo wider than half the read extent on any axis;
3. a `GLOBAL_STAT` preceded by a stage that changes values;
4. a `GLOBAL_STAT` whose statistic cannot be read from disk;
5. a `REGRID` that cannot size the region it reads (no geometry, no bound);
6. a chain whose folded shapes do not land on the target grid.

Rule 3 is the one that surprises people. `[Clip(-200, 400), Standardize()]` does
not stream: the statistic on disk belongs to the **stored** volume, while
`Standardize`'s input here is the clipped one, and `Clip` moves values. Streaming
would standardize every patch by the wrong statistic, so KonfAI loads instead.
`[Canonical(), Normalize()]` does stream, because a reorientation is the one kind
that preserves every statistic: it moves voxels without changing any of them.
`TensorCast` declares the same for itself, but only for a target that holds every
value: `float32` streams, `uint8` and `float16` do not.

Rule 2 is about cost, not correctness. Every patch pays its halo on each side, so
streaming reads `prod(1 + 2·halo/extent)` times the case's bytes. At half the
extent that is 8× in 3D, against the single load streaming was avoiding. At patch
8, `Dilate(4)` streams and `Dilate(5)` does not.

### What each built-in declares

| Kind | Transforms |
| --- | --- |
| `POINTWISE` | `Argmax`, `Softmax`, `Sum` (all with `dim=0`), `OneHot`, `MergeLabels`, `FlatLabel`, `SelectLabel`, `UnNormalize`, `Percentage`, `Variance`, `StandardDeviation`, `SegmentationDisagreement`, `Magnitude`, `TensorCast` to a value-preserving target (`float32`, `float64`), `Mask`, `Clip` with fixed bounds, `Standardize` with both `mean` and `std`, `Dilate(0)` |
| `GLOBAL_STAT` | `Normalize`, `Standardize`, `Clip` with `'min'`/`'max'` bounds, `Statistics` |
| `HALO` | `Dilate(n>0)`, `Gradient` |
| `ORIENTATION` | `Flip`, `Permute`, `Canonical` on axis-aligned direction cosines |
| `CROP` | `Crop`, once its box is on the case |
| `REGRID` | `Resample`; `Padding` in every mode (`constant` is a translation into a filled, larger volume; `reflect` and `replicate` pull the border they mirror, which the region's own window carries) |

Augmentations declare per **(case, draw)**, so two copies of one case can answer
differently. `Permute`, `Flip` (with `vector_field: false`) and `Rotate` on a
quarter turn are `ORIENTATION`; `ColorTransform` and its subclasses are
`POINTWISE`; `Translate` is `HALO`. A free-angle `Rotate` and `Scale` are
`REGRID`, pulling their own window through the affine; `Noise` and `CutOUT` are
`POINTWISE`, their field and their box being functions of the voxel's position in
the whole volume. The `Mask` DRAW and `Elastix` load the volume (the draw's output grid is the
mask's own, which is already resident); the `Mask` TRANSFORM above is pointwise and reads its
mask by region, and declares those reads to the decoded-chunk cache ahead of a sweep or of a
case's patches, as the reader declares its own.

The transforms that load the volume do so because their answer needs it:
`Clip` and `Standardize` under a `mask` read a second full volume a patch cannot
locate itself in; `Clip` with percentile bounds and `HistogramMatching` need the
whole histogram; `Argmax`, `Softmax` and `Sum` over a spatial `dim` reduce across
the extent; `Canonical` on an oblique direction resamples.

`Save` is the useful exception. A `Save` whose cache exists becomes the streaming
source, and only the transforms after it are planned. A `Save` whose cache is
missing is **materialized slab by slab** when the transforms before it stream:
each slab is read through the composed region plan and region-written, the entry
appears only once complete, then the case streams from it. That runs the prefix
once instead of once per patch per epoch, and it lets a statistic seed after a
value-changing stage: `[Clip, Save, Standardize]` streams where
`[Clip, Standardize]` cannot. Only a `Save` fed by an unstreamable prefix, or
writing to a format without region writes, still loads the volume.

A custom transform inherits `WHOLE_VOLUME` too, and is correct without knowing
streaming exists. {doc}`custom-models` has the contract for
declaring a locality yourself.

### Reading regions from disk

Streaming is only as cheap as the format underneath.

| Backend | Serves a disk region |
| --- | --- |
| HDF5 | yes, natively |
| OME-Zarr | yes, chunked, `level` selects the pyramid resolution |
| DICOM | yes, per slice |
| SimpleITK | uncompressed MetaImage and non-gzipped NIfTI only |

A format that cannot serve a region still returns the right voxels: it decodes
the whole volume for every patch. That costs speed, never correctness, and KonfAI
warns once per format. Convert those datasets to OME-Zarr, HDF5, or uncompressed
`.mha`/`.nii`.

The same table governs the `GLOBAL_STAT` seed: on a backend that serves regions
the statistic is a chunked running pass in float64, never the whole volume in
RAM.

### The write side

The output streams too, and for the same reason: each slab is finalized and
written as soon as its patches complete, so a huge prediction at original
resolution never exists whole.

Geometry inverses **compose** on the way out. A `Canonical`, `Flip` or `Permute`
inverse remaps each slab to its written region; a `Padding` inverse crops it in
flight; a `spacing`/`shape` `Resample` inverse resamples back through a sliding
window. Chain any number, each pulling through the next. A masked finalize
(`Mask`) streams as well, reading only its aligned mask region per slab.

This matters most where the output *is* the peak. Resampling multi-class
probabilities back to a native grid is tens of GB whole, and a `combine: Concat`
ensemble multiplies it by the number of members; streamed, it is one window.

What streaming cannot honour splits instead: the pointwise prefix still streams
into a light buffer and the remaining stages run once on it. Four things keep the
whole-volume path: a TTA draw whose inverse moves the slab axis (a z-flip, a
z-moving permute), a case too light to be worth slab synchronization, a
non-voxel-local reduction, or a destination without region writes.
`KONFAI_STREAMED_WRITES=0` forces the whole-volume path globally, which is the
reference to compare against.

### `transforms` vs `patch_transforms`

`transforms` runs once on the case, `patch_transforms` on each patch after it is
cut. Only `POINTWISE` and `GLOBAL_STAT` are admissible per patch, and KonfAI
rejects anything else at config time with the remedy: move it to `transforms`.

A per-patch `GLOBAL_STAT` derives its statistic from that patch. To standardize
patches by the volume's statistic instead, pair a case-level
`Standardize(lazy=True)` with a per-patch `Standardize()`.

### How close is a streamed patch

Byte-identical to the same patch cut from the loaded volume, border padding
included, for `POINTWISE`, `HALO`, `ORIENTATION` and `CROP`.

Two cases carry a bounded difference. A `GLOBAL_STAT` seeded from `Mean`/`Std`
reads its statistic through a numpy pass while the whole-volume path recomputes
it in torch: same values, different summation order, so a voxel may land a few
float32 ulp away. Seeded from `Min`/`Max` it is exact, since a min has no
summation order to disagree on.

A streamed `REGRID` walks global float64 coordinates (`precision: exact`, the
default; `precision: fast` walks in float32 and the bounds below then no longer
hold), so a slab computes the very numbers the whole volume computes. On the host the blend is ITK's own
resampler on a window at its true origin, and on an axis-aligned volume streamed
equals whole **bit for bit**, whatever the map. Two things cost an ulp: on CUDA a
*linear* blend through a map that does not factorise (a rotation, a stored
field) goes through `grid_sample`, which normalises coordinates by the window it
is handed; and on oblique direction cosines a region's origin is one rounding
the whole volume never takes. Either way streamed and whole agree to about 1e-5
of the data's range, the deviation following the local gradient (within 1 LSB on
integer volumes). Nearest-neighbour, which is what a `uint8` label volume gets,
picks on the exact index; cubic walks its own corners; an axis-aligned change of
density is read one axis at a time on global coordinates: all three are
bit-identical everywhere.

The slab height follows the budget, so it can differ between machines; through
that same non-separable linear resample two runs of one chain under different
budgets then differ by the same ~1e-5, and the plan says so when the budget
lowers the height below the default. Everything else is independent of the
slabbing: the same chain writes the same bytes under any budget.

The same holds across slab heights. A `TRANSFORM` sweep cuts a case into slabs
whose height follows the memory budget, so it depends on the machine: the first
slab is priced against half of the budget, and each slab that holds under a
third of it doubles the next, up to eight chunk rows (64 rows, unchanged, without
a budget). A pointwise, halo,
orientation or crop chain and an axis-aligned `Resample` write the same bytes at
8 rows as at 64; only the non-separable linear resample above can differ, and by
that same 1e-5. An OME-Zarr store's chunk layout does follow the slab, so the
values are portable and the layout is not: see {doc}`../config_guide/transform`.

## Reproducing the numbers

Every performance number the documentation carries is meant to be re-runnable.
The tracked
[`benchmarks/`](https://github.com/fideus-labs/KonfAI/tree/main/benchmarks)
directory holds the harness; this section says what each script evidences and
under which protocol, so a published figure and your own re-run are compared on
the same footing.

### Protocol

- Wall time is the median of 3 runs after 1 warmup, on an otherwise idle
  machine.
- Host memory is the peak resident set of the whole process tree (`psutil`),
  sampled at 50 ms, so DataLoader workers and spawned ranks count.
- Device memory is `torch.cuda.max_memory_allocated()`, with the NVML
  per-process figure reported beside it when available; the two overlap and are
  never summed.
- Every report line carries the konfai/torch/SimpleITK versions, CPU model,
  GPU model, and the input's shape, dtype and checksum.

### The streaming claim

[Patch streaming](#patch-streaming) states that a streamable case never has to fit in
RAM: a 16 GiB uncompressed volume is transformed with peak host memory bounded
by the declared `memory_budget`, not by the volume. A chain that cannot stream
falls back to the whole volume, which TRANSFORM sizes against the same budget
and refuses when it does not fit. Reproduce the claim with one command from
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

### The app comparison tables

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

## Next steps

- {doc}`../reference/components/storage-backends`: backend capabilities,
  layouts, and reading a root from object storage.
- {doc}`../config_guide/prediction`: batching, TTA, ensembles, reductions,
  output writing.
- {doc}`../config_guide/transform`: the per-case plan a TRANSFORM run prints.
- {doc}`custom-models`: declaring a locality for a transform of your own.
- {doc}`../examples/index`: the large-images notebook, a public 2.4 GB
  OME-Zarr read where it lives.
