# Patch streaming

**KonfAI works out of core.** A case never has to fit in RAM: each patch's source
region is read straight from the file, and the result is written slab by slab as
it completes. Neither the input nor the output is ever held whole. A 16 GiB
uncompressed volume trains at a peak of **0.46 GiB of host RAM**, stable across
epochs, with VRAM equal to one batch.

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
app pages.

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

## The three regimes

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
decoded-chunk cache among them: a third of the budget, never more than the
budget, and never below a 256 MiB floor unless the budget itself is smaller,
set on every rank of every workflow. The
interpreter, torch and its CUDA context, the model and each worker's own
working set sit outside it, so the peak RSS a run reports is the budget plus a
floor that does not move when the budget does: lowering the budget lowers the
peak by roughly what you took off, never to the budget itself.

## What decides whether a chain streams

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

## What each built-in declares

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
mask by region.

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
streaming exists. {doc}`../reference/api/extension-points` has the contract for
declaring a locality yourself.

## Reading regions from disk

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

## The write side

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

## `transforms` vs `patch_transforms`

`transforms` runs once on the case, `patch_transforms` on each patch after it is
cut. Only `POINTWISE` and `GLOBAL_STAT` are admissible per patch, and KonfAI
rejects anything else at config time with the remedy: move it to `transforms`.

A per-patch `GLOBAL_STAT` derives its statistic from that patch. To standardize
patches by the volume's statistic instead, pair a case-level
`Standardize(lazy=True)` with a per-patch `Standardize()`.

## How close is a streamed patch

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
whose height follows the memory budget, so it depends on the machine (64 rows
without a budget, fewer under a tight or `auto` one). A pointwise, halo,
orientation or crop chain and an axis-aligned `Resample` write the same bytes at
8 rows as at 64; only the non-separable linear resample above can differ, and by
that same 1e-5. An OME-Zarr store's chunk layout does follow the slab, so the
values are portable and the layout is not: see {doc}`../config_guide/transform`.

## Next steps

- {doc}`../usage/large-images` to tune it: OME-Zarr chunks, patch size, workers.
- {doc}`../config_guide/training` for `memory_budget` and `shuffle_window` in a
  full config.
- {doc}`../reference/components/storage-backends` for format tokens and layouts.
- {doc}`../config_guide/transform` for the per-case plan a TRANSFORM run prints.
