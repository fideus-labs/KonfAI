# Patch streaming

When a preprocessing chain allows it, KonfAI reads each patch's source region
straight from disk and never materializes the volume. This page explains what
decides whether that happens, and what you control.

Streaming is **derived, not configured**. Nothing in YAML asks for it. KonfAI
reads your preprocessing chain, works out whether each patch's answer can be
computed from a bounded region of the file, and streams when it can. When it
cannot, it loads the volume. Both paths give the same patches; they differ in
memory and speed.

A 16 GiB uncompressed `.mha` at patch 64³, batch 2, two workers, cache off,
under an 8 GiB memory cap trains at a peak anonymous RSS of **0.46 GiB**, stable
across epochs, with VRAM equal to one batch.

## The three regimes

Which one a case takes follows from `use_cache` and from the chain itself.

| Regime | When | Memory held |
| --- | --- | --- |
| **Cache** | `use_cache: true` | every case, resident for the whole run |
| **Stream** | `use_cache: false` and the chain is streamable | one patch |
| **Buffer** | `use_cache: false` and the chain is not streamable | a FIFO of `batch_size + 1` cases (or `shuffle_window` cases, whichever is larger) |

**The cache wins over streaming.** `use_cache: true` preloads every case, and a
preloaded case is cut from the resident volume even when its chain would stream.
Set `use_cache: false` to get the streaming path at all.

Stream and buffer coexist inside one run. The decision is made per case *and*
per augmented copy, not once per dataset, so a chain that streams for one
augmentation draw may load the volume for the next.

`use_cache` is a training key: only `Trainer:` accepts it. Prediction declares
`false` and evaluation declares `true`. A `memory_budget` overrides that
declared value in every workflow — see below.

## What you control

Three keys under `Dataset:`:

| Key | Default | Where | Effect |
| --- | --- | --- | --- |
| `use_cache` | `true` | `Trainer:` | `false` opens the stream/buffer path. |
| `memory_budget` | `null` | all three workflows | Derives `use_cache` from the dataset's size. |
| `subset.shuffle_window` | `null` | `Trainer:` | Bounds how many cases stay resident on the buffer path. |

All three are listed in {doc}`../config_guide/training`.

`memory_budget` compares the estimated per-rank dataset size against the budget
and sets `use_cache` accordingly, printing its decision once. A bare number is
GiB (`24` means 24 GiB), a string may name its unit (`"24GB"`, `"32 GiB"`,
`"512mb"`), and `"auto"` offers 80% of the detected node memory, cgroup limit
included, divided by the ranks sharing it. `null` leaves `use_cache` exactly as
declared.

`memory_budget` replaces the declared `use_cache`, in both directions and in
every workflow. A prediction under `memory_budget: auto` caches its dataset
whenever it fits the budget; an evaluation under a budget smaller than its
dataset does not cache. Set the budget on the workflow you mean to bound.

To take the stream/buffer path on every case, declare `use_cache` and leave the
budget out:

```yaml
Dataset:
  use_cache: false
```

To let the dataset's size decide, declare a budget. It settles `use_cache`
whatever the config says, so the two keys together are not a way to force
streaming on a dataset that fits:

```yaml
Dataset:
  memory_budget: auto
```

```{warning}
`memory_budget` is an **estimate, not a guarantee**. The figure is computed from
file headers alone — `prod(shape) × 4 bytes` per source group — so it ignores
the dtype you actually stored, any size-changing transform, the augmented copies
of each case, and the transient peak while a case is being built. Treat it as a
coarse switch, not as a memory bound.
```

```{warning}
Do not combine `shuffle_window` with multi-GPU training. Each rank sizes its
sampler from its own shard, so ranks can disagree on the number of batches and
the run hangs in NCCL.
```

## What decides whether a chain streams

Every transform declares how its output at one voxel depends on its input. That
declaration — its **patch locality** — is what the dispatcher reads to work out
which region of the file a patch needs.

| Kind | Meaning | What KonfAI reads |
| --- | --- | --- |
| `POINTWISE` | the voxel depends only on itself | the exact patch |
| `HALO` | a bounded neighbourhood of radius `halo` | the patch enlarged by `halo`, cropped after |
| `ORIENTATION` | flip or permute | the index-remapped region |
| `CROP` | the source region is the target translated | the region — reading it *is* the answer |
| `GLOBAL_STAT` | needs whole-volume `Min`/`Max`/`Mean`/`Std` | the statistic once from disk, then the exact patch |
| `RESCALE` | resample | the region through the scale mapping, plus an interpolation halo |
| `WHOLE_VOLUME` | genuinely needs everything | the volume — the fallback |

`WHOLE_VOLUME` is the default. A transform that declares nothing loads the
volume and is correct.

A chain streams when it has the shape

```text
[pointwise*] [at most one region stage] [pointwise*]
```

where `GLOBAL_STAT` counts as pointwise, and the region stages are `HALO`,
`ORIENTATION`, `CROP`, and `RESCALE`. The chain planned is the group's
`transforms` followed by the copy's own augmentation draw — one list, so a
region transform and a region augmentation are two regions.

Six conditions reject streaming:

1. any `WHOLE_VOLUME` declaration;
2. more than one region stage;
3. a halo wider than half the read extent on any axis;
4. a `GLOBAL_STAT` preceded by a stage that does not preserve statistics;
5. a `GLOBAL_STAT` whose statistic cannot be read from disk;
6. a `RESCALE` on a case with no `Spacing`.

### Why `[Clip(-200, 400), Standardize()]` does not stream

This is rule 4.

`Standardize` is `GLOBAL_STAT`: it needs the volume's `Mean` and `Std`, which
KonfAI reads once from disk. The statistic on disk is the **stored** volume's,
while `Standardize`'s input here is the *clipped* volume. `Clip` maps values, so
the two differ. Streaming this chain would standardize every patch by the
pre-`Clip` statistic, so KonfAI loads the volume instead.

Swap the first stage and the chain streams:

```yaml
transforms:
  Canonical: {}
  Normalize: {}
```

`Canonical` is `ORIENTATION`, the one kind that preserves statistics: a flip or a
permute moves voxels without changing any of them, so the multiset of values —
and every statistic over it — is untouched. The stored volume's `Min` and `Max`
are `Normalize`'s own input, and the chain streams.

The rule generalizes: a `GLOBAL_STAT` stage streams only when **every** stage
before it preserves statistics. `TensorCast` is the one built-in that declares
this for itself — casting to a float dtype changes no value, so
`[TensorCast('float32'), Standardize()]` streams, while
`[TensorCast('uint8'), Standardize()]` does not.

### Why two region stages do not stream

A region stage is a rewrite of *which* source voxels a target patch needs. One
such rewrite composes with the read: KonfAI maps the patch back through it and
asks the file for the result. A second stage's region is expressed in the first
stage's output space, which does not exist on disk. `[Dilate(1), Gradient()]` is
two halos; `[Canonical(), Permute('2|1|0')]` is two reorientations. Both load the
volume.

Rule 3 is about cost, not correctness. Every patch pays its halo on every side,
so streaming reads `prod(1 + 2·halo/extent)` times the case's bytes, where the
extent is the patch or the volume, whichever is smaller. At half the extent that
is 8× in 3D, against the single load streaming avoids. At patch 8, `Dilate(4)`
streams and `Dilate(5)` does not.

## What each built-in declares

| Kind | Transforms |
| --- | --- |
| `POINTWISE` | `Argmax`, `Softmax`, `Sum` (all `dim=0`), `OneHot`, `MergeLabels`, `FlatLabel`, `SelectLabel`, `UnNormalize`, `Percentage`, `Variance`, `StandardDeviation`, `SegmentationDisagreement`, `TensorCast`, `Clip` with fixed bounds, `Standardize` with both `mean` and `std`, `Dilate(0)` |
| `GLOBAL_STAT` | `Normalize`, `Standardize`, `Clip` with `'min'`/`'max'` bounds |
| `HALO` | `Dilate(n>0)`, `Gradient` |
| `ORIENTATION` | `Flip`, `Permute`, `Canonical` (only on axis-aligned direction cosines) |
| `CROP` | `Crop` (only once its box is on the case) |
| `RESCALE` | `ResampleToResolution`, `ResampleToShape` |

Augmentations declare per **(case, draw)** — two copies of the same case can
answer differently. `Permute`, `Flip` (when `vector_field: false`), and `Rotate`
(when the draw is a quarter turn) are `ORIENTATION`; `ColorTransform` and its
subclasses are `POINTWISE`; `Translate` is `HALO`. `Scale`, `Noise`, `CutOUT`,
`Mask`, and `Elastix` load the volume.

These transforms load the volume because their answer needs it:

- `Mask`, and `Clip`/`Standardize` with a `mask`, read a second full volume that
  a patch cannot locate itself in.
- `Clip` with percentile bounds needs the whole histogram, as does
  `HistogramMatching`.
- `Save` writes the whole preprocessed volume. A `Save` whose cache already
  exists becomes the streaming source instead, and only the transforms after it
  are planned.
- `Argmax`, `Softmax`, and `Sum` over a spatial `dim` reduce across the whole
  extent.
- `Canonical` on an oblique direction resamples.

## Reading regions from disk

Streaming is only as cheap as the format underneath it.

| Backend | Serves a disk region |
| --- | --- |
| HDF5 | yes, natively |
| OME-Zarr | yes, chunked (`level` selects the pyramid resolution) |
| DICOM | yes, per slice |
| SimpleITK | only uncompressed MetaImage and non-gzipped NIfTI |

A format that cannot serve a region — NRRD, or any compressed file — still
returns the right voxels: it decodes the whole volume for every patch. That
costs speed, never correctness, and KonfAI warns once per format with the
remedy. Convert such datasets to OME-Zarr, HDF5, or uncompressed `.mha`/`.nii`.

The same table governs the `GLOBAL_STAT` seed. On a backend that serves regions,
the statistic is a chunked running pass in float64 — slab by slab, never the
whole volume in RAM. On a format that cannot serve one, reading the statistic
decodes the volume in one go, once per case.

## `transforms` vs `patch_transforms`

`transforms` runs once on the case; `patch_transforms` runs on each patch after
it is cut. Only `POINTWISE` and `GLOBAL_STAT` are admissible per patch, and
KonfAI rejects anything else at config time with the reason and the remedy —
move it to `transforms`.

A per-patch `GLOBAL_STAT` means *derive the statistic from this patch*. To
standardize patches by the volume's statistic instead, pair a case-level
`Standardize(lazy=True)` with a per-patch `Standardize()`.

## Custom transforms

Subclass `konfai.data.transform.Transform` and you inherit the safe default:
your transform reports `WHOLE_VOLUME`, KonfAI loads the volume, and your patches
are correct without you knowing streaming exists.

Subclassing is **required**. A class that merely looks like a transform fails
inside the planner with a bare `AttributeError` rather than a KonfAI error with a
remedy.

To opt in, override `patch_locality` under three rules:

- **Read-only.** Never write to `cache_attribute`. The declaration is made once
  for the whole case; a write is handed a private copy and lost.
- **No I/O.** Read the attribute in hand and nothing else. Whether the
  declaration can be honoured is the dispatcher's call.
- **Total.** Answer for any case, including one with no metadata. A missing key
  returns `WHOLE_VOLUME`; it never raises.

`ORIENTATION` and `CROP` must also implement `stream_region_source`, mapping a
target patch to the source region. Declaring a region kind without it raises a
`TransformError` on the first patch read. `HALO` and `RESCALE` need no remap; the
dispatcher derives their regions.

## Equivalence

A streamed patch is byte-identical to the same patch cut from the loaded volume
— border padding included — for `POINTWISE`, `HALO`, `ORIENTATION`, and `CROP`.

Two cases carry a bounded numerical difference instead.

A `GLOBAL_STAT` stage seeded from `Mean`/`Std` — `Standardize` — reads its
statistic from a numpy pass over the file while the whole-volume path recomputes
it in torch. Same values, different summation order, so a standardized voxel may
land a few float32 ulp away. A stage seeded from `Min`/`Max` — `Normalize`,
`Clip('min', 'max')` — is byte-identical: a min and a max have no summation
order to disagree on.

A streamed `RESCALE` computes interpolation weights in the sub-region's
coordinate frame, so the deviation scales with the local gradient rather than
with the voxel's own value: within a few ulp of the volume's peak on float
volumes, within 1 LSB on integer ones. A nearest-neighbour resample — what a
`uint8` label volume gets — uses no weights and stays exact. The `Translate`
augmentation interpolates on the same terms as `RESCALE`.

## Next steps

- {doc}`datasets` — the grouped layout, `groups_src`, and where patching sits
- {doc}`../config_guide/training` — `use_cache`, `memory_budget`, and `shuffle_window` in a full config
- {doc}`../reference/components/storage-backends` — format tokens and the per-backend APIs
