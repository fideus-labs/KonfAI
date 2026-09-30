# Transform configuration

`TRANSFORM` prepares a dataset: it reads cases, runs a chain of stages on each one, and writes the
result. Its config lives under the `Transformer` root. If you never train anything, this is the only
page you need.

```yaml
Transformer:
  name: RESAMPLE_TO_ISO
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

This writes `./Out/<case>/CT_iso.ome.zarr/` for every case in `./Raw`. Cases are processed slab by slab,
so a volume never has to fit in memory.

## Running it

```bash
konfai TRANSFORM --config Transform.yml          # run
konfai TRANSFORM --config Transform.yml --plan   # print the plan and stop
konfai TRANSFORM --config Transform.yml --cpu 4  # split the work over 4 processes
konfai TRANSFORM --config Transform.yml --gpu 0  # run the stages on a GPU
```

- **Resume.** A case whose output already exists is skipped, so rerunning after an interruption picks up
  where it stopped. `-y/--overwrite` recomputes everything.
- **Failures.** A case that fails does not stop the others. The run lists the failed cases at the end and
  exits non-zero; a rerun retries exactly those.
- **`--cpu N`** splits the work between N processes, largest cases first. Every `Save` and `Write` must then
  write to a directory format (`mha`, `nii.gz`, `omezarr`), not to a single file (`h5`).
- **`--gpu`** runs the stages on the device. The output is the same as on the CPU, except a linear
  `Resample` through a rotation or a displacement field, which can differ by about 1e-5 of the data's range.
- **`--plan`** reads the config the way a run does, so it writes the resolved defaults back into
  `Transform.yml` (copy the file first to keep your version).

A run that swept for more than a second ends with one line saying where the time went (the chain, the
reads, the writes). The longest of the three is the one to work on.

```{note}
`konfai_apps.transforms:KonfAIInference` lets a chain call a packaged model as one stage. It loads the model for every case and
its memory is outside `memory_budget`. To run a model over a cohort, use `PREDICTION`, which loads it once.
```

## The plan

Every run plans before writing anything. The console gets one summary line, and the full plan is in
`./Transforms/<name>/log_0.txt`:

```text
[KonfAI] plan over 1 rank(s) | 120 entr(ies): 18 LOAD, 100 STREAM, 2 WHOLE-VOLUME | per-rank
budget 7.45 GiB ('8G') | 2 note(s) -> full plan in ./Transforms/CT_ISO/log_0.txt
```

Each case gets a verdict:

| Verdict | Meaning |
| --- | --- |
| `STREAM` | Read and written region by region. Memory stays at one slab. |
| `LOAD` | Read once into memory because its format cannot serve regions (NRRD). It fits the budget. |
| `WHOLE-VOLUME` | A stage needs the whole volume, so the case is assembled in memory. The plan names the stage and why. |
| `SKIP` | The output already exists. |
| `REDUCE` / `REFUSED` | For a chain that folds the cohort into one output (`Reduce`, below): it streams, or it cannot run. |

A case that cannot stream and does not fit `memory_budget` stops the run before anything is written.

## When a chain cannot stream

`on_fallback` says what a `WHOLE-VOLUME` verdict should do:

| Value | Effect |
| --- | --- |
| `allow` | Go ahead. The plan still names the case. |
| `warn` (default) | Go ahead and warn. |
| `error` | Refuse the run. |

The usual causes and their fixes:

| The plan says | Fix |
| --- | --- |
| a stage needs the whole volume | Nothing to fix (`Squeeze` and `Norm` change the tensor's rank). Check it fits the budget. |
| a statistic after a stage that changes values | Put a `Save` before the statistic (below). |
| the destination cannot write regions | Write to `:omezarr`, `:h5`, or `:mha`/`:nii` for an image with geometry. |

`[Clip, Standardize]` cannot stream: once `Clip` has run, the statistics stored with the volume are no
longer those of `Standardize`'s input. A `Save` in between fixes it, because `Standardize` then reads its
statistics from the saved copy:

```yaml
transforms:
  Clip: {min_value: 0.0, max_value: 400.0}
  Save: {dataset: ./Work:h5}
  Standardize: {inverse: false}
  Write: {dataset: ./Out:omezarr}
```

`Write` is the output of the run. `Save` is an intermediate copy: it is skipped when the `Write` after it
already exists. Both need their own `dataset`.

## What the config refuses

These are refused before any data is read:

- a chain that does not end with a `Write`, or a stage after the `Write`;
- two chains writing the same group to the same dataset;
- a `Write` inside a source dataset;
- a `Save` without a `dataset`;
- any key nothing reads (a misspelled `memory_budge:`, a `Clip: {min_val: …}`). The error names the key and
  the closest valid one.

A store in `uint16` (common in microscopy) can fail in the first stage that computes on it, because torch
supports few operations on `uint16`. Put a cast at the head of the chain:

```yaml
transforms:
  TensorCast: {dtype: int32, inverse: false}
  Clip: {min_value: 0.0, max_value: 40000.0}
  Write: {dataset: ./Out:omezarr}
```

## Fields

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `name` | string | `TRANSFORM_01` | The run folder under `--transforms-dir`. |
| `on_fallback` | `allow` \| `warn` \| `error` | `warn` | What a `WHOLE-VOLUME` case does. |
| `manual_seed` | int | `0` | The seed `Expand` draws from: same seed, same copies. |
| `cudnn_benchmark` | bool | `false` | Faster convolutions on the GPU, without a bit-for-bit replay. |
| `Dataset` | mapping | | Sources, chains, budget. |

Under `Dataset:`:

| Field | Type | Default | Effect |
| --- | --- | --- | --- |
| `dataset_filenames` | list of `path[:format]` | `["./Dataset:mha"]` | Where cases are read. |
| `memory_budget` | size | `auto` | Memory each process may use for its buffers. `"8G"` is 8 x 10^9 bytes, `"8GiB"` is 8 x 2^30, a bare number is GiB. `auto` takes 80% of the machine, split between processes. |
| `subset` | string or list | `null` | Which cases to run: a case name, a case-list file, `~file` to exclude, a `start:end` slice, or a list of these. |
| `groups_src` | mapping | | The chains, by source group then destination group. |

There is no `patch`, `batch_size` or `validation` here: the planner cuts the slabs itself. With several
source groups, only the cases present in all of them are processed; the plan says how many were left out.

## Changing the number of cases

A chain maps one case to one output. Two stages change that from the point where they appear:

| Stage | Effect | Writes |
| --- | --- | --- |
| `Reduce` | N cases to 1 | one entry, named by `output` |
| `Expand` | 1 case to N copies | one entry per copy, named by `pattern` |

A chain uses at most one of them. Both exist only in `TRANSFORM`.

### `Reduce`: one volume from a cohort

Stages before `Reduce` run on each case, `Reduce` combines the cases voxel by voxel, and stages after it
run on the result. Memory stays at a few regions, whatever the number of cases.

```yaml
transforms:
  Clip: {min_value: 0.0, max_value: 400.0}
  Reduce:
    operator: Mean
    output: template
  Write: {dataset: ./Atlas:h5}
```

| Field | Default | Effect |
| --- | --- | --- |
| `operator` | `Median` | `Mean`, `Median`, `Vote`, `Concat`, or your own `Reduction` class. Its parameters go next to `operator`. |
| `output` | required | The name of the single entry written. |
| `grid` | `strict` | How the cases must agree on their grid (below). |
| `grid_tolerance` | `1e-6` | The tolerance of `strict`. |
| `provenance` | `true` | Record the operator and the list of cases in the output's header. |

```{warning}
`Mean` and `Median` are for intensities: on labels they produce values that are no label. Combine
segmentations with `Vote`, which keeps the label most cases agree on.
```

`grid` decides when two cases count as the same space:

- `strict`: same extent, spacing, origin and direction;
- `shape_only`: same extent only, for cases already resampled together but with approximate headers;
- `reference:<case>`: same extent, and the output takes that case's geometry.

`shape_only` and `reference:` cannot check that the cases are really aligned. To put a cohort on one grid,
resample every case onto a reference first (`Resample: {reference: …}`, below).

A `Reduce` that cannot stream refuses the run: there is no whole-volume fallback. A cohort stored in a
format that cannot serve regions (NRRD) is decoded once per region; put a `Save: {dataset: ./Cache:h5}`
before the `Reduce` so each case is decoded once. A reduction is one unit of work, so `--cpu N` cannot
split it.

### `Expand`: copies of each case

After `Expand`, each stage runs once per copy. Random stages (augmentations) and ordinary transforms can be
mixed freely:

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
              Clip: {min_value: 0.0, max_value: 400.0}   # once per case
              Expand: {nb: 8, pattern: "{name}_r{a:02d}"}
              Rotate: {is_quarter: true}                 # a random draw, per copy
              Resample: {spacing: [2.0, 2.0, 2.0]}       # a transform, per copy
              Brightness: {b_std: 0.2}                   # another draw
              Write: {dataset: ./Augmented:omezarr}      # once per copy
```

This writes `./Augmented/<case>_r01/` to `_r08/` for each case.

- `pattern` must contain `{name}` and `{a}` (the copy number, from 1), or copies would overwrite each other.
  Quote it, or YAML reads `{name}` as a mapping.
- A random stage before `Expand`, or an `Expand` with no random stage after it, is refused.
- A copy's draws depend only on `manual_seed`, the case name and the stage, so a rerun, a subset or another
  machine produces the same copies. `Expand` takes its own `seed` to draw different copies on purpose.
- Resume works per copy.

To augment an image and its mask the same way, give both chains the same `Expand` and the same geometric
draws. Draws that only change intensity (`Brightness`) can differ between the chains:

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
              Brightness: {b_std: 0.2}
              Write: {dataset: ./Augmented:omezarr}
      SEG:
        groups_dest:
          SEG_aug:
            transforms:
              Expand: {nb: 8, pattern: "{name}_r{a:02d}"}
              Rotate: {is_quarter: true}
              Write: {dataset: ./AugmentedSeg:omezarr}
```

**Cost.** Copies whose draws only change voxel values (`Brightness`, `Contrast`, `Noise`, `CutOUT`) share
one read of the case. Draws that move voxels (`Rotate`, `Flip`, `Scale`, `Elastix`, `Translate`) read the
case once per copy. When the stages before `Expand` are expensive, put a `Save` before it.

```{note}
`Flip`, `Permute` and `Foreign` exist both as a transform and as a random draw. Before `Expand` the name
means the transform, after it the draw. Write `konfai.data.transform:Flip` to force the transform.
```

A chain can also be a YAML list, which lets the same stage appear several times:

```yaml
transforms:
  - Clip: {min_value: -1000.0, max_value: 1000.0}
  - Clip: {min_value: -200.0, max_value: 400.0}
  - Write: {dataset: ./Out:mha}
```

The resolved config names the repeats `Clip`, `Clip#2`, `Clip#3`. A stage written with nothing under it
(`Canonical:`) takes its defaults.

## `Resample`

One stage covers every resampling. It answers two questions, and asked together they cost a single
interpolation:

| Question | Key | Meaning |
| --- | --- | --- |
| Which grid to write on | (none) | the case's own grid |
| | `spacing` | same field of view, another voxel size |
| | `shape` | same field of view, another voxel count |
| | `reference` | the grid of a stored image |
| Through which map | (none) | none: only the grid changes |
| | `field` | a displacement field, in world units |
| | `transforms` | transforms stored beside the cases (rigid, affine, BSpline, field) |

A `spacing` or `shape` value of 0 or less keeps that axis as it is. `align: extent` (the default) keeps the
field of view; `align: origin` keeps the first voxel in place. With `extent`, the voxel count is rounded, so
the written spacing is the closest one that fits the field of view.

**Onto a reference grid.** The output takes the extent, spacing, origin and direction of a stored image.
This is how to put a cohort on one grid before a `Reduce`:

```yaml
transforms:
  Resample: {reference: case_0, reference_group: CT, fill: 0.0}
  Reduce: {operator: Median, output: template, grid: strict}
  Write: {dataset: ./Template:mha}
```

`reference_dataset: ./Raw:mha` reads the reference from another store. `reference: '{case}'` uses each
case's own entry in `reference_group`, which puts registered images on their field's grid.

**Through a displacement field.** Add `field:` to apply a registration in one interpolation. The field can
be coarser than the image; outside its extent the displacement is zero. Fields stored beside the cases need
only `field_group`:

```yaml
transforms:
  Resample: {reference: case_0, reference_group: CT, field: ./Fields:mha, field_group: DVF}
  Write: {dataset: ./Registered:mha}
```

**Through stored transforms.** Each key of `transforms:` is a group holding one transform per case; the
value says whether to invert it. Several groups compose, the last one applied first, as in SimpleITK:

```yaml
transforms:
  Resample: {transforms: {reg: false}}
  Write: {dataset: ./Registered:mha}
```

Inverting a BSpline or a displacement field needs the whole volume: store the inverse to keep the case
streamed.

A streamed case reads, for each region, the part of the image the region's faces map to. A registration
result that does not fold is read exactly; one that folds can reach past its faces, and those voxels come out
wrong.

**Label maps.** Without `interpolation`, `uint8`, `int64` and `bool` volumes take the nearest voxel and the
rest are interpolated. A label map stored in another type must say `interpolation: nearest`, or labels are
blended into values that are no label.

`Resample` refuses a case without geometry (`Origin`, `Spacing`, `Direction`), a reference whose direction
differs from the case's (run `Canonical` first), and a case that does not overlap the reference grid at all.
A partial overlap is fine: the rest is `fill`, and the plan prints how much of the grid each case covers.

## Reproducibility

The output depends only on the config and the data. The slab height depends on the machine (`auto` budget),
and it changes nothing for pointwise stages, crops, reorientations and axis-aligned resampling. Only a linear
`Resample` through a rotation or a displacement field can differ by about 1e-5 of the data's range between
two slab heights.

## Statistics

`Normalize`, `Standardize` and other stages that need a figure of the whole volume (`Min`, `Max`, `Mean`,
`Std`, or their `PerChannel` versions) read it from the stored case before streaming it. That figure is the
stored volume's, so it is only valid when no stage before it changes the values: this is why `Clip` then
`Standardize` needs a `Save` in between.

## Writing your own transform

A transform streams when it declares what it reads. This one averages each voxel's neighbourhood:

```python
# BoxFilter.py, importable from the directory you run konfai from
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
        # Each output voxel reads a neighbourhood of `radius` voxels.
        return PatchLocality(LocalityKind.HALO, halo=(self.radius,))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        k = 2 * self.radius + 1
        return F.avg_pool3d(tensor.to(torch.float32), k, stride=1, padding=self.radius).to(tensor.dtype)
```

```yaml
transforms:
  BoxFilter:BoxFilter: {radius: 2}
  Write: {dataset: ./Out:omezarr}
```

The declaration must match what `__call__` reads: nothing checks it, and a wrong one leaves seams at slab
borders. Without a declaration the transform takes the whole volume, which is always correct.

| `__call__` reads | Declare | Also implement |
| --- | --- | --- |
| the same voxel | `POINTWISE` | nothing |
| a neighbourhood | `HALO`, with `halo=(r,)` | nothing |
| the volume flipped or permuted | `ORIENTATION` | `stream_region_source()` |
| a shifted sub-box | `CROP` | `stream_region_source()` |
| another grid | `REGRID` | `stream_region_source()` and `stream_region()` |
| a whole-volume statistic | `GLOBAL_STAT`, with `stat_keys` | nothing |
| really the whole volume | nothing | nothing |

## Python API

```python
from konfai.transformer import build_transform

workflow = build_transform(transform_file="Transform.yml", transforms_dir="./Transforms")
plan = workflow.compute_plan(world_size=1)
print(plan.report())
```

`build_transform` builds the workflow without running it. `konfai.transform()` and
`konfai.plan_transform()` are the same workflow from Python ({doc}`../usage/python-api`).
