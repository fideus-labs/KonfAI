# Evaluation configuration

Evaluation configuration lives under the `Evaluator` root object.

```yaml
Evaluator:
  metrics:
    SEG:
      targets_criterions:
        SEG_PRED:
          criterions_loader:
            Dice:
              labels: [1, 2, 3]
  Dataset:
    ...
  train_name: SEG_BASELINE
```

## Running it

From the directory that contains `Evaluation.yml`:

```bash
konfai EVALUATION -y --config Evaluation.yml
```

The output directory is controlled by `Evaluator.train_name` in the YAML and
`--evaluations-dir` on the CLI.

Evaluation persists per case as it goes: each rank appends finished cases to a
`*.cases.rank<N>.jsonl` file beside the metric JSON, so a rerun after an
interruption pays only the cases that are not yet recorded. A case recorded with
other metrics than the ones the config names now is scored again.

A case whose prediction or reference cannot be read (a truncated, empty or
corrupt file) is set aside and the other cases go on. A warning names the case,
the file and the error when it happens, and a last warning lists every case set
aside; the run exits 0. The per-case values and the aggregates cover the cases
evaluated, and the metric JSON lists the others under `set_aside` (see Output
files). Only a read error of the case's own files is set aside: a configuration
error, an out-of-memory or an error raised by a transform or a metric still
stops the run. Under several ranks, each rank warns about the cases of its own
shard and the metric JSON lists them all. A split none of whose cases can be
read fails instead, and writes no metric JSON, unless the config names no metric
(`metrics: {}`): its metric JSON then lists the cases under `set_aside`.

## Top-level fields

| Field | Type | Default in code | Required | Effect |
| --- | --- | --- | --- | --- |
| `metrics` | mapping | default target criterions loader | Yes in practice | Declares what metrics should be computed and between which groups. |
| `Dataset` | mapping | `DataMetric()` | Yes | Defines how targets and predictions are loaded. |
| `train_name` | string | `TRAIN_01` | Yes in practice | Names the evaluation output folder. |

## `metrics`

The evaluation structure mirrors `outputs_criterions`, but without the model.

```yaml
metrics:
  sCT:
    targets_criterions:
      CT;MASK:
        criterions_loader:
          MAE:
            reduction: mean
          PSNR:
            dynamic_range: 4095
```

Structure:

- output group → the predicted group to evaluate
- `targets_criterions` → one or more target groups, optionally composed with `;`
- `criterions_loader` → one or more metric implementations

Some metrics also accept attributes or write auxiliary datasets. This behavior is
implemented in `konfai.evaluator.Evaluator.update()` and `konfai.metric.measure`.

An output and each target it is scored against voxel to voxel must have the same
spatial shape. A case where they differ is refused with an error naming the case,
the two groups and their shapes, before any value of it is recorded. Left out of
the check: a metric that reads the geometry itself (a `CriterionWithAttribute`,
such as the IMPACT metrics), a metric that puts its target on the output grid
itself (`Dice` and `FocalLoss`, nearest neighbour), a metric that never compares
its target voxel to voxel (`Mean`, `Variance`, `Gram`, `BCE`, `PatchGanLoss`,
`KLDivergence`), and a pair that is not two images (landmarks scored by `TRE`).

The same pairs are then compared on their geometry, as each group lands after
its `transforms`, and so are `Dice` and `FocalLoss` when the output and the
target have the same shape: they then take the target as it is. When an output
and a target have the same shape but a different origin, spacing or direction,
each voxel is scored against one at another place: the case is still scored, and
one warning per case names it, the output, the target and each value that
differs, on both sides. The origin is compared within a thousandth of the
smallest spacing (a thousandth of a voxel): a header stored in float32, as NIfTI
stores it, moves the origin by more than ITK's tolerance without moving a voxel.
The spacing and the direction keep ITK's tolerance: 1e-6 times the first spacing,
and 1e-6. A group without a geometry is not compared: an h5 entry without
attributes, a `.npy`, and the formats that store no origin (`png`, `jpg`,
`jpeg`, `bmp`, `tif`, `tiff`). Nor is a 2D image against a single-slice 3D one.

## `Evaluator.Dataset`

Evaluation datasets are instantiated through `DataMetric`.

Common fields:

| Field | Type | Effect |
| --- | --- | --- |
| `dataset_filenames` | list[str] | Pairs or merges the datasets needed for evaluation. |
| `groups_src` | mapping | Defines how the compared tensors are loaded. A group without `transforms` is compared as stored. |
| `subset` | string / list / null | Restricts evaluated cases: a flat selector: a case name, a case-list file, `~file` to exclude, a `start:end` slice, or a list of those. Not a nested mapping. |
| `validation` | string / list / null | Optional validation selector for a separate JSON report. Supports a case-list file, a list of case names, or a list of case-list files. |
| `num_workers` | int or null | DataLoader workers. `None` resolves to `0`, or to `max(1, min(cpu_count, 4))` when reading one patch decodes a whole volume (a store that cannot serve a region). |
| `pin_memory` | bool | Pinned host memory for the batches (`false` when absent). |
| `prefetch_factor` | int or null | Prefetched batches per worker, only with workers; `None` resolves to `2`. |
| `persistent_workers` | bool or null | Keep the workers alive, only with workers; `None` resolves to `false`. |

### `memory_budget`: memory-bounded evaluation

Evaluation bounds itself by default: an absent `memory_budget` means `auto`
(80% of the detected memory), and explicit values (a bare number in GiB,
`"24GB"`) narrow it. Each run sizes itself from image headers alone: a case that
fits the budget is evaluated whole, and a case that does not is cut into the largest
DISJOINT patches that fit. Metrics accumulate running partial sums per patch and
combine them into the whole-case value (never a mean of per-patch values). MAE, MSE,
ME and PSNR sum in float32, so they agree with the whole-volume value to float32
rounding (a relative difference near 1e-7), not bit for bit.
MAE, MSE, ME, PSNR, SSIM and Dice (masked or not) support this, and the SaveMap
error maps stream region by region into their `dataset` (mha, h5 or omezarr). One
caveat on the first two: `MAE` and `MSE` are reducible only for `reduction: mean` or
`sum`, so a `reduction: none` on either forces the whole-volume path for the whole
run, by the same rule as a non-reducible metric below.

A metric that scores each voxel through a window declares the window's radius as
its `halo`, and the reader serves it: SSIM (7-voxel window) declares 3, so every
patch is read 3 voxels past each face of its slot, clamped at the volume's faces,
and the sizing counts that band in the budget. SSIM sums the map voxels centred
in the slot, which is exactly the whole-volume map's share of it (the whole-volume
map is cropped by the same radius at the faces); the metrics without a halo see
the slot alone, so their values are the ones they had without SSIM in the run.
A custom metric declares `halo` beside `reducible` and receives `core=` in
`partial_metric`, the slot's slices within the patch it is handed.
One metric that cannot recombine (LPIPS, or any custom metric that does not
declare `reducible`) keeps the whole-volume path for the entire run: correct
beats bounded. Evaluation streams its data whatever the budget says, one pass,
a cache is never re-read; in training the same budget also picks cache versus
streaming.

## Output files

Evaluation writes JSON files, not CSV files. The main outputs are:

- `Metric_TRAIN.json`
- optionally `Metric_VALIDATION.json`

The JSON structure contains:

- per-case values under `case`
- aggregated statistics under `aggregates`, such as mean, std, percentiles,
  min, max, and count
- `directions`: per metric, `"max"` or `"min"`, emitted whenever a metric declares
  one so a consumer can rank runs without guessing which way is better
- `set_aside`: present only when some case of the split could not be read, each
  such case with the error that set it aside, so an aggregate is never read as
  the whole cohort's

This behavior comes from `konfai.evaluator.Statistics.write()`.

### Where the split's time went

A split that ran for more than a second closes with a line accounting for it,
phase by phase, in the rank's log:

```text
[KonfAI] evaluation TRAIN 4.0 s = wait(load) 0.6 + h2d 0.2 + MAE 0.2 + PSNR 0.1 + SSIM 2.5 + map 0.3 + other 0.1
```

`wait(load)` is the wait for the loader's next case or patch, `h2d` the move to
the metric device, then one figure per metric name, `map` the error-map writes of
the SaveMap metrics and `flush` the combination of a streamed case's partial
states; what the named phases do not account for is `other`, so the sum closes
exactly. A phase that spent nothing is left out: the run above read its cases
whole, so it carries no `flush`. On a GPU a metric's figure is the time to
enqueue its kernels, not to run them: a slow kernel shows up in whatever next
waits on the device, typically the next `h2d` or a metric that reads a value
back.

## Examples

See:

- `examples/Segmentation/Evaluation.yml`
- `examples/Synthesis/Evaluation.yml`

## Troubleshooting

Common evaluation mistakes:

- the evaluation file still points to an old prediction folder
- label definitions in the metric do not match the dataset encoding

## Next steps

- {doc}`index`: the `dataset_filenames` merge flags and the
  `validation` selector used here.
- {doc}`prediction`: to produce the prediction dataset this file scores.
