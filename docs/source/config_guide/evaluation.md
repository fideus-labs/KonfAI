# Evaluation configuration

Evaluation compares predictions with references. It is configured under the `Evaluator` root, in
`Evaluation.yml`.

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

```bash
konfai EVALUATION -y --config Evaluation.yml
```

The results go to `Evaluations/<train_name>/` (`--evaluations-dir` moves them).

- **Resume.** Each finished case is recorded as it goes, so a rerun only scores the missing cases. A case
  scored with other metrics than the config now names is scored again.
- **Unreadable files.** A case whose prediction or reference cannot be read is set aside with a warning; the
  others are scored, and the results list it under `set_aside`. A split where no case can be read fails.

## `metrics`

`metrics` has the shape of `outputs_criterions`, with groups instead of model outputs:

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

The key is the group to evaluate (the prediction); `targets_criterions` names the reference groups (joined
with `;`, here a CT and a mask); `criterions_loader` lists the metrics
({doc}`../reference/components/losses-metrics`).

**Same grid.** A prediction and a reference compared voxel by voxel must have the same shape: otherwise the
case is refused with both shapes. `Dice` and `FocalLoss` resample the reference themselves, and metrics that
do not compare voxels (`Mean`, `KLDivergence`, `TRE` on landmarks) are not checked. When the shapes match but
the origin, spacing or direction differ, each voxel is compared with one at another place: the case is scored
and a warning names the differences. Put both on one grid first, with a `Resample` onto the reference.

## Fields

| Field | Default | Effect |
| --- | --- | --- |
| `metrics` | | What to compute, between which groups. |
| `Dataset` | | The predictions and references (below). |
| `train_name` | `TRAIN_01` | Names the output folder. |

Under `Dataset:`:

| Field | Effect |
| --- | --- |
| `dataset_filenames` | The datasets to read, usually the references and the predictions joined with `:i:` ({doc}`index`). |
| `groups_src` | The groups and their transforms. A group without `transforms` is compared as stored. |
| `subset` | Which cases to score. |
| `validation` | Cases scored in a separate `Metric_VALIDATION.json`. |
| `memory_budget` | Memory per case (`auto` by default, 80% of the machine). A case too large is scored in patches. |
| `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers` | DataLoader settings, as in prediction. |

Scoring a large case in patches gives the whole-case value (not a mean of patch values) for `MAE`, `MSE`,
`ME`, `PSNR`, `SSIM` and `Dice`, to float32 rounding. A metric that cannot be split this way (`LPIPS`, a
custom metric without `reducible`) makes the whole run read cases whole.

## Output files

- `Metric_TRAIN.json`, and `Metric_VALIDATION.json` when `validation` is set.
- `case`: the value of each metric for each case.
- `aggregates`: mean, standard deviation, percentiles, minimum, maximum and count.
- `directions`: for each metric, whether higher (`max`) or lower (`min`) is better.
- `set_aside`: the cases that could not be read, with the error, when there are some.

A split that took more than a second ends with one line saying where the time went: loading, and each
metric.

## Common mistakes

- The evaluation still points at an old prediction folder.
- The labels named in the metric do not match those in the data.

## Examples

- `examples/Segmentation/Evaluation.yml`
- `examples/Synthesis/Evaluation.yml`
