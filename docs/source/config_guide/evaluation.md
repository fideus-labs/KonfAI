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
            dynamic_range: None
```

Structure:

- output group → the predicted group to evaluate
- `targets_criterions` → one or more target groups, optionally composed with `;`
- `criterions_loader` → one or more metric implementations

Some metrics also accept attributes or write auxiliary datasets. This behavior is
implemented in `konfai.evaluator.Evaluator.update()` and `konfai.metric.measure`.

## `Evaluator.Dataset`

Evaluation datasets are instantiated through `DataMetric`.

Common fields:

| Field | Type | Effect |
| --- | --- | --- |
| `dataset_filenames` | list[str] | Pairs or merges the datasets needed for evaluation. |
| `groups_src` | mapping | Defines how the compared tensors are loaded. |
| `subset` | string / list / null | Restricts evaluated cases: a flat selector: a case name, a case-list file, `~file` to exclude, a `start:end` slice, or a list of those. Not a nested mapping. |
| `validation` | string / list / null | Optional validation selector for a separate JSON report. Supports a case-list file, a list of case names, or a list of case-list files. |

### `memory_budget`: memory-bounded evaluation

Evaluation bounds itself by default: an absent `memory_budget` means `auto`
(80% of the detected memory), and explicit values (a bare number in GiB,
`"24GB"`) narrow it. Each run sizes itself from image headers alone: a case that
fits the budget is evaluated whole, and a case that does not is cut into the largest
DISJOINT patches that fit. Metrics accumulate running partial sums per patch and
combine them into the exact whole-case value (never a mean of per-patch values).
MAE, MSE, ME, PSNR and Dice (masked or not) support this, and the SaveMap
error maps stream region by region into their `dataset` (mha, h5 or omezarr). One
caveat on the first two: `MAE` and `MSE` are reducible only for `reduction: mean` or
`sum`, so a `reduction: none` on either forces the whole-volume path for the whole
run, by the same rule as a non-reducible metric below.
One metric that cannot recombine (SSIM, LPIPS, or any custom metric that does
not declare `reducible`) keeps the whole-volume path for the entire run: correct
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

This behavior comes from `konfai.evaluator.Statistics.write()`.

## Examples

See:

- `examples/Segmentation/Evaluation.yml`
- `examples/Synthesis/Evaluation.yml`

## Troubleshooting

Common evaluation mistakes:

- the evaluation file still points to an old prediction folder
- label definitions in the metric do not match the dataset encoding

## Next steps

- {doc}`../concepts/datasets`: the `dataset_filenames` merge flags and the
  `validation` selector used here.
- {doc}`prediction`: to produce the prediction dataset this file scores.
