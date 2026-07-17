# Prediction configuration

Prediction configuration lives under the `Predictor` root object.

```yaml
Predictor:
  Model:
    classpath: UNet.yml
    UNet:
      ...
  Dataset:
    ...
  outputs_dataset:
    ...
  train_name: SEG_BASELINE
```

## Running it

From the directory that contains `Prediction.yml` and the `Checkpoints/`
folder written by training:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml \
  --models Checkpoints/SEG_BASELINE/<checkpoint>.pt
```

You can also pass multiple checkpoints:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml \
  --models ckpt_a.pt ckpt_b.pt ckpt_c.pt
```

When multiple checkpoints are provided, the predictor combines them using the
`combine` strategy from the YAML, usually `Mean` or `Median`.

## Top-level fields

| Field | Type | Default in code | Required | Effect |
| --- | --- | --- | --- | --- |
| `Model` | mapping | `ModelLoader()` | Yes | Selects the model class used for prediction. |
| `Dataset` | mapping | `DataPrediction()` | Yes | Defines inference data loading and test-time augmentation. |
| `outputs_dataset` | mapping | default output dataset | Yes in practice | Controls which outputs are written to disk and how. |
| `combine` | string | `Mean` | No | Reduces outputs across multiple checkpoints. |
| `train_name` | string | `"name"` | Yes in practice | Names the prediction run and output folder. |
| `manual_seed` | int or null | `None` | No | Optional seed. |
| `gpu_checkpoints` | list or null | `None` | No | Module placement optimization. |
| `autocast` | bool | `false` | No | Enables AMP during inference. |
| `data_log` | list or null | `None` | No | Optional TensorBoard logging. |

## `Predictor.Model`

Prediction uses the same `classpath` convention as training:

```yaml
Model:
  classpath: Model:UNetpp5
  outputs_criterions: {}
```

In most prediction configs:

- you select the architecture
- you keep only the inference-relevant parameters
- you disable or simplify training-only criteria

Checkpoint loading is controlled by the CLI argument `--models`, not by the YAML
file itself.

## `Predictor.Dataset`

Prediction datasets are instantiated through `DataPrediction`.

Key fields:

| Field | Type | Effect |
| --- | --- | --- |
| `dataset_filenames` | list[str] | Input dataset sources. |
| `groups_src` | mapping | Input groups and preprocessing transforms. |
| `augmentations` | mapping | Test-time augmentation definitions. |
| `Patch` | mapping | Sliding-window or slice-wise inference setup. |
| `subset` | object | Restricts evaluated cases. |
| `batch_size` | int | Number of patches per inference batch. |

Use `Dataset.Patch` when:

- the full input does not fit in memory
- you want slice-wise or sliding-window inference
- you need the same spatial strategy as training

## `outputs_dataset`

`outputs_dataset` defines how selected model outputs become files on disk.

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutSameAsGroupDataset
      group: sCT
      same_as_group: MR:MR
      reduction: Mean
```

Important nested fields:

| Field | Effect |
| --- | --- |
| output key | Selects the model output to export. |
| `name_class` | Selects the output dataset implementation. |
| `group` | Output group name written to disk. |
| `dataset_filename` | Destination dataset path and format. |
| `same_as_group` | Geometry reference group for exported volumes. |
| `before_reduction_transforms` | Applied before combining ensemble or TTA outputs. |
| `after_reduction_transforms` | Applied after reduction. |
| `final_transforms` | Final transforms applied before writing. |
| `reduction` | Combines multiple predictions, usually `Mean` or `Median`. |
| `patch_combine` | Optional patch reassembly strategy. |

```{note}
One `Prediction.yml` can be shared between different checkpoints as long as
the exported output name stays consistent.
```

```{note}
**Streamed writes are automatic — there is no config key.** When an output can be finalized slab by slab
identically to the assembled volume (a single augmentation, a voxel-local reduction, and an
`mha`/`h5`/`omezarr` destination), each slab is written to disk as its patches complete, bounding RAM at
one patch window instead of the whole volume. Geometry inverses stream too, composed in any number
(`Canonical`/`Flip`/`Permute`, `Padding`, nearest-mode `ResampleToResolution`/`ResampleToShape`):
each slab is remapped, cropped, or resampled through a sliding window straight to its written region.
A chain streaming cannot honour streams its pointwise prefix into a light post-reduction buffer and
runs the rest whole-volume on it. Streamed outputs match the assembled path voxel for voxel on a given
device; only a transcendental-terminated float chain (Softmax/Sigmoid) can differ by ~1 ULP between a
GPU window and a CPU whole-volume run. Set `KONFAI_STREAMED_WRITES=0` to force the whole-volume path
globally (ops/debug or exact bit-reproducibility against a CPU run).
```

## Examples

See:

- `examples/Segmentation/Prediction.yml`
- `examples/Synthesis/Prediction.yml`

## Troubleshooting

- If geometry or intensity range is wrong, review the final transforms in
  `outputs_dataset`.

## Next steps

- {doc}`evaluation` — to score the written predictions against ground truth.
- {doc}`../concepts/datasets` — the shared `dataset_filenames`, `groups_src`,
  and `subset` conventions.
- {doc}`../concepts/model-graph` — how the model output paths referenced by
  `outputs_dataset` are named.
