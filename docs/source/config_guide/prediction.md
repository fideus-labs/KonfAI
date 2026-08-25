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
  --models Checkpoints/SEG_BASELINE/*.pt
```

Checkpoints are named after the moment they were written, so there is no fixed
filename to type; the glob picks up whatever training kept.

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
| `autocast` | bool | `false` | No | Enables AMP during inference. On the shipped Segmentation example: 4.2 s to 2.7 s, 11110 of 58.4 million label voxels change, at boundaries. |
| `channels_last` | bool | `false` | No | Lays the convolution weights and inputs out channels-last (4-D and 5-D). With `autocast`, 2.7 s to 2.2 s on the same example and no further voxel changes; alone, no gain and 3199 voxels moved by the kernels cuDNN then picks. |
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
| `subset` | string / list / null | Restricts which cases are predicted: a flat selector: a case name, a case-list file, `~file` to exclude, a `start:end` slice, or a list of those. Not a nested mapping. |
| `batch_size` | int | Number of patches per inference batch. |

Use `Dataset.Patch` when:

- the full input does not fit in memory
- you want slice-wise or sliding-window inference
- you need the same spatial strategy as training

### Free patch axes: sizing by measurement

A `patch_size` entry of `0` declares a FREE axis the framework sizes itself. The
patch starts at the axis's full extent (the whole volume when every axis is
free), and shrinks only if the device actually runs out of memory:

```yaml
Patch:
  patch_size: [0, 0, 0]   # whole volume when it fits; [1, 0, 0] = full 2D slices
  overlap: 0
```

There is no budget key: the budget is the GPU's measured free VRAM. On a CUDA
out-of-memory the run reads what the failed forward cost (the measurement is
free (it already ran), shrinks the free axes by that ratio (pinned axes never
move), re-plans the patch grid and re-runs the rank's cases) typically one
restart. The chosen size also reserves room for the accumulation, so the blend
stays on the GPU; when that reservation cannot fit (or cannot be measured), the
forward is sized alone and the writer blends on the host instead. `overlap`
accepts a voxel count (`8`), a percent string (`"20%"`), or `null` (a 20%
default), resolved after the size; an axis a single patch spans gets none. A
`patch_size` without a `0` is never resized: the OOM propagates.

`overlap` accepts four forms and the binder keeps each one's type: a voxel count
(`16`), a fraction (`0.2`), a percent string (`"20%"`), and a per-axis list
(`[10, 20, 0]`). Declaration-order coercion once turned `overlap: 0.25` into
`int(0.25) == 0`: silent no-overlap; that is fixed and pinned by
`tests/unit/test_config.py::test_apply_config_union_keeps_the_value_type_over_lossy_coercion`.

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

One `Prediction.yml` can be shared between different checkpoints as long as
the exported output name stays consistent.

**Streamed writes are automatic: there is no config key.** When an output can be finalized slab by slab
identically to the assembled volume (a single augmentation, a voxel-local reduction, and an
`mha`/`h5`/`omezarr` destination), each slab is written to disk as its patches complete, bounding RAM at
one patch window instead of the whole volume. Geometry inverses stream too, composed in any number
(`Canonical`/`Flip`/`Permute`, `Padding`, a nearest-mode `Resample`):
each slab is remapped, cropped, or resampled through a sliding window straight to its written region.
A chain streaming cannot honour streams its pointwise prefix into a light post-reduction buffer and
runs the rest whole-volume on it. Streamed outputs match the assembled path voxel for voxel on a given
device; only a transcendental-terminated float chain (Softmax/Sigmoid) can differ by ~1 ULP between a
GPU window and a CPU whole-volume run. Set `KONFAI_STREAMED_WRITES=0` to force the whole-volume path
globally (ops/debug or exact bit-reproducibility against a CPU run).

### Where the run's time went

A run whose loop took more than a second closes with a line accounting for it, phase
by phase, in the same shape as the transform workflow's sweep line:

```text
[KonfAI] prediction 84.2 s = fetch 3.1 + forward 41.5 + blend 22.0 + finalize(stream) 9.8 + finalize(case) 0.0 + drain 1.2 + other 6.6 | writer 30.4 s, waited on 8.9 s
```

The sum before the bar is the loop's own thread and it closes exactly: what the named
phases do not account for is `other`. `fetch` is the wait for the loader's next batch,
`forward` the model, `blend` a patch's inverses and its blend into the accumulator (the
copy home included when the case accumulates on the host), the two `finalize` figures
the slabs and the cases handed to the writer, and `drain` the writes still queued when
the loop ends. On a GPU the loop only enqueues the forward and the blend, so the device's
time is waited for where a result crosses to the host.

After the bar is the writer: its own thread's time, and how long the loop stood waiting
on it inside the finalize phases. The background writer overlaps the disk with the next
forward only while its queue has room; `waited on` close to `writer` means the
destination is the floor of the run, and the writer has become synchronous.

## Examples

See:

- `examples/Segmentation/Prediction.yml`
- `examples/Synthesis/Prediction.yml`

## Troubleshooting

- If geometry or intensity range is wrong, review the final transforms in
  `outputs_dataset`.

## Next steps

- {doc}`evaluation`: to score the written predictions against ground truth.
- {doc}`../concepts/datasets`: the shared `dataset_filenames`, `groups_src`,
  and `subset` conventions.
- {doc}`../concepts/model-graph`: how the model output paths referenced by
  `outputs_dataset` are named.
