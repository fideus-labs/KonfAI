# Prediction configuration

Prediction is configured under the `Predictor` root, in `Prediction.yml`.

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

From the folder holding `Prediction.yml` and the training's `Checkpoints/`:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml \
  --models Checkpoints/SEG_BASELINE/SELECTED_MODEL.pt
```

`SELECTED_MODEL.pt` is the dated checkpoint training kept (not `resume_latest.pt` or `crash_*.pt`). Several
`--models` make an ensemble:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models ckpt_a.pt ckpt_b.pt ckpt_c.pt
```

- **Ensembles.** `combine` merges the members: `Mean`, `Median`, `Std`, `Vote`, `Concat`, or your own
  reduction. For a segmentation, save the `Softmax` output and apply `Argmax` in `final_transforms`, so the
  probabilities are averaged before the labels are chosen; or merge label maps with `Vote`. `Mean` refuses
  label maps (the mean of two labels is a third).
- **Precision.** Members are combined in float16, which halves the memory of a many-class ensemble. Values
  above 65,504 overflow: keep model outputs in a normalised range and restore the scale in the output
  transforms.
- **Resume.** A case whose outputs are all written is skipped; `-y` predicts everything again.
- **Unreadable inputs.** A case whose file cannot be read (truncated, empty, corrupt) is set aside with a
  warning, and the others are predicted. It gets no output, so a rerun predicts it once the file is fixed.
  A run where no case can be read fails. Only read errors are set aside: a config error or an out-of-memory
  still stops the run.

## Top-level fields

| Field | Default | Effect |
| --- | --- | --- |
| `Model` | | The model, with the same `classpath` as in training. The weights come from `--models`. |
| `Dataset` | | The input data (below). |
| `outputs_dataset` | | What to write and how (below). |
| `combine` | `Mean` | How the members of an ensemble are merged. |
| `checkpoint_cache_gib` | `1.0` | Memory for keeping the ensemble's checkpoints loaded (below). |
| `train_name` | | Names the output folder `Predictions/<train_name>/`. |
| `manual_seed` | `null` | Seeds the test-time augmentation draws (with the case name, so a case's draws never depend on the others). |
| `autocast` | `false` | Mixed precision: about 1.6 times faster; a few labels may change at boundaries. |
| `channels_last` | `false` | Channels-last layout: a little faster on top of `autocast`, on some models. |
| `cudnn_benchmark` | `false` | Fastest cuDNN kernels even with a seed, without exact replay. |
| `torch_compile` | `false` | Compile the model once for all the members. Helps only models bound by their kernels. |
| `gpu_checkpoints` | `null` | Modules to place on other GPUs. |
| `data_log` | `null` | Outputs to log in TensorBoard. |
| `check_training_transforms` | `true` | Warn when an input is preprocessed differently from training (below). |

### Checkpoint memory

The members of an ensemble run one after the other in one model. Their weights stay in a host cache of
`checkpoint_cache_gib` (per process) and are reloaded when they do not fit; only the model's weights are kept,
not the optimizer's. `0` turns the cache off.

## `Predictor.Dataset`

| Field | Effect |
| --- | --- |
| `dataset_filenames` | Where the inputs are ({doc}`index`). |
| `groups_src` | The input groups and their transforms. |
| `augmentations` | Test-time augmentation. |
| `Patch` | How the volume is cut (below). |
| `subset` | Which cases to predict. |
| `batch_size` | Patches per batch. `0` measures the largest batch that fits on the GPU. |
| `num_workers` | Loader workers (`null`: 0, or up to 4 when the format cannot read regions). Each worker holds the case it prepares, so more workers use more RAM. |
| `pin_memory`, `prefetch_factor`, `persistent_workers` | DataLoader settings, as in training. |

| `Patch` field | Default | Effect |
| --- | --- | --- |
| `patch_size` | `[128, 128, 128]` | The patch the model sees. A `0` lets KonfAI size that axis. |
| `overlap` | `null` | Voxels (`16`), a fraction (`0.2`), `"20%"`, or one per axis. `null`: 20%. |
| `pad_value` | `null` | Padding past the volume. `null`: the data's minimum. |
| `extend_slice` | `0` | 2.5-D: neighbouring slices added as channels (with `patch_size[0] == 1`). |

### Letting KonfAI size the patch

```yaml
Patch:
  patch_size: [0, 0, 0]   # the whole volume when it fits; [1, 0, 0] for whole 2-D slices
  overlap: 0
```

A `0` axis starts at the whole extent. If the GPU runs out of memory, KonfAI measures what the forward
needed and cuts the axis into the fewest equal patches that fit, usually in one retry. It also keeps room to
blend the result on the GPU. A patch size without `0` is never changed.

### The training-chain check

A checkpoint does not record how its inputs were preprocessed. A `Prediction.yml` that prepares an input
differently from the `Config.yml` it was trained with runs, and gives wrong results: the Synthesis example
once used a different `Standardize` mask in prediction, and its error went from 98 to 409 HU with the same
weights.

So prediction compares the input transforms with the training config it finds in
`Statistics/<train_name>/`, and warns about each difference:

```text
[KonfAI] WARNING: this run preprocesses a model input differently from TRAIN_01:
[KonfAI]   'MR:MR' transforms[1] Standardize: mask: 'None' in training, 'MASK' here
```

It only warns, since a difference can be intended. Stages that do not change values (`Statistics`, `Save`)
and output transforms are ignored. Without the training config at hand (an app, a copied `.pt`), it says it
could not check. `check_training_transforms: false` turns it off.

## `outputs_dataset`

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutputDataset
      group: sCT
      same_as_group: MR:MR
      reduction: Mean
```

The key is the model output to save ({doc}`../reference/components/models`).

| Field | Effect |
| --- | --- |
| `name_class` | The writer class (`OutputDataset`). |
| `group` | The name of the written group. |
| `dataset_filename` | Where to write, and in which format. |
| `same_as_group` | `source:destination` groups whose geometry the output takes. |
| `before_reduction_transforms` | Transforms applied to each copy before the copies are merged. |
| `reduction` | How the test-time augmentation copies are merged (`Mean`, `Vote` for labels, …). |
| `after_reduction_transforms` | Transforms after the merge. |
| `final_transforms` | Transforms just before writing (an `Argmax`, an inverse normalisation). |
| `attributes` | Header values to set, as `key=value` (`key=` removes one). |
| `patch_combine` | How overlapping patches are blended: `Trim` (default), `Mean`, `Cosinus`, `Gaussian`. Label maps take only `Trim`. |

The output is written slab by slab as patches complete, so a large output never sits whole in memory.
There is no key for it: it happens whenever the output allows it ({doc}`../usage/large-images`).

A run that took more than a second ends with one line saying where the time went (loading, forward,
blending, writing). When the writer's time is close to the total, the disk is the limit.

## Examples

- `examples/Segmentation/Prediction.yml`
- `examples/Synthesis/Prediction.yml`

Next: {doc}`evaluation`, to score the predictions.
