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

- **Ensembles.** `combine` merges the members: `Mean`, `Sum`, `Median`, `Std`, `Vote`, `Concat`, or your own
  reduction. For a segmentation, save the `Softmax` output and apply `Argmax` in `final_transforms`, so the
  probabilities are averaged before the labels are chosen; or merge label maps with `Vote`. `Mean` refuses
  label maps (the mean of two labels is a third).
  `Sum` adds the members without dividing by their count. A custom reduction derived from `Mean`
  keeps its own behavior. With one member, `Concat` preserves it and `Std` returns a zero spread map,
  without intermediate volume buffers.
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
| `autocast` | `false` | Mixed precision. Speed depends on the model and hardware; labels may change at boundaries. |
| `channels_last` | `false` | Channels-last layout: a little faster on top of `autocast`, on some models. |
| `cudnn_benchmark` | `false` | Fastest cuDNN kernels even with a seed, without exact replay. |
| `torch_compile` | `false` | Compile the model once for all the members. Pays on long runs (TTA, many patches); the first run compiles. Where it cannot compile (no Triton), the model runs uncompiled. |
| `gpu_checkpoints` | `null` | Modules to place on other GPUs. |
| `data_log` | `null` | Outputs to log in TensorBoard. |

When TensorBoard is unavailable, `data_log` skips image preparation and the extra forward for model outputs.
For a 3-D `IMAGE` or `IMAGES` log, only the displayed central slice is copied to CPU.

### Checkpoint memory

The members of an ensemble run one after the other in one model. Their weights stay in a host cache of
`checkpoint_cache_gib` (per process) and are reloaded when they do not fit; only the model's weights are kept,
not the optimizer's. `0` turns the cache off.

On a GPU, the members stay resident when that costs the batch nothing: each loads once, and a forward
switches to its weights instead of copying a checkpoint in. When holding them would shrink the measured batch,
they load per batch instead, as for a model whose class defines its own `load`; a rank that runs out of memory
with them resident restarts with them loading per batch. At one batch size the outputs are the same to the bit.

For nested models, weights are matched by the network's full path in the checkpoint. A short name is
accepted only when it identifies one entry; an ambiguous name is refused instead of loading another
network's weights. The same matching applies to training resumes.

## `Predictor.Dataset`

| Field | Effect |
| --- | --- |
| `dataset_filenames` | Where the inputs are ({doc}`index`). |
| `groups_src` | The input groups and their transforms. |
| `augmentations` | Test-time augmentation. A `Flip` gives the copies the distinct mirrors its `f_prob` allows, in turn (7 copies: all seven), rather than drawing them. |
| `Patch` | How the volume is cut (below). |
| `subset` | Which cases to predict. |
| `batch_size` | Patches per batch. `0`, the default, measures the largest batch that fits on the GPU. A `0` patch axis then takes the largest extent among the cases, the smaller ones padded up to it, so the patches share one shape. |
| `num_workers` | Loader workers (`null`: 0, or up to 4 when the format cannot read regions). Each worker holds the case it prepares, so more workers use more RAM. |
| `pin_memory`, `prefetch_factor`, `persistent_workers` | DataLoader settings, as in training. |

| `Patch` field | Default | Effect |
| --- | --- | --- |
| `patch_size` | `[128, 128, 128]` | The patch the model sees. A `0` lets KonfAI size that axis. |
| `overlap` | `null` | Voxels (`16`), a fraction (`0.2`), `"20%"`, or one per axis. `null`: 20%. |
| `pad_value` | `null` | Padding past the volume. `null`: the data's minimum. |
| `extend_slice` | `0` | 2.5-D: neighbouring slices added as channels (with `patch_size[0] == 1`). |
| `mode` | `tile` | What a case over `max_voxels` gets: `tile` cuts it into patches, `resample` runs it whole on a grid coarse enough and brings the output back onto the case's grid. |
| `max_voxels` | `null` | The voxels a patch (`tile`) or a case's coarse grid (`resample`) holds. `null`: sized from the two costs below. |
| `vram_bytes_per_voxel`, `ram_bytes_per_voxel` | `null` | What a voxel costs the pass on the GPU and in RAM. With `max_voxels` unset, KonfAI holds the tighter of the free GPU memory and the rank's RAM budget. |

### Letting KonfAI size the patch

```yaml
Patch:
  patch_size: [0, 0, 0]   # the whole volume when it fits; [1, 0, 0] for whole 2-D slices
  overlap: 0
```

A `0` axis starts at the whole extent. If the GPU runs out of memory, KonfAI measures what the forward
needed and cuts the axis into the fewest equal patches that fit, usually in one retry. It also keeps room to
blend the result on the GPU. A patch size without `0` is never changed.

### Preprocessing must match training

Use the same input preprocessing as training, including normalization masks and
intensity ranges. A mismatch can change predictions without causing an execution error.
Check `Prediction.yml` against the training configuration before using a checkpoint.

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

### Writing a layer as it is

`OutputDataset` writes an output on the grid of an input group (`same_as_group`) and undoes that group's
transforms. To write an output that lives on a grid of its own, such as an intermediate layer or features
at a lower resolution, use `OutputLayerDataset`:

```yaml
outputs_dataset:
  UNetBlock_0:UNetBlock_1:DownConvBlock:Activation_1:   # features after the first pooling, half size
    OutputDataset:
      name_class: OutputLayerDataset
      group: Features
```

It takes the same keys except `same_as_group`. No input transform is undone.
The output keeps the case's origin and scales its spacing by the ratio between
the input and layer sizes: a layer half the input's size gets twice the spacing.
With several patches, their positions must scale to whole voxels.

Compatible `OutputDataset` routes write completed slabs as patches arrive
({doc}`../usage/large-images`). `OutputLayerDataset` assembles the layer output
in memory and refuses accumulators whose estimated size exceeds `memory_budget`.

The first output voxel is placed on the first input voxel. This matches some
strided convolutions, but the writer does not infer the layer's actual sampling
grid from its stride, crop or padding. Pooling can shift voxel centres. Check the
physical alignment before using an intermediate layer as a registered medical image.

A run that took more than a second ends with one line saying where the time went (loading, forward,
blending, writing). When the writer's time is close to the total, the disk is the limit.

## Examples

- `examples/Segmentation/Prediction.yml`
- `examples/Synthesis/Prediction.yml`

Next: {doc}`evaluation`, to score the predictions.
