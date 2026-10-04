# Training configuration

Training is configured under the `Trainer` root, in `Config.yml`.

```yaml
Trainer:
  Model:
    classpath: UNet.yml
    UNet:
      ...
  Dataset:
    ...
  train_name: SEG_BASELINE
  epochs: 100          # examples/Segmentation ships 5, so a first run finishes quickly
```

## Running it

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

Use `--cpu 1` instead of `--gpu 0` without a GPU. Add `-tb` to start TensorBoard; KonfAI prints the address
to open. TensorBoard listens on `127.0.0.1` only, because it has no authentication: from a remote machine,
forward the port with `ssh -N -L <port>:127.0.0.1:<port> user@server`, or set `KONFAI_TENSORBOARD_HOST`.
Without the `tensorboard` extra, training runs normally and only the curves are lost.

Checkpoints go to `Checkpoints/<train_name>/`, curves and the resolved config to `Statistics/<train_name>/`
(`--checkpoints-dir` and `--statistics-dir` move them).

### Resuming

```bash
konfai RESUME -y --config Config.yml --model Checkpoints/SEG_BASELINE/resume_latest.pt
```

- `resume_latest.pt` is the latest checkpoint saved at the end of an epoch. Resuming from it continues
  exactly where training stopped: optimizer, schedulers, EMA, early stopping and random generators included.
- A checkpoint saved in the middle of an epoch holds weights for prediction and cannot be resumed.
- `epochs` counts from the start of training: to train longer, raise it.
- An early stop is decided again with the configured `patience`, so a larger one trains on.
- A save after a crash is named `crash_<date>.pt`; it is never pruned, and yours to delete.

The resumed run is bit for bit the same on the CPU with `num_workers: 0` and no augmentation; otherwise it
continues at the next epoch and says the replay is not exact.

### Seeds and the validation split

Without `manual_seed`, training draws a seed and records it in `Statistics/<train_name>/Seed.txt`. `RESUME`
reads it back, and setting `manual_seed` to it replays the run.

The train/validation split is drawn on the cases found at each launch. If cases were added, removed or
renamed, `RESUME` warns and names the cases that changed side. To keep a split fixed, give `validation` as a
list of case names.

## Top-level fields

| Field | Default | Effect |
| --- | --- | --- |
| `Model` | | The model (below). |
| `Dataset` | | The data (below). |
| `train_name` | `TRAIN_01` | Names the run and its output folders. |
| `manual_seed` | `null` | Seeds the model, the split and the batch order. `null` draws one and records it. |
| `epochs` | `100` | Number of epochs. |
| `it_validation` | `null` | Validate and save every N iterations. `null`: once per epoch. |
| `it_lr_update` | `null` | Step the schedulers every N batches. `null`: once per epoch. |
| `autocast` | `false` | Mixed precision. Measure speed and numerical differences on your model and hardware. |
| `channels_last` | `false` | Channels-last memory layout. Faster on some models and slower on others: measure it with `benchmarks/perf/bench_train_step.py`. |
| `cudnn_benchmark` | `false` | Let cuDNN pick the fastest kernels even with a seed, at the cost of an exact replay. |
| `torch_compile` | `false` | Compile the network with `torch.compile`. Compilation adds startup cost; speed depends on the model and hardware. |
| `gradient_checkpoints` | `null` | Modules to run with gradient checkpointing (less memory, more compute). |
| `gpu_checkpoints` | `null` | Modules to place on other GPUs. |
| `ema_decay` | `0` | Keep an exponential moving average of the weights when above 0. |
| `data_log` | `null` | Groups or outputs to log as images in TensorBoard. |
| `EarlyStopping` | `null` | Stop when the score stops improving (below). |
| `save_checkpoint_mode` | `BEST` | `BEST` keeps the checkpoint with the lowest validation loss, `ALL` keeps every save. |

Image logging is skipped when TensorBoard is unavailable. Model outputs requested by `data_log` use
an extra forward in evaluation mode: BatchNorm statistics and the random generators used by training
are preserved, and each module returns to its previous mode afterwards.
For a 3-D `IMAGE` or `IMAGES` log, only the displayed central slice is copied to CPU.

### `EarlyStopping`

| Field | Default | Effect |
| --- | --- | --- |
| `monitor` | `null` | The logged losses or metrics summed into the score. `null`: the losses. |
| `patience` | `10` | Validations without improvement before stopping. |
| `min_delta` | `0.0` | The smallest change that counts as an improvement. |
| `mode` | `min` | `min` or `max`: the direction that improves. |

An undefined score (`NaN` or infinity) counts as an evaluation without improvement and consumes
patience. Only a finite score can become the best reference; the first finite score resets the counter.
`BEST` checkpoint retention also prefers finite scores. Until one is available, it keeps the latest
checkpoint, including after a resume.

## `Trainer.Model`

A `classpath` selects the model; its arguments go in a section named after the class:

```yaml
Model:
  classpath: UNet.yml
  UNet:
    optimizer:
      name: AdamW
      lr: 0.001
```

| Field | Effect |
| --- | --- |
| `classpath` | The model: a catalog `.yml`, a Python class, or `default\|<Name>.yml` ({doc}`../reference/components/models`). |
| `optimizer` | `name` is a `torch.optim` class; the other keys are its arguments. |
| `schedulers` | Learning-rate schedulers, by name. |
| `outputs_criterions` | The losses and metrics, by model output (below). |
| `ModelPatch` | A second level of patching inside the network ({doc}`../usage/large-images`). |
| `dim` | 2 or 3. |
| `allow_head_resize` | Let a checkpoint with a different number of classes initialise the part that matches. |
| `pretrained_from` | Start from another framework's weights ({doc}`../reference/components/models`). |

On a GPU the optimizer uses its fused step when torch has one; `fused: false` keeps the default one.

### `outputs_criterions`

Losses and metrics are attached to named outputs of the model:

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:
    targets_criterions:
      SEG:
        criterions_loader:
          CrossEntropyLoss:
            is_loss: true
            schedulers:
              Constant:
                nb_step: 0
                value: 1
```

- The key is a path in the model's graph (`UNetBlock_0:Head:Conv`).
- `targets_criterions` names the target: a dataset group, or another output by its path.
- `criterions_loader` lists the criteria. Each takes `is_loss`, `group` (criteria of one group are summed),
  `start` and `stop` (the iterations it is active), `accumulation`, `schedulers` (its weight over time), and
  its own arguments.
- Without `is_loss`, a criterion takes its own role: a loss, except `PSNR` and `SSIM`, which are metrics.

Criteria with `is_loss: false` run without gradient tracking: they report a score without keeping
backward buffers. Criteria used as losses keep their gradients.

An output can carry several criteria: `examples/Segmentation` puts a cross entropy on
`UNetBlock_0:Head:Conv` and a Dice loss on `UNetBlock_0:Head:Softmax`. A model with no loss is refused.

## `Trainer.Dataset`

| Field | Default | Effect |
| --- | --- | --- |
| `dataset_filenames` | `["default\|./Dataset:mha"]` | Where the cases are ({doc}`index`). |
| `groups_src` | | The groups to load and their transforms (below). Required in practice. |
| `augmentations` | `null` | Augmentations ({doc}`../reference/components/transforms`). |
| `inline_augmentations` | `false` | Build augmented copies when they are needed, drawing them again each epoch. |
| `Patch` | | Patch extraction (below). |
| `memory_budget` | `auto` | Memory for the data: the dataset stays in RAM if it fits, otherwise it streams (below). |
| `subset` | `null` | Which cases to use: a case-list file, a list of names, or a `start:end` slice. |
| `batch_size` | `1` | Batch size. |
| `num_workers` | `null` | DataLoader workers. `null`: 0 when the dataset is in RAM, up to 4 when it streams. |
| `pin_memory` | `false` | Pinned host memory for faster copies to the GPU. |
| `prefetch_factor` | `null` | Batches prepared ahead per worker (2 by default). |
| `persistent_workers` | `null` | Keep workers between epochs (yes by default; off with inline augmentations). |
| `validation` | `0.2` | A share of the cases, or a list or file of case names. |
| `validation_augmentations` | `false` | Also validate on augmented copies. |
| `shuffle` | `true` | Shuffle the training order. |
| `shuffle_window` | `null` | Keep only this many cases in play at a time (below). |

### `Patch`

| Field | Default | Effect |
| --- | --- | --- |
| `patch_size` | `[128, 128, 128]` | The patch the model sees. A `0` lets KonfAI size that axis. |
| `overlap` | `null` | Overlap between patches: voxels, a fraction, `"20%"`, or one per axis. `null`: 20%. |
| `pad_value` | `null` | Padding past the volume. `null`: the data's minimum. |
| `extend_slice` | `0` | 2.5-D: neighbouring slices added as channels (with `patch_size[0] == 1`). |
| `max_voxels` | `null` | The voxels a patch (`tile`) or a case's coarse grid (`resample`) holds. `null`: sized from the two costs below. |
| `vram_bytes_per_voxel`, `ram_bytes_per_voxel` | `null` | What a voxel costs the pass on the GPU and in RAM. With `max_voxels` unset, KonfAI holds the tighter of the free GPU memory and the rank's RAM budget. |

With a `0` in `patch_size`, KonfAI starts from the whole axis and, on a GPU out-of-memory at the first step,
restarts with the fewest equal patches that fit. A size without `0` is never changed.

### Loading: in RAM or streamed

KonfAI estimates the dataset's size from the image headers:

- if it fits `memory_budget`, every case is loaded and preprocessed once and kept in RAM (the default for
  most datasets);
- otherwise each patch is read from its region of the file (**stream**), or, for a case whose transforms
  need the whole volume, the case is loaded into a small buffer.

The decision and the numbers behind it are printed at startup. A budget below the dataset's size forces
streaming.

| `memory_budget` | Read as |
| --- | --- |
| `24` | 24 GiB |
| `"24GB"`, `"512MB"` | powers of 10 |
| `"32GiB"`, `"512MiB"` | powers of 2 |
| `"1024b"` | bytes |
| `auto` (default) | 80% of the memory (or of the container's limit), split between the processes on the machine |

The budget is per process. The estimate counts 4 bytes per voxel and one copy per augmentation: leave some
headroom.

`shuffle_window` helps a streamed dataset whose cases must be loaded whole: instead of shuffling all patches
(which reloads a case for almost every patch), it keeps `shuffle_window` cases in play and shuffles their
patches together, so each case is read about once per epoch. It works with several GPUs.

### `groups_src`

Each source group on disk gives one or more tensors:

```yaml
groups_src:
  CT:
    groups_dest:
      CT:
        transforms:
          Standardize:
            lazy: false
            mean: None
            std: None
            mask: None
            inverse: false
        patch_transforms: None
        is_input: true
```

`transforms` defines case preprocessing, replayed on regions when streaming; `patch_transforms` runs on each patch, and `is_input` marks the model's inputs
(the others are targets). No `transforms` means the group reaches the model as stored.

## Examples

- `examples/Segmentation/Config.yml`
- `examples/Synthesis/Config.yml`
- `examples/Synthesis/Config_GAN.yml`

Next: {doc}`prediction`, to predict with the trained model.
