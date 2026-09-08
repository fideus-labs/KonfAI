# Training configuration

Training configuration lives under the `Trainer` root object.

```yaml
Trainer:
  Model:
    classpath: UNet.yml
    UNet:
      ...
  Dataset:
    ...
  train_name: SEG_BASELINE
  epochs: 100          # examples/Segmentation ships 5, sized so a first run finishes
```

## Running it

From the directory that contains `Config.yml`:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

If you do not have a GPU available, use `--cpu 1` instead of `--gpu 0`.

Add `-tb` to enable TensorBoard: KonfAI allocates a free local port
automatically:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml -tb
```

TensorBoard is optional: without the `tensorboard` extra the run trains
normally with a no-op writer and one warning naming
`pip install konfai[tensorboard]`; only the scalar logs are lost.

Resume from an existing checkpoint with `RESUME`. Checkpoints are named after the
moment they were written, so substitute the one training produced: `--model`
takes exactly one:

```bash
konfai RESUME -y --config Config.yml \
  --model Checkpoints/SEG_BASELINE/2026_08_03_02_36_00.pt
```

A run preserves its preparation seed: the seed every preparation draw comes from
(the train/validation split first) is recorded in
`Statistics/<train_name>/Seed.txt`, and RESUME of an unseeded run reads it
back, so resuming never re-splits the cohort. Set `manual_seed` only to pick
the seed yourself. A save on an exceptional exit is named `crash_<date>.pt` and
sits outside the `save_checkpoint_mode` pruning: never a contender for best,
and yours to delete.

New checkpoints distinguish completed epochs from intermediate snapshots using
`resume.version: 1` and `resume.kind`. An eligible completed epoch stores
`resume.next_epoch`: resuming a checkpoint written after epoch 2 starts epoch 3,
without replaying epoch 2 or adding an initial validation pass. The optimizer,
scheduler/scaler, update counters, EMA and early-stopping state are restored.
Each rank's Python, NumPy and PyTorch generator states are restored after startup;
CUDA generator states are included for GPU training.
The bounded metric windows and running totals/counts are also restored per rank,
including the historical mean read by `ReduceLROnPlateau`. Criterion-weight
schedules shipped with KonfAI (`Constant`, `CosineAnnealing`) use the restored
iteration counter; custom stateful criteria/schedules need their own state contract.

With both `save_checkpoint_mode: BEST` and `ALL`, a separate `resume_latest.pt`
keeps the latest eligible epoch boundary:

```bash
konfai RESUME -y --config Config.yml \
  --model Checkpoints/SEG_BASELINE/resume_latest.pt
```

In `BEST`, the best scored model remains the dated `.pt` file. It can refer to a
different epoch from `resume_latest.pt`. These files share storage while they
contain the same checkpoint; otherwise, plan for up to two checkpoints of disk
space. `ALL` retains its dated checkpoints, including completed epochs even when
the validation interval does not land on the final batch. `it_validation` still
controls scored checkpoints within the epoch.

Intermediate saves and crash saves contain model weights for prediction, but
new-format `RESUME` refuses them because sample positions and pending gradients
are not serialized. An epoch is eligible only when **every optimizer's gradient
accumulation window has closed**. If an epoch ends with pending gradients, training
continues with those gradients into the next epoch and emits a warning; it adds
no optimizer step and retains the previous `resume_latest.pt`. For example,
3 batches per epoch with `nb_batch_per_step: 2` produce eligible boundaries after
epochs 2, 4, and so on. A run stopped before its first eligible boundary has no
new-format continuation checkpoint. Choose a sufficient epoch count or a batch
count/cadence that closes the windows. Checkpoints without the versioned cursor
keep the historical behavior: their stored `epoch` is replayed, and exact
continuation is not promised.

Bit-for-bit continuation is tested on CPU with the same configuration and data,
`num_workers: 0`, no augmented copies, and stochastic model operations using the
saved global generators. The rank count and number of batches must match.
DataLoader workers' RNG/cache state and augmentation draws/cache state are not
serialized: those configurations still continue at `next_epoch`, with a warning
that stochastic replay is not exact. Custom generators, changed data/configuration,
and nondeterministic GPU kernels are also outside the exact-replay guarantee.

You can also change the output directories:

```bash
konfai TRAIN -y --config Config.yml \
  --checkpoints-dir ./Checkpoints \
  --statistics-dir ./Statistics
```

## Top-level fields

| Field | Type | Default in code | Required | Effect |
| --- | --- | --- | --- | --- |
| `Model` | mapping | `ModelLoader()` | Yes | Selects and configures the model graph. |
| `Dataset` | mapping | `DataTrain()` | Yes | Defines training data loading, transforms, augmentation, and patching. |
| `train_name` | string | `TRAIN_01` | No | Names the run and its output folders. |
| `manual_seed` | int or null | `None` | No | Seeds training generators and preparation. With `None`, TRAIN still records its preparation seed in `Statistics/<train_name>/Seed.txt` for RESUME's cohort split; this does not promise deterministic GPU training. |
| `epochs` | int | `100` | No | Number of training epochs. |
| `it_validation` | int or null | `None` | No | Validation and checkpoint interval in iterations. |
| `it_lr_update` | int or null | `None` | No | Scheduler-step interval in iterations. `None` steps once per epoch (it resolves to the training dataloader's length). Every resolved config on disk carries this key. |
| `autocast` | bool | `false` | No | Enables AMP during training. On a 3D UNet (five levels to 256 channels, 96 cubed patches, batch 2, twenty 128 cubed cases, one RTX PRO 5000) an epoch runs 11.7 s against 27.0 s in fp32. The shipped Segmentation example trains with it on, with `channels_last`: 17.0 ms per step against 38.2 in fp32 (`benchmarks/perf/bench_train_step.py`). On a small 3D toy (64 cubed, batch 4, 1.4 M parameters) autocast alone was 49 % slower than fp32 and the pair 14 % faster; on SynthRAD 2025's UNet++ (2.5D, five slices of 320 squared, batch 32, 26 M parameters) a step goes 852 ms in fp32, 521 with autocast alone, 741 with `channels_last` alone and 387 with both (2.2x); on CURVAS's ResidualEncoderUNet (3D, nnU-Net style, 102 M parameters, batch 2 of 128x160x160) autocast alone is 2.0x (1244 to 617 ms, 17.6 to 9.6 GB) and `channels_last` costs 20 to 26 % with or without it (1532 ms alone, 779 with both). Turn `autocast` on; `channels_last` depends on the model, so measure it on yours with `benchmarks/perf/bench_train_step.py --model-classpath`. |
| `channels_last` | bool | `false` | No | Lays the convolution weights and inputs out channels-last (4-D and 5-D). cuDNN picks its kernels by layout: with `autocast` the shipped Segmentation example predicts 1.25x faster and the 3D UNet above trains an epoch in 10.0 s against 11.7 s, while CURVAS's ResidualEncoderUNet trains 26 % slower with it (see `autocast`), so measure before turning it on; the kernels chosen differ, so labels can move at boundaries (3199 of 58.4 million voxels in fp32 on that example). |
| `gradient_checkpoints` | list or null | `None` | No | Activates gradient checkpointing on selected modules. |
| `gpu_checkpoints` | list or null | `None` | No | Pins selected modules to dedicated GPUs. |
| `ema_decay` | float | `0` | No | Enables exponential moving average tracking when greater than zero. |
| `data_log` | list or null | `None` | No | TensorBoard logging directives for dataset groups or model outputs. |
| `EarlyStopping` | mapping or null | `None` | No | Configures early stopping. |
| `save_checkpoint_mode` | string | `BEST` | No | `BEST` keeps the best checkpoint, `ALL` keeps every saved checkpoint. |

## `Trainer.Model`

`Trainer.Model` always starts with a `classpath`, then a section named after the
selected class.

```yaml
Model:
  classpath: UNet.yml
  UNet:
    optimizer:
      name: AdamW
      lr: 0.001
```

Common nested fields used by built-in and local models:

| Field | Type | Required | Effect |
| --- | --- | --- | --- |
| `classpath` | string | Yes | Selects the model class to import. |
| `<SelectedClass>` | mapping | Yes | Constructor arguments for the chosen class. |
| `optimizer` | mapping | Usually | Optimizer configuration passed through `OptimizerLoader`. |
| `schedulers` | mapping | Optional | Learning-rate schedulers keyed by classpath. |
| `outputs_criterions` | mapping | Usually | Declares losses and metrics attached to specific model outputs. |
| `ModelPatch` | mapping or null | Optional | Enables a second, model-level patch inside the network (a config key named `ModelPatch`, distinct from `Dataset.Patch`). |
| `dim` | int | Model-dependent | Declares whether the network operates in 2D or 3D. |

### `optimizer`

`name` selects a class from `torch.optim`; the rest of the section is that
class's own signature, so a resolved config carries every one of its keys.

`fused` and `foreach` both left at `None` ask for the widest batched step the
optimizer implements: fused where torch has one, foreach otherwise. The run
decides, not the parameters: a run that places the graph on a GPU takes the
batched step, a CPU run keeps torch's own default. On the Segmentation example
(1.93 M parameters in 40 tensors, one RTX PRO 5000) an AdamW step costs
0.059 ms of host time fused against 0.188 ms foreach, in two kernels against
eight. A fused step sums in another order, so its parameters drift from the
foreach ones: 3.7e-05 after 100 steps, 8.3e-06 of their own scale. Write
`fused: false` to pin the foreach step.

### `outputs_criterions`

This is the most important training structure after the dataset definition.

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

Several criteria can share one output, and several outputs can each carry their own: `examples/Segmentation` pairs the cross entropy above with a `Dice` loss on
`UNetBlock_0:Head:Softmax`, because the two want different forms of the same head.

Structure:

- output key → model output or module path
- `targets_criterions` → one or more target groups
- `criterions_loader` → one or more criteria for that target
- each criterion can define `is_loss`, `group`, `start`, `stop`, `accumulation`, and scheduler weights

## `Trainer.Dataset`

Training datasets are instantiated through `DataTrain`.

Common fields:

| Field | Type | Default in code | Effect |
| --- | --- | --- | --- |
| `dataset_filenames` | list[str] | `["default\|./Dataset:mha"]` | Dataset sources and selection mode. |
| `groups_src` | mapping | `{Labels: Group()}` | Maps on-disk groups to loaded tensors. The default binds a single group named `Labels`, which is almost never what you want: treat it as required. |
| `augmentations` | mapping or null | one default augmentation list | Data augmentations sampled during training. |
| `inline_augmentations` | bool | `false` | Keeps base samples cached and generates augmentation tensors only when an augmented sample is requested; augmentation states are re-sampled on each epoch. |
| `Patch` | mapping or null | `DatasetPatch()` | Dataset-level patch extraction. |
| `memory_budget` | number / string / null | `null` = `auto` | RAM budget the loading regime is derived from: the dataset caches when its per-rank share fits, streams otherwise. An absent key (`null`) means `auto`: 80% of the detected memory decides. |
| `subset` | string / list / null | `null` | Restricts which cases are used: a flat selector (a case-list file, a list of names, or a `start:end` slice), not a nested object. `shuffle` and `shuffle_window` are sibling `Dataset` keys. |
| `batch_size` | int | `1` | Batch size. |
| `num_workers` | int or null | `None` | Number of DataLoader workers. `None` resolves to `0` on the cache regime, and to `max(1, min(cpu_count, 4))` on the stream/buffer regime. A `KonfAIInference` transform in any group forces `0` whatever the value. |
| `pin_memory` | bool | `false` | Enables pinned host memory for DataLoader batches. |
| `prefetch_factor` | int or null | `None` | Prefetched batches per worker. Applies only when workers are enabled, where `None` resolves to `2`. |
| `persistent_workers` | bool or null | `None` | Keep workers alive across epochs. Applies only when workers are enabled, where `None` resolves to `true`. **Forced to `false`**: an explicit `true` included, when `inline_augmentations` is on with any augmentation declared, because persistent workers freeze the per-epoch redraw. |
| `validation` | float / string / list / null | `0.2` | Validation split or explicit validation set. |
| `validation_augmentations` | bool | `true` | Whether validation also iterates over augmented variants. Set `false` to validate only on base (non-augmented) samples. |
| `shuffle` | bool | `true` through subset | Shuffles the training sampler. |
| `shuffle_window` | int or null | `null` through subset | Locality-aware training order: shuffles cases, then keeps this many cases in play at a time with their patches shuffled together. Safe under DDP. |

### Cache, stream, and buffer

The loading regime picks how a loader turns a case into patches. It applies to
the training and validation subsets alike. Training defaults to the cache;
`memory_budget` switches the regime from the dataset's measured size.

- **Cache** (the training default). Every case is loaded, preprocessed,
  and held in RAM before the first epoch; patches are cut from the resident
  volume. RAM follows the dataset. `num_workers` defaults to `0` here.
- **Stream** (budget exceeded, patch-compatible preprocessing). Each patch is
  read from its own region of the source file. No volume is materialized.
- **Buffer** (budget exceeded, preprocessing that needs the whole volume).
  The case is loaded whole into a FIFO of `batch_size + 1` cases (`max(batch_size + 1, shuffle_window)` when a window is set) evicting the
  oldest.

Nothing in YAML selects streaming. KonfAI derives it from the declared transforms
and augmentations, per case and per augmented copy, so stream and buffer coexist
in one run: a group that needs its whole volume loads that case, while the others
still stream.

A cached case is resident, so its patches are cut from RAM even when its chain
would stream.

A 16 GiB uncompressed `.mha` at patch 64³, batch 2, 2 workers on the streaming
regime, run under an 8 GiB memory cap, streams at a peak anonymous
RSS of 0.46 GiB, flat across epochs, with one batch (2 MiB) resident on the GPU.

### `memory_budget`

`memory_budget` derives the loading regime from a RAM budget. KonfAI estimates
the dataset size from image headers alone (no voxel read), caches when the
per-rank share fits the budget, and takes the streaming/buffer path otherwise: a budget below the dataset's size therefore forces streaming. The decision
is made once on the launcher, before any rank is spawned, and the estimate, the
budget, its source, and the chosen regime are printed. `null` (the default)
means `auto`: the detected memory decides -- a dataset that fits caches exactly
as before, one that does not streams instead of overrunning the node.

| Value | Read as |
| --- | --- |
| `24`, `"24"` | 24 GiB: a bare number is GiB |
| `"24GB"`, `"512MB"` | decimal, 10^n |
| `"32GiB"`, `"512MiB"` | binary, 2^n |
| `"1024b"` | bytes |
| `"auto"` | 80% of the detected RAM |

Case is folded and the space before the unit is optional: `"32 gib"` and
`"32GiB"` name the same budget.

An explicit budget is **per rank**: the comparison is
`dataset_size / world_size <= budget`, because cases are sharded across ranks.
`"auto"` divides the detected memory by the ranks sharing **one node**
(`KONFAI_LOCAL_RANKS`), not by `world_size`. On a single node the two coincide, the
ranks cancel, and it reduces to "does the whole dataset fit 80% of the detected
memory"; across nodes the numerator still uses `world_size`, so they do not cancel. `"auto"`
takes whichever is tighter of the **cgroup limit** (set under a container or
SLURM), and the host's available RAM; the log names which one won.

The dataset size is an estimate, not a guarantee. It sums `prod(header_shape) x 4`
bytes over the source groups: it models a float32 cached tensor, so a `uint8`
source is over-counted and a `float64` one under-counted. It reads the raw header
shape and ignores transforms that shrink (resample-down, crop) or grow (pad,
one-hot) the tensor. It also counts one copy per case, while the cache holds every
augmented copy of every case, with augmentations declared, the real footprint is a
multiple of the estimate. `inline_augmentations: true` defers those copies rather
than dropping them: they are built on demand and released once per epoch, so the
peak is the same. Caching also peaks above its steady state while it runs. Leave
headroom.

### `shuffle_window`

Each non-streamable case is loaded into the FIFO buffer, so a global patch shuffle
reloads a volume once per patch that lands after an eviction. A window keeps
`shuffle_window` cases in play at a time (their patches shuffled together, all
emitted before advancing), which reads each volume about once per epoch. `1` is
perfect locality and no decorrelation; larger windows trade one back for the other.

The window applies to the training loader only. Validation is scored over the whole
subset whatever the order, so it follows `shuffle` without a window.

The window resolves back to a plain global shuffle (byte for byte), when it is
`null` (the default), when it is `>=` the number of cases, or when `num_workers`
exceeds the number of cases. Under a window, cases are partitioned across workers
and the per-worker batches interleaved, so every volume is read by exactly one
worker. The buffer is sized to hold the window, so a non-streamable run holds
`max(batch_size + 1, shuffle_window)` volumes per worker.

`shuffle_window` works under DDP: the sampler's length is the mapping's length, so
the window reorders without changing the count, and each training shard is padded to
the longest one. Ranks stay in step.

### Free patch axes: sizing by measurement

`Patch.patch_size` accepts the same free-axis convention as prediction: `0`
entries are sized by the framework, starting at the full extent and shrinking
only on a CUDA out-of-memory: the failed step (forward, backward and optimizer)
already measured its cost, so the shrink lands near the target and the run
restarts on the re-planned grid. Training runs out of memory at the first step
when it does at all (its memory is maximal from step one), so a restart loses no
meaningful work. Under DDP the failing ranks agree on the per-axis minimum
before restarting, so every rank trains the same grid; a single rank failing
alone dies at the collective timeout, exactly as an unhandled OOM does. A
`patch_size` without a `0` is never resized: the OOM propagates.

### `groups_src`

Each source group contains one or more destination groups:

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

Use this section to define:

- what exists on disk
- preprocessing transforms
- patch-specific transforms
- whether the tensor is a model input

## Examples

The most practical examples in the repository are:

- `examples/Segmentation/Config.yml`
- `examples/Synthesis/Config.yml`
- `examples/Synthesis/Config_GAN.yml`

## Next steps

- {doc}`index`: the shared `dataset_filenames`, `groups_src`,
  `subset`, and `validation` conventions used above.
- {doc}`../reference/components/models`: how module names become the output paths
  used in `outputs_criterions`.
- {doc}`prediction`: to configure inference with the trained model.
