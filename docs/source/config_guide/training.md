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
  epochs: 100
```

## Running it

From the directory that contains `Config.yml`:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

If you do not have a GPU available, use `--cpu 1` instead of `--gpu 0`.

Add `-tb` to enable TensorBoard — KonfAI allocates a free local port
automatically:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml -tb
```

Resume from an existing checkpoint with `RESUME`:

```bash
konfai RESUME -y --config Config.yml \
  --model Checkpoints/SEG_BASELINE/<checkpoint>.pt
```

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
| `manual_seed` | int or null | `None` | No | Sets the random seed when provided. |
| `epochs` | int | `100` | No | Number of training epochs. |
| `it_validation` | int or null | `None` | No | Validation and checkpoint interval in iterations. |
| `autocast` | bool | `false` | No | Enables AMP during training. |
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
| `Patch` | mapping | Optional | Enables model-level patching via `ModelPatch`. |
| `dim` | int | Model-dependent | Declares whether the network operates in 2D or 3D. |

### `outputs_criterions`

This is the most important training structure after the dataset definition.

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:
    targets_criterions:
      SEG:
        criterions_loader:
          torch:nn:CrossEntropyLoss:
            is_loss: true
            schedulers:
              Constant:
                nb_step: 0
                value: 1
```

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
| `groups_src` | mapping | required in practice | Maps on-disk groups to loaded tensors. |
| `augmentations` | mapping or null | one default augmentation list | Data augmentations sampled during training. |
| `inline_augmentations` | bool | `false` | Keeps base samples cached and generates augmentation tensors only when an augmented sample is requested; augmentation states are re-sampled on each epoch. |
| `Patch` | mapping or null | `DatasetPatch()` | Dataset-level patch extraction. |
| `use_cache` | bool | `true` | Holds the whole prepared dataset in RAM. `false` opens the stream/buffer path: a streamable chain reads each patch from disk, and any other loads whole cases into a bounded FIFO. |
| `memory_budget` | number / string / null | `null` | RAM budget from which `use_cache` is derived. `null` keeps `use_cache` as given. |
| `subset` | object | `TrainSubset()` | Restricts which cases are used. |
| `batch_size` | int | `1` | Batch size. |
| `num_workers` | int or null | `None` | Number of DataLoader workers. `None` resolves to `0` under `use_cache: true`, and to `max(1, min(cpu_count, 4))` otherwise. A `KonfAIInference` transform in any group forces `0` whatever the value. |
| `pin_memory` | bool | `false` | Enables pinned host memory for DataLoader batches. |
| `prefetch_factor` | int or null | `None` | Prefetched batches per worker. Applies only when workers are enabled, where `None` resolves to `2`. |
| `persistent_workers` | bool or null | `None` | Keep workers alive across epochs. Applies only when workers are enabled, where `None` resolves to `true`. |
| `validation` | float / string / list / null | `0.2` | Validation split or explicit validation set. |
| `validation_augmentations` | bool | `true` | Whether validation also iterates over augmented variants. Set `false` to validate only on base (non-augmented) samples. |
| `shuffle` | bool | `true` through subset | Shuffles the training sampler. |
| `shuffle_window` | int or null | `null` through subset | Locality-aware training order: shuffles cases, then keeps this many cases in play at a time with their patches shuffled together. Single-process training only. |

### Cache, stream, and buffer

`use_cache` picks how a loader turns a case into patches. It applies to the
training and validation subsets alike.

- **Cache** (`use_cache: true`, the default). Every case is loaded, preprocessed,
  and held in RAM before the first epoch; patches are cut from the resident
  volume. RAM follows the dataset. `num_workers` defaults to `0` here.
- **Stream** (`use_cache: false`, patch-compatible preprocessing). Each patch is
  read from its own region of the source file. No volume is materialized.
- **Buffer** (`use_cache: false`, preprocessing that needs the whole volume).
  The case is loaded whole into a FIFO of `batch_size + 1` cases —
  `max(batch_size + 1, shuffle_window)` when a window is set — evicting the
  oldest.

Nothing in YAML selects streaming. KonfAI derives it from the declared transforms
and augmentations, per case and per augmented copy, so stream and buffer coexist
in one run: a group that needs its whole volume loads that case, while the others
still stream.

A cached case is resident, so its patches are cut from RAM even when its chain
would stream.

A 16 GiB uncompressed `.mha` at patch 64³, batch 2, 2 workers and
`use_cache: false`, run under an 8 GiB memory cap, streams at a peak anonymous
RSS of 0.46 GiB, flat across epochs, with one batch (2 MiB) resident on the GPU.

### `memory_budget`

`memory_budget` derives `use_cache` from a RAM budget. KonfAI estimates the
dataset size from image headers alone (no voxel read), caches when the per-rank
share fits the budget, and takes the streaming/buffer path otherwise. A budget
decides the regime on its own: the declared `use_cache` is ignored. The decision
is made once on the launcher, before any rank is spawned, and the estimate, the
budget, its source, and the chosen regime are printed. `null` (the default)
leaves `use_cache` exactly as declared.

| Value | Read as |
| --- | --- |
| `24`, `"24"` | 24 GiB — a bare number is GiB |
| `"24GB"`, `"512MB"` | decimal, 10^n |
| `"32GiB"`, `"512MiB"` | binary, 2^n |
| `"1024b"` | bytes |
| `"auto"` | 80% of the detected RAM |

Case is folded and the space before the unit is optional: `"32 gib"` and
`"32GiB"` name the same budget.

An explicit budget is **per rank**: the comparison is
`dataset_size / world_size <= budget`, because cases are sharded across ranks.
`"auto"` divides the detected memory by `world_size` too, so the ranks cancel and
it reduces to "does the whole dataset fit 80% of the detected memory". `"auto"`
takes whichever is tighter of the **cgroup limit** — set under a container or
SLURM — and the host's available RAM; the log names which one won.

```{warning}
The dataset size is an estimate, not a guarantee. It sums `prod(header_shape) x 4`
bytes over the source groups: it models a float32 cached tensor, so a `uint8`
source is over-counted and a `float64` one under-counted. It reads the raw header
shape and ignores transforms that shrink (resample-down, crop) or grow (pad,
one-hot) the tensor. It also counts one copy per case, while the cache holds every
augmented copy of every case — with augmentations declared, the real footprint is a
multiple of the estimate. `inline_augmentations: true` defers those copies rather
than dropping them: they are built on demand and released once per epoch, so the
peak is the same. Caching also peaks above its steady state while it runs. Leave
headroom.
```

### `shuffle_window`

Each non-streamable case is loaded into the FIFO buffer, so a global patch shuffle
reloads a volume once per patch that lands after an eviction. A window keeps
`shuffle_window` cases in play at a time — their patches shuffled together, all
emitted before advancing — which reads each volume about once per epoch. `1` is
perfect locality and no decorrelation; larger windows trade one back for the other.

The window applies to the training loader only. Validation is scored over the whole
subset whatever the order, so it follows `shuffle` without a window.

The window resolves back to a plain global shuffle — byte for byte — when it is
`null` (the default), when it is `>=` the number of cases, or when `num_workers`
exceeds the number of cases. Under a window, cases are partitioned across workers
and the per-worker batches interleaved, so every volume is read by exactly one
worker. The buffer is sized to hold the window, so a non-streamable run holds
`max(batch_size + 1, shuffle_window)` volumes per worker.

```{warning}
`shuffle_window` is for single-process training. Each rank windows its own shard,
so the ranks can disagree on the number of batches in an epoch and the collective
will hang. Leave it `null` for multi-GPU runs.
```

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

- {doc}`../concepts/datasets` — the shared `dataset_filenames`, `groups_src`,
  `subset`, and `validation` conventions used above.
- {doc}`../concepts/model-graph` — how module names become the output paths
  used in `outputs_criterions`.
- {doc}`prediction` — to configure inference with the trained model.
