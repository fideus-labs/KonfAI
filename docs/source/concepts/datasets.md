# Datasets and groups

KonfAI works with **grouped datasets**. Each case lives in its own directory,
and each file in that directory belongs to a named group such as `CT`, `MR`,
`SEG`, or `MASK`.

## Expected layout

Typical layouts in the repository look like this:

```text
Dataset/
├── CASE_001/
│   ├── CT.mha
│   └── SEG.mha
└── CASE_002/
    ├── CT.mha
    └── SEG.mha
```

```text
Dataset/
├── CASE_001/
│   ├── MR.mha
│   ├── CT.mha
│   └── MASK.mha
└── CASE_002/
    ├── MR.mha
    ├── CT.mha
    └── MASK.mha
```

The concrete file extension is not restricted to `.mha`. KonfAI supports the
extensions listed in `konfai.utils.utils.SUPPORTED_EXTENSIONS`.

Directory-backed formats use the same case/group model:

```text
DicomDataset/CASE_001/CT/*.dcm
OmeDataset/CASE_001/CT.ome.zarr/
```

## Two layers: format readers and dataset loaders

Data handling is split across two packages with distinct responsibilities.

**`konfai/utils/` — format readers.** These modules turn an on-disk file into a
channel-first array plus its physical geometry; they are the only place that
knows about file formats, and the per-backend table (format tokens, optional
extras, API details) lives in {doc}`../reference/components/storage-backends`.

**`konfai/data/` — PyTorch datasets and dataloaders.** These modules build the
`torch.utils.data.Dataset` / `DataLoader` machinery on top of the readers:

| Module | Role |
| --- | --- |
| `konfai/data/data_manager.py` | grouped `Data*` datasets, `GroupTransform`, subset/validation splitting |
| `konfai/data/augmentation.py` | `DataAugmentationsList` — on-the-fly augmentation |
| `konfai/data/patching.py` | `DatasetPatch` — patch extraction and reassembly |

### Never load a full volume into RAM

The central rule of the data layer is that a full volume is **never** loaded into
memory just to feed the model. The DICOM and OME-Zarr readers expose
slice-/patch-level entry points (an OME-Zarr array is already chunked and lazy),
and `DatasetPatch` crops large volumes before they reach the network. Always
reach for lazy or patch-based access; see **When to use dataset patching** below.

### The `Attribute` class

Reading a medical image is not just reading pixels — the physical geometry must
travel with the array so predictions can be written back into the same space.
`konfai.utils.dataset.Attribute` is the container that carries it.

`Attribute` is a `dict[str, Any]` subclass that stores, among other metadata, the
three values that define an image in physical space:

- **`Origin`** — physical position of the first voxel
- **`Spacing`** — voxel size along each axis
- **`Direction`** — the flattened direction-cosine matrix

Numeric values are stored as strings and recovered with `get_np_array(key)` or
`get_tensor(key)`. Keys use a stack-like naming scheme (`Origin_0`, `Origin_1`,
…) so a chain of transforms can push successive geometries and pop them to invert
the chain — which is how KonfAI restores the original geometry when exporting a
prediction.

## `groups_src` and `groups_dest`

Each workflow describes how on-disk groups should be loaded through the
`Dataset.groups_src` mapping.

Example:

```yaml
Dataset:
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
          is_input: true
```

Conceptually:

- `groups_src` identifies what must exist on disk
- `groups_dest` identifies how the loaded tensors are exposed to the workflow
- `is_input: true` marks tensors that are fed into the model

The logic lives in `konfai.data.data_manager.GroupTransform` and the `Data*`
dataset classes.

## Dataset file selectors

The `dataset_filenames` field accepts strings in the form:

- `path`
- `path:format`
- `path:flag:format`

This behavior is implemented in `konfai.data.data_manager.Data.get_data()`.

The most important conventions are:

- `a` means “append / union”
- `i` means “intersection / keep only common cases”

Examples:

- `./Dataset:a:mha`
- `./Predictions/TRAIN_01/Dataset:i:mha`
- `./DicomDataset:a:dicom`
- `./OmeDataset:a:omezarr`

## Training subsets and validation

KonfAI supports several ways to define subsets and validation sets.

From the dataset code, `subset` may be:

- `None`
- a slice string such as `0:10`
- a path to a text file listing case names
- a `~path.txt` exclusion file
- a list of indices
- a list of case names
- a list of case-list files

From the dataset code, `validation` may be:

- `None`
- a float such as `0.2`
- a slice string such as `0:10`
- a path to a text file listing case names
- a list of indices
- a list of case names
- a list mixing case names and case-list files

Three semantics are worth remembering:

- `subset: None` keeps the full dataset;
- `validation: None` disables the split;
- `~` exclusion applies to `subset` but **not** to `validation`.

The `subset` object is applied before validation splitting and can exclude or
include items. The exact logic is implemented by `TrainSubset` and
`PredictionSubset`.

## Caching, augmentation, and patching

At the dataset level, KonfAI can:

- cache transformed data in memory
- generate multiple augmentations per item
- split volumes into patches before they reach the model

This is handled by:

- `konfai.data.data_manager.DataTrain`
- `konfai.data.augmentation.DataAugmentationsList`
- `konfai.data.patching.DatasetPatch`

## When to use dataset patching

Use `Dataset.Patch` when:

- volumes are too large to process at once
- you want 2D, 2.5D, or 3D crops sampled from larger volumes
- you need sliding-window style training or inference

Dataset patching is separate from **model patching**, which applies inside the
network itself. See {doc}`model-graph`.

## Next steps

- {doc}`model-graph` — to attach losses and metrics that target these groups
- {doc}`../reference/components/storage-backends` — format tokens, optional extras, and the DICOM / OME-Zarr APIs
- {doc}`../config_guide/training` — the `Dataset` keys in the context of a full training config
