# Configuration

A KonfAI config is YAML that says which Python objects to build and how they connect.

## Four files, four commands

Each workflow has one file with one root key:

| Command | File | Root key | Writes |
| --- | --- | --- | --- |
| `konfai TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | `Checkpoints/<train_name>/`, `Statistics/<train_name>/` |
| `konfai PREDICTION` | `Prediction.yml` | `Predictor:` | `Predictions/<train_name>/` |
| `konfai EVALUATION` | `Evaluation.yml` | `Evaluator:` | `Evaluations/<train_name>/Metric_*.json` |
| `konfai TRANSFORM` | `Transform.yml` | `Transformer:` | wherever each `Write:` points; logs in `Transforms/<name>/` |

```{mermaid}
flowchart LR
    C[Config.yml<br/>Trainer:]:::cfg --> T([konfai TRAIN]):::cmd
    P[Prediction.yml<br/>Predictor:]:::cfg --> R([konfai PREDICTION]):::cmd
    E[Evaluation.yml<br/>Evaluator:]:::cfg --> V([konfai EVALUATION]):::cmd
    X[Transform.yml<br/>Transformer:]:::cfg --> W([konfai TRANSFORM]):::cmd

    T --> TO[Checkpoints/&lt;train_name&gt;<br/>Statistics/&lt;train_name&gt;]:::out
    R --> RO[Predictions/&lt;train_name&gt;]:::out
    V --> VO[Evaluations/&lt;train_name&gt;<br/>Metric_*.json]:::out
    W --> WO[wherever each Write: points<br/>Transforms/&lt;name&gt;/outputs.json]:::out

```

The model workflows share a workspace named by `train_name`, so use the same `train_name` in the files of
one experiment. Read {doc}`training` first; {doc}`transform` can be read on its own.

## How YAML becomes objects

Each YAML key is a constructor argument. KonfAI reads the class's signature, takes each argument from the
key of the same name, and builds nested objects the same way:

```{mermaid}
flowchart TB
    Y["Trainer:<br/>&nbsp;&nbsp;epochs: 100<br/>&nbsp;&nbsp;Model:<br/>&nbsp;&nbsp;&nbsp;&nbsp;classpath: UNet.yml"]:::yaml
    S["signature of Trainer.__init__<br/>(epochs, model, dataset, …)"]:::sig
    O["Trainer(epochs=100, model=…)"]:::obj
    N["the Model subtree, read the same way<br/>against Network.__init__"]:::obj
    B["the resolved defaults, written back<br/>into the same YAML file"]:::back

    Y -- "the subtree this callable owns" --> S
    S -- "one argument per parameter name" --> O
    O -. "recurses on each nested @config object" .-> N
    O --> B

```

```{note}
**Running a config rewrites it.** Every default is written into the file, so after a run the file is the
complete record of the experiment. A run that fails while building leaves the file as it was; `--init`
writes it on purpose. The file keeps its line endings and permissions.
```

So keys are the constructor's argument names, in `snake_case`. The values are converted to the argument's
type: a number, a string, a list, a nested object. A few rules:

- **A missing key takes its default.** `konfai <COMMAND> --init` writes every default into the file for you
  to edit.
- **`null` (or an empty value) means `None`**, which turns a feature off. It is never replaced by the default.
  `Trainer: {}` (not an empty `Trainer:`) takes every default.
- **A required key must be written**: the error names it (`missing required key '...OneHot.num_classes'`).
- **A value of the wrong shape is refused**, with its path: a block where a number is expected, `2.5` for an
  integer, a list for a single object.
- **A key nothing reads is reported** with the closest valid name. `TRANSFORM` refuses it; the other
  workflows warn, and refuse it when its value would be lost (`epoch: 20` beside `epochs`).

(config-discover)=
## Finding a component's arguments

The tables in this documentation give the main arguments. For the complete list:

- run `konfai <COMMAND> --init`: the config is filled with every argument and its default;
- run `konfai list <kind>` for the exact names;
- or read the class's `__init__`.

## `classpath`

A `classpath` picks the class to build:

```yaml
Model:
  classpath: segmentation.UNet.UNet
```

| Form | Example | Resolves to |
| --- | --- | --- |
| bare name | `Dice`, `Standardize`, `Flip` | KonfAI's package for that kind |
| `module:Class` | `torch:nn:L1Loss`, `monai.losses:DiceLoss`, `Model:UNetpp5` | any importable module, including a `.py` file in the directory you run from |
| `default\|<Name>.yml` | `default\|UNet.yml` | a model of KonfAI's YAML catalog |

Where a bare name is looked up:

| Kind | Package |
| --- | --- |
| criteria | `konfai.metric.measure` |
| transforms | `konfai.data.transform`, then `konfai.data.augmentation` |
| augmentations | `konfai.data.augmentation` |
| models | `konfai.models.python` |
| learning-rate schedulers | `torch.optim.lr_scheduler`, then `konfai.metric.schedulers` |
| loss-weight schedulers | `konfai.metric.schedulers` |
| `patch_combine` | `konfai.data.patching` |
| reductions (`combine`, `Reduce`) | `konfai.data.reduction` |

Optimizers are `torch.optim` classes, named by `name`. A model's arguments go under its class name
(`Trainer.Model.UNetpp5`); `@config("Key")` on a class moves them to `Key`. `examples/Synthesis` shows a
local model (`Model.py`) and transform (`UnNormalize.py`). {doc}`../usage/custom-models` explains how to
write your own.

A value written `default|...` in the code is a default the config can override: `train_name` defaults to
`default|TRAIN_01`, meaning `TRAIN_01` unless you write another name.

## The `Dataset` block

Every workflow reads its data through a `Dataset:` block. The keys below are shared; each workflow page lists
its own (`batch_size`, `Patch`, `memory_budget`).

### Layout

One folder per case, one file per group:

```text
Dataset/
├── CASE_001/
│   ├── CT.mha
│   └── SEG.mha
└── CASE_002/
    ├── CT.mha
    └── SEG.mha
```

A DICOM series is a folder (`CASE_001/CT/*.dcm`), an OME-Zarr store a folder too (`CASE_001/CT.ome.zarr/`).
Any format of {doc}`../reference/components/storage-backends` works.

### `dataset_filenames`

A list of `path`, `path:format` or `path:flag:format`:

- `./Dataset:mha`, `./DicomDataset:dicom`, `./OmeDataset:omezarr`;
- the flag `a` adds a dataset's cases to the others (union), `i` keeps only the cases present in all
  (intersection): `./Predictions/TRAIN_01/Dataset:i:mha`.

### `groups_src` and `groups_dest`

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

`groups_src` names the groups read from disk. Each gives one or more tensors in `groups_dest`, each with its
own `transforms`. `is_input: true` marks the model's inputs.

### `subset` and `validation`

Both take the same selectors:

- a slice, `0:10` (`0:-2` counts from the end);
- a case name, or a list of names;
- a text file listing case names, one per line;
- `~file.txt` to exclude the cases it lists;
- a list mixing these.

`subset` picks the cases to use, `validation` the ones held out for validation. `validation` also takes a
share, such as `0.2`: the last cases of the run order, at least one. `null` keeps all cases, or disables
the split.

Case-list files are read in UTF-8. With non-ASCII case names, run under a UTF-8 locale
(`LC_ALL=C.UTF-8`).

## When a key does not bind as expected

- The root key must match the command (`Trainer:` for `TRAIN`).
- Keys must be the constructor's argument names.
- A local `classpath` module must be importable from the directory you run from.
- The YAML nesting must follow the objects: a model's arguments under its class name.

## Environment variables

The CLI sets `KONFAI_config_file` (the config path) and `KONFAI_CONFIG_MODE` (`Done` to bind from the file,
`Import` to build objects without reading it). Set both yourself only to build configurable classes
outside the CLI, from a notebook or a test.

## Next steps

- {doc}`training`: every key of `Config.yml`.
- {doc}`../reference/components/models`: how model outputs are named for losses and metrics.
- {doc}`../usage/python-api`: the same workflows from Python, with the config as a dictionary.
