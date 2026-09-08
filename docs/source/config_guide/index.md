# Configuration

KonfAI is a configuration-driven object builder. A YAML file does not pass
values into a fixed script: it decides which Python classes are instantiated
and how they are connected, and reading it resolves every default back into
the file, so a run leaves a complete record of the experiment. This page is
the engine behind every `Trainer`, `Predictor`, `Evaluator` and `Transformer`,
the binding rules, where a bare name resolves, and the `Dataset` conventions
the four workflow pages share. Read it when a key is not binding the way you
expect, or before you expose a custom class to YAML.

```{note}
Reading a config **mutates it**: loading a run resolves every default and
rewrites the YAML file in place, so the file on disk becomes the fully-resolved
record of the experiment. One consequence: a `None` value round-trips as the
literal string `"None"`: it is written back as `"None"` and reparsed to
`None` on the next read. An explicit `name: null` (or an empty `name:`) also
binds `None`: null is the disabled spelling and is never replaced by the
default.
```

## Four files, four commands

The root key of a YAML file selects the workflow object, and one command reads
each file:

| Command | File | Root key | Class | Writes |
| --- | --- | --- | --- | --- |
| `konfai TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | `konfai.trainer.Trainer` | `Checkpoints/<train_name>/`, `Statistics/<train_name>/` |
| `konfai PREDICTION` | `Prediction.yml` | `Predictor:` | `konfai.predictor.Predictor` | `Predictions/<train_name>/` |
| `konfai EVALUATION` | `Evaluation.yml` | `Evaluator:` | `konfai.evaluator.Evaluator` | `Evaluations/<train_name>/Metric_*.json` |
| `konfai TRANSFORM` | `Transform.yml` | `Transformer:` | `konfai.transformer.Transformer` | wherever each `Write:` points, plus `Transforms/<name>/outputs.json` |

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

One root key per file, one command per file. The same reflection engine builds
each one: only the root key changes what it builds. The model workflows write
into a single workspace keyed by `train_name`, so the `train_name` in each
config file must name the run you intend to touch. `TRANSFORM` is the
exception: it is keyed by `name`, and only its log, its plan and a copy of its
config land in the workspace (`Transforms/<name>/`): the data goes wherever
each `Write:` stage says, which that run directory records in `outputs.json`.

Each page of this guide starts with the commands that run its workflow. Read
{doc}`training` first: it introduces the structures (`Model`, `Dataset`,
`outputs_criterions`) the other pages reuse. Then {doc}`prediction` and
{doc}`evaluation` as you reach those workflows. {doc}`transform` stands apart:
dataset preparation has no `Model:` block, so it is the one page you can read
on its own if you only process data. The pages focus on the fields that are
stable and visible in the codebase and the shipped examples; for built-in
models, transforms and metrics, the exact available parameters depend on the
selected classpath, and {ref}`config-discover` says how to get the exhaustive
list.

## How YAML becomes Python objects

This behavior is implemented by `konfai.utils.config.Config`, `config()`, and
`apply_config()`.

In practice, the mapping is straightforward:

1. a class or function is optionally annotated with `@config("...")`
2. `apply_config()` inspects the constructor signature
3. YAML fields are matched against constructor parameter names
4. nested objects are recursively instantiated from nested mappings

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

The write-back is the last arrow, and it is why reading a config changes the file:
whatever the signature defaulted is now spelled out in it.

This is why KonfAI configuration keys should generally:

- use **snake_case**
- match the actual Python constructor arguments
- stay close to the shipped examples when you introduce a custom class

One important detail about models: the nesting is the same with or without a
decorator. `@config()` defaults to the class name, and an *undecorated* class gets
its class name appended by the model loader, so `classpath: Model:UNetpp5` reads
from `Trainer.Model.UNetpp5` either way. A decorator only *renames* that subtree; it
never removes it. `examples/Synthesis` shows the shape: `Model.py` decorates
`UNetpp5` with `@config()`, and `Config.yml` nests its parameters under
`Trainer.Model.UNetpp5`.

## Runtime environment variables

Two environment variables drive configuration loading at runtime. Both are read
directly from `os.environ` by `konfai.utils.config.Config`:

| Variable | Meaning |
| --- | --- |
| `KONFAI_config_file` | Path to the active YAML config file. `Config.__init__` reads it directly, so it must be set before any configurable object is built. |
| `KONFAI_CONFIG_MODE` | Controls what happens when the config file or individual keys are missing. See **Config modes** below. |

The KonfAI CLI sets both variables for you from the `--config` argument. You only
need to set them by hand when you call configurable classes directly: for
example from a test or a notebook (see the testing notes in `AGENTS.md`).

## The `apply_config` decorator and `Config` context manager

Two cooperating pieces implement YAML → Python binding:

- **`Config(key)`** is a context manager. On `__enter__` it loads the YAML file
  named by `KONFAI_config_file`, walks down the dot-separated `key` to the
  matching subtree, and exposes it. On `__exit__` it merges the visited subtree
  back into the file, so a run that materializes defaults also *records* them in
  the YAML for reproducibility.
- **`apply_config("Root.Path")`** is a decorator placed on a configurable class
  or function. When the decorated object is called, it opens a `Config` for its
  subtree and binds arguments from the YAML before the callable runs.

A class additionally annotated with `@config("Name")` overrides the YAML key it
binds to; without it, the key defaults to the object's own name.

### How YAML keys map to arguments via reflection

`apply_config` does not hard-code any field names. It inspects the target with
`inspect.signature()` and, for each parameter, reads a value from the active YAML
subtree using the parameter's **type annotation** to decide how to convert it:

- `int`, `float`, `bool`, `str`, `torch.Tensor`: cast directly from the YAML scalar
- `Literal[...]`: validated against the allowed set (an invalid value raises `ConfigError`)
- `pathlib.Path`: wrapped as a `Path`; a non-existent path only logs a warning
- `list[...]` / `dict[str, ...]`: parsed element-wise
- a nested configurable class: instantiated recursively by re-entering
  `apply_config` on the nested subtree

Scalar conversion also applies inside typed lists and dictionaries, including
container alternatives of a union. For example, quoted `"false"` becomes
`False` in `bool`, `list[bool]`, `dict[str, bool]`, and `list[bool] | str`.
A union first preserves a matching value and its element types: `0.25` stays
a float in `int | float`, and `["001"]` stays a string list in
`list[int] | list[str]`. Invalid elements are reported with their key or index.
String sentinels such as `auto` remain strings when the union permits them;
`None` remains available for optional values. Explicit YAML values remain in
the resolved record and bind the same way when read again.

Because the parameter *names* are the YAML keys, configuration keys should use
the exact constructor argument names (typically `snake_case`). A missing key
falls back to the parameter default, or to a `default|...` marker when one is
provided (see below).

## Keys nothing reads

The binder reads a key when a parameter names it and materializes the default
when none does, so a key nothing reads (a typo, a parameter an older version
had) is carried along and the default used in its place. Each workflow builder
reads its file inside `konfai.utils.config.strict_config(root)`, which records,
level by level, what the file holds against what the binder read, and reports
the difference by path with the keys read at that level: `TRANSFORM` refuses
(its config is the deliverable), `TRAIN`/`PREDICTION`/`EVALUATION` warn, since
files written back by earlier versions carry such keys. The check closes when
the builder returns, so everything a workflow reads from its file is bound at
construction.

## Config modes

`KONFAI_CONFIG_MODE` selects how KonfAI reacts to a missing file:

| Mode | Behavior |
| --- | --- |
| `Done` | Normal run mode. The config file must already exist; values are read and the visited subtree is written back. A missing file raises `ConfigError` naming `konfai <COMMAND> --init` as the way to generate one. |
| `Import` | Skip config binding entirely. The decorated object is called with the arguments it was given, without reading the YAML: used when importing or constructing classes outside the config-driven flow. |

An *unknown* value behaves like `Done`. An **unset** `KONFAI_CONFIG_MODE` is
different: `apply_config` binds nothing at all (as under `Import`), and using
`Config` directly raises `KeyError` on exit. Tests that build configurable objects
directly must therefore set **both** variables explicitly.

**Generating a config is a CLI verb, not a mode**: `konfai <COMMAND> --init`
creates the file when missing (seeded with its root key), binds the workflow
once so every default resolves into it, and exits without running. The former
generation modes (`default`, `interactive`, `remove`) are gone.

Two binding rules worth knowing:

- **An explicit null stays null.** `name:` (empty) or `name: null` binds
  `None`, the disabled spelling, exactly like the string `"None"`. The default
  is not substituted: that would silently reactivate the very thing the line
  was written to suppress.
- **A wrong-shaped value is refused, by dotted path.** A key given a nested
  block or a list where its parameter takes a scalar raises `ConfigError`
  naming the path (`Parameter 'Trainer.Dataset.batch_size' was given a nested
  block, but it takes a int.`) instead of binding something silently.

## `classpath`

Many configurable components are selected dynamically through a `classpath`
string. The exact resolution logic is implemented by
`konfai.utils.utils.get_module()`.

Typical examples:

```yaml
Model:
  classpath: segmentation.UNet.UNet
```

```yaml
Model:
  classpath: Model:UNetpp5
```

The two main styles are:

- `package.module.ClassName`-style references resolved relative to a KonfAI namespace
- `module:ClassName` references for explicit imports, often used for local files next to the YAML

Use the second form when you add custom files inside an example or project
directory. It is usually the least ambiguous option.

### How a name is resolved

Most component names in a config are resolved by `konfai.utils.utils.get_module` in
one of two ways. Three kinds do **not** go through it: loss-weight schedulers and
optimizers are looked up directly inside `konfai.metric.schedulers` and `torch.optim`
(so `module:Class` is not accepted for them), and a storage backend is never named at
all, you pick a format token in `dataset_filenames`.

| Form | Example | Resolves to |
| --- | --- | --- |
| **bare name** | `Dice`, `Standardize`, `Flip` | inside that kind's package (`konfai.metric.measure`, `konfai.data.transform`, `konfai.data.augmentation`, …) |
| **`module:Class`** | `torch:nn:L1Loss`, `monai.losses:DiceLoss`, `Loss:MyWrapper` | *any* importable module: an installed library **or** a local `.py` file next to your config (the current working directory is on `sys.path`) |

So the component pages list the **bare name** for built-ins; you are never limited
to them: any importable class that satisfies the same contract works via the
`module:Class` form. The binding rules above are the full resolution rules;
{doc}`../usage/custom-models` says how to write your own.

(config-discover)=
### How to discover a component's parameters

The tables give the **key** constructor arguments and defaults, but the exact,
always-current parameter set is whatever the class's `__init__` declares: the
reflection engine binds YAML keys directly to constructor parameter names. Two
ways to get the exhaustive list for any component:

1. **Let KonfAI materialise the defaults.** Reference the component in a config
   and run `konfai <COMMAND> --init` (or the workflow itself). KonfAI writes
   every resolved default back into the YAML file, giving you a complete,
   fully-expanded subtree to edit. (This is the same
   [config-mutation behaviour](#configuration) that surprises
   new users: here it is a feature.) `konfai list <kind>` prints every
   component's exact YAML spelling.
2. **Read the signature.** Where a bare name is looked up depends on the kind:

   | Kind | Bare name resolves in |
   | --- | --- |
   | criteria | `konfai/metric/measure/` |
   | transforms | `konfai/data/transform/` (one module per family), then `konfai/data/augmentation/` |
   | augmentations | `konfai/data/augmentation/` |
   | models | `konfai/models/python/**` |
   | learning-rate schedulers | `torch.optim.lr_scheduler` **first**, then `konfai/metric/schedulers.py` |
   | loss-weight schedulers | `konfai/metric/schedulers.py` only |
   | patch blending (`patch_combine`) | `konfai/data/patching/blend.py` |
   | reduction operators (a prediction's copies *and* a cohort's cases) | `konfai/data/reduction.py` |

   So a bare `StepLR` resolves *outside* KonfAI, in torch.

## `default|...` values

The `default|...` prefix is an important KonfAI convention. Its behavior is
inferred directly from `konfai.utils.config.Config._get_input_default()`.

It is used to express a fallback value that can still be overridden by the
config, and it is what `--init` materialises into a generated file. Examples
from the codebase include:

- `train_name: str = "default|TRAIN_01"`
- `classpath: str = "default|segmentation.UNet.UNet"`
- default dictionary keys such as `default|Labels`

In practice, you can read it as:

- **use the value after the pipe if nothing else is provided**

## Configuration is recursive

Because nested constructors are instantiated recursively, the shape of the YAML
mirrors the shape of the Python object graph. For example, a training config can
nest:

- `Trainer`
- `Model`
- a chosen model class
- `optimizer`
- `schedulers`
- `outputs_criterions`

This is why KonfAI examples are such a good source of truth: they show real
constructor trees that the framework accepts.

## Practical mapping rules

When a config does not behave as expected, check these rules first:

- the YAML root must match the workflow you are launching
- nested section names must match constructor parameters or any explicit
  `@config("...")` keys you chose
- local `classpath` modules must be importable from the current working directory
- the YAML shape should mirror the Python object graph, not just the names you
  want conceptually

## When to use local Python modules

Use a local module when you need:

- a custom model architecture
- a custom transform
- a project-specific helper that is not part of the built-in package

The `examples/Synthesis` workflow is the clearest repository example:

- `Model.py` defines local model classes
- `UnNormalize.py` defines a local transform
- the YAML references them with `Model:...` and `UnNormalize:...`

## The `Dataset` block

Every workflow reads its data through a `Dataset:` block, and four conventions
are shared by all of them: the on-disk layout, the `groups_src` mapping, the
`dataset_filenames` selectors and the `subset` / `validation` grammar. The
per-workflow keys (`batch_size`, `memory_budget`, `Patch`, augmentations) are
on each workflow's page; patch extraction and the memory regimes are explained
on {doc}`../usage/large-images`.

### Expected layout

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
extensions listed in `konfai.utils.utils.SUPPORTED_EXTENSIONS`. A spec may also
name a format that is a **backend rather than a suffix** (`:itktransform`, whose
entries are `<group>.h5`): those live in `SUPPORTED_BACKEND_FORMATS`, and
`SUPPORTED_FORMATS` is the union a `path[:flag]:format` spec is checked against.

Directory-backed formats use the same case/group model:

```text
DicomDataset/CASE_001/CT/*.dcm
OmeDataset/CASE_001/CT.ome.zarr/
```

### `groups_src` and `groups_dest`

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

### Dataset file selectors

The `dataset_filenames` field accepts strings in the form:

- `path`
- `path:format`
- `path:flag:format`

This behavior is implemented in `konfai.data.data_manager.DataSources._resolve_dataset_sources()`,
which delegates the parsing to `konfai.utils.utils.split_path_spec()`.

The most important conventions are:

- `a` means “append / union”
- `i` means “intersection / keep only common cases”

Examples:

- `./Dataset:a:mha`
- `./Predictions/TRAIN_01/Dataset:i:mha`
- `./DicomDataset:a:dicom`
- `./OmeDataset:a:omezarr`

### Training subsets and validation

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
- a slice string such as `0:10` (a negative end counts from the end,
  Python-style: `0:-2`)
- a path to a text file listing case names
- a `~path.txt` exclusion file
- a list of indices
- a list of case names
- a list mixing case names and case-list files

Three semantics are worth remembering:

- `subset: None` keeps the full dataset;
- `validation: None` disables the split;
- `subset` and `validation` accept the same selector spellings (slices, names,
  files, `~` exclusion): one grammar, implemented by `Subset`.

The `subset` object is applied before validation splitting and can exclude or
include items.

## Next steps

- {doc}`training`: every `Config.yml` key the training workflow reads.
- {doc}`../reference/components/models`: how `Model` sections address named
  module outputs for losses, metrics and exported predictions.
- {doc}`../usage/custom-models`: exposing a class of your own to YAML.
- {doc}`../usage/python-api`: the same workflows as Python callables, with the
  config tree as a dict.
