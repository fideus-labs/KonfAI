# Configuration model

This page explains how KonfAI turns a YAML file into live Python objects — the
reflection engine behind every `Trainer`, `Predictor`, and `Evaluator`. Read it
when a config key is not binding the way you expect, or before you expose a
custom class to YAML.

```{note}
Reading a config **mutates it**: loading a run resolves every default and
rewrites the YAML file in place, so the file on disk becomes the fully-resolved
record of the experiment. One consequence: a `None` value round-trips as the
literal string `"None"` — it is written back as `"None"` and reparsed to
`None` on the next read.
```

KonfAI is fundamentally a **configuration-driven object builder**.

The YAML file does not just pass values into a fixed script. It determines
which Python classes are instantiated and how they are connected.

## Root workflow objects

The root key of the YAML selects the high-level workflow object:

- `Trainer` for training
- `Predictor` for inference
- `Evaluator` for metrics

These names map directly to the public classes in:

- `konfai.trainer.Trainer`
- `konfai.predictor.Predictor`
- `konfai.evaluator.Evaluator`

## How YAML becomes Python objects

This behavior is implemented by `konfai.utils.config.Config`, `config()`, and
`apply_config()`.

In practice, the mapping is straightforward:

1. a class or function is optionally annotated with `@config("...")`
2. `apply_config()` inspects the constructor signature
3. YAML fields are matched against constructor parameter names
4. nested objects are recursively instantiated from nested mappings

This is why KonfAI configuration keys should generally:

- use **snake_case**
- match the actual Python constructor arguments
- stay close to the shipped examples when you introduce a custom class

One important detail in the current codebase: `@config()` defaults to the class
name. For local custom classes, this usually adds an unnecessary extra subtree.
Without any decorator, a custom class `UNetpp5` loaded through
`classpath: Model:UNetpp5` reads its parameters directly from `Trainer.Model`.

## Runtime environment variables

Two environment variables drive configuration loading at runtime. Both are read
directly from `os.environ` by `konfai.utils.config.Config`:

| Variable | Meaning |
| --- | --- |
| `KONFAI_config_file` | Path to the active YAML config file. `Config.__init__` reads it directly, so it must be set before any configurable object is built. |
| `KONFAI_CONFIG_MODE` | Controls what happens when the config file or individual keys are missing. See **Config modes** below. |

The KonfAI CLI sets both variables for you from the `--config` argument. You only
need to set them by hand when you call configurable classes directly — for
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

- `int`, `float`, `bool`, `str`, `torch.Tensor` — cast directly from the YAML scalar
- `Literal[...]` — validated against the allowed set (an invalid value raises `ConfigError`)
- `pathlib.Path` — wrapped as a `Path`; a non-existent path only logs a warning
- `list[...]` / `dict[str, ...]` — parsed element-wise
- a nested configurable class — instantiated recursively by re-entering
  `apply_config` on the nested subtree

Because the parameter *names* are the YAML keys, configuration keys should use
the exact constructor argument names (typically `snake_case`). A missing key
falls back to the parameter default, or to a `default|...` marker when one is
provided (see below).

## Config modes

`KONFAI_CONFIG_MODE` selects how KonfAI reacts to a missing file or missing keys:

| Mode | Behavior |
| --- | --- |
| `Done` | Normal run mode. The config file must already exist; values are read and the visited subtree is written back. A missing file raises `ConfigError`. |
| `default` | Materialize defaults non-interactively. Missing files or keys are created from each field's `default\|...` value (or its Python default), and the file is written. |
| `interactive` | Like `default`, but prompt on stdin for each `default\|...` field so a config can be generated interactively. Falls back to `default` when stdin is unavailable. |
| `Import` | Skip config binding entirely. The decorated object is called with the arguments it was given, without reading the YAML — used when importing or constructing classes outside the config-driven flow. |
| `remove` | Delete the config file on context exit instead of writing it back — used for throwaway configs, for example in tests. |

An unset or unknown value behaves like `Done`. Tests that build configurable
objects directly must therefore set **both** variables explicitly.

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

## `default|...` values

The `default|...` prefix is an important KonfAI convention. Its behavior is
inferred directly from `konfai.utils.config.Config._get_input_default()`.

It is used to express a fallback value that can still be overridden by config or
interactive generation. Examples from the codebase include:

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

## Next steps

- {doc}`datasets` — how the configured dataset sections map onto lazy, patch-based storage.
- {doc}`model-graph` — how `Model` sections address named module outputs for losses and metrics.
- {doc}`../config_guide/index` — the key-by-key reference for the three workflow config files.
