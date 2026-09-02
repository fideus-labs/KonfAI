# KonfAI in Python

The four CLI commands are callables: `konfai.transform` (with `konfai.plan_transform`, its
dry-run twin), `konfai.evaluate`, `konfai.predict` and `konfai.train`. One engine, two spellings: everything below builds the same config tree the YAML file would hold and hands it to the same
binder, so nothing here can drift from what a YAML run does.

```python
import konfai
from konfai.data.transform import Resample, Write

result = konfai.transform(
    "moved",
    "./Staged:mha",
    {"Moving": {"Moved": [
        Resample(reference="{case}", reference_group="DVF", field_group="DVF"),
        Write(dataset="./Output:mha"),
    ]}},
    memory_budget="8G",
)
result.outputs      # every chain's terminal Write: where the deliverables landed
result.config       # the resolved YAML the run kept -- commit this file to version the experiment
```

A chain is a list of **live stage objects** (the very classes the YAML names, with the very same
constructor arguments, which the extension bases record as given) or the equivalent mapping
(`{"Resample": {...}, "Write": {...}}`), or a whole tree loaded from an existing YAML and modified
in place. Two stages of the same class in one chain spell the second one module-qualified
(`konfai.data.transform:Resample`), exactly as the YAML file must.

## The contract, and how it differs from the CLI

- **A designed refusal raises** `KonfAIError`: the message and the remedy are the exception; the
  caller decides. Only the CLI catches and exits.
- **Results come back structured**: `transform` returns the `outputs.json` destinations and the
  workspace; `evaluate` returns the parsed `Metric_*.json` as a dict.
- **The process is left as found**: the `KONFAI_*` environment is restored around every call, and
  one workflow runs at a time per process: a second concurrent call is refused with the remedy
  (subprocesses), never allowed to corrupt the first.
- **The record remains.** Every call materializes the resolved YAML in the run's workspace:
  promoting a notebook run to a versioned experiment is copying `result.config`: nothing to
  rewrite, and the run stays resumable like any other.

`konfai.plan_transform(...)` takes the same arguments and returns the `TransformPlan` without
running anything: plan first is the same reflex in Python as on the CLI.

```{note}
**Migration note for Python callers of `Network`.** `Network.state_dict()` now
honors the torch signature and returns the torch-native flat dict (still
skipping nested `Network`s); the KonfAI aggregate that checkpoints are built
from is `network_states()`. Checkpoint **files on disk are unchanged**: nothing
saved by an earlier version needs converting, and RESUME/PREDICTION read them
as before. Only code that builds or unpacks checkpoint dicts in Python must
switch from `state_dict()` to `network_states()`. The KonfAI traversals moved
with it: `graph_parameters(pretrained=...)` replaces the old `parameters(pretrained)`
override, and `graph_apply()` the custom `apply()`; torch's native
`parameters()` / `named_parameters()` / `apply()` are back to their own
semantics.
```

## Which spelling fits which workflow

| Workflow | Its config is… | The Python spelling |
| --- | --- | --- |
| TRANSFORM | a chain of stage objects | `konfai.transform(name, datasets, chains, ...)` with live stages |
| EVALUATION | criteria per group | `konfai.evaluate(name, datasets, metrics={"PRED": {"GT": [MAE(), Dice()]}}, ...)` |
| PREDICTION | wiring (checkpoints, patches, TTA) | `konfai.predict(models=[...], config=tree_or_path, ...)`: the tree or the file |
| TRAIN / RESUME | the full graph (model, losses, optimizers) | the tree: load the YAML into a dict, change the keys under study, call `konfai.train(config=tree)` |

Every workflow entry point accepts the config **tree as a dict** wherever it accepts a file path, that alone is the sweep idiom for TRAIN: the resolved config each run keeps *is* the record of what
was tried. The object spelling exists where a config is a list of objects (TRANSFORM chains,
EVALUATION criteria); rebuilding a training graph in nested kwargs would add nothing over the YAML
that publishes it.
