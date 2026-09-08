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

## Building a workflow without running it

Below `konfai.api`, each workflow's builder returns the configured object
without launching the runtime: `build_train(...)`, `build_predict(...)`,
`build_evaluate(...)` and `build_transform(transform_file=..., transforms_dir=...)`.
That is the way to validate a config before the full run; `konfai.plan_transform`
above is the same idea one step further, and the `--plan` flag's entrypoint.

## Signatures

```{eval-rst}
.. currentmodule:: konfai.api

.. autofunction:: transform
   :no-index:

.. autofunction:: plan_transform
   :no-index:

.. autofunction:: evaluate
   :no-index:

.. autofunction:: predict
   :no-index:

.. autofunction:: train
   :no-index:

.. autofunction:: train_model
   :no-index:

.. autofunction:: predict_model
   :no-index:

.. autofunction:: import_bundle
   :no-index:

.. autofunction:: export_bundle
   :no-index:

.. autoclass:: TransformResult
   :members:
   :no-index:

.. autoclass:: EvaluationResult
   :members:
   :no-index:
```

The root classes behind them (`Trainer`, `Predictor`, `Evaluator`,
`Transformer`, `TransformPlan`) are in the {doc}`full module reference <../reference/api/index>`.

## Apps: the `konfai_apps` package

Besides the {doc}`CLI <../reference/cli>` and the {doc}`HTTP server <../reference/app-server-api>`,
KonfAI Apps expose a small **Python API** in the standalone `konfai_apps`
package (install it separately: see {doc}`../getting-started/installation`).
Use it to run an app from a script or notebook, locally or against a remote
server, with the same behaviour as the CLI. It is a layer on top of the
workflow API above.

### Public exports

`from konfai_apps import ...`: `KonfAIApp`, `KonfAIAppClient`, `AbstractKonfAIApp`,
`run_distributed_app`, `run_remote_job`, `main_apps`, `main_apps_server`. Plus
`from konfai import RemoteServer`.

### `KonfAIApp`: run an app locally

```python
from konfai_apps import KonfAIApp
from pathlib import Path

app = KonfAIApp("VBoussot/ImpactSynth:MR", download=False, force_update=False)
app.infer(
    inputs=[[Path("case_0000.nii.gz")]],   # list of input groups; each group is a list of paths
    output=Path("./Output"),
    ensemble=3, tta=4, gpu=[0],
)
```

`KonfAIApp(app, download, force_update)` resolves `app` to a **local directory** or
a **Hugging Face repo** (`repo_id:app_name`, optionally `repo_id@revision:app_name`).
A remote identifier raises: use `KonfAIAppClient` for that. Each call runs inside
an isolated temporary workspace.

The full method signatures of `KonfAIApp` and `KonfAIAppClient` (`infer`,
`evaluate`, `uncertainty`, `pipeline`, `fine_tune`) are single-sourced from the
docstrings on [App signatures](#app-signatures) below.

`inputs` (and `gt`, `mask`) are a **list of groups**, where each group is a list of
file paths: `inputs=[[Path("a.nii.gz")]]` is one group of one file. Multi-modality
apps take one group per modality.

### `KonfAIAppClient`: run an app on a remote server

```python
from konfai import RemoteServer
from konfai_apps import KonfAIAppClient

client = KonfAIAppClient(
    "VBoussot/ImpactSynth:MR",
    RemoteServer("127.0.0.1", 8000, token="changeme"),
)
client.pipeline(
    inputs=[[Path("case_0000.nii.gz")]],
    gt=[[Path("ref_0000.nii.gz")]],
    output=Path("./RemoteOutput"),
    tta=4, ensemble=3, gpu=[0],
)
```

`KonfAIAppClient(app, remote_server)` mirrors `KonfAIApp`'s methods, but each one
submits a job to the {doc}`HTTP server <../reference/app-server-api>`, streams the logs,
downloads and unpacks the result zip into `output`, and kills the remote job on
interrupt. `RemoteServer(host, port, token)` builds the base URL
(`http://host:port`) and the `Authorization: Bearer` header.

```{warning}
`RemoteServer` uses **plain HTTP**: the token and the medical volumes travel
unencrypted. Put the server behind a TLS-terminating reverse proxy for anything
beyond localhost. Remote `patch_size` / `batch_size` **are** carried: each job endpoint takes an
`options` form field, and the client refuses the submission if the server does not
echo the tunables back in `accepted_options`: a server too old to honour them fails
loudly instead of ignoring them.
```

### Bundle & ONNX export

`konfai_apps.bundle` assembles an app bundle offline and (experimentally) exports
ONNX for the `konfai-rs` portable-inference path:

```python
from konfai_apps.bundle import assemble_bundle, export_onnx_into_bundle

b = assemble_bundle(
    "MR", "dist", "app.json",
    ["Prediction.yml", "Evaluation.yml"], ["CV_0.pt", "CV_1.pt"],
    model_py="Model.py",
)
export_onnx_into_bundle(b, checkpoint="CV_0.pt")   # writes model.onnx + manifest.json
```

| Function | Purpose |
| --- | --- |
| `assemble_bundle(name, out_dir, app_json, configs, checkpoints, model_py=None, requirements=None)` | Validate `app.json` and stage configs / checkpoints / `Model.py` / `requirements.txt` into a bundle dir. |
| `export_onnx_into_bundle(bundle, *, patch_size=None, in_channels=None, prediction_config="Prediction.yml", checkpoint=None, output_module=None, root="Predictor")` | Load the model, export `model.onnx` + `manifest.json` via `konfai.export`. |
| `derive_requirements(py_files)` | Best-effort AST import scan → PyPI names (a draft to review, not authoritative). |

There is no `konfai` subcommand for ONNX export, but there **is** a
`konfai-apps` one: `konfai-apps bundle <name> --onnx …` exports `model.onnx` plus a
manifest into a bundle (and `--patch-size`, `--in-channels`, `--output-module` size
it).
It exports a **single, static-shape** head of a feed-forward model; custom-`forward`
models (diffusion/StyleGAN/…) do not round-trip. See `konfai/export.py`.

### Trust model

```{danger}
Resolving an app **copies its `.py` files into the run workspace and imports
them** unconditionally: running a model by classpath (`Model:MyNet`) executes
the app's own Python, i.e. arbitrary code. Resolving also **pip-installs the
app's `requirements.txt` by default**: only missing or version-mismatched
packages are installed, core packages (`torch`, `konfai`, …) are never touched,
and non-PEP 508 lines (`-r`, `--extra-index-url`, `git+…`) are skipped. Set
`KONFAI_APPS_INSTALL_REQUIREMENTS=0` to opt out (offline / CI / reproducible
environments). **Only resolve apps from sources you trust.** On the server
side, the `--apps` allowlist is the trust boundary; keep it tightly scoped.
```

### App signatures

```{eval-rst}
.. currentmodule:: konfai_apps

.. autoclass:: KonfAIApp
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: KonfAIAppClient
   :members:
   :show-inheritance:
   :no-index:

.. currentmodule:: konfai

.. autoclass:: RemoteServer
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: check_server
   :no-index:
.. autofunction:: get_available_devices
   :no-index:
.. autofunction:: get_ram
   :no-index:
.. autofunction:: get_vram
   :no-index:
```

## Next steps

- {doc}`making-data`: the TRANSFORM workflow as a guide, with its Python spelling.
- {doc}`adopting-konfai`: `train_model` and `predict_model` on a model you already have.
- {doc}`../reference/cli`: the same workflows from the command line.
- {doc}`../reference/app-server-api`: the HTTP endpoints `KonfAIAppClient` calls.
