# KonfAI in Python

The four commands are Python functions: `konfai.transform` (and `konfai.plan_transform`, which only plans),
`konfai.evaluate`, `konfai.predict` and `konfai.train`. They build the same config a YAML file would hold and
run the same workflow.

```python
import konfai
from konfai.data.transform import Resample, Write

if __name__ == "__main__":
    result = konfai.transform(
        "moved",
        "./Staged:mha",
        {"Moving": {"Moved": [
            Resample(reference="{case}", reference_group="DVF", field_group="DVF"),
            Write(dataset="./Output:mha"),
        ]}},
        memory_budget="8G",
    )
    result.outputs      # where each chain's Write landed
    result.config       # the resolved YAML of the run: keep it to reproduce the experiment
```

A chain is a list of stage objects (the classes the YAML names, with the same arguments), or the same thing
as a dictionary (`{"Resample": {...}, "Write": {...}}`), or a tree loaded from a YAML file and modified.

| Workflow | Python |
| --- | --- |
| TRANSFORM | `konfai.transform(name, datasets, chains, ...)` with stage objects |
| EVALUATION | `konfai.evaluate(name, datasets, metrics={"PRED": {"GT": [MAE(), Dice()]}}, ...)` |
| PREDICTION | `konfai.predict(models=[...], config=tree_or_path, ...)` |
| TRAIN / RESUME | load the YAML into a dictionary, change what you study, `konfai.train(config=tree)` |

Every function takes a config as a file path or as a dictionary, which makes sweeps simple: each run keeps
its resolved config as the record of what was tried.

## How a call behaves

- **Errors raise.** A refusal is a `KonfAIError` with its message and remedy; your code decides what to do.
- **Results are objects.** `transform` returns the outputs and the workspace; `evaluate` returns the metrics
  as a dictionary.
- **Your config file is not modified.** It is copied to a scratch folder, which receives the resolved
  defaults.
- **One workflow at a time per process.** The `KONFAI_*` environment and the memory budget are restored
  after each call.
- **Run it under `if __name__ == "__main__":`.** Worker processes import your script again; without the
  guard, each would start the workflow.
- **A local classpath works as on the CLI**: `Model:UNet` finds `./Model.py` in the working directory.
- **`gpu=[...]`** takes the same ids as `--gpu`.
- The first call sets torch's, ITK's and zarr's thread pools for the life of the process.

`konfai.plan_transform(...)` takes the same arguments as `transform` and returns the plan without running
anything. To build a workflow without running it, use `konfai.trainer.build_train`,
`konfai.predictor.build_predict`, `konfai.evaluator.build_evaluate` or `konfai.transformer.build_transform`.

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

The workflow classes (`Trainer`, `Predictor`, `Evaluator`, `Transformer`, `TransformPlan`) are in the
{doc}`module reference <../reference/api/index>`.

## Apps: the `konfai_apps` package

`konfai_apps` (installed separately, {doc}`../getting-started/installation`) runs apps from Python, locally or
on a server, like the {doc}`konfai-apps CLI <../reference/cli>`.

```python
from pathlib import Path
from konfai_apps import KonfAIApp

app = KonfAIApp("VBoussot/ImpactSynth:MR", download=False, force_update=False)
app.infer(
    inputs=[[Path("case_0000.nii.gz")]],   # a list of input groups, each a list of files
    output=Path("./Output"),
    ensemble=3, tta=4, gpu=[0],
)
```

`KonfAIApp` takes a local folder or a Hugging Face reference (`repo:app`, or `repo@revision:app`). Its methods
are `infer`, `evaluate`, `uncertainty`, `pipeline` and `fine_tune` ([signatures](#app-signatures)). `inputs`,
`gt` and `mask` are lists of groups: one group per modality.

To run on a {doc}`server <../reference/app-server-api>`, use `KonfAIAppClient` with the same methods:

```python
from konfai import RemoteServer
from konfai_apps import KonfAIAppClient

client = KonfAIAppClient("VBoussot/ImpactSynth:MR", RemoteServer("127.0.0.1", 8000, token="changeme"))
client.pipeline(
    inputs=[[Path("case_0000.nii.gz")]],
    gt=[[Path("ref_0000.nii.gz")]],
    output=Path("./RemoteOutput"),
    tta=4, ensemble=3, gpu=[0],
)
```

It uploads the inputs, streams the logs, downloads the result into `output`, and stops the remote job if
you interrupt it.

```{warning}
`RemoteServer` speaks plain HTTP: the token and the images are not encrypted. Put the server behind a TLS
reverse proxy beyond localhost.
```

```{danger}
Resolving an app **runs its Python code** and, by default, **installs its `requirements.txt`** (never touching
`torch` or `konfai`; `KONFAI_APPS_INSTALL_REQUIREMENTS=0` turns this off). Only use apps from sources you
trust. On a server, the `--apps` list decides which apps can run.
```

### Bundles and ONNX export

`konfai_apps.bundle` builds an app folder from Python, and can export an ONNX model (experimental):

```python
from konfai_apps.bundle import assemble_bundle, export_onnx_into_bundle

bundle = assemble_bundle(
    "MR", "dist", "app.json",
    ["Prediction.yml", "Evaluation.yml"], ["CV_0.pt", "CV_1.pt"],
    model_py="Model.py",
)
export_onnx_into_bundle(bundle, checkpoint="CV_0.pt")   # writes model.onnx and manifest.json
```

`konfai-apps bundle <name> --onnx` does the same from the command line. The export takes one output of fixed
shape from a feed-forward model; models with a custom `forward` (diffusion, StyleGAN) do not export. It uses
the `Patch.overlap` of the inference config: declare it, or the exported model tiles without overlap.

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

- {doc}`making-data`: preparing a dataset, in YAML and in Python.
- {doc}`adopting-konfai`: `train_model` and `predict_model` on a model you already have.
