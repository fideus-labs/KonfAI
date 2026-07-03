# Python API (apps)

Besides the {doc}`CLI <cli>` and the {doc}`HTTP server <app-server-api>`, KonfAI
Apps expose a small **Python API** in the standalone `konfai_apps` package. Use it
to run an app from a script or notebook — locally or against a remote server —
with the same behaviour as the CLI.

```{note}
This is the `konfai_apps` package API (install it separately — see
{doc}`../getting-started/installation`). The low-level workflow functions
`konfai.trainer.train` / `konfai.predictor.predict` / `konfai.evaluator.evaluate`
are documented under {doc}`api/workflows`; here we cover the app layer on top.
```

## Public exports

`from konfai_apps import ...`: `KonfAIApp`, `KonfAIAppClient`, `AbstractKonfAIApp`,
`run_distributed_app`, `run_remote_job`, `main_apps`, `main_apps_server`. Plus
`from konfai import RemoteServer`.

## `KonfAIApp` — run an app locally

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
A remote identifier raises — use `KonfAIAppClient` for that. Each call runs inside
an isolated temporary workspace.

| Method | What it does |
| --- | --- |
| `infer(inputs, output, ensemble=0, ensemble_models=[], tta=0, mc=0, patch_size=None, batch_size=None, uncertainty=False, prediction_file="Prediction.yml", gpu=…, cpu=None, quiet=False, tmp_dir=None)` | Build a dataset from `inputs`, run KonfAI prediction, copy results into `output`. |
| `evaluate(inputs, gt, output, mask=None, evaluation_file="Evaluation.yml", …)` | Evaluate predictions against `gt` (auto ones-mask if `mask is None`). |
| `uncertainty(inputs, output, uncertainty_file="Uncertainty.yml", …)` | Uncertainty over a multi-channel inference stack. |
| `pipeline(inputs, gt, output, …, uncertainty=True, …)` | `infer` → (if `gt`) `evaluate` → (if `uncertainty`) `uncertainty`. `gt` is optional locally. |
| `fine_tune(dataset, name="Finetune", output, epochs=10, it_validation=1000, models=[], config_file="Config.yml", lr=None, …)` | Resume-train the app's checkpoint(s) on a local `dataset`, copy new weights into the output bundle. |

```{note}
`inputs` (and `gt`, `mask`) are a **list of groups**, where each group is a list of
file paths: `inputs=[[Path("a.nii.gz")]]` is one group of one file. Multi-modality
apps take one group per modality.
```

## `KonfAIAppClient` — run an app on a remote server

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
submits a job to the {doc}`HTTP server <app-server-api>`, streams the logs,
downloads and unpacks the result zip into `output`, and kills the remote job on
interrupt. `RemoteServer(host, port, token)` builds the base URL
(`http://host:port`) and the `Authorization: Bearer` header.

```{warning}
`RemoteServer` uses **plain HTTP** — the token and the medical volumes travel
unencrypted. Put the server behind a TLS-terminating reverse proxy for anything
beyond localhost. Also note **remote `patch_size`/`batch_size` are dropped** (the
HTTP endpoints don't accept them); those overrides only apply to local runs.
```

## Bundle & ONNX export

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

```{warning}
ONNX export is **experimental and Python-API-only** (there is no `konfai` CLI
subcommand for it). It exports a **single, static-shape** head of a feed-forward
model; custom-`forward` models (diffusion/StyleGAN/…) do not round-trip. See
`konfai/export.py` and {doc}`stability`.
```

## Trust model

```{danger}
Resolving/installing an app **copies its `.py` files into the run workspace and
imports them** — running a model by classpath (`Model:MyNet`) executes the app's
own Python. **Only resolve apps from sources you trust.** On the server side, the
`--apps` allowlist is the trust boundary; keep it tightly scoped. (The
`requirements.txt` auto-install mechanism exists but is currently gated off in the
shipped CLI paths.)
```

## See also

- {doc}`cli` — the `konfai-apps` / `konfai-apps-server` command line
- {doc}`app-server-api` — the HTTP endpoints these clients call
- {doc}`../ecosystem/index` — where the app bundles live
- {doc}`api/workflows` — the low-level `train`/`predict`/`evaluate` functions
