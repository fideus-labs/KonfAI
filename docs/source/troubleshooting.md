# Troubleshooting

Symptom-first fixes for the most common KonfAI failures. Scan the headings for
what you are seeing — a missing command, a rejected config key, an empty
metric file — and each entry gives the likely cause and the fix. Come here
whenever a command fails or an expected output never appears.

## Installation problems

### `ModuleNotFoundError` after installation

KonfAI was probably installed into a different Python environment than the one
you are using to run the CLI.

Reinstall with the exact interpreter you plan to use:

```bash
python -m pip install -e .
```

### `konfai-apps-server` or `konfai-cluster` is missing

`konfai-cluster` comes from the optional `cluster` extra on the core package.
`konfai-apps` and `konfai-apps-server` come from the standalone `konfai-apps`
package.

Install the relevant package or extra:

```bash
python -m pip install konfai-apps
python -m pip install "konfai[cluster]"
```

### GPU works in Python but not in KonfAI

KonfAI relies on PyTorch device discovery and `CUDA_VISIBLE_DEVICES`.

Check both:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
echo "$CUDA_VISIBLE_DEVICES"
```

If you are using Docker, also verify the container runtime and the `--gpus all`
flag.

## Configuration problems

### The dataset groups do not match the YAML

KonfAI expects the groups declared under `groups_src` to exist on disk. If the
config uses `CT` and `SEG`, each case directory must contain `CT.<ext>` and
`SEG.<ext>`.

Start from one of the shipped examples before renaming groups.

### `classpath` cannot import my module

This usually means one of these:

- the Python file is not in the current working directory or import path
- the class name in YAML does not match the Python symbol
- the YAML points to a local module, but the command is launched from the wrong directory

**KonfAI examples assume you run commands from the example directory itself.**
When in doubt, `cd` into the directory that contains the YAML before launching
`konfai`.

### A metric or output path is rejected

Keys used in `outputs_criterions` and similar sections must match real module
paths in the model graph. The runtime validates these names against the actual
submodules and raises an error if they do not exist.

When in doubt:

- start from a working example
- rename output paths gradually
- keep training, prediction, and evaluation aligned on the same output names

### Validation split behaves unexpectedly

`Dataset.validation` is flexible. In code it can be:

- `None`
- a float ratio
- a `start:stop` slice string
- a path to a text file
- an explicit list of indices
- an explicit list of case names
- a list mixing case names and text-file paths

If the split looks wrong, check which form your config is actually using.

## Runtime problems

### KonfAI asks before overwriting an existing run

Add `-y` to skip the interactive confirmation:

```bash
konfai TRAIN -y --config Config.yml
```

### Training fails in a restricted environment with socket or port errors

This can happen in sandboxes, some notebooks, or hardened servers.

This behavior is inferred from the runtime code: KonfAI's distributed launcher
allocates a free TCP port and initializes PyTorch distributed communication even
for local execution paths. If the environment forbids socket binding, startup
can fail before training begins.

In practice, test the workflow on a normal local machine or GPU server first.

### Live logs do not match TensorBoard exactly

That is expected. KonfAI logs some values live through the textual training
description, while validation summaries are written on their own schedule.

If you are debugging live behavior, inspect the log stream first. If you are
ranking completed runs, inspect the saved evaluation JSON files.

### Evaluation runs but the metric file is empty or missing

Check:

- that `Prediction.yml` wrote outputs into the expected `Predictions/<train_name>/`
  folder
- that `Evaluation.yml` points to the same `train_name`
- that masks, predictions, and references use compatible group names
- that the evaluation dataset uses the same case names as the prediction folder

## KonfAI Apps and remote server

### Remote app execution returns `401`

The server expects a bearer token when `KONFAI_API_TOKEN` is configured.

Make sure the client uses the same token:

```bash
konfai-apps infer my_app ... --host server --port 8000 --token my-token
```

### Remote app execution cannot connect

Check:

- server host and port
- firewall rules
- whether `konfai-apps-server` is actually running
- whether `/health` is reachable

## Next steps

- {doc}`getting-started/installation` — the extras and verification steps
  behind most install-time symptoms.
- {doc}`reference/environment` — the `KONFAI_*` environment variables that
  drive runtime behavior.
- {doc}`usage/apps` — running packaged apps locally or against a remote
  `konfai-apps-server`.
