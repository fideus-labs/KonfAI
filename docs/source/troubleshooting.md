# Troubleshooting

Find what you see in the headings; each entry gives the likely cause and the fix.

## Installation problems

### `ModuleNotFoundError` after installation

KonfAI was probably installed into a different Python environment than the one
you are using to run the CLI.

Reinstall with the exact interpreter you plan to use:

```bash
python -m pip install "konfai[imaging]"
```

`-e ".[imaging]"` is the editable install from a checkout.

### `konfai-apps` or `konfai-apps-server` is missing

Both come from the standalone `konfai-apps` package:

```bash
python -m pip install konfai-apps
```

`konfai-cluster` comes with `konfai`; submitting jobs needs `pip install "konfai[cluster]"`.

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

An `outputs_criterions` or `outputs_dataset` key must be the exact path of a module of the model
(`UNetBlock_0:Head:Softmax`). Start from a working example, and use the same output names in training,
prediction and evaluation ({doc}`reference/components/models`).

### Validation split behaves unexpectedly

`validation` takes a share (`0.2`), or a list of case indices, slices (`0:10`), case names or files of case
names (`~` excludes). Check which form your config uses ({doc}`config_guide/index`). To keep a split fixed across runs,
give case names.

## Runtime problems

### The streaming regime still uses memory proportional to a case

A stage in the chain needs the whole volume: percentile `Clip`, `HistogramMatching`, `Canonical` on an
oblique volume, a statistic after a stage that changes values, or a custom transform that declares nothing.
{doc}`usage/large-images` lists which stages stream. Remove stages one at a time to find it, or run the
expensive part once with `Save` and stream from the saved copy.

### Patch inference is slow

Look at, one change at a time on the same cases:

- patch size and overlap (overlap repeats reads and forward passes);
- OME-Zarr chunk shape, DICOM decoding;
- `batch_size`, `num_workers`, `prefetch_factor`, `pin_memory`;
- the number of TTA copies and checkpoints;
- the number of output channels.

### CUDA OOM occurs during reconstruction or reduction

The peak is the output being assembled, not the model. Lower the batch size, the TTA or ensemble size, or
the output channels. When the output does not fit the GPU, KonfAI assembles it in host memory, more slowly.

### KonfAI asks before overwriting an existing run

Add `-y` to skip the interactive confirmation:

```bash
konfai TRAIN -y --config Config.yml
```

### Training fails in a restricted environment with socket or port errors

KonfAI opens a local port to coordinate its processes. A sandbox that forbids opening ports stops the run
before training starts. Run it on a normal machine.

### Live logs do not match TensorBoard exactly

That is expected: the console shows running values, TensorBoard the validation summaries. Compare finished
runs with the evaluation JSON files.

### Evaluation refuses: a group not found, or no case in common

EVALUATION stops before it writes a metric file when its groups do not meet:

```text
[DatasetManager] Group source 'PRED' not found in any dataset.
[DatasetManager] No data was found for groups ['PRED', 'SEG']: although each group contains data from a dataset, there are no common dataset names shared across all groups, the intersection is empty.
```

The first lists the groups each dataset holds. Check:

- that `Prediction.yml` wrote outputs into the expected `Predictions/<train_name>/`
  folder, and that `Evaluation.yml` points to the same `train_name`
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

### DICOM series is not detected

Use the `dicom` dataset token and the layout
`<root>/<case>/<group>/*.dcm`. The `dcm` token is a single-file SimpleITK path.
Extensionless Part-10 files are detected by their `DICM` marker; non-Part-10
extensionless exports may need conversion or explicit filenames.

### OME-Zarr metadata or geometry is rejected

Verify that each case/group store is valid OME-NGFF, that the selected pyramid
level exists, and that scale/translation metadata dimensionality matches the
array. Start with level 0, then consult
{doc}`reference/components/storage-backends` before selecting another level.

## Next steps

- {doc}`getting-started/installation`: the extras and verification steps
  behind most install-time symptoms.
- {doc}`reference/cli`: the `KONFAI_*` environment variables that
  drive runtime behavior.
- {doc}`usage/apps`: running packaged apps locally or against a remote
  `konfai-apps-server`.
