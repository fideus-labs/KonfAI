# Installation

KonfAI needs **Python 3.11 or newer**. This is the line most people want:

```bash
python -m pip install "konfai[imaging]"
```

`konfai` on its own brings PyTorch, NumPy, `ruamel.yaml`, `psutil` and
the engine, but **no image reader**, so it cannot open a `.mha`. The `[imaging]`
extra adds SimpleITK, h5py, pydicom, zarr and ngff-zarr, which covers all four
storage backends at once. You do not need `[dicom]` or `[omezarr]` on top of it.

Check the result:

```bash
python -c "import konfai; print(konfai.__version__)"
konfai --help
```

The examples do not travel in the wheel. To install a checkout as a wheel and
run its examples:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
python -m pip install ".[imaging]"
```

## The extras

Take one when you need a specific reader, metric or tool. `[all]` takes
every extra but `monai`, `[dev]` adds the test, lint and docs tooling.

| Extra | Pulls in | Use it for |
| --- | --- | --- |
| `imaging` | `SimpleITK`, `h5py`, `pydicom`, `zarr`, `ngff-zarr` | all four storage backends at once: ITK, HDF5, DICOM, OME-Zarr |
| `itk` | `SimpleITK` | ITK formats only (`.mha`, `.nii.gz`, …) |
| `hdf5` | `h5py` | HDF5-backed datasets |
| `dicom` | `pydicom` | DICOM series, see {doc}`../reference/components/storage-backends` |
| `omezarr` | `zarr`, `ngff-zarr` | OME-Zarr / OME-NGFF, see {doc}`../reference/components/storage-backends` |
| `s3` | `s3fs`, `aiohttp` | reading a dataset root from `s3://`, see {doc}`../reference/components/storage-backends` |
| `tensorboard` | `tensorboard` | TensorBoard logging |
| `monitoring` | `nvidia-ml-py` | GPU monitoring |
| `smp` | `segmentation-models-pytorch` | the SMP model bridge, **required by `examples/Synthesis`** |
| `monai` | `monai` | MONAI networks and losses named by classpath (`monai.networks.nets:UNet`) and MONAI Bundles, see {doc}`../usage/adopting-konfai` |
| `lpips` | `lpips` | the `LPIPS` metric |
| `ssim` | `scikit-image` | the `SSIM` metric |
| `vtk` | `vtk` | VTK rendering and mesh features |
| `export` | `onnx`, `onnxruntime`, `onnxscript` | ONNX export, see {doc}`../usage/python-api` |
| `cluster` | `submitit` | the `konfai-cluster` submitter |
| `all` | everything above but `monai`, plus `huggingface_hub` | one shot; `huggingface_hub` serves the `IMPACT*` criteria's feature-extractor downloads |
| `dev` | pytest, ruff, mypy, sphinx, … | working on KonfAI itself |

## Running packaged apps

Apps live in their own package:

```bash
python -m pip install konfai-apps
```

It gives you the `konfai-apps` and `konfai-apps-server` commands, plus the
Python API under `konfai_apps`. Check them with `konfai-apps --help` and
`konfai-apps-server --help`. `konfai-cluster` ships with the core `konfai`
package itself; the `cluster` extra only adds `submitit`, which actual SLURM
submission needs.

When developing one of the bundled apps (`apps/impact_seg`, `apps/impact_synth`,
`apps/impact_reg`, `apps/mrsegmentator`, `apps/totalsegmentator`), install the
three packages from the same checkout to exercise your local changes:

```bash
python -m pip install -e . -e konfai-apps -e apps/impact_seg
```

At a release tag each bundle pins `konfai==` and `konfai-apps==` to that
release; from a working tree the pin accepts the closest release or newer.

## Pixi

[Pixi](https://pixi.sh) pins system libraries as well as Python packages, so it
is the reproducible option and the one to use when working on KonfAI itself.

```bash
pixi add konfai        # a released version, in your own project
```

From a checkout:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
pixi install           # every environment, locked
pixi run test-fast     # the iteration loop, about 25 s
pixi run lint          # ruff over the source tree
pixi run check         # lint + format + tests, before pushing
```

{doc}`../development` has the full task list.

## From source, without Pixi

An editable install works when Pixi is not available or when you already have an
environment:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
python -m pip install -e ".[imaging,dev]"
pytest tests/
```

## GPU

KonfAI declares `torch` as a dependency but cannot pick the right wheel for your
drivers and CUDA version. If your PyTorch already matches your machine, there is
nothing to do. If you need a specific CUDA or a CPU-only build, install PyTorch
first, then KonfAI. For containers, see [Docker](#docker) below.

## Platforms

KonfAI runs on Linux, macOS and Windows. Two things differ outside Linux:

- **GPUs are CUDA devices.** KonfAI does not use Apple's Metal (MPS) backend, so
  on macOS every workflow runs on the CPU: leave `--gpu` out and choose the
  number of processes with `--cpu N`.
- **Windows opens no process group.** TRAIN, RESUME and EVALUATION run there on
  one process (`--cpu 1` or a single `--gpu`) and refuse more. PREDICTION and
  TRANSFORM, whose processes never talk to each other, can use several. See
  [How a run is launched](../reference/cli.md#how-a-run-is-launched).

## If something is missing

- **`ModuleNotFoundError` after installing**: the install landed in a different
  environment than the one you are running. Reinstall with the same interpreter,
  `python -m pip install "konfai[imaging]"` (`-e ".[imaging]"` from a
  checkout).
- **PyTorch sees your GPU but KonfAI does not**: KonfAI goes through PyTorch
  device discovery and `CUDA_VISIBLE_DEVICES`. Check both with
  `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
  and `echo "$CUDA_VISIBLE_DEVICES"`.
- **`konfai-apps-server` not found**: `pip install konfai-apps`.
- **`konfai-cluster` not found**: the command ships with `konfai` itself, so
  "not found" means the environment mismatch of the first bullet. Install
  `konfai[cluster]` only when submission fails on a missing `submitit`.


```{include} ../../../docker/README.md
:heading-offset: 1
```

The `docker run` examples above mount the current directory as `/workspace`:
run them from the directory that contains your configs, data, and checkpoints.
{doc}`../reference/cli` lists the flags used in the container commands.

Next: {doc}`../quickstart` trains, predicts and evaluates four synthetic CT
volumes on CPU, then verifies the outputs. {doc}`../reference/cli` lists every
command and flag.
