# Installation

This page covers every supported way to install KonfAI — PyPI, Pixi, and from
source — plus the optional extras and how to verify the result. Read it before
your first run, or come back when a format reader or CLI entrypoint is missing.
KonfAI targets **Python 3.10+** and depends on PyTorch, SimpleITK, TensorBoard,
and a set of medical-imaging utilities.

## Install from PyPI

```bash
python -m pip install konfai
```

This installs the core CLI entrypoints:

- `konfai`
- `konfai-cluster` with the `cluster` extra

Optional extras exposed by the package metadata:

```bash
python -m pip install "konfai[imaging]"   # all imaging backends: SimpleITK, h5py, pydicom, zarr, ngff-zarr
python -m pip install "konfai[dicom]"     # pydicom — DICOM series reader
python -m pip install "konfai[omezarr]"   # zarr + ngff-zarr — OME-Zarr dataset read/write
python -m pip install "konfai[all]"       # every optional extra at once
python -m pip install "konfai[dev]"       # test, docs, lint, and server tooling
```

### Optional extras

**A bare `pip install konfai` includes no imaging backend** — reading `.mha`,
HDF5, DICOM, or OME-Zarr data requires the matching extra below (`[imaging]`
covers all four).

| Extra | Pulls in | Use it for |
| --- | --- | --- |
| `itk` | `SimpleITK` | reading/writing ITK formats (`.mha`, `.nii.gz`, …) |
| `hdf5` | `h5py` | HDF5-backed datasets |
| `imaging` | `SimpleITK`, `h5py`, `pydicom`, `zarr`, `ngff-zarr` | **all four storage backends at once** (ITK + HDF5 + DICOM + OME-Zarr) — the common medical-imaging stack |
| `dicom` | `pydicom` | DICOM series input — see {doc}`../reference/components/storage-backends` |
| `omezarr` | `zarr`, `ngff-zarr` | OME-Zarr / OME-NGFF input — see {doc}`../reference/components/storage-backends` |
| `tensorboard` | `tensorboard` | TensorBoard logging |
| `monitoring` | `nvidia-ml-py` | GPU monitoring |
| `vtk` | `vtk` | VTK-dependent rendering and mesh features |
| `lpips` | `lpips` | the `LPIPS` perceptual metric |
| `ssim` | `scikit-image` | the `SSIM` metric |
| `fid` | `scipy`, `torchvision` | the `FID` metric |
| `export` | `onnx`, `onnxruntime`, `onnxscript` | ONNX export (experimental; see {doc}`../reference/python-api`) |
| `cluster` | `submitit` | `konfai-cluster` job submission |
| `all` | all of the above | install every optional extra at once |
| `dev` | pytest, ruff, sphinx, fastapi, … | local development, tests, docs, and the app server |

```{tip}
`konfai[imaging]` already covers DICOM and OME-Zarr — you do **not** need to add
`[dicom]` or `[omezarr]` on top of it. See {doc}`../reference/components/storage-backends`
for which format token maps to which backend.
```

Install the standalone apps package separately when you need packaged app
execution:

```bash
python -m pip install konfai-apps
```

This provides:

- `konfai-apps`
- `konfai-apps-server`
- the Python API under `konfai_apps`

## Install with Pixi

[Pixi](https://pixi.sh) is the recommended tool for reproducible environments
because it pins both Python packages and system libraries.

Install a released version:

```bash
pixi add konfai
```

Or, for a fully locked development environment from the repository:

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
pixi install        # resolves and installs all environments
pixi run test       # run the test suite
pixi run lint       # ruff lint the source tree
pixi run check      # lint + format-check + test (run before pushing)
```

See {doc}`../development` for the full developer workflow and the complete task
list.

## Install from source (pip)

Use an editable pip install when Pixi is not available or when you need to
install into an existing environment:

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
python -m pip install -e ".[imaging,dev]"
pytest -q tests/    # verify
```

## PyTorch and GPU notes

KonfAI declares `torch` as a dependency, but the correct GPU-enabled PyTorch
wheel still depends on your platform, drivers, and CUDA setup. In practice:

- if your default PyTorch install already matches your machine, `pip install konfai` is enough
- if you need a specific CUDA or CPU-only wheel, install PyTorch first, then install KonfAI
- for containerized usage, see {doc}`../usage/docker`

## Verify the installation

Check that the package imports correctly:

```bash
python -c "import konfai; print(konfai.__version__)"
```

Check that the main CLIs are available:

```bash
konfai --help
```

If you installed the standalone apps package or the `cluster` extra:

```bash
konfai-apps --help
konfai-apps-server --help
konfai-cluster --help
```

For a first real run after installation, go to {doc}`../quickstart`.

## Common installation issues

### `ModuleNotFoundError` after installation

This usually means the package was installed into a different Python
environment than the one you are currently using. Re-run the install with the
same interpreter you will use to launch KonfAI:

```bash
python -m pip install -e .
```

### GPU is available in Python but not in KonfAI

KonfAI relies on PyTorch device discovery and `CUDA_VISIBLE_DEVICES`. Check both:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
echo "$CUDA_VISIBLE_DEVICES"
```

### `konfai-apps-server` is missing

Install the standalone apps package:

```bash
python -m pip install konfai-apps
```

### `konfai-cluster` is missing

Install the `cluster` extra:

```bash
python -m pip install "konfai[cluster]"
```

## Next steps

- {doc}`../quickstart` — run the full train → predict → evaluate loop on the
  shipped segmentation example.
- {doc}`../usage/docker` — containerized installs when you cannot manage the
  host environment.
- {doc}`../reference/cli` — every `konfai` command and flag in one place.
