# Docker

This directory contains the Docker assets used to package KonfAI from the PyPI release.

Included files:

- `Dockerfile`: build a KonfAI image with PyTorch preinstalled
- `entrypoint.sh`: forwards commands to `konfai` and related CLIs

## Docker Hub

The official image is available on Docker Hub as `vboussot/konfai`.

Pull the published image:

```bash
docker pull vboussot/konfai
```

If you build the image locally instead, replace `vboussot/konfai` with your local tag such as `konfai` in the examples below.

Run the default help command:

```bash
docker run --rm vboussot/konfai
```

## Local Build

Build the default image from this repository:

```bash
docker build -f docker/Dockerfile -t konfai .
```

Build a specific KonfAI version from PyPI:

```bash
docker build -f docker/Dockerfile \
  --build-arg KONFAI_PYPI_VERSION=1.6.0 \
  -t konfai .
```

Build with additional optional dependencies:

```bash
docker build -f docker/Dockerfile \
  --build-arg KONFAI_EXTRAS=imaging,cluster \
  -t konfai .
```

Build a CPU-only image:

```bash
docker build -f docker/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t konfai-cpu .
```

## What The Image Provides

The image exposes the KonfAI CLI entrypoints:

- `konfai`
- `konfai-apps`
- `konfai-apps-server`
- `konfai-cluster`

If no command is provided, the container runs `konfai --help`.
If the first argument is not one of the known executables, it is forwarded to `konfai`.

## GPU Runtime

The default image installs a CUDA-enabled PyTorch wheel. GPU access still depends on the host runtime.

Quick CUDA check:

```bash
docker run --rm -it --gpus all vboussot/konfai \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If CUDA is not visible inside the container, check:

- NVIDIA drivers on the host with `nvidia-smi`
- Docker GPU support with `docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu24.04 nvidia-smi`
- `nvidia-container-toolkit` installation on the host

## Run KonfAI

Train from the repository root:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai TRAIN --gpu 0 -c examples/Synthesis/Config.yml
```

Run prediction or evaluation:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai PREDICTION --models checkpoint.pt --gpu 0 -c examples/Synthesis/Prediction.yml
```

```bash
docker run --rm -it \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai EVALUATION -c examples/Synthesis/Evaluation.yml
```

## Run KonfAI Apps

Run an app command:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  vboussot/konfai konfai-apps infer my_app -i input.mha -o ./Output
```

Run the apps server:

```bash
docker run --rm -it -p 8000:8000 \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e KONFAI_API_TOKEN=my-token \
  vboussot/konfai konfai-apps-server --host 0.0.0.0 --port 8000 --apps konfai-apps/tests/assets/apps.json
```

## Notes

- The image is intended for CLI workflows executed from a mounted workspace.
- The default image is GPU-oriented; use a custom `TORCH_INDEX_URL` if you want a CPU-only variant.
- For reproducibility, pin the PyPI version with `KONFAI_PYPI_VERSION` when rebuilding locally.
