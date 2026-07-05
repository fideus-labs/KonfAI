# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level helpers and runtime utilities exposed by the KonfAI package."""

import datetime
import os
from importlib import metadata
from pathlib import Path

import psutil

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

# ``requests`` (remote-server helpers only) and ``torch`` (device-name lookup only) are imported lazily
# at their point of use so that ``import konfai`` stays light — in particular it no longer forces the
# ~1s torch import for CLI paths that never touch a GPU (``--help``/``--version``, light apps helpers).
from konfai.utils.errors import KonfAIError

try:
    __version__ = metadata.version("konfai")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


def checkpoints_directory() -> Path:
    """Return the configured checkpoint output directory."""
    return Path(_get_env("KONFAI_CHECKPOINTS_DIRECTORY"))


def predictions_directory() -> Path:
    """Return the configured prediction output directory."""
    return Path(_get_env("KONFAI_PREDICTIONS_DIRECTORY"))


def evaluations_directory() -> Path:
    """Return the configured evaluation output directory."""
    return Path(_get_env("KONFAI_EVALUATIONS_DIRECTORY"))


def statistics_directory() -> Path:
    """Return the configured statistics output directory."""
    return Path(_get_env("KONFAI_STATISTICS_DIRECTORY"))


def config_file() -> Path:
    """Return the active configuration file used by the current workflow."""
    return Path(_get_env("KONFAI_config_file"))


def konfai_state() -> str:
    """Return the current KonfAI workflow state stored in the environment."""
    return _get_env("KONFAI_STATE")


def konfai_root() -> str:
    """Return the root configuration section name for the current workflow."""
    return _get_env("KONFAI_ROOT")


class RemoteServer:
    """Connection settings for a remote KonfAI Apps server."""

    def __init__(self, host: str, port: int, token: str | None) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.timeout = 10

    def __str__(self) -> str:
        return f"{self.host}|{self.port}"

    def get_headers(self) -> dict[str, str]:
        """Return the HTTP headers required to talk to the remote server."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def get_url(self) -> str:
        """Return the base URL of the remote server."""
        return f"http://{self.host}:{self.port}"


def cuda_visible_devices() -> list[int]:
    """
    Return the GPU indices visible to the current process.

    Returns
    -------
    list[int]
        GPU ids exposed through ``CUDA_VISIBLE_DEVICES`` or detected by PyTorch.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu != ""]
    else:
        import torch

        devices = []
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
    return devices


def get_available_devices(
    remote_server: RemoteServer | None = None, timeout_s: float = 2.0
) -> tuple[list[int], list[str]]:
    """
    Return the available GPU indices and their display names.

    Parameters
    ----------
    remote_server : RemoteServer | None, optional
        Remote server to query instead of the local machine.
    timeout_s : float, optional
        HTTP timeout used for remote requests.

    Returns
    -------
    tuple[list[int], list[str]]
        Available device indices and the corresponding device names.
    """
    if remote_server is not None:
        import requests

        r = requests.get(
            f"{remote_server.get_url()}/available_devices", headers=remote_server.get_headers(), timeout=timeout_s
        )
        r.raise_for_status()
        data = r.json()
        return data["devices_index"], data["devices_name"]
    else:
        from torch.cuda import get_device_name

        devices_index = cuda_visible_devices()
        # Torch reindexes devices after CUDA_VISIBLE_DEVICES masking, so the
        # visible names must be resolved through local ordinals (0..N-1) while
        # we keep returning the original user-facing device ids.
        return devices_index, [get_device_name(local_index) for local_index in range(len(devices_index))]


def get_ram(remote_server: RemoteServer | None = None, timeout_s: float = 2.0) -> tuple[float, float]:
    """
    Return used and total RAM in gigabytes.

    Parameters
    ----------
    remote_server : RemoteServer | None, optional
        Remote server to query instead of the local machine.
    timeout_s : float, optional
        HTTP timeout used for remote requests.

    Returns
    -------
    tuple[float, float]
        Used RAM and total RAM in gigabytes.
    """
    if remote_server is not None:
        import requests

        r = requests.get(
            f"{remote_server.get_url()}/ram",
            headers=remote_server.get_headers(),
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        return data["used_gb"], data["total_gb"]
    else:
        ram = psutil.virtual_memory()
        used_gb = (ram.total - ram.available) / (1024**3)
        total_gb = ram.total / (1024**3)
        return used_gb, total_gb


def get_vram(
    devices: list[int], remote_server: RemoteServer | None = None, timeout_s: float = 2.0
) -> tuple[float, float]:
    """
    Return used and total VRAM in gigabytes for the selected devices.

    Parameters
    ----------
    devices : list[int]
        GPU indices to inspect.
    remote_server : RemoteServer | None, optional
        Remote server to query instead of the local machine.
    timeout_s : float, optional
        HTTP timeout used for remote requests.

    Returns
    -------
    tuple[float, float]
        Used VRAM and total VRAM in gigabytes.
    """
    if remote_server is not None:
        import requests

        r = requests.get(
            f"{remote_server.get_url()}/vram",
            params=[("devices", device_index) for device_index in devices],
            headers=remote_server.get_headers(),
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        return data["used_gb"], data["total_gb"]
    else:
        if not _PYNVML_AVAILABLE:
            raise KonfAIError(
                "GPU monitoring",
                "nvidia-ml-py is required for local VRAM queries. Install it with `pip install konfai[monitoring]`.",
            )
        used_gb = 0.0
        total_gb = 0.0
        pynvml.nvmlInit()
        for device_index in devices:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device_index))
            used_gb += info.used / (1024**3)
            total_gb += info.total / (1024**3)
        return used_gb, total_gb


def current_date() -> str:
    """Return the current timestamp formatted for KonfAI output folders."""
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def _get_env(var: str) -> str:
    value = os.environ.get(var)
    if value is None:
        raise RuntimeError(f"Environment variable '{var}' is not set.")
    return value


_KONFAI_DEPS: dict[str, str] = {
    "torch": "torch",
    "tqdm": "tqdm",
    "numpy": "numpy",
    "ruamel.yaml": "ruamel.yaml",
    "psutil": "psutil",
    "tensorboard": "tensorboard",
    "SimpleITK": "SimpleITK",
    "lxml": "lxml",  # often used as lxml.etree
    "h5py": "h5py",
    "nvidia-ml-py": "pynvml",  # IMPORTANT: pip != import
    "requests": "requests",
    "huggingface_hub": "huggingface_hub",
}


def _try_import(import_name: str) -> str | None:
    try:
        __import__(import_name)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def check_server(remote_server: RemoteServer, timeout_s: float = 2.0) -> tuple[bool, str]:
    """
    Check whether a remote KonfAI Apps server is reachable and healthy.

    Parameters
    ----------
    remote_server : RemoteServer
        Remote server connection settings.
    timeout_s : float, optional
        HTTP timeout used for the health check.

    Returns
    -------
    tuple[bool, str]
        A boolean success flag and a human-readable status message.
    """
    import requests

    try:
        r = requests.get(
            f"{remote_server.get_url()}/health",
            headers=remote_server.get_headers(),
            timeout=timeout_s,
        )

        if r.status_code == 401:
            return False, "Unauthorized (invalid or missing token)"
        if r.status_code == 403:
            return False, "Forbidden"
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        data = r.json()
        if data.get("status") != "ok":
            return False, f"Unexpected response: {data}"

        return True, "OK"

    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_konfai_install() -> tuple[bool, dict]:
    """
    Checks that KonfAI dependencies are importable.

    Returns
    -------
    tuple[bool, dict]
        A pair containing a global success flag and a report dictionary with the
        keys ``missing``, ``errors``, and ``versions``.
    """
    missing: list[str] = []
    errors: dict[str, str] = {}
    versions: dict[str, str] = {}

    deps = dict(_KONFAI_DEPS)
    for pip_name, import_name in deps.items():
        # best effort version lookup
        try:
            versions[pip_name] = metadata.version(pip_name)
        except metadata.PackageNotFoundError:
            versions[pip_name] = "not installed"
        except Exception:
            versions[pip_name] = "unknown"

        err = _try_import(import_name)
        if err is None:
            continue

        if versions[pip_name] == "not installed":
            missing.append(pip_name)
        else:
            errors[pip_name] = err

    return len(missing) == 0 and len(errors) == 0, {
        "missing": missing,
        "errors": errors,
        "versions": versions,
    }


class KonfAIPackagesError(RuntimeError):
    """Raised when required Python packages for KonfAI are missing/broken."""


def assert_konfai_install() -> None:
    """
    Raise :class:`KonfAIPackagesError` if the KonfAI dependency check fails.
    """
    is_konfai_install, report = check_konfai_install()
    if not is_konfai_install:
        lines = ["KonfAI dependency check failed."]

        if report["missing"]:
            lines.append("\nMissing packages:")
            lines.extend(f"  - {p}" for p in report["missing"])

        if report["errors"]:
            lines.append("\nImport/runtime errors:")
            for p, e in report["errors"].items():
                lines.append(f"  - {p}: {e}")

        raise KonfAIPackagesError("\n".join(lines))
