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

"""Local and remote execution helpers for packaged KonfAI Apps."""

import inspect
import os
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, cast

import numpy as np
import requests
import SimpleITK as sitk
from konfai import RemoteServer, check_server, cuda_visible_devices, get_vram
from konfai.utils.dataset import Dataset
from konfai.utils.errors import AppRepositoryError, KonfAIAppClientError
from konfai.utils.runtime import MinimalLog, State, safe_torch_load
from konfai.utils.utils import SUPPORTED_EXTENSIONS, split_format_level, split_path_spec
from ruamel.yaml import YAML

from .app_repository import LocalAppRepository, get_app_repository_info


class CancelProcess(RuntimeError):
    """
    Exception used to convert SIGINT/SIGTERM signals into a regular Python error.

    This is primarily used to ensure that `finally` blocks are executed even when
    the process receives an interrupt/termination signal (Ctrl+C, system kill).

    Notes
    -----
    - This exception is intentionally raised from a signal handler.
    - It should typically be caught at a high level to perform cleanup.
    """

    pass


@contextmanager
def ensure_finally_on_signals():
    """
    Context manager that guarantees `finally` blocks run on SIGINT/SIGTERM.

    Inside this context:

    - SIGINT and SIGTERM handlers are temporarily replaced.
    - Receiving one of these signals raises `CancelProcess`, which unwinds the
      stack normally and therefore triggers `finally` clauses.

    On exit:

    - original signal handlers are restored.

    Typical usage
    -------------

    .. code-block:: python

       with ensure_finally_on_signals():
           try:
               ...
           finally:
               cleanup()

    Caveats
    -------
    - Signal handlers are process-global: using this concurrently in multiple
      threads is not recommended.
    - Only SIGINT/SIGTERM are handled here.
    """
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):
        raise CancelProcess(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def run_distributed_app(
    func: Callable[..., None],
) -> Callable[..., None]:
    """
    Decorator that runs a KonfAI app entrypoint inside an isolated temporary workspace.

    This wrapper:
    - Creates (or reuses) a temporary working directory (`tmp_dir`)
    - Changes the current working directory to that temporary directory
    - Adds that directory to `sys.path` (so local imports work)
    - Executes the wrapped function inside a minimal logging context (`MinimalLog`)
    - Restores the user's original working directory
    - Deletes the temporary directory if it was created automatically

    The decorated function may declare a `tmp_dir` argument. If provided, that
    directory is used and NOT automatically deleted (unless it lives under the
    system temp directory and the code chooses to clean it).

    Parameters
    ----------
    func : Callable[..., None]
        Function implementing a local app action (infer/evaluate/etc.).

    Returns
    -------
    Callable[..., None]
        Wrapped function with identical signature and behavior, executed in
        an isolated workspace.
    """

    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        params = sig.parameters
        kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

        bound = sig.bind_partial(*args, **kwargs_fun)
        bound.apply_defaults()

        tmp_dir = bound.arguments.get("tmp_dir")
        auto_created = tmp_dir is None
        if auto_created:
            workspace_dir = Path(tempfile.mkdtemp(prefix="konfai_app_"))
        else:
            workspace_dir = Path(cast(str | os.PathLike[str], tmp_dir))
        workspace_dir = workspace_dir.resolve()
        user_dir = os.getcwd()
        # Resolve every caller-supplied path against the caller's directory before chdir'ing
        # into the workspace: a relative path would otherwise be interpreted inside the
        # (possibly auto-created and then deleted) temporary workspace.
        if bound.arguments.get("output") is not None:
            bound.arguments["output"] = Path(bound.arguments["output"]).resolve()
        for key in ("inputs", "gt", "mask"):
            value = bound.arguments.get(key)
            if value is not None:
                bound.arguments[key] = [[Path(p).resolve() for p in group] for group in value]
        if bound.arguments.get("dataset") is not None:
            bound.arguments["dataset"] = Path(bound.arguments["dataset"]).resolve()
        added_to_syspath = False
        try:
            os.makedirs(workspace_dir, exist_ok=True)
            os.chdir(str(workspace_dir))
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
                added_to_syspath = True
            with MinimalLog():
                func(*bound.args, **bound.kwargs)
        except KeyboardInterrupt:
            print("\n[KonfAI-Apps] Manual interruption (Ctrl+C)")
            raise SystemExit(130) from None
        finally:
            if added_to_syspath and str(workspace_dir) in sys.path:
                sys.path.remove(str(workspace_dir))
            if Path(os.getcwd()).resolve() != Path(user_dir).resolve():
                os.chdir(user_dir)
            if auto_created:
                shutil.rmtree(str(workspace_dir), ignore_errors=True)

    return wrapper


class AbstractKonfAIApp:
    """Common base class for local and remote KonfAI App runners."""


class KonfAIAppClient(AbstractKonfAIApp):
    """
    Client-side helper to submit jobs to a remote KonfAI app server.

    This class implements:
    - job submission to endpoints like `/apps/{app}/{action}`
    - streaming logs via SSE (`/jobs/{job_id}/logs`)
    - result retrieval (`/jobs/{job_id}/result`)
    - remote job termination (`/jobs/{job_id}/kill`)

    It is intended to mirror the server's execution model:
    submit → stream logs → download results → (optional) kill on interruption.
    """

    def __init__(self, app: str, remote_server: RemoteServer) -> None:
        """
        Create a client bound to a given application and remote server.

        Parameters
        ----------
        app : str
            Application identifier/path on the server.
        remote_server : RemoteServer
            Server connection parameters (base URL, auth headers).

        Raises
        ------
        KonfAIAppClientError
            If the server cannot be reached or does not respond as expected.
        """
        self.app = app
        self.remote_server = remote_server
        ok, msg = check_server(remote_server)
        if not ok:
            raise KonfAIAppClientError(
                f"{msg}."
                "Unable to connect to the KonfAI app server.\n\n"
                "Please verify the host and port, or select another remote server."
            )

    def stream_logs(self, job_id: str, connect_timeout: int = 60, read_timeout: int = 600):
        """
        Stream server-side job logs using Server-Sent Events (SSE).

        This method connects to:
            GET /jobs/{job_id}/logs

        It prints each received SSE "data:" message to stdout. The stream ends when
        one of the terminal markers is received:
        - "__DONE__"
        - "__ERROR__ ..."

        Parameters
        ----------
        job_id : str
            Remote job identifier returned by the server.
        connect_timeout : int
            Max seconds to wait for the initial connection.
        read_timeout : int
            Max seconds to wait for new bytes before considering the stream stalled.

        Raises
        ------
        RuntimeError
            For auth errors, forbidden access, stream stalls, or other request failures.
        """
        url = f"{self.remote_server.get_url()}/jobs/{job_id}/logs"
        try:
            with requests.get(
                url,
                headers=self.remote_server.get_headers(),
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as r:
                if r.status_code == 401:
                    raise RuntimeError("Unauthorized: invalid or missing token")
                if r.status_code == 403:
                    raise RuntimeError("Forbidden")

                r.raise_for_status()

                for line in r.iter_lines(decode_unicode=True):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        msg = line[6:]
                        if msg == "__DONE__":
                            return
                        if msg.startswith("__ERROR__"):
                            detail = msg[len("__ERROR__") :].strip()
                            raise RuntimeError(f"Remote job failed: {detail}" if detail else "Remote job failed")
                        print(msg, flush=True)

        except requests.exceptions.ReadTimeout as e:
            raise RuntimeError(f"Log stream stalled (no data received for {read_timeout}s)") from e
        except requests.exceptions.ConnectTimeout as e:
            raise RuntimeError("Connection timeout") from e
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to stream logs from {url}: {e}") from e

    def kill_job(self, job_id: str, timeout_s: float = 60) -> None:
        """
        Request termination of a remote job.

        Sends:
            POST /jobs/{job_id}/kill

        If successful, prints a confirmation message.

        Parameters
        ----------
        job_id : str
            Remote job identifier.
        timeout_s : float
            Timeout (seconds) for the kill request.

        Raises
        ------
        TimeoutError
            If the request times out.
        RuntimeError
            For auth errors or other HTTP failures.
        """
        url = f"{self.remote_server.get_url()}/jobs/{job_id}/kill"
        try:
            r = requests.post(
                url,
                headers=self.remote_server.get_headers(),
                timeout=timeout_s,
            )

            if r.status_code == 401:
                raise RuntimeError("Unauthorized: invalid or missing token")

            r.raise_for_status()
            print(f"[KonfAI-Apps] Remote job {job_id} successfully killed.")
        except requests.exceptions.ConnectTimeout as e:
            raise TimeoutError("Connection timeout while sending kill request") from e

        except requests.exceptions.ReadTimeout as e:
            raise TimeoutError(f"Kill request stalled (no response for {timeout_s:.0f}s)") from e

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to kill job {job_id}: {e}") from e

    def download_result(
        self, job_id: str, out_dir: Path, connect_timeout: int = 60, read_timeout: int = 600, max_wait_s: int = 600
    ):
        """
        Download and unpack the result archive for a remote job.

        Polls:
            GET /jobs/{job_id}/result

        Server behavior expected:
        - HTTP 202: result not ready → keep polling
        - HTTP 200: returns a zip archive → download then unpack

        The downloaded archive is saved as:
            <out_dir>/result.zip

        Then extracted into `out_dir`.

        Parameters
        ----------
        job_id : str
            Remote job identifier.
        out_dir : Path
            Destination directory where the result is extracted.
        connect_timeout : int
            Connection timeout for each poll attempt.
        read_timeout : int
            Read timeout for each download attempt.
        max_wait_s : int
            Maximum total time to wait for the result to become available.

        Returns
        -------
        bool
            True if the result was successfully downloaded and extracted.

        Raises
        ------
        TimeoutError
            If the result does not become ready within `max_wait_s`.
        RuntimeError
            For request failures, auth issues, or download errors.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "result.zip"

        poll_interval = 0.5
        deadline = time.monotonic() + max_wait_s

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Result not ready after {max_wait_s:.0f}s")

            try:
                with requests.get(
                    f"{self.remote_server.get_url()}/jobs/{job_id}/result",
                    headers=self.remote_server.get_headers(),
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                ) as r:
                    if r.status_code == 401:
                        raise RuntimeError("Unauthorized: invalid or missing token")

                    if r.status_code == 202:
                        time.sleep(poll_interval)
                        continue

                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
            except requests.exceptions.ConnectTimeout as e:
                raise TimeoutError("Connection timeout while downloading result") from e
            except requests.exceptions.ReadTimeout as e:
                raise TimeoutError(f"Download stalled (no data for {read_timeout:.0f}s)") from e
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download result for job {job_id}: {e}") from e

            break
        shutil.unpack_archive(zip_path, out_dir)
        print(f"[KonfAI-Apps] Result written to: {out_dir}")
        return True

    @staticmethod
    def run_remote_job(func: Callable[..., None]) -> Callable[..., None]:
        """
        Decorator for KonfAIAppClient methods that submit work to the remote server.

        The wrapped method is treated as an "action" endpoint. For example, wrapping
        ``infer()`` will call ``POST /apps/{self.app}/infer``.

        Behavior:

        1. Introspects the wrapped function signature to filter kwargs.
        2. Builds a multipart request containing file fields for inputs, ground
           truth, and masks, plus scalar fields for other parameters.
        3. Submits the job and retrieves a ``job_id``.
        4. Streams logs until completion markers are received.
        5. Downloads and extracts results into the requested output directory.
        6. On SIGINT or SIGTERM, triggers cleanup and kills the remote job if it
           is still running.
        7. Always closes local file handles.

        Signal handling
        ---------------
        Uses ``ensure_finally_on_signals()`` so that SIGINT or SIGTERM raises
        ``CancelProcess``. This ensures the ``finally`` block runs and the remote
        kill request is attempted when needed.

        Notes
        -----
        - The decorated methods are "declarative": they do not implement logic
          themselves and typically contain only ``pass``.
        - Output directory is taken from the wrapped method's ``output``
          argument.

        Returns
        -------
        Callable[..., None]
            Wrapped method that performs remote submission + monitoring + download.
        """
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> None:
            job_id: str | None = None

            params = sig.parameters
            kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

            bound = sig.bind_partial(self, *args, **kwargs_fun)
            bound.apply_defaults()
            bound.arguments.pop("self", None)

            output = bound.arguments.pop("output", None)
            files = []
            data = {}
            dataset_zip_dir: str | None = None
            unit_zip_dirs: list[str] = []
            finished = False
            with ensure_finally_on_signals():
                try:
                    data_arguments = ["inputs", "gt", "mask"]
                    for k, v in bound.arguments.items():
                        if k in data_arguments:
                            if v is not None:
                                group_sizes: list[int] = []
                                for group in v:
                                    count = 0
                                    for source, unit_suffix in KonfAIApp._list_input_units([Path(p) for p in group]):
                                        source = Path(source)
                                        if source.is_dir():
                                            # A DICOM series / OME-Zarr store must travel whole: zip it and mark
                                            # the filename so the server re-extracts it as one directory volume.
                                            unit_dir = tempfile.mkdtemp(prefix="konfai_unit_")
                                            unit_zip_dirs.append(unit_dir)
                                            zip_base = os.path.join(unit_dir, "unit")
                                            shutil.make_archive(zip_base, "zip", root_dir=str(source))
                                            filename = f"unit_{count}{unit_suffix}.konfaidir.zip"
                                            files.append((k, (filename, open(f"{zip_base}.zip", "rb"))))
                                        else:
                                            files.append((k, (source.name, open(source, "rb"))))
                                        count += 1
                                    group_sizes.append(count)
                                data[f"{k}_groups"] = ",".join(str(c) for c in group_sizes)
                        elif k == "dataset":
                            if v is not None:
                                dataset_zip_dir = tempfile.mkdtemp(prefix="konfai_dataset_")
                                zip_base = os.path.join(dataset_zip_dir, "dataset")
                                shutil.make_archive(zip_base, "zip", root_dir=str(Path(v)))
                                files.append(("dataset", open(f"{zip_base}.zip", "rb")))
                        else:
                            data[k] = v

                    if "ensemble_models" in data:
                        if len(data["ensemble_models"]) > 0:
                            del data["ensemble"]
                            data["ensemble_models"] = ",".join(data["ensemble_models"])
                        else:
                            del data["ensemble_models"]
                    if "models" in data:
                        if len(data["models"]) > 0:
                            data["models"] = ",".join(data["models"])
                        else:
                            del data["models"]
                    if "gpu" in data:
                        data["gpu"] = ",".join(str(x) for x in data["gpu"])
                    connect_timeout = 60
                    read_timeout: int = 600

                    with requests.post(
                        f"{self.remote_server.get_url()}/apps/{self.app}/{func.__name__}",
                        files=files,
                        data=data,
                        headers=self.remote_server.get_headers(),
                        timeout=(connect_timeout, read_timeout),
                    ) as r:
                        if r.status_code == 401:
                            raise RuntimeError("Unauthorized: invalid or missing token")
                        r.raise_for_status()
                        resp = r.json()
                        job_id = resp["job_id"]

                    self.stream_logs(job_id)
                    self.download_result(job_id, output)
                    finished = True
                except (CancelProcess, KeyboardInterrupt):
                    print("[KonfAI-Apps] Interrupted (SIGINT/SIGTERM)")
                    raise SystemExit(130) from None
                except requests.RequestException as e:
                    raise RuntimeError(f"Failed to submit job to remote KonfAI server: {e}") from e
                finally:
                    for _, fh in files:
                        handle = fh[1] if isinstance(fh, tuple) else fh
                        try:
                            handle.close()
                        except (OSError, ValueError):
                            pass

                    for unit_dir in unit_zip_dirs:
                        shutil.rmtree(unit_dir, ignore_errors=True)

                    if dataset_zip_dir is not None:
                        shutil.rmtree(dataset_zip_dir, ignore_errors=True)

                    if job_id is not None and not finished:
                        try:
                            self.kill_job(job_id)
                        except (RuntimeError, TimeoutError) as kill_error:
                            print(f"[KonfAI-Apps] Failed to kill remote job {job_id}: {kill_error}")

        return wrapper

    @run_remote_job
    def infer(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/").resolve(),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        patch_size: list[int] | None = None,
        batch_size: int | None = None,
        uncertainty: bool = False,
        prediction_file: str = "Prediction.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def evaluate(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]],
        output: Path = Path("./Output/"),
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def uncertainty(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/"),
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def pipeline(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]] | None,
        output: Path = Path("./Output/"),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        patch_size: list[int] | None = None,
        batch_size: int | None = None,
        prediction_file: str = "Prediction.yml",
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        uncertainty: bool = True,
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        pass

    @run_remote_job
    def fine_tune(
        self,
        dataset: Path,
        name: str = "Finetune",
        output: Path = Path("./Output/"),
        epochs: int = 10,
        it_validation: int = 1000,
        models: list[str] = [],
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        config_file: str = "Config.yml",
        lr: float | None = None,
        tmp_dir: Path | None = None,
    ) -> None:
        pass


class KonfAIApp(AbstractKonfAIApp):
    """
    Local runner for KonfAI applications.

    This class executes inference/evaluation/uncertainty/fine-tuning locally by:
    - building a dataset folder structure expected by KonfAI
    - installing the appropriate model/config assets (HF or local directory)
    - invoking KonfAI predictor/evaluator/trainer functions
    - collecting outputs into a user-defined output folder

    The public methods (infer/evaluate/uncertainty/fine_tune) are wrapped by
    `run_distributed_app`, which runs each operation in an isolated temporary
    workspace.
    """

    def __init__(self, app: str, download: bool, force_update: bool) -> None:
        """
        Create a local KonfAI app runner from either a HuggingFace model spec
        or a local directory.

        Parameters
        ----------
        app : str
            Either:
            - "repo_id:app_name" (HuggingFace style), or
            - a local path identifying a model directory.

        Notes
        -----
        Sets `self.app_repository` to either:
        - LocalAppRepositoryFromHF
        - LocalAppRepositoryFromDirectory
        """
        self.app_repository: LocalAppRepository
        app_repository_info = get_app_repository_info(app, force_update or download)
        if not isinstance(app_repository_info, LocalAppRepository):
            raise TypeError(
                f"KonfAI apps can only be executed from a local application repository. "
                f"App '{app}' resolves to a {type(app_repository_info).__name__}, which is not local."
            )
        if download:
            app_repository_info.download_app()
        self.app_repository = app_repository_info

    @staticmethod
    def _match_supported(file: Path) -> bool:
        """
        Check whether the file has an extension supported by KonfAI datasets.

        Parameters
        ----------
        file : Path
            Candidate file.

        Returns
        -------
        bool
            True if the file matches one of `SUPPORTED_EXTENSIONS`.
        """
        lower = file.name.lower()
        return any(lower.endswith("." + ext) for ext in SUPPORTED_EXTENSIONS)

    @staticmethod
    def _supported_suffix(file: Path) -> str:
        """
        Return the registered format extension of `file`, with a leading dot.

        For names carrying extra dots (e.g. ``patient.1.nii.gz``) this returns the
        longest matching supported extension (``.nii.gz``) rather than every dotted
        segment (``.1.nii.gz``), so the ``Volume_i`` copy keeps a name KonfAI can read.

        Parameters
        ----------
        file : Path
            A file already known to have a supported extension.

        Returns
        -------
        str
            The extension (dot-prefixed); falls back to `file.suffix` if nothing matches.
        """
        lower = file.name.lower()
        matches = [ext for ext in SUPPORTED_EXTENSIONS if lower.endswith("." + ext)]
        if not matches:
            return file.suffix
        return "." + max(matches, key=len)

    @staticmethod
    def _list_supported_files(paths: list[Path]) -> list[Path]:
        """
        Expand a list of input paths into a flat list of supported files.

        Each element in `paths` may be:
        - a file: must match supported extensions
        - a directory: recursively scanned for supported files, listed in
          sorted path order so that cases pair consistently across groups

        Parameters
        ----------
        paths : list[Path]
            Files and/or directories provided by the user.

        Returns
        -------
        list[Path]
            All discovered supported files.

        Raises
        ------
        FileNotFoundError
            If a path does not exist, or contains no supported files.
        """
        files = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: '{path}'")

            if path.is_file():
                if KonfAIApp._match_supported(path):
                    files.append(path)
                else:
                    raise FileNotFoundError(f"No supported file found: '{path.name}' is not a supported format.")
            else:
                files.extend(sorted(f for f in path.rglob("*") if f.is_file() and KonfAIApp._match_supported(f)))
                if not files:
                    raise FileNotFoundError(f"No supported files found in directory: '{path}'.")
        return files

    @staticmethod
    def _directory_volume_suffix(path: Path) -> str | None:
        """Return the staging suffix if ``path`` is a directory that is ITSELF one volume.

        A DICOM series and an OME-Zarr store are directories, not files, so they must be staged
        whole (not fragmented into their slices/chunks). Returns the extension the KonfAI dataset
        backend resolves the staged volume under -- ``.ome.zarr``/``.zarr`` for an OME-Zarr store,
        ``""`` for a DICOM series directory (read as a bare ``Volume_i`` directory) -- or ``None``
        for a plain directory that holds separate per-case files.
        """
        if not path.is_dir():
            return None
        lower = path.name.lower()
        if lower.endswith(".ome.zarr"):
            return ".ome.zarr"
        if lower.endswith(".zarr"):
            return ".zarr"
        entries = sorted(path.iterdir(), key=lambda entry: entry.name)
        names = {entry.name for entry in entries}
        if names & {".zgroup", ".zattrs", "zarr.json"}:
            return ".ome.zarr"
        if any(name.lower().endswith((".dcm", ".dicom")) for name in names):
            return ""
        # A DICOM series is commonly exported with no extension at all, so the suffixes above miss it;
        # the Part-10 magic (``DICM`` at offset 128) in the first file is what identifies it then.
        files = [entry for entry in entries if entry.is_file()]
        if any(KonfAIApp._is_dicom_file(file) for file in files):
            return ""
        return None

    @staticmethod
    def _is_dicom_file(path: Path) -> bool:
        """Whether a file carries the DICOM Part-10 magic: ``DICM`` at offset 128.

        Reimplemented here (not imported from ``konfai``) to stay on KonfAI's public API rather than
        reach into a private helper.
        """
        try:
            with open(path, "rb") as file:
                return file.read(132)[128:132] == b"DICM"
        except OSError:
            return False

    @staticmethod
    def _list_input_units(paths: list[Path]) -> list[tuple[Path, str]]:
        """Expand input paths into ordered ``(source, staged_suffix)`` units for the ``Dataset/`` layout.

        A unit is either a single supported file, or a directory that is itself one volume (an
        OME-Zarr store or a DICOM series directory -- see :meth:`_directory_volume_suffix`). A plain
        directory is walked so each contained file / store / series becomes its own case, in sorted
        order so cases pair consistently across input groups.

        Raises
        ------
        FileNotFoundError
            If a path does not exist, is an unsupported file, or yields no volume.
        """
        units: list[tuple[Path, str]] = []
        visited: set[Path] = set()

        def walk(path: Path) -> None:
            directory_suffix = KonfAIApp._directory_volume_suffix(path)
            if path.is_file():
                if KonfAIApp._match_supported(path):
                    units.append((path, KonfAIApp._supported_suffix(path)))
            elif directory_suffix is not None:
                units.append((path, directory_suffix))
            else:
                # Guard against symlink loops: recurse into each real directory once (a self-referential
                # link would otherwise walk until the OS raises ELOOP and aborts the whole staging).
                real = path.resolve()
                if real in visited:
                    return
                visited.add(real)
                # A cyclic symlink raises OSError (ELOOP) at iterdir() itself, before the resolve()/visited
                # guard above can see the child -- skip the unreadable entry instead of aborting staging.
                try:
                    children = sorted(path.iterdir(), key=lambda entry: entry.name)
                except OSError:
                    return
                for child in children:
                    walk(child)

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: '{path}'")
            if path.is_file() and not KonfAIApp._match_supported(path):
                raise FileNotFoundError(f"No supported file found: '{path.name}' is not a supported format.")
            before = len(units)
            walk(path)
            if len(units) == before:
                raise FileNotFoundError(f"No supported inputs found in: '{path}'.")
        return units

    @staticmethod
    def _detect_group_format(dataset_dir: Path, group: str) -> str:
        """Detect the on-disk format token of a staged group, for reading it back.

        Directory-volume inputs (a DICOM series or an OME-Zarr store) need their real backend token
        (``dicom`` / ``omezarr``); single-file inputs keep the historical ``mha`` default so existing
        behaviour is unchanged.
        """
        root = Path(dataset_dir)
        if root.is_dir():
            for case in sorted(path for path in root.iterdir() if path.is_dir()):
                matches = sorted(case.glob(f"{group}*"))
                if not matches:
                    continue
                suffix = KonfAIApp._directory_volume_suffix(matches[0])
                if suffix is not None:
                    return "omezarr" if suffix in (".ome.zarr", ".zarr") else "dicom"
                return "mha"
        return "mha"

    @staticmethod
    def symlink(src: Path, dst: Path) -> None:
        """
        Create a symlink from `dst` pointing to `src`, with safe replacement.

        If `dst` already exists:
        - directories are removed (unless they are symlinks)
        - files are unlinked

        On platforms or filesystems that do not support symlinks (Windows without
        Developer Mode raises OSError WinError 1314), this falls back to copying:
        - directories via copytree
        - files via copy2

        Parameters
        ----------
        src : Path
            Source file or directory.
        dst : Path
            Destination symlink path.
        """
        # ``is_symlink()`` also catches a BROKEN link (its target is gone), which ``exists()`` reports
        # as absent -- left in place it makes both os.symlink and the copy fallback fail on FileExists.
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except OSError:
            # Windows without Developer Mode (WinError 1314), or a filesystem without symlink support.
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def _write_inputs_to_dataset(self, inputs: list[list[Path]]) -> None:
        """
        Build the on-disk Dataset/ structure for inference/evaluation inputs.

        Expected structure:
            ./Dataset/P{idx}/Volume_{i}{suffix}

        Where:
        - i is the input-group index (e.g., channel/modalities)
        - idx is the patient/case index

        Parameters
        ----------
        inputs : list[list[Path]]
            Nested list of paths. Each inner list is scanned for supported files.
        """
        dataset_path = Path("./Dataset/")
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for i, input_path in enumerate(inputs):
            for idx, (source, suffix) in enumerate(KonfAIApp._list_input_units(input_path)):
                KonfAIApp.symlink(source, dataset_path / f"P{idx:03d}" / f"Volume_{i}{suffix}")

    @staticmethod
    def _dataset_level(prediction_file: str, dataset_dir: Path) -> int:
        """The level ``dataset_dir`` will be read at, per the resolved config's ``dataset_filenames``.

        ``@N`` is a property of a dataset ENTRY: every group of that entry is read at that one level. So a
        synthesised default must be readable at the same level as the inputs it stands in for, and the
        config is the only place that level is declared.

        Format-neutral by construction: ``split_format_level`` knows nothing about any backend, and an
        entry that declares no ``@N`` — which is every non-pyramid format — yields 0, the level those
        backends have always been read at.

        Only the entry pointing at ``dataset_dir`` counts; a config listing several sources may read each
        at its own level, and the staged inputs live in exactly one of them.
        """
        path = Path(prediction_file)
        if not path.is_file():
            return 0
        from ruamel.yaml import YAML

        with open(path, encoding="utf-8") as file:
            data = YAML().load(file)
        dataset = ((data or {}).get("Predictor") or {}).get("Dataset") or {}
        target = dataset_dir.resolve()
        for entry in dataset.get("dataset_filenames") or []:
            filename, _, file_format = split_path_spec(
                str(entry),
                default_format="mha",
                allowed_flags={"a", "i"},
                supported_extensions=SUPPORTED_EXTENSIONS,
            )
            if Path(filename).resolve() == target:
                return split_format_level(file_format)[1]
        return 0

    def _fill_optional_inputs(self, provided: int, prediction_file: str = "Prediction.yml") -> None:
        """Synthesise declared defaults for optional inputs the caller did not provide.

        Inputs map positionally to ``Volume_0..N-1`` in ``app.json`` declaration order, so only trailing
        inputs can be omitted. An optional input may declare a ``default`` in app.json (``"ones"`` /
        ``"zeros"``); konfai-apps then creates that ``Volume_i`` for every case, shaped and geo-referenced
        like ``Volume_0`` but read from its header only (no pixel load) — so an app runs from its required
        inputs alone. The registration mask branches use ``"ones"`` (a whole-image mask restricts nothing).
        Optional inputs with no ``default`` are left absent; a genuinely-missing required input still fails
        downstream.

        "Like ``Volume_0``" has to include the LEVEL it is read at. On a multiscale input, level 1 is not
        the shape of level 0, so a default sized from level 0 is both the wrong shape and unopenable at
        ``@1`` — the store has one level and the reader asks for the second. Taking the level from the
        config keeps this one code path for every format: a flat input reports level 0 and behaves exactly
        as before.
        """
        declared = list(self.app_repository.get_inputs().items())
        fills = {
            i: entry.default
            for i, (_, entry) in enumerate(declared)
            if i >= provided and not entry.required and entry.default is not None
        }
        if not fills:
            return
        fill_value = {"ones": 1, "zeros": 0}
        dataset_dir = Path("Dataset")
        file_format = KonfAIApp._detect_group_format(dataset_dir, "Volume_0")
        level = KonfAIApp._dataset_level(prediction_file, dataset_dir)
        dataset = Dataset("Dataset", f"{file_format}@{level}" if level else file_format)
        for name in dataset.get_names("Volume_0"):
            shape, attributes = dataset.get_infos("Volume_0", name)  # header only, no pixel read
            for i, default in fills.items():
                dataset.write(f"Volume_{i}", name, np.full(shape, fill_value[default], dtype=np.uint8), attributes)

    def _write_inference_stack_to_dataset(self, inputs: list[list[Path]]) -> None:
        """
        Build the Dataset/ structure for uncertainty estimation.

        This method enforces that each input file is a multi-component volume
        (e.g., an inference stack) by reading metadata with SimpleITK.

        Raises
        ------
        FileNotFoundError
            If a provided input is not multi-channel (single-component).
        """
        dataset_path = Path("./Dataset/")
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for i, input_path in enumerate(inputs):
            for idx, file in enumerate(KonfAIApp._list_supported_files(input_path)):
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(file))
                reader.ReadImageInformation()
                n_channels = reader.GetNumberOfComponents()
                if n_channels > 1:
                    suffix = KonfAIApp._supported_suffix(file)
                    KonfAIApp.symlink(file, dataset_path / f"P{idx:03d}" / f"Volume_{i}{suffix}")
                else:
                    raise FileNotFoundError(
                        "Invalid input volume for inference: a multi-channel volume stack is required, "
                        "but a single-channel volume was provided."
                    )

    def _write_gt_to_dataset(self, gt: list[list[Path]]) -> None:
        """
        Write ground-truth volumes into the Dataset/ structure.

        Expected structure:
            ./Dataset/P{idx}/Reference_{i}{suffix}

        Parameters
        ----------
        gt : list[list[Path]]
            Ground truth file paths grouped similarly to inputs.
        """
        for i, gt_path in enumerate(gt):
            for idx, (source, suffix) in enumerate(KonfAIApp._list_input_units(gt_path)):
                KonfAIApp.symlink(source, Path(f"./Dataset/P{idx:03d}/Reference_{i}{suffix}"))

    def _write_mask_or_default(self, mask: list[list[Path]] | None) -> None:
        """
        Write mask volumes into the Dataset/ structure or generate default masks.

        If `mask` is None:
        - creates a mask of ones for each case using the shape/metadata of Volume_0

        If `mask` is provided:
        - symlinks mask files as:
            ./Dataset/P{idx}/Mask_{i}{suffix}

        Parameters
        ----------
        mask : list[list[Path]] | None
            Optional mask paths.
        """
        if mask is None:
            dataset = Dataset("Dataset", KonfAIApp._detect_group_format(Path("Dataset"), "Volume_0"))
            names = dataset.get_names("Volume_0")
            for name in names:
                shape, attr = dataset.get_infos("Volume_0", name)  # header only, no pixel read
                dataset.write("Mask_0", name, np.ones(shape, dtype=np.uint8), attr)
        else:
            for i, mask_path in enumerate(mask):
                for idx, (source, suffix) in enumerate(KonfAIApp._list_input_units(mask_path)):
                    KonfAIApp.symlink(source, Path(f"./Dataset/P{idx:03d}/Mask_{i}{suffix}"))

    @run_distributed_app
    def infer(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/").resolve(),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        patch_size: list[int] | None = None,
        batch_size: int | None = None,
        config_overrides: list[str] | None = None,
        uncertainty: bool = False,
        prediction_file: str = "Prediction.yml",
        gpu: list[int] | None = None,
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run inference locally for the given inputs.

        Steps:
        1. Build Dataset/ from `inputs`
        2. Install inference assets (models/config) via `self.app_repository.install_inference`
        3. Call `konfai.predictor.predict(...)`
        4. Copy generated predictions into `output` if they exist

        Notes
        -----
        - Executes inside an isolated temporary workspace (via run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        gpu = cuda_visible_devices() if gpu is None else gpu
        self._write_inputs_to_dataset(inputs)
        available_vram = None
        if len(gpu):
            available_vram_per_device: list[float] = []
            for device in gpu:
                used_gb, total_gb = get_vram([device])
                available_vram_per_device.append(total_gb - used_gb)

            available_vram = min(available_vram_per_device)
        models_path = self.app_repository.install_inference(
            tta,
            ensemble,
            ensemble_models,
            mc,
            uncertainty,
            prediction_file,
            available_vram,
            forced_patch_size=patch_size,
            forced_batch_size=batch_size,
            config_overrides=config_overrides,
        )
        # After install_inference, not before: the defaults are sized from the level the config asks for,
        # and that config only exists here -- installation is what writes it out with any --set applied.
        self._fill_optional_inputs(len(inputs), prediction_file)
        from konfai.predictor import predict

        # predictions_dir is passed explicitly: predict()'s default is resolved when konfai.predictor is
        # first imported, so a host that imports it before this chdir'd workspace (e.g. the konfai-mcp
        # job runner) would silently write predictions outside ./Predictions and break collection below.
        predict(
            models_path,
            True,
            gpu,
            cpu,
            quiet,
            False,
            Path(prediction_file).resolve(),
            predictions_dir=Path("./Predictions").resolve(),
        )
        if Path("./Predictions").absolute().exists():
            shutil.copytree(Path("./Predictions").absolute(), output, dirs_exist_ok=True)

    @run_distributed_app
    def evaluate(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]],
        output: Path = Path("./Output/"),
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        gpu: list[int] | None = None,
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run evaluation locally against ground-truth.

        Steps:
        1. Build Dataset/ from inputs and gt
        2. Ensure masks exist (provided or generated)
        3. Install evaluation assets via `self.app_repository.install_evaluation`
        4. Call `konfai.evaluator.evaluate(...)`
        5. Copy evaluation outputs into `output`

        Notes
        -----
        - Runs inside an isolated workspace (run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        gpu = cuda_visible_devices() if gpu is None else gpu
        self._write_inputs_to_dataset(inputs)
        self._write_gt_to_dataset(gt)
        self._write_mask_or_default(mask)
        self.app_repository.install_evaluation(evaluation_file)
        from konfai.evaluator import evaluate

        # evaluations_dir passed explicitly for the same import-time-default reason as predict() above.
        evaluate(
            True,
            gpu,
            cpu,
            quiet,
            False,
            Path(evaluation_file).resolve(),
            evaluations_dir=Path("./Evaluations").resolve(),
        )
        if Path("./Evaluations").exists():
            shutil.copytree("./Evaluations", output, dirs_exist_ok=True)

    @run_distributed_app
    def uncertainty(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/"),
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] | None = None,
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run uncertainty estimation locally.

        Steps:
        1. Validate that inputs are multi-component inference stacks
        2. Install uncertainty assets via `self.app_repository.install_uncertainty`
        3. Call evaluator with an explicit output directory (./Uncertainties)
        4. Copy uncertainty results into `output`

        Notes
        -----
        - Runs inside an isolated workspace (run_distributed_app).
        - GPU defaults to `cuda_visible_devices()`.
        """
        gpu = cuda_visible_devices() if gpu is None else gpu
        self._write_inference_stack_to_dataset(inputs)
        self.app_repository.install_uncertainty(uncertainty_file)
        from konfai.evaluator import evaluate

        evaluate(True, gpu, cpu, quiet, False, Path(uncertainty_file).resolve(), Path("./Uncertainties/"))
        if Path("./Uncertainties").exists():
            shutil.copytree("./Uncertainties", output, dirs_exist_ok=True)

    def pipeline(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]] | None,
        output: Path = Path("./Output/"),
        ensemble: int = 0,
        ensemble_models: list[str] = [],
        tta: int = 0,
        mc: int = 0,
        patch_size: list[int] | None = None,
        batch_size: int | None = None,
        config_overrides: list[str] | None = None,
        prediction_file: str = "Prediction.yml",
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        uncertainty: bool = True,
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] | None = None,
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Run a full pipeline locally: inference → evaluation → uncertainty.

        This is a convenience method that orchestrates multiple stages and organizes
        outputs into subfolders:

        - ``<output>/Predictions``
        - ``<output>/Evaluations``
        - ``<output>/Uncertainties``

        Behavior:

        - always runs inference
        - runs evaluation only if ``gt`` is provided
        - runs uncertainty only if ``uncertainty=True``
        """
        gpu = cuda_visible_devices() if gpu is None else gpu
        self.infer(
            inputs=inputs,
            output=output / "Predictions",
            ensemble=ensemble,
            ensemble_models=ensemble_models,
            tta=tta,
            mc=mc,
            patch_size=patch_size,
            batch_size=batch_size,
            config_overrides=config_overrides,
            uncertainty=uncertainty,
            prediction_file=prediction_file,
            gpu=gpu,
            cpu=cpu,
            quiet=quiet,
            tmp_dir=tmp_dir,
        )
        outputs: list[Path] = []
        inference_stacks: list[Path] = []

        def _collect(path: Path) -> None:
            # Treat a directory that is itself one volume (DICOM series / OME-Zarr store) as a single
            # output unit instead of descending into its slices/chunks.
            if path.is_file():
                if KonfAIApp._match_supported(path):
                    (inference_stacks if path.name == "InferenceStack.mha" else outputs).append(path)
            elif KonfAIApp._directory_volume_suffix(path) is not None:
                outputs.append(path)
            else:
                for child in sorted(path.iterdir(), key=lambda entry: entry.name):
                    _collect(child)

        predictions_dir = output / "Predictions"
        if predictions_dir.exists():
            _collect(predictions_dir)
        if gt is not None:
            self.evaluate([outputs], gt, output / "Evaluations", mask, evaluation_file, gpu, cpu, quiet, tmp_dir)
        if uncertainty:
            self.uncertainty([inference_stacks], output / "Uncertainties", uncertainty_file, gpu, cpu, quiet, tmp_dir)

    @staticmethod
    def _weights_only_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
        """
        Load a checkpoint and keep only its model weights.

        Fine-tuning must start from the pretrained weights but with fresh training counters,
        optimizer and LR schedule. Returning only the ``Model`` key (dropping ``epoch``/``it``/
        ``loss``, ``Model_EMA`` and the optimizer/scheduler state) makes a subsequent RESUME leave
        ``epoch``/``it`` at 0, so ``range(0, epochs)`` runs all fine-tuning epochs with a fresh LR
        schedule.
        """
        state_dict = safe_torch_load(checkpoint_path, "cpu")
        if "Model" not in state_dict:
            raise AppRepositoryError(f"Checkpoint '{checkpoint_path}' has no 'Model' weights to fine-tune from.")
        return {"Model": state_dict["Model"]}

    @run_distributed_app
    def fine_tune(
        self,
        dataset: Path,
        name: str = "Finetune",
        output: Path = Path("./Output/"),
        epochs: int = 10,
        it_validation: int = 1000,
        models: list[str] = [],
        gpu: list[int] | None = None,
        cpu: int | None = None,
        quiet: bool = False,
        config_file: str = "Config.yml",
        lr: float | None = None,
        tmp_dir: Path | None = None,
    ) -> None:
        """
        Fine-tune one or several checkpoints of the app locally.

        Steps:
        1. Install training assets/config and resolve the selected checkpoint(s) via
           `self.app_repository.install_fine_tune`.
        2. Link the user dataset into ./Dataset.
        3. For each selected checkpoint: sanitize it to weights-only, write a per-model config with a
           distinct ``train_name``, run `konfai.trainer.train(...)` in resume mode, then copy the
           produced checkpoint back into the output app.

        The output directory is left as a clean, resolvable app bundle (app.json + config + code +
        fine-tuned checkpoint(s)); training artifacts (Checkpoints/Statistics) are kept out of it.

        Notes
        -----
        - Runs inside an isolated workspace (run_distributed_app). The CLI passes ``tmp_dir=output``
          (the workspace IS the output dir); other callers get the bundle files copied into ``output``.
        - Fine-tuning requires a CUDA GPU when the loss relies on GPU-only components.
        - `models` selects which checkpoint(s) to fine-tune (default: the first available).
        """
        gpu = cuda_visible_devices() if gpu is None else gpu
        import torch

        selected_models = self.app_repository.install_fine_tune(
            config_file, Path("./"), name, epochs, it_validation, models
        )
        KonfAIApp.symlink(dataset, Path("./Dataset").absolute())

        from konfai.trainer import train

        # Keep training artifacts (Checkpoints/Statistics) outside the output so it stays a clean app.
        work_dir = Path(tempfile.mkdtemp(prefix="konfai_finetune_")).resolve()
        config_path = Path(config_file)
        # Relative classpaths (e.g. 'classpath: sub/UNet.yml') resolve against the config file's parent,
        # so the app's support files -- including any that live in a subpackage -- must sit next to the
        # per-model config copies written into work_dir, at the same relative path.
        for support in Path(".").rglob("*"):
            if not support.is_file() or "__pycache__" in support.parts:
                continue
            if support.suffix.lower() not in {".py", ".yml", ".yaml"}:
                continue
            if support.resolve() == config_path.resolve():
                continue
            dest = work_dir / support
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(support, dest)
        try:
            for checkpoint_name, checkpoint_src in selected_models:
                stem = Path(checkpoint_name).stem
                train_name = f"{name}_{stem}"

                sanitized_ckpt = work_dir / f"{stem}_init.pt"
                torch.save(KonfAIApp._weights_only_checkpoint(checkpoint_src), sanitized_ckpt)  # nosec B614

                model_config = work_dir / f"{config_path.stem}_{stem}{config_path.suffix}"
                yaml = YAML()
                with open(config_path) as file:
                    data = yaml.load(file)
                data["Trainer"]["train_name"] = train_name
                with open(model_config, "w") as file:
                    yaml.dump(data, file)

                train(
                    State.RESUME,
                    True,
                    sanitized_ckpt,
                    gpu,
                    cpu,
                    quiet,
                    False,
                    model_config,
                    work_dir / "Checkpoints",
                    work_dir / "Statistics",
                    lr=lr,
                )

                produced_dir = work_dir / "Checkpoints" / train_name
                produced = sorted(produced_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
                if not produced:
                    raise AppRepositoryError(
                        f"Fine-tuning of '{checkpoint_name}' produced no checkpoint in '{produced_dir}'."
                    )
                shutil.copy2(produced[-1], Path(checkpoint_name).name)
            # Honour `output` for every caller: the CLI passes tmp_dir=output (workspace IS the output
            # dir, nothing to do), but any other caller runs in a throwaway workspace that the decorator
            # deletes -- so collect the resulting bundle files (app.json + configs + code + fine-tuned
            # checkpoints) into `output`, mirroring how infer collects ./Predictions.
            output_dir = Path(output).resolve()
            if Path.cwd().resolve() != output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                # Walk the whole bundle so nested assets survive (a subpackaged .yml/.py, requirements.txt,
                # app.json, and the fine-tuned .pt) -- a root-only iterdir silently drops everything nested.
                # The ./Dataset symlink is the user's input, not a bundle asset, so it is excluded.
                skip = {"Dataset"}
                for artifact in Path(".").rglob("*"):
                    if not artifact.is_file() or "__pycache__" in artifact.parts or skip.intersection(artifact.parts):
                        continue
                    dst = output_dir / artifact.relative_to(".")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(artifact, dst)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            dataset_link = Path("./Dataset")
            if dataset_link.is_symlink() or dataset_link.is_file():
                dataset_link.unlink()
            elif dataset_link.is_dir():
                shutil.rmtree(dataset_link, ignore_errors=True)

    def __str__(self) -> str:
        return str(self.app_repository)


run_remote_job = KonfAIAppClient.run_remote_job
