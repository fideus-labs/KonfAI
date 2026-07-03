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
from konfai.utils.utils import SUPPORTED_EXTENSIONS
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
        # On déclenche une exception -> remonte -> finally exécuté
        raise CancelProcess(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        # On restaure les handlers d'origine
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

    def __init__(self) -> None:
        super().__init__()


class Cancelprocess(CancelProcess):
    """Backward-compatible cancellation exception kept for app integrations."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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
                                    for p in group:
                                        p = Path(p)
                                        group_files = KonfAIApp._list_supported_files([p]) if p.is_dir() else [p]
                                        for file_path in group_files:
                                            files.append((k, open(file_path, "rb")))
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
                        try:
                            fh.close()
                        except (OSError, ValueError):
                            pass

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
            - "repo_id:revision" (HuggingFace style), or
            - a local path identifying a model directory.

        Notes
        -----
        Sets `self.app_repository` to either:
        - AppRepositoryOnHF
        - AppRepository
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
    def symlink(src: Path, dst: Path) -> None:
        """
        Create a symlink from `dst` pointing to `src`, with safe replacement.

        If `dst` already exists:
        - directories are removed (unless they are symlinks)
        - files are unlinked

        On platforms or filesystems that do not support symlinks (e.g. some Windows
        environments), this falls back to copying:
        - directories via copytree
        - files via copy2

        Parameters
        ----------
        src : Path
            Source file or directory.
        dst : Path
            Destination symlink path.
        """
        if dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except OSError:
            # fallback Windows / FS without symlink
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
            for idx, file in enumerate(KonfAIApp._list_supported_files(input_path)):
                suffix = KonfAIApp._supported_suffix(file)
                KonfAIApp.symlink(file, dataset_path / f"P{idx:03d}" / f"Volume_{i}{suffix}")

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
            for idx, file in enumerate(KonfAIApp._list_supported_files(gt_path)):
                suffix = KonfAIApp._supported_suffix(file)
                KonfAIApp.symlink(file, Path(f"./Dataset/P{idx:03d}/Reference_{i}{suffix}"))

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
            dataset = Dataset("Dataset", "mha")
            names = dataset.get_names("Volume_0")
            for name in names:
                data, attr = dataset.read_data("Volume_0", name)
                dataset.write("Mask_0", name, np.ones_like(data), attr)
        else:
            for i, mask_path in enumerate(mask):
                for idx, file in enumerate(KonfAIApp._list_supported_files(mask_path)):
                    suffix = KonfAIApp._supported_suffix(file)
                    KonfAIApp.symlink(file, Path(f"./Dataset/P{idx:03d}/Mask_{i}{suffix}"))

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
        uncertainty: bool = False,
        prediction_file: str = "Prediction.yml",
        gpu: list[int] = cuda_visible_devices(),
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
        )
        from konfai.predictor import predict

        predict(models_path, True, gpu, cpu, quiet, False, Path(prediction_file).resolve())
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
        gpu: list[int] = cuda_visible_devices(),
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
        self._write_inputs_to_dataset(inputs)
        self._write_gt_to_dataset(gt)
        self._write_mask_or_default(mask)
        self.app_repository.install_evaluation(evaluation_file)
        from konfai.evaluator import evaluate

        evaluate(True, gpu, cpu, quiet, False, Path(evaluation_file).resolve())
        if Path("./Evaluations").exists():
            shutil.copytree("./Evaluations", output, dirs_exist_ok=True)

    @run_distributed_app
    def uncertainty(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/"),
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        self._write_inference_stack_to_dataset(inputs)
        self.app_repository.install_uncertainty(uncertainty_file)
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
        prediction_file: str = "Prediction.yml",
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        uncertainty: bool = True,
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = cuda_visible_devices(),
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
        self.infer(
            inputs=inputs,
            output=output / "Predictions",
            ensemble=ensemble,
            ensemble_models=ensemble_models,
            tta=tta,
            mc=mc,
            patch_size=patch_size,
            batch_size=batch_size,
            uncertainty=uncertainty,
            prediction_file=prediction_file,
            gpu=gpu,
            cpu=cpu,
            quiet=quiet,
            tmp_dir=tmp_dir,
        )
        outputs = []
        inference_stacks = []
        for f in sorted((output / "Predictions").rglob("*")):
            if f.is_file() and KonfAIApp._match_supported(f):
                if f.name == "InferenceStack.mha":
                    inference_stacks.append(f)
                else:
                    outputs.append(f)
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
        gpu: list[int] = cuda_visible_devices(),
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
        - Runs inside an isolated workspace (run_distributed_app); the workspace is the output dir.
        - Fine-tuning requires a CUDA GPU when the loss relies on GPU-only components.
        - `models` selects which checkpoint(s) to fine-tune (default: the first available).
        """
        import torch

        selected_models = self.app_repository.install_fine_tune(
            config_file, Path("./"), name, epochs, it_validation, models
        )
        KonfAIApp.symlink(dataset, Path("./Dataset").absolute())

        from konfai.trainer import train

        # Keep training artifacts (Checkpoints/Statistics) outside the output so it stays a clean app.
        work_dir = Path(tempfile.mkdtemp(prefix="konfai_finetune_")).resolve()
        config_path = Path(config_file)
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
