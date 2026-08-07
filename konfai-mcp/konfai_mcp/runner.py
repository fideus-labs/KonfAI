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

from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path
from queue import Empty
from typing import Any, cast
from uuid import uuid4

from konfai.evaluator import build_evaluate
from konfai.predictor import build_predict
from konfai.trainer import build_train
from konfai.transformer import build_transform
from konfai.utils.runtime import State, execute_distributed_object


def _subprocess_entry(queue: Any, target: str, kwargs: dict[str, Any], output_path: str = "") -> None:
    """Run one runner API and post its result back, with the child's stdio off the server's.

    The server speaks JSON-RPC over its own stdio, and this child inherits those descriptors. Anything it
    or its descendants write to fd 1 (KonfAI's own progress bars, a pip install during app resolution, a
    library banner at import) lands inside a protocol frame and desynchronises the stream: the response
    to THIS call is delivered and then destroyed, and the caller waits forever for an answer that already
    arrived. Redirecting at fd level covers the whole process tree; ``sys.stdout`` alone would not, since a
    grandchild inherits descriptors rather than Python objects.

    The output is not discarded: it is the child's only diagnostic, and ``run_api_in_subprocess`` attaches
    its tail to a failure. fd 0 goes to devnull so a stray read gets EOF instead of eating a request.
    """
    if output_path:
        try:
            with open(output_path, "ab") as sink:
                try:
                    with open(os.devnull, "rb") as devnull:
                        os.dup2(devnull.fileno(), 0)
                except OSError:
                    # A closed fd 0 makes a stray read fail instantly; left inherited, it could eat
                    # the client's protocol bytes. And it must not cost fds 1/2 their detach.
                    with contextlib.suppress(OSError):
                        os.close(0)
                os.dup2(sink.fileno(), 1)
                os.dup2(sink.fileno(), 2)
        except OSError as exc:
            # The call goes on: refusing every validation over an exotic fd state would be worse --
            # with fd 1 possibly still on the protocol. Said in the sink, by path: open() does not need
            # the fds that just failed, and a wedged or failing call then carries its explanation.
            with contextlib.suppress(OSError), open(output_path, "ab") as sink:
                sink.write(f"[konfai-mcp] could not detach the child stdio from the server's: {exc}\n".encode())
    try:
        module_name, function_name = target.split(":", 1)
        result = getattr(importlib.import_module(module_name), function_name)(**kwargs)
        queue.put({"ok_transport": True, "payload": result})
    except BaseException:
        queue.put({"ok_transport": False, "error": traceback.format_exc()})


def run_api_in_subprocess(target: str, kwargs: dict[str, Any], timeout_s: float | None = None) -> Any:
    """Run one runner API in a fresh spawn interpreter and return its result.

    Isolation properties: agent-authored workspace code never executes in the server process,
    the child's os.chdir/env mutations cannot race other server threads, and a fresh interpreter
    guarantees an edited local Model.py/Loss.py is re-imported instead of read from a stale cache.

    A wall-clock ``timeout_s`` (default from ``KONFAI_MCP_SUBPROCESS_TIMEOUT``, else 1800s) bounds the wait
    so an alive-but-hung child cannot loop the server thread forever; on expiry the child is killed and a
    TimeoutError is raised. Pass ``0`` to wait unbounded.
    """
    if timeout_s is None:
        raw_timeout = os.environ.get("KONFAI_MCP_SUBPROCESS_TIMEOUT", "1800")
        try:
            timeout_s = float(raw_timeout)
        except ValueError:
            warnings.warn(
                f"Ignoring non-numeric KONFAI_MCP_SUBPROCESS_TIMEOUT={raw_timeout!r}; using 1800s.",
                stacklevel=2,
            )
            timeout_s = 1800.0
    deadline = time.monotonic() + timeout_s if timeout_s and timeout_s > 0 else None
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    # Resolve the entry through the live module so spawn-pickling stays valid even if this module was
    # reloaded (pickle requires the exact object currently registered as konfai_mcp.runner._subprocess_entry).
    entry = importlib.import_module("konfai_mcp.runner")._subprocess_entry
    # The child's stdio sink. Created here so the path survives a child that dies before saying anything.
    output = Path(tempfile.mkdtemp(prefix="konfai-mcp-")) / "subprocess.log"
    process = context.Process(target=entry, args=(queue, target, kwargs, str(output)), daemon=True)
    process.start()
    result: dict[str, Any] | None = None
    while result is None:
        try:
            result = queue.get(timeout=0.5)
        except Empty:
            if deadline is not None and time.monotonic() > deadline and process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                raise TimeoutError(
                    f"Isolated subprocess '{target}' exceeded {timeout_s:.0f}s and was terminated. "
                    "Raise KONFAI_MCP_SUBPROCESS_TIMEOUT if this is a large model, or simplify the config."
                ) from None
            if not process.is_alive():
                try:
                    result = queue.get(timeout=0.5)
                except Empty:
                    result = {
                        "ok_transport": False,
                        "error": (
                            f"The isolated subprocess died with exit code {process.exitcode} before returning "
                            "a result (native crash or OOM)."
                        ),
                    }
    # Bounded join: the result is already in hand, so the child should exit near-instantly. If it wedged
    # during teardown (e.g. a native/CUDA context that will not exit), an unbounded join would hang the
    # server thread forever (exactly what the timeout above exists to prevent), so escalate to
    # terminate/kill instead, matching the timeout branch.
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
    printed = _drain(output)
    if not result.get("ok_transport"):
        # The child's own output is where a native crash or an OOM leaves its only trace: the traceback
        # says the process died, this says what it was saying when it did.
        error = str(result.get("error") or "Isolated subprocess failed.")
        raise RuntimeError(f"{error}\n\nSubprocess output (tail):\n{printed}" if printed else error)
    return result["payload"]


@contextmanager
def preserved_config(config_path: Path):
    """Keep the authored config bytes across a child that builds a workflow from them.

    Building materializes every default into the file, and the child puts it back in its own
    ``finally``: which a timeout kill (SIGTERM) never reaches. Only the parent is guaranteed to
    outlive the child, so the snapshot belongs on this side of the spawn.
    """
    backup = config_path.read_text(encoding="utf-8") if config_path.is_file() else None
    try:
        yield
    finally:
        if backup is not None and config_path.is_file() and config_path.read_text(encoding="utf-8") != backup:
            # Written aside then renamed, like the restore in validate_workflow_api: this is the last
            # guard on the file, so a write that fails part-way must leave the config KonfAI rewrote
            # rather than a fragment of the authored one.
            staging = config_path.with_name(f".{config_path.name}.{uuid4().hex}.tmp")
            try:
                staging.write_text(backup, encoding="utf-8")
                os.replace(staging, config_path)
            except OSError:
                staging.unlink(missing_ok=True)
                raise


_OUTPUT_TAIL = 4000  # enough for a traceback and the lines around it, short enough to hand to an agent


def _drain(output: Path) -> str:
    """The tail of the child's captured stdio, then remove it. '' when it printed nothing."""
    try:
        text = output.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        text = ""
    shutil.rmtree(output.parent, ignore_errors=True)
    return text[-_OUTPUT_TAIL:]


def _ensure_local_imports() -> None:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _purge_workspace_modules(workspace: Path) -> None:
    """Drop cached imports of workspace-local modules so edited files are re-imported.

    KonfAI resolves local ``File:Class`` classpaths with a bare ``importlib.import_module``, which
    caches the module. In-process validation would otherwise keep validating the FIRST import of
    ``Model.py``/``Loss.py`` after the agent edits the file, silently returning stale results.
    """
    root = str(workspace.resolve())
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            resolved = str(Path(module_file).resolve())
        except OSError:
            continue
        if resolved.startswith(root + os.sep):
            del sys.modules[name]


def _apply_single_process_patches() -> None:
    """Force the KonfAI workflow to run inline in this process, without DDP or CUDA init.

    Validation/smoke children need a single deterministic process: no GPU setup (may be CUDA-less),
    no ``mp.spawn`` (the fn runs inline as rank 0), no ``dist.barrier`` (no process group exists).
    Only safe in a throwaway child; never call in a real training run.
    """
    import konfai.trainer as konfai_trainer
    import konfai.utils.runtime as konfai_runtime

    konfai_runtime.setup_gpu = lambda world_size, rank=None: (0, 0)  # type: ignore[assignment]
    konfai_runtime.mp.spawn = (  # type: ignore[assignment]
        lambda fn, nprocs, args=(), join=True, daemon=False, start_method="spawn": fn(0, *args)
    )
    konfai_trainer.dist.barrier = lambda: None  # type: ignore[assignment]


def _build_workflow(
    command: str,
    config: str,
    models: list[str] | None = None,
    model: str | None = None,
    lr: float | None = None,
):
    resolved_config = Path(config).resolve()
    if command in ("TRAIN", "RESUME"):
        resume_model: Path | str | None = None
        if model is not None:
            # KonfAI accepts https:// checkpoint URLs for RESUME; only local paths get resolved.
            resume_model = model if model.startswith(("http://", "https://")) else Path(model).resolve()
        return build_train(
            command=State.RESUME if command == "RESUME" else State.TRAIN,
            model=resume_model,
            config=resolved_config,
            checkpoints_dir=Path("./Checkpoints").resolve(),
            statistics_dir=Path("./Statistics").resolve(),
            lr=lr,
        )
    if command == "PREDICTION":
        return build_predict(
            models=[Path(model).resolve() for model in models or []],
            prediction_file=resolved_config,
            predictions_dir=Path("./Predictions").resolve(),
        )
    if command == "EVALUATION":
        return build_evaluate(
            evaluations_file=resolved_config,
            evaluations_dir=Path("./Evaluations").resolve(),
        )
    if command == "TRANSFORM":
        return build_transform(
            transform_file=resolved_config,
            transforms_dir=Path("./Transforms").resolve(),
        )
    raise ValueError(f"Unsupported command: {command}")


@contextmanager
def _runtime_context(cwd: Path | None = None, env_updates: dict[str, str] | None = None):
    previous_cwd = Path.cwd()
    previous_env = os.environ.copy()
    try:
        if cwd is not None:
            os.chdir(cwd)
        if env_updates:
            os.environ.update(env_updates)
        yield
    finally:
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous_env)


def run_workflow_api(
    *,
    command: str,
    config: str,
    models: list[str] | None = None,
    model: str | None = None,
    lr: float | None = None,
    gpu: list[int] | None = None,
    cpu: int | None = None,
    overwrite: bool = False,
    quiet: bool = False,
    tensorboard: bool = False,
    single_process: bool = False,
    cluster_kwargs: dict[str, Any] | None = None,
    cwd: str | None = None,
) -> None:
    """Child entrypoint that runs one KonfAI workflow (TRAIN/RESUME/PREDICTION/EVALUATION/TRANSFORM)."""
    with _runtime_context(cwd=Path(cwd).resolve() if cwd is not None else None):
        _ensure_local_imports()
        if single_process:
            _apply_single_process_patches()
        # Build-time sizing (the evaluation auto-patch) divides the auto memory budget by the per-node
        # rank count it finds in the environment. KonfAI's own launcher exports it, but this child
        # entrypoint builds the workflow itself: export it here too (the spawn child owns its env).
        os.environ["KONFAI_LOCAL_RANKS"] = str(max(1, len(gpu or []) or int(cpu or 1)))
        workflow = _build_workflow(command, config, models, model=model, lr=lr)
        execute_distributed_object(
            workflow,
            gpu=gpu,
            cpu=cpu,
            overwrite=overwrite,
            quiet=quiet,
            tensorboard=tensorboard,
            cluster_kwargs=cluster_kwargs,  # type: ignore[arg-type]
        )


def run_app_api(
    *,
    ref: str,
    inputs: list[list[str]],
    output: str,
    gpu: list[int] | None = None,
    cpu: int | None = None,
    tta: int = 0,
    ensemble: int = 0,
    ensemble_models: list[str] | None = None,
    patch_size: list[int] | None = None,
    batch_size: int | None = None,
    config_overrides: list[str] | None = None,
    uncertainty: bool = False,
    force_update: bool = False,
    quiet: bool = False,
    cwd: str | None = None,
) -> None:
    """Child entrypoint that runs a KonfAI app's inference.

    Resolving the app imports its Python code and pip-installs its requirements (the trust boundary,
    gated in the parent tool), so it runs here in the spawn subprocess. Local and HuggingFace apps
    only: a remote app server is not driven from the MCP.
    """
    with _runtime_context(cwd=Path(cwd).resolve() if cwd is not None else None):
        _ensure_local_imports()
        from konfai_apps.app import KonfAIApp

        app = KonfAIApp(ref, download=True, force_update=force_update)
        common: dict[str, Any] = {
            "inputs": [[Path(path) for path in group] for group in inputs],
            "output": Path(output).resolve(),
            "ensemble": ensemble,
            "ensemble_models": ensemble_models or [],
            "tta": tta,
            "patch_size": patch_size,
            "batch_size": batch_size,
            "uncertainty": uncertainty,
            "quiet": quiet,
        }
        if gpu is not None:
            common["gpu"] = gpu
        if cpu is not None:
            common["cpu"] = cpu
        app.infer(config_overrides=config_overrides, **common)


def run_app_action_api(
    *,
    ref: str,
    action: str,
    output: str,
    inputs: list[list[str]],
    gt: list[list[str]] | None = None,
    mask: list[list[str]] | None = None,
    extra: dict[str, Any] | None = None,
    force_update: bool = False,
    cwd: str | None = None,
) -> None:
    """Child entrypoint for an app action beyond plain infer: evaluate / uncertainty / pipeline.

    The parent (``AppService._prepare_action``) supplies the exact scalar kwargs for ``action`` in
    ``extra``; this converts the path groups and dispatches to the matching ``KonfAIApp`` method.
    Resolving the app imports its code and pip-installs (gated in the parent).
    """

    def _groups(value: list[list[str]]) -> list[list[Path]]:
        return [[Path(path) for path in group] for group in value]

    with _runtime_context(cwd=Path(cwd).resolve() if cwd is not None else None):
        _ensure_local_imports()
        from konfai_apps.app import KonfAIApp

        call: dict[str, Any] = {"inputs": _groups(inputs), "output": Path(output).resolve(), **(extra or {})}
        if action in ("evaluate", "pipeline"):
            call["gt"] = _groups(gt) if gt else None
            if mask is not None:
                call["mask"] = _groups(mask)
        getattr(KonfAIApp(ref, download=True, force_update=force_update), action)(**call)


def run_finetune_api(
    *,
    ref: str,
    dataset: str,
    output: str,
    name: str = "Finetune",
    epochs: int = 10,
    it_validation: int = 1000,
    models: list[str] | None = None,
    lr: float | None = None,
    config_overrides: list[str] | None = None,
    gpu: list[int] | None = None,
    cpu: int | None = None,
    config_file: str = "Config.yml",
    force_update: bool = False,
    quiet: bool = False,
    cwd: str | None = None,
) -> None:
    """Child entrypoint that fine-tunes a KonfAI app on the user's dataset, producing a bundle.

    Resolving the app imports its Python code and pip-installs its requirements (gated in the parent
    tool). Local and HuggingFace apps only.
    """
    with _runtime_context(cwd=Path(cwd).resolve() if cwd is not None else None):
        _ensure_local_imports()
        from konfai_apps.app import KonfAIApp

        app = KonfAIApp(ref, download=True, force_update=force_update)
        common: dict[str, Any] = {
            "dataset": Path(dataset).resolve(),
            "output": Path(output).resolve(),
            "name": name,
            "epochs": epochs,
            "it_validation": it_validation,
            "models": models or [],
            "lr": lr,
            "config_file": config_file,
            "quiet": quiet,
        }
        if gpu is not None:
            common["gpu"] = gpu
        if cpu is not None:
            common["cpu"] = cpu
        app.fine_tune(**common, config_overrides=config_overrides)


def app_parameters_api(*, ref: str, force_update: bool = False) -> dict[str, Any]:
    """Child entrypoint that reads an app's tunable parameters (``{values, constraints}``).

    ``get_parameters`` imports the app's model class to derive constraints from its type hints --
    the trust boundary, so it runs here in the spawn subprocess, never in the server process.
    """
    from konfai_apps.app_repository import LocalAppRepository, get_app_repository_info

    info = get_app_repository_info(ref, force_update)
    if not isinstance(info, LocalAppRepository):
        raise ValueError("Reading tunable parameters is only supported for local or HuggingFace apps.")
    parameters = info.get_parameters()
    return {
        "source": "hf" if "FromHF" in type(info).__name__ else "local",
        "values": parameters.get("values", {}),
        "constraints": parameters.get("constraints", {}),
    }


def import_app_api(
    *,
    ref: str,
    target: str,
    config_overrides: list[str] | None = None,
    display_name: str | None = None,
    force_update: bool = False,
) -> dict[str, Any]:
    """Child entrypoint: resolve + download an app bundle into the session root as a normal experiment.

    Copies the app's config(s), custom code, and .pt checkpoints into ``target`` and pip-installs its
    requirements. The resolve + download + install runs here in the spawn subprocess, never in the server
    process. Returns the copied filenames, the checkpoint names, and which config files landed.
    """
    from konfai_apps.app_repository import LocalAppRepository, get_app_repository_info

    info = get_app_repository_info(ref, force_update)
    if not isinstance(info, LocalAppRepository):
        raise ValueError("Importing an app into the session is only supported for local or HuggingFace apps.")
    destination = Path(target).resolve()
    filenames = info.download_bundle(destination, display_name=display_name, config_overrides=config_overrides)
    checkpoints = sorted(name for name in filenames if name.endswith(".pt"))
    configs = {
        key: name
        for key, name in (("train", "Config.yml"), ("prediction", "Prediction.yml"), ("evaluation", "Evaluation.yml"))
        if (destination / name).is_file()
    }
    return {"files": filenames, "checkpoints": checkpoints, "configs": configs}


def _collect_model_outputs(workflow_object: Any, workflow: str) -> dict[str, list[dict[str, Any]]]:
    """Enumerate every Network's addressable module paths (the valid outputs_criterions keys)."""
    networks: dict[str, Any] = {}

    def probe(label: str, candidate: Any) -> None:
        if hasattr(candidate, "named_module_args_dict") and label not in networks:
            networks[label] = candidate

    for attr, value in vars(workflow_object).items():
        probe(attr, value)
        if not hasattr(value, "named_module_args_dict") and hasattr(value, "get_model"):
            try:
                probe(attr, value.get_model(train=workflow == "train"))
            except Exception:  # a loader that cannot build outside setup is simply skipped
                continue
    return {
        label: [
            {
                "path": path,
                "terminal": bool(getattr(args, "_isEnd", False)),
                "module": type(module).__name__,
            }
            for path, module, args in network.named_module_args_dict()
        ]
        for label, network in networks.items()
    }


def smoke_test_component(
    *,
    classpath: str,
    kind: str,
    workspace_dir: str,
    shape: list[int] | None = None,
    init_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one component's runtime contract on dummy tensors.

    transform: asserts transform_shape() == __call__ output shape (a mismatch silently corrupts
    patch reassembly). criterion: reports Tensor-vs-tuple return (loss vs metric convention) and
    whether backward() works. Runs in the caller's spawn subprocess (never the server process);
    imports workspace code, so still only run it on trusted code.
    """
    kind_defaults = {
        "transform": "konfai.data.transform",
        "criterion": "konfai.metric.measure",
        "loss": "konfai.metric.measure",
        "metric": "konfai.metric.measure",
    }
    if kind not in kind_defaults:
        raise ValueError(f"Unsupported kind '{kind}'. Expected one of {sorted(kind_defaults)}.")
    env_updates = {"KONFAI_CONFIG_MODE": "Import", "CUDA_VISIBLE_DEVICES": ""}
    with _runtime_context(cwd=Path(workspace_dir).resolve(), env_updates=env_updates):
        _ensure_local_imports()
        _purge_workspace_modules(Path(workspace_dir))
        import torch
        from konfai.utils.utils import get_module

        dims = [int(value) for value in (shape or [1, 8, 8, 8])]
        payload: dict[str, Any] = {"classpath": classpath, "kind": kind, "shape": dims}
        try:
            module, name = get_module(classpath, kind_defaults[kind])
            instance = getattr(module, name)(**(init_kwargs or {}))
        except Exception as exc:
            return {
                **payload,
                "ok": False,
                "stage": "construct",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        try:
            if kind == "transform":
                from konfai.utils.dataset import Attribute

                predicted = [int(v) for v in instance.transform_shape("SMOKE", "case_0", list(dims), Attribute())]
                result = instance("case_0", torch.rand(dims), Attribute())
                actual = [int(v) for v in result.shape]
                payload.update(
                    {
                        "ok": actual == predicted,
                        "stage": "contract",
                        "predicted_shape": predicted,
                        "actual_shape": actual,
                        "dtype": str(result.dtype),
                    }
                )
                if not payload["ok"]:
                    payload["error"] = (
                        "transform_shape() does not match the __call__ output shape - patch planning "
                        "would silently corrupt reassembly. Fix transform_shape to predict the shape exactly."
                    )
            else:
                output = torch.rand(dims, requires_grad=True)
                result = instance(output, torch.rand(dims))
                if isinstance(result, tuple):
                    payload.update({"ok": True, "stage": "contract", "returns": "tuple", "behaves_as": "metric"})
                else:
                    payload.update({"ok": True, "stage": "contract", "returns": "tensor", "behaves_as": "loss"})
                    payload["value"] = float(result.detach().mean())
                    try:
                        result.mean().backward()
                        payload["backward_ok"] = True
                    except Exception as exc:
                        # A loss that returns a Tensor but cannot backprop cannot train a model: this is a
                        # contract failure, so propagate it into ok (the caller branches on ok alone).
                        payload["backward_ok"] = False
                        payload["backward_error"] = str(exc)
                        payload["ok"] = False
                        payload["error"] = (
                            "The criterion returned a loss Tensor but backward() failed, so it cannot train "
                            f"a model ({payload['backward_error']}). Make its output differentiable."
                        )
        except Exception as exc:
            return {
                **payload,
                "ok": False,
                "stage": "call",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        return payload


def _force_single_process_loading(workflow_object: Any) -> int:
    """Force num_workers=0 on the dataset before setup builds the dataloader; return the requested value.

    The validation subprocess is daemonic (run_api_in_subprocess), and daemonic processes cannot spawn
    children, so a DataLoader with num_workers>0 (the default on the streaming path) would raise
    'daemonic processes are not allowed to have children'. Loading single-process also means the real
    error surfaces here instead of being masked by a dying worker. Guarded so a core rename degrades
    gracefully rather than crashing the dry-run. The original ``num_workers`` is returned so the caller
    can run the worker-spawn picklability check only when the real run would actually spawn workers.
    """
    dataset = getattr(workflow_object, "dataset", None)
    args = getattr(dataset, "dataLoader_args", None)
    requested = 0
    if isinstance(args, dict):
        requested = int(args.get("num_workers", 0) or 0)
        args["num_workers"] = 0
        args.pop("prefetch_factor", None)  # invalid when num_workers == 0
        args.pop("persistent_workers", None)
    return requested


def _check_worker_spawn_picklability(workflow_object: Any, requested_num_workers: int) -> dict[str, Any]:
    """Pre-flight the exact operation a real num_workers>0 run performs: pickling the dataset for spawn.

    Validation loads single-process, so an unpicklable dataset member (an open file handle, a lambda in
    a transform, a SimpleITK object) passes validation and then kills the DataLoader workers at the real
    run's setup: the classic "validation passes, training crashes" gap. ``pickle.dumps`` on each
    dataloader's dataset reproduces the spawn transfer without spawning anything.
    """
    result: dict[str, Any] = {"requested_num_workers": requested_num_workers, "checked": False}
    if requested_num_workers <= 0:
        return result
    import pickle

    datasets: list[Any] = []
    for group in getattr(workflow_object, "dataloader", []) or []:
        loaders = group if isinstance(group, (list, tuple)) else [group]
        datasets.extend(dataset for dataset in (getattr(dl, "dataset", None) for dl in loaders) if dataset)
    for dataset in datasets:
        try:
            pickle.dumps(dataset)
        except Exception as exc:
            result.update(
                checked=True,
                picklable=False,
                error=f"{type(exc).__name__}: {exc}",
                hint=(
                    f"The config requests Dataset.num_workers={requested_num_workers}: the real run will "
                    "spawn DataLoader workers by pickling this dataset, and that pickling fails, so "
                    "training would die at setup with a masked worker crash. Make every dataset/transform "
                    "attribute picklable (no lambdas, open handles, or SimpleITK objects kept on self), "
                    "or set Dataset.num_workers: 0."
                ),
            )
            return result
    result.update(checked=True, picklable=True, datasets=len(datasets))
    return result


def _run_one_train_step(workflow_object: Any) -> dict[str, Any]:
    """Run ONE forward (+ best-effort backward) on ONE batch to catch runtime-only config errors.

    setup() builds the dataloader but never runs a step, so loss/target dtype and shape mismatches, an
    outputs_criterions key that does not resolve to a produced output, and a detached loss all surface
    only at run time. This mirrors `_Trainer.run`'s dataset load plus `_Trainer.train`'s load-bearing
    steps (forward via Measure.update, backward) WITHOUT building _Trainer, so no SummaryWriter/checkpoint
    is written. It runs single-process
    (num_workers forced to 0 before setup) and CPU-only (CUDA_VISIBLE_DEVICES="" is set by the caller),
    so the real error is not masked by a dying DataLoader worker. It couples to core internals (Model/
    NetState/Network.backward); the attribute guards below degrade to a clear message instead of a crash
    if the core training API moves.
    """
    import torch
    from konfai.network.network import Model, NetState

    torch.manual_seed(0)
    if not hasattr(workflow_object, "dataloader") or not hasattr(workflow_object, "model"):
        return {"ran": False, "reason": "workflow object exposes no dataloader/model after setup; train_step skipped."}
    rank0 = workflow_object.dataloader[0]
    train_dl = rank0[0] if isinstance(rank0, (list, tuple)) else rank0
    model = Model(workflow_object.model)
    model.train()
    model.module.set_state(NetState.TRAIN)
    train_dl.dataset.load("Train")  # mirrors _Trainer.run's dataset load before the first step
    batch_sample = next(iter(train_dl))
    model(batch_sample)  # forward: runs the module graph AND computes every loss via Measure.update
    result: dict[str, Any] = {"ran": True, "forward": True}
    try:
        model.module.backward(model)  # backward + optimizer step (no-op if scaler/optimizer unset)
        result["backward"] = True
    except Exception as exc:  # a detached/non-differentiable loss surfaces here, not a config-build error
        result["backward"] = False
        result["backward_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def plan_transform_api(
    *,
    workspace_dir: str,
    config: str,
    cpu: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Child entrypoint for the TRANSFORM dry-run: plan every case, write nothing.

    The plan is the workflow's own verdict, not a prediction (it opens and removes a real
    region-write on each destination), so it is the one thing an agent should read before launching
    a job that writes a dataset. Runs in a spawn child like every other workflow API: the config is
    rewritten in place by KonfAI's own binder, so its bytes are snapshotted and restored.
    """
    from konfai.transformer import Transformer, build_transform

    config_path = Path(config).resolve()
    config_backup = config_path.read_text(encoding="utf-8") if config_path.is_file() else None
    workspace = Path(workspace_dir).resolve()
    with _runtime_context(cwd=workspace, env_updates={"KONFAI_VERBOSE": "False"}):
        _ensure_local_imports()
        _purge_workspace_modules(workspace)
        try:
            workflow = cast(
                Transformer,
                build_transform(
                    transform_file=config_path,
                    transforms_dir=(workspace / "Transforms").resolve(),
                ),
            )
            plan = workflow.compute_plan(max(1, int(cpu)), bool(overwrite))
        finally:
            if config_backup is not None:
                config_path.write_text(config_backup, encoding="utf-8")
    counts: dict[str, int] = {}
    for entry in plan.entries:
        counts[entry.verdict] = counts.get(entry.verdict, 0) + 1
    return {
        "ok": True,
        "config_path": str(config_path),
        "report": plan.report(),
        "verdict_counts": counts,
        "budget_bytes": plan.budget_bytes,
        "budget": plan.budget_desc,
        "world_size": plan.world_size,
        # The entries a reader has to act on. STREAM lines are the expected case and would drown them.
        "needs_attention": [
            {
                "case": entry.case,
                "group_src": entry.group_src,
                "group_dest": entry.group_dest,
                "verdict": entry.verdict,
                "reason": entry.reason,
                "working_set_bytes": entry.working_set_bytes,
            }
            for entry in plan.entries
            if entry.verdict in ("WHOLE-VOLUME", "REFUSED")
        ][:50],
        "over_budget": [entry.case for entry in plan.budget_violations()],
    }


def validate_workflow_api(
    *,
    workflow: str,
    level: str,
    workspace_dir: str,
    config: str,
    models: list[str] | None = None,
    single_process: bool = False,
    validate_root: str | None = None,
    collect_model_outputs: bool = False,
) -> dict[str, Any]:
    """Child entrypoint that builds (and optionally sets up / one-steps) a workflow, and puts back what it touched.

    Levels: 'instantiate' builds the workflow object, 'setup' also builds datasets/dataloaders,
    'train_step' additionally runs one forward+backward on one batch. The authored config bytes are
    snapshotted and restored, and all outputs go to a throwaway validate root, with one exception.
    ``workflow='transform'`` at ``level='setup'`` runs ``Transformer.setup``, whose plan opens and
    removes a real region-write probe on every destination the config's Write/Save stages name: those
    are the author's own stores, not the validate root. A directory store gets a probe case created
    then removed; a single-file store (h5) is created if it was not already there. ``plan_transform``
    is the tool for a plan.
    """
    resolved_validate_root = (
        Path(validate_root).resolve()
        if validate_root is not None
        else Path(os.environ["KONFAI_MCP_VALIDATE_ROOT"]).resolve()
    )
    resolved_validate_root.mkdir(parents=True, exist_ok=True)
    validate_checkpoints = resolved_validate_root / "Checkpoints"
    validate_statistics = resolved_validate_root / "Statistics"
    validate_predictions = resolved_validate_root / "Predictions"
    validate_evaluations = resolved_validate_root / "Evaluations"
    validate_transforms = resolved_validate_root / "Transforms"
    for path in (
        validate_checkpoints,
        validate_statistics,
        validate_predictions,
        validate_evaluations,
        validate_transforms,
    ):
        path.mkdir(parents=True, exist_ok=True)
    env_updates = {
        "KONFAI_VERBOSE": "False",
        "KONFAI_OVERWRITE": "True",
        "CUDA_VISIBLE_DEVICES": "",
    }

    # Building a workflow runs KonfAI with KONFAI_CONFIG_MODE='Done', whose Config.__exit__ rewrites
    # the config file in place (materialising every default). Validation must be side-effect-free on
    # the agent's authored config, so snapshot its bytes and restore them afterwards.
    config_path = Path(config).resolve()
    config_backup = config_path.read_text(encoding="utf-8") if config_path.is_file() else None

    with _runtime_context(cwd=Path(workspace_dir).resolve(), env_updates=env_updates):
        _ensure_local_imports()
        # Runs in the spawn subprocess (never the server process): purge cached workspace imports so an
        # edited Model.py/Loss.py is re-imported fresh instead of silently validating stale code.
        _purge_workspace_modules(Path(workspace_dir))
        try:
            if single_process:
                _apply_single_process_patches()

            if workflow == "train":
                workflow_object = build_train(
                    command=State.TRAIN,
                    config=Path(config).resolve(),
                    checkpoints_dir=validate_checkpoints,
                    statistics_dir=validate_statistics,
                )
            elif workflow == "prediction":
                dummy_model = resolved_validate_root / "dummy_prediction_model.pt"
                if models is None:
                    dummy_model.write_text("konfai-mcp validation placeholder\n", encoding="utf-8")
                workflow_object = build_predict(
                    models=[Path(model).resolve() for model in (models or [str(dummy_model)])],
                    prediction_file=Path(config).resolve(),
                    predictions_dir=validate_predictions,
                )
            elif workflow == "evaluation":
                workflow_object = build_evaluate(
                    evaluations_file=Path(config).resolve(),
                    evaluations_dir=validate_evaluations,
                )
            elif workflow == "transform":
                workflow_object = build_transform(
                    transform_file=Path(config).resolve(),
                    transforms_dir=validate_transforms,
                )
            else:
                raise ValueError(f"Unsupported validation workflow: {workflow}")

            payload: dict[str, Any] = {
                "ok": True,
                "workflow": workflow,
                "level": level,
                "object_type": type(workflow_object).__name__,
                "name": getattr(workflow_object, "name", None),
                "config_path": str(Path(config).resolve()),
            }
            if collect_model_outputs:
                payload["model_outputs"] = _collect_model_outputs(workflow_object, workflow)
            if level in ("setup", "train_step"):
                requested_num_workers = _force_single_process_loading(workflow_object)
                workflow_object.setup(1)
                payload["setup"] = {
                    "dataloaders": len(getattr(workflow_object, "dataloader", [])),
                    "size": getattr(workflow_object, "size", None),
                }
                payload["worker_spawn_check"] = _check_worker_spawn_picklability(workflow_object, requested_num_workers)
            if level == "train_step" and workflow == "train":
                payload["train_step"] = _run_one_train_step(workflow_object)
            return payload
        except Exception as exc:  # pragma: no cover - error shape tested through caller
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            # Restore the authored config so validation never mutates the file it validated.
            if config_backup is not None:
                try:
                    # Atomic restore: an OSError mid-write must not truncate the author's config. Write a
                    # sibling temp file and rename it into place (mirrors server_jobs._persist_job).
                    tmp = config_path.with_name(f".{config_path.name}.{uuid4().hex}.tmp")
                    try:
                        tmp.write_text(config_backup, encoding="utf-8")
                        os.replace(tmp, config_path)
                    except OSError:
                        tmp.unlink(missing_ok=True)
                        raise
                except OSError as restore_exc:
                    # The side-effect-free invariant is broken: KonfAI already rewrote the config with all
                    # defaults materialised and the author's bytes could not be put back. Never swallow
                    # this silently: a "success" payload would then hide a mutated config on disk.
                    warnings.warn(
                        f"Failed to restore the validated config at {config_path}: {restore_exc}. The file "
                        "was left rewritten by KonfAI (defaults materialised, not the authored bytes).",
                        stacklevel=2,
                    )
                    if "payload" in locals() and isinstance(payload, dict):
                        payload["config_restore_failed"] = str(restore_exc)
