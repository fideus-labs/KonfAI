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

"""Command-line entrypoints for standalone KonfAI Apps workflows and services."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from konfai import RemoteServer
from konfai.utils.errors import EXIT_OUT_OF_MEMORY, KonfAIError
from konfai.utils.utils import env_flag

from . import app as app_module
from .app_repository import (
    LocalAppRepository,
    LocalAppRepositoryFromHF,
    did_you_mean,
    get_app_repository_info,
    get_available_apps_on_hf_repo,
)
from .errors import AppRepositoryError
from .options import add_device, add_tmp_dir

if TYPE_CHECKING:
    from .app import AbstractKonfAIApp


#: Where a designed refusal also writes its message, when set: the app server reads it back to answer
#: the job's result with the refusal (422) rather than a bare exit code.
REFUSAL_FILE_ENV = "KONFAI_APPS_REFUSAL_FILE"


@contextmanager
def _exit_on_refusal() -> Iterator[None]:
    """A designed refusal (the app layer's or a workflow's it runs) prints its message and exits 1, as the
    konfai CLI does; the traceback only under KONFAI_DEBUG=1."""
    try:
        yield
    except KonfAIError as error:
        if env_flag("KONFAI_DEBUG", False):
            raise
        message = str(error).strip()
        print(message, file=sys.stderr)
        if refusal_file := os.environ.get(REFUSAL_FILE_ENV):
            Path(refusal_file).write_text(message + "\n", encoding="utf-8")
        sys.exit(1)
    except RuntimeError as error:  # torch.cuda.OutOfMemoryError, named without importing torch here
        if type(error).__name__ != "OutOfMemoryError":
            raise
        # Not a designed refusal but one a caller acts on: IMPACT-Reg re-plans smaller on this exit code.
        print(f"[KonfAI] out of GPU memory: {(str(error).splitlines() or ['CUDA'])[0]}", file=sys.stderr)
        sys.exit(EXIT_OUT_OF_MEMORY)


def _package_version() -> str:
    try:
        return importlib.metadata.version("konfai-apps")
    except importlib.metadata.PackageNotFoundError:
        return "0+local"


def add_common_konfai_apps(parser: argparse.ArgumentParser, with_uncertainty: bool = True) -> dict[str, Any]:
    """Add shared CLI arguments for app-focused commands and parse them."""
    parser.add_argument(
        "-i",
        "--inputs",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        required=True,
        help="Input path(s): provide one or multiple volume files, or a dataset directory.",
    )

    parser.add_argument(
        "--gt",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        help="Ground-truth path(s): provide one or multiple data files, or a dataset directory.",
    )

    parser.add_argument(
        "--mask",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        help="Optional evaluation mask path: provide one or multiple volume files, or a dataset directory.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=lambda x: Path(x).resolve(),
        default=Path("./Output").resolve(),
        help="Output directory / file",
    )
    if with_uncertainty:
        parser.add_argument("-uncertainty", action="store_true", help="Run uncertainty workflow.")

    add_device(parser)

    kwargs = vars(parser.parse_args())
    if kwargs["cpu"] is not None:
        kwargs["gpu"] = []
    if not with_uncertainty:
        kwargs["uncertainty"] = False
    return kwargs


def _resolved_path(value: str) -> Path:
    return Path(value).resolve()


def _add_app_io(parser: argparse.ArgumentParser, fine_tune: bool = False) -> None:
    """Add the input/output/device options shared by every app operation. Fine-tuning reads a dataset
    instead of inputs and works in its output directory, so it takes no temporary directory."""
    if fine_tune:
        parser.add_argument(
            "-d", "--dataset", type=_resolved_path, required=True, help="Dataset directory to fine-tune on."
        )
    else:
        parser.add_argument(
            "-i",
            "--inputs",
            type=_resolved_path,
            nargs="+",
            action="append",
            required=True,
            help="Input path(s): one or multiple volume files, or a dataset directory.",
        )
    parser.add_argument(
        "-o", "--output", type=_resolved_path, default=Path("./Output").resolve(), help="Output directory / file."
    )
    if not fine_tune:
        add_tmp_dir(parser)
    add_device(parser)


def _add_gt(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument(
        "--gt",
        type=_resolved_path,
        nargs="+",
        action="append",
        required=required,
        help="Ground-truth path(s): one or multiple data files, or a dataset directory.",
    )


def _add_mask(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mask", type=_resolved_path, nargs="+", action="append", help="Optional evaluation mask path(s)."
    )


def _add_patch_overrides(parser: argparse.ArgumentParser) -> None:
    """Add the optional patch/batch overrides (``--patch-size`` / ``--batch-size``).

    When omitted the app keeps its config's patch and batch (``batch_size: 0`` there measures the batch on
    the GPU); when given they force the inference ``Patch.patch_size`` / ``batch_size``. A single
    ``--patch-size`` value is an isotropic cube.
    """
    parser.add_argument(
        "--patch-size",
        "--patch_size",
        dest="patch_size",
        type=int,
        nargs="+",
        default=None,
        help="Override the inference patch size, e.g. '192' (cube) or '192 192 192'. Default: the app's config.",
    )
    parser.add_argument(
        "--max-voxels",
        "--max_voxels",
        dest="max_voxels",
        type=int,
        default=None,
        help="Override the voxels a pass holds before KonfAI resamples or tiles it (the config's Patch.max_voxels). "
        "Default: sized from the device and the cost the app declares.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override the inference batch size. Default: the app's config (`batch_size: 0` measures it on the GPU).",
    )


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    """Add the generic config override (``--set NAME=VALUE``, repeatable).

    A bare ``NAME`` tunes a model parameter (e.g. ``--set iterations=300``); a dotted ``NAME`` is a full
    path from the config root (e.g. ``Predictor.Dataset.batch_size=2``). The value is parsed as YAML (int /
    float / bool / list / string). This is the generic mechanism a UI drives to tune a preset's parameters.
    """
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        metavar="NAME=VALUE",
        default=None,
        help="Tune a model parameter, e.g. --set iterations=300 (repeatable; a dotted NAME targets any config key).",
    )


def app_id(repo: str, name: str) -> str:
    """The id of app ``name`` in ``repo``: a local directory of app folders, or a Hugging Face ``repo[@revision]``."""
    root = Path(repo).expanduser()
    return str(root / name) if root.is_dir() else f"{repo}:{name}"


def app_file(repo: str, name: str, filename: str) -> Path:
    """One file of app ``name`` in ``repo``, read cache first as the app's own run reads it."""
    root = Path(repo).expanduser()
    if root.is_dir():
        return root / name / filename
    return LocalAppRepositoryFromHF.download(repo, f"{name}/{filename}", False)


def app_names(repo: str, force_update: bool = False) -> list[str]:
    """The apps of ``repo``, sorted: its folders holding an ``app.json``."""
    root = Path(repo).expanduser()
    if root.is_dir():
        return sorted(folder.name for folder in root.iterdir() if (folder / "app.json").is_file())
    return get_available_apps_on_hf_repo(repo, force_update)


def list_apps(repo: str, task: str | None = None) -> list[tuple[str, dict]]:
    """Each app of ``repo`` with its ``app.json``, by name; with ``task``, those declaring another task left out (one
    declaring none is kept)."""
    apps = [
        (name, json.loads(app_file(repo, name, "app.json").read_text(encoding="utf-8"))) for name in app_names(repo)
    ]
    return [(name, manifest) for name, manifest in apps if task is None or manifest.get("task", task) == task]


def _show(app: str) -> None:
    """``show``: ``describe_app`` of ``app`` (``repo:name`` or a local app folder), printed."""
    repo, name = _split_app(app)
    print_app(name, describe_app(repo, name))


def _split_app(app: str) -> tuple[str, str]:
    """``repo:name`` or a local app folder -> ``(repo, name)``."""
    path = Path(app).expanduser()
    if path.is_dir():
        return str(path.resolve().parent), path.name
    repo, colon, name = app.partition(":")
    if not colon:
        raise AppRepositoryError(f"'{app}' names no app.", "Name it 'repo_id:app_name', or give a local app folder.")
    return repo, name


def describe_app(repo: str, name: str) -> dict[str, Any]:
    """What app ``name`` of ``repo`` runs, read from its own files before it runs (a warm cache answers offline): its
    ``app.json`` (``manifest``), the model ``classpath``, the ``--set`` ``parameters`` (``get_parameters``), the
    model files it downloads on its first run (``model_files`` in ``app.json``: ``repo_id``, ``revision``,
    ``filename``) with their size in bytes once cached, else None (``models``), and the ``requirements`` installed on
    first use."""
    from huggingface_hub import try_to_load_from_cache
    from ruamel.yaml import YAML

    try:
        info = get_app_repository_info(app_id(repo, name), False)
    except AppRepositoryError:
        names = app_names(repo)  # only now: a warm cache resolves one app offline, not the listing
        if name in names:
            raise
        raise AppRepositoryError(
            f"No app '{name}' in {repo}.{did_you_mean(name, names, 3)}", f"Its apps: {', '.join(names)}."
        ) from None
    if not isinstance(info, LocalAppRepository):
        raise AppRepositoryError(f"'{name}' is not a local or Hugging Face app.")
    files = {path.name: path for path in info.download_config_file()}
    config = YAML(typ="safe").load(files["Prediction.yml"].read_text(encoding="utf-8"))
    parameters = info.get_parameters()
    manifest = json.loads(files["app.json"].read_text(encoding="utf-8"))
    models: dict[str, int | None] = {}
    for model in manifest.get("model_files", []):
        cached = try_to_load_from_cache(model["repo_id"], model["filename"], revision=model.get("revision"))
        models[f"{model['repo_id']}:{model['filename']}"] = (
            Path(cached).stat().st_size if isinstance(cached, str) else None
        )
    lines = files["requirements.txt"].read_text(encoding="utf-8").splitlines() if "requirements.txt" in files else []
    return {
        "manifest": manifest,
        "classpath": config["Predictor"]["Model"]["classpath"],
        "parameters": parameters,
        "models": models,
        "requirements": [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")],
    }


def _megabytes(size: int) -> str:
    """A file size as read in a listing: 0.015 MB, 24 MB, 123 MB."""
    return f"{size / 1e6:.2g} MB" if size < 1e7 else f"{size / 1e6:.0f} MB"


def print_apps(prog: str, apps: list[tuple[str, dict]]) -> None:
    """``list``: each app's name and display name, its short description wrapped under them."""
    width = max((len(name) for name, _ in apps), default=0)
    columns = max(shutil.get_terminal_size().columns - width - 2, 40)
    for name, manifest in apps:
        print(f"{name:<{width}}  {manifest.get('display_name', '')}".rstrip())
        for line in textwrap.wrap(manifest.get("short_description", ""), columns):
            print(f"{'':<{width}}  {line}")
    print(f"\n'{prog} show NAME' says what an app runs, what it needs and what --set tunes.")


#: The ``app.json`` keys ``show`` prints in its header or leaves to a UI; the others are printed as they are.
_SHOWN_KEYS = {
    "display_name",
    "description",
    "short_description",
    "inputs",
    "outputs",
    "terminology",
    "icon",
    "model_files",
}


def print_app(name: str, described: dict[str, Any]) -> None:
    """``show NAME``: what the app does, runs and needs, its other ``app.json`` fields, and its --set parameters."""
    manifest = described["manifest"]
    fill = functools.partial(textwrap.fill, width=shutil.get_terminal_size().columns, break_on_hyphens=False)
    print(f"{name}: {manifest.get('display_name', '')}\n")
    print(fill(manifest.get("description") or manifest.get("short_description", "")), end="\n\n")
    model_class = described["classpath"].partition(":")[2]
    models = [
        f"{ref} ({_megabytes(size)})" if size else f"{ref} (not cached yet: downloaded on the first run)"
        for ref, size in described["models"].items()
    ]
    rows = [
        ("Model", described["classpath"]),
        ("Models", "; ".join(models) or "none"),
        ("Installs", f"{', '.join(described['requirements'])}, on first use" if described["requirements"] else ""),
    ]
    rows += [(key, json.dumps(value)) for key, value in manifest.items() if key not in _SHOWN_KEYS]
    width = max(len(label) for label, _ in rows) + 1
    for label, value in rows:
        if value:
            print(fill(value, initial_indent=f"  {label:<{width}}", subsequent_indent=" " * (width + 2)))

    print("\nParameters: --set NAME=VALUE (repeatable); a nested one takes its full path.")
    parameters, described_already = described["parameters"], set()

    def rows_of(values: dict, constraints: dict, prefix: str) -> list[tuple[str, object, dict]]:
        found = []
        for key, value in values.items():
            constraint = constraints.get(key) or constraints.get("*") or {}
            if isinstance(value, dict) and value:
                found += rows_of(value, constraint, f"{prefix or f'Predictor.Model.{model_class}.'}{key}.")
            else:
                found.append((f"{prefix}{key}", value, constraint))
        return found

    for path, value, constraint in rows_of(parameters["values"], parameters["constraints"], ""):
        shown = value if isinstance(value, str) else json.dumps(value)
        choices = constraint.get("choices") or []
        if 0 < len(choices) <= 8:
            shown += f"  ({' | '.join(map(str, choices))})"
        elif choices:
            shown += f"  (one of {len(choices)})"
        elif "min" in constraint:
            shown += f"  ({constraint['min']} to {constraint['max']})"
        print(fill(f"{path} = {shown}", initial_indent="  ", subsequent_indent="      "))
        description = constraint.get("description")
        if description and description not in described_already:
            described_already.add(description)
            print(fill(description, initial_indent="      ", subsequent_indent="      "))


def build_app_cli(
    prog: str,
    description: str,
    *,
    resolve_app: Callable[[argparse.Namespace], str],
    add_selection: Callable[[argparse.ArgumentParser], None] | None = None,
    add_infer_knobs: Callable[[argparse.ArgumentParser], None] | None = None,
    resolve_infer: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    infer_command: str = "infer",
    with_uncertainty: bool = True,
    repo: str | None = None,
) -> Callable[[], None]:
    """Build a ``main()`` for a repo-pinned app CLI exposing ``<infer_command>``/eval/uncertainty/pipeline.

    The set of *operations* is uniform across apps, but the *arguments* stay domain-specific through hooks:

    - ``resolve_app(args) -> "repo:app"``          resolves the pinned app id from the parsed selection;
    - ``add_selection(subparser)``                 adds the model/task selection args (shared by every command);
    - ``add_infer_knobs(subparser)``               adds the inference knobs (ensemble/folds/models/tta/mc …);
    - ``resolve_infer(args) -> dict``              maps those knobs to ``KonfAIApp.infer`` kwargs.

    ``infer_command`` names the inference operation with the app's own vocabulary (e.g. ``segment``,
    ``synthesize``); ``with_uncertainty=False`` drops the uncertainty command for models that do not support it.
    ``show`` says what the selected app runs and tunes; ``repo``, the repository the apps come from, adds ``list``.
    """
    select = add_selection or (lambda parser: None)
    knobs = add_infer_knobs or (lambda parser: None)
    infer_kwargs = resolve_infer or (lambda args: {})

    @_exit_on_refusal()
    def main() -> None:
        parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False,
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        if repo is not None:
            subparsers.add_parser("list", help=f"List the apps of {repo}.")
        select(subparsers.add_parser("show", help="Say what an app runs, what it needs and what --set tunes."))

        run_p = subparsers.add_parser(infer_command, help=f"Run {infer_command} (model inference).")
        select(run_p)
        _add_app_io(run_p)
        knobs(run_p)
        _add_patch_overrides(run_p)
        _add_config_overrides(run_p)
        if with_uncertainty:
            run_p.add_argument("-uncertainty", action="store_true", help="Also write the inference stack.")
        run_p.add_argument(
            "--prediction-file",
            "--prediction_file",
            dest="prediction_file",
            default="Prediction.yml",
            help="Prediction config filename inside the app.",
        )

        eval_p = subparsers.add_parser("eval", help="Evaluate the model against ground-truth labels.")
        select(eval_p)
        _add_app_io(eval_p)
        _add_gt(eval_p, required=True)
        _add_mask(eval_p)
        eval_p.add_argument(
            "--evaluation-file",
            "--evaluation_file",
            dest="evaluation_file",
            default="Evaluation.yml",
            help="Evaluation config filename inside the app.",
        )

        if with_uncertainty:
            unc_p = subparsers.add_parser("uncertainty", help="Compute model uncertainty.")
            select(unc_p)
            _add_app_io(unc_p)
            unc_p.add_argument(
                "--uncertainty-file",
                "--uncertainty_file",
                dest="uncertainty_file",
                default="Uncertainty.yml",
                help="Uncertainty config filename inside the app.",
            )

        pipe_p = subparsers.add_parser(
            "pipeline", help=f"Run {infer_command}, then evaluation and uncertainty in a single command."
        )
        select(pipe_p)
        _add_app_io(pipe_p)
        knobs(pipe_p)
        _add_patch_overrides(pipe_p)
        _add_config_overrides(pipe_p)
        _add_gt(pipe_p, required=False)
        _add_mask(pipe_p)
        pipe_p.add_argument(
            "--prediction-file",
            "--prediction_file",
            dest="prediction_file",
            default="Prediction.yml",
            help="Prediction config filename inside the app.",
        )
        pipe_p.add_argument(
            "--evaluation-file",
            "--evaluation_file",
            dest="evaluation_file",
            default="Evaluation.yml",
            help="Evaluation config filename inside the app.",
        )
        if with_uncertainty:
            pipe_p.add_argument(
                "--uncertainty-file",
                "--uncertainty_file",
                dest="uncertainty_file",
                default="Uncertainty.yml",
                help="Uncertainty config filename inside the app.",
            )
            pipe_p.add_argument("-uncertainty", action="store_true", help="Also run the uncertainty workflow.")

        args = parser.parse_args()
        if args.command == "list":
            print_apps(prog, list_apps(str(repo)))
            return
        if args.command == "show":
            _show(resolve_app(args))
            return
        gpu = [] if args.cpu is not None else args.gpu
        konfai_app = app_module.KonfAIApp(resolve_app(args), args.download, args.force_update)

        if args.command == infer_command:
            konfai_app.infer(
                inputs=args.inputs,
                output=args.output,
                tmp_dir=args.tmp_dir,
                gpu=gpu,
                cpu=args.cpu,
                quiet=args.quiet,
                uncertainty=getattr(args, "uncertainty", False),
                prediction_file=args.prediction_file,
                patch_size=args.patch_size,
                max_voxels=getattr(args, "max_voxels", None),
                batch_size=args.batch_size,
                config_overrides=args.config_overrides,
                **infer_kwargs(args),
            )
        elif args.command == "eval":
            konfai_app.evaluate(
                inputs=args.inputs,
                gt=args.gt,
                mask=args.mask,
                output=args.output,
                tmp_dir=args.tmp_dir,
                evaluation_file=args.evaluation_file,
                gpu=gpu,
                cpu=args.cpu,
                quiet=args.quiet,
            )
        elif args.command == "uncertainty":
            konfai_app.uncertainty(
                inputs=args.inputs,
                output=args.output,
                tmp_dir=args.tmp_dir,
                uncertainty_file=args.uncertainty_file,
                gpu=gpu,
                cpu=args.cpu,
                quiet=args.quiet,
            )
        elif args.command == "pipeline":
            konfai_app.pipeline(
                inputs=args.inputs,
                gt=args.gt,
                mask=args.mask,
                output=args.output,
                tmp_dir=args.tmp_dir,
                prediction_file=args.prediction_file,
                evaluation_file=args.evaluation_file,
                uncertainty=getattr(args, "uncertainty", False),
                uncertainty_file=getattr(args, "uncertainty_file", "Uncertainty.yml"),
                gpu=gpu,
                cpu=args.cpu,
                quiet=args.quiet,
                patch_size=args.patch_size,
                max_voxels=getattr(args, "max_voxels", None),
                batch_size=args.batch_size,
                config_overrides=args.config_overrides,
                **infer_kwargs(args),
            )

    return main


def run_download_cli(kwargs: dict[str, Any]) -> None:
    """Download app files from Hugging Face into the local cache, with minimal console output."""
    from konfai.utils.runtime import MinimalLog

    from .app_repository import LocalAppRepositoryFromHF

    repository = get_app_repository_info(kwargs["app"], False)
    if not isinstance(repository, LocalAppRepositoryFromHF):
        raise SystemExit("'download' only applies to Hugging Face apps (expected 'repo_id:app_name').")

    force_update = not kwargs.get("no_force_update", False)
    filenames = kwargs.get("files") or repository.get_app_filenames()
    with MinimalLog():
        for filename in filenames:
            repository.download_files([filename], force_update)
            print(f"[KonfAI-Apps] {filename} is ready.")


@_exit_on_refusal()
def main_apps() -> None:
    """Entry point for the `konfai-apps` command-line interface."""
    parser = argparse.ArgumentParser(
        prog="konfai-apps", description="KonfAI Apps - Apps for Medical AI Models", allow_abbrev=False
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(parser: argparse.ArgumentParser, is_fine_tune: bool = False) -> None:
        parser.add_argument("app", type=str, help="KonfAI App name")
        parser.add_argument("--host", type=str, default=None, help="Server host")
        parser.add_argument("--port", type=int, default=8000, help="Server port")
        parser.add_argument(
            "--token",
            type=str,
            default=os.environ.get("KONFAI_API_TOKEN"),
            help="Bearer token (or use KONFAI_API_TOKEN env var)",
        )
        _add_app_io(parser, fine_tune=is_fine_tune)

    infer_p = subparsers.add_parser("infer", help="Run inference using a KonfAI App.")
    add_common_args(infer_p)
    group = infer_p.add_mutually_exclusive_group()
    group.add_argument("--ensemble", type=int, default=0, help="Number of models in the ensemble (auto-select).")
    group.add_argument(
        "--ensemble-models",
        "--ensemble_models",
        nargs="+",
        default=[],
        help="Explicit list of model identifiers/paths to use.",
    )
    infer_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations")
    infer_p.add_argument(
        "--mc", type=int, default=0, help="Monte Carlo dropout: not implemented, any value but 0 is refused."
    )
    _add_patch_overrides(infer_p)
    _add_config_overrides(infer_p)
    infer_p.add_argument("-uncertainty", action="store_true", help="If enabled, inference write the inference stack")
    infer_p.add_argument(
        "--prediction-file",
        "--prediction_file",
        type=str,
        default="Prediction.yml",
        help="Optional prediction config filename",
    )

    eval_p = subparsers.add_parser("eval", help="Evaluate a KonfAI App using ground-truth labels.")
    add_common_args(eval_p)
    _add_gt(eval_p, required=True)
    _add_mask(eval_p)
    eval_p.add_argument(
        "--evaluation-file",
        "--evaluation_file",
        type=str,
        default="Evaluation.yml",
        help="Optional evaluation config filename",
    )

    unc_p = subparsers.add_parser("uncertainty", help="Compute model uncertainty for a KonfAI App.")
    add_common_args(unc_p)
    unc_p.add_argument(
        "--uncertainty-file",
        "--uncertainty_file",
        type=str,
        default="Uncertainty.yml",
        help="Optional uncertainty config filename",
    )

    pipe_p = subparsers.add_parser(
        "pipeline", help="Run inference and optionally evaluation and uncertainty in a single command."
    )
    add_common_args(pipe_p)
    group = pipe_p.add_mutually_exclusive_group()
    group.add_argument("--ensemble", type=int, default=0, help="Number of models in the ensemble (auto-select).")
    group.add_argument(
        "--ensemble-models",
        "--ensemble_models",
        nargs="+",
        default=[],
        help="Explicit list of model identifiers/paths to use.",
    )
    pipe_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations.")
    pipe_p.add_argument(
        "--mc", type=int, default=0, help="Monte Carlo dropout: not implemented, any value but 0 is refused."
    )
    _add_patch_overrides(pipe_p)
    _add_config_overrides(pipe_p)
    pipe_p.add_argument(
        "--prediction-file",
        "--prediction_file",
        type=str,
        default="Prediction.yml",
        help="Optional prediction config filename",
    )
    _add_gt(pipe_p, required=False)
    _add_mask(pipe_p)
    pipe_p.add_argument(
        "--evaluation-file",
        "--evaluation_file",
        type=str,
        default="Evaluation.yml",
        help="Optional evaluation config filename",
    )
    pipe_p.add_argument(
        "--uncertainty-file",
        "--uncertainty_file",
        type=str,
        default="Uncertainty.yml",
        help="Optional uncertainty config filename",
    )
    pipe_p.add_argument("-uncertainty", action="store_true", help="Run uncertainty workflow.")

    ft_p = subparsers.add_parser("fine-tune", help="Fine-tune a KonfAI App on a dataset.")
    add_common_args(ft_p, True)
    ft_p.add_argument("name", type=str, help="New KonfAI App display name")
    ft_p.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Checkpoint name(s) to fine-tune, e.g. 'CV_0 CV_1'. If omitted, the first available "
        "checkpoint is fine-tuned. Each selected checkpoint is fine-tuned independently.",
    )
    ft_p.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    ft_p.add_argument(
        "--it-validation",
        "--it_validation",
        type=int,
        default=1000,
        help="Number of training iterations between validation runs.",
    )
    ft_p.add_argument(
        "--config",
        "--config-file",
        "--config_file",
        dest="config_file",
        type=str,
        default="Config.yml",
        help="Training configuration filename inside the app.",
    )
    ft_p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate. If omitted, the checkpoint learning rate is resumed and the "
        "scheduler continues; if set, the learning rate restarts from this value.",
    )
    ft_p.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override the training batch size (Trainer.Dataset.batch_size). Default: the app config's value.",
    )
    _add_config_overrides(ft_p)

    bundle_p = subparsers.add_parser(
        "bundle", help="Assemble an app bundle (HF layout), optionally with a portable ONNX model."
    )
    bundle_p.add_argument("name", type=str, help="Bundle/app variant name (the folder name).")
    bundle_p.add_argument("--out", required=True, help="Output directory; the bundle is written to <out>/<name>/.")
    bundle_p.add_argument("--app-json", required=True, help="Path to the app.json metadata file.")
    bundle_p.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="Config file(s), space-separated: Prediction.yml Evaluation.yml …",
    )
    bundle_p.add_argument("--checkpoint", nargs="+", required=True, help="Checkpoint .pt file(s): CV_0.pt CV_1.pt …")
    bundle_p.add_argument("--model-py", help="Optional custom Model.py to include.")
    bundle_p.add_argument(
        "--support-file",
        action="append",
        default=[],
        metavar="DESTINATION=SOURCE",
        help="Explicit helper file/package/asset directory mapping; paths are relative, directories copy recursively. Repeat for more paths.",
    )
    bundle_p.add_argument(
        "--support-root",
        help="Source root required with --support-file; imports are not recursively discovered.",
    )
    bundle_p.add_argument(
        "--requirements", help="Optional requirements.txt; if omitted, a draft is derived from Model.py imports."
    )
    bundle_p.add_argument(
        "--onnx", action="store_true", help="Also export model.onnx + manifest.json (portable runtime)."
    )
    bundle_p.add_argument(
        "--patch-size", type=int, nargs="+", help="Override --onnx patch size (else read from config)."
    )
    bundle_p.add_argument("--in-channels", type=int, help="Override --onnx input channels (else read from config).")
    bundle_p.add_argument("--output-module", help="Named head to export for --onnx (default: last graph output).")

    list_p = subparsers.add_parser("list", help="List the apps of a repository.")
    list_p.add_argument("repo", help="A Hugging Face repository ('repo_id[@revision]') or a local directory of apps.")
    list_p.add_argument("--task", default=None, help="Only the apps declaring this task.")
    show_p = subparsers.add_parser("show", help="Say what an app runs, what it needs and what --set tunes.")
    show_p.add_argument("app", help="'repo_id:app_name', or a local app folder.")

    download_p = subparsers.add_parser(
        "download", help="Download KonfAI App files from Hugging Face into the local cache."
    )
    download_p.add_argument("app", type=str, help="KonfAI App name, e.g. 'VBoussot/ImpactSynth:AppName'")
    download_p.add_argument(
        "files",
        nargs="*",
        help="File(s) to download, relative to the app folder. Downloads the whole app when omitted.",
    )
    download_p.add_argument(
        "--no-force-update",
        "--no_force_update",
        action="store_true",
        help="Reuse cached files when present instead of refreshing them from the Hub.",
    )

    parser.add_argument("--version", action="version", version=_package_version())

    kwargs = vars(parser.parse_args())

    if kwargs.get("command") == "bundle":
        from konfai_apps.bundle import run_bundle_cli

        run_bundle_cli(kwargs)
        return

    if kwargs.get("command") == "download":
        run_download_cli(kwargs)
        return

    if kwargs.get("command") == "list":
        print_apps("konfai-apps", list_apps(kwargs["repo"], kwargs["task"]))
        return

    if kwargs.get("command") == "show":
        _show(kwargs["app"])
        return

    host = kwargs.pop("host")
    port = kwargs.pop("port")
    token = kwargs.pop("token")

    konfai_app: AbstractKonfAIApp
    if host is not None:
        konfai_app = app_module.KonfAIAppClient(kwargs.pop("app"), RemoteServer(host, port, token))
    else:
        konfai_app = app_module.KonfAIApp(kwargs.pop("app"), kwargs.pop("download"), kwargs.pop("force_update"))

    command = kwargs.pop("command")
    if command == "infer":
        konfai_app.infer(**kwargs)
    elif command == "eval":
        konfai_app.evaluate(**kwargs)
    elif command == "uncertainty":
        konfai_app.uncertainty(**kwargs)
    elif command == "pipeline":
        konfai_app.pipeline(**kwargs)
    elif command == "fine-tune":
        kwargs["tmp_dir"] = kwargs["output"]
        konfai_app.fine_tune(**kwargs)


def _configure_server_auth_env(auth: str, token: str | None, token_env: str) -> None:
    """Resolve the bearer token into ``KONFAI_API_TOKEN`` (the variable the server actually reads).

    ``--auth off`` clears it so a leftover ``KONFAI_API_TOKEN`` cannot silently re-enable auth; bearer
    mode copies the token from ``token_env`` (or ``--token``) into ``KONFAI_API_TOKEN`` so a custom
    ``--token-env`` is honoured instead of silently disabling authentication.
    """
    if auth == "off":
        os.environ.pop("KONFAI_API_TOKEN", None)
        return
    if token:
        os.environ[token_env] = token
    resolved = os.environ.get(token_env)
    if not resolved:
        raise SystemExit(f"Auth is enabled but no token found. Set {token_env} or pass --token (dev).")
    os.environ["KONFAI_API_TOKEN"] = resolved


def main_apps_server() -> None:
    """Entry point for launching the KonfAI Apps FastAPI server."""
    try:
        import fastapi  # noqa: F401  # the server app imports it
        import uvicorn
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"konfai-apps-server needs the server extra ({error.name} is missing): pip install 'konfai-apps[server]'"
        ) from None

    parser = argparse.ArgumentParser(description="KonfAI apps server", allow_abbrev=False)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auth", choices=["off", "bearer"], default="bearer", help="Auth mode (default: bearer)")
    parser.add_argument(
        "--token-env", "--token_env", type=str, default="KONFAI_API_TOKEN", help="Env var name holding the bearer token"
    )
    parser.add_argument("--token", type=str, default=None, help="(dev) Bearer token override (NOT recommended in prod)")
    parser.add_argument("--apps", type=Path, required=True, help="Config file listing available apps (json).")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pre-download all apps listed in --apps into the local cache before starting the server.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all apps listed in --apps (no download) and exit; with --download, download and serve.",
    )

    args = parser.parse_args()

    _configure_server_auth_env(args.auth, args.token, args.token_env)

    if not args.apps.exists():
        raise SystemExit(f"Config file not found: {args.apps}")

    data = json.loads(args.apps.read_text(encoding="utf-8"))
    if "apps" not in data or not isinstance(data["apps"], list):
        raise SystemExit("Invalid config file: expected a JSON object with an 'apps' list.")

    os.environ["KONFAI_APPS_CONFIG"] = json.dumps(data)
    apps = []
    if args.check or args.download:
        errors = []
        for app_id in data["apps"]:
            try:
                apps.append(get_app_repository_info(str(app_id), True))
                print(f"[KonfAI-Apps] OK: {app_id}", flush=True)
            except Exception as exc:
                errors.append((app_id, str(exc)))
                print(f"[KonfAI-Apps] ERROR: {app_id} -> {exc}", flush=True)

        if errors:
            raise SystemExit("One or more apps are invalid:\n" + "\n".join(f"  - {a}: {err}" for a, err in errors))

        print("[KonfAI-Apps] All apps validated successfully.")
        if not args.download:
            return

    if args.download:
        for app in apps:
            try:
                if isinstance(app, LocalAppRepository):
                    _ = app.download_app()
                    print(f"[KonfAI-Apps] Cached: {app.get_name()}", flush=True)
            except Exception as exc:
                print(f"[KonfAI-Apps] Failed to cache '{app.get_name()}': {exc}", flush=True)

    uvicorn.run("konfai_apps.app_server:app", host=args.host, port=args.port, log_level="info", reload=False)
