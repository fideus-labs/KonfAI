#!/usr/bin/env python3
#
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

"""Command-line entrypoints for KonfAI workflows, apps, and services."""

import argparse
import importlib
import importlib.metadata
import os
import sys
from typing import Any

from konfai.utils import State

_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("CPU value must be > 0")
    return ivalue


class _VersionAction(argparse.Action):
    """``--version``: the installed version, looked up when asked for. ``importlib.metadata.version``
    scans the installed distributions (17 ms cold), which every other invocation would pay at
    parser construction."""

    def __init__(self, option_strings: list[str], dest: str, help: str | None = None) -> None:
        super().__init__(option_strings, dest, default=argparse.SUPPRESS, nargs=0, help=help)

    def __call__(self, parser: argparse.ArgumentParser, namespace: Any, values: Any, option_string=None) -> None:
        print(importlib.metadata.version("konfai"))
        parser.exit()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """The arguments TRAIN / RESUME / PREDICTION / EVALUATION share; TRANSFORM declares its own set."""
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (YAML). "
        "If omitted, a command-specific default is used: Config.yml, Prediction.yml, Evaluation.yml.",
    )
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (checkpoints, logs, predictions) without prompting.",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[],
        help="GPU device ids to use, space separated: --gpu 0, or --gpu 0 1 2. If omitted runs on CPU.",
    )
    device_group.add_argument(
        "--cpu",
        type=_positive_int,
        default=None,
        help="Number of CPU worker processes when no --gpu is given; the run stays on CPU unless --gpu is passed.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output for a quieter execution")
    parser.add_argument("-tb", "--tensorboard", action="store_true", help="Launch TensorBoard.")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create the config file if missing, resolve every default into it, and exit without running.",
    )


def _add_dir_argument(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    """A workspace-directory flag in the house style: ``--<name>-dir``/``--<name>_dir``, default ``./<Name>/``."""
    default = f"./{name.capitalize()}/"
    parser.add_argument(
        f"--{name}-dir",
        f"--{name}_dir",
        type=str,
        default=default,
        help=f"{help_text} (default: {default}).",
    )


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(str(State.TRAIN), help="Train a model from scratch.")
    _add_common_args(parser)
    _add_dir_argument(parser, "checkpoints", "Directory where checkpoints are saved")
    _add_dir_argument(parser, "statistics", "Directory where training statistics/logs are saved")


def _add_resume(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(str(State.RESUME), help="Resume training from existing checkpoints.")
    _add_common_args(parser)
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path to resume from")
    _add_dir_argument(parser, "checkpoints", "Directory where checkpoints are saved")
    _add_dir_argument(parser, "statistics", "Directory where training statistics/logs are saved")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate on resume. If omitted, the checkpoint learning rate is "
        "resumed and the scheduler continues; if set, the learning rate restarts from this value.",
    )


def _add_predict(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(str(State.PREDICTION), help="Run inference using a trained model.")
    _add_common_args(parser)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="PATH",
        required=True,
        help="One or more checkpoint/model paths to resume from.",
    )
    _add_dir_argument(parser, "predictions", "Directory where predictions are written")


def _add_evaluate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(str(State.EVALUATION), help="Evaluate model.")
    _add_common_args(parser)
    _add_dir_argument(parser, "evaluations", "Directory where evaluation outputs are written")


def _add_transform(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        str(State.TRANSFORM), help="Prepare a dataset: apply a transform chain to every case and write the result."
    )
    # Deliberately NOT _add_common_args: -tb has no scalar to show, so refusing it here is a parse
    # error, not a silent no-op.
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (YAML). If omitted, Transform.yml is used.",
    )
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="Rewrite outputs that already exist. Without it, a case whose output exists is skipped.",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[],
        help="GPU device ids the chain runs on: every stage, the streamed replays and the folds "
        "included, executes on the rank's device. If omitted the run stays on CPU.",
    )
    device_group.add_argument(
        "--cpu",
        type=_positive_int,
        default=None,
        help="Shard the cases over N worker processes (default: 1).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output for a quieter execution")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the per-case streaming plan and exit without transforming. The plan probes each"
        " destination with a real region-write open, so it reports the run's actual verdict, and"
        " takes back what the probe created (the entry, and the store when it did not exist). The"
        " plan is printed even with -q.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create the config file if missing, resolve every default into it, and exit without running.",
    )
    _add_dir_argument(
        parser, "transforms", "Directory where run logs are written; --plan prints and writes nothing there"
    )


#: The component families `konfai list` prints, spelled as the CLI takes them.
_LIST_KINDS = ("transforms", "augmentations", "criteria", "reductions", "models", "blocks")


def _add_list(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "list", help="List the components a YAML config can reference (name and one-line doc)."
    )
    parser.add_argument("kind", choices=_LIST_KINDS, help="Component family to list.")


def _run_list(kind: str) -> None:
    # Lazy: the catalog imports the component families (torch included), which --help must not pay for.
    from konfai.utils.catalog import list_components

    components = list_components(kind)
    width = max((len(component.config_reference) for component in components), default=0)
    for component in components:
        print(f"{component.config_reference:<{width}}  {component.doc or ''}".rstrip())


# Command -> (implementation module, entrypoint, the kwarg the config path travels under).
# Imports stay lazy and by name: the heavy modules load only for the command that runs.
_COMMANDS: dict[str, tuple[str, str, str]] = {
    str(State.TRAIN): ("konfai.trainer", "train", "config"),
    str(State.RESUME): ("konfai.trainer", "train", "config"),
    str(State.PREDICTION): ("konfai.predictor", "predict", "prediction_file"),
    str(State.EVALUATION): ("konfai.evaluator", "evaluate", "evaluations_file"),
    str(State.TRANSFORM): ("konfai.transformer", "transform", "transform_file"),
}

# Command -> (default config filename, root key, pure build function beside the entrypoint).
_INIT_TARGETS: dict[str, tuple[str, str, str]] = {
    str(State.TRAIN): ("Config.yml", "Trainer", "build_train"),
    str(State.RESUME): ("Config.yml", "Trainer", "build_train"),
    str(State.PREDICTION): ("Prediction.yml", "Predictor", "build_predict"),
    str(State.EVALUATION): ("Evaluation.yml", "Evaluator", "build_evaluate"),
    str(State.TRANSFORM): ("Transform.yml", "Transformer", "build_transform"),
}


def _run_init(args: dict[str, Any]) -> None:
    """``--init``: bind the workflow once so every default resolves and lands in the file, then exit.

    The file is created seeded with its root key when missing (an empty tree would be refused as
    holding no root). The build itself never runs anything; a build error after partial binding
    still leaves what resolved on disk (the strict block flushes on exceptional exit too).
    """
    import inspect
    from pathlib import Path

    command = args["command"]
    module_name, _, config_key = _COMMANDS[command]
    default_name, root, builder_name = _INIT_TARGETS[command]
    config_path = Path(args.get(config_key) or args.get("config") or default_name)
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(f"{root}: {{}}\n", encoding="utf-8")
    args[config_key] = config_path
    builder = getattr(importlib.import_module(module_name), builder_name)
    accepted = inspect.signature(builder).parameters
    try:
        builder(**{name: value for name, value in args.items() if name in accepted})
    except Exception as error:
        print(f"[KonfAI] Wrote what resolved before the error to '{config_path}'.")
        print(f"[KonfAI] {error}")
        sys.exit(1)
    print(f"[KonfAI] Resolved default configuration written to '{config_path}'.")


def _check_gpu_ids(parser: argparse.ArgumentParser, gpu: list[int]) -> None:
    """The ``--gpu`` choices, checked after parsing: resolving them imports torch, which
    ``--help`` and a usage error must not pay for."""
    if not gpu:
        return
    from konfai import cuda_visible_devices

    visible = cuda_visible_devices()
    unknown = [device for device in gpu if device not in visible]
    if unknown:
        choices = ", ".join(str(device) for device in visible) or "no GPU visible"
        parser.error(f"argument --gpu: invalid choice: {unknown[0]} (choose from {choices})")


def _dispatch(parser: argparse.ArgumentParser, args: dict[str, Any]) -> None:
    if args["command"] == "list":
        # Before the workflow machinery: `list` declares only its kind, none of the run flags.
        _run_list(args["kind"])
        return
    if args["command"] not in _COMMANDS:
        # Exhaustive on purpose: a fallback would silently launch the trainer for any command it
        # does not know: a new workflow would train a UNet instead of failing.
        parser.error(f"Unknown command '{args['command']}'.")
    _check_gpu_ids(parser, args["gpu"])
    module_name, function_name, config_key = _COMMANDS[args["command"]]
    if args["config"] is None:
        del args["config"]  # the entrypoint's own default config filename applies
    elif config_key != "config":
        args[config_key] = args.pop("config")
    if args.pop("init", False):
        # --init must SHORT-CIRCUIT for the same reason --plan does just below.
        _run_init(args)
        return
    if args.pop("plan", False):
        # --plan must SHORT-CIRCUIT here: the distributed wrapper filters kwargs by the entrypoint's
        # signature, so a 'plan' passed through would be silently dropped and the run would proceed
        # as if the flag had never been given.
        if "num_nodes" in args:
            parser.error("--plan is a dry run on this machine and submits nothing: use `konfai TRANSFORM --plan`.")
        # plan_transform declares the TRANSFORM flags and nothing else; the command name is not one.
        del args["command"]
        function_name = "plan_transform"
    entrypoint = getattr(importlib.import_module(module_name), function_name)
    entrypoint(**args)


def _run(parser: argparse.ArgumentParser) -> None:
    """Declare the five workflow subcommands on ``parser``, then parse and dispatch."""
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train(subparsers)
    _add_resume(subparsers)
    _add_predict(subparsers)
    _add_evaluate(subparsers)
    _add_transform(subparsers)
    _add_list(subparsers)
    parser.add_argument("--version", action=_VersionAction, help="Print KonfAI version and exit.")
    _dispatch(parser, vars(parser.parse_args()))


def main():
    """Entry point for the ``konfai`` command-line interface."""
    parser = argparse.ArgumentParser(
        prog="konfai", description="KonfAI - Deep learning framework for Medical AI Models", allow_abbrev=False
    )
    _run(parser)


def cluster():
    """Entry point for the ``konfai-cluster`` CLI: the standard commands plus SLURM job arguments."""
    parser = argparse.ArgumentParser(
        prog="konfai-cluster", description="KonfAI - Deep learning framework for Medical AI Models", allow_abbrev=False
    )
    cluster_args = parser.add_argument_group("Cluster manager arguments")
    cluster_args.add_argument("--name", type=str, help="Task name", required=True)
    cluster_args.add_argument("--num-nodes", "--num_nodes", default=1, type=int, help="Number of nodes")
    cluster_args.add_argument("--memory", type=int, default=16, help="Amount of memory per node")
    cluster_args.add_argument(
        "--time-limit",
        "--time_limit",
        type=int,
        default=1440,
        help="Job time limit in minute",
    )
    _run(parser)


if __name__ == "__main__":
    main()
