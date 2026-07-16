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
import importlib.metadata
import os
import sys

from konfai import cuda_visible_devices
from konfai.utils.runtime import State

_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)


def _run(parser: argparse.ArgumentParser) -> None:
    """
    Shared CLI builder and dispatcher for the main KonfAI training/inference commands.

    This function:
    1) defines common arguments used by TRAIN / RESUME / PREDICTION / EVALUATION
       (config file, overwrite, device selection, quiet, tensorboard)
    2) defines subcommands and their command-specific arguments
    3) parses CLI args and dispatches to the correct implementation:
       - `konfai.trainer.train` for TRAIN and RESUME
       - `konfai.predictor.predict` for PREDICTION
       - `konfai.evaluator.evaluate` for EVALUATION

    Device selection
    ----------------
    GPU and CPU are mutually exclusive:
    - `--gpu` accepts one or more GPU ids, constrained to available devices
      returned by `cuda_visible_devices()`.
    - `--cpu` accepts a strictly positive integer (number of workers).

    Config handling
    ---------------
    If `--config` is omitted, the `config` key is removed from the argument dict,
    so downstream functions can use their own default config filename.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The top-level parser created by the caller.
    """

    def add_common_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            help="Path to the configuration file (YAML). "
            "If omitted, a command-specific default is used (e.g. Train.yml, Prediction.yml, Evaluation.yml).",
        )
        parser.add_argument(
            "-y",
            "--overwrite",
            action="store_true",
            help="Overwrite existing outputs (checkpoints, logs, predictions) without prompting.",
        )

        device_group = parser.add_mutually_exclusive_group()
        devices = cuda_visible_devices()
        device_group.add_argument(
            "--gpu",
            type=int,
            nargs="+",
            choices=devices,
            default=[],
            help="GPU device ids to use, e.g. '0' or '0,1,2'. If omitted runs on CPU.",
        )

        def positive_int(value: str) -> int:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError("CPU value must be > 0")
            return ivalue

        device_group.add_argument(
            "--cpu",
            type=positive_int,
            default=None,
            help="Run on CPU using N worker processes/cores. If omitted, uses GPU when available.",
        )

        parser.add_argument(
            "-q", "--quiet", action="store_true", help="Suppress console output for a quieter execution"
        )
        parser.add_argument("-tb", "--tensorboard", action="store_true", help="Launch TensorBoard.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser(str(State.TRAIN), help="Train a model from scratch.")
    add_common_args(train_p)
    train_p.add_argument(
        "--checkpoints-dir",
        "--checkpoints_dir",
        type=str,
        default="./Checkpoints/",
        help="Directory where checkpoints are saved (default: ./Checkpoints/).",
    )

    train_p.add_argument(
        "--statistics-dir",
        "--statistics_dir",
        type=str,
        default="./Statistics/",
        help="Directory where training statistics/logs are saved (default: ./Statistics/).",
    )

    resume_p = subparsers.add_parser(str(State.RESUME), help="Resume training from existing checkpoints.")
    add_common_args(resume_p)
    resume_p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint path to resume from",
    )

    resume_p.add_argument(
        "-checkpoints-dir",
        "-checkpoints_dir",
        type=str,
        default="./Checkpoints/",
        help="Directory where checkpoints are saved (default: ./Checkpoints/)",
    )

    resume_p.add_argument(
        "-statistics-dir",
        "-statistics_dir",
        type=str,
        default="./Statistics/",
        help="Directory where training statistics/logs are saved (default: ./Statistics/).",
    )

    resume_p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate on resume. If omitted, the checkpoint learning rate is "
        "resumed and the scheduler continues; if set, the learning rate restarts from this value.",
    )

    predict_p = subparsers.add_parser(str(State.PREDICTION), help="Run inference using a trained model.")
    add_common_args(predict_p)

    predict_p.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="PATH",
        required=True,
        help="One or more checkpoint/model paths to resume from.",
    )

    predict_p.add_argument(
        "--predictions-dir",
        "--predictions_dir",
        type=str,
        default="./Predictions/",
        help="Directory where predictions are written (default: ./Predictions/).",
    )

    eval_p = subparsers.add_parser(str(State.EVALUATION), help="Evaluate model.")
    add_common_args(eval_p)

    eval_p.add_argument(
        "--evaluations-dir",
        "--evaluations_dir",
        type=str,
        default="./Evaluations/",
        help="Directory where evaluation outputs are written (default: ./Evaluations/).",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=importlib.metadata.version("konfai"),
        help="Print KonfAI version and exit.",
    )

    args = vars(parser.parse_args())

    if args["command"] == "PREDICTION":
        from konfai.predictor import predict

        if args["config"] is not None:
            args["prediction_file"] = args.pop("config")
        predict(**args)
    elif args["command"] == "EVALUATION":
        from konfai.evaluator import evaluate

        if args["config"] is not None:
            args["evaluations_file"] = args.pop("config")

        evaluate(**args)
    else:
        from konfai.trainer import train

        if args["config"] is None:
            del args["config"]
        train(**args)


def main():
    """
    Entry point for the ``konfai`` command-line interface.

    This function builds the top-level CLI parser and delegates the full argument
    parsing and command dispatching to `_run(parser)`.

    Supported commands are:
    - TRAIN
    - RESUME
    - PREDICTION
    - EVALUATION

    Notes
    -----
    The actual execution logic is implemented in `konfai.trainer.train`,
    `konfai.predictor.predict`, and `konfai.evaluator.evaluate`.
    """
    parser = argparse.ArgumentParser(
        prog="konfAI", description="KonfAI - Deep learning framework for Medical AI Models", allow_abbrev=False
    )
    _run(parser)


def cluster():
    """
    Entry point for running KonfAI with cluster-oriented CLI arguments.

    This command extends the standard KonfAI CLI with a "Cluster manager arguments"
    group (job name, nodes, memory, time limit, resubmit), then delegates parsing
    and command dispatching to `_run(parser)`.

    Notes
    -----
    - This function only defines extra CLI arguments before delegating to
      ``_run``.
    """
    parser = argparse.ArgumentParser(
        prog="konfAI", description="KonfAI - Deep learning framework for Medical AI Models", allow_abbrev=False
    )

    # Cluster manager arguments
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
    cluster_args.add_argument(
        "--resubmit",
        action="store_true",
        help="Automatically resubmit job just before timout",
    )
    _run(parser)
