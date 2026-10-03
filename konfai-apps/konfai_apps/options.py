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


"""The command-line options every app CLI shares; argparse only, so that ``--help`` imports nothing heavy."""

import argparse
from collections.abc import Callable
from pathlib import Path


def local_path(value: str) -> Path:
    """An argparse type: a local path, made absolute. A URI is refused: every input is staged from the local
    filesystem."""
    if "://" in value:
        raise argparse.ArgumentTypeError(f"{value!r} is a URI; inputs are read from local paths, so copy it first.")
    return Path(value).resolve()


def at_least(minimum: int) -> Callable[[str], int]:
    """An argparse type: an integer no smaller than ``minimum``, refused at parse time rather than in a child."""

    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            number = minimum - 1
        if number < minimum:
            raise argparse.ArgumentTypeError(f"expected an integer >= {minimum}, got {value!r}")
        return number

    return parse


def add_tmp_dir(parser: argparse.ArgumentParser, help: str = "Temporary directory (optional).") -> None:
    """Add ``--tmp-dir``."""
    parser.add_argument("--tmp-dir", "--tmp_dir", dest="tmp_dir", type=local_path, default=None, help=help)


def add_device(parser: argparse.ArgumentParser, download: bool = True) -> None:
    """Add the device (``--gpu`` / ``--cpu``), ``--quiet`` and, with ``download``, the app-download options."""
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--gpu",
        type=at_least(0),
        nargs="+",
        default=[],
        help="GPU id(s), e.g. '0'; the run's CUDA_VISIBLE_DEVICES becomes them, replacing any mask already set. "
        "CPU if omitted.",
    )
    device.add_argument("--cpu", type=at_least(1), default=None, help="Run on CPU using N worker processes.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output.")
    if download:
        parser.add_argument("--download", action="store_true", help="Download the full app(s) upfront.")
        parser.add_argument(
            "--force-update",
            "--force_update",
            dest="force_update",
            action="store_true",
            help="Refresh required app files before running.",
        )
