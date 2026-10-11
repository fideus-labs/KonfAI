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

"""Command-line wrapper for the baseline apps: ``baselines-konfai register <baseline> -i fixed -i moving -o out``.

The baselines live on Hugging Face (``fideus-labs/Baselines``); ``BASELINES_KONFAI_REPO`` points at a local
directory of app folders instead.
"""

import argparse
import os
from pathlib import Path

from konfai_apps.cli import build_app_cli

BASELINES_KONFAI_REPO = os.environ.get("BASELINES_KONFAI_REPO", "fideus-labs/Baselines")


def resolve_app(args: argparse.Namespace) -> str:
    """The app's id: a folder of a local directory of apps, or ``repo:app`` on Hugging Face."""
    if Path(BASELINES_KONFAI_REPO).is_dir():
        return str(Path(BASELINES_KONFAI_REPO) / args.baseline)
    return f"{BASELINES_KONFAI_REPO}:{args.baseline}"


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("baseline", help="Which baseline to run (an app of the repository, e.g. ConvexAdam_MIND).")


main = build_app_cli(
    "baselines-konfai",
    "The registration baselines of ImagingBenchmarks: the authors' code at a pinned version, through KonfAI.",
    resolve_app=resolve_app,
    add_selection=_add_selection,
    resolve_infer=lambda args: {"ensemble": 0, "tta": 0, "mc": 0},
    infer_command="register",
    with_uncertainty=False,
)

if __name__ == "__main__":
    main()
