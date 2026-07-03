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

"""Command-line wrapper for running TotalSegmentator through KonfAI Apps.

Exposes the app operations with segmentation vocabulary: ``segment`` / ``eval`` / ``pipeline``
(TotalSegmentator models do not provide an uncertainty workflow).
"""

import argparse

from konfai_apps.cli import build_app_cli

TOTAL_SEGMENTATOR_KONFAI_REPO = "VBoussot/TotalSegmentator-KonfAI"


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "task",
        help="Which anatomical task/model to use (determines what is predicted). "
        "An unknown name is reported when the app is resolved.",
    )


def _add_infer_knobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", default=[], help="Explicit list of model identifiers/paths to ensemble.")


main = build_app_cli(
    "totalsegmentator-konfai",
    "TotalSegmentator (KonfAI app wrapper): whole-body CT segmentation.",
    resolve_app=lambda args: f"{TOTAL_SEGMENTATOR_KONFAI_REPO}:{args.task}",
    add_selection=_add_selection,
    add_infer_knobs=_add_infer_knobs,
    resolve_infer=lambda args: {"ensemble": 0, "ensemble_models": args.models, "tta": 0, "mc": 0},
    infer_command="segment",
    with_uncertainty=False,
)


if __name__ == "__main__":
    main()
