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

"""Command-line wrapper for running MRSegmentator through KonfAI Apps.

Exposes the app operations with segmentation vocabulary: ``segment`` / ``eval`` / ``uncertainty`` / ``pipeline``.
"""

import argparse

from konfai_apps.cli import build_app_cli

MR_SEGMENTATOR_KONFAI_REPO = "VBoussot/MRSegmentator-KonfAI"


def _add_infer_knobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--folds",
        choices=[1, 2, 3, 4, 5],
        default=2,
        type=int,
        help="Number of cross-validation folds to ensemble.",
    )


main = build_app_cli(
    "mrsegmentator-konfai",
    "MRSegmentator (KonfAI app wrapper): multi-organ MR segmentation.",
    resolve_app=lambda args: f"{MR_SEGMENTATOR_KONFAI_REPO}:MRSegmentator",
    add_infer_knobs=_add_infer_knobs,
    resolve_infer=lambda args: {"ensemble": args.folds, "ensemble_models": [], "tta": 0, "mc": 0},
    infer_command="segment",
)


if __name__ == "__main__":
    main()
