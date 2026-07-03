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

"""Command-line wrapper for running IMPACT-Seg through KonfAI Apps.

Exposes the app operations with segmentation vocabulary: ``segment`` / ``eval`` / ``uncertainty`` / ``pipeline``.
"""

import argparse

from konfai_apps.cli import build_app_cli

IMPACT_SEG_KONFAI_REPO = "VBoussot/ImpactSeg"


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        help="Which published IMPACT-Seg model to use. An unknown name is reported when the app is resolved.",
    )


def _add_infer_knobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ensemble", type=int, default=0, help="Size of model ensemble.")
    parser.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations.")
    parser.add_argument("--mc", type=int, default=0, help="Monte Carlo dropout samples.")


main = build_app_cli(
    "impact-seg-konfai",
    "IMPACT-Seg (KonfAI app wrapper): multimodal segmentation.",
    resolve_app=lambda args: f"{IMPACT_SEG_KONFAI_REPO}:{args.model}",
    add_selection=_add_selection,
    add_infer_knobs=_add_infer_knobs,
    resolve_infer=lambda args: {"ensemble": args.ensemble, "ensemble_models": [], "tta": args.tta, "mc": args.mc},
    infer_command="segment",
)


if __name__ == "__main__":
    main()
