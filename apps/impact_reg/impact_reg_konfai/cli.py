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

"""Command-line orchestrator for IMPACT-Reg registration presets running as KonfAI Apps.

Three composable sub-commands, mirroring ``konfai-apps`` (infer/eval/uncertainty):

- ``register``    : run one or more preset apps on a fixed/moving pair and ensemble their DVFs;
- ``eval``        : evaluate a registration on any subset of modalities (image/seg/fid);
- ``uncertainty`` : voxel-wise spread map from an ensemble of displacement fields.
"""

import argparse
from pathlib import Path

from impact_reg_konfai.impact_reg import ImpactRegKonfAIApp, get_available_presets


def _paths(value: str) -> Path:
    return Path(value).resolve()


def _default_preset() -> str:
    """First available preset, or an actionable error when none resolve (used when --preset is omitted)."""
    presets = get_available_presets()
    if not presets:
        raise SystemExit(
            "No registration preset resolved. Pass --preset, set KONFAI_IMPACTREG_REPO to a directory of "
            "preset folders, or check network/Hugging Face access."
        )
    return presets[0]


def _add_device(parser: argparse.ArgumentParser, download: bool = True) -> None:
    """Add the shared device / verbosity / download options to a sub-parser."""
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", type=int, nargs="+", default=[], help="GPU device ids (e.g. '0'). CPU if omitted.")
    device.add_argument("--cpu", type=int, default=None, help="Run on CPU using N worker processes.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output.")
    if download:
        parser.add_argument("--download", action="store_true", help="Download the full preset app(s) upfront.")
        parser.add_argument("--force_update", action="store_true", help="Refresh required app files before running.")


def main() -> None:
    """Parse CLI arguments and run the requested IMPACT-Reg operation."""
    parser = argparse.ArgumentParser(
        prog="impact-reg-konfai",
        description="IMPACT-Reg (KonfAI app orchestrator): model-based registration presets with ensembling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register ---------------------------------------------------------------
    reg = subparsers.add_parser(
        "register",
        help="Register a fixed/moving pair with one or more presets (several presets are ensembled).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reg.add_argument(
        "presets",
        nargs="+",
        help="One or more preset apps; several presets are ensembled (their DVFs are averaged). "
        "Any invalid name is reported when the preset app is resolved.",
    )
    reg.add_argument("-f", "--fixed-images", type=_paths, nargs="+", required=True, help="Fixed image(s).")
    reg.add_argument("-m", "--moving-images", type=_paths, nargs="+", required=True, help="Moving image(s).")
    reg.add_argument(
        "--fixed-mask",
        type=_paths,
        nargs="+",
        default=[],
        help="Optional fixed mask(s) restricting the metric region (whole-image mask auto-filled if omitted).",
    )
    reg.add_argument("--moving-mask", type=_paths, nargs="+", default=[], help="Optional moving mask(s).")
    reg.add_argument("-o", "--output", type=_paths, default=Path("./Output").resolve(), help="Output directory.")
    reg.add_argument(
        "--tta",
        type=int,
        default=0,
        help="Number of test-time augmentations per preset (flipped registrations averaged by each preset app).",
    )
    reg.add_argument(
        "--uncertainty",
        action="store_true",
        help="Keep each preset's displacement field (under Ensemble/) so the ensemble spread can be "
        "measured afterwards by 'uncertainty'.",
    )
    _add_device(reg)

    # eval -------------------------------------------------------------------
    ev = subparsers.add_parser(
        "eval",
        help="Evaluate a registration on any subset of modalities (image/seg/fid); at least one is required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ev.add_argument(
        "--preset",
        default=None,
        help="Preset providing the evaluation configs (default: first available).",
    )
    ev.add_argument("-f", "--fixed-images", type=_paths, nargs="+", default=[], help="Fixed image(s) [image modality].")
    ev.add_argument(
        "-m", "--moving-images", type=_paths, nargs="+", default=[], help="Moving image(s) [image modality]."
    )
    ev.add_argument(
        "--transform",
        type=_paths,
        nargs="+",
        default=[],
        help="Transform(s) warping moving onto fixed (identity if omitted).",
    )
    ev.add_argument("--gt-fixed-seg", type=_paths, nargs="+", default=[], help="Fixed segmentation(s) [seg modality].")
    ev.add_argument(
        "--gt-moving-seg", type=_paths, nargs="+", default=[], help="Moving segmentation(s) [seg modality]."
    )
    ev.add_argument("--gt-fixed-fid", type=_paths, nargs="+", default=[], help="Fixed landmark file(s) [fid modality].")
    ev.add_argument(
        "--gt-moving-fid", type=_paths, nargs="+", default=[], help="Moving landmark file(s) [fid modality]."
    )
    ev.add_argument(
        "--mask", type=_paths, nargs="+", default=None, help="Optional evaluation mask(s) [image modality]."
    )
    ev.add_argument("-o", "--output", type=_paths, default=Path("./Output").resolve(), help="Output directory.")
    _add_device(ev)

    # uncertainty ------------------------------------------------------------
    unc = subparsers.add_parser(
        "uncertainty",
        help="Voxel-wise spread map from an ensemble of displacement fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    unc.add_argument(
        "--preset",
        default=None,
        help="Preset providing the uncertainty config (default: first available).",
    )
    unc.add_argument(
        "--dvf",
        type=_paths,
        nargs="+",
        required=True,
        help="Two or more ensemble displacement fields (e.g. the per-preset DVFs written by 'register').",
    )
    unc.add_argument("-o", "--output", type=_paths, default=Path("./Output").resolve(), help="Output directory.")
    _add_device(unc)

    args = parser.parse_args()
    app = ImpactRegKonfAIApp(
        download=getattr(args, "download", False), force_update=getattr(args, "force_update", False)
    )

    if args.command == "register":
        gpu = [] if args.cpu is not None else args.gpu
        app.register(
            args.presets,
            args.fixed_images,
            args.moving_images,
            fixed_masks=args.fixed_mask,
            moving_masks=args.moving_mask,
            output=args.output,
            gpu=gpu,
            cpu=args.cpu,
            quiet=args.quiet,
            tta=args.tta,
            keep_dvf=args.uncertainty,
        )

    elif args.command == "eval":
        # Nothing is mandatory except at least one complete modality (image / seg / fid).
        has_image = bool(args.fixed_images and args.moving_images)
        has_seg = bool(args.gt_fixed_seg and args.gt_moving_seg)
        has_fid = bool(args.gt_fixed_fid and args.gt_moving_fid)
        if not (has_image or has_seg or has_fid):
            ev.error(
                "provide at least one modality: image (-f/-m), seg (--gt-fixed-seg/--gt-moving-seg), "
                "or fid (--gt-fixed-fid/--gt-moving-fid)."
            )
        gpu = [] if args.cpu is not None else args.gpu
        app.evaluate(
            preset=args.preset or _default_preset(),
            fixed_images=args.fixed_images,
            moving_images=args.moving_images,
            transforms=args.transform,
            gt_fixed_seg=args.gt_fixed_seg,
            gt_moving_seg=args.gt_moving_seg,
            gt_fixed_fid=args.gt_fixed_fid,
            gt_moving_fid=args.gt_moving_fid,
            mask=args.mask,
            output=args.output,
            gpu=gpu,
            cpu=args.cpu,
            quiet=args.quiet,
        )

    elif args.command == "uncertainty":
        gpu = [] if args.cpu is not None else args.gpu
        app.uncertainty(
            preset=args.preset or _default_preset(),
            dvfs=args.dvf,
            output=args.output,
            gpu=gpu,
            cpu=args.cpu,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
