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

Six sub-commands; register, eval and uncertainty are composable and mirror ``konfai-apps``
(infer/eval/uncertainty):

- ``list``        : list the registration presets;
- ``show``        : say what one preset runs, needs and tunes;
- ``register``    : run one or more preset apps on a fixed/moving pair and ensemble their DVFs;
- ``eval``        : evaluate a registration on any subset of modalities (image/seg/fid);
- ``uncertainty`` : voxel-wise spread map from an ensemble of displacement fields;
- ``apply``       : warp more moving-side images onto the fixed grid through a transform.
"""

import argparse
import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

# argparse only: --help and --version stay free of torch and KonfAI.
from konfai_apps.options import add_device, add_tmp_dir, at_least, local_path

from impact_reg_konfai import PRESETS_REPO


def _stop_on_sigterm() -> None:
    """End the run on SIGTERM the way Ctrl+C does, stopping what it started first.

    SIGTERM is how SlicerImpactReg's Stop ends the command. The children are asked to stop first (konfai-apps
    turns SIGTERM into its own cleanup) and given one second, then the exit unwinds this process through its
    ``finally``, which removes the work directory; ``subprocess.run`` kills a child still running on the way. One
    second: Slicer kills this process 2 s after its SIGTERM.
    """

    def stop(signum: int, _frame: object) -> None:
        import psutil

        children = psutil.Process().children(recursive=True)
        for child in children:
            with contextlib.suppress(psutil.Error):
                child.terminate()
        psutil.wait_procs(children, timeout=1)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)


def _version() -> str:
    """What ``--version`` prints: this package, the KonfAI stack it runs on, and where its presets come from."""
    from importlib.metadata import PackageNotFoundError, version

    def of(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    stack = ", ".join(f"{name} {of(name)}" for name in ("konfai", "konfai-apps", "torch"))
    return f"impact-reg-konfai {of('impact-reg-konfai')} ({stack}); presets: {PRESETS_REPO}"


def main() -> None:
    """Parse CLI arguments and run the requested IMPACT-Reg operation."""
    parser = argparse.ArgumentParser(
        prog="impact-reg-konfai",
        description="""IMPACT-Reg: register a moving image onto a fixed one with registration presets (KonfAI apps
on Hugging Face), ensemble several, evaluate a transform, and measure the spread of an ensemble.""",
        epilog=f"""examples:
  impact-reg-konfai list
  impact-reg-konfai show FireANTs_SyN
  impact-reg-konfai register FireANTs_SyN -f fixed.nii.gz -m moving.nii.gz -o Output --gpu 0

The presets come from {PRESETS_REPO}. KONFAI_IMPACTREG_REPO points elsewhere: a directory
of preset folders, or <repo>@<revision>. IMPACT_REG_DEBUG=1 prints the traceback of a failure.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_version())
    subparsers = parser.add_subparsers(dest="command", required=True)
    tmp_dir_help = (
        "Directory for the volume-sized intermediates (a hidden directory beside --output when unset, on the results' "
        "own filesystem rather than a tmpfs system temporary directory)."
    )

    # list / show ------------------------------------------------------------
    subparsers.add_parser("list", help="List the registration presets.")
    show = subparsers.add_parser(
        "show",
        help="Say what a preset runs, what it needs and what --set tunes.",
        epilog="example:\n  impact-reg-konfai show FireANTs_SyN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    show.add_argument("name", help="The preset.")

    # register ---------------------------------------------------------------
    reg = subparsers.add_parser(
        "register",
        help="Register a fixed/moving pair with one or more presets (several presets are ensembled).",
        description="""Register each moving image onto its fixed image with one or more presets; several presets
are ensembled, their displacement fields averaged. Each case, P000, P001, ... in input order, gets:

  <output>/<case>/Transform.h5            the transform, a displacement field on the fixed grid
  <output>/<case>/Moved.<moving's ext>    the moving image resampled through it (not with --fields-only)
  <output>/<case>/Ensemble/<preset>.h5    each preset's own field, with --keep-fields

The transform maps a point of the fixed image to the matching point of the moving image (ITK's
convention): resampling the moving image through it gives Moved.""",
        epilog="""examples:
  impact-reg-konfai register FireANTs_SyN -f fixed.nii.gz -m moving.nii.gz -o Output --gpu 0
  impact-reg-konfai register -f fixed.ome.zarr -m moving.ome.zarr -p ConvexAdam_Composite FireANTs_SyN \\
      --keep-fields -o Output --gpu 0""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    reg.add_argument(
        "presets",
        nargs="*",
        metavar="PRESET",
        help="One or more preset apps ('impact-reg-konfai list' lists them); several presets are ensembled "
        "(their DVFs are averaged). An unknown name is reported before anything runs.",
    )
    # The same, as an option: a name written after -f/-m is read as one more image, so the positional form only
    # works with the presets first.
    reg.add_argument(
        "-p",
        "--preset",
        dest="preset_options",
        nargs="+",
        action="extend",
        default=[],
        metavar="PRESET",
        help="Preset(s) to run, anywhere on the line; added to the positional ones.",
    )
    reg.add_argument("-f", "--fixed-images", type=local_path, nargs="+", required=True, help="Fixed image(s).")
    reg.add_argument("-m", "--moving-images", type=local_path, nargs="+", required=True, help="Moving image(s).")
    reg.add_argument(
        "--fixed-mask",
        type=local_path,
        nargs="+",
        default=[],
        help="Optional fixed mask(s) restricting the metric region (whole-image mask auto-filled if omitted).",
    )
    reg.add_argument("--moving-mask", type=local_path, nargs="+", default=[], help="Optional moving mask(s).")
    reg.add_argument(
        "-o",
        "--output",
        type=local_path,
        default=Path("./Output").resolve(),
        help="Output directory (default: ./Output).",
    )
    reg.add_argument(
        "--tta",
        type=at_least(0),
        default=0,
        help="Number of test-time augmentations per preset (flipped registrations averaged by each preset app).",
    )
    reg.add_argument(
        "--keep-fields",
        dest="keep_fields",
        action="store_true",
        help="Keep each preset's displacement field as <output>/<case>/Ensemble/<preset>.<ext>, for 'uncertainty "
        "--dvf' to measure the spread afterwards; runs into the same --output add their members beside the earlier "
        "ones.",
    )
    reg.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        metavar="[PRESET:][global:|tile:]NAME=VALUE",
        default=None,
        help="Tune a preset parameter (repeatable), checked against each preset before anything runs. NAME=VALUE "
        "applies to every preset, PRESET:NAME=VALUE to that one alone, since engines name their knobs differently: "
        "e.g. --set FireANTs_SyN:deformable_iterations=[400,200,100] --set Generic_Rigid:max_iterations=500. "
        "'impact-reg-konfai show NAME' lists the parameters a preset takes. A pair registered in tiles gets it "
        "in the preset's global pass and in its tiles where their configs agree, in the global pass alone where "
        "the tile config differs (its deformable stage alone); global: or tile: gives it to that pass alone, "
        "e.g. --set FireANTs_SyN:tile:deformable_iterations=[100,50,25].",
    )
    reg.add_argument(
        "--max-voxels",
        "--max_voxels",
        dest="max_voxels",
        type=at_least(1),
        default=None,
        help="Register whole a pair of at most this many voxels, overriding what the device holds when the run "
        "starts at the costs the preset declares. A larger pair runs whole on a coarser grid, resampled by KonfAI, "
        "and a preset that declares a tile pass then refines it on native tiles sized in proportion.",
    )
    reg.add_argument(
        "--fields-only",
        "--fields_only",
        dest="fields_only",
        action="store_true",
        help="Write the transforms only: skip the moved image, which is derived from them. For a "
        "caller that composes the transform itself and would delete it.",
    )
    add_device(reg)
    add_tmp_dir(reg, tmp_dir_help)

    # eval -------------------------------------------------------------------
    ev = subparsers.add_parser(
        "eval",
        help="Evaluate a registration on any subset of modalities (image/seg/fid); at least one is required.",
        description="""Evaluate a transform on any subset of three modalities, at least one:

  image  -f/-m                              MAE between the fixed image and the warped moving one
  seg    --gt-fixed-seg/--gt-moving-seg     Dice per label, and their mean
  fid    --gt-fixed-fid/--gt-moving-fid     TRE in mm: the fixed landmarks (Slicer .fcsv) pushed
                                            through the transform, against the moving ones, row by row

The moving side is the ORIGINAL moving data: the transform is applied here, and omitted it is the
identity (the misalignment before registration). A displacement-field transform also gets its
Jacobian determinant statistics (folded fraction, minimum, SD of its log). The metrics go to
<output>/<case>/Evaluation/<modality>/, and <output>/Evaluation_summary.json gathers the cohort.""",
        epilog="""example:
  impact-reg-konfai eval --transform Output/P000/Transform.h5 -f fixed.nii.gz -m moving.nii.gz \\
      --gt-fixed-seg fixed_seg.nii.gz --gt-moving-seg moving_seg.nii.gz -o Evaluation""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ev.add_argument(
        "-f", "--fixed-images", type=local_path, nargs="+", default=[], help="Fixed image(s) [image modality]."
    )
    ev.add_argument(
        "-m", "--moving-images", type=local_path, nargs="+", default=[], help="Moving image(s) [image modality]."
    )
    ev.add_argument(
        "--transform",
        type=local_path,
        nargs="+",
        default=[],
        help="Transform(s) warping moving onto fixed, one per case or one for every case (identity if omitted): an "
        "ITK transform file (.h5, .tfm, .itk.txt: linear, B-spline or displacement field, as 'register' writes) or "
        "a displacement field image. A composite of a linear transform and a field is not read.",
    )
    ev.add_argument(
        "--gt-fixed-seg", type=local_path, nargs="+", default=[], help="Fixed segmentation(s) [seg modality]."
    )
    ev.add_argument(
        "--gt-moving-seg", type=local_path, nargs="+", default=[], help="Moving segmentation(s) [seg modality]."
    )
    ev.add_argument(
        "--gt-fixed-fid", type=local_path, nargs="+", default=[], help="Fixed landmark file(s) [fid modality]."
    )
    ev.add_argument(
        "--gt-moving-fid", type=local_path, nargs="+", default=[], help="Moving landmark file(s) [fid modality]."
    )
    ev.add_argument(
        "--mask", type=local_path, nargs="+", default=None, help="Optional evaluation mask(s) [image modality]."
    )
    ev.add_argument(
        "-o",
        "--output",
        type=local_path,
        default=Path("./Output").resolve(),
        help="Output directory (default: ./Output).",
    )
    add_device(ev)
    add_tmp_dir(ev, tmp_dir_help)

    # uncertainty ------------------------------------------------------------
    unc = subparsers.add_parser(
        "uncertainty",
        help="Voxel-wise spread map from an ensemble of displacement fields.",
        description="""Measure, voxel by voxel and in mm, the spread of an ensemble of displacement fields on one
grid, such as the fields 'register --keep-fields' keeps under <case>/Ensemble/: the root mean square
distance of the members' displacement vectors to their mean (sample, N-1). Writes
<output>/uncertainty/Uncertainty.mha.""",
        epilog="example:\n  impact-reg-konfai uncertainty --dvf Output/P000/Ensemble/*.h5 -o Output/P000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    unc.add_argument(
        "--dvf",
        type=local_path,
        nargs="+",
        required=True,
        help="Two or more ensemble displacement fields (e.g. the per-preset DVFs written by 'register').",
    )
    unc.add_argument(
        "-o",
        "--output",
        type=local_path,
        default=Path("./Output").resolve(),
        help="Output directory (default: ./Output).",
    )
    add_device(unc, download=False)
    add_tmp_dir(unc, tmp_dir_help)

    # apply ------------------------------------------------------------------
    app_ = subparsers.add_parser(
        "apply",
        help="Warp more moving-side images (a segmentation, another sequence) onto the fixed grid.",
        description="""Warp images of the moving side onto the fixed grid through a transform 'register' wrote:
a segmentation of the moving image (--labels: nearest-neighbour, no invented label values),
another sequence of the same subject, a mask. Each is written as <output>/<its name>, in its own
form (file, OME-Zarr store or DICOM series); -o must not be the directory holding the images.

The transform maps fixed points to moving points, so it brings images from the moving side; no
inverse is computed for the other direction.""",
        epilog="""example:
  impact-reg-konfai apply --transform Output/P000/Transform.h5 -f fixed.nii.gz \\
      -i moving_organs.nii.gz --labels -o Output/P000""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    app_.add_argument("--transform", type=local_path, required=True, help="The transform 'register' wrote.")
    app_.add_argument("-f", "--fixed-image", type=local_path, required=True, help="The fixed image: the output grid.")
    app_.add_argument(
        "-i",
        "--images",
        type=local_path,
        nargs="+",
        required=True,
        help="Moving-side image(s), or a directory of them.",
    )
    app_.add_argument("--labels", action="store_true", help="The images are label maps: nearest-neighbour.")
    app_.add_argument(
        "-o",
        "--output",
        type=local_path,
        default=Path("./Output").resolve(),
        help="Output directory (default: ./Output).",
    )
    add_device(app_, download=False)
    add_tmp_dir(app_, tmp_dir_help)

    args = parser.parse_args()
    if args.command == "register":
        args.presets += args.preset_options
        if not args.presets:
            reg.error("name a preset, first or with -p/--preset: a name after -f/-m is read as one more image.")
    if args.command == "uncertainty" and len(args.dvf) < 2:
        unc.error("--dvf needs two or more displacement fields: a spread is measured across an ensemble.")
    _stop_on_sigterm()

    # konfai's Python API raises designed refusals (message + remedy); the CLI's job is to print
    # them and exit 1: the same contract the konfai CLI itself offers. A missing file, a bad value or
    # a failed step says what went wrong in its message as well, and its traceback only buried that
    # line; a preset that failed has already printed its own error, so the parent's is one more.
    # IMPACT_REG_DEBUG=1 keeps the tracebacks. Those other errors also cover a bug, whose message
    # alone has no location, so their line says how to get the traceback.
    from konfai.utils.errors import KonfAIError

    try:
        _dispatch(args, ev)
    except subprocess.CalledProcessError as error:
        if os.environ.get("IMPACT_REG_DEBUG"):
            raise
        command = " ".join(str(part) for part in error.cmd[:3])
        print(f"'{command}' failed (exit {error.returncode}); its error is printed above.", file=sys.stderr)
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except (KonfAIError, OSError, ValueError, RuntimeError) as error:
        if os.environ.get("IMPACT_REG_DEBUG"):
            raise
        hint = "" if isinstance(error, KonfAIError) else " (IMPACT_REG_DEBUG=1 prints the traceback)"
        print(str(error).strip() + hint, file=sys.stderr)
        sys.exit(1)


def _report_ensemble(output: Path, presets: list[str]) -> None:
    """Say what each case's ``Ensemble/`` holds once a run kept its fields.

    Members accumulate: runs into the same ``--output`` add theirs beside the earlier ones, a legitimate way to build
    an ensemble one preset at a time, and ``uncertainty --dvf <case>/Ensemble/*`` takes them all. So the members this
    run did not write are named, and a lone member is flagged: a spread needs two.
    """
    for ensemble in sorted(output.glob("*/Ensemble")):
        members = sorted(entry.name.partition(".")[0] for entry in ensemble.iterdir())
        earlier = [member for member in members if member not in presets]
        note = f" ({', '.join(earlier)} from an earlier run)" if earlier else ""
        if len(members) < 2:
            note += "; 'uncertainty' needs two or more: run another preset into the same --output"
        print(f"[ImpactReg] {ensemble}: {', '.join(members)}{note}.")


def _dispatch(args: argparse.Namespace, ev: argparse.ArgumentParser) -> None:
    # Imported once the command line is parsed: it brings torch and KonfAI, seconds that --help and --version skip.
    from impact_reg_konfai import impact_reg

    if args.command in ("list", "show"):
        from konfai_apps.cli import describe_app, list_apps, print_app, print_apps

        if args.command == "list":
            print_apps("impact-reg-konfai", list_apps(impact_reg.IMPACT_REG_KONFAI_REPO, "registration"))
        else:
            print_app(args.name, describe_app(impact_reg.IMPACT_REG_KONFAI_REPO, args.name))
        return
    app = impact_reg.ImpactRegKonfAIApp(
        download=getattr(args, "download", False), force_update=getattr(args, "force_update", False)
    )
    if args.command == "register":
        if not args.gpu and args.cpu is None and not args.quiet:
            import torch

            if torch.cuda.is_available():
                print(
                    "[ImpactReg] No --gpu: registering on the CPU, much slower for most presets. Pass --gpu 0 to use "
                    "the GPU, or --cpu N to stay on the CPU without this note.",
                    flush=True,
                )
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
            keep_dvf=args.keep_fields,
            config_overrides=args.config_overrides,
            tmp_dir=args.tmp_dir,
            fields_only=args.fields_only,
            max_voxels=args.max_voxels,
        )
        if args.keep_fields and not args.quiet:
            _report_ensemble(args.output, args.presets)

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
            tmp_dir=args.tmp_dir,
        )

    elif args.command == "apply":
        for path in app.apply(
            args.transform,
            args.fixed_image,
            args.images,
            output=args.output,
            labels=args.labels,
            gpu=args.gpu,
            cpu=args.cpu,
            quiet=args.quiet,
            tmp_dir=args.tmp_dir,
        ):
            if not args.quiet:
                print(f"[ImpactReg] {path}")

    elif args.command == "uncertainty":
        gpu = [] if args.cpu is not None else args.gpu
        app.uncertainty(
            dvfs=args.dvf,
            output=args.output,
            gpu=gpu,
            cpu=args.cpu,
            quiet=args.quiet,
            tmp_dir=args.tmp_dir,
        )


if __name__ == "__main__":
    main()
