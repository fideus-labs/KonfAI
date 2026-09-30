#!/usr/bin/env bash
# Tutorial 4: tuning a preset with --set, and choosing its feature layers.
#
# FireANTs_IMPACT drives SyN with the early layers of a TotalSegmentator CT model: the pairing for CT/CBCT, where
# texture carries the alignment. On an MR/CT pair, the organ-level layers of the MR model suit better. Both are one
# --set away, and the registration is scored after each change.
#
#   bash 04_tuning.sh            # GPU=0 by default
set -euo pipefail
cd "$(dirname "$0")"
DEV="--gpu ${GPU-0}"
[ -f data/fixed_ct.mha ] || python prepare_data.py data
MODEL=Predictor.Model.RegistrationNet.models.0

# What can be tuned, with values, ranges and descriptions:
impact-reg-konfai show FireANTs_IMPACT

run() {  # run <name> [--set ...]: register, evaluate, print Dice and TRE
    name=$1; shift
    impact-reg-konfai register FireANTs_IMPACT -f data/fixed_ct.mha -m data/moving_mr.mha --fixed-mask data/fixed_mask.mha \
        -o out/04_$name $DEV -q "$@"
    impact-reg-konfai eval --transform out/04_$name/P000/Transform.h5 \
        --gt-fixed-seg data/fixed_labels.mha --gt-moving-seg data/moving_labels.mha \
        --gt-fixed-fid data/fixed_landmarks.fcsv --gt-moving-fid data/moving_landmarks.fcsv -o out/04_$name/eval $DEV -q
    python -c "import json; a = json.load(open('out/04_$name/eval/Evaluation_summary.json'))['aggregates'];\
print(f\"$name: Dice {a['MovingSeg:FixedSeg:Dice']['mean']:.3f}, TRE {a['FixedFid:MovingFid:TRE']['mean']:.2f} mm\")"
}

run as_shipped                                                       # CT model, early layer ('01')
run mr_model_deep --set $MODEL.ref=VBoussot/impact-torchscript-models:TS/M730.pt --set $MODEL.layers_mask=0000001
run longer --set deformable_iterations=[400,200,100]                # more deformable iterations
run smoother --set smooth_warp_sigma=1.5                             # a more regular field

# Several presets in one run take PRESET:NAME=VALUE, since engines name their knobs differently:
#   --set FireANTs_SyN:deformable_iterations=[400,200,100] --set Generic_Rigid_BSpline:max_iterations=500
