#!/usr/bin/env bash
# Tutorial 6: evaluating a registration, and warping more images through it.
#
# The tutorial pair comes with a label map and landmarks on each image, so the registration can be scored with every
# metric eval knows: MAE between the images, Dice of the labels, TRE of the landmarks, and the Jacobian determinant of
# the field (does it fold?).
#
#   bash 06_evaluation.sh            # GPU=0 by default
set -euo pipefail
cd "$(dirname "$0")"
DEV="--gpu ${GPU-0}"
[ -f data/fixed_ct.mha ] || python prepare_data.py data
[ -f out/06_reg/P000/Transform.h5 ] || \
    impact-reg-konfai register Elastix_IMPACT_Static -f data/fixed_ct.mha -m data/moving_mr.mha \
        --fixed-mask data/fixed_mask.mha -o out/06_reg $DEV -q

LABELS="--gt-fixed-seg data/fixed_labels.mha --gt-moving-seg data/moving_labels.mha"
POINTS="--gt-fixed-fid data/fixed_landmarks.fcsv --gt-moving-fid data/moving_landmarks.fcsv"

# Before: no --transform means the identity, the misalignment the registration started from.
impact-reg-konfai eval $LABELS $POINTS -o out/06_before $DEV -q
# After. The moving side is always the ORIGINAL moving data: eval warps it through the transform itself.
# (MAE between a CT and an MRI measures nothing useful; for a same-modality pair add -f fixed -m moving.)
impact-reg-konfai eval --transform out/06_reg/P000/Transform.h5 $LABELS $POINTS --mask data/fixed_mask.mha \
    -o out/06_after $DEV -q

python - <<'EOF'
import json
for name in ("before", "after"):
    aggregates = json.load(open(f"out/06_{name}/Evaluation_summary.json"))["aggregates"]
    shown = {key: round(value["mean"], 3) for key, value in aggregates.items() if "Landmarks_" not in key}
    print(name, json.dumps(shown, indent=1))
EOF

# The transform maps a FIXED point to its MOVING partner (ITK's convention): eval pushes the fixed landmarks through
# it and measures their distance to the moving ones. The same transform warps any other moving-side image onto the
# fixed grid; a label map with nearest-neighbour interpolation:
impact-reg-konfai apply --transform out/06_reg/P000/Transform.h5 -f data/fixed_ct.mha \
    -i data/moving_labels.mha --labels -o out/06_reg/P000 -q
ls out/06_reg/P000
