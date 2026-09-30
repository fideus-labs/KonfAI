#!/usr/bin/env bash
# Tutorial 1: a first registration, from installation to the moved image.
#
#   pip install impact-reg-konfai "torch==2.12.*" matplotlib
#   bash 01_first_registration.sh            # GPU=0 by default; GPU= (empty) to run on the CPU
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${GPU-0}
DEV=$([ -n "$DEVICE" ] && echo "--gpu $DEVICE" || echo "--cpu 4")

# 1. The pair: a head and neck CT and the MRI of the same patient, the MRI moved by a known rigid transform.
[ -f data/fixed_ct.mha ] || python prepare_data.py data

# 2. What can register it?
impact-reg-konfai list

# 3. A rigid registration first: elastix, mutual information, a few seconds. The first run downloads the elastix
#    binary (and, if your torch is another version than the one it was built with, the matching LibTorch).
impact-reg-konfai register Generic_Rigid -f data/fixed_ct.mha -m data/moving_mr.mha -o out/01_rigid $DEV

# 4. A deformable one on deep features: elastix driven by IMPACT on TotalSegmentator and MIND, the preset for MR/CT.
#    The fixed mask keeps its metrics on the patient: the CT also holds the head rest, and the MRI covers less than
#    the CT, and without it the deformation pulls the MRI onto what it does not show.
impact-reg-konfai register Elastix_IMPACT_Static -f data/fixed_ct.mha -m data/moving_mr.mha --fixed-mask data/fixed_mask.mha \
    -o out/01_static $DEV

# 5. What came out: the transform (a displacement field on the fixed grid), the moved MRI, and a record of the run.
ls out/01_static out/01_static/P000

# 6. Before and after: the MRI's edges over the CT.
python show.py data/fixed_ct.mha data/moving_mr.mha out/01_rigid/P000/Moved.mha out/01_static/P000/Moved.mha \
    --titles "before" "Generic_Rigid" "Elastix_IMPACT_Static" -o out/01_before_after.png
