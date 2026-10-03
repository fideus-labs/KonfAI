#!/usr/bin/env bash
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

# Tutorial 3: the same MR/CT pair through the three engines, scored the same way.
#
# elastix, FireANTs and ConvexAdam each come with an intensity preset and IMPACT presets; Generic_Rigid is the
# reference, the rigid part of the known transform recovered and its deformation left. On an MR/CT pair the
# intensity metrics of the deformable stages (FireANTs' local correlation) compare grey values that do not correspond,
# where the IMPACT presets compare features that do. The table shows what each costs and what it recovers.
#
#   bash 03_three_backends.sh           # GPU=0 by default; FireANTs and the IMPACT presets want a GPU
set -euo pipefail
cd "$(dirname "$0")"
DEV="--gpu ${GPU-0}"
[ -f data/fixed_ct.mha ] || python prepare_data.py data

PRESETS="Generic_Rigid Generic_Rigid_BSpline Elastix_IMPACT_Static FireANTs_SyN FireANTs_Anatomix ConvexAdam_Composite"
for preset in $PRESETS; do
    # The fixed mask keeps each metric on the patient, off the CT's head rest and past the MRI's field of view.
    impact-reg-konfai register $preset -f data/fixed_ct.mha -m data/moving_mr.mha --fixed-mask data/fixed_mask.mha \
        -o out/03_$preset $DEV -q
    impact-reg-konfai eval --transform out/03_$preset/P000/Transform.h5 \
        --gt-fixed-seg data/fixed_labels.mha --gt-moving-seg data/moving_labels.mha \
        --gt-fixed-fid data/fixed_landmarks.fcsv --gt-moving-fid data/moving_landmarks.fcsv \
        -o out/03_$preset/eval $DEV -q
done
# Before registration, for reference: eval without a transform scores the identity.
impact-reg-konfai eval --gt-fixed-seg data/fixed_labels.mha --gt-moving-seg data/moving_labels.mha \
    --gt-fixed-fid data/fixed_landmarks.fcsv --gt-moving-fid data/moving_landmarks.fcsv -o out/03_before $DEV -q

python - $PRESETS <<'EOF'
import json, sys
from pathlib import Path

def score(folder):
    aggregates = json.loads((folder / "Evaluation_summary.json").read_text())["aggregates"]
    mean = lambda key: aggregates[key]["mean"] if key in aggregates else float("nan")
    return mean("MovingSeg:FixedSeg:Dice"), mean("FixedFid:MovingFid:TRE"), mean("Transform:Jacobian:folded_fraction")

dice, tre, _ = score(Path("out/03_before"))
print(f"{'preset':<24}{'time':>8}{'Dice':>8}{'TRE mm':>9}{'folded':>9}")
print(f"{'(before)':<24}{'':>8}{dice:>8.3f}{tre:>9.2f}")
for preset in sys.argv[1:]:
    runtime = json.loads(Path(f"out/03_{preset}/register.json").read_text())["runtime_s"]
    dice, tre, folded = score(Path(f"out/03_{preset}/eval"))
    print(f"{preset:<24}{runtime:>7.0f}s{dice:>8.3f}{tre:>9.2f}{folded:>9.2%}")
EOF
