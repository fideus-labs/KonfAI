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

# Tutorial 7: an ensemble of presets, and where its members disagree.
#
# Several presets registering the same pair give several fields; register averages them into one transform, and with
# --keep-fields keeps each member so that uncertainty can map, voxel by voxel, how far they are from one another
# (the root mean square distance of the members' vectors to their mean, in mm). Ensemble presets of one nature:
# deformable ones here.
#
#   bash 07_ensemble_uncertainty.sh            # GPU=0 by default
set -euo pipefail
cd "$(dirname "$0")"
DEV="--gpu ${GPU-0}"
[ -f data/fixed_ct.mha ] || python prepare_data.py data

impact-reg-konfai register Generic_Rigid_BSpline FireANTs_Anatomix Elastix_IMPACT_Static \
    -f data/fixed_ct.mha -m data/moving_mr.mha --fixed-mask data/fixed_mask.mha -o out/07_ensemble --keep-fields $DEV
ls out/07_ensemble/P000 out/07_ensemble/P000/Ensemble

impact-reg-konfai uncertainty --dvf out/07_ensemble/P000/Ensemble/*.h5 -o out/07_ensemble/P000 $DEV -q

# Each member and the ensemble, scored on the labels and landmarks:
for field in out/07_ensemble/P000/Ensemble/*.h5 out/07_ensemble/P000/Transform.h5; do
    name=$(basename $field .h5); [ $name = Transform ] && name=ensemble
    impact-reg-konfai eval --transform $field --gt-fixed-seg data/fixed_labels.mha --gt-moving-seg data/moving_labels.mha \
        --gt-fixed-fid data/fixed_landmarks.fcsv --gt-moving-fid data/moving_landmarks.fcsv -o out/07_eval_$name $DEV -q
    python -c "import json; a = json.load(open('out/07_eval_$name/Evaluation_summary.json'))['aggregates'];\
print(f\"{'$name':<24} Dice {a['MovingSeg:FixedSeg:Dice']['mean']:.3f}  TRE {a['FixedFid:MovingFid:TRE']['mean']:.2f} mm\")"
done

# The spread map: high where the members disagree, which is where to look before trusting the result. Read inside the
# body: in the air around it no image constrains the fields, and each engine leaves its own there.
python show.py data/fixed_ct.mha out/07_ensemble/P000/Moved.mha --titles "ensemble" -o out/07_ensemble.png
python -c "import SimpleITK as sitk, numpy as np; u = sitk.GetArrayFromImage(sitk.ReadImage('out/07_ensemble/P000/uncertainty/Uncertainty.mha'));\
body = sitk.GetArrayFromImage(sitk.ReadImage('data/fixed_mask.mha')) > 0; u = u[body];\
print(f'spread in the body: median {np.median(u):.2f} mm, 95th percentile {np.percentile(u, 95):.2f} mm, max {u.max():.2f} mm')"
