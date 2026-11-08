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
# --uncertainty keeps every member in one stack (InferenceStack.mha) for uncertainty to measure how far they are from
# one another: the spread of their vectors (a map, in mm), of the fixed landmarks pushed through each, whether that
# spread predicts the landmark error, and how the moving labels they propagate disagree. Ensemble presets of one
# nature: deformable ones here.
#
#   bash 07_ensemble_uncertainty.sh            # GPU=0 by default
set -euo pipefail
cd "$(dirname "$0")"
DEV="--gpu ${GPU-0}"
[ -f data/fixed_ct.mha ] || python prepare_data.py data

impact-reg-konfai register Elastix#rigid_bspline FireANTs#anatomix Elastix#impact_static \
    -f data/fixed_ct.mha -m data/moving_mr.mha --fixed-mask data/fixed_mask.mha -o out/07_ensemble --uncertainty $DEV
ls out/07_ensemble/P000

# The ensemble, scored on the labels and landmarks:
impact-reg-konfai eval --transform out/07_ensemble/P000/Transform.h5 --gt-fixed-seg data/fixed_labels.mha \
    --gt-moving-seg data/moving_labels.mha --gt-fixed-fid data/fixed_landmarks.fcsv \
    --gt-moving-fid data/moving_landmarks.fcsv -o out/07_eval $DEV -q
python -c "import json; a = json.load(open('out/07_eval/Evaluation_summary.json'))['aggregates'];\
print(f\"ensemble Dice {a['MovingSeg:FixedSeg:Dice']['mean']:.3f}  TRE {a['FixedFid:MovingFid:TRE']['mean']:.2f} mm\")"

# How its members disagree, from their stack:
impact-reg-konfai uncertainty -i out/07_ensemble/P000/InferenceStack.mha --gt-fixed-fid data/fixed_landmarks.fcsv \
    --gt-moving-fid data/moving_landmarks.fcsv --gt-moving-seg data/moving_labels.mha -o out/07_uncertainty $DEV -q
python -c "import json; a = json.load(open('out/07_uncertainty/Uncertainty_summary.json'))['aggregates'];\
[print(f'{key:<56} {stats[\"mean\"]:.3f}') for key, stats in a.items()]"

# The spread map: high where the members disagree, which is where to look before trusting the result. Read inside the
# body: in the air around it no image constrains the fields, and each engine leaves its own there.
python show.py data/fixed_ct.mha out/07_ensemble/P000/Moved.mha --titles "ensemble" -o out/07_ensemble.png
python -c "import glob, SimpleITK as sitk, numpy as np;\
u = sitk.GetArrayFromImage(sitk.ReadImage(glob.glob('out/07_uncertainty/P000/Uncertainty/Map/*/Output/P000/Uncertainty.mha')[0]));\
body = sitk.GetArrayFromImage(sitk.ReadImage('data/fixed_mask.mha')) > 0; u = u[body];\
print(f'spread in the body: median {np.median(u):.2f} mm, 95th percentile {np.percentile(u, 95):.2f} mm, max {u.max():.2f} mm')"
