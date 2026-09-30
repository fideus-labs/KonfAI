#!/usr/bin/env bash
# Tutorial 5: a pair larger than the GPU, registered at native resolution in tiles.
#
# Two mouse brains from the public ExaSPIM data, 190 million voxels each at level 6. No preset registers that whole on
# a 24 GB card: register runs the preset once on a coarse copy of the whole pair, read from the stores' own pyramid,
# then its deformable stage on native tiles, and composes the two. The plan is printed before it runs.
#
#   pip install "fsspec[http]"             # read the public bucket over HTTP
#   bash 05_large_images.sh [level]         # level 6 by default; GPU=0 by default
set -euo pipefail
cd "$(dirname "$0")"
LEVEL=${1-6}
DEV="--gpu ${GPU-0}"
python download_exaspim.py $LEVEL exaspim
F=exaspim/fixed_level$LEVEL.ome.zarr
M=exaspim/moving_level$LEVEL.ome.zarr

# Before registration: how the two brains' masks overlap as they came.
impact-reg-konfai eval --gt-fixed-seg exaspim/fixed_level${LEVEL}_brain.mha --gt-moving-seg exaspim/moving_level${LEVEL}_brain.mha \
    -o out/05_before $DEV -q
python -c "import json; a = json.load(open('out/05_before/Evaluation_summary.json'))['aggregates'];\
print('before: brain Dice', round(a['MovingSeg:FixedSeg:Dice']['mean'], 3))"

# Each engine on the pair as downloaded. Generic_Rigid has no deformable stage: its global pass is the result. The others
# refine on native tiles. The tissue lies in a few tens of grey values beside lone voxels in the tens of thousands, which
# every mutual-information stage clamps away (0.01-99.99 percentiles) before binning.
for preset in Generic_Rigid Generic_Rigid_BSpline ConvexAdam_Composite FireANTs_SyN; do
    # --tmp-dir: the intermediates are volume-sized (about 60 bytes a voxel at the peak of a tiled run).
    /usr/bin/time -f "$preset: %e s, peak RAM %M kB" impact-reg-konfai register $preset -f $F -m $M \
        -o out/05_$preset --tmp-dir out/05_tmp $DEV
    # The brains' masks, warped and compared: how well the two brains overlap.
    impact-reg-konfai eval --transform out/05_$preset/P000/Transform.h5 \
        --gt-fixed-seg exaspim/fixed_level${LEVEL}_brain.mha --gt-moving-seg exaspim/moving_level${LEVEL}_brain.mha \
        -o out/05_$preset/eval $DEV -q
    python -c "import json; a = json.load(open('out/05_$preset/eval/Evaluation_summary.json'))['aggregates'];\
print('$preset: brain Dice', round(a['MovingSeg:FixedSeg:Dice']['mean'], 3))"
done
du -sh out/05_*/P000/*
