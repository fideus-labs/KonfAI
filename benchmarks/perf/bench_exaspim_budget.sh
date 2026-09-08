#!/bin/bash
# Region-height sweep on a chunked OME-Zarr store: one TRANSFORM run per budget in a fresh process,
# wall, peak RSS, the sweep clock line, and three z slices against a reference store.
#
#   KONFAI_EXASPIM_BENCH=~/.cache/ngff-bench/konfai benchmarks/perf/bench_exaspim_budget.sh 1G 2G 4G 8G
#
# The bench directory holds Resample.yml (its `memory_budget:` line is rewritten per run), the
# Moving/Fixed stores it names, and the reference under $KONFAI_EXASPIM_REFERENCE (default:
# s3run/Out/fusion2halves/SPIM_on_773889.ome.zarr). Results append to results/<host>/exaspim.txt.
set -u
BENCH=${KONFAI_EXASPIM_BENCH:?the bench directory (Resample.yml + stores)}
REF=${KONFAI_EXASPIM_REFERENCE:-$BENCH/s3run/Out/fusion2halves/SPIM_on_773889.ome.zarr}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=$HERE/results/$(hostname)
mkdir -p "$OUT"
cd "$BENCH" || exit 1
NAME=$(grep -oE '^\s*name: .*' Resample.yml | head -1 | awk '{print $2}')
if [ -z "$NAME" ]; then
  echo "no 'name:' value in $BENCH/Resample.yml: refusing to clear Transforms/" >&2
  exit 1
fi
for B in "$@"; do
  for REP in 1 2; do
    cp Resample.yml Resample_sweep.yml
    sed -i "s/memory_budget: .*/memory_budget: $B/" Resample_sweep.yml
    rm -rf Registered "Transforms/$NAME"
    LOG=$OUT/exaspim_${B}_${REP}.log
    uptime > "$LOG"
    /usr/bin/time -v pixi run --manifest-path "$ROOT/pyproject.toml" --environment dev konfai TRANSFORM --config Resample_sweep.yml -y >> "$LOG" 2>&1
    WALL=$(grep -oE 'done in [0-9.]+ s' "$LOG" | head -1)
    RSS=$(grep "Maximum resident" "$LOG" | awk '{print $NF}')
    SWEEP=$(grep -E "\[KonfAI\] sweep" "Transforms/$NAME/log_0.txt" 2>/dev/null | head -1 | cut -c1-160)
    CMP=$(pixi run --manifest-path "$ROOT/pyproject.toml" --environment dev python -W ignore - "$REF" <<'PY' 2>/dev/null
import sys, glob
import numpy as np
from konfai.utils.ome_zarr import read_ome_zarr_data_slice
out = glob.glob("Registered/*/*.ome.zarr")[0]
diffs = []
for z in (0, 256, 512):
    window = (slice(0, 1), slice(z, z + 1), slice(None), slice(None))
    a = read_ome_zarr_data_slice(out, window)[0].astype(np.int64)
    b = read_ome_zarr_data_slice(sys.argv[1], window)[0].astype(np.int64)
    diffs.append(int(np.abs(a - b).max()))
print(" ".join(map(str, diffs)))
PY
)
    echo "$(git -C "$ROOT" rev-parse --short HEAD) budget=$B rep=$REP $WALL rss_kib=$RSS max|diff|=[$CMP] | $SWEEP" | tee -a "$OUT/exaspim.txt"
  done
done
rm -f Resample_sweep.yml
