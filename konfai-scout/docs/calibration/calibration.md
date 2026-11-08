# Calibration on real registrations: L2R HippocampusMR, ConvexAdam

Five development pairs of Learn2Reg HippocampusMR (`0001_0000_0021_0000`, `0034_0000_0200_0000`,
`0045_0000_0155_0000`, `0048_0000_0038_0000`, `0049_0000_0051_0000`; 64^3 voxels of 1 mm, the fixed and moving
hippocampus labels given as references), each registered by ImpactReg's `ConvexAdam#composite` (impact-reg-konfai
1.8.9.dev30, a displacement field on the fixed grid). `examples/calibrate_real.py` with `--repeats 2
--defects-per-case 1 --max-tool-calls 25 --max-seconds 900`: per case the clean copy and one planted copy (the
defect types rotate over the cases), each inspected twice, 20 inspections. Provider `claude-code`, model
`claude-opus-5-5`, active mode, blind, prompt 1.3, Claude Agent SDK 0.2.179. Every inspection concluded by itself in 8 to 12
tool calls (no budget exhausted), 36 s on average; $3.25 in all at API prices. `calibration.json` holds every
inspection, the planted boxes and the minimum Jacobian determinant of each local defect.

| defect | planted | found | sensitivity | clean | false alarms | specificity |
|---|---|---|---|---|---|---|
| local_fold | 4 | 4 | 1.0 | 10 | 1 | 0.9 |
| local_shift | 4 | 4 | 1.0 | 10 | 9 | 0.1 |
| global_shift | 2 | 2 | 1.0 | 10 | 9 | 0.1 |

Test-retest: defect matches agree in 0.85 of 20 pairs, verdicts in 0.8 of 10.

A match is an error observation of the defect's type over its box (anywhere, for the global shift). What the clean
copies show:

- **The shifts' specificity measures the controls, not Scout.** On 9 of the 10 clean inspections Scout recorded a
  residual misalignment of the hippocampus, and its own check of the box backs it: the warped labels' Dice against
  the fixed ones is 0.68 to 0.82 per label, below the 0.8 threshold in all but one case. ConvexAdam leaves these
  pairs misaligned by that measure, so a planted shift adds to an error already there. A control for a shift needs a
  registration that the threshold accepts, or a threshold set for the task (`ScoutConfig.thresholds`).
- **The one "false" fold is a real one.** `0034_0000_0200_0000`'s clean field folds near the hippocampus (Scout's
  check: folded fraction 0.001 and 0.068 in the boxes noted); the planted folds are found 4 of 4, with their
  measured folding.
- Inspections of one copy disagreed on 3 of 20 matches and 2 of 10 verdicts (`acceptable` against `defective` for a
  misalignment around the threshold).

Five cases and two repeats make this a proof that the protocol runs end to end on real fields, not a measure of
Scout's accuracy.
