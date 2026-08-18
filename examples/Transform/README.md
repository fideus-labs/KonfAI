# Transform: the workflow that makes a dataset

`konfai TRANSFORM` reads a dataset, applies a chain of transforms, and writes a
dataset: it is the dataset-preparation workflow. `EVALUATION` measures, this
one is the workflow you reach for when the thing you need is *data*.

Two configs here, one per direction the cardinality can change.

| File | Shape | What it builds |
| --- | --- | --- |
| `Transform.yml` | N → 1 | One median template from a cohort that does not share a grid |
| `Transform_expand.yml` | 1 → N | Four drawn copies of every case |

Everything runs on CPU in under a minute, on ~3.5 MB of synthetic data this
directory generates. Nothing is downloaded.

## Run it

```bash
pip install "konfai[imaging]"                     # make_dataset.py writes .mha through SimpleITK
python make_dataset.py                            # ./Raw/<case>/CT.mha, 6 cases

konfai TRANSFORM --config Transform.yml --plan    # read the plan first
konfai TRANSFORM --config Transform.yml           # ./Template/template/CT_template.mha

konfai TRANSFORM --config Transform_expand.yml    # ./Augmented/<case>_r01..r04/
```

`--plan` writes none of the deliverable, but it is not read-only: it opens a real region-write on
each destination and removes it, so the output store itself may be created.

## What the cohort looks like, and why that matters

`make_dataset.py` writes six volumes that agree about nothing: extents differ by
a few voxels, spacings by up to 30%, origins by more than a voxel. That is the
ordinary state of a cohort as acquired (an acquisition's stage coordinates are
not an anatomical frame), and it is why `Reduce` refuses it as stored:

```text
case 'CASE_001' lands on extent [44, 60, 52] where 'CASE_000' lands on [48, 56, 56]
```

`Resample: {reference: ...}` is what makes the agreement true rather than waived. It
puts every member on one named member's grid (extent, spacing, origin and
direction), so `grid: strict` passes because the cohort really is on one grid,
not because the check was relaxed.

```{warning}
`grid: shape_only` and `grid: reference:<case>` will happily average volumes
that do not overlap. The result still looks like a volume. Put the cohort on one
grid first.
```

## Read the plan before you read anything else

`--plan` prints what the run will do and stops. Nothing is written first, and
the plan is the run's own verdict, not an estimate of it:

```text
  CT -> CT_template: REDUCE 6 case(s) -> 1 output 'template': REDUCE
    19 resident region(s) of 64 row(s) = 0.01 GiB  (every case resident per region)
    peak ~= 0.01 GiB vs per-rank budget 1.86 GiB
```

Nineteen regions for six cases is `Median` being honest: it needs every case
resident to name the middle one, then stacks them into a new tensor and sorts a
copy of that. `Mean` folds one case at a time and holds two regions whatever the
cohort's size: swap `operator: Mean` and watch the line change.

The plan also reports how much of the reference each member actually covers:

```text
NOTE: case 'CASE_005' covers 82.0% of reference 'CASE_000'; the rest of what it writes is fill (0)
```

That is worth reading. A member covering 60% of the reference is contributing
fill to nearly half the template, and nothing about the written volume would
look wrong.

## Where the cardinality changes

In `Transform.yml`, everything above `Reduce` runs once per case and everything
below it runs once, on the folded result:

```yaml
Clip: {min_value: 0.0, max_value: 400.0}      # per case
Resample: {reference: CASE_000, ...}          # per case
Reduce: {operator: Median, output: template}  # <- N becomes 1 here
Write: {dataset: ./Template:mha}              # once
```

Only voxel-local stages may follow a `Reduce`, because each is handed one
*region* of the result. Anything reading across space belongs in a second chain
that reads the written template back.

`Transform_expand.yml` is the mirror image: `Expand` marks where one case
becomes four, and the draws go **after** it. A draw declared before the marker
is applied once per case, which is a random transform rather than an
augmentation, and the run refuses it and says so.

Because `Brightness` is pointwise, the four copies share a single read pass over
the source: the plan says `4 STREAM (shared read pass), 0 STREAM (own pass)`. A draw that moves voxels
around (`Rotate`, `Flip`) cannot share it, and each copy gets its own pass.

## Reproducibility

`manual_seed` in `Transform_expand.yml` is what makes an image chain and its mask
chain draw the *same* copies. The two chains never meet: each derives its draws
from `(seed, the case's name, the draw's class and its rank among the draws of
that class)`, so they agree without coordinating, and a case keeps its copies
whatever subset of the cohort a run covers. A mask rotated by a different angle
than its image is a silently ruined dataset, not an error, which is why the seed
is not optional in practice.

## Next

- The full reference: [TRANSFORM configuration](https://konfai.readthedocs.io/en/latest/config_guide/transform.html)
- Chains that embed a trained model (`KonfAIInference`), streaming rules and the
  memory budget are all documented there.
