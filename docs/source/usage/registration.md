# Register two images with IMPACT-Reg

IMPACT-Reg aligns a **moving** image to a **fixed** image using a published registration
preset. The fixed image defines the output grid. This guide runs an existing preset;
{doc}`../examples/registration` shows how to train a registration model.

## Install and choose a preset

```bash
python -m pip install impact-reg-konfai
impact-reg-konfai list
impact-reg-konfai show Elastix_IMPACT_Static
```

`list` gives the available presets. `show` describes one preset's engine, requirements
and adjustable parameters. The first run downloads its models and engine dependencies.
The installation requirements differ between elastix, ConvexAdam and FireANTs; consult
the [IMPACT-Reg installation guide](https://github.com/fideus-labs/KonfAI/blob/c3e275ca0e7fbb1dba60b1469ad3368713c2d4db/apps/impact_reg/docs/troubleshooting.md)
before changing engines.

## Register the pair

For an MR image and a CT reference:

```bash
impact-reg-konfai register Elastix_IMPACT_Static \
  -f ./fixed_ct.nii.gz -m ./moving_mr.nii.gz \
  -o ./Registration --gpu 0
```

Replace the two image paths with your pair. `--gpu 0` selects the first GPU.
When the CT includes a table or anatomy outside the MR field of view, supply its body
mask with `--fixed-mask ./ct_body.nii.gz` to restrict the metric region.

For a single pair, inspect:

- `Registration/P000/Moved.<ext>`: the moving image resampled onto the fixed grid;
- `Registration/P000/Transform.h5`: the transform used to produce it;
- `register.json` under the output directory: the record of the run.

Overlay the moved and fixed images and check anatomical alignment. Use the
[evaluation guide](https://github.com/fideus-labs/KonfAI/blob/c3e275ca0e7fbb1dba60b1469ad3368713c2d4db/apps/impact_reg/docs/evaluation.md)
when you have reference labels or landmarks.

## Apply the transform to labels

Use the original moving-side labels. `--labels` selects nearest-neighbour interpolation:

```bash
impact-reg-konfai apply \
  --transform ./Registration/P000/Transform.h5 \
  -f ./fixed_ct.nii.gz -i ./moving_labels.nii.gz \
  --labels -o ./WarpedLabels
```

## Large images and output resolution

A large pair can be optimized on a coarser grid to fit the declared memory budget.
Some presets then refine the result in native-resolution tiles. The output transform
is placed on the fixed grid in either case: its grid alone does not tell you the
resolution used for optimization. See the large-image guide below for each preset's
route.

Loading the moved image and ensemble fields into Slicer requires memory for those
volumes, even when the registration itself used tiles.

## Choose the next step

- [Presets](https://github.com/fideus-labs/KonfAI/blob/c3e275ca0e7fbb1dba60b1469ad3368713c2d4db/apps/impact_reg/docs/presets.md):
  choose an engine and feature model for your image pair.
- [Large images](https://github.com/fideus-labs/KonfAI/blob/c3e275ca0e7fbb1dba60b1469ad3368713c2d4db/apps/impact_reg/docs/large-images.md):
  understand the coarse pass, native-resolution tiles and memory requirements.
- [Parameters](https://github.com/fideus-labs/KonfAI/blob/c3e275ca0e7fbb1dba60b1469ad3368713c2d4db/apps/impact_reg/docs/parameters.md):
  change a preset with `--set`.
- [IMPACT-Reg notebook](https://github.com/fideus-labs/KonfAI/blob/main/examples/ImpactReg/register_demo.ipynb):
  download a demo pair and inspect its registration.

Several presets can be listed in one `register` command. Their displacement fields are
averaged; pass `--keep-fields` to retain each field for a later uncertainty calculation.
Field disagreement measures variation between those registrations, not registration accuracy.
