# Examples

Choose an example by what you want to produce. The CPU quickstart creates its own
small dataset; the app notebooks download demo inputs and model weights. Runtime
and download size depend on the example and your machine.

## Learn a workflow

| Task | Start here | Data and result |
| --- | --- | --- |
| First training run | {doc}`../quickstart` | Four synthetic cases; train, predict, evaluate and verify on CPU |
| Prepare data | {doc}`transform` | Procedural volumes; transform and write a dataset |
| Segment anatomy | {doc}`segmentation` | Public pelvis CT; train and evaluate a 41-class model |
| Synthesize CT | {doc}`synthesis` | Public paired MR/CT; image-to-image training and masked evaluation |
| Train registration | {doc}`registration` | CT pairs with known deformation; train and evaluate VoxelMorph |

Run repository examples from their own directory: the configurations refer to local
models and datasets. Each tutorial states its dependencies and commands. Short training
runs demonstrate the workflow; their scores are not benchmarks of the method.

## Run a published model

These notebooks provide the app, demo input and result display. To use your own image,
start with {doc}`../usage/apps` or {doc}`../usage/registration`.

| Notebook | Task |
| --- | --- |
| [ImpactSeg](https://github.com/fideus-labs/KonfAI/blob/main/examples/ImpactSeg/ImpactSeg_demo.ipynb) | 11-label segmentation from CT, MR or CBCT |
| [TotalSegmentator](https://github.com/fideus-labs/KonfAI/blob/main/examples/TotalSegmentator/TotalSegmentator_demo.ipynb) | Whole-body CT segmentation |
| [MRSegmentator](https://github.com/fideus-labs/KonfAI/blob/main/examples/MRSegmentator/MRSegmentator_demo.ipynb) | Whole-body MR segmentation |
| [ImpactSynth](https://github.com/fideus-labs/KonfAI/blob/main/examples/ImpactSynth/ImpactSynth_demo.ipynb) | Synthetic CT from MR or CBCT |
| [IMPACT-Reg](https://github.com/fideus-labs/KonfAI/blob/main/examples/ImpactReg/register_demo.ipynb) | Align a fixed/moving pair and score it against reference labels |

The {doc}`visual-gallery` shows recorded outputs, their data sources and measurements.

```{include} ../../../examples/BringYourModel/README.md
:heading-offset: 1
```

See {doc}`../usage/adopting-konfai` for classpaths, imported weights and the limits of
training a live Python model.

```{include} ../../../examples/LargeImages/README.md
:heading-offset: 1
```

See {doc}`../usage/large-images` for storage choices and memory planning.
