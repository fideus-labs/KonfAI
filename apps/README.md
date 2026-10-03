# KonfAI Apps

These packages provide task-specific commands for published KonfAI models. An app
contains its configuration, weights and any custom Python code.

| Package | Task | Command |
| --- | --- | --- |
| [ImpactSeg](impact_seg/README.md) | CT, MR and CBCT segmentation | `impact-seg-konfai segment` |
| [ImpactSynth](impact_synth/README.md) | Synthetic CT from MR or CBCT | `impact-synth-konfai synthesize` |
| [TotalSegmentator](totalsegmentator/README.md) | Whole-body CT and MR segmentation | `totalsegmentator-konfai segment` |
| [MRSegmentator](mrsegmentator/README.md) | Whole-body MR segmentation | `mrsegmentator-konfai segment` |
| [IMPACT-Reg](impact_reg/README.md) | Fixed/moving image registration | `impact-reg-konfai register` |

Start with [Run a published app](https://konfai.readthedocs.io/en/latest/usage/apps.html)
for installation, a first prediction and result inspection. Use the package READMEs
above for model-specific inputs and variants.

Each app imports Python code and installs requirements by default. Only run apps
from sources you trust.

## Build or integrate an app

- [Fine-tune and package an app](https://konfai.readthedocs.io/en/latest/usage/packaging-apps.html)
- [Python API](https://konfai.readthedocs.io/en/latest/usage/python-api.html)
- [App server HTTP API](https://konfai.readthedocs.io/en/latest/reference/app-server-api.html)
- [SlicerKonfAI](https://github.com/vboussot/SlicerKonfAI)

These references cover the shared interfaces. The directories here contain the
thin commands and metadata for each task-specific package.
