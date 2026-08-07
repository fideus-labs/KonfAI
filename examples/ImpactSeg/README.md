# ImpactSeg Example

Colab-ready demo for [`impact-seg-konfai`](https://github.com/fideus-labs/KonfAI): multimodal / multi-organ **segmentation** via a
published model on [`VBoussot/ImpactSeg`](https://huggingface.co/VBoussot/ImpactSeg), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/ImpactSeg/ImpactSeg_demo.ipynb)

**Run all the cells of `ImpactSeg_demo.ipynb`.** It downloads one public demo case (CT), runs the
model, and plots a segmentation label map over the input. The model is fetched from the Hub on first use
(a few hundred MB) and a GPU is strongly recommended.

The same thing from a terminal:

```bash
cd /path/to/KonfAI                              # the paths below are relative to the repo root
pip install . ./konfai-apps ./apps/impact_seg   # all three from this checkout: the app pins konfai== and konfai-apps==
impact-seg-konfai segment body -i input.mha -o ./Output/ --gpu 0
```

produces a segmentation label map under `Output/`.
