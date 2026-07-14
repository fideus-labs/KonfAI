# ImpactSeg Example

Colab-ready demo for [`impact-seg-konfai`](https://github.com/vboussot/ImpactLoss) — multimodal / multi-organ **segmentation** via a
published model on [`VBoussot/ImpactSeg`](https://huggingface.co/VBoussot/ImpactSeg), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussot/KonfAI/blob/main/examples/ImpactSeg/ImpactSeg_demo.ipynb)

`ImpactSeg_demo.ipynb` downloads one public demo case (CT), inspects it, and shows the exact
command. Inference is toggle-gated (`RUN_INFER = False`) because the model is fetched from the Hub and a GPU
is recommended.

```bash
pip install ./apps/impact_seg
impact-seg-konfai segment body -i input.mha -o ./Output/ --gpu 0
```

produces a segmentation label map under `Output/`.
