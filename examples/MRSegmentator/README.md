# MRSegmentator Example

Colab-ready demo for [`mrsegmentator-konfai`](https://github.com/vboussot/KonfAI) — multi-organ **MR segmentation** via a
published model on [`VBoussot/MRSegmentator-KonfAI`](https://huggingface.co/VBoussot/MRSegmentator-KonfAI), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussot/KonfAI/blob/main/examples/MRSegmentator/MRSegmentator_demo.ipynb)

`MRSegmentator_demo.ipynb` downloads one public demo case (MR), inspects it, and shows the exact
command. Inference is toggle-gated (`RUN_INFER = False`) because the model is fetched from the Hub and a GPU
is recommended.

```bash
pip install ./apps/mrsegmentator
mrsegmentator-konfai segment -i input.mha -o ./Output/ --gpu 0
```

produces a multi-organ label map under `Output/`.
