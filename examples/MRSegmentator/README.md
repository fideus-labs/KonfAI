# MRSegmentator Example

Colab-ready demo for [`mrsegmentator-konfai`](https://github.com/fideus-labs/KonfAI) — multi-organ **MR segmentation** via a
published model on [`VBoussot/MRSegmentator-KonfAI`](https://huggingface.co/VBoussot/MRSegmentator-KonfAI), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/MRSegmentator/MRSegmentator_demo.ipynb)

**Run all the cells of `MRSegmentator_demo.ipynb`.** It downloads one public demo case (MR), runs the
model, and plots a multi-organ label map over the input. The model is fetched from the Hub on first use
(a few hundred MB) and a GPU is strongly recommended.

The same thing from a terminal:

```bash
pip install . ./konfai-apps ./apps/mrsegmentator   # all three from this checkout: the app pins konfai== and konfai-apps==
mrsegmentator-konfai segment -i input.mha -o ./Output/ --gpu 0
```

produces a multi-organ label map under `Output/`.
