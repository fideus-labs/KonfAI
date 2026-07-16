# TotalSegmentator Example

Colab-ready demo for [`totalsegmentator-konfai`](https://github.com/vboussot/KonfAI) — whole-body **CT segmentation** via a
published model on [`VBoussot/TotalSegmentator-KonfAI`](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussot/KonfAI/blob/main/examples/TotalSegmentator/TotalSegmentator_demo.ipynb)

`TotalSegmentator_demo.ipynb` downloads one public demo case (CT), inspects it, and shows the exact
command. Inference is toggle-gated (`RUN_INFER = False`) because the model is fetched from the Hub and a GPU
is recommended.

```bash
pip install ./apps/totalsegmentator
totalsegmentator-konfai segment total -i input.mha -o ./Output/ --gpu 0
```

produces a whole-body label map under `Output/`.
