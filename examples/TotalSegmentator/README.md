# TotalSegmentator Example

Colab-ready demo for [`totalsegmentator-konfai`](https://github.com/fideus-labs/KonfAI): whole-body **CT segmentation** via a
published model on [`VBoussot/TotalSegmentator-KonfAI`](https://huggingface.co/VBoussot/TotalSegmentator-KonfAI), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/TotalSegmentator/TotalSegmentator_demo.ipynb)

**Run all the cells of `TotalSegmentator_demo.ipynb`.** It downloads one public demo case (CT), runs the
model, and plots a whole-body label map over the input. The model is fetched from the Hub on first use
(a few hundred MB) and a GPU is strongly recommended.

The same thing from a terminal:

```bash
cd /path/to/KonfAI                                    # the paths below are relative to the repo root
pip install . ./konfai-apps ./apps/totalsegmentator   # all three from this checkout: the app pins konfai== and konfai-apps==
totalsegmentator-konfai segment total -i input.mha -o ./Output/ --gpu 0
```

produces a whole-body label map under `Output/`.
