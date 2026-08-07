# ImpactSynth Example

Colab-ready demo for [`impact-synth-konfai`](https://github.com/fideus-labs/KonfAI): whole-body **synthetic CT** from MR/CBCT via a
published model on [`VBoussot/ImpactSynth`](https://huggingface.co/VBoussot/ImpactSynth), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/ImpactSynth/ImpactSynth_demo.ipynb)

**Run all the cells of `ImpactSynth_demo.ipynb`.** It downloads one public demo case (MR), runs the
model, and plots a synthetic CT on the input grid over the input. The model is fetched from the Hub on first use
(a few hundred MB) and a GPU is strongly recommended.

The same thing from a terminal:

```bash
cd /path/to/KonfAI                                # the paths below are relative to the repo root
pip install . ./konfai-apps ./apps/impact_synth   # all three from this checkout: the app pins konfai== and konfai-apps==
impact-synth-konfai synthesize MR -i input.mha -o ./Output/ --gpu 0
```

produces a synthetic CT on the input grid under `Output/`.
