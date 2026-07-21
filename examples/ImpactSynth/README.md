# ImpactSynth Example

Colab-ready demo for [`impact-synth-konfai`](https://github.com/fideus-labs/KonfAI) — whole-body **synthetic CT** from MR/CBCT via a
published model on [`VBoussot/ImpactSynth`](https://huggingface.co/VBoussot/ImpactSynth), run through the KonfAI runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/ImpactSynth/ImpactSynth_demo.ipynb)

`ImpactSynth_demo.ipynb` downloads one public demo case (MR), inspects it, and shows the exact
command. Inference is toggle-gated (`RUN_INFER = False`) because the model is fetched from the Hub and a GPU
is recommended.

```bash
pip install ./apps/impact_synth
impact-synth-konfai synthesize MR -i input.mha -o ./Output/ --gpu 0
```

produces a synthetic CT on the input grid under `Output/`.
