# Bring your model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/BringYourModel/BringYourModel_demo.ipynb)

A model you already have, trained and run on a KonfAI dataset through `konfai.train_model` and
`konfai.predict_model`: no YAML, the same engine. Open
`BringYourModel_demo.ipynb` (Colab-ready); it fetches the Segmentation
example's five cases, trains MONAI's `UNet` for three epochs and predicts with it.

The workspaces it writes (`Checkpoints/`, `Statistics/`, `Predictions/`) and the dataset it fetches
(`Dataset/`) stay in this directory and are git-ignored.
