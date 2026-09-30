# Fine-tune and package an app

Use this guide to turn an existing KonfAI workflow into an app that another person
can run. To use a published model first, see {doc}`apps`.

## Fine-tune an existing app

Replace `APP` with a local app folder or Hugging Face app reference, and `NAME` with
the name of the new app. Your dataset must match the groups in its training configuration.

```bash
konfai-apps fine-tune APP NAME -d ./Dataset --epochs 10 --gpu 0
```

Training starts from the first checkpoint with a fresh optimizer and schedule. Use
`--models CV_0 CV_1` to select named checkpoints. The run works in `./Output` and links
your dataset there; it refuses a folder that already contains its own `Dataset`.

## Package a trained workflow

From the directory containing your configuration, create `app.json`:

```json
{
  "display_name": "My segmentation model",
  "description": "Segments the target anatomy from CT.",
  "short_description": "CT segmentation",
  "tta": 0,
  "mc_dropout": 0
}
```

This example expects `Prediction.yml`, `Evaluation.yml`, `Model.py` and a trained
checkpoint under `Checkpoints/SEG_BASELINE/`:

```bash
konfai-apps bundle CT_SEG \
  --out dist \
  --app-json app.json \
  --config Prediction.yml Evaluation.yml \
  --checkpoint Checkpoints/SEG_BASELINE/[0-9]*.pt \
  --model-py Model.py
```

It writes `dist/CT_SEG/` with the configurations, checkpoints and Python code. Omit
`--model-py` when there is no custom code. The generated `requirements.txt` is inferred
from imports: check it before distributing the app. Include `Uncertainty.yml` only
when your workflow provides that operation.

## Check the packaged result

Run the package on a known input and inspect the resulting image:

```bash
konfai-apps infer ./dist/CT_SEG -i ./input.mha -o ./AppCheck --gpu 0
```

Compare the written geometry and predictions with the original workflow, and evaluate
against an aligned reference. Once checked, the `CT_SEG/` directory can be uploaded to a
Hugging Face model repository and addressed as `owner/repository:CT_SEG`.

See {doc}`python-api` to package from Python, or {doc}`../reference/cli` for all bundle options.
