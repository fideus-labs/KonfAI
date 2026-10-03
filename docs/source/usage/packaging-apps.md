# Package and fine-tune an app

An app holds the configuration, checkpoints and any custom code needed to run a
workflow. Install `konfai-apps` as described in {doc}`apps`. Start with the completed {doc}`CPU quickstart <../quickstart>` and stay
in its `konfai-first-run` directory.

## Package the quickstart

Create an `app.json` file with this content:

```json
{
  "display_name": "Two-class CT segmentation",
  "description": "Segments the foreground in the synthetic quickstart data.",
  "short_description": "Quickstart segmentation",
  "tta": 0,
  "mc_dropout": 0
}
```

The Apps CLI stages `-i` inputs as `Volume_0`, `Volume_1`, and so on, and `--gt`
references as `Reference_0`, `Reference_1`. Adapt copies of the prediction and
evaluation configs to those source names. Keep the destination names that the
model and metrics use. Save this as `prepare_app.py`:

```python
from pathlib import Path
import shutil
from ruamel.yaml import YAML

yaml = YAML()
root = Path("app-source")
root.mkdir(exist_ok=True)
shutil.copy2("Config.yml", root / "Config.yml")

prediction = yaml.load(Path("Prediction.yml"))
groups = prediction["Predictor"]["Dataset"]["groups_src"]
groups["Volume_0"] = groups.pop("CT")
for outputs in prediction["Predictor"]["outputs_dataset"].values():
    outputs["OutputDataset"]["same_as_group"] = "Volume_0:CT"
yaml.dump(prediction, root / "Prediction.yml")

evaluation = yaml.load(Path("Evaluation.yml"))
dataset = evaluation["Evaluator"]["Dataset"]
dataset["dataset_filenames"] = ["./Dataset:mha"]
groups = dataset["groups_src"]
groups["Volume_0"] = groups.pop("PRED")
groups["Reference_0"] = groups.pop("SEG")
yaml.dump(evaluation, root / "Evaluation.yml")
```

Prepare the configs and build the app:

```bash
python prepare_app.py
konfai-apps bundle CT_SEG \
  --out dist \
  --app-json app.json \
  --config app-source/Config.yml app-source/Prediction.yml app-source/Evaluation.yml \
  --checkpoint Checkpoints/CT_TWO_CLASSES/[0-9]*.pt
```

The result is `dist/CT_SEG/`. The quickstart uses the shipped YAML model, so there
is no custom Python file to include. For your own model, add `--model-py Model.py`.
Use `--support-file DESTINATION=SOURCE` with `--support-root` for helper files.
Check the generated `requirements.txt` before distributing the app.

## Check the packaged result

```bash
konfai-apps infer ./dist/CT_SEG \
  -i ./Dataset/CASE_000/CT.mha -o ./AppCheck --cpu 1
```

The label image is `AppCheck/CT_TWO_CLASSES/Dataset/P000/PRED.mha`. Compare it and its geometry
with the quickstart prediction for CASE_000. Once checked, the `CT_SEG/` folder
can be shared as a local app or placed in a Hugging Face model repository.

## Fine-tune it

The app includes `Config.yml`, which describes its training inputs and loss.
Choose a new output directory and keep your dataset's group names:

```bash
konfai-apps fine-tune ./dist/CT_SEG CT_SEG_TUNED \
  -d ./Dataset -o ./FineTune --epochs 2 --cpu 1
```

Fine-tuning starts from the first checkpoint with a fresh optimizer and schedule.
`--models CV_0 CV_1` selects named checkpoints and trains each independently.
The resulting app is written under the output directory. A directory containing
its own `Dataset` is refused so the run cannot replace your data.

See {doc}`python-api` to package from Python, or {doc}`../reference/cli` for all
bundle and fine-tuning options.
