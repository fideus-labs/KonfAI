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

The result is `dist/CT_SEG/`. Each checkpoint is written as a `.safetensors` file
holding the model weights only: the optimizer state and the training counters stay in
the run's `.pt`. An app still reads the `.pt` weights of an older bundle, and takes the
`.safetensors` when both are there. The quickstart uses the shipped YAML model, so there
is no custom Python file to include. For your own model, add `--model-py Model.py`.
Use `--support-file DESTINATION=SOURCE` with `--support-root` for helper files. A helper
file cannot end in `.pt` or `.safetensors`: an app reserves those for its models.
Check the generated `requirements.txt` before distributing the app.

## Check the packaged result

```bash
konfai-apps infer ./dist/CT_SEG \
  -i ./Dataset/CASE_000/CT.mha -o ./AppCheck --cpu 1
```

The label image is `AppCheck/CT_TWO_CLASSES/Dataset/P000/PRED.mha`. Compare it and its geometry
with the quickstart prediction for CASE_000. Once checked, the `CT_SEG/` folder
can be shared as a local app or placed in a Hugging Face model repository.

An app can read another app's output. Declare it in `app.json` as `"needs": {"body": "CT_BODY_SEG"}`:
before it predicts, the app runs the needed app on the same inputs (as many as that app declares) and
stages its output for each case as the group `body`, beside `Volume_0`. A stage of the config then reads it,
for example `Crop: {source: body, labels: [10, 11], margin_mm: 20}`. A bare name is a sibling of the app (the
same folder, or the same Hugging Face repository); `repo_id:app_name` or a path names any other.

`--set NAME=VALUE` reaches the model's parameters by name. To expose another key of the prediction config, such as
an argument of a transform stage, map a name to its dotted path from the config root in `app.json`:
`"parameters": {"crop_shape": "Predictor.Dataset.groups_src.Volume_0.groups_dest.Volume.transforms.model:CentreCrop.shape"}`.
`konfai-apps show` and a UI's parameters dialog then list `crop_shape` with its value and the constraints the
stage's types declare, and `--set crop_shape=[160,192,160]` writes that key. A name the model already takes, or a
path the config does not have, is refused.

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
