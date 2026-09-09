# CT segmentation with two classes, on CPU

Four synthetic CT-like volumes, background `0` and foreground `1`, 32 x 32 x 4 voxels
with a non-unit spacing and a rotated direction. Nothing to download: `quickstart.py`
writes the cases and `default|UNet.yml` comes from the installed wheel. The model sees
one slice at a time; `CASE_003` is held out during training.

Copy this directory out of the checkout, install `konfai[itk]`, and run from the copy:

```bash
export OMP_NUM_THREADS=1
python quickstart.py prepare
konfai TRAIN -y --cpu 1 --config Config.yml
konfai PREDICTION -y --cpu 1 --config Prediction.yml --models "$(python quickstart.py checkpoint)"
konfai EVALUATION -y --cpu 1 --config Evaluation.yml
python quickstart.py verify
```

Twenty epochs take about seven seconds on one core and reach a Dice of about 0.98 on
every case, the held-out one included. `quickstart.py checkpoint` prints the one dated
file `BEST` kept (`resume_latest.pt` is a training continuation, `crash_*.pt` an
exceptional save, and several `--models` paths run an ensemble). `verify` reads the four
predictions, checks their labels and physical geometry against `CT` and `SEG`, and
recomputes the Dice values `Metric_TRAIN.json` reports.

A run writes its resolved defaults back into the YAML files: copy the directory again
for another fresh run. Three keys are pinned to `None` on purpose, `patch_transforms` on
every group and the two `*_reduction_transforms` on the output: an absent key defaults
to a `Normalize` to `[-1, 1]`.

To adapt the three configs together:

| Change | Training | Prediction | Evaluation |
|---|---|---|---|
| Run name | `Trainer.train_name` | `Predictor.train_name` | `Evaluator.train_name` and the predictions folder |
| Cases | `Dataset.dataset_filenames` | the same CT folder | the reference folder plus the predictions folder |
| CT values | `Standardize` under `CT` | the same preprocessing | none: label maps are compared |
| Classes | `nb_class`, `Dice.labels` | `nb_class` | `Dice.labels` |
| Held-out cases | `validation` names | the cases to predict | a `subset` for held-out scoring |

`verify` knows these four case names; give it yours. The larger
[Segmentation](../README.md) example trains 41 classes on real pelvis CT.
