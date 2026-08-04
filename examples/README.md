# Examples

Every example here is a notebook you can **open and run top to bottom**. Each one downloads (or
generates) its own data, runs the real KonfAI commands, and ends by showing you the result. Nothing
is hidden behind a flag you have to flip.

They all work from a fresh environment, including **Google Colab** — click the badge in any example's
README, then *Runtime > Run all*.

## Start here

These three are the framework itself: a YAML config, the `konfai` CLI, and nothing else.

| Example | What you get | Time on a GPU |
|---|---|---|
| [`Registration`](Registration/) | Train a `VoxelMorph` to align real pelvis CT slices, and measure how much of a *known* deformation it recovered. | ~3 min |
| [`Segmentation`](Segmentation/) | Train a UNet on five pelvis CT cases, predict the labels, score them with Dice. | ~7 min |
| [`Synthesis`](Synthesis/) | Turn an MR volume into a synthetic CT, scored with MAE / PSNR / SSIM inside the body mask. | ~7 min |

`Registration` is the shortest way to see the whole `TRAIN -> PREDICTION -> EVALUATION` loop.
`Segmentation` is the best template to copy for your own data. `Synthesis` shows the richer patterns:
a custom Python model, a perceptual loss, test-time augmentation, and an optional GAN variant.

Their training runs are **deliberately short** — enough to see the pipeline work end to end, not
enough to produce a usable model. Each README says what score to expect and which knob to raise.

## Then: run a published model

No training, no config to write. Each of these is a KonfAI **app**: a model published on the Hugging
Face Hub behind a single command. The model downloads on first use, so allow a few hundred MB and
prefer a GPU.

| Example | Command | Task |
|---|---|---|
| [`TotalSegmentator`](TotalSegmentator/) | `totalsegmentator-konfai segment total` | whole-body CT segmentation |
| [`MRSegmentator`](MRSegmentator/) | `mrsegmentator-konfai segment` | multi-organ MR segmentation |
| [`ImpactSeg`](ImpactSeg/) | `impact-seg-konfai segment body` | multimodal / multi-organ segmentation |
| [`ImpactSynth`](ImpactSynth/) | `impact-synth-konfai synthesize MR` | synthetic CT from MR or CBCT |
| [`ImpactReg`](ImpactReg/) | `impact-reg-konfai register FireANTs_SyN` | register two real patients, scored on their reference labels |

`ImpactReg` is the most instructive of the five, and the only fully unsupervised task in this folder:
it aligns **two different patients** — a real anatomical difference, with no ground-truth field to
recover — and scores the result by propagating one patient's 41-label reference through the recovered
displacement field and measuring its Dice against the other's.

## Demo data

The public demo dataset lives on the Hugging Face Hub at
[`VBoussot/konfai-demo`](https://huggingface.co/datasets/VBoussot/konfai-demo) and provides a
`Segmentation/` subset (pelvis CT with a 41-label reference) and a `Synthesis/` subset (paired MR /
CT / body mask). Every notebook fetches what it needs, and the Hub caches it after the first run —
`Registration` and `ImpactReg` both build on the pelvis CT.

## Running from the command line instead

Every notebook only calls the CLI, so you can do the same by hand. From an example directory:

```bash
konfai TRAIN      -y --gpu 0 --config Config.yml
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/<train_name>/<checkpoint>.pt
konfai EVALUATION -y          --config Evaluation.yml
```

Each run writes a workspace keyed by `train_name`: `Checkpoints/`, `Statistics/` (TensorBoard logs
and the resolved config), `Predictions/`, and `Evaluations/` (the metric JSON).

## Where to go next

- adapt one of the three configs to your own dataset — each example's `README.md` lists the fields to
  change first, in order;
- package a mature workflow as an app of your own — see [`apps/`](../apps/);
- drive KonfAI from an LLM agent — see [`konfai-mcp/`](../konfai-mcp/).
