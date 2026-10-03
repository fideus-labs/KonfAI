# Run a published app

A KonfAI app packages a trained model, its weights and its configuration. This guide runs
ImpactSeg on one CT, MR or CBCT image and writes a segmentation with 11 labels.
For a fixed/moving image pair, use {doc}`registration`.

## Install and run

Use Python 3.11 or newer in your working environment. See {doc}`../getting-started/installation`
for environment and GPU setup.

```bash
python -m pip install "konfai[imaging]" konfai-apps
konfai-apps infer VBoussot/ImpactSeg:body -i ./scan.mha -o ./Output --gpu 0
```

Replace `./scan.mha` with your image path. `--gpu 0` selects the first GPU;
use `--cpu 1` instead to run on the CPU. The first run downloads the app's weights
and installs any missing app requirements, so it takes longer than later runs.

An app imports Python code and installs its requirements by default. Use a source
you trust. Set `KONFAI_APPS_INSTALL_REQUIREMENTS=0` to manage requirements yourself.

## Check the result

For this app, open `Output/ImpactSeg-Body/Output/P000/Output.mha`. The command
prints the files written for each case and keeps the run logs. Open the label image
alongside your input in an image viewer, such as 3D Slicer: check that they align and
that the labels identify the expected anatomy. A successful command does not establish
accuracy on your data.

For an example that downloads its own input and displays the result, open the
[ImpactSeg notebook](https://github.com/fideus-labs/KonfAI/blob/main/examples/ImpactSeg/ImpactSeg_demo.ipynb).
The {doc}`gallery <../examples/visual-gallery>` shows other completed app runs.

## Choose another app

| App reference | Input | Output |
| --- | --- | --- |
| `VBoussot/ImpactSeg:body` | CT, MR or CBCT | 11-label segmentation |
| `VBoussot/ImpactSynth:MR` | MR | Synthetic CT |
| `VBoussot/ImpactSynth:CBCT` | CBCT | Synthetic CT |
| `VBoussot/MRSegmentator-KonfAI:MRSegmentator` | MR | 40-label segmentation |
| `VBoussot/TotalSegmentator-KonfAI:total` | CT | 117-label segmentation |

Replace the app reference in the command above. App-specific notebooks are listed in
{doc}`../examples/index`; each bundle's README describes its variants and benchmarks.

A reference is `owner/repository:app` or a local app folder. To select an exact revision,
use `owner/repository@revision:app`. Without a revision, the installed release tries its
matching version tag, then `main`. `konfai-apps download APP` fetches an app and its weights
before a run; replace `APP` with the reference you chose.

## Evaluate or estimate uncertainty

These operations depend on the configurations the app provides:

| Operation | Input it needs | Result |
| --- | --- | --- |
| `infer` | Original image | Prediction |
| `eval` | Prediction and matching reference | Metrics |
| `uncertainty` | Saved multi-channel inference stack | Uncertainty map |
| `pipeline` | Original image; reference for evaluation | Inference and supported follow-up stages |

For example, evaluate an ImpactSynth prediction against its paired CT:

```bash
konfai-apps eval VBoussot/ImpactSynth:MR -i ./prediction.mha --gt ./reference_ct.mha
```

Replace both paths with aligned images from the same case. To retain the copies needed
for uncertainty, pass `-uncertainty` to `infer`. The `uncertainty` command consumes that
saved stack, not the original input image. See {doc}`../reference/cli` for the options.

## Run on a server or from Slicer

A remote run uploads the inputs, streams logs and downloads the result. Install
`"konfai-apps[server]"` on the server and follow the {doc}`server reference <../reference/app-server-api>`
for its app allowlist, authentication and deployment.

[SlicerKonfAI](https://github.com/vboussot/SlicerKonfAI) runs apps from 3D Slicer,
locally or on a server, and loads their outputs into the scene.

## The ecosystem around an app

| Tool | Use it to |
| --- | --- |
| KonfAI | Train, predict, evaluate or prepare data from a configuration |
| KonfAI Apps | Run and distribute a packaged workflow |
| KonfAI MCP | Operate those workflows through MCP tools |
| KonfAI Studio | Use those tools through a chat interface |

To adapt weights or distribute your own model, continue with {doc}`packaging-apps`.
