# Examples

KonfAI ships three low-level, YAML-driven examples under `examples/`. They are
the best starting point when you want to understand the framework before
building a reusable KonfAI App.

Segmentation and Synthesis are backed by public demo data on Hugging Face:

- `VBoussot/konfai-demo/Synthesis`
- `VBoussot/konfai-demo/Segmentation`

Registration instead generates eight small fixed/moving pairs locally, with a
known translation and no patient data. Segmentation and Synthesis also include
notebooks intended for a fresh environment, including Google Colab.

<figure class="kf-visual kf-visual--execution">
  <a class="kf-visual-frame" href="../_static/readme/execution-flow.svg" aria-label="Open the KonfAI execution-flow diagram at full resolution">
    <picture>
      <source media="(max-width: 640px)" srcset="../_static/readme/execution-flow-mobile.svg" width="720" height="1330">
      <img src="../_static/readme/execution-flow.svg" alt="Conceptual diagram showing medical data entering KonfAI through regional reads, passing through patch planning, transforms, model execution, and reconstruction, then leaving as medical datasets, applications, services, Slicer workflows, or agent-operated experiments." width="1200" height="500" fetchpriority="high" decoding="async">
    </picture>
  </a>
  <figcaption>
    <span class="kf-visual-copy">
      <strong>The shared execution path—not an example result.</strong>
      <span class="kf-visual-meta">All three tutorials use part of this conceptual storage → execution → delivery path; their measured outputs are documented on their own pages.</span>
    </span>
    <a class="kf-visual-inspect" href="../_static/readme/execution-flow.svg">Inspect the diagram <span aria-hidden="true">↗</span></a>
  </figcaption>
</figure>

```{toctree}
:maxdepth: 1

visual-gallery
segmentation
registration
synthesis
```

## Choosing an example

Start with **Segmentation** when you want the smallest conservative baseline:

- one input group (`CT`)
- one label-map target (`SEG`)
- built-in `UNet`
- training with `CrossEntropyLoss`
- final evaluation with Dice

Start with **Synthesis** when you want to understand more of KonfAI's
configuration model:

- custom local Python modules loaded through `classpath`
- paired image-to-image training
- masked evaluation
- shared prediction and evaluation configs
- a GAN variant with nested patching scopes

Start with **Registration** when you want to learn the two-input spatial
workflow:

- generated `FIXED` / `MOVING` image pairs with a known offset
- the built-in 2D diffeomorphic `VoxelMorph`
- a named warped-image output materialised as `MOVED.mha`
- MAE and MSE measured before and after registration

A good adoption pattern is:

1. get **Segmentation** to run once
2. use **Registration** when your model consumes ordered fixed/moving inputs
3. move to **Synthesis** when you need custom modules or more advanced workflow structure

## Working from the repository

**All example commands in this documentation assume you are running from the
example directory itself**, for example:

```bash
cd examples/Segmentation
```

or:

```bash
cd examples/Registration
```

or:

```bash
cd examples/Synthesis
```

That matters because the shipped YAML files refer to local modules and dataset
paths relative to the current working directory.

## Next steps

- {doc}`segmentation` — the smallest end-to-end run; start here
- {doc}`registration` — train, materialise, and evaluate a fixed/moving image workflow
- {ref}`gallery-registration` — inspect a separate real IMPACT-Reg App execution
- {doc}`../concepts/configuration` — understand the YAML the examples are built from
- {doc}`../usage/custom-models` — the step after Synthesis's local `classpath` modules
