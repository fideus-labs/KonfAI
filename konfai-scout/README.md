<p align="center">
  <img src="assets/konfai-scout-logo.png" width="520" alt="KonfAI Scout">
</p>

<p align="center"><b>An agent that looks at your segmentation, registration or synthesis the way you would, and tells you where it is wrong.</b></p>

---

A score says how good a result is. Scout says where and why. It reads one case slice by slice, zooms on what looks
off, measures the suspect region with KonfAI Eval, and writes a short report: each defect with its place, its type,
its severity and the views that show it.

## Install

```bash
pip install "konfai-scout[anthropic]"     # Claude, with an Anthropic API key
```

Other models: `[openai]` for GPT or a local model behind an OpenAI-compatible server (vLLM, Ollama),
`[claude-code]` or `[codex]` to use your Claude Code or ChatGPT login without an API key.

## Inspect a case

```bash
konfai-scout inspect --task segmentation --name case01 \
    --input image=ct.nii.gz --input prediction=seg.nii.gz --labels 1=kidney \
    --out reports/case01 --provider anthropic --allow-cloud-images
```

A few minutes later, `reports/case01/report.md` opens with the verdict and the findings:

```markdown
**Verdict: defective** (confidence 0.88, by model)

The left kidney label stops in a flat cut at z≈747 mm. The kidney's upper half, visible up to z≈790, has no label
(o001, major). Small hilar holes are left out (o002, minor). The right kidney otherwise follows the image.

| Finding | Defect             | Kind      | Severity | Confidence  | Where                              |
|---------|--------------------|-----------|----------|-------------|------------------------------------|
| o001    | under_segmentation | suspected | major    | 0.90 (firm) | (98, 185, 771) mm, 75 x 70 x 48 mm |
| o002    | holes              | suspected | minor    | 0.60        | (-66, 188, 757) mm, 17 x 11 x 44 mm |
```

Each finding then has its own section, with the views that show it (saved beside the report) and the measurements
it cites. A *firm* finding is one Scout stands behind; the others are doubts to check. `report.json` holds the same
for a program.

The other tasks take their own inputs:

```bash
konfai-scout inspect --task registration --name pair07 \
    --input fixed=fixed.nii.gz --input moving=moving.nii.gz --input field=dvf.nii.gz \
    --out reports/pair07 --provider anthropic --allow-cloud-images

konfai-scout inspect --task synthesis --name ct01 \
    --input source=mr.nii.gz --input synthetic=sct.nii.gz \
    --out reports/ct01 --provider anthropic --allow-cloud-images
```

## With or without a reference

- **Without one**, Scout judges the result against the image: a label the image's edges do not support, a structure
  that is not in the source. Name each label with what it holds (`--labels "1=liver without its vessels"`).
- **With one** (`--input truth=gt.nii.gz`, `target=` for a synthesis, labels or landmarks for a registration),
  every disagreement is checked and measured against it, and an error can be *confirmed*.

## Useful options

- `--depth quick|standard|deep`: 10, 25 or 40 tool calls; `deep` adds a blind second reading of each error.
- `--max-usd`, `--max-seconds`: stop an inspection at a cost or a time.
- `--allow-cloud-images`: required to send images of the case to a cloud model. A local server (`--base-url
  http://localhost:11434/v1`) needs none.
- `--provider mock`: the whole loop with a heuristic model, no API at all.

## From Python

```python
from konfai_scout import InspectionCase, ScoutConfig, inspect

case = InspectionCase("segmentation", "case01", {"image": "ct.nii.gz", "prediction": "seg.nii.gz"}, labels={1: "kidney"})
report = inspect(case, ScoutConfig(provider="anthropic", allow_cloud_images=True), "reports/case01")
print(report.conclusion.verdict, [(o.defect, o.severity, o.box_mm) for o in report.observations])
```

## Learn more

- [How it works](https://github.com/fideus-labs/KonfAI/blob/main/konfai-scout/docs/how-it-works.md): the inspection, the inputs of each task, the model's tools and views, the
  budget, the providers and privacy.
- [The report](https://github.com/fideus-labs/KonfAI/blob/main/konfai-scout/docs/report.md): every field, and `konfai-scout export --level regions|counts` to share a report
  without the case's name or images.
- [Integrations](https://github.com/fideus-labs/KonfAI/blob/main/konfai-scout/docs/integrations.md): KonfAI Eval, konfai-mcp, KonfAI Studio and FidOpt.

Scout is part of the [KonfAI](https://github.com/fideus-labs/KonfAI) family and reads, warps and measures every
volume through KonfAI. From a checkout: `pip install -e ".[dev]"` then `pip install -e "./konfai-scout[dev]"`, and
`pytest konfai-scout/tests` (no API key needed).

Apache-2.0. Copyright (c) 2026 Fideus Labs LLC.
