# Schedulers

KonfAI has **two distinct scheduler families**, resolved by different loaders.
Don't confuse them.

## A. Criterion-weight schedulers

These schedule the **weight of a loss** over training, in the `schedulers:`
subtree of a criterion (`konfai/metric/schedulers.py`, base class `Scheduler`):

```yaml
CrossEntropyLoss:
  is_loss: true
  schedulers:
    Constant: { nb_step: 0, value: 1 }
```

| Name | Purpose | Config args (defaults) |
| --- | --- | --- |
| `Constant` | Fixed weight for all iterations. | `value=1` (+ `nb_step`) |
| `CosineAnnealing` | Cosine anneal from `start_value` to `eta_min` over `t_max`. | `start_value=1, eta_min=1e-5, t_max=100` (+ `nb_step`) |

```{note}
Each entry carries an `nb_step` (window width). Multiple schedulers can be
**chained** into consecutive iteration windows; an `nb_step: 0` (or `None`)
window is the terminal, always-on schedule. `Constant` is the one used across
tests and examples; `CosineAnnealing` is implemented but unexercised.
```

## B. Learning-rate schedulers

These schedule the **optimizer learning rate**, in the model's `schedulers:`
block. The loader searches **both** `torch.optim.lr_scheduler` **and**
`konfai.metric.schedulers`, so you can use any torch scheduler (`StepLR`,
`ReduceLROnPlateau`, `CosineAnnealingLR`, …) **plus** the two KonfAI additions:

```yaml
schedulers:
  StepLR: { step_size: 20, gamma: 0.5 }     # any torch LR scheduler works
```

| Name | Purpose | Config args (defaults) |
| --- | --- | --- |
| `Warmup` | Linear LR warmup wrapper (`LambdaLR`). | `warmup_steps=10, last_epoch=-1` (+ `nb_step`) |
| `PolyLRScheduler` | nnU-Net-style polynomial LR decay `lr = initial_lr·(1 − step/max_steps)^exponent`. | `initial_lr` (**required**), `max_steps` (**required**), `exponent=0.9` (+ `nb_step`) |

```{note}
`PolyLRScheduler` is tested and stable; `Warmup` is implemented but not covered
by tests/examples. The `optimizer` itself is injected by the framework — you do
not write it under `schedulers:`.
```

## See also

- {doc}`losses-metrics` — where the weight schedulers live
- {doc}`../../config_guide/training` — the `optimizer:` / `schedulers:` blocks
