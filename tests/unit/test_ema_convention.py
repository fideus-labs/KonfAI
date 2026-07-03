from types import SimpleNamespace

import pytest
import torch
from konfai.trainer import Trainer


def test_avg_fn_follows_standard_ema_convention() -> None:
    stub = SimpleNamespace(ema_decay=0.9)
    averaged = torch.tensor(1.0)
    model = torch.tensor(0.0)

    result = Trainer._avg_fn(stub, averaged, model, 0)

    assert result.item() == pytest.approx(0.9)


def test_avg_fn_high_decay_keeps_running_average_dominant() -> None:
    stub = SimpleNamespace(ema_decay=0.999)
    averaged = torch.tensor(10.0)
    model = torch.tensor(0.0)

    result = Trainer._avg_fn(stub, averaged, model, 5)

    assert result.item() == pytest.approx(9.99)
