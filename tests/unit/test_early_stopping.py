import pytest

from konfai.trainer import EarlyStopping, EarlyStoppingBase
from konfai.utils.errors import TrainerError


def test_early_stopping_base_starts_running_and_can_be_stopped() -> None:
    stopper = EarlyStoppingBase()

    assert stopper.is_stopped() is False

    stopper.stop()

    assert stopper.is_stopped() is True


def test_early_stopping_inherits_stop_from_base() -> None:
    stopper = EarlyStopping(monitor=[], patience=10)

    assert stopper.is_stopped() is False

    stopper.stop()

    assert stopper.is_stopped() is True


def test_early_stopping_triggers_after_patience_without_improvement() -> None:
    stopper = EarlyStopping(monitor=[], patience=2, mode="min")

    assert stopper(1.0) is False  # first score sets the baseline
    assert stopper(1.0) is False  # no improvement (counter = 1)
    assert stopper(1.0) is True  # no improvement (counter = 2 >= patience)
    assert stopper.is_stopped() is True


def test_get_score_reports_missing_metric_and_available_keys() -> None:
    stopper = EarlyStopping(monitor=["val_loss"], patience=3)

    with pytest.raises(TrainerError) as exc_info:
        stopper.get_score({"train_loss": 1.0, "dice": 0.5})

    message = str(exc_info.value)
    assert "val_loss" in message  # the missing monitored metric is named
    assert "train_loss" in message  # the keys actually available are listed
    assert "dice" in message
    assert "{}" not in message  # the placeholder is interpolated, not left raw
