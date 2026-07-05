# SPDX-License-Identifier: Apache-2.0
"""Smoke tests verifying package-level import contracts and public API surface."""

import konfai
import pytest
from konfai.utils.errors import KonfAIError, TransformError


def test_package_importable() -> None:
    import konfai

    assert isinstance(konfai.__version__, str)
    assert konfai.__version__


def test_config_module_importable() -> None:
    from konfai.utils.config import Config, apply_config, config  # noqa: F401


def test_errors_module_importable() -> None:
    from konfai.utils.errors import ConfigError, KonfAIError, TrainerError

    assert issubclass(KonfAIError, Exception)
    assert issubclass(ConfigError, Exception)
    assert issubclass(TrainerError, Exception)


def test_local_vram_query_requires_monitoring_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(konfai, "_PYNVML_AVAILABLE", False)

    with pytest.raises(KonfAIError, match="nvidia-ml-py"):
        konfai.get_vram([0])


def test_itk_helper_requires_simpleitk(monkeypatch: pytest.MonkeyPatch) -> None:
    import konfai.utils.ITK as itk_module

    monkeypatch.setattr(itk_module, "sitk", None)

    with pytest.raises(TransformError, match="SimpleITK"):
        itk_module.resample_resize(None)


def test_main_module_importable() -> None:
    import konfai.main

    assert callable(konfai.main.main)
