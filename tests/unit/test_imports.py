# SPDX-License-Identifier: Apache-2.0
"""Smoke tests confirming the konfai package imports cleanly with no optional deps."""

import subprocess
import sys


def test_import_konfai_succeeds() -> None:
    import konfai  # noqa: F401


def test_version_attribute_exists() -> None:
    import konfai

    assert hasattr(konfai, "__version__"), "konfai must expose __version__"
    assert isinstance(konfai.__version__, str)
    assert konfai.__version__  # non-empty


def test_konfai_utils_config_imports_without_simpleitk() -> None:
    """konfai.utils.config must be importable even if SimpleITK is absent."""
    script = """
import builtins

real_import = builtins.__import__

def import_without_simpleitk(name, *args, **kwargs):
    if name == "SimpleITK":
        raise ImportError("SimpleITK unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_simpleitk
import konfai.utils.config
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_transform_and_blocks_import_without_simpleitk() -> None:
    """Modules using SimpleITK guard it at point-of-use, so import must succeed without it."""
    script = """
import builtins

real_import = builtins.__import__

def import_without_simpleitk(name, *args, **kwargs):
    if name == "SimpleITK":
        raise ImportError("SimpleITK unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_simpleitk
import konfai.data.transform
import konfai.network.blocks

assert konfai.data.transform.sitk is None
assert konfai.network.blocks.sitk is None
try:
    konfai.data.transform._require_simpleitk()
except Exception as exc:
    assert "pip install konfai[itk]" in str(exc), str(exc)
else:
    raise AssertionError("_require_simpleitk must raise without SimpleITK")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
