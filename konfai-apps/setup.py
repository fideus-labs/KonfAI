from email import message_from_string
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parents[1]


def _release_version() -> str:
    pkg_info = Path(__file__).with_name("PKG-INFO")
    if pkg_info.exists():
        return message_from_string(pkg_info.read_text())["Version"]
    from setuptools_scm import get_version

    return get_version(root=str(_ROOT), tag_regex=r"^v(?P<version>.*)$", local_scheme="no-local-version")


_version = _release_version()

setup(install_requires=[f"konfai=={_version}", "SimpleITK", "fastapi", "uvicorn", "python-multipart"])
