# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from email import message_from_string
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parents[2]
# Only an sdist carries it: the metadata of the tree it was built from, git history included.
_PKG_INFO = Path(__file__).with_name("PKG-INFO")


def _release_version() -> str:
    if _PKG_INFO.exists():
        return message_from_string(_PKG_INFO.read_text())["Version"]
    from setuptools_scm import get_version

    return get_version(root=str(_ROOT), tag_regex=r"^v(?P<version>.*)$", local_scheme="no-local-version")


def _sibling(name: str, version: str) -> str:
    """The family ships in lockstep: at a release tag the sibling is pinned to the exact version.
    From a working tree (a ``.dev`` version) the pin is the closest release tag or newer, so
    ``pip install -e`` resolves against the core installed beside it; the publish workflow
    builds at the tag, where the pin is exact. A wheel built from an sdist keeps the pins the
    sdist recorded (see below)."""
    if ".dev" not in version:
        return f"{name}=={version}"
    try:
        from setuptools_scm import get_version

        floor = get_version(
            root=str(_ROOT),
            tag_regex=r"^v(?P<version>.*)$",
            version_scheme=lambda scm_version: str(scm_version.tag),
            local_scheme="no-local-version",
        )
    except Exception:  # no git history to read a tag from: the dev version's own floor
        floor = version
    return f"{name}>={floor}"


_version = _release_version()

# h5py: every preset writes its transform through konfai's ':itktransform' backend, which fills the
# parameters region by region rather than holding the field in float64. The backend requires it.
# konfai[monitoring] (nvidia-ml-py): register reads the free GPU memory with konfai's get_vram to size the
# tiles of a pair too large for the card; without it every pair would be handed to the preset whole.
# konfai[omezarr,dicom]: OME-Zarr stores and DICOM series go in and come out as they are, the README's own
# examples; without the readers the first such input stopped on an install hint inside the preset's child.
# A wheel built from the sdist (what `python -m build` does) has no git history to read the closest release
# from, and pinned '>= <its own dev version>', which no released core satisfies. The sdist computed the pins
# with that history: they are kept. An sdist that recorded none gets them computed, never a wheel without any.
_recorded = message_from_string(_PKG_INFO.read_text()).get_all("Requires-Dist") if _PKG_INFO.exists() else None
_requirements = _recorded or [
    _sibling("konfai[monitoring,omezarr,dicom]", _version),
    _sibling("konfai-apps", _version),
    "h5py",
]
setup(install_requires=_requirements)
