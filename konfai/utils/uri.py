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

"""Where a dataset root lives: the local filesystem, or a URI some other filesystem serves.

KonfAI walks a root before it reads a byte of it. Everything here answers that walk for both kinds
of root, or raises naming the one it could not reach: an unreachable root that answers "empty" is a
run that does nothing and reports success.

Remote roots are READ-ONLY: publication is a rename, which object stores do not have.

Optional dependency: the filesystem for the scheme (``pip install konfai[s3]`` for ``s3://``);
``fsspec`` itself already ships with the OME-Zarr extra.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from konfai.utils.errors import DatasetManagerError

#: A scheme of two characters or more: a Windows drive letter (``C:\\Data``, ``C://``) is a path.
_URI = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]+)://")

#: The pip extra carrying each scheme's fsspec implementation. A scheme absent from here is named
#: in the refusal without an install hint.
_SCHEME_EXTRAS = {"s3": "s3", "s3a": "s3"}


def is_uri(path: str | Path) -> bool:
    """Whether ``path`` names a remote root."""
    return bool(_URI.match(str(path)))


def scheme(path: str | Path) -> str:
    """The URI scheme of ``path``, or ``''`` for a local path."""
    match = _URI.match(str(path))
    return match.group(1) if match else ""


def split_scheme(path: str) -> tuple[str, str]:
    """``path`` as its ``scheme://`` prefix and what follows it; ``('', path)`` for a local path."""
    match = _URI.match(path)
    return (match.group(0), path[match.end() :]) if match else ("", path)


def normalize(path: str | Path) -> str:
    """``path`` as KonfAI stores a root: a local path through ``Path.as_posix``, a URI untouched.

    ``Path('s3://bucket/key').as_posix()`` is ``'s3:/bucket/key'``: one slash, which no filesystem
    resolves and none rejects.
    """
    return str(path) if is_uri(path) else Path(path).as_posix()


def join(root: str, *parts: str) -> str:
    """``root`` extended by ``parts``, on the separator both kinds of root use."""
    return "/".join([root.rstrip("/"), *(str(part).strip("/") for part in parts if part)])


def filesystem(path: str | Path) -> Any:
    """The fsspec filesystem serving ``path``.

    Opened with fsspec's own configuration and nothing else: it merges ``FSSPEC_<PROTO>_<KEY>`` from
    the environment and ``~/.config/fsspec/*.json`` into every filesystem it builds, so a public S3
    bucket is ``FSSPEC_S3_ANON=true`` and a private one the ``AWS_*`` variables botocore already
    reads. KonfAI declares none of that itself.
    """
    protocol = scheme(path)
    try:
        import fsspec
    except ImportError as exc:
        raise DatasetManagerError(
            f"Reading a dataset from '{protocol}://' needs fsspec.",
            "Install it with: pip install konfai[omezarr]",
        ) from exc
    # Routing first, arguments second: `get_filesystem_class` answers whether the protocol reaches an
    # implementation at all (ValueError if nothing registers it, ImportError naming the package if
    # something does), so what `filesystem` then refuses is the configuration and nothing else.
    try:
        fsspec.get_filesystem_class(protocol)
    except ValueError as exc:
        raise DatasetManagerError(
            f"No filesystem is registered for '{protocol}://' ({exc}).",
            f"Install the fsspec implementation for {protocol}, or check the scheme for a typo.",
        ) from exc
    except ImportError as exc:
        extra = _SCHEME_EXTRAS.get(protocol)
        raise DatasetManagerError(
            f"No filesystem is installed for '{protocol}://' ({exc}).",
            f"Install it with: pip install konfai[{extra}]" if extra else str(exc),
        ) from exc
    try:
        return fsspec.filesystem(protocol)
    except (TypeError, ValueError) as exc:
        raise DatasetManagerError(
            f"'{protocol}://' refused its fsspec configuration: {exc}.",
            "Check the FSSPEC_" + protocol.upper() + "_* variables and ~/.config/fsspec/.",
        ) from exc


def _info(path: str | Path, action: str) -> dict[str, Any] | None:
    """What the remote filesystem knows of ``path``, ``None`` when it holds nothing there.

    Asked through ``info`` and not through ``exists``/``isdir``: fsspec answers those ``False`` on
    any failure, so a denied bucket and a missing key read alike. A filesystem that cannot be
    reached RAISES; only one that answers gets to say no.
    """
    fs = filesystem(path)
    try:
        return dict(fs.info(str(path)))
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise _unreachable(path, action, exc) from exc


def exists(path: str | Path) -> bool:
    """Whether ``path`` is there, asked of whichever filesystem owns it."""
    return _info(path, "probed") is not None if is_uri(path) else os.path.exists(path)


def is_dir(path: str | Path) -> bool:
    """Whether ``path`` is a directory (a key prefix, on an object store)."""
    if not is_uri(path):
        return os.path.isdir(path)
    info = _info(path, "probed")
    return info is not None and info.get("type") == "directory"


def list_names(path: str | Path) -> list[str]:
    """The names of ``path``'s direct children, sorted; ``[]`` when ``path`` is not a directory.

    Never ``[]`` because the listing FAILED: an expired credential, a bucket that denies listing and
    a typo in a URI all raise here, each of them reading as an empty cohort otherwise.
    """
    if not is_uri(path):
        return sorted(os.listdir(path)) if os.path.isdir(path) else []
    if not is_dir(path):
        return []
    fs, target = filesystem(path), str(path)
    try:
        children = fs.ls(target, detail=False)
    except Exception as exc:
        raise _unreachable(path, "listed", exc) from exc
    # An implementation may or may not echo the protocol back, and a directory of its own is one of
    # the children it lists: compare on the key, and take the last component of it.
    return sorted({_key(child).rsplit("/", 1)[-1] for child in children if _key(child) != _key(target)})


def _key(path: str) -> str:
    """``path`` as the filesystem keys it: no protocol, no trailing separator."""
    return path.rstrip("/").split("://", 1)[-1]


def refuse_remote_walk(path: str | Path, file_format: str) -> None:
    """Refuse a recursive walk of a remote root, which only the store backends can serve."""
    if is_uri(path):
        raise DatasetManagerError(
            f"'{path}' is a remote root, which the '{file_format}' backend cannot walk.",
            "Only ':omezarr' reads a remote dataset. Declare the root as ':omezarr', or copy it locally first.",
        )


def refuse_write(path: str | Path) -> None:
    """Refuse a write to a remote root, which is a read-only source here."""
    if is_uri(path):
        raise DatasetManagerError(
            f"'{path}' is a remote root, and KonfAI writes only to local ones.",
            "A remote store is where a cohort is read from. Point the Write (or the output"
            " dataset) at a local path and upload the result separately.",
        )


def _unreachable(path: str | Path, action: str, exc: Exception) -> DatasetManagerError:
    return DatasetManagerError(
        f"'{path}' could not be {action}: {type(exc).__name__}: {exc}",
        "Check the URI, the credentials fsspec has for it (anonymous S3 access is"
        " FSSPEC_S3_ANON=true) and that the bucket allows listing. A remote root that cannot be"
        " reached is an error, never an empty cohort.",
    )
