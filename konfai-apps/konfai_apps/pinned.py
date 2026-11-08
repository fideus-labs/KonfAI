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

"""What an app that runs its authors' tool needs to fetch it: the authors' repository at a commit, a release archive
or a file checked by its hash, each once into a cache and reused; and the environment of a tool's subprocess on the
app's GPU. The app's engine (``konfai.network.external``) calls them.
"""

import contextlib
import hashlib
import os
import shutil
import subprocess  # nosec B404
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path


def cache_dir() -> Path:
    """Where the tools' code, binaries and environments live once fetched: ``$KONFAI_TOOLS_CACHE``, or
    ``~/.cache/konfai/tools``."""
    return Path(os.environ.get("KONFAI_TOOLS_CACHE", "~/.cache/konfai/tools")).expanduser().resolve()


@contextlib.contextmanager
def _staged(target: Path) -> Iterator[Path]:
    """A path of this process's own to build ``target`` in, moved into place at once when the block ends: a directory
    another process put there first is kept, a file there is replaced, and an attempt that fails leaves nothing
    behind."""
    part = target.with_name(f".{target.name}.{os.getpid()}.part")

    def clear() -> None:
        if part.is_dir():
            shutil.rmtree(part, ignore_errors=True)
        part.unlink(missing_ok=True)

    clear()  # what an earlier attempt of the same process id left
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield part
        try:
            part.replace(target)
        except OSError:
            if not target.exists():  # not another process's copy already in place
                raise
    finally:
        clear()


def pinned_source(name: str, url: str, commit: str) -> Path:
    """The authors' repository at ``commit``, cloned once into the cache: an app runs the code it was measured with,
    never a moving branch."""
    target = cache_dir() / f"{name}-{commit[:12]}"
    if not target.exists():
        with _staged(target) as part:
            subprocess.run(["git", "clone", "--quiet", url, str(part)], check=True)  # nosec B603 B607
            subprocess.run(["git", "-C", str(part), "checkout", "--quiet", commit], check=True)  # nosec B603 B607
    return target


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url: str, target: Path, sha256: str) -> Path:
    """``url`` downloaded once to ``target`` and checked against its hash, as is a file already there: another release
    at the same path, or a broken download, is fetched again."""
    if not target.exists() or _sha256(target) != sha256:
        with _staged(target) as part:
            urllib.request.urlretrieve(url, part)  # nosec B310 - the authors' release, checked below
            if _sha256(part) != sha256:
                raise RuntimeError(f"{url} does not match its recorded hash")
    return target


def pinned_archive(name: str, url: str, sha256: str) -> Path:
    """A release archive at ``url``, downloaded once into the cache, checked against ``sha256`` and unpacked: the
    binaries an app was measured with."""
    target = cache_dir() / f"{name}-{sha256[:12]}"
    if not target.exists():
        with _staged(target) as part:
            archive = fetch(url, part / "archive", sha256)
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as packed:
                    for entry in packed.infolist():  # zipfile drops the modes: the binaries must stay executable
                        written = packed.extract(entry, part)  # nosec B202 - our own pinned archive
                        if mode := (entry.external_attr >> 16) & 0o777:
                            os.chmod(written, mode)
            else:
                shutil.unpack_archive(str(archive), str(part), "gztar")
            archive.unlink()
    return target


def gpu_environment(device: int) -> dict[str, str]:
    """The environment of a tool's subprocess on the CUDA device ``device`` of this process (-1: none): the physical
    GPU behind that index, since the subprocess reads CUDA_VISIBLE_DEVICES afresh."""
    from konfai import cuda_visible_devices

    visible = str(cuda_visible_devices()[device]) if device >= 0 else ""
    return {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
