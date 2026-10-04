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


"""The dataset: groups of named entries over one backend, read whole, by region or by statistics."""

from __future__ import annotations

import contextlib
import os
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from typing import TYPE_CHECKING

from konfai.utils import uri
from konfai.utils.dataset import statistics
from konfai.utils.dataset.abstract import AbstractFile as _AbstractFile
from konfai.utils.dataset.attribute import (
    Attribute,
    as_channel_first,
    data_to_image,
    data_to_transform,
)
from konfai.utils.dataset.backend import File as _File
from konfai.utils.dataset.backend import backend_for
from konfai.utils.dataset.dicom_file import DicomFile
from konfai.utils.dataset.h5 import H5File
from konfai.utils.dataset.itk_transform_file import ItkTransformFile
from konfai.utils.dataset.ome_zarr_file import OmeZarrFile
from konfai.utils.dataset.sitk_file import SitkFile
from konfai.utils.dataset.staging import _recover_orphaned_backup, is_staging_entry
from konfai.utils.dataset.statistics import (
    _finalize_running_statistics,
    _lerp_like_numpy,
    _order_statistics,
    _scan_rows,
    _statistics_chunk_length,
    _update_pieces,
    _update_running_extrema,
    _update_running_statistics,
    needs_moments,
)
from konfai.utils.errors import CaseReadError, DatasetManagerError, KonfAIError
from konfai.utils.utils import (
    STORE_FORMS,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
    is_dir_entry,
    is_file_entry,
    is_store_name,
    listed_volume_form,
    split_format_level,
    storage_form,
)

if TYPE_CHECKING:
    from konfai.utils.dataset.stream import DataStream

_T = TypeVar("_T")


#: How many of a local root's directories are probed for its store form, how many sub-directories of each
#: that hold something are opened, and how many files of those are read for the DICOM magic: a probe costs a
#: bounded number of listings and reads, whatever the cohort.
_PROBED = 16

_VOLUME_FILES = tuple(f".{extension}" for extension in SUPPORTED_EXTENSIONS)


def _local_store_form(root: str) -> str | None:
    """The store form (``omezarr`` / ``dicom``) most of a sample of ``root``'s directories hold, ``None``
    when that is plain files or when none of them holds a volume.

    The sample is spread evenly over the directories by name, hidden ones aside, and stops once a form holds
    a majority of it; one that holds no volume or cannot be read does not vote, and a tie goes to the form
    the first of them by name holds. A link is followed only once sampled, and one to a file or to nothing
    does not vote. When no sampled directory could be read at all, the permission error is raised: nothing
    was seen to decide on.
    """
    with os.scandir(root) as listing:
        names = sorted(
            entry.name
            for entry in listing
            if not entry.name.startswith(".") and (entry.is_symlink() or entry.is_dir(follow_symlinks=False))
        )
    count = min(_PROBED, len(names))
    votes: Counter[str] = Counter()
    denied: PermissionError | None = None
    read = False
    for index in range(count):
        try:
            form = _held_volume_form(os.path.join(root, names[index * len(names) // count]))
        except PermissionError as error:
            denied = denied or error
            continue
        except (FileNotFoundError, NotADirectoryError):
            continue
        read = True
        if form is not None:
            votes[form] += 1
            if votes[form] > count // 2:
                break
    if not votes:
        if denied is not None and not read:
            raise denied
        return None
    form = max(votes, key=votes.__getitem__)
    return None if form == "file" else form


def _held_volume_form(directory: str) -> str | None:
    """``omezarr`` / ``dicom`` for the first volume directory ``directory`` holds by name, ``file`` when it
    holds volume files only, ``None`` when it holds no volume.

    A sub-directory named as a store is told by its name; of the others, only the first ``_PROBED`` that
    hold something are looked into (a directory of cases holds hundreds), with at most ``_PROBED`` of their
    files read for the DICOM magic, and one that cannot be read is passed over. When nothing was found and
    one was passed over, its permission error is raised.
    """
    with os.scandir(directory) as listing:
        entries = sorted(listing, key=lambda entry: entry.name)
    holds_file = False
    opened = 0
    denied: PermissionError | None = None
    for entry in entries:
        if not is_dir_entry(entry):
            name = entry.name
            holds_file |= not name.startswith(".") and name.lower().endswith(_VOLUME_FILES) and is_file_entry(entry)
            continue
        if is_store_name(entry.name):
            return "omezarr"
        if opened == _PROBED or (entry.name.startswith(".") and is_staging_entry(entry.name)):
            continue  # a writer's staging directory holds another entry's files until it is moved in
        try:
            with os.scandir(entry.path) as listing:
                held = list(listing)
        except PermissionError as error:
            denied = denied or error
            continue
        except (FileNotFoundError, NotADirectoryError):
            continue
        opened += bool(held)
        volume = listed_volume_form(held, _PROBED)
        if volume is not None:
            return "dicom" if volume == "" else "omezarr"
    if holds_file:
        return "file"
    if denied is not None:
        raise denied
    return None


def _is_listed_name(name: str) -> bool:
    """Whether ``name`` is one component of a directory listing, which is how a root spells its cases."""
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name


class Dataset:
    """Filesystem or HDF5-backed dataset abstraction used across KonfAI."""

    # The backends are addressed as ``Dataset.<Backend>``: the names stay on the class.
    AbstractFile = _AbstractFile
    H5File = H5File
    SitkFile = SitkFile
    OmeZarrFile = OmeZarrFile
    DicomFile = DicomFile
    ItkTransformFile = ItkTransformFile
    File = _File

    def __init__(
        self,
        filename: str | Path,
        file_format: str,
        scale_factors: list[int] | str | None = None,
        downsample_method: str | None = None,
    ) -> None:
        base_format, self.level = split_format_level(file_format)
        normalized_format = base_format.lower().removeprefix(".")
        # Every spelling the walk accepts on disk for a store is a token here, the dotted one included.
        file_format = "omezarr" if f".{normalized_format}" in STORE_FORMS else normalized_format
        if file_format not in SUPPORTED_FORMATS:
            # Unchecked, the token reaches the SimpleITK writer, which may write a file no backend probes.
            raise DatasetManagerError(
                f"'{base_format}' is not a format KonfAI writes.",
                "Use one of: " + ", ".join(sorted(SUPPORTED_FORMATS)) + ".",
            )
        self.filename, self.is_directory = Dataset._normalize_path(filename, file_format)
        self.file_format = file_format
        # A store backend (OME-Zarr / Zarr / DICOM) is detected from disk; the token then only carries
        # the write format and the OME-Zarr pyramid level (``@N``).
        detected = Dataset._detect_directory_store_format(self.filename) if self.is_directory else None
        if detected is not None:
            self.file_format = detected
        # A pyramid asked of a format without levels is refused, never silently written as one level.
        if scale_factors and not backend_for(self.file_format).writes_pyramid:
            raise DatasetManagerError(
                f"A pyramid was asked of a '{self.file_format}' destination, which has no levels.",
                "Only ':omezarr' stores levels. Drop scale_factors, or write to ':omezarr'.",
            )
        self.scale_factors = scale_factors or None
        self.downsample_method = downsample_method
        self._names_cache: dict[str, list[str]] = {}
        self._infos_cache: dict[tuple[str, str], tuple[list[int], Attribute]] = {}
        #: A root seen once is not re-probed, and a case resolved once keeps its path until a write
        #: drops the caches (one round-trip each on a remote root).
        self._root_seen = False
        self._case_paths: dict[tuple[str, str], str] = {}
        #: The file an entry of a case resolved to, kept under the same rule.
        self._entry_paths: dict[str, str] = {}
        #: Facts a stage derived from an entry's pixels (a Crop's foreground box), keyed by
        #: ``(group, name)``: computed once per volume.
        self.case_facts: dict[tuple[str, str], dict[str, Any]] = {}

    def _file(self, filename: str, read: bool) -> _File:
        """One entry's backing file, opened as this dataset's root is."""
        return self.File(filename, read, self.file_format, self.level)

    @property
    def _backend(self) -> type[_AbstractFile]:
        """The class serving this dataset's format."""
        return backend_for(self.file_format)

    @staticmethod
    def _normalize_path(filename: str | Path, file_format: str) -> tuple[str, bool]:
        # A single store is one file, every other backend a directory of cases: only the latter gets the
        # trailing slash that marks ``is_directory``. The separator stays forward on every OS.
        path = uri.normalize(filename)
        if not backend_for(file_format).single_store and not path.endswith("/"):
            path += "/"
        return path, path.endswith("/")

    def rebase(self, prefix: Path) -> None:
        """Prepend ``prefix`` to this dataset's path, re-deriving ``is_directory`` from the format.

        A rebased root is an output root: a remote root is refused.
        """
        uri.refuse_write(self.filename)
        self.filename, self.is_directory = Dataset._normalize_path(prefix / self.filename, self.file_format)

    @staticmethod
    def _detect_directory_store_format(root: str) -> str | None:
        """The store backend of a directory dataset, from disk (``omezarr`` / ``dicom``), or ``None`` for
        plain per-file volumes."""
        if not uri.is_dir(root):
            return None
        if uri.is_uri(root):
            # A remote store is told by its name, never probed as a path.
            names = Dataset._first_case_entries(root)
            return "omezarr" if any(is_store_name(name.name) for name in names) else None
        return _local_store_form(root)

    @staticmethod
    def _first_case_entries(root: str) -> list[Path]:
        """What remote ``root``'s first case directory by name holds, empty when it has none."""
        cases = (name for name in uri.list_names(root))
        case = next((name for name in cases if uri.is_dir(uri.join(root, name))), None)
        return [] if case is None else [Path(name) for name in uri.list_names(uri.join(root, case))]

    @property
    def store_root(self) -> str:
        """Where the store lives, as text (a URI has no ``Path``): its root directory, or the ``.h5``
        file for a single-file store. :attr:`path_on_disk` is the local-only view."""
        root = self.filename
        suffix = self._backend.case_file_suffix
        if self._backend.single_store and suffix and not root.endswith(suffix):
            return f"{root}{suffix}"
        return root

    @property
    def path_on_disk(self) -> Path:
        """:attr:`store_root` as a path. Local roots only: a URI has no filesystem path."""
        return Path(self.store_root)

    def exists_on_disk(self) -> bool:
        """Whether the store is there. A remote root that cannot be reached raises."""
        return uri.exists(self.store_root)

    def concurrent_write_safe(self) -> bool:
        """Whether writes to different entries land in disjoint files, so a background writer may
        flush one entry while another thread writes elsewhere (the backend's own declaration)."""
        return self._backend.concurrent_write_safe

    def _forget_paths(self) -> None:
        """Drop what this dataset memoised of its files: a write, before it starts and once it has published,
        may change which file an entry resolves to."""
        self._names_cache.clear()
        self._infos_cache.clear()
        self._case_paths.clear()
        self._entry_paths.clear()
        self.case_facts.clear()

    def _write_target(self, group: str, name: str) -> tuple[_File, str]:
        """The file a ``(group, name)`` write lands in and the entry name inside it, caches dropped.

        A directory dataset routes any sub-directory prefix of ``group`` into the file path (one file
        per case); a single store keeps one file and a ``group/name`` entry.
        """
        uri.refuse_write(self.filename)
        self._forget_paths()
        if self.is_directory:
            os.makedirs(self.filename, exist_ok=True)
            s_group = group.split("/")
            if len(s_group) > 1:
                name = f"{'/'.join(s_group[:-1])}/{name}"
                group = s_group[-1]
            return (
                self.File(
                    f"{self.filename}{name}",
                    False,
                    self.file_format,
                    self.level,
                    self.scale_factors,
                    self.downsample_method,
                ),
                group,
            )
        return (
            self.File(self.filename, False, self.file_format, self.level, self.scale_factors, self.downsample_method),
            f"{group}/{name}",
        )

    def write(
        self,
        group: str,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        attributes = attributes if attributes is not None else Attribute()
        if isinstance(data, np.ndarray):
            data = as_channel_first(data, attributes)
        target, entry = self._write_target(group, name)
        with target as file:
            file.data_to_file(entry, data, attributes)
        self._forget_paths()

    def can_stream_data(self, attributes: Attribute) -> bool:
        """Whether ``open_data_stream`` can serve this dataset's write format: H5 and OME-Zarr always;
        ``mha``, ``nii`` and ``itktransform`` (a displacement field) with image geometry; every other
        format only writes whole volumes."""
        return self._backend.can_stream(self.file_format, attributes)

    def open_data_stream(
        self,
        group: str,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute | None = None,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        """Open one entry for incremental region writes.

        Returns ``None`` when the write format cannot serve region writes; the caller then assembles
        the volume and uses ``write``. The returned stream is a context manager: a clean exit
        finalizes the entry, an exception removes the partial one.

        ``region_shape`` is the extent the caller will write at a time, channels included; a store
        that chunks on it never pays a read-modify-write.
        """
        if attributes is None:
            attributes = Attribute()
        file, entry = self._write_target(group, name)
        backend = file.__enter__()
        try:
            stream = backend.open_data_stream(entry, shape, dtype, attributes, region_shape)
        except BaseException:
            file.__exit__(None, None, None)
            raise
        if stream is None:
            file.__exit__(None, None, None)
            return None
        stream._file = file
        stream._on_finish = self._forget_paths
        return stream

    def _case_path(self, sub_directory: str, name: str) -> str | None:
        """The file a directory dataset stores case ``name`` under, or ``None`` if absent on disk.

        The returned path omits the implicit ``.h5`` suffix h5 case files carry. A case found once
        is not probed again; an absent case stays a fresh question.
        """
        memo_key = (sub_directory, name)
        memoised = self._case_paths.get(memo_key)
        if memoised is not None:
            return memoised
        path = f"{self.filename}{sub_directory}{name}"
        on_disk = f"{path}{self._backend.case_file_suffix or ''}"
        if uri.exists(on_disk):
            self._case_paths[memo_key] = path
            return path
        if uri.is_uri(on_disk):
            return None  # no writer of a remote root, so no backup to recover
        # A writer killed mid-replacement leaves the previous version under its backup name.
        _recover_orphaned_backup(Path(on_disk))
        return path if os.path.exists(on_disk) else None

    def _holds(self, sub_directory: str, group: str, name: str) -> bool:
        """Whether the case file ``name`` under ``sub_directory`` holds ``group``."""
        path = self._case_path(sub_directory, name)
        if path is None:
            return False
        with self._file(path, True) as file:
            return file.is_exist(group)

    def _resolve_entry(self, groups: str, name: str, action: Callable[[_AbstractFile, str, str], _T]) -> _T:
        """Run ``action`` on the open file holding ``(groups, name)``.

        ``action`` receives the backend and the entry's coordinates inside that file: ``("", group)``
        on a directory dataset (one case per file, the entry keyed by the group path's last
        component), ``(groups, name)`` on a single-file dataset. Raises ``DatasetManagerError`` when
        the dataset or the entry is missing.
        """
        if not self._root_seen and not self.exists_on_disk():
            raise DatasetManagerError(
                f"The dataset '{self.filename}' does not exist.",
                "Check 'dataset_filenames' and the path it names.",
            )
        self._root_seen = True
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                path = self._case_path(sub_directory, name)
                if path is not None:
                    with self._unreadable_named(path, groups, name), self._file(path, True) as file:
                        file.case = name
                        file.resolved_paths = self._entry_paths
                        return action(file, "", groups.split("/")[-1])
            raise DatasetManagerError(
                f"The entry '{groups}/{name}' is not in '{self.filename}'.",
                "Check the groups_src spelling and that the case carries every group it names.",
            )
        with self._unreadable_named(self.filename, groups, name), self._file(self.filename, True) as file:
            # is_exist would take a wildcard group's '*' literally.
            exists = name in file.get_names(groups) if "*" in groups else file.is_exist(groups, name)
            if not exists:
                raise DatasetManagerError(
                    f"The entry '{groups}/{name}' is not in '{self.filename}'.",
                    "Check the groups_src spelling and that the case carries every group it names.",
                )
            return action(file, groups, name)

    @contextlib.contextmanager
    def _unreadable_named(self, where: str, groups: str, name: str) -> Iterator[None]:
        """Raise what the backend's library raises on an entry it cannot read as a ``CaseReadError``
        naming the case, the entry and the file."""
        try:
            yield
        except self._backend.read_errors as error:
            detail = " ".join(str(error).split())
            raise CaseReadError(
                f"The '{groups}' entry of case '{name}' in '{where}' cannot be read: {type(error).__name__}: {detail}"
            ) from error

    def read_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self._resolve_entry(groups, name, lambda file, group, entry: file.file_to_data(group, entry))

    def read_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        return self._resolve_entry(
            groups, name, lambda file, group, entry: file.file_to_data_slice(group, entry, slices)
        )

    def read_granularity(self, groups: str, name: str) -> tuple[int, ...] | None:
        """The stored block reads of ``(groups, name)`` are served in, or ``None`` when a read costs
        exactly what it asks for."""
        with contextlib.suppress(Exception):
            # The entry's path inside the file: a single-file store (h5) keys it by its group as well.
            return self._resolve_entry(
                groups, name, lambda file, group, entry: file.read_granularity(f"{group}/{entry}" if group else entry)
            )
        return None

    def plan_region_reads(self, groups: str, name: str, windows: Sequence[tuple[slice, ...]]) -> None:
        """Declare the region reads about to happen on ``(groups, name)``, in order. A hint the
        backend may ignore."""
        with contextlib.suppress(DatasetManagerError):
            self._resolve_entry(groups, name, lambda file, _group, entry: file.plan_region_reads(entry, windows))

    def iter_data_blocks(self, groups: str, name: str) -> Callable[[], Iterator[np.ndarray]]:
        """A factory of passes over one entry, block by block along the first spatial axis, each
        block about ``_STATISTICS_CHUNK_ELEMENTS`` elements: what the statistics fold and the
        quantile scan iterate. A store that cannot serve bounded region reads (GIPL) is read whole
        once and kept for every pass: the declared whole-volume route, which the plan names LOAD and
        refuses when the volume does not fit the budget."""
        shape, _ = self.get_infos(groups, name)
        if len(shape) < 2 or not self.bounded_region_reads(groups, name):
            resident: list[np.ndarray] = []

            def whole() -> Iterator[np.ndarray]:
                if not resident:
                    resident.append(self.read_data(groups, name)[0])
                yield resident[0]

            return whole
        # A whole number of update pieces: the fold sees the same pieces in the same order whatever
        # the read grain, so the budget never changes what the running mean and std answer.
        rows = _scan_rows(
            [(self, groups)], name, shape, _statistics_chunk_length(shape, 1, statistics._STATISTICS_UPDATE_ELEMENTS)
        )

        def slabs() -> Iterator[np.ndarray]:
            for start in range(0, int(shape[1]), rows):
                slices = (
                    slice(None),
                    slice(start, min(int(shape[1]), start + rows)),
                    *(slice(None) for _ in shape[2:]),
                )
                yield self.read_data_slice(groups, name, slices)[0]

        return slabs

    def _scanned_element_bytes(self, groups: str, name: str, shape: list[int]) -> int:
        """What one element of a scanned block costs: the store's own element size, read off a
        one-voxel region."""
        probe = (slice(0, 1),) * len(shape)
        return max(1, int(self.read_data_slice(groups, name, probe)[0].dtype.itemsize))

    def read_data_quantile(self, groups: str, name: str, q: float) -> Any:
        """``numpy.quantile(volume, q)`` (the default ``linear`` method, to the value) without
        holding the volume: bounded passes over :meth:`iter_data_blocks`."""
        low, high, weight = _order_statistics(self.iter_data_blocks(groups, name), float(q))
        if not np.issubdtype(np.asarray(low).dtype, np.inexact):
            # numpy.quantile promotes an integer input to float64 before it interpolates.
            low, high = np.float64(low), np.float64(high)
        return _lerp_like_numpy(low, high, weight) if weight else low

    def bounded_region_reads(self, groups: str, name: str) -> bool:
        """Whether a region read of this entry decodes only the region, or the whole volume (GIPL, a
        compressed file with no uncompressed twin). ``False`` for a missing entry."""
        try:
            return self._resolve_entry(groups, name, lambda file, _, entry: file.bounded_region_reads(entry))
        except DatasetManagerError:
            return False

    def read_data_statistics(
        self,
        groups: str,
        name: str,
        channels: list[int] | None = None,
        keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Min/max/mean/std of one entry, over the volume and per channel (``channels`` restricts
        both to those), folded over :meth:`iter_data_blocks`: the volume is never held. ``keys``
        names the statistics the caller reads (``min``, ``max_per_channel``, ...): without a mean or
        std the fold stays in the stored dtype and computes the extrema only. ``None`` computes the four."""
        update = _update_running_statistics if needs_moments(keys) else _update_running_extrema
        state = None
        for block in self.iter_data_blocks(groups, name)():
            for piece in _update_pieces(block if channels is None else block[channels]):
                state = update(state, piece)
        return _finalize_running_statistics(state)

    def read_transform(self, group: str, name: str) -> sitk.Transform:
        if not self.exists_on_disk():
            raise DatasetManagerError(
                f"The dataset '{self.filename}' does not exist.",
                "Check 'dataset_filenames' and the path it names.",
            )
        data, attribute = self.read_data(group, name)
        return data_to_transform(data, attribute, name)

    def read_image(self, group: str, name: str) -> sitk.Image:
        data, attribute = self.read_data(group, name)
        return data_to_image(data, attribute)

    def get_size(self, group: str) -> int:
        return len(self.get_names(group))

    def is_group_exist(self, group: str, requested: set[str] | None = None) -> bool:
        """Whether this root holds ``group``.

        ``requested`` is what the caller is about to select (:meth:`select_names`): with it, the
        first case holding the group answers. Without it the whole listing is taken and cached.
        """
        if requested is None or not self.is_directory:
            return bool(self.get_names(group))
        names = self._iter_names(group)
        try:
            return next(names, None) is not None
        finally:
            names.close()

    def is_dataset_exist(self, group: str, name: str) -> bool:
        """Whether ``(group, name)`` is on disk, asked of disk at the moment it is asked.

        Never a slice of :meth:`get_names`: a group the run itself produces gains cases while it is
        read, from another ``Dataset`` object and possibly another process.
        """
        if not self.exists_on_disk():
            return False
        if self.is_directory:
            # Not _resolve_entry: answers False instead of raising.
            entry_group = group.split("/")[-1]
            return any(
                self._holds(sub_directory, entry_group, name) for sub_directory in self._get_sub_directories(group)
            )
        with self._file(self.filename, True) as file:
            # Only the store's own listing expands a wildcard group.
            return name in file.get_names(group) if "*" in group else file.is_exist(group, name)

    def _get_sub_directories(self, groups: str, sub_directory: str = ""):
        group = groups.split("/")[0]
        sub_directories = []
        if len(groups.split("/")) == 1:
            sub_directories.append(sub_directory)
        elif group == "*":
            root = f"{self.filename}{sub_directory}"
            for k in uri.list_names(root):
                if uri.is_dir(f"{root}{k}"):
                    sub_directories.extend(
                        self._get_sub_directories(
                            "/".join(groups.split("/")[1:]),
                            f"{sub_directory}{k}/",
                        )
                    )
        else:
            sub_directory = f"{sub_directory}{group}/"
            if uri.exists(f"{self.filename}{sub_directory}"):
                sub_directories.extend(self._get_sub_directories("/".join(groups.split("/")[1:]), sub_directory))
        return sub_directories

    def _iter_names(self, groups: str) -> Generator[str, None, None]:
        """Every case of ``groups`` this root holds, one entry open at a time and in no order."""
        if not self.is_directory:
            with self._file(self.filename, True) as file:
                yield from file.get_names(groups)
            return
        group = groups.split("/")[-1]
        suffix = self._backend.case_file_suffix
        for sub_directory in self._get_sub_directories(groups):
            root = f"{self.filename}{sub_directory}"
            for name in uri.list_names(root):
                if suffix and uri.is_dir(f"{root}{name}"):
                    continue
                with self._file(f"{root}{name}", True) as file:
                    if file.is_exist(group):
                        yield name.removesuffix(suffix) if suffix else name

    def get_names(self, groups: str, index: list[int] | None = None) -> list[str]:
        if index is None and groups in self._names_cache:
            return self._names_cache[groups]

        sorted_names = sorted(self._iter_names(groups))
        if index is None:
            self._names_cache[groups] = sorted_names
            return sorted_names
        return [name for i, name in enumerate(sorted_names) if i in index]

    def select_names(self, groups: str, requested: set[str] | None) -> list[str]:
        """The names of ``groups`` this root holds, narrowed to ``requested``.

        ``requested`` is the set the caller will keep, or ``None`` for the whole cohort. A directory
        root probes each requested name instead of enumerating; a name is probed only as the listing
        would spell it, one path component, so ``case/`` or ``./case`` selects nothing.
        """
        if requested is None or not self.is_directory:
            names = self.get_names(groups)
            return names if requested is None else sorted(requested.intersection(names))
        group = groups.split("/")[-1]
        return sorted(
            {
                name
                for sub_directory in self._get_sub_directories(groups)
                for name in requested
                if _is_listed_name(name) and self._holds(sub_directory, group, name)
            }
        )

    def get_group(self) -> list[str]:
        if self.is_directory:
            if self._backend.lists_case_entries:
                groups_set = set()
                for case in uri.list_names(self.filename):
                    case_path = uri.join(self.filename, case)
                    if uri.is_dir(case_path):
                        with self._file(case_path, True) as dataset_file:
                            groups_set.update(dataset_file.get_group())
                return sorted(groups_set)
            uri.refuse_remote_walk(self.filename, self.file_format)
            groups_set = set()
            for root_dir, directories, files in os.walk(self.filename):
                # A writer's staging directory holds an entry under its final name until it is moved in.
                directories[:] = [name for name in directories if not (name.startswith(".") and is_staging_entry(name))]
                for file in files:
                    # A MetaImage pixel file (.raw/.zraw, one per slice at times) is a half its header names.
                    if file.startswith(".") or is_staging_entry(file) or file.lower().endswith((".raw", ".zraw")):
                        continue
                    # A dot in an entry's stem belongs to its name: only its storage form is cut.
                    form = storage_form(Path(root_dir, file))
                    cut = form.lower()[1:] in SUPPORTED_EXTENSIONS
                    stem = file[: -len(form)] if cut else file.split(".")[0]
                    path = Path(root_dir, stem).relative_to(self.filename).as_posix()
                    parts = path.split("/")
                    if len(parts) >= 2:
                        del parts[-2]
                    groups_set.add("/".join(parts))
            groups = list(groups_set)
        else:
            with self._file(self.filename, True) as dataset_file:
                groups = dataset_file.get_group()
        return list(groups)

    def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        # The header read is memoised; copies go in and out so a caller cannot mutate the cache.
        cache_key = (groups, name)
        cached = self._infos_cache.get(cache_key)
        if cached is None:
            shape, attr = self._resolve_entry(groups, name, lambda file, group, entry: file.get_infos(group, entry))
            cached = self._infos_cache[cache_key] = (list(shape), Attribute(attr))
        shape, attr = cached
        return list(shape), Attribute(attr)


def refuse_shared_single_file(world_size: int, destinations: Iterable[Dataset], error: type[KonfAIError]) -> None:
    """Refuse a multi-rank run before any rank writes into a single-file store.

    Ranks shard by case, and a directory dataset gives each case its own file or store, so their writes
    are disjoint; a single-file store (h5) puts every case in one handle, and no lock spans processes.
    NOT ``concurrent_write_safe()``: that asks whether two entries of one shared store may be written at
    once, and answers no for omezarr, whose cases are disjoint stores.
    """
    if world_size <= 1:
        return
    for destination in destinations:
        if not destination.is_directory:
            raise error(
                f"{world_size} processes: destination '{destination.store_root}' is a single-file store,"
                " and every rank would write into the same file.",
                "Use one process, or a directory destination (omezarr, mha, nii.gz).",
            )
