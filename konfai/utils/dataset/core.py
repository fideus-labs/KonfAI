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
from collections.abc import Callable, Generator, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from typing import TYPE_CHECKING

from konfai.utils import uri
from konfai.utils.budget import format_bytes, per_rank_budget_bytes
from konfai.utils.dataset import statistics
from konfai.utils.dataset.abstract import AbstractFile as _AbstractFile
from konfai.utils.dataset.attribute import (
    Attribute,
    as_channel_first,
    data_to_image,
    data_to_transform,
    is_an_image,
)
from konfai.utils.dataset.backend import File as _File
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
    _scan_block_on_the_store_grid,
    _statistics_block_elements,
    _statistics_chunk_length,
    _statistics_plane_elements,
    _update_pieces,
    _update_running_statistics,
)
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import (
    STORE_FORMS,
    SUPPORTED_FORMATS,
    directory_volume_form,
    is_store_name,
    split_format_level,
)

if TYPE_CHECKING:
    from konfai.utils.dataset.stream import DataStream

_T = TypeVar("_T")


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
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        base_format, self.level = split_format_level(file_format)
        normalized_format = base_format.lower().removeprefix(".")
        # One vocabulary for a store: every spelling the walk accepts on disk is a token here, the
        # dotted one included, since that is the suffix a store actually carries.
        file_format = "omezarr" if f".{normalized_format}" in STORE_FORMS else normalized_format
        if file_format not in SUPPORTED_FORMATS:
            # Unchecked, the token reaches the SimpleITK writer, which either raises on an extension
            # it cannot name or writes a file no backend of ours probes: 'hd5' for 'h5' wrote and read
            # back correctly while ``is_dataset_exist`` said no, so a resumed run redid the work.
            raise DatasetManagerError(
                f"'{base_format}' is not a format KonfAI writes.",
                "Use one of: " + ", ".join(sorted(SUPPORTED_FORMATS)) + ".",
            )
        self.filename, self.is_directory = Dataset._normalize_path(filename, file_format)
        self.file_format = file_format
        # The store backend is auto-detected from what is actually on disk (like SitkFile already probes
        # every supported extension): an OME-Zarr / Zarr / DICOM store is a directory whose type is
        # knowable from its structure, so a ``:mha`` token never forces it to be mis-read. The token then
        # only carries the WRITE format and the OME-Zarr pyramid level (``@N``).
        detected = Dataset._detect_directory_store_format(self.filename) if self.is_directory else None
        if detected is not None:
            self.file_format = detected
        # Write-side pyramid, declared by the Save/Write that owns this destination. Refused here
        # rather than ignored: only OME-NGFF has multiple levels, so a pyramid asked of an mha or an
        # h5 is a request the format cannot serve, and silently writing one level would leave the
        # consumer's ``@1`` resolving to a level that does not exist.
        if scale_factors and self.file_format != "omezarr":
            raise DatasetManagerError(
                f"A pyramid was asked of a '{self.file_format}' destination, which has no levels.",
                "Only ':omezarr' stores levels. Drop scale_factors, or write to ':omezarr'.",
            )
        self.scale_factors = list(scale_factors) if scale_factors else None
        self.downsample_method = downsample_method
        self._names_cache: dict[str, list[str]] = {}
        self._infos_cache: dict[tuple[str, str], tuple[list[int], Attribute]] = {}
        #: Facts a stage derived from an entry's pixels (a Crop's foreground box), keyed by
        #: ``(group, name)``: computed once per volume, whatever the number of chains reading it.
        self.case_facts: dict[tuple[str, str], dict[str, Any]] = {}

    def _file(self, filename: str, read: bool) -> _File:
        """One entry's backing file, opened as this dataset's root is."""
        return self.File(filename, read, self.file_format, self.level)

    @staticmethod
    def _normalize_path(filename: str | Path, file_format: str) -> tuple[str, bool]:
        # A single-store h5 is one file, every other backend a directory of cases: only the latter gets the
        # trailing slash that marks ``is_directory``. Keep the two in lock-step so a path never ends up a
        # directory-flagged h5 (which would write the hidden dotfile ``<dir>/.h5``). ``as_posix`` keeps the
        # separator forward on every OS, so the stored filename (and the trailing-slash marker) is the same
        # on Windows, where ``prefix / name`` would otherwise render backslashes.
        path = uri.normalize(filename)
        if file_format != "h5" and not path.endswith("/"):
            path += "/"
        return path, path.endswith("/")

    def rebase(self, prefix: Path) -> None:
        """Prepend ``prefix`` to this dataset's path, re-deriving ``is_directory`` from the format.

        A rebased root is an output root, and ``prefix / uri`` folds the scheme's second slash away:
        refused as a remote root before it can stop looking like one.
        """
        uri.refuse_write(self.filename)
        self.filename, self.is_directory = Dataset._normalize_path(prefix / self.filename, self.file_format)

    @staticmethod
    def _detect_directory_store_format(root: str) -> str | None:
        """Detect a directory dataset's store backend from disk (``omezarr`` / ``dicom``), independent of the
        format token; ``None`` when it is plain per-file volumes (the SitkFile path, which auto-detects the
        extension itself). Probes the first case's entries only: cheap, and cases share one layout."""
        if not uri.is_dir(root):
            return None
        if uri.is_uri(root):
            # Only the store backend reads a remote root, and a store is told by its name: a
            # remote entry is never probed as a path, which on a bare name asks the working directory.
            names = Dataset._first_case_entries(root)
            return "omezarr" if any(is_store_name(name.name) for name in names) else None
        for entry in Dataset._first_case_entries(root):
            volume = directory_volume_form(entry)
            if volume is not None:
                return "dicom" if volume == "" else "omezarr"
        return None

    @staticmethod
    def _first_case_entries(root: str) -> list[Path]:
        """What ``root``'s first case directory holds, empty when it has none.

        Unsorted on a local root: any case is representative of the layout, and ``iterdir`` stops at
        the first directory where a listing materialises the whole of a resume's output tree. A
        remote listing is one request either way, and arrives sorted.
        """
        if uri.is_uri(root):
            cases = (name for name in uri.list_names(root))
            case = next((name for name in cases if uri.is_dir(uri.join(root, name))), None)
            return [] if case is None else [Path(name) for name in uri.list_names(uri.join(root, case))]
        case_path = next((child for child in Path(root).iterdir() if child.is_dir()), None)
        return [] if case_path is None else sorted(case_path.iterdir())

    @property
    def store_root(self) -> str:
        """Where the store lives, as text: its root directory, or the ``.h5`` file for a single-file
        store (named with or without the suffix, as the backend opens it).

        Text, because ``Path`` eats the second slash of a URI; :attr:`path_on_disk` is the local-only
        view, for the callers that manipulate the path.
        """
        root = self.filename
        if self.file_format == "h5" and not root.endswith(".h5"):
            return f"{root}.h5"
        return root

    @property
    def path_on_disk(self) -> Path:
        """:attr:`store_root` as a path. Local roots only: a URI has no filesystem path."""
        return Path(self.store_root)

    def exists_on_disk(self) -> bool:
        """Whether the store is there, asked of whichever filesystem owns it. A remote root that
        cannot be reached raises; only one that answers gets to say no."""
        return uri.exists(self.store_root)

    def concurrent_write_safe(self) -> bool:
        """Whether writes to different entries land in disjoint files, so a background writer may
        flush one entry while another thread writes elsewhere in the dataset.

        Mirrors the backend dispatch in ``File.__enter__``: everything that is not a single-store
        backend is a :class:`SitkFile` directory, one image file per ``(group, name)``. A single
        store (one HDF5 file, one zarr hierarchy, a DICOM series) shares handles and metadata across
        entries and must stay serial.
        """
        return self.file_format not in ("h5", "omezarr", "dicom")

    def _write_target(self, group: str, name: str) -> tuple[_File, str]:
        """The file a ``(group, name)`` write lands in and the entry name inside it, caches dropped.

        A directory dataset routes any sub-directory prefix of ``group`` into the file path (one file
        per case); a single store keeps one file and a ``group/name`` entry.
        """
        # Ahead of the makedirs below, which would take a URI for a directory name.
        uri.refuse_write(self.filename)
        self._names_cache.clear()
        self._infos_cache.clear()
        self.case_facts.clear()
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

    def can_stream_data(self, attributes: Attribute) -> bool:
        """Whether ``open_data_stream`` can serve this dataset's write format.

        H5 and OME-Zarr always can; MetaImage ``mha`` needs image geometry to write its header up
        front; every other format only writes whole volumes (use ``write``).
        """
        if self.file_format in ("h5", "omezarr"):
            return True
        if self.file_format == "itktransform":
            return is_an_image(attributes)
        return self.file_format in ("mha", "nii") and is_an_image(attributes)

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

        ``region_shape`` is the extent the caller will write at a time, channels included. A store
        that chunks on it never pays a read-modify-write; a store left to guess pays one on every
        region that straddles a chunk. Declaring it is the writer's job, it is the only party that
        knows its own access pattern.
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
        return stream

    def _case_path(self, sub_directory: str, name: str) -> str | None:
        """The file a directory dataset stores case ``name`` under, or ``None`` if absent on disk.

        The returned path omits the implicit ``.h5`` suffix h5 case files carry: ``H5File``
        re-appends it on open.
        """
        path = f"{self.filename}{sub_directory}{name}"
        on_disk = f"{path}{'.h5' if self.file_format == 'h5' else ''}"
        if uri.exists(on_disk):
            return path
        if uri.is_uri(on_disk):
            return None  # no writer of a remote root, so no backup of one to recover
        # Absent is not always absent: a writer killed mid-replacement leaves the previous version
        # under its backup name, which the listings hide. Asked of disk again after the attempt: the
        # recovery declines to a publish that landed meanwhile, and that publish is the entry.
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
        """Run ``action`` on the open file holding ``(groups, name)``: THE place entry resolution lives.

        ``action`` receives the backend and the entry's coordinates INSIDE that file: a directory
        dataset stores one case per file, addressed by ``name``, with the entry keyed by the group
        path's last component, so the coordinates are ``("", group)`` there and ``(groups, name)``
        on a single-file dataset. Raises ``NameError`` when the dataset or the entry is missing.
        """
        if not self.exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                path = self._case_path(sub_directory, name)
                if path is not None:
                    with self._file(path, True) as file:
                        return action(file, "", groups.split("/")[-1])
            raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")
        with self._file(self.filename, True) as file:
            return action(file, groups, name)

    def read_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self._resolve_entry(groups, name, lambda file, group, entry: file.file_to_data(group, entry))

    def read_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        return self._resolve_entry(
            groups, name, lambda file, group, entry: file.file_to_data_slice(group, entry, slices)
        )

    def read_granularity(self, groups: str, name: str) -> tuple[int, ...] | None:
        """The stored block reads of ``(groups, name)`` are served in, or ``None`` when a read costs
        exactly what it asks for. What a decomposition is sized and aligned against."""
        with contextlib.suppress(Exception):
            return self._resolve_entry(groups, name, lambda file, _group, entry: file.read_granularity(entry))
        return None

    def plan_region_reads(self, groups: str, name: str, windows: Sequence[tuple[slice, ...]]) -> None:
        """Declare the region reads about to happen on ``(groups, name)``, in order. A backend that
        can use it does; the rest ignore it, and so does a caller that declares nothing."""
        with contextlib.suppress(NameError):
            self._resolve_entry(groups, name, lambda file, _group, entry: file.plan_region_reads(entry, windows))

    def iter_data_blocks(self, groups: str, name: str) -> Callable[[], Iterator[np.ndarray]]:
        """A factory of passes over one entry, block by block along the first spatial axis, each
        block about ``_STATISTICS_CHUNK_ELEMENTS`` elements: what a scan that must never hold the
        volume iterates (the statistics fold, the quantile scan). A store that cannot serve bounded
        region reads (gzipped NIfTI, compressed MetaImage) is read whole ONCE and kept for every
        pass the factory serves: those formats have no bounded reader to use instead, a block read
        decodes the whole volume anyway, so reading per block would hold the same peak N times over.
        This is the declared whole-volume route, not a way around the streaming invariant: a case
        that needs it plans as LOAD, and the plan refuses it when the volume does not fit the
        budget, before a byte is written."""
        shape, _ = self.get_infos(groups, name)
        if len(shape) < 2 or not self.bounded_region_reads(groups, name):
            resident: list[np.ndarray] = []

            def whole() -> Iterator[np.ndarray]:
                if not resident:
                    resident.append(self.read_data(groups, name)[0])
                yield resident[0]

            return whole
        # A whole number of update pieces, so the fold sees the same sequence of pieces in the same
        # order whatever the read grain: the running mean and std are then the budget's business
        # only in how much is held, never in what they answer.
        budget = per_rank_budget_bytes()
        # Only a declared budget sizes anything from it, so only a declared budget pays the probe.
        element_bytes = (
            statistics._STATISTICS_ELEMENT_BYTES if budget is None else self._scanned_element_bytes(groups, name, shape)
        )
        piece = _statistics_chunk_length(shape, 1, statistics._STATISTICS_UPDATE_ELEMENTS)
        rows = max(
            piece, _statistics_chunk_length(shape, 1, _statistics_block_elements(element_bytes)) // piece * piece
        )
        plane = _statistics_plane_elements(shape, 1)
        granularity = self.read_granularity(groups, name)
        rows, held = _scan_block_on_the_store_grid(
            rows, int(shape[1]), plane, granularity[1:2] if granularity else None, budget, element_bytes
        )
        if budget is not None and held > budget:
            raise DatasetManagerError(
                f"'{name}': the shortest block a whole-volume scan of '{groups}' can read holds"
                f" {format_bytes(held)}, over the per-rank memory budget ({format_bytes(budget)}).",
                "Raise 'memory_budget'.",
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
        one-voxel region.

        A block of a scan is the bytes the store hands over, never a cast copy of them, so a
        float64 source held twice what the budget was told at every block in flight. The probe is a
        bounded read, which is the route this entry is on, and one voxel of it.
        """
        probe = (slice(0, 1),) * len(shape)
        return max(1, int(self.read_data_slice(groups, name, probe)[0].dtype.itemsize))

    def read_data_quantile(self, groups: str, name: str, q: float) -> Any:
        """``numpy.quantile(volume, q)`` (the default ``linear`` method, to the value) without
        holding the volume: bounded passes over :meth:`iter_data_blocks`."""
        low, high, weight = _order_statistics(self.iter_data_blocks(groups, name), float(q))
        if not np.issubdtype(np.asarray(low).dtype, np.inexact):
            # numpy.quantile promotes an integer input to float64 before it interpolates: the
            # difference of two order statistics would wrap on a narrow signed type, and an exact
            # index would answer in the stored dtype where numpy answers in float64.
            low, high = np.float64(low), np.float64(high)
        return _lerp_like_numpy(low, high, weight) if weight else low

    def bounded_region_reads(self, groups: str, name: str) -> bool:
        """Whether a region read of this entry decodes only the region, or the whole volume.

        What it prices is the ROUTE, never the answer: a store that decodes the whole volume once
        per slab (compressed MetaImage, NRRD, gzipped NIfTI) makes streaming read the source many
        times over, where loading reads it once. ``False`` for a missing entry: pessimistic, and
        only ever costing speed.
        """
        try:
            return self._resolve_entry(groups, name, lambda file, _, entry: file.bounded_region_reads(entry))
        except NameError:
            return False

    def read_data_statistics(
        self,
        groups: str,
        name: str,
        channels: list[int] | None = None,
    ) -> dict[str, Any]:
        """Min/max/mean/std of one entry, over the volume and per channel (``channels`` restricts
        both to those), folded over :meth:`iter_data_blocks`: the volume is never held."""
        state = None
        for block in self.iter_data_blocks(groups, name)():
            for piece in _update_pieces(block if channels is None else block[channels]):
                state = _update_running_statistics(state, piece)
        return _finalize_running_statistics(state)

    def read_transform(self, group: str, name: str) -> sitk.Transform:
        if not self.exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        data, attribute = self.read_data(group, name)
        return data_to_transform(data, attribute, name)

    def read_image(self, group: str, name: str) -> sitk.Image:
        data, attribute = self.read_data(group, name)
        return data_to_image(data, attribute)

    def get_size(self, group: str) -> int:
        return len(self.get_names(group))

    def is_group_exist(self, group: str, requested: set[str] | None = None) -> bool:
        """Whether this root holds ``group``, asked as narrowly as the caller will read it.

        ``requested`` is what the caller is about to select (:meth:`select_names`): with it, the
        first case holding the group answers, where counting would walk a cohort the run then
        discards. Without it the caller reads the whole listing next, so this takes that listing
        and leaves it cached.
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

        Deliberately NOT a slice of :meth:`get_names`: that listing is a planning-time snapshot, and a
        group the run itself produces (a ``Save`` writing into the dataset being read) gains cases
        while it is read, through a different ``Dataset`` object and, when the loader has workers, a
        different PROCESS. No memo can be invalidated across that boundary, so membership asks the disk.
        One entry, one probe: O(1) in the number of cases, where the listing is O(N) headers, and cheaper
        than the listing it replaces.
        """
        if not self.exists_on_disk():
            # A store that is not there yet holds nothing: the first probe of every fresh
            # destination, which a single-file backend would otherwise turn into an open error.
            return False
        if self.is_directory:
            # Not _resolve_entry: membership keeps scanning past a case file whose group is absent,
            # and answers False instead of raising.
            entry_group = group.split("/")[-1]
            return any(
                self._holds(sub_directory, entry_group, name) for sub_directory in self._get_sub_directories(group)
            )
        with self._file(self.filename, True) as file:
            # A wildcard group is a path pattern; only the store's own listing expands it.
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
        """Every case of ``groups`` this root holds, one entry open at a time and in no order.

        Lazy so a caller that only needs to know whether there IS one stops at the first.
        """
        if not self.is_directory:
            with self._file(self.filename, True) as file:
                yield from file.get_names(groups)
            return
        group = groups.split("/")[-1]
        for sub_directory in self._get_sub_directories(groups):
            root = f"{self.filename}{sub_directory}"
            for name in uri.list_names(root):
                if self.file_format == "h5" and uri.is_dir(f"{root}{name}"):
                    continue
                with self._file(f"{root}{name}", True) as file:
                    if file.is_exist(group):
                        yield name.replace(".h5", "") if self.file_format == "h5" else name

    def get_names(self, groups: str, index: list[int] | None = None) -> list[str]:
        if index is None and groups in self._names_cache:
            return self._names_cache[groups]

        sorted_names = sorted(self._iter_names(groups))
        if index is None:
            self._names_cache[groups] = sorted_names
            return sorted_names
        return [name for i, name in enumerate(sorted_names) if i in index]

    def select_names(self, groups: str, requested: set[str] | None) -> list[str]:
        """The names of ``groups`` this root holds, asked of it as narrowly as the caller can ask.

        ``requested`` is the set the caller will keep, or ``None`` when only the whole cohort
        answers its selection. A root holding one entry per case is opened once per case it HOLDS
        to enumerate, and once per case the caller ASKED for to answer this, which is the whole
        difference between a wide root and a narrow subset. A root that is one entry answers
        either from the single listing it already takes. A name is probed only as the listing
        would have spelled it: one path component, so ``case/`` or ``./case`` selects nothing.
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
            if self.file_format in {"dicom", "omezarr"}:
                groups_set = set()
                for case in uri.list_names(self.filename):
                    case_path = uri.join(self.filename, case)
                    if uri.is_dir(case_path):
                        with self._file(case_path, True) as dataset_file:
                            groups_set.update(dataset_file.get_group())
                return sorted(groups_set)
            uri.refuse_remote_walk(self.filename, self.file_format)
            groups_set = set()
            for root_dir, _, files in os.walk(self.filename):
                for file in files:
                    if file.startswith(".") or is_staging_entry(file):  # a staging write, or its crashed leftover
                        continue
                    path = Path(root_dir, file.split(".")[0]).relative_to(self.filename).as_posix()
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
        # Memoize the header read (SITK reader + ReadImageInformation, or the HDF5/Zarr
        # metadata parse): get_infos is called once per name per group per build-pass at
        # setup, so caching it (like get_names) avoids re-parsing the same header N times.
        # Cache and hand back copies so a caller mutating the geometry cannot poison it.
        cache_key = (groups, name)
        cached = self._infos_cache.get(cache_key)
        if cached is not None:
            shape, attr = cached
            return list(shape), Attribute(attr)
        result = self._resolve_entry(groups, name, lambda file, group, entry: file.get_infos(group, entry))
        self._infos_cache[cache_key] = (list(result[0]), Attribute(result[1]))
        return result
