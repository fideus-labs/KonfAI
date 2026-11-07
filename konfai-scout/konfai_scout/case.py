# SPDX-License-Identifier: Apache-2.0
"""What Scout inspects: one case of a task, its files by role, read through KonfAI's dataset backends.

The files are staged as KonfAI reads a case (``<root>/<case>/<ROLE><suffix>``, one root per storage backend, linked
not copied, as konfai-apps stages its inputs), so an image file, an OME-Zarr store, a DICOM series or an ITK
transform file are all read the same way: by region, with their geometry, through ``konfai.utils.dataset.Dataset``.

Roles by task (``*`` required):
  segmentation  image*, prediction*, truth, mask
  registration  fixed*, moving*, field* (a displacement field image, mm, LPS, on the fixed grid: the fixed point x
                matches x + u(x) in the moving image) or transform* (an ITK transform file), fixed_labels,
                moving_labels, warped_labels, fixed_landmarks, moving_landmarks, fixed_mask
  synthesis     source*, synthetic*, target, mask
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from konfai.data.geometry import Grid
from konfai.utils.dataset import Dataset, entry_backend, link_entry, read_landmarks
from konfai.utils.errors import KonfAIError
from konfai.utils.utils import directory_volume_form, storage_form

from .planes import promoted

TASKS = ("segmentation", "registration", "synthesis")
ROLES: dict[str, dict[str, bool]] = {
    "segmentation": {"image": True, "prediction": True, "truth": False, "mask": False},
    "registration": {
        "fixed": True,
        "moving": True,
        "field": False,
        "transform": False,
        "fixed_labels": False,
        "moving_labels": False,
        "warped_labels": False,
        "fixed_landmarks": False,
        "moving_landmarks": False,
        "fixed_mask": False,
    },
    "synthesis": {"source": True, "synthetic": True, "target": False, "mask": False},
}
#: The role a task's truth is read from: with it, reference metrics are possible.
TRUTH_ROLES = {
    "segmentation": ("truth",),
    "registration": ("fixed_labels", "fixed_landmarks"),
    "synthesis": ("target",),
}
#: Roles read as point sets, not volumes.
POINT_ROLES = ("fixed_landmarks", "moving_landmarks")
#: The group a role is staged under: the role's name in upper case.
BACKENDS = ("mha", "omezarr", "dicom", "itktransform")


class CaseError(ValueError):
    """A case Scout cannot inspect as given."""


def sha256_of(path: Path, limit: int = 64 << 20) -> str:
    """The SHA-256 of a file (its first 64 MiB and its size for a larger one: a fingerprint, not a checksum); for
    a store or a series directory, of its file names and sizes."""
    digest = hashlib.sha256()
    if path.is_dir():
        for part in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(f"{part.relative_to(path)}:{part.stat().st_size}\n".encode())
        return digest.hexdigest()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    if size > limit:
        digest.update(f"size={size}".encode())
    return digest.hexdigest()


class Volume:
    """One staged entry read through KonfAI: its header at once, its planes and regions on demand, always as a
    rank-3 ``[(C,) Z, Y, X]`` array on a rank-3 grid (a 2-D image is one plane thick)."""

    def __init__(self, dataset: Dataset, group: str, name: str, role: str) -> None:
        self.dataset, self.group, self.name, self.role = dataset, group, name, role
        try:
            shape, self.attribute = dataset.get_infos(group, name)
        except KonfAIError as error:
            raise CaseError(f"{role}: {error}") from error
        self.components = int(shape[0])
        self.rank = len(shape) - 1
        if self.rank not in (2, 3):
            raise CaseError(f"{role} has {self.rank} spatial dimensions; Scout inspects 2-D and 3-D images")
        try:
            self.grid = promoted(Grid.of([int(v) for v in shape[1:]], self.attribute, role))
        except KonfAIError as error:
            raise CaseError(f"{role}: {error}") from error
        self._dtype: np.dtype | None = None

    @property
    def is_vector(self) -> bool:
        return self.components == self.rank and self.components > 1

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._dtype = self.region((0, 0, 0), (1, 1, 1)).dtype
        return self._dtype

    @property
    def is_integer(self) -> bool:
        return np.issubdtype(self.dtype, np.integer) or self.dtype == np.bool_

    def region(self, lo_zyx: tuple[int, int, int], hi_zyx: tuple[int, int, int]) -> np.ndarray:
        """The voxels ``lo:hi`` (array order, ``hi`` exclusive) as ``[(C,) Z, Y, X]``, read as a region of the
        store: KonfAI reads what the region needs, never the whole volume for a slice."""
        lo = [int(v) for v in lo_zyx]
        hi = [int(min(h, n)) for h, n in zip(hi_zyx, self.grid.size_zyx, strict=True)]
        if any(h <= lo_ for h, lo_ in zip(hi, lo, strict=True)) or min(lo) < 0:
            raise CaseError(f"empty region {lo}:{hi} in {self.role} of shape {self.grid.size_zyx}")
        spatial = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        if self.rank == 2:
            spatial = spatial[1:]  # the z plane is the whole 2-D image
        try:
            array, _ = self.dataset.read_data_slice(self.group, self.name, (slice(None), *spatial))
        except KonfAIError as error:
            raise CaseError(f"{self.role}: {error}") from error
        array = np.asarray(array)
        if self.rank == 2:
            array = array[:, None]
        return array

    def plane(self, array_axis: int, index: int) -> np.ndarray:
        """One slice along an array axis, as ``[(C,) H, W]`` with the in-plane axes in array order; the channel
        axis is dropped for a scalar image."""
        lo = [0, 0, 0]
        hi = list(self.grid.size_zyx)
        lo[array_axis], hi[array_axis] = int(index), int(index) + 1
        block = self.region(tuple(lo), tuple(hi))  # type: ignore[arg-type]
        plane = np.take(block, 0, axis=1 + array_axis)
        return plane[0] if self.components == 1 else plane

    def quantile(self, q: float) -> float:
        """A quantile of the whole entry, by KonfAI's bounded passes: the volume is never held."""
        return float(self.dataset.read_data_quantile(self.group, self.name, q))

    def describe(self) -> dict[str, object]:
        from .planes import describe_grid

        return {"components": self.components, "dimension": self.rank, **describe_grid(self.grid)}


@dataclass
class InspectionCase:
    """One case to inspect: its task, its files by role, what is known about its labels and its prior evaluation."""

    task: str
    name: str
    inputs: dict[str, Path]
    #: Label value -> name, when known (a segmentation's terminology).
    labels: dict[int, str] = field(default_factory=dict)
    #: KonfAI Eval's report(s) of this case (``Metric_*.json`` paths) or values already read: ``{metric: value}``.
    evaluation: list[Path] | dict[str, float | None] | None = None
    #: What the caller knows: modality, dataset, the method under inspection, anything a reader needs.
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise CaseError(f"unknown task {self.task!r}; one of {', '.join(TASKS)}")
        self.inputs = {role: Path(path) for role, path in self.inputs.items()}
        roles = ROLES[self.task]
        unknown = sorted(set(self.inputs) - set(roles))
        if unknown:
            raise CaseError(f"unknown role(s) {unknown} for {self.task}; roles: {', '.join(roles)}")
        missing = sorted(role for role, required in roles.items() if required and role not in self.inputs)
        if missing:
            raise CaseError(f"{self.task} needs {', '.join(missing)}")
        if self.task == "registration" and "field" not in self.inputs and "transform" not in self.inputs:
            raise CaseError(
                "registration needs a field (a displacement field image) or a transform (an ITK transform file)"
            )
        for role, path in self.inputs.items():
            if not path.exists():
                raise CaseError(f"{role}: {path} does not exist")
        self.labels = {int(k): str(v) for k, v in self.labels.items()}
        self._volumes: dict[str, Volume] = {}
        self._stage: Path | None = None
        self._datasets: dict[str, Dataset] = {}

    # ------------------------------------------------------------------ staging through KonfAI

    @property
    def slug(self) -> str:
        """The case's name as KonfAI names a case on disk."""
        return self.name.replace("/", "__") or "case"

    def stage(self, work: Path | str) -> Path:
        """Link every volume role under ``work``, one root per storage backend (``Dataset_<backend>/<case>/<ROLE>``),
        as konfai-apps stages its inputs; the roots feed every KonfAI read, transform and evaluation of the case."""
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        self._datasets = {}
        for role, path in self.inputs.items():
            if role in POINT_ROLES:
                continue
            suffix = (directory_volume_form(path) if path.is_dir() else None) or storage_form(path)
            backend = entry_backend(path, suffix)
            root = work / f"Dataset_{backend}"
            name = f"{role.upper()}{suffix.lower() if backend == 'itktransform' else suffix}"
            link_entry(path, root / self.slug / name)
            if backend not in self._datasets:
                self._datasets[backend] = Dataset(root, backend)
        self._stage = work
        self._volumes = {}
        return work

    def staged(self) -> Path:
        if self._stage is None:
            self.stage(Path(tempfile.mkdtemp(prefix="konfai_scout_")))
        assert self._stage is not None
        return self._stage

    def dataset_roots(self) -> list[str]:
        """The staged roots as a KonfAI config names them (``path:format``), for a TRANSFORM or an EVALUATION run."""
        self.staged()
        return [f"{self._stage}/Dataset_{backend}:{backend}" for backend in self._datasets]

    def dataset_of(self, role: str) -> Dataset:
        self.staged()
        backend = entry_backend(self.inputs[role])
        return self._datasets[backend]

    @property
    def truth_available(self) -> bool:
        return any(role in self.inputs for role in TRUTH_ROLES[self.task])

    @property
    def image_role(self) -> str:
        """The role every view is drawn over: the image, the fixed image, the synthetic image."""
        return {"segmentation": "image", "registration": "fixed", "synthesis": "synthetic"}[self.task]

    def volume(self, role: str) -> Volume:
        if role not in self.inputs:
            raise CaseError(f"this case has no {role}")
        if role in POINT_ROLES or role == "transform":
            raise CaseError(f"{role} is not a volume")
        if role not in self._volumes:
            self._volumes[role] = Volume(self.dataset_of(role), role.upper(), self.slug, role)
        return self._volumes[role]

    @cached_property
    def grid(self) -> Grid:
        """The reference grid, the image every view is drawn over."""
        return self.volume(self.image_role).grid

    def landmarks(self, role: str) -> np.ndarray | None:
        """Points ``[N, 3]`` in LPS, read as KonfAI reads a fiducial file."""
        if role not in self.inputs:
            return None
        try:
            points = read_landmarks(self.inputs[role])
        except (KonfAIError, ValueError, OSError) as error:
            raise CaseError(f"{role}: {error}") from error
        return None if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def fingerprint(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for role, path in self.inputs.items():
            entry: dict[str, object] = {"path": str(path), "sha256": sha256_of(path)}
            if path.is_file():
                entry["bytes"] = path.stat().st_size
            if role not in POINT_ROLES and role != "transform":
                try:
                    entry.update(self.volume(role).describe())
                except CaseError:
                    pass
            out[role] = entry
        return out

    def to_dict(self) -> dict[str, object]:
        evaluation: object = self.evaluation
        if isinstance(evaluation, list):
            evaluation = [str(p) for p in evaluation]
        return {
            "task": self.task,
            "name": self.name,
            "inputs": {role: str(path) for role, path in self.inputs.items()},
            "labels": {str(k): v for k, v in self.labels.items()},
            "evaluation": evaluation,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict[str, object], base: Path | None = None) -> InspectionCase:
        def resolve(value: object) -> Path:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() or base is None else base / path

        evaluation: object = data.get("evaluation")
        if isinstance(evaluation, list):
            evaluation = [resolve(p) for p in evaluation]
        elif isinstance(evaluation, str):
            evaluation = [resolve(evaluation)]
        return InspectionCase(
            task=str(data["task"]),
            name=str(data.get("name") or "case"),
            inputs={str(role): resolve(path) for role, path in dict(data.get("inputs") or {}).items()},  # type: ignore[union-attr]
            labels={int(k): str(v) for k, v in dict(data.get("labels") or {}).items()},  # type: ignore[union-attr]
            evaluation=evaluation,  # type: ignore[arg-type]
            notes=str(data.get("notes") or ""),
        )

    @staticmethod
    def load(path: Path | str) -> InspectionCase:
        path = Path(path)
        return InspectionCase.from_dict(json.loads(path.read_text(encoding="utf-8")), base=path.parent)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
