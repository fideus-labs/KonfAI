# SPDX-License-Identifier: Apache-2.0
"""What Scout inspects: one case of a task, its files by role, read lazily with their geometry.

Roles by task (``*`` required):
  segmentation  image*, prediction*, truth, mask
  registration  fixed*, moving*, field* (a displacement field image, mm, LPS, on the fixed grid: the fixed point x
                matches x + u(x) in the moving image) or transform* (an ITK transform file), fixed_labels,
                moving_labels, warped_labels, fixed_landmarks, moving_landmarks, fixed_mask
  synthesis     source*, synthetic*, target, mask
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .geometry import Grid

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
LANDMARK_SUFFIXES = (".fcsv", ".csv", ".txt")
TRANSFORM_SUFFIXES = (".h5", ".hdf5", ".tfm", ".txt", ".mat")


class CaseError(ValueError):
    """A case Scout cannot inspect as given."""


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover - the dependency is declared
        raise CaseError("Scout reads images with SimpleITK: pip install SimpleITK") from error
    return sitk


def sha256_of(path: Path, limit: int = 64 << 20) -> str:
    """The SHA-256 of a file (its first 64 MiB and its size for a larger one: a fingerprint, not a checksum)."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    if size > limit:
        digest.update(f"size={size}".encode())
    return digest.hexdigest()


class Volume:
    """One image file read lazily: its header at once, its planes and regions on demand."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise CaseError(f"{self.path} does not exist")
        sitk = _sitk()
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self.path))
        try:
            reader.ReadImageInformation()
        except RuntimeError as error:
            raise CaseError(f"{self.path} is not an image SimpleITK can read: {error}") from error
        self.dimension = int(reader.GetDimension())
        if self.dimension not in (2, 3):
            raise CaseError(f"{self.path} has {self.dimension} dimensions; Scout inspects 2-D and 3-D images")
        self.components = int(reader.GetNumberOfComponents())
        self.pixel_type = sitk.GetPixelIDValueAsString(reader.GetPixelID())
        self.grid = Grid.of(reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection())
        self._whole: np.ndarray | None = None

    @property
    def is_integer(self) -> bool:
        return "int" in self.pixel_type.lower() and "float" not in self.pixel_type.lower()

    @property
    def is_vector(self) -> bool:
        return self.components == self.dimension and self.components > 1

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.grid.shape_zyx

    def _as_array(self, image) -> np.ndarray:
        sitk = _sitk()
        array = sitk.GetArrayFromImage(image)
        if self.dimension == 2:
            array = array[None] if self.components == 1 else array[None]
        if self.components > 1:
            array = np.moveaxis(array, -1, 0)
        return array

    def region(self, lo_zyx: tuple[int, int, int], hi_zyx: tuple[int, int, int]) -> np.ndarray:
        """The voxels ``lo:hi`` (array order, ``hi`` exclusive) as ``[(C,) Z, Y, X]``, read as a region: the whole
        volume is never loaded for a slice. A 2-D image is read whole once (one plane) and cached."""
        sitk = _sitk()
        lo = [int(v) for v in lo_zyx]
        hi = [int(min(h, n)) for h, n in zip(hi_zyx, self.shape_zyx, strict=True)]
        if any(h <= lo_ for h, lo_ in zip(hi, lo, strict=True)) or min(lo) < 0:
            raise CaseError(f"empty region {lo}:{hi} in {self.path.name} of shape {self.shape_zyx}")
        if self.dimension == 2:
            if self._whole is None:
                self._whole = self._as_array(sitk.ReadImage(str(self.path)))
            return self._whole[..., lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self.path))
        reader.ReadImageInformation()
        reader.SetExtractIndex([lo[2], lo[1], lo[0]])
        reader.SetExtractSize([hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]])
        return self._as_array(reader.Execute())

    def plane(self, array_axis: int, index: int) -> np.ndarray:
        """One slice along an array axis, as ``[(C,) H, W]`` with the in-plane axes in array order."""
        lo = [0, 0, 0]
        hi = list(self.shape_zyx)
        lo[array_axis] = int(index)
        hi[array_axis] = int(index) + 1
        block = self.region(tuple(lo), tuple(hi))  # type: ignore[arg-type]
        return np.squeeze(block, axis=block.ndim - 3 + array_axis)

    def image(self):
        """The whole image, for a resample (a warp needs every voxel)."""
        # ponytail: a warp reads the moving image whole; stream it by region when cases outgrow RAM
        return _sitk().ReadImage(str(self.path))

    def sample_percentiles(self, lower: float = 1.0, upper: float = 99.0, planes: int = 5) -> tuple[float, float]:
        """An intensity window from a few planes of the longest axis, never the whole volume."""
        axis = int(np.argmax(self.shape_zyx))
        count = self.shape_zyx[axis]
        samples = [
            self.plane(axis, i).astype(np.float32).ravel()
            for i in np.linspace(0, count - 1, min(planes, count)).astype(int)
        ]
        values = np.concatenate(samples)
        if self.components > 1:
            values = np.abs(values)
        low, high = np.percentile(values, (lower, upper))
        return float(low), float(high) if high > low else float(low) + 1.0

    def describe(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "pixel_type": self.pixel_type,
            "components": self.components,
            "dimension": self.dimension,
            **self.grid.describe(),
        }


def read_landmarks(path: Path) -> np.ndarray:
    """Points ``[N, 3]`` in LPS from a Slicer fiducial file or a plain ``x,y,z`` CSV (``x,y`` in 2-D, z = 0); a
    ``# CoordinateSystem = RAS`` header is converted."""
    system = "LPS"
    rows: list[list[float]] = []
    with Path(path).open(newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                key, _, value = line[1:].partition("=")
                if key.strip() == "CoordinateSystem":
                    system = value.strip().upper()
                continue
            cells = next(csv.reader([line.strip()]))
            if not cells or not line.strip():
                continue
            numbers = []
            for cell in cells:
                try:
                    numbers.append(float(cell))
                except ValueError:
                    continue
            if len(cells) >= 4 and not _is_number(cells[0]):  # id,x,y,z[,...]: the id first
                numbers = numbers[:3]
            elif len(numbers) >= 2:
                numbers = numbers[:3]
            else:
                continue
            rows.append(numbers + [0.0] * (3 - len(numbers)))
    points = np.asarray(rows, dtype=float).reshape(-1, 3)
    if system in ("RAS", "0"):
        points[:, :2] *= -1
    return points


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


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
        if role not in self._volumes:
            self._volumes[role] = Volume(self.inputs[role])
        return self._volumes[role]

    @cached_property
    def grid(self) -> Grid:
        """The reference grid, the image every view is drawn over."""
        return self.volume(self.image_role).grid

    def landmarks(self, role: str) -> np.ndarray | None:
        return read_landmarks(self.inputs[role]) if role in self.inputs else None

    def fingerprint(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for role, path in self.inputs.items():
            entry: dict[str, object] = {"path": str(path), "sha256": sha256_of(path), "bytes": path.stat().st_size}
            if path.suffix.lower() not in LANDMARK_SUFFIXES + TRANSFORM_SUFFIXES or role in ("field",):
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
