# SPDX-License-Identifier: Apache-2.0
"""What a registration case needs derived before it is looked at, by KonfAI's TRANSFORM workflow: the moving image
and labels pulled onto the fixed grid through the registration (``Resample``), the moving image on the fixed grid
with no registration (the start), and the fixed landmarks carried by the registration (``TransformPoints``). One run,
streamed by KonfAI, its results written as a dataset beside the inspection and read back like any other volume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from konfai.utils.dataset import Dataset, read_landmarks
from konfai.utils.errors import KonfAIError

from .case import CaseError, InspectionCase, Volume

#: The groups the run writes, by what they hold.
WARPED, MOVING_ON_FIXED, WARPED_LABELS, MOVED_LANDMARKS, FIELD = (
    "WARPED",
    "MOVING_ON_FIXED",
    "WARPED_LABELS",
    "MOVED_LANDMARKS",
    "FIELD",
)


class Derived:
    """The derived entries of a registration case, made once per inspection under ``work``."""

    def __init__(self, case: InspectionCase, work: Path) -> None:
        self.case = case
        self.root = Path(work) / "derived"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset(self.root / "Dataset", "mha")
        self._ran = False
        self._volumes: dict[str, Volume] = {}

    # ------------------------------------------------------------------ the map

    def _map(self) -> dict[str, object]:
        """How ``Resample`` reads the registration: a field group, or a transform file group."""
        return {"field_group": "FIELD"} if "field" in self.case.inputs else {"transforms": {"TRANSFORM": False}}

    def _point_map(self) -> dict[str, bool]:
        return {"FIELD" if "field" in self.case.inputs else "TRANSFORM": False}

    # ------------------------------------------------------------------ the run

    def ensure(self) -> None:
        """Run KonfAI's TRANSFORM once: every derived group this case can have, on the CPU, quietly."""
        if self._ran:
            return
        from konfai import transform
        from konfai.data.transform import Resample, TransformPoints, Write

        out = f"{self.root / 'Dataset'}:mha"
        onto_fixed = {"reference": "{case}", "reference_group": "FIXED"}
        chains: dict[str, dict[str, list[object]]] = {
            "MOVING": {
                WARPED: [Resample(**onto_fixed, **self._map()), Write(dataset=out, group=WARPED)],
                MOVING_ON_FIXED: [Resample(**onto_fixed), Write(dataset=out, group=MOVING_ON_FIXED)],
            }
        }
        if "moving_labels" in self.case.inputs and "warped_labels" not in self.case.inputs:
            chains["MOVING_LABELS"] = {
                WARPED_LABELS: [
                    Resample(**onto_fixed, **self._map(), interpolation="nearest"),
                    Write(dataset=out, group=WARPED_LABELS),
                ]
            }
        if "fixed_landmarks" in self.case.inputs:
            chains["FIXED_LANDMARKS"] = {
                MOVED_LANDMARKS: [
                    TransformPoints(transforms=self._point_map()),
                    Write(dataset=out, group=MOVED_LANDMARKS),
                ]
            }
        roots = self.case.dataset_roots()
        if "fixed_landmarks" in self.case.inputs:  # points are read by the SimpleITK backend from their own root
            roots.append(f"{self._points_root()}:mha")
        try:
            transform(
                "derived", roots, chains, cpu=1, quiet=True, overwrite=True, transforms_dir=self.root / "Transforms"
            )
        except KonfAIError as error:
            raise CaseError(f"KonfAI could not derive the registration's warped data: {error}") from error
        self._ran = True

    def _points_root(self) -> Path:
        """The fixed landmarks staged as a KonfAI entry (a fiducial file beside the images)."""
        from konfai.utils.dataset import link_entry

        root = self.root / "Points"
        link_entry(self.case.inputs["fixed_landmarks"], root / self.case.slug / "FIXED_LANDMARKS.fcsv")
        return root

    def _derived(self, group: str) -> Volume:
        if group not in self._volumes:
            self.ensure()
            self._volumes[group] = Volume(self.dataset, group, self.case.slug, group.lower())
        return self._volumes[group]

    # ------------------------------------------------------------------ what is read back

    def warped(self) -> Volume:
        """The moving image pulled onto the fixed grid through the registration."""
        return self._derived(WARPED)

    def moving_on_fixed(self) -> Volume:
        """The moving image on the fixed grid with no registration: the start."""
        return self._derived(MOVING_ON_FIXED)

    def warped_labels(self) -> Volume | None:
        if "warped_labels" in self.case.inputs:
            return self.case.volume("warped_labels")
        if "moving_labels" not in self.case.inputs:
            return None
        return self._derived(WARPED_LABELS)

    def moved_landmarks(self) -> np.ndarray | None:
        """The fixed landmarks carried by the registration into the moving image: ``x + u(x)``, as ITK reads it."""
        if "fixed_landmarks" not in self.case.inputs:
            return None
        self.ensure()
        path = self.root / "Dataset" / self.case.slug / f"{MOVED_LANDMARKS}.fcsv"
        points = read_landmarks(path)
        return None if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def field(self) -> Volume:
        """The displacement field on the fixed grid: the given one, or the transform file sampled on it once."""
        if "field" in self.case.inputs:
            return self.case.volume("field")
        if FIELD not in self._volumes:
            if not self.dataset.is_dataset_exist(FIELD, self.case.slug):
                self._sample_transform()
            self._volumes[FIELD] = Volume(self.dataset, FIELD, self.case.slug, "field")
        return self._volumes[FIELD]

    def _sample_transform(self) -> None:
        """A dense field of a transform file on the fixed grid, for the views that draw a field; the transform is
        read by KonfAI's transform backend, sampled by SimpleITK (KonfAI resamples through a transform but writes
        no field of it)."""
        import SimpleITK as sitk

        fixed = self.case.volume("fixed")
        transform = self.case.dataset_of("transform").read_transform("TRANSFORM", self.case.slug)
        grid = fixed.grid
        reference = sitk.Image([int(n) for n in reversed(grid.size_zyx)], sitk.sitkUInt8)
        reference.SetSpacing([float(s) for s in grid.spacing_xyz])
        reference.SetOrigin([float(o) for o in grid.origin_xyz])
        reference.SetDirection([float(v) for v in np.asarray(grid.direction_xyz).ravel()])
        sampler = sitk.TransformToDisplacementFieldFilter()
        sampler.SetReferenceImage(reference)
        sampler.SetOutputPixelType(sitk.sitkVectorFloat32)
        self.dataset.write(FIELD, self.case.slug, sampler.Execute(transform))  # the image carries its geometry
