# SPDX-License-Identifier: Apache-2.0
"""What a registration case needs derived before it is looked at: the moving data on the fixed grid, before and
after the registration, and the fixed landmarks carried by the field. SimpleITK does the resampling, with ITK's
reading of a displacement field (the fixed point x matches x + u(x) in the moving image)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .case import CaseError, InspectionCase, Volume


class Derived:
    """Files Scout derives once per inspection, under ``work``, read back as volumes."""

    def __init__(self, case: InspectionCase, work: Path) -> None:
        self.case = case
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self._volumes: dict[str, Volume] = {}

    def _sitk(self):
        import SimpleITK as sitk

        return sitk

    def transform(self):
        """The registration as a SimpleITK transform: the field image, or the transform file."""
        sitk = self._sitk()
        if "field" in self.case.inputs:
            field = sitk.ReadImage(str(self.case.inputs["field"]))
            if field.GetNumberOfComponentsPerPixel() != field.GetDimension():
                raise CaseError("field: a displacement field has one component per spatial axis")
            return sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
        return sitk.ReadTransform(str(self.case.inputs["transform"]))

    def field(self) -> Volume:
        """The displacement field on the fixed grid: the given one, or the transform sampled on it once."""
        if "field" in self.case.inputs:
            return self.case.volume("field")
        return self._derived("field", self._sample_transform)

    def _sample_transform(self, path: Path) -> None:
        sitk = self._sitk()
        fixed = self.case.volume("fixed").image()
        sampler = sitk.TransformToDisplacementFieldFilter()
        sampler.SetReferenceImage(fixed)
        sampler.SetOutputPixelType(sitk.sitkVectorFloat32)
        sitk.WriteImage(sampler.Execute(self.transform()), str(path))

    def _derived(self, name: str, make) -> Volume:
        if name not in self._volumes:
            path = self.work / f"{name}.mha"
            if not path.exists():
                make(path)
            self._volumes[name] = Volume(path)
        return self._volumes[name]

    def _resample(self, role: str, transform, interpolation: str, key: str) -> Volume:
        sitk = self._sitk()

        def make(path: Path) -> None:
            moving = self.case.volume(role).image()
            reference = self.case.volume("fixed").image()
            interpolator = sitk.sitkNearestNeighbor if interpolation == "nearest" else sitk.sitkLinear
            fill = 0.0
            if interpolation != "nearest":
                fill = float(sitk.GetArrayViewFromImage(moving).min())
            resampled = sitk.Resample(moving, reference, transform, interpolator, fill, moving.GetPixelID())
            sitk.WriteImage(resampled, str(path))

        return self._derived(key, make)

    def warped(self) -> Volume:
        """The moving image pulled onto the fixed grid through the registration."""
        return self._resample("moving", self.transform(), "linear", "moving_warped")

    def moving_on_fixed(self) -> Volume:
        """The moving image on the fixed grid with no registration: the start."""
        return self._resample("moving", self._sitk().Transform(), "linear", "moving_on_fixed")

    def warped_labels(self) -> Volume | None:
        if "warped_labels" in self.case.inputs:
            return self.case.volume("warped_labels")
        if "moving_labels" not in self.case.inputs:
            return None
        return self._resample("moving_labels", self.transform(), "nearest", "moving_labels_warped")

    def moved_landmarks(self) -> np.ndarray | None:
        """The fixed landmarks carried by the registration into the moving image: ``x + u(x)``, as ITK reads it."""
        points = self.case.landmarks("fixed_landmarks")
        if points is None:
            return None
        transform = self.transform()
        return np.asarray([transform.TransformPoint([float(v) for v in point]) for point in points], dtype=float)
