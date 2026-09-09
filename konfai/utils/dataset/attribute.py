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


"""The geometry sidecar and the conversions between arrays, SimpleITK images and transforms."""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.errors import DatasetManagerError


def _attribute_text(value: Any) -> str:
    """One value as an attribute holds it: its printed form, on one line, complete and exact.

    The printed form is left as each type prints it: :meth:`Attribute.get_np_array` reads both
    forms of a sequence and ``ast.literal_eval`` (:meth:`Dataset.read_transform`) needs the Python
    one. No value is elided, and a float is printed as the shortest text that reads back to the
    same float64.
    """
    if type(value) is str:
        return value.replace("\n", "")
    if isinstance(value, torch.Tensor):
        # A tensor from any device: attributes are host-side strings.
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic | np.ndarray) and np.issubdtype(value.dtype, np.floating):
        value = np.asarray(value, dtype=np.float64)[()] if isinstance(value, np.generic) else value.astype(np.float64)
    with np.printoptions(threshold=sys.maxsize, floatmode="unique"):
        return str(value).replace("\n", "")


def region_geometry(
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray,
    spatial_slices: tuple[slice, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """The geometry record of the samples a normalized spatial slice keeps: the first sample's
    world position as the origin, the spacing scaled by the step.

    Shared by every backend. ``spatial_slices`` arrive array-ordered (``(Z)YX``,
    ``slice.indices``-normalized); geometry is ``(x, y, z)``.
    """
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    matrix = np.asarray(direction, dtype=np.float64).reshape(len(spacing), len(spacing))
    start_xyz = np.asarray([item.start for item in reversed(spatial_slices)], dtype=np.float64)
    step_xyz = np.asarray([item.step for item in reversed(spatial_slices)], dtype=np.float64)
    return origin + matrix @ (start_xyz * spacing), spacing * step_xyz


class Attribute(dict[str, Any]):
    """Metadata container storing repeated values with a stack-like naming scheme.

    Values are text, always; assignment and construction both normalize. Copying one is a dict
    copy.
    """

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        super().__init__()
        if not attributes:
            return
        if type(attributes) is Attribute:
            super().update(attributes)
            return
        for k, v in attributes.items():
            super().__setitem__(k if type(k) is str else copy.deepcopy(k), _attribute_text(v))

    @staticmethod
    def _is_stack_member(stored_key: str, key: str) -> bool:
        # Values are stacked as ``{key}_{n}``; a sibling sharing a prefix (``SpacingOriginal``) is not one.
        if stored_key == key:
            return True
        prefix = f"{key}_"
        return stored_key.startswith(prefix) and stored_key[len(prefix) :].isdigit()

    def _count_key(self, key: str) -> int:
        return sum(1 for k in super().keys() if Attribute._is_stack_member(k, key))

    def __getitem__(self, key: str) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return str(super().__getitem__(f"{key}_{i - 1}"))
        if key in super().keys():
            return str(super().__getitem__(key))
        raise DatasetManagerError(
            f"'{key}' is not in the case's attributes.",
            "A stage reads a statistic an earlier one records: check the chain's order.",
        )

    def __setitem__(self, key: str, value: Any) -> None:
        result = _attribute_text(value)
        if "_" not in key:
            super().__setitem__(f"{key}_{self._count_key(key)}", result)
        else:
            super().__setitem__(key, result)

    def pop(self, key: str, default: Any = None) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return super().pop(f"{key}_{i - 1}")
        if key in super().keys():
            return super().pop(key)
        raise DatasetManagerError(
            f"'{key}' is not in the case's attributes.",
            "A stage reads a statistic an earlier one records: check the chain's order.",
        )

    @staticmethod
    def _parse_array(text: str) -> np.ndarray:
        """Both printed forms of a sequence: NumPy's ``[1.5 1.5 2.]`` and Python's ``[1.5, 1.5, 2.0]``."""
        return np.fromstring(text[1:-1].replace(",", " "), sep=" ", dtype=np.double)

    @staticmethod
    def _parsed_array(key: str, text: str) -> np.ndarray:
        """:meth:`_parse_array`, refusing by name what does not parse back as a flat array."""
        try:
            return Attribute._parse_array(text)
        except ValueError:
            raise DatasetManagerError(
                f"'{key}' does not parse back as a flat array: it holds '{text}'.",
                "Only flat scalars and 1-D arrays round-trip through the sidecar:"
                " flatten the value where it is recorded.",
            ) from None

    def get_np_array(self, key: str) -> np.ndarray:
        return Attribute._parsed_array(key, self[key])

    def get_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)

    def pop_np_array(self, key: str) -> np.ndarray:
        return Attribute._parsed_array(key, self.pop(key))

    def pop_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(Attribute._is_stack_member(k, key) for k in super().keys())


def is_an_image(attributes: Attribute) -> bool:
    """Return whether the given attribute set contains image geometry metadata."""
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes


def as_channel_first(data: np.ndarray, attributes: Attribute) -> np.ndarray:
    """Give back its channel axis to a block that folded it away, where the header says it did.

    An array with as many axes as the geometry has spatial axes is a single-channel image. A block
    with no header is handed back untouched.
    """
    if "Spacing" in attributes and data.ndim == len(attributes.get_np_array("Spacing")):
        return data[None]
    return data


def data_to_image(data: np.ndarray | torch.Tensor, attributes: Attribute) -> sitk.Image:
    """Convert a NumPy array and KonfAI attributes into a SimpleITK image."""
    if isinstance(data, torch.Tensor):
        # A tensor on any device: SimpleITK works on host arrays.
        data = data.detach().cpu().numpy()
    if not is_an_image(attributes):
        raise DatasetManagerError(
            "The entry is not an image.",
            "This reader serves volumes; a transform or a point set is read by its own backend.",
        )
    if data.dtype == np.float16:
        # ITK has no half-float pixel type; the streamed .mha writer widens the same way.
        data = data.astype(np.float32)
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i + 1 for i in range(len(data.shape) - 1)] + [0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    for k, v in attributes.items():
        if v and len(v):
            image.SetMetaData(k, v)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
    return image


# Set on an entry read back from a store that types its component axis as an RFC-5 displacement
# field, so ``Dataset.read_transform`` can rebuild the transform. The underscore keeps the key out
# of ``Attribute.__setitem__``'s stack renaming.
DISPLACEMENT_FIELD_ATTRIBUTE = "konfai_displacement_field"


def displacement_field_to_data(transform: sitk.Transform, name: str) -> tuple[np.ndarray, Attribute]:
    """A displacement-field transform as a channel-first array plus its geometry.

    The counterpart of ``_encode_transform_leaves`` for a field, which travels as an image; the
    store records what it is (``write_ome_zarr(displacement_field=True)``).
    """
    if not isinstance(transform, sitk.DisplacementFieldTransform):
        raise DatasetManagerError(
            f"Expected a DisplacementFieldTransform for entry '{name}', got '{type(transform).__name__}'."
        )
    return image_to_data(transform.GetDisplacementField())


def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    """Convert a SimpleITK image into a channel-first NumPy array and attributes."""
    attributes = Attribute()
    for k in image.GetMetaDataKeys():
        # ``ITK_*`` keys are the reader's own bookkeeping, not the volume's metadata.
        if not k.startswith("ITK_"):
            attributes[k] = image.GetMetaData(k)
    # After the metadata import: the metadata may carry a stale geometry stack, and the header must
    # land as its latest version.
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    if image.GetNumberOfComponentsPerPixel() == 1:
        return np.expand_dims(sitk.GetArrayFromImage(image), 0), attributes
    # One contiguous channel-first copy off ITK's interleaved buffer. np.array and not
    # ascontiguousarray: a view of ITK's buffer would outlive the image.
    return np.array(np.moveaxis(sitk.GetArrayViewFromImage(image), -1, 0), order="C"), attributes


def ome_zarr_attributes(metadata: dict[str, Any]) -> Attribute:
    """A KonfAI ``Attribute`` (Origin / Spacing / Direction) from an OME-Zarr entry's metadata.

    The store's konfai sidecar carries the Direction matrix, which NGFF cannot express, and every
    other key it recorded; Direction defaults to identity without it. The sidecar describes one
    level, the finest: its Spacing and Origin are trusted only where its Spacing is this level's
    scale, and any other level takes both from its own transforms.
    """
    attributes = Attribute(metadata.get("attributes", {}))
    axes = metadata["axes"]
    scale = dict(zip(axes, metadata.get("scale", []), strict=False))
    translation = dict(zip(axes, metadata.get("translation", []), strict=False))
    spatial_axes = [axis for axis in ("x", "y", "z") if axis in axes]
    level_spacing = np.asarray([scale.get(axis, 1.0) for axis in spatial_axes])
    level_origin = np.asarray([translation.get(axis, 0.0) for axis in spatial_axes])
    if "Spacing" in attributes:
        recorded = attributes.get_np_array("Spacing")
        if recorded.shape != level_spacing.shape or not np.allclose(recorded, level_spacing, rtol=1e-6, atol=0.0):
            # Another level than the sidecar's. Popped then set, so the key keeps its place in the stack.
            attributes.pop("Spacing")
            attributes["Spacing"] = level_spacing
            if "Origin" in attributes:
                attributes.pop("Origin")
            attributes["Origin"] = level_origin
    if "Spacing" not in attributes:
        attributes["Spacing"] = level_spacing
    if "Origin" not in attributes:
        attributes["Origin"] = level_origin
    if "Direction" not in attributes:
        attributes["Direction"] = np.eye(len(spatial_axes), dtype=np.float64).flatten()
    attributes["OMEAxes"] = np.asarray(axes)
    return attributes


def _flatten_transforms(transform: sitk.Transform) -> list[sitk.Transform]:
    """The leaf transforms of a (possibly nested) composite, in application order."""
    if isinstance(transform, sitk.CompositeTransform):
        leaves: list[sitk.Transform] = []
        for i in range(transform.GetNumberOfTransforms()):
            leaves.extend(_flatten_transforms(transform.GetNthTransform(i)))
        return leaves
    return [transform]


def _transform_codec() -> list[tuple[type, str, Any]]:
    """(sitk class, serialized type tag, decode factory) for every supported transform kind."""
    return [
        (sitk.Euler3DTransform, "Euler3DTransform_double_3_3", sitk.Euler3DTransform),
        (sitk.AffineTransform, "AffineTransform_double_3_3", lambda: sitk.AffineTransform(3)),
        (sitk.BSplineTransform, "BSplineTransform_double_3_3", lambda: sitk.BSplineTransform(3)),
    ]


def _encode_transform_leaves(transform: sitk.Transform, name: str, attributes: Attribute) -> list[np.ndarray]:
    """Serialize a (possibly composite) transform: record each leaf's type tag and fixed parameters
    into ``attributes`` (``{i}:Transform`` / ``{i}:FixedParameters``) and return the per-leaf
    parameter arrays, in application order."""
    datas: list[np.ndarray] = []
    for i, leaf in enumerate(_flatten_transforms(transform)):
        type_tag = next((tag for sitk_class, tag, _ in _transform_codec() if isinstance(leaf, sitk_class)), None)
        if type_tag is None:
            raise DatasetManagerError(f"Unsupported transform type '{type(leaf).__name__}' for entry '{name}'.")
        attributes[f"{i}:Transform"] = type_tag
        attributes[f"{i}:FixedParameters"] = leaf.GetFixedParameters()

        datas.append(np.asarray(leaf.GetParameters()))
    return datas


def _decode_transform(transform_type: str, name: str) -> sitk.Transform:
    """A fresh transform instance for a serialized type tag."""
    for _, type_tag, factory in _transform_codec():
        if transform_type == type_tag:
            return factory()
    raise DatasetManagerError(f"Unsupported transform type '{transform_type}' for entry '{name}'.")


def data_to_transform(data: np.ndarray, attributes: Attribute, name: str) -> sitk.Transform:
    """The transform a stored entry holds: a displacement field is its image in float64, which
    ``DisplacementFieldTransform`` requires; any other entry is the parameter rows and type keys of
    ``_encode_transform_leaves``."""
    if DISPLACEMENT_FIELD_ATTRIBUTE in attributes:
        return sitk.DisplacementFieldTransform(data_to_image(np.asarray(data, dtype=np.float64), attributes))
    transforms = []
    for i, transform_type in enumerate(v for k, v in attributes.items() if k.endswith(":Transform_0")):
        transform = _decode_transform(transform_type, name)
        transform.SetFixedParameters(ast.literal_eval(attributes[f"{i}:FixedParameters"]))
        transform.SetParameters(tuple(data[i]))
        transforms.append(transform)
    return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]


def get_infos(filename: str | Path) -> tuple[list[int], Attribute]:
    """Read shape and metadata from an image file without loading its full pixel data."""
    attributes = Attribute()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(filename))
    file_reader.ReadImageInformation()
    attributes["Origin"] = np.asarray(file_reader.GetOrigin())
    attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
    attributes["Direction"] = np.asarray(file_reader.GetDirection())
    for k in file_reader.GetMetaDataKeys():
        attributes[k] = file_reader.GetMetaData(k)
    # SimpleITK GetSize() is (x, y, [z], ...); KonfAI arrays are [C, (Z), Y, X]: reversed for every rank.
    size = list(reversed(file_reader.GetSize()))
    size = [file_reader.GetNumberOfComponents(), *size]
    return size, attributes
