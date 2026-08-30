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
    """One value as an attribute holds it: its printed form, on one line and complete.

    The single place a value stops being a live object, because every consumer takes text --
    ``SetMetaData`` on a SimpleITK image, an h5 attribute, a zarr sidecar. The printed form is left
    as each type prints it: a sequence has two, :meth:`Attribute.get_np_array` reads both, and
    ``ast.literal_eval`` (:meth:`Dataset.read_transform`, on the parameter keys) needs the Python
    one, so normalising to either here breaks the other reader.

    Complete, because an attribute is a record and not a display: NumPy's own printing elides values
    past a threshold, and an elided record is one no reader can parse back. Exact, for the same
    reason: a float is printed as the shortest text that reads back to the very same float64 --
    NumPy's default (8 decimals for an array, the float32-shortest form for a float32 scalar) is a
    display, and a statistic that came back a few ulps off made the whole-volume and the streamed
    path of one chain disagree by that much (measured: a Min/Max rescale, 28% of voxels one ulp
    apart on CUDA).
    """
    if type(value) is str:
        return value.replace("\n", "")  # what the printing below does to a str, without entering it
    if isinstance(value, torch.Tensor):
        # Accept a tensor from any device: attributes are host-side strings, and finalize transforms
        # (Normalize, Statistics, ...) may hand over stats computed on a CUDA-resident volume.
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic | np.ndarray) and np.issubdtype(value.dtype, np.floating):
        value = np.asarray(value, dtype=np.float64)[()] if isinstance(value, np.generic) else value.astype(np.float64)
    with np.printoptions(threshold=sys.maxsize, floatmode="unique"):
        return str(value).replace("\n", "")


class Attribute(dict[str, Any]):
    """Metadata container storing repeated values with a stack-like naming scheme.

    Values are text, always. Both doors normalize (assignment and construction), so an attribute
    built from a store's own sidecar, which JSON hands back as live lists, is the same thing as one
    assigned in Python. Anything less and a value can be stored and not written back out.

    Copying one is a dict copy: its values are text already, and the streamed route of every
    workflow copies the case's attributes three to five times per patch (measured 69 us for a
    23-key copy through the normalising door, 2 us at dict level).
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
        # Values are stacked as ``{key}_{n}``; match that exact pattern (or the bare key) so a sibling that
        # merely shares a prefix (``SpacingOriginal`` vs ``Spacing``) is not miscounted as another entry.
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
        """Both printed forms of a sequence: NumPy's ``[1.5 1.5 2.]`` and Python's ``[1.5, 1.5, 2.0]``.

        Which one an attribute holds follows from what the writer handed over (an ``ndarray``, or
        the plain list a JSON sidecar gives back), and no reader should have to make that
        distinction. ``np.fromstring`` reads whitespace only, so the commas go first.
        """
        return np.fromstring(text[1:-1].replace(",", " "), sep=" ", dtype=np.double)

    def get_np_array(self, key: str) -> np.ndarray:
        return Attribute._parse_array(self[key])

    def get_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)

    def pop_np_array(self, key: str) -> np.ndarray:
        return Attribute._parse_array(self.pop(key))

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

    A stage may hand back a volume without its channel axis (``Sum(dim=0)`` and ``MergeLabels`` fold
    the leading axis, which in a TRANSFORM chain is the channel one). An array with as many axes as
    the geometry has spatial axes IS a single-channel image: read as channel-first it would be a 2-D
    image with a plane's worth of channels, refused by ITK or, worse, stored that way in silence.

    The header is what declares the spatial rank, so a block that comes with none is handed back
    untouched: only the caller knows whether it can be written as it is or must be refused.
    """
    if "Spacing" in attributes and data.ndim == len(attributes.get_np_array("Spacing")):
        return data[None]
    return data


def data_to_image(data: np.ndarray, attributes: Attribute) -> sitk.Image:
    """Convert a NumPy array and KonfAI attributes into a SimpleITK image."""
    if isinstance(data, torch.Tensor):
        # Accept a torch tensor on any device: SimpleITK works on host arrays, so a SITK-backed transform
        # fed a CUDA-resident volume converts here and naturally returns on the CPU (the pipeline then
        # continues on the CPU). This keeps every transform usable regardless of the volume's device.
        data = data.detach().cpu().numpy()
    if not is_an_image(attributes):
        raise DatasetManagerError(
            "The entry is not an image.",
            "This reader serves volumes; a transform or a point set is read by its own backend.",
        )
    if data.dtype == np.float16:
        # ITK has no half-float pixel type (GetImageFromArray rejects float16), so widen to float32 --
        # exact and lossless. The streamed .mha writer widens the same way, so both write identical bytes.
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
# field, so ``Dataset.read_transform`` can rebuild the transform. The underscore matters: a key
# without one is stack-renamed by ``Attribute.__setitem__`` (``Transform`` becomes ``Transform_0``).
DISPLACEMENT_FIELD_ATTRIBUTE = "konfai_displacement_field"


def displacement_field_to_data(transform: sitk.Transform, name: str) -> tuple[np.ndarray, Attribute]:
    """A displacement-field transform as a channel-first array plus its geometry.

    The counterpart of ``_encode_transform_leaves`` for the one transform kind that cannot go through
    it: a displacement field's parameters ARE the field, so serialising it as a parameter vector
    would drop the geometry that makes it meaningful. It travels as an image instead, and the store
    records what it is (see ``write_ome_zarr(displacement_field=True)``).
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
        # ``ITK_*`` keys are the reader's own bookkeeping (the input filter's name, the file's original
        # direction and spacing), not the volume's metadata: carried into an output they describe the
        # source of a resampled volume, which nothing should read as the output's.
        if not k.startswith("ITK_"):
            attributes[k] = image.GetMetaData(k)
    # AFTER the metadata import, deliberately. data_to_image stamps every attribute -- the
    # geometry stack included -- back onto the image as metadata text, and the loop above imports
    # it verbatim (versioned keys carry a '_', so they land as-is). Recorded first, the header
    # landed as Origin_0 and the stale text then OVERWROTE that very key: an image read, moved
    # (SetOrigin) and written back kept its old origin, silently. Recorded last, the header
    # appends the next version of the stack, which is the one every reader takes.
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    if image.GetNumberOfComponentsPerPixel() == 1:
        return np.expand_dims(sitk.GetArrayFromImage(image), 0), attributes
    # One copy, written channel-first straight off ITK's interleaved buffer: the array is contiguous
    # for whatever holds it next, where the copy of the buffer transposed was a strided view every
    # consumer needing a contiguous field copied again (a 3x128^3 float64 field: 50 MiB each time).
    # np.array and not ascontiguousarray: a one-voxel image is contiguous however its axes are moved,
    # and a view of ITK's buffer would outlive the image.
    return np.array(np.moveaxis(sitk.GetArrayViewFromImage(image), -1, 0), order="C"), attributes


def ome_zarr_attributes(metadata: dict[str, Any]) -> Attribute:
    """A KonfAI ``Attribute`` (Origin / Spacing / Direction) from an OME-Zarr entry's metadata.

    The store's konfai sidecar wins when present (it carries the full Direction matrix, which NGFF
    scale/translation cannot express) otherwise geometry falls back to the NGFF transforms, Direction
    defaulting to identity. Shared by the Dataset OME-Zarr reader and ``ITK.read_displacement_field``
    so both recover geometry the one same way.

    THE SIDECAR DESCRIBES ONE LEVEL: the one the writer was handed, and it writes the finest. Every
    level of a pyramid carries its own scale and translation, so a sidecar taken at its word on a
    coarser level put level 0's spacing and origin on level 1's voxels: half the extent along every
    axis, a brain that reads at half its size for anything that asks for ``@1``. The sidecar is
    therefore trusted for Spacing and Origin only where its Spacing IS this level's scale; on any other level those two come from the level's
    own transforms, and the sidecar still supplies what NGFF cannot: the Direction, and every other
    key it recorded.
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
            # Another level than the one the sidecar was written for: its own geometry, not the
            # sidecar's. Popped then set, so the key keeps its place in the stack rather than
            # gaining a rung that a later write would record twice.
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
    """The leaf transforms of a (possibly nested) composite, in application order.

    ``CompositeTransform.GetNthTransform`` can itself return a composite, so a single-level walk
    leaves a nested composite in the list and the serializer rejects it. Recurse to the leaves.
    """
    if isinstance(transform, sitk.CompositeTransform):
        leaves: list[sitk.Transform] = []
        for i in range(transform.GetNumberOfTransforms()):
            leaves.extend(_flatten_transforms(transform.GetNthTransform(i)))
        return leaves
    return [transform]


def _transform_codec() -> list[tuple[type, str, Any]]:
    """(sitk class, serialized type tag, decode factory) for every supported transform kind.

    Built lazily because ``sitk`` is an optional import.
    """
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
    """The transform a stored entry holds: a displacement field is its image in float64, what
    ``DisplacementFieldTransform`` requires, widened here exactly so the image is built once in that
    type; any other entry is the parameter rows and type keys of ``_encode_transform_leaves``."""
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
    # SimpleITK GetSize() is (x, y, [z], ...); KonfAI arrays are numpy-order [C, (Z), Y, X], so the
    # spatial size must be reversed for EVERY rank: a 3-D-only reversal transposes 2-D/4-D data.
    size = list(reversed(file_reader.GetSize()))
    size = [file_reader.GetNumberOfComponents(), *size]
    return size, attributes
