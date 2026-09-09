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


"""A group's transform chain and what a patched route requires of it."""

from collections.abc import Iterator, Mapping

from konfai import konfai_root, konfai_state
from konfai.data.transform import (
    Expand,
    LocalityKind,
    Transform,
    TransformInverse,
    TransformLoader,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError
from konfai.utils.runtime import State


def _check_patch_transform_locality(transform: Transform, group_src: str, group_dest: str) -> None:
    """Reject a transform whose per-patch result cannot equal its case-level result.

    Only POINTWISE and GLOBAL_STAT are correct on one patch; per-patch GLOBAL_STAT derives the statistic
    from that patch. The probe passes an empty ``Attribute``, so a kind decided by the image answers
    WHOLE_VOLUME.
    """
    kind = transform.patch_locality(Attribute()).kind
    if kind in (LocalityKind.POINTWISE, LocalityKind.GLOBAL_STAT):
        return
    name = type(transform).__name__
    location = f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}"
    reasons = {
        LocalityKind.SLAB: (f"'{name}' needs the output's slabs in order, which only the whole-volume write hands it."),
        LocalityKind.HALO: (
            f"'{name}' reads a neighbourhood around each voxel, so run per-patch it would see no data"
            " beyond the patch border and corrupt every patch edge. Running it per-patch is not"
            " supported yet."
        ),
        LocalityKind.ORIENTATION: (
            f"'{name}' reorients its input: applied to one patch it reorients that patch about its own"
            " extent, which is not the whole volume reoriented and then cut into patches."
        ),
        LocalityKind.CROP: (
            f"'{name}' crops its input to a box measured on the whole volume: applied to one patch it"
            " crops that patch about its own extent, and cuts the patch grid predictions are"
            " reassembled onto down to what is left."
        ),
        LocalityKind.REGRID: (
            f"'{name}' resamples its input onto another grid: applied to one patch it would rescale"
            " that patch about its own extent, or hand back the whole target extent: neither of"
            " which is the patch grid predictions are reassembled onto."
        ),
        LocalityKind.WHOLE_VOLUME: f"'{name}' needs the whole volume.",
    }
    raise ConfigError(
        f"{location}.patch_transforms: {reasons.get(kind, f'{name!r} declares {kind.name}, which a patch cannot serve.')}",
        f"Move '{name}' to {location}.transforms, where it runs once on the whole volume.",
    )


def _check_patch_transform_shape(transform: Transform, group_src: str, group_dest: str) -> None:
    """Reject a patch_transform that resizes the patch it is handed.

    The patch grid is folded from the CASE-level ``transforms`` only, so a patch_transform must be spatially
    shape-preserving. Only the spatial shape is checked: ``transform_shape`` is asked with the
    channel-stripped shape, so a transform that changes only the channel count is not caught here.
    """
    spatial_shape = [7, 11, 13]
    shape = list(transform.transform_shape(group_src, "", list(spatial_shape), Attribute()))
    if shape == spatial_shape:
        return
    name = type(transform).__name__
    location = f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}"
    raise ConfigError(
        f"{location}.patch_transforms: '{name}' changes the spatial shape of its input"
        f" ({spatial_shape} -> {shape}), but a patch must keep the shape the patch grid cut it to.",
        f"Move '{name}' to {location}.transforms, where the patch grid is folded from its"
        " transform_shape(); a patch_transform must be spatially shape-preserving.",
    )


def _check_patch_transform_invertible(
    transform: Transform, case_transforms: list[Transform], group_src: str, group_dest: str
) -> None:
    """Reject a per-patch global statistic at prediction, whose inverse cannot be reconstructed.

    A per-patch statistic never reaches the case attribute, so the finalize inverse, seeded from the case
    attribute, has nothing to pop. A case-level ``transforms`` entry deriving the same statistic
    (``Standardize(lazy=True)`` caches Mean/Std and applies nothing) rescues it. Gated on the prediction
    state, the only one that inverts patch_transforms.
    """
    if konfai_state() != str(State.PREDICTION):
        return
    if not (isinstance(transform, TransformInverse) and transform.apply_inverse):
        return
    locality = transform.patch_locality(Attribute())
    if locality.kind is not LocalityKind.GLOBAL_STAT:
        return
    if any(
        (case_locality := case.patch_locality(Attribute())).kind is LocalityKind.GLOBAL_STAT
        and locality.stat_keys <= case_locality.stat_keys
        for case in case_transforms
    ):
        return
    name = type(transform).__name__
    location = f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}"
    raise ConfigError(
        f"{location}.patch_transforms: '{name}' derives its statistic from each patch, but prediction"
        " must invert it and a per-patch statistic cannot be un-applied to the reassembled volume.",
        f"Capture the volume-global statistic case-level instead: put '{name}(lazy=True)' in"
        f" {location}.transforms (it traverses the whole volume, caches the statistic and applies"
        f" nothing), and keep '{name}()' in patch_transforms to consume it.",
    )


class GroupTransform:
    """Collection of transforms attached to one source-to-destination group path."""

    def __init__(
        self,
        transforms: dict[str, TransformLoader] | None = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
        patch_transforms: dict[str, TransformLoader] | None = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
        is_input: bool = True,
    ) -> None:
        self._transforms = transforms
        self._patch_transforms = patch_transforms
        self.transforms: list[Transform] = []
        self.patch_transforms: list[Transform] = []
        self.is_input = is_input
        self._prepared = False

    def prepare(self, group_src: str, group_dest: str) -> None:
        # Binds ONCE: a workflow may inspect its chains before Data.prepare() calls this for every group,
        # and a stage's __init__ is user code that may not be run twice.
        if self._prepared:
            return
        self._prepared = True
        self.transforms = []
        self.patch_transforms = []
        if self._transforms is not None:
            for classpath, transform_loader in self._transforms.items():
                transform = transform_loader.get_transform(
                    classpath,
                    konfai_args=f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}.transforms",
                    # Past an Expand marker the chain is the copies' draws: a name both packages have
                    # (Flip, Mask, Permute) is the draw there, the transform before it.
                    prefer_augmentation=any(isinstance(stage, Expand) for stage in self.transforms),
                )
                self.transforms.append(transform)
        if self._patch_transforms is not None:
            for classpath, transform_loader in self._patch_transforms.items():
                transform = transform_loader.get_transform(
                    classpath,
                    konfai_args=f"{konfai_root()}.Dataset.groups_src.{group_src}"
                    f".groups_dest.{group_dest}.patch_transforms",
                )
                _check_patch_transform_locality(transform, group_src, group_dest)
                _check_patch_transform_shape(transform, group_src, group_dest)
                _check_patch_transform_invertible(transform, self.transforms, group_src, group_dest)
                self.patch_transforms.append(transform)

    def set_datasets(self, datasets: list[Dataset]) -> None:
        for transform in self.transforms:
            transform.set_datasets(datasets)
        for transform in self.patch_transforms:
            transform.set_datasets(datasets)

    def to(self, device: int):
        for transform in self.transforms:
            transform.to(device)
        for transform in self.patch_transforms:
            transform.to(device)

    def __str__(self) -> str:
        params = {"transforms": self.transforms, "patch_transforms": self.patch_transforms}
        return str(params)

    def __repr__(self) -> str:
        return str(self)


class GroupTransformMetric(GroupTransform):
    """Metric-specific group transform that omits patch-time transforms."""

    def __init__(
        self,
        transforms: dict[str, TransformLoader] = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
    ):
        super().__init__(transforms, {})


class GroupTransformOut(GroupTransform):
    """Transform-workflow group: a plain chain, no patch-time transforms, no ``is_input``.

    Every group of a dataset-preparation workflow is an input."""

    def __init__(
        self,
        transforms: dict[str, TransformLoader] = {
            "default|Normalize|Standardize|TensorCast|ResampleIsotropic|Write": TransformLoader()
        },
    ):
        super().__init__(transforms, {})


class Group(dict[str, GroupTransform]):
    """Mapping of destination group names to transform pipelines."""

    def __init__(
        self,
        groups_dest: dict[str, GroupTransform] = {"default|Labels": GroupTransform()},
    ):
        super().__init__(groups_dest)


class GroupMetric(dict[str, GroupTransformMetric]):
    """Metric-oriented variant of :class:`Group` used during evaluation."""

    def __init__(
        self,
        groups_dest: dict[str, GroupTransformMetric] = {"default|group_dest": GroupTransformMetric()},
    ):
        super().__init__(groups_dest)


class GroupOut(dict[str, GroupTransformOut]):
    """Transform-workflow variant of :class:`Group`."""

    def __init__(
        self,
        groups_dest: dict[str, GroupTransformOut] = {"default|group_dest": GroupTransformOut()},
    ):
        super().__init__(groups_dest)


def _chains(groups_src: Mapping[str, Group | GroupMetric | GroupOut]) -> Iterator[tuple[str, str, GroupTransform]]:
    """Every ``(group_src, group_dest, chain)`` of the config, in declaration order."""
    for group_src, group in groups_src.items():
        for group_dest, chain in group.items():
            yield group_src, group_dest, chain
