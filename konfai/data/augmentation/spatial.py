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


"""Draws that move the grid: translation, rotation, scale, flips, permutations, elastic fields."""

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.augmentation.base import (
    DataAugmentation,
    _reflect_interval,
    _rotation_2d_matrix,
    _rotation_3d_matrix,
    _scale_matrix,
    _translate_matrix,
)
from konfai.data.geometry import (
    SIGNED_PERMUTATION_ATOL_FLOAT32,
    AffineMap,
    AxisRemap,
    DisplacementStage,
    Grid,
    WorldBox,
    apply_remap,
    remap_region,
    remap_shape,
    signed_permutation,
)
from konfai.data.sampling import _apply, _displacement_at, _to_index
from konfai.data.transform import LocalityKind, PatchLocality, RegionContext
from konfai.utils.dataset import Attribute


class EulerTransform(DataAugmentation):
    """A draw that resamples the copy through an affine map about the volume's centre.

    The map is stated in the normalised coordinates ``affine_grid`` spans over the whole extent
    (``[-1, 1]`` per axis, ``align_corners=True``), output to source. Sampling goes through
    :meth:`_sample_region`: the whole volume is the region that covers everything, so a streamed
    region and the whole-volume copy run the very same arithmetic and agree to float rounding.

    A REGRID by default, and the declared kind is what routes a streamed region: a REGRID samples
    the target region out of the source box it pulled; any other kind (a HALO shift, an ORIENTATION
    quarter turn) is handed the block its declaration asked for and applies the draw to it whole.
    """

    # A map about the centre displaces a voxel by an amount that grows with its distance to it,
    # so no constant halo bounds the read: each target region pulls the source box it maps to.
    locality = LocalityKind.REGRID

    def __init__(self) -> None:
        super().__init__()
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _grid_matrix(self, index: int, a: int, shape: list[int]) -> torch.Tensor:
        """Copy *a*'s affine, in the normalised coordinates ``affine_grid`` spans over ``shape``."""
        return self.matrix[index][a]

    #: How far inside the grid a mapped box must lie, in voxels, for the float32 coordinates built
    #: from it to lie inside too: the GEMM and the scaling to voxels err by a few float32 ulps of
    #: the extent (under 2e-6 of it), and a coordinate within an ulp of the far face folds to its
    #: mirror. The larger of the two margins applies.
    _INTERIOR_MARGIN = 1e-3
    _INTERIOR_MARGIN_RELATIVE = 1e-5

    @staticmethod
    def _mapped_box(
        matrix: torch.Tensor, target: tuple[slice, ...], full: tuple[int, ...]
    ) -> list[tuple[float, float]]:
        """Where the region's corners map to, per array axis, in source voxel indices of the full
        grid: the hull of the affine image of the region's box, in float64."""
        n = len(full)
        affine = matrix[0].to(torch.float64).numpy()

        def normalised(position: int, extent: int) -> float:
            # affine_grid's coordinate of a voxel of the FULL extent (a singleton axis sits at 0).
            return -1.0 + 2.0 * position / (extent - 1) if extent > 1 else 0.0

        # The region's first and last voxel per axis, in (x, y, z): affine_grid's order.
        ends = np.array(
            [
                [normalised(part.start, extent), normalised(part.stop - 1, extent)]
                for part, extent in zip(target, full, strict=True)
            ]
        )[::-1]
        box = WorldBox(ends[:, 0], ends[:, 1]).image_under(AffineMap(affine[:n, :n], affine[:n, n]))
        return [
            (
                float((box.low_xyz[n - 1 - axis] + 1.0) / 2.0 * float(extent - 1)),
                float((box.high_xyz[n - 1 - axis] + 1.0) / 2.0 * float(extent - 1)),
            )
            for axis, extent in enumerate(full)
        ]

    @classmethod
    def _interior(cls, box: list[tuple[float, float]], full: tuple[int, ...]) -> bool:
        """Whether every coordinate of a region mapping to ``box`` lies in ``[0, extent - 1]``."""
        for (low, high), extent in zip(box, full, strict=True):
            span = float(extent - 1)
            margin = max(cls._INTERIOR_MARGIN, cls._INTERIOR_MARGIN_RELATIVE * span)
            if extent < 2 or low < margin or high > span - margin:
                return False
        return True

    @staticmethod
    def _source_coordinates(
        matrix: torch.Tensor, target: tuple[slice, ...], full: tuple[int, ...], device: torch.device | None = None
    ) -> torch.Tensor:
        """Where each target voxel samples from, in source VOXEL indices of the full grid, per axis in
        array order: ``[*region_shape, n]``, on ``device``. Reflected into the volume as
        ``padding_mode='reflection'`` would (``align_corners=True``: mirrored about the outer voxel
        centres) and clipped.

        In place past the affine map: each step is the op it always was, rounded once, on the one
        tensor, where a chain of full-grid temporaries moved ~200 bytes per voxel for a 12-byte
        result (measured 867 MiB of growth for an 81 MiB result at 192^3). A region whose mapped
        box is interior skips the reflection and the clip: both are the identity on ``[0, n - 1]``.
        """
        n = len(full)
        # affine_grid's own base grid, restricted to the region: linspace over the FULL extent (a
        # singleton axis sits at 0, as affine_grid places it). Built on the host and moved, so every
        # device meshes the same bits.
        axes = [
            (torch.linspace(-1.0, 1.0, extent, dtype=torch.float32) if extent > 1 else torch.zeros(1))[part].to(device)
            for extent, part in zip(full, target, strict=True)
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        # affine_grid orders a coordinate (x, y, z): the last array axis first.
        homogeneous = torch.stack([*reversed(mesh), torch.ones_like(mesh[0])], dim=-1)
        weights = matrix[0, :n, :].T
        if device is None or device.type == "cpu":
            source = homogeneous.reshape(-1, n + 1) @ weights.to(torch.float32)
        else:
            # In float64 on a device: a float32 matmul there follows the process's TF32 setting,
            # ten bits of mantissa, a thousandth of a voxel on a 512 axis. Sixteen flops per voxel.
            source = (homogeneous.reshape(-1, n + 1).to(torch.float64) @ weights.to(device, torch.float64)).to(
                torch.float32
            )
        del homogeneous, mesh
        index = source.reshape(*[part.stop - part.start for part in target], n).flip(-1)  # back to array order
        del source
        extents = torch.tensor([float(extent - 1) for extent in full], dtype=torch.float32, device=device)
        index.add_(1.0).div_(2.0).mul_(extents)
        if EulerTransform._interior(EulerTransform._mapped_box(matrix, target, full), full):
            return index
        # reflect_coordinates(in, 0, 2 * (size - 1)) then clip, as torch's kernel does.
        span = extents.clamp(min=1.0)
        magnitude = index.abs_()
        extra = torch.remainder(magnitude, span)
        flips = magnitude.div_(span).floor_()
        mirrored = torch.sub(span, extra)
        reflected = torch.where(flips.remainder_(2) == 0, extra, mirrored, out=extra)
        del flips, mirrored
        reflected.masked_fill_(extents <= 0, 0.0)
        torch.maximum(reflected, torch.zeros((), dtype=torch.float32, device=device), out=reflected)
        return torch.minimum(reflected, extents, out=reflected)

    def _sample_region(
        self,
        matrix: torch.Tensor,
        block: torch.Tensor,
        source: tuple[slice, ...],
        target: tuple[slice, ...],
        full: tuple[int, ...],
    ) -> torch.Tensor:
        """Sample the target region from ``block`` (the source region ``source`` of the full grid).

        Integer tensors are label maps: interpolating them blends class ids into non-existent labels,
        so they are resampled with nearest-neighbour instead. The coordinates are the full grid's,
        re-expressed on the block: every one lies inside it, since the pull kept what a reflection
        reads back from.
        """
        mode = "nearest" if not block.dtype.is_floating_point else "bilinear"
        coordinates = self._source_coordinates(matrix, target, full, block.device)
        starts = torch.tensor([float(part.start) for part in source], dtype=torch.float32, device=block.device)
        sizes = torch.tensor(
            [float(part.stop - part.start - 1) for part in source], dtype=torch.float32, device=block.device
        )
        local = torch.where(
            sizes > 0, (coordinates - starts) * 2.0 / sizes.clamp(min=1.0) - 1.0, torch.zeros_like(coordinates)
        )
        grid = local.flip(-1).unsqueeze(0)  # grid_sample orders a coordinate (x, y, z)
        return (
            F.grid_sample(
                block.unsqueeze(0).type(torch.float32), grid, align_corners=True, mode=mode, padding_mode="border"
            )
            .type(block.dtype)
            .squeeze(0)
        )

    def _sample(self, matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        full = tuple(int(extent) for extent in tensor.shape[1:])
        whole = tuple(slice(0, extent) for extent in full)
        return self._sample_region(matrix, tensor, whole, whole, full)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._sample(self._grid_matrix(index, a, list(tensor.shape[1:])), tensor)

    def _stream_region_source(
        self, index: int, a: int, target_slices: tuple[slice, ...], source_spatial_shape: list[int]
    ) -> list[slice]:
        """The source box a target region samples from: the affine image of the region's box,
        widened by one voxel for the interpolation taps and to what a reflection at the border
        reads back from, clamped to the volume."""
        full = tuple(int(extent) for extent in source_spatial_shape)
        box = self._mapped_box(self._grid_matrix(index, a, list(full)), tuple(target_slices), full)
        pull: list[slice] = []
        for axis, (low, high) in enumerate(box):
            extent = full[axis]
            low, high = _reflect_interval(low - 1.0, high + 1.0, float(extent - 1))
            start = max(0, int(np.floor(low)))
            stop = min(extent, int(np.ceil(high)) + 1)
            pull.append(slice(start, max(stop, start + 1)))
        return pull

    def _stream_region(
        self, name: str, index: int, a: int, tensor: torch.Tensor, context: RegionContext
    ) -> torch.Tensor:
        if self._patch_locality(index, a, Attribute()).kind is not LocalityKind.REGRID:
            return self._compute(name, index, a, tensor)
        return self._sample_region(
            self._grid_matrix(index, a, list(context.source_shape)),
            tensor,
            tuple(context.source),
            tuple(context.target),
            tuple(int(extent) for extent in context.source_shape),
        )

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._sample(self._grid_matrix(index, a, list(tensor.shape[1:])).inverse(), tensor)


class Translate(EulerTransform):
    def __init__(self, t_min: float = -10, t_max=10, is_int: bool = False):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.is_int = is_int
        self.translate: dict[int, torch.Tensor] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        translate = torch.rand((len(shapes), dim)) * torch.tensor(self.t_max - self.t_min) + torch.tensor(self.t_min)
        self.translate[index] = torch.round(translate) if self.is_int else translate
        return shapes

    def _grid_matrix(self, index: int, a: int, shape: list[int]) -> torch.Tensor:
        # The draw is a shift in VOXELS, in (x, y, z). ``affine_grid`` spans [-1, 1] over whatever
        # extent it is given, so the same shift is a different matrix on a patch than on the volume:
        # normalise it against the extent it is about to be applied to, never against a fixed one.
        sizes = torch.tensor(list(reversed(shape)), dtype=torch.float32)
        return torch.unsqueeze(_translate_matrix(self.translate[index][a] * 2.0 / (sizes - 1)), dim=0)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # A uniform shift sends a target patch to that same patch displaced by the draw, so the source
        # is a bounded neighbourhood of it. One voxel past the ceiling covers the far tap a fractional
        # shift interpolates from. The draw is in (x, y, z); a halo is in array order.
        radius = (torch.ceil(self.translate[index][a].abs()).to(torch.int64) + 1).tolist()
        return PatchLocality(LocalityKind.HALO, halo=tuple(int(r) for r in reversed(radius)))


class Rotate(EulerTransform):
    """Rotate a copy of the case about its centre.

    A quarter draw is a signed permutation of the axes: an exact index remap (permute + flip), never an
    interpolation, and it transposes the extents it swaps, so the copy is cut on its own grid. A free
    angle resamples: it streams as a REGRID, each target region pulling the source box its corners map
    to (a slab of a rotated volume pulls a wide band, which the plan prices).
    """

    def __init__(self, a_min: float = 0, a_max: float = 360, is_quarter: bool = False):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.is_quarter = is_quarter

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        func = _rotation_3d_matrix if dim == 3 else _rotation_2d_matrix
        angles = []

        if self.is_quarter:
            quarter_angles = torch.tensor([90.0, 180.0, 270.0])
            choices = torch.randint(0, quarter_angles.numel(), (len(shapes), dim))
            angles = torch.deg2rad(quarter_angles[choices])
        else:
            angles = torch.deg2rad(
                torch.rand((len(shapes), dim)) * torch.tensor(self.a_max - self.a_min) + torch.tensor(self.a_min)
            )

        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in angles]
        # A quarter turn transposes the extents it swaps, so a copy whose draw is one is cut on the grid
        # that draw lands on. A sampled draw keeps the grid it was applied to.
        return [Rotate._draw_shape(self.matrix[index][a], shape) for a, shape in enumerate(shapes)]

    @classmethod
    def _index_remap(cls, matrix: torch.Tensor) -> AxisRemap | None:
        """The exact index remap this draw is, or ``None`` if it must be sampled.

        ``matrix`` maps an output coordinate onto the input it comes from, so it is a signed
        permutation exactly for a quarter turn: the shared predicate decides, at the tolerance of
        the float32 cosines the matrix is composed from.
        """
        return signed_permutation(matrix[0, :-1, :-1], SIGNED_PERMUTATION_ATOL_FLOAT32)

    @classmethod
    def _draw_shape(cls, matrix: torch.Tensor, shape: list[int]) -> list[int]:
        """The spatial extents a draw lands on, given the ones it is applied to.

        A quarter turn carries each extent with the axis it reads; a sampled draw spans the extent
        it is given.
        """
        remap = cls._index_remap(matrix)
        if remap is None:
            return list(shape)
        return remap_shape(shape, remap)

    def _reorient(self, index: int, a: int, matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        remap = Rotate._index_remap(matrix)
        if remap is None:
            return self._sample(matrix, tensor)
        # apply_remap materialises, so the copy never aliases the tensor it was drawn from.
        return apply_remap(tensor, remap)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._reorient(index, a, self._grid_matrix(index, a, list(tensor.shape[1:])), tensor)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._reorient(index, a, self._grid_matrix(index, a, list(tensor.shape[1:])).inverse(), tensor)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # Permuting and mirroring voxels is a bijection on them, which is what ORIENTATION promises and
        # what LocalityKind.preserves_statistics lets a later stage trust. Only the draw can say whether
        # this one is that, and the draw is a property of the copy rather than of the case. Any other
        # angle resamples: a REGRID whose target region pulls the source box its corners map to.
        if Rotate._index_remap(self.matrix[index][a]) is None:
            return PatchLocality(LocalityKind.REGRID)
        return PatchLocality(LocalityKind.ORIENTATION)

    def _stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        # The same extent carry state_init applied to the copy's grid.
        return Rotate._draw_shape(self.matrix[index][a], list(shape))

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        remap = Rotate._index_remap(self.matrix[index][a])
        if remap is None:
            return super()._stream_region_source(index, a, target_slices, source_spatial_shape)
        return remap_region(target_slices, source_spatial_shape, remap)


class Scale(EulerTransform):
    """Scale a copy about the volume centre."""

    def __init__(self, s_std: float = 0.2):
        super().__init__()
        self.s_std = s_std

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        scale = torch.Tensor.repeat(
            torch.exp2(torch.randn(len(shapes)) * self.s_std).unsqueeze(1),
            [1, len(shapes[0])],
        )
        self.matrix[index] = [torch.unsqueeze(_scale_matrix(value), dim=0) for value in scale]
        return shapes


class Flip(DataAugmentation):
    def __init__(self, f_prob: list[float] = [0.33, 0.33, 0.33], vector_field: bool = False) -> None:
        super().__init__()
        self.f_prob = f_prob
        self.vector_field = vector_field
        self.flip: dict[int, list[list[int]]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        prob = torch.rand((len(shapes), len(self.f_prob))) < torch.tensor(self.f_prob)
        dims = torch.tensor([1, 2, 3][: len(self.f_prob)])
        self.flip[index] = [dims[mask].tolist() for mask in prob]
        return shapes

    def _flip(self, tensor: torch.Tensor, dims: list[int]) -> torch.Tensor:
        result = torch.flip(tensor, dims=dims)
        # A displacement/vector field (one channel per spatial axis, channel-first [C=(dx,dy,dz),(D),H,W])
        # is not mirror-invariant: flipping a spatial axis must also negate its component channel
        # (channel = tensor.dim() - 1 - dim, as channels are in (x,y,z) order and axes are reversed).
        # Enable ``vector_field`` only in configs whose augmented tensors are single-channel (scalars/masks,
        # left untouched) or genuine vector fields: any OTHER multi-channel tensor whose channel count
        # equals the spatial rank (e.g. a 3-contrast volume in 3D) would be wrongly negated by this guard.
        if self.vector_field and tensor.shape[0] == tensor.dim() - 1:
            for dim in dims:
                result[tensor.dim() - 1 - dim] = -result[tensor.dim() - 1 - dim]
        return result

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # A mirror is a bijection on the voxels (ORIENTATION). Negating a component channel is not: it
        # maps values, so a later GLOBAL_STAT could no longer seed from the stored volume, and only
        # the tensor's channel count says whether it fires, which a header-time declaration cannot see.
        if self.vector_field:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="vector_field: true negates the mirrored component channel, so the stored"
                " volume's statistics are not this stage's output's",
            )
        return PatchLocality(LocalityKind.ORIENTATION)

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        # A mirror moves no axis: the remap is the identity permutation, mirrored on the flipped
        # axes. ``flip`` holds channel-first tensor dims, so spatial axis k is dim k + 1.
        dims = self.flip[index][a]
        remap: AxisRemap = [(k, (k + 1) in dims) for k in range(len(target_slices))]
        return remap_region(target_slices, source_spatial_shape, remap)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._flip(tensor, self.flip[index][a])

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._flip(tensor, self.flip[index][a])


class Permute(DataAugmentation):
    def __init__(self, prob_permute: list[float] | None = [0.5, 0.5]) -> None:
        super().__init__()
        self._permute_dims = torch.tensor([[0, 2, 1, 3], [0, 3, 1, 2]])
        self.prob_permute = prob_permute
        self.permute: dict[int, torch.Tensor] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if len(shapes):
            dim = len(shapes[0])
            if dim != 3:
                raise ValueError("The permute augmentation only support 3D images")
            if self.prob_permute:
                if len(self.prob_permute) != 2:
                    raise ValueError("Size of prob_permute must be equal 2")
                self.permute[index] = torch.rand((len(shapes), len(self.prob_permute))) < torch.tensor(
                    self.prob_permute
                )
            else:
                if len(shapes) != 2:
                    raise ValueError("The number of augmentation images must be equal to 2")
                self.permute[index] = torch.eye(2, dtype=torch.bool)
            for i in range(len(shapes)):
                shapes[i] = remap_shape(shapes[i], self._remap(index, i))
        return shapes

    def _source_axes(self, index: int, a: int) -> list[int]:
        """Which source spatial axis each output spatial axis is drawn from, for copy *a*."""
        axes = list(range(3))
        for permute in self._permute_dims[self.permute[index][a]]:
            axes = [axes[dim - 1] for dim in permute[1:]]
        return axes

    # Reordering axes moves every voxel and touches none, so the multiset of values is the input's:
    # a bijection, which is what ORIENTATION promises.
    locality = LocalityKind.ORIENTATION

    def _remap(self, index: int, a: int) -> AxisRemap:
        # Output axis k is source axis ``_source_axes()[k]``, never mirrored.
        return [(axis, False) for axis in self._source_axes(index, a)]

    def _stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        # The same reorder state_init applied to the copy's grid.
        return remap_shape(shape, self._remap(index, a))

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        return remap_region(target_slices, source_spatial_shape, self._remap(index, a))

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        for permute in self._permute_dims[self.permute[index][a]]:
            tensor = tensor.permute(tuple(permute))
        return tensor

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        for permute in reversed(self._permute_dims[self.permute[index][a]]):
            tensor = tensor.permute(tuple(np.argsort(permute)))
        return tensor


#: Voxels one chunk of an Elastix warp evaluates at once: the float64 corner walk holds tens of
#: bytes per voxel it evaluates, so the sampling grid is filled in chunks and only the grid (12
#: bytes per voxel of the region) stands at the peak.
_ELASTIX_CHUNK_VOXELS = 1 << 21


class Elastix(DataAugmentation):
    """A random cubic-B-spline elastic warp, drawn as a control-point lattice per copy.

    The only state of a draw is its lattice (a :class:`DisplacementStage`, O(control points)); the
    displacement at a voxel is evaluated lazily with the order-3 kernel ITK uses
    (``konfai.data.sampling``), so a region computes exactly its part and the copies stream.

    ``grid_spacing`` and ``max_displacement`` are in the case's world units (its header spacing; a
    headerless case counts voxels). Control values are uniform in ``[-max_displacement,
    max_displacement]`` and the kernel is a convex combination of them, so no voxel's displacement
    exceeds ``max_displacement``: what bounds the source box a target region pulls.
    """

    # A warp through a bounded field: each target region pulls its own box, widened by the field's
    # reach, and samples it. The bound is constant, but REGRID (not HALO) keeps the pull exactly
    # the mapped box and prices the sampling grid the warp builds beside its block.
    locality = LocalityKind.REGRID

    def __init__(self, grid_spacing: int = 16, max_displacement: int = 16) -> None:
        super().__init__()
        self.grid_spacing = grid_spacing
        self.max_displacement = max_displacement
        #: Per case index, per selected copy: the drawn lattice and the case grid it warps.
        self.draws: dict[int, list[tuple[DisplacementStage, Grid]]] = {}

    def reset_state(self, index: int | None = None) -> None:
        super().reset_state(index)
        if index is None:
            self.draws.clear()
        else:
            self.draws.pop(index, None)

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.draws[index] = []
        for shape, cache_attribute in zip(shapes, caches_attribute, strict=False):
            dim = len(shape)
            grid, _missing = Grid.from_header(list(shape), cache_attribute, "the case this draw warps")
            # The transform domain covers the volume's physical footprint (voxel edges). The
            # coefficient grid is what sitk.BSplineTransform(order=3) stores for that domain: one
            # node's spacing before its origin, mesh + 3 nodes per axis (verified against
            # GetCoefficientImages; the parity test holds it to TransformToDisplacementFieldFilter).
            physical_xyz = np.array(list(reversed(shape)), dtype=np.float64) * grid.spacing_xyz
            mesh_xyz = np.maximum(1, (physical_xyz / float(self.grid_spacing) + 0.5).astype(np.int64))
            node_spacing_xyz = physical_xyz / mesh_xyz
            domain_origin_xyz = grid.origin_xyz - grid.direction_xyz @ (0.5 * grid.spacing_xyz)
            coefficient_grid = Grid(
                tuple(int(nodes) + 3 for nodes in reversed(mesh_xyz)),
                domain_origin_xyz - grid.direction_xyz @ node_spacing_xyz,
                node_spacing_xyz,
                grid.direction_xyz,
            )
            control = torch.rand((dim, *(int(nodes) + 3 for nodes in reversed(mesh_xyz))), dtype=torch.float64)
            control = (control - 0.5) * (2.0 * self.max_displacement)
            self.draws[index].append((DisplacementStage(coefficient_grid, control.numpy(), order=3), grid))
        return shapes

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        # |displacement| <= max_displacement in world units by convexity: in voxels that is the
        # bound over the axis's spacing, plus one voxel for the far interpolation tap.
        _stage, grid = self.draws[index][a]
        rank = len(source_spatial_shape)
        pull: list[slice] = []
        for k, (part, extent) in enumerate(zip(target_slices, source_spatial_shape, strict=True)):
            reach = int(np.ceil(self.max_displacement / float(grid.spacing_xyz[rank - 1 - k]))) + 1
            start = max(0, part.start - reach)
            stop = min(extent, part.stop + reach)
            pull.append(slice(start, max(stop, start + 1)))
        return pull

    def _sampling_grid(
        self,
        stage: DisplacementStage,
        grid: Grid,
        target: tuple[slice, ...],
        source: tuple[slice, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Where each target voxel samples from, normalised on the source block for ``grid_sample``.

        Per target voxel: its world point, plus the lattice's displacement there, back to a
        continuous index, re-expressed on the block. Filled slab by slab along the first target
        axis: the corner walk's float64 temporaries then stay chunk-sized while the float32 grid is
        the one region-sized tensor held.
        """
        target_shape = tuple(int(part.stop - part.start) for part in target)
        rank = len(target_shape)
        starts = torch.tensor([float(part.start) for part in reversed(source)], dtype=torch.float64, device=device)
        spans = torch.tensor(
            [float(part.stop - part.start - 1) for part in reversed(source)], dtype=torch.float64, device=device
        )
        out = torch.empty((*target_shape, rank), dtype=torch.float32, device=device)
        plane = int(np.prod(target_shape[1:], dtype=np.int64)) if rank > 1 else 1
        rows = max(1, _ELASTIX_CHUNK_VOXELS // max(1, plane))
        for begin in range(0, target_shape[0], rows):
            slab = (slice(target[0].start + begin, min(target[0].stop, target[0].start + begin + rows)), *target[1:])
            axes = [torch.arange(part.start, part.stop, dtype=torch.float64, device=device) for part in slab]
            mesh = torch.meshgrid(*axes, indexing="ij")
            index_xyz = torch.stack(list(reversed(mesh)), dim=-1)
            world = _apply(index_xyz, grid.index_to_world, device)
            world = world + _displacement_at(stage, world, device)
            coordinates = _to_index(world, grid, device)
            local = torch.where(
                spans > 0, (coordinates - starts) * 2.0 / spans.clamp(min=1.0) - 1.0, torch.zeros_like(coordinates)
            )
            out[begin : begin + (slab[0].stop - slab[0].start)] = local.to(torch.float32)
        return out

    def _warp(
        self,
        stage: DisplacementStage,
        grid: Grid,
        tensor: torch.Tensor,
        source: tuple[slice, ...],
        target: tuple[slice, ...],
    ) -> torch.Tensor:
        # Integer tensors are label maps: nearest-neighbour keeps class ids intact. The whole
        # volume is the region that covers everything, so a streamed region and the whole-volume
        # copy run the same arithmetic and agree to grid_sample's own float rounding.
        mode = "nearest" if not tensor.dtype.is_floating_point else "bilinear"
        sampling = self._sampling_grid(stage, grid, target, source, tensor.device).unsqueeze(0)
        return (
            F.grid_sample(
                tensor.type(torch.float32).unsqueeze(0),
                sampling,
                align_corners=True,
                mode=mode,
                padding_mode="border",
            )
            .type(tensor.dtype)
            .squeeze(0)
        )

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        stage, grid = self.draws[index][a]
        whole = tuple(slice(0, int(extent)) for extent in tensor.shape[1:])
        return self._warp(stage, grid, tensor, whole, whole)

    def _stream_region(
        self, name: str, index: int, a: int, tensor: torch.Tensor, context: RegionContext
    ) -> torch.Tensor:
        stage, grid = self.draws[index][a]
        return self._warp(stage, grid, tensor, tuple(context.source), tuple(context.target))

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Elastix augmentation has no inverse; do not use it for invertible TTA.")
