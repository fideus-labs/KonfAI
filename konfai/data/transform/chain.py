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


"""Chain structure: N-cases-to-one reduction, copy expansion, the expand marker."""

import inspect

import torch

from konfai.data.reduction import Reduction
from konfai.data.transform.base import LocalityKind, PatchLocality, Transform
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute
from konfai.utils.errors import ReductionError, TransformError
from konfai.utils.utils import get_module

#: Keys the ``Reduce`` stage reads from its own mapping. An operator sharing one of these names is
#: refused.
_REDUCE_OWN_KEYS = frozenset({"operator", "output", "grid", "grid_tolerance", "provenance"})


def resolve_operator(reduce: "Reduce") -> Reduction:
    """The operator the stage names, as an instance, refusing one that cannot fold a region.

    Its constructor arguments are bound from the same mapping the ``Reduce`` itself was read from
    (``Reduce: {operator: mypkg:TrimmedMean, output: template, trim: 0.2}`` gives it ``trim``). A
    chain assembled in Python builds the operator from its own defaults.
    """
    module, name = get_module(reduce.operator_classpath, "konfai.data.reduction")
    factory = getattr(module, name)
    shadowed = sorted(set(inspect.signature(factory.__init__).parameters) & _REDUCE_OWN_KEYS)
    if shadowed:
        raise ReductionError(
            f"'{reduce.operator_classpath}' has parameter(s) {shadowed}, which the Reduce stage"
            " reads for itself from the same mapping.",
            f"Rename them: {sorted(_REDUCE_OWN_KEYS)} belong to Reduce, everything else in the"
            " mapping is the operator's.",
        )
    try:
        operator = apply_config(reduce.konfai_args)(factory)()
    except TypeError as error:
        raise ReductionError(
            f"'{reduce.operator_classpath}' could not be built: {error}.",
            "Give its parameters under the Reduce stage, next to 'operator' and 'output', or give them defaults.",
        ) from error
    if not isinstance(operator, Reduction):
        raise ReductionError(
            f"'{reduce.operator_classpath}' is not a Reduction.",
            "Subclass konfai.data.reduction.Reduction, or use Mean / Median / Concat.",
        )
    if not operator.voxel_local:
        raise ReductionError(
            f"'{reduce.operator_classpath}' does not declare itself voxel-local, so it cannot be"
            " folded one region at a time.",
            "Set voxel_local = True when every output voxel depends only on the same voxel of each"
            " case; an operator reading across space cannot stream.",
        )
    return operator


class Reduce(Transform):
    """Fold every case of a group into one volume, at fixed voxel.

    Everything before it runs once per case, this folds the cases together, everything after it runs
    once on the result. A chain carrying one is driven by the reduction engine.

    ``operator`` is a classpath resolved against :mod:`konfai.data.reduction` (``Mean``, ``Median``,
    ``Concat``, or your own :class:`~konfai.data.reduction.Reduction`). ``output`` is the required
    entry name the result is written under. ``grid`` decides how much agreement between members is
    demanded before a byte is read: ``strict`` compares extents and geometry
    (Spacing/Origin/Direction) within ``grid_tolerance``; ``shape_only`` compares extents alone;
    ``reference:<case>`` adopts that member's geometry and still demands equal extents.
    """

    def __init__(
        self,
        operator: str = "Median",
        output: str = "",
        grid: str = "strict",
        grid_tolerance: float = 1e-6,
        provenance: bool = True,
    ) -> None:
        super().__init__()
        if not output or not str(output).strip():
            raise TransformError(
                "'Reduce' needs an 'output': the name its single result is written under.",
                "Declare it, e.g. Reduce: {operator: Median, output: template}. A reduction has no"
                " case name to inherit: borrowing a member's would tie the deliverable to"
                " iteration order.",
            )
        policy = str(grid)
        reference = policy.split(":", 1)[1].strip() if policy.startswith("reference:") else ""
        if policy not in ("strict", "shape_only") and not reference:
            raise TransformError(
                f"'Reduce' has an unknown grid policy '{grid}'.",
                "Use 'strict' (extents + geometry), 'shape_only' (extents only) or"
                " 'reference:<case>' (adopt that member's geometry): 'reference:' alone names no"
                " case.",
            )
        self.operator_classpath = str(operator)
        # Where this stage was configured from, so its operator binds its own parameters from the
        # same mapping. None for a chain built in Python.
        self.konfai_args: str | None = None
        self._operator: Reduction | None = None
        self.output = str(output).strip()
        self.grid = f"reference:{reference}" if reference else policy
        self.grid_tolerance = float(grid_tolerance)
        self.provenance = bool(provenance)

    def prepare(self, konfai_args: str) -> None:
        # Bound here, not lazily: a strict read of the config counts a key only if something read it.
        self.konfai_args = konfai_args
        self._operator = resolve_operator(self)

    @property
    def operator(self) -> Reduction:
        """The bound operator; a stage built in Python, never prepared, binds it from its defaults."""
        if self._operator is None:
            self._operator = resolve_operator(self)
        return self._operator

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A cardinality marker, not a per-case stage: the reduction engine splits it out of the
        # chain before any manager is built.
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        raise TransformError(
            f"'Reduce' (output '{self.output}') was applied to one case, which reduces nothing.",
            "A chain containing Reduce is run by the reduction engine of the TRANSFORM workflow; it"
            " has no meaning as an ordinary per-case transform.",
        )


class Expand(Transform):
    """Turn one case into ``nb`` copies, at a declared point of the chain: ``Reduce``'s mirror.

    Everything before it runs once per case (a ``Save`` there is a cache every copy shares);
    everything after it runs once per copy, and a ``Save``/``Write`` there writes one entry per copy.

    It multiplies, and nothing else: the draws are ordinary stages of the chain, so transforms and
    augmentations interleave freely after the marker::

        transforms:
          Clip:   {min_value: 0.0, max_value: 400.0}   # once per case
          Expand: {nb: 8, pattern: "{name}_r{a:02d}"}
          Rotate: {a_min: -15, a_max: 15}              # a draw, per copy
          Write:  {dataset: ./Augmented:omezarr}

    ``pattern`` names each copy's entry: ``str.format`` over ``{name}`` (the case) and ``{a}`` (the
    copy ordinal, 1-based). Both tokens are required.

    Every draw after this marker is parameterised from ``(seed, case, which draw this is)``, not from
    a shared RNG. Left unset, ``seed`` is the run's ``manual_seed``, so an image chain and its mask
    chain produce matching copies. Set it to ask two chains for different copies of the same cases.
    """

    def __init__(self, nb: int = 2, pattern: str = "{name}_{a:02d}", seed: int | None = None) -> None:
        super().__init__()
        self.seed = None if seed is None else int(seed)
        if int(nb) < 1:
            raise TransformError(
                f"'Expand' asks for {nb} copies.",
                "A cardinality is at least one: nb: 8 writes eight entries per case.",
            )
        self.nb = int(nb)
        pattern = str(pattern)
        try:
            first, second = pattern.format(name="case", a=1), pattern.format(name="case", a=2)
        except (KeyError, IndexError, ValueError) as error:
            raise TransformError(
                f"'Expand' cannot format its pattern '{pattern}': {error}.",
                "The pattern is a str.format template over {name} and {a}, e.g. pattern: '{name}_r{a:02d}'.",
            ) from error
        if "{name" not in pattern or first == second:
            raise TransformError(
                f"'Expand' has a pattern ('{pattern}') that does not vary over "
                + ("{name}" if "{name" not in pattern else "{a}")
                + ", so its entries would collide.",
                "Use both tokens, e.g. pattern: '{name}_r{a:02d}': {name} keeps cases apart,"
                " {a} keeps a case's copies apart.",
            )
        self.pattern = pattern

    @property
    def draw_seed(self) -> int:
        """The seed the copies are actually drawn from: this marker's own, or the run's."""
        return 0 if self.seed is None else self.seed

    def entry(self, name: str, a: int) -> str:
        """The entry name copy ``a`` of case ``name`` writes under."""
        return self.pattern.format(name=name, a=a)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A cardinality marker, not a per-case stage: the dispatcher splices the copy's own draw at
        # this position and never runs the marker itself.
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        raise TransformError(
            "'Expand' was applied to one tensor, which expands nothing.",
            "A chain containing Expand is run by the TRANSFORM workflow, which replaces the marker"
            " with each copy's draw; it has no meaning as an ordinary per-case transform.",
        )


def split_expand(transforms: list[Transform]) -> tuple[list[Transform], "Expand | None", list[Transform]]:
    """A chain around its ``Expand``: what runs once per case, the marker, what runs per copy."""
    for index, transform in enumerate(transforms):
        if isinstance(transform, Expand):
            return list(transforms[:index]), transform, list(transforms[index + 1 :])
    return list(transforms), None, []
