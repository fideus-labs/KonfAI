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

"""The streamed-oracle property over the EXPANSION axis: one case expanded into copies (see the
family note in ``test_streamed_oracle_decomposition``).

Every copy of an ``Expand`` carries its own draw, and the decomposition must not change it: the
copies of a swept case must equal the copies of the whole-volume case, draw by draw, at every
region count, and take the regime (one shared read pass, or a solo pass each) their draw declares.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from konfai.data.augmentation import CutOUT, DataAugmentation, Elastix, Noise, Rotate, Scale
from konfai.data.augmentation import Flip as FlipDraw
from konfai.data.materialize import CaseMaterializer, Regime, Verdict
from konfai.data.patching import DatasetManager
from konfai.data.transform import Clip, Expand, Mask, Transform, Write
from konfai.utils.dataset import Dataset
from konfai.utils.errors import PatchError
from oracle_support import (
    AUGMENTATION_ATOL,
    CASE_NAME,
    GEOMETRIES,
    MAIN,
    ROUTES,
    Route,
    budget_for,
    build_case,
    manager,
)

pytest.importorskip("SimpleITK")


@dataclass(frozen=True)
class Draw:
    """One draw, the regime its copies must take, and how far a copy of it may round.

    A per-voxel draw is exactly its own block, so its copies ride ONE read pass; a draw that reads
    elsewhere than its target block cannot, and sweeps its own. Which one is not a detail: the
    shared pass is the whole point of the regime, and a pass that fails falls back to solo passes
    that write the same bytes, so only the regime says whether the optimisation still happens.

    A draw that resamples reaches its copy through grid_sample on coordinates normalised by the
    region's own extent rather than the volume's, which is the deviation ``AUGMENTATION_ATOL``
    bounds (ulps of the phantom's step; measured here at 1.5e-4 on a 500-wide range, 3e-7 of it).
    The exact remaps and the per-voxel fields are byte-identical at any region count.
    """

    build: Callable[[], DataAugmentation]
    regime: Regime
    atol: float = 0.0


def _draws() -> dict[str, Draw]:
    """One draw per way a copy is read: a per-voxel field, a box, two exact remaps, two pull maps.
    Built per call, because a draw caches the parameters it drew for a case."""
    return {
        "Noise": Draw(lambda: Noise(1.0), Regime.SHARED),
        "CutOUT": Draw(lambda: CutOUT(1.0, 0.5, 0.0), Regime.SHARED),
        "Flip": Draw(lambda: FlipDraw(f_prob=[1.0, 1.0, 1.0]), Regime.SOLO),
        "QuarterRotate": Draw(lambda: Rotate(is_quarter=True), Regime.SOLO),
        "Rotate": Draw(lambda: Rotate(a_min=10.0, a_max=10.0), Regime.SOLO, AUGMENTATION_ATOL),
        "Scale": Draw(lambda: Scale(), Regime.SOLO, AUGMENTATION_ATOL),
    }


def _expanded(dataset: Dataset, augmentation: DataAugmentation, copies: int, destination: Path) -> DatasetManager:
    return manager(
        dataset,
        [
            Clip(-200.0, 300.0),
            Expand(nb=copies, pattern="{name}_c{a:02d}"),
            augmentation,
            Write(f"{destination}:h5"),
        ],
        group="Intensity",
    )


@pytest.mark.parametrize("name", list(_draws()), ids=list(_draws()))
@pytest.mark.parametrize("copies", [2, 3])
@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_streamed_copy_equals_the_whole_volume_copy(name: str, copies: int, route: Route, tmp_path: Path) -> None:
    """Every copy of an ``Expand`` carries its own draw, and the decomposition must not change it.

    Pointwise is not place-independent: a noise field and a cutout box are functions of the voxel's
    position, so a copy's stages must be told where their block sits exactly as the shared prefix's
    are. Without that, the copies agreed with the whole volume on a case that fitted one region and
    diverged over its whole extent on anything larger.

    Rank 3 only: the draws are declared three-dimensional (``Permute`` refuses anything else), so a
    2-D row would exercise that refusal rather than this property.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    draw = _draws()[name]
    augmentation = draw.build()
    augmentation.load(1.0)

    streamed = _expanded(dataset, augmentation, copies, tmp_path / "streamed")
    budget = budget_for(streamed, route)
    outcomes = CaseMaterializer(streamed).materialize_copies(list(range(1, copies + 1)), fallback_budget_bytes=budget)
    whole = _expanded(dataset, augmentation, copies, tmp_path / "whole")
    for a in range(1, copies + 1):
        CaseMaterializer(whole)._assemble_and_write(a)

    assert set(outcomes.values()) == {(Verdict.STREAM, draw.regime)}
    for a in range(1, copies + 1):
        entry = f"{CASE_NAME}_c{a:02d}"
        got, _ = Dataset(tmp_path / "streamed", "h5").read_data("Intensity", entry)
        want, _ = Dataset(tmp_path / "whole", "h5").read_data("Intensity", entry)
        np.testing.assert_allclose(got, want, rtol=0, atol=draw.atol)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_transform_after_the_marker_reads_its_companion_where_the_block_sits(route: Route, tmp_path: Path) -> None:
    """A copy's tail is not only its draw: a pointwise TRANSFORM there reads a second volume.

    ``Mask`` takes its foreground from a companion aligned with the case, so it needs the block's
    place as much as a noise field does, and it is the half of the fix whose failure is not silent:
    handed a block as a whole volume it raises, the shared pass gives up, and the copies fall back
    to a solo pass each that writes exactly the same bytes. Which is why the REGIME is what says
    whether the shared pass still happens.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    draw = _draws()["Noise"].build()
    draw.load(1.0)
    mask = Mask(path="Labels", value_outside=-7)

    def chain(destination: Path) -> list[Transform]:
        return [
            Expand(nb=2, pattern="{name}_c{a:02d}"),
            draw,
            mask,
            Write(f"{destination}:h5"),
        ]

    mask.set_datasets([dataset])
    streamed = manager(dataset, chain(tmp_path / "streamed"), group="Intensity")
    budget = budget_for(streamed, route)
    outcomes = CaseMaterializer(streamed).materialize_copies([1, 2], fallback_budget_bytes=budget)
    assert set(outcomes.values()) == {(Verdict.STREAM, Regime.SHARED)}

    whole = manager(dataset, chain(tmp_path / "whole"), group="Intensity")
    for a in (1, 2):
        CaseMaterializer(whole)._assemble_and_write(a)
    for a in (1, 2):
        entry = f"{CASE_NAME}_c{a:02d}"
        got, _ = Dataset(tmp_path / "streamed", "h5").read_data("Intensity", entry)
        want, _ = Dataset(tmp_path / "whole", "h5").read_data("Intensity", entry)
        np.testing.assert_array_equal(got, want)
        assert (got == -7).any(), "the mask fell outside the copy: nothing was masked"


def test_a_copy_that_cannot_stream_is_refused_under_a_budget_its_whole_volume_exceeds(tmp_path: Path) -> None:
    """``Elastix`` solves its field over the whole volume, so its copies take the whole-volume path.

    That path is priced, not free: under a budget the assembled case does not fit, the copies must
    be refused with the working set named, and nothing written. Given room, the same copies land."""
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    draw = Elastix()
    draw.load(1.0)
    refused = _expanded(dataset, draw, 2, tmp_path / "refused")
    with pytest.raises(PatchError, match="exceeds the per-rank budget"):
        CaseMaterializer(refused).materialize_copies([1, 2], fallback_budget_bytes=1.0)
    assert not (tmp_path / "refused").exists()

    written = _expanded(dataset, draw, 2, tmp_path / "written")
    outcomes = CaseMaterializer(written).materialize_copies([1, 2])
    assert {verdict for verdict, _regime in outcomes.values()} == {Verdict.WHOLE_VOLUME}
    assert Dataset(tmp_path / "written", "h5").is_dataset_exist("Intensity", f"{CASE_NAME}_c01")


def test_the_copies_of_a_case_are_not_the_same_copy(tmp_path: Path) -> None:
    """The property above compares two routes of ONE draw, so it would hold if every copy were the
    identity. The copies must differ from each other and from the source."""
    geometry = GEOMETRIES[MAIN]
    dataset = build_case(tmp_path / "case", geometry)
    augmentation = _draws()["Noise"].build()
    augmentation.load(1.0)
    CaseMaterializer(_expanded(dataset, augmentation, 2, tmp_path / "out")).materialize_copies([1, 2])

    out = Dataset(tmp_path / "out", "h5")
    first, _ = out.read_data("Intensity", f"{CASE_NAME}_c01")
    second, _ = out.read_data("Intensity", f"{CASE_NAME}_c02")
    source, _ = dataset.read_data("Intensity", CASE_NAME)
    assert not np.array_equal(first, second) and not np.array_equal(first, source)
