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

"""A store's length unit, on the way in and on the way out.

KonfAI's Spacing and Origin are ITK's: plain numbers that everything downstream reads as
millimetres. NGFF states a unit instead and makes it optional, so a micrometre store read at its
numbers is a volume a thousand times too large -- which is how one lands in a viewer as a 12-metre
brain. These tests pin the conversion in both directions, and that a store KonfAI writes says which
unit its numbers are in.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")

import ngff_zarr as nz
from konfai.utils.dataset.attribute import ome_zarr_attributes
from konfai.utils.ome_zarr import (
    DEFAULT_LENGTH_UNIT,
    clear_ome_zarr_cache,
    get_ome_zarr_info,
    millimetres_per_unit,
    write_ome_zarr,
)

MICRON_SPACING = {"z": 40.0, "y": 30.08, "x": 30.08}  # an ExaSPIM level 0, as published
MICRON_ORIGIN = {"z": -23819.0, "y": 11000.0, "x": -40600.0}


def _micrometre_store(path: Path) -> Path:
    """A store in micrometres, written the way the published ExaSPIM ones are."""
    image = nz.to_ngff_image(
        np.zeros((4, 5, 6), np.uint16),
        dims=["z", "y", "x"],
        scale=MICRON_SPACING,
        translation=MICRON_ORIGIN,
        axes_units=dict.fromkeys("zyx", "micrometer"),
    )
    nz.to_ngff_zarr(str(path), nz.to_multiscales(image, scale_factors=[], cache=False), version="0.4")
    clear_ome_zarr_cache(path)
    return path


def _axis_units(store: Path) -> dict[str, str | None]:
    return {axis.name: axis.unit for axis in nz.from_ngff_zarr(str(store)).metadata.axes}


def test_a_micrometre_store_is_read_in_millimetres(tmp_path: Path) -> None:
    # Spacing and Origin go to SimpleITK and into NIfTI headers, neither of which carries a unit.
    info = get_ome_zarr_info(_micrometre_store(tmp_path / "micron.ome.zarr"))
    attributes = ome_zarr_attributes(info)

    assert info["units"] == {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
    assert np.allclose(attributes.get_np_array("Spacing"), [0.03008, 0.03008, 0.04])  # (x, y, z)
    assert np.allclose(attributes.get_np_array("Origin"), [-40.6, 11.0, -23.819])


def test_a_store_without_a_unit_is_taken_at_its_numbers(tmp_path: Path) -> None:
    # NGFF makes the unit optional, and converting an unstated one would be guessing.
    store = tmp_path / "bare.ome.zarr"
    image = nz.to_ngff_image(np.zeros((4, 5, 6), np.uint16), dims=["z", "y", "x"], scale=MICRON_SPACING)
    nz.to_ngff_zarr(str(store), nz.to_multiscales(image, scale_factors=[], cache=False), version="0.4")
    clear_ome_zarr_cache(store)

    attributes = ome_zarr_attributes(get_ome_zarr_info(store))

    assert np.allclose(attributes.get_np_array("Spacing"), [30.08, 30.08, 40.0])
    assert "OMEUnits" not in attributes


def test_a_micrometre_store_read_and_written_back_keeps_its_numbers(tmp_path: Path) -> None:
    # The round trip is what a pipeline does to every volume it touches; it must not move the grid.
    source = _micrometre_store(tmp_path / "micron.ome.zarr")
    attributes = ome_zarr_attributes(get_ome_zarr_info(source))
    destination = tmp_path / "again.ome.zarr"

    write_ome_zarr(
        destination,
        np.zeros((1, 4, 5, 6), np.uint16),
        spacing=attributes.get_np_array("Spacing"),
        origin=attributes.get_np_array("Origin"),
        attributes=dict(attributes),
    )

    written = get_ome_zarr_info(destination)
    assert _axis_units(destination) == {"c": None, "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    assert np.allclose(written["geometry"]["z"]["scale"], MICRON_SPACING["z"])
    assert np.allclose(written["geometry"]["x"]["translation"], MICRON_ORIGIN["x"])
    # And read back through the same door, it is millimetres again.
    assert np.allclose(ome_zarr_attributes(written).get_np_array("Spacing"), [0.03008, 0.03008, 0.04])


def test_a_store_written_from_millimetres_says_so(tmp_path: Path) -> None:
    # Without a unit the numbers mean whatever the reader assumes: Slicer reads them as millimetres,
    # Neuroglancer as metres. KonfAI holds millimetres, so the store states it.
    store = tmp_path / "mm.ome.zarr"
    write_ome_zarr(store, np.zeros((1, 4, 5, 6), np.uint16), spacing=(0.5, 0.5, 1.0), origin=(1.0, 2.0, 3.0))

    assert _axis_units(store) == {
        "c": None,
        "z": DEFAULT_LENGTH_UNIT,
        "y": DEFAULT_LENGTH_UNIT,
        "x": DEFAULT_LENGTH_UNIT,
    }
    assert np.allclose(get_ome_zarr_info(store)["geometry"]["z"]["scale"], 1.0)


@pytest.mark.parametrize(
    ("unit", "millimetres"),
    [("micrometer", 1e-3), ("MICROMETER", 1e-3), ("millimeter", 1.0), ("meter", 1e3), ("inch", 25.4)],
)
def test_known_units_convert(unit: str, millimetres: float) -> None:
    assert millimetres_per_unit(unit) == pytest.approx(millimetres)


@pytest.mark.parametrize("unit", [None, "", "pixel", "second"])
def test_an_unconvertible_unit_does_not_convert(unit: str | None) -> None:
    # Not a length, or not stated: the numbers stand as they are rather than being scaled by a guess.
    assert millimetres_per_unit(unit) is None
