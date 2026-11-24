# SPDX-License-Identifier: Apache-2.0
"""Scout reads through KonfAI's dataset backends: a case given as OME-Zarr stores is inspected like one given as
image files, with the same geometry and the same planes."""

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset
from konfai_scout import InspectionCase, ScoutConfig, inspect


def test_a_case_of_ome_zarr_stores_is_read_through_konfai(fixtures, tmp_path):
    pytest.importorskip("ngff_zarr")
    source, _defects = fixtures["segmentation"]
    source.stage(tmp_path / "stage")
    root = tmp_path / "stores"
    dataset = Dataset(root, "omezarr")
    for role in ("image", "prediction", "truth"):
        volume = source.volume(role)
        array = volume.region((0, 0, 0), tuple(volume.grid.size_zyx))
        attribute = Attribute()
        attribute["Origin"] = np.asarray(volume.grid.origin_xyz)
        attribute["Spacing"] = np.asarray(volume.grid.spacing_xyz)
        attribute["Direction"] = np.asarray(volume.grid.direction_xyz).ravel()
        dataset.write(role.upper(), "zarr_case", np.ascontiguousarray(array), attribute)
    stores = {role: root / "zarr_case" / f"{role.upper()}.ome.zarr" for role in ("image", "prediction", "truth")}
    assert all(path.is_dir() for path in stores.values())
    case = InspectionCase("segmentation", "zarr_case", stores, labels={1: "sphere", 2: "cube"})
    case.stage(tmp_path / "stage2")
    assert case.volume("image").grid.same_as(source.grid)
    plane = case.volume("prediction").plane(0, 12)
    assert np.array_equal(plane, source.volume("prediction").plane(0, 12))
    assert np.allclose(case.grid.index_to_world.apply([3, 4, 5]), source.grid.index_to_world.apply([3, 4, 5]))
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path / "report")
    assert report.error is None and report.observations
    assert (
        report.case["inputs"]["image"]["path"].endswith("IMAGE.ome.zarr") and report.case["inputs"]["image"]["sha256"]
    )
    measured = [m for m in report.measurements if m.kind == "konfai_eval" and m.values]
    assert measured and any(":Dice" in k for k in measured[0].values)  # KonfAI Eval read the OME-Zarr stores
