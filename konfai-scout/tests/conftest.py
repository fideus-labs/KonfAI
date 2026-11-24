# SPDX-License-Identifier: Apache-2.0
"""Synthetic cases with planted defects, built once per session. No patient data, no API."""

from __future__ import annotations

import pytest
from konfai_scout.synthetic import registration_case, segmentation_case, synthesis_case


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple]:
    root = tmp_path_factory.mktemp("cases")
    return {
        "segmentation": segmentation_case(root / "seg"),
        "segmentation_oblique": segmentation_case(root / "seg_oblique", oblique=True),
        "segmentation_no_truth": segmentation_case(root / "seg_no_truth", with_truth=False),
        "segmentation_clean": segmentation_case(root / "seg_clean", clean=True),  # no lead: the loop's own tests
        "registration": registration_case(root / "reg", oblique=True),
        "registration_no_ref": registration_case(root / "reg_no_ref", with_references=False, fold=False),
        "registration_clean": registration_case(root / "reg_clean", with_references=False, clean=True),
        "synthesis": synthesis_case(root / "syn"),
        "synthesis_no_target": synthesis_case(root / "syn_no_target", with_target=False),
    }
