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

"""What a cohort walk asks the roots, and what it must never ask them.

A subset that names its cases is answered by asking the roots about those. Enumerating a root and
then discarding what the subset already excluded costs one entry open per case the root HOLDS, and
a root is as wide as its owner made it, not as wide as the run.

Counted, never timed: the count is what a narrow subset and a wide root differ by, and it is the
same count on a filesystem and on an object store.
"""

from pathlib import Path

import numpy as np
import pytest
from konfai.data.data_manager import DataPrediction, Group, PredictionSubset, Subset
from konfai.utils.dataset import Attribute, Dataset

pytest.importorskip("SimpleITK")

CASES = [f"CASE_{index:03d}" for index in range(8)]


def _attributes() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


def _root(tmp_path: Path, file_format: str) -> Dataset:
    """A cohort of eight cases: ``mha`` is one entry per case, ``h5`` one entry for all of them."""
    dataset = Dataset(tmp_path / ("cases" if file_format == "mha" else "store.h5"), file_format)
    for name in CASES:
        dataset.write("CT", name, np.zeros((1, 2, 2, 2), np.float32), _attributes())
    return dataset


class _CountingOpens:
    """How many entries the root was opened for, whatever the backend keeps them in."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        opener = Dataset._file
        monkeypatch.setattr(Dataset, "_file", lambda dataset, *args: self._counted(opener, dataset, *args))

    def _counted(self, opener, dataset, *args):
        self.count += 1
        return opener(dataset, *args)


# ---------------------------------------------------------------- what a subset can name up front


@pytest.mark.parametrize(
    "subset,required",
    [
        (None, None),
        ("CASE_003", {"CASE_003"}),
        (["CASE_003", "CASE_005"], {"CASE_003", "CASE_005"}),
        (["CASE_003", "~CASE_005"], {"CASE_003"}),  # an exclusion only removes what an inclusion brought
        ("~CASE_003", None),  # excluding alone is defined against everything the roots hold
        ("0:4", None),  # so is a slice
        (3, None),  # and so is a position
    ],
)
def test_a_subset_names_its_cases_when_it_can(subset, required) -> None:
    assert Subset(subset).required_names() == required


def test_a_subset_reading_its_names_from_a_file_names_them(tmp_path: Path) -> None:
    names = tmp_path / "fold.txt"
    names.write_text("CASE_001\nCASE_004\n", encoding="utf-8")
    assert Subset(str(names)).required_names() == {"CASE_001", "CASE_004"}


def test_a_subset_selecting_on_geometry_still_gets_the_cohort() -> None:
    """``requires_infos`` marks a subclass that picks from the headers: it is handed every case."""

    class _Biggest(Subset):
        def __call__(self, names, infos):
            return {max(names)}

    assert _Biggest("CASE_003").required_names() is None


# ---------------------------------------------------------------- what the root is then asked


@pytest.mark.parametrize("file_format", ["mha", "h5"])
def test_asking_for_two_cases_answers_the_same_as_listing_all_of_them(tmp_path: Path, file_format: str) -> None:
    dataset = _root(tmp_path, file_format)
    requested = {"CASE_002", "CASE_005"}
    assert dataset.select_names("CT", None) == CASES
    assert dataset.select_names("CT", requested) == sorted(requested)
    assert dataset.select_names("CT", {"CASE_002", "CASE_404"}) == ["CASE_002"]


def test_a_root_of_one_entry_per_case_is_opened_once_per_case_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _root(tmp_path, "mha")

    listing = _CountingOpens(monkeypatch)
    assert dataset.select_names("CT", None) == CASES
    assert listing.count == len(CASES), "enumerating opens every case the root holds"

    dataset._names_cache.clear()
    narrowed = _CountingOpens(monkeypatch)
    assert dataset.select_names("CT", {"CASE_002"}) == ["CASE_002"]
    assert narrowed.count == 1, "one case asked for, one entry opened, whatever the root holds"


def test_a_root_of_one_entry_for_all_cases_keeps_its_single_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrowing must not cost anywhere it does not pay: a store that lists every case in one open
    answers from that open, and probing it case by case would be seven opens more."""
    dataset = _root(tmp_path, "h5")

    counted = _CountingOpens(monkeypatch)
    assert dataset.select_names("CT", {"CASE_002"}) == ["CASE_002"]
    assert counted.count == 1


# ---------------------------------------------------------------- and what the walk then costs


def _walk(tmp_path: Path, file_format: str, subset, cases: list[str], monkeypatch) -> tuple[list[str], int]:
    """One cohort walk, and the number of entries it opened to make it."""
    root = tmp_path / ("cases" if file_format == "mha" else "store.h5")
    if not root.exists():
        dataset = Dataset(root, file_format)
        for name in cases:
            dataset.write("CT", name, np.zeros((1, 2, 2, 2), np.float32), _attributes())
    counted = _CountingOpens(monkeypatch)
    data = DataPrediction(
        dataset_filenames=[f"{root}:{file_format}"],
        groups_src={"CT": Group()},
        augmentations=None,
        patch=None,
        subset=PredictionSubset(subset),
    )
    names, _ = data._select_cases()
    return sorted(names), counted.count


@pytest.mark.parametrize("file_format", ["mha", "h5"])
def test_a_named_subset_costs_what_it_asked_for_not_what_the_root_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: str
) -> None:
    """The walk is what a wide root and a narrow subset differ by: it must not scale with the root.

    Doubling the cohort and asking for the same one case must not move the count, or a root is as
    slow to select from as its owner made it wide.
    """
    small, opens_small = _walk(tmp_path / "small", file_format, "CASE_003", CASES, monkeypatch)
    wide = [f"CASE_{index:03d}" for index in range(64)]
    large, opens_large = _walk(tmp_path / "wide", file_format, "CASE_003", wide, monkeypatch)

    assert small == large == ["CASE_003"]
    assert opens_small == opens_large, "the walk scales with the root, not with the subset"


@pytest.mark.parametrize("file_format", ["mha", "h5"])
def test_a_subset_that_cannot_name_its_cases_still_reads_the_cohort_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: str
) -> None:
    """A slice is defined against the full sorted list, so it gets it, and gets it ONCE: the
    existence probe and the name walk are the same listing, not two."""
    names, opens = _walk(tmp_path, file_format, "0:2", CASES, monkeypatch)
    assert names == ["CASE_000", "CASE_001"]
    assert opens == (len(CASES) if file_format == "mha" else 1)
