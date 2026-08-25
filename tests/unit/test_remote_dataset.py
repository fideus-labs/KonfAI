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

"""A dataset root named by a URI: the cohort walk, and what it must never answer quietly.

A mistyped bucket, an expired credential and a listing-denied prefix must each raise. Answering
"empty" is a run that does nothing and reports success.

The remote store is fsspec's in-memory filesystem: a real URI through the real code path, with no
network and no credentials.
"""

import numpy as np
import pytest
from konfai.utils import uri
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError

pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")
fsspec = pytest.importorskip("fsspec")


@pytest.fixture
def memory_root() -> str:
    """An empty ``memory://`` root, cleared of whatever a previous test left behind."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append("")
    return "memory://cohort"


def _attributes() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


def _publish(memory_root: str, case: str, group: str, volume: np.ndarray) -> None:
    """Write one entry into the memory root as an OME-Zarr store, through ngff-zarr itself."""
    from konfai.utils.ome_zarr import write_ome_zarr

    write_ome_zarr(
        f"{memory_root}/{case}/{group}.ome.zarr",
        volume,
        spacing=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        attributes=dict(_attributes()),
    )


# ---------------------------------------------------------------- telling the two kinds apart


@pytest.mark.parametrize(
    "path,remote",
    [
        ("s3://bucket/key", True),
        ("gs://bucket/key", True),
        ("memory://cohort", True),
        ("./Dataset", False),
        ("/srv/data/Dataset", False),
        (r"C:\Data\Dataset", False),
        ("C://Data/Dataset", False),  # a drive letter is one character: not a protocol
    ],
)
def test_a_uri_is_told_from_a_path(path: str, remote: bool) -> None:
    assert uri.is_uri(path) is remote


def test_normalising_a_uri_keeps_both_slashes() -> None:
    """``Path('s3://b/k').as_posix()`` is ``'s3:/b/k'``: one slash, a path nothing resolves and
    nothing raises on either."""
    assert uri.normalize("s3://bucket/key") == "s3://bucket/key"
    assert uri.normalize("./Dataset") == "Dataset"


def test_a_dataset_keeps_the_uri_it_was_given(memory_root: str) -> None:
    dataset = Dataset(memory_root, "omezarr")
    assert dataset.filename == f"{memory_root}/"
    assert dataset.store_root == f"{memory_root}/"


# ---------------------------------------------------------------- the cohort walk


def test_a_remote_cohort_is_walked_like_a_local_one(memory_root: str) -> None:
    rng = np.random.default_rng(0)
    for case in ("CASE_000", "CASE_001"):
        _publish(memory_root, case, "CT", (rng.random((1, 6, 8, 10)) * 100).astype(np.float32))

    dataset = Dataset(memory_root, "omezarr")

    assert dataset.exists_on_disk()
    assert dataset.get_group() == ["CT"]
    assert dataset.get_names("CT") == ["CASE_000", "CASE_001"]
    assert dataset.is_dataset_exist("CT", "CASE_000")
    assert not dataset.is_dataset_exist("CT", "CASE_404")


def test_a_remote_entry_is_read_region_by_region(memory_root: str) -> None:
    volume = (np.random.default_rng(1).random((1, 6, 8, 10)) * 100).astype(np.float32)
    _publish(memory_root, "CASE_000", "CT", volume)

    dataset = Dataset(memory_root, "omezarr")
    shape, _attribute = dataset.get_infos("CT", "CASE_000")
    assert shape == [1, 6, 8, 10]

    region = (slice(0, 1), slice(2, 5), slice(0, 8), slice(0, 10))
    read, _attributes = dataset.read_data_slice("CT", "CASE_000", region)
    np.testing.assert_allclose(read, volume[region])


def test_an_unreachable_remote_root_raises_instead_of_reporting_no_cases(
    memory_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: an empty answer here is a run that does nothing and says it succeeded."""
    _publish(memory_root, "CASE_000", "CT", np.zeros((1, 4, 4, 4), np.float32))
    dataset = Dataset(memory_root, "omezarr")

    def denied(*_args, **_kwargs):
        raise PermissionError("listing denied")

    monkeypatch.setattr(fsspec.filesystem("memory").__class__, "ls", denied)

    with pytest.raises(DatasetManagerError, match="could not be listed"):
        dataset.get_names("CT")


def test_konfai_declares_no_credentials_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """fsspec merges ``FSSPEC_<PROTO>_<KEY>`` and ``~/.config/fsspec/*.json`` into every filesystem
    it builds, so a public bucket is ``FSSPEC_S3_ANON=true`` and a private one the ``AWS_*``
    variables. KonfAI hands it the protocol and nothing else, or it would override that."""
    seen: dict = {}
    original = fsspec.filesystem

    def spy(protocol, **kwargs):
        seen.update(protocol=protocol, kwargs=kwargs)
        return original(protocol, **kwargs)

    monkeypatch.setattr(fsspec, "filesystem", spy)
    uri.exists("memory://cohort/CASE_000")
    assert seen == {"protocol": "memory", "kwargs": {}}


# ---------------------------------------------------------------- what a remote root cannot do


def test_an_unregistered_scheme_is_not_reported_as_a_configuration_to_check() -> None:
    """fsspec answers "Protocol not known" for a scheme nothing registers, and the variables the
    configuration hint names do not exist for it: pointing a user at them sends them looking for a
    setting they cannot write."""
    pytest.importorskip("fsspec")
    with pytest.raises(DatasetManagerError) as raised:
        uri.filesystem("nosuchproto://bucket/key")
    message = " ".join(str(part) for part in raised.value.args)
    assert "No filesystem is registered for 'nosuchproto://'" in message
    assert "FSSPEC_" not in message


def test_which_refusal_a_scheme_gets_is_read_off_the_registry_not_off_the_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same exception types carry both refusals, and only the registry tells them apart.

    Classified on the text of the exception, a registered filesystem refusing a bucket it does not
    know was reported as a scheme nothing implements (install something), and any wording fsspec
    changes turns an unregistered scheme into configuration variables the user cannot write.
    """
    fsspec = pytest.importorskip("fsspec")

    def refuse(protocol: str, **kwargs: object) -> object:
        raise ValueError(f"bucket for {protocol} not known to this endpoint")

    monkeypatch.setattr(fsspec, "filesystem", refuse)
    with pytest.raises(DatasetManagerError) as configured:
        uri.filesystem("memory://bucket/key")
    with pytest.raises(DatasetManagerError) as unregistered:
        uri.filesystem("nosuchproto://bucket/key")

    assert "refused its fsspec configuration" in " ".join(str(part) for part in configured.value.args)
    assert "No filesystem is registered" in " ".join(str(part) for part in unregistered.value.args)


def test_a_remote_root_in_a_format_that_reads_files_is_refused(memory_root: str) -> None:
    """Only the store backend reads a URI; the others open a path, and a path is what a URI is not."""
    with pytest.raises(DatasetManagerError, match="only ':omezarr' can read"):
        with Dataset.File(f"{memory_root}/CASE_000", True, "mha"):
            pass


def test_writing_to_a_remote_root_is_refused(memory_root: str) -> None:
    dataset = Dataset(memory_root, "omezarr")
    with pytest.raises(DatasetManagerError, match="writes only to local ones"):
        dataset.write("CT", "CASE_000", np.zeros((1, 4, 4, 4), np.float32), _attributes())


def test_an_unknown_scheme_names_itself(memory_root: str) -> None:
    del memory_root
    with pytest.raises(DatasetManagerError, match="quicksand"):
        uri.exists("quicksand://bucket/key")


def test_a_probe_the_filesystem_cannot_answer_raises(memory_root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """fsspec's ``isdir`` and ``exists`` answer False on any failure, so a denied bucket would read
    as an absent one: every probe here goes through ``info`` and raises on anything but absence."""

    def denied(*_args, **_kwargs):
        raise PermissionError("AccessDenied")

    monkeypatch.setattr(fsspec.filesystem("memory").__class__, "info", denied)

    for probe in (uri.exists, uri.is_dir, uri.list_names):
        with pytest.raises(DatasetManagerError, match="could not be"):
            probe(f"{memory_root}/CASE_000")


def test_an_absent_remote_key_is_absent_and_not_an_error(memory_root: str) -> None:
    assert not uri.exists(f"{memory_root}/CASE_404")
    assert not uri.is_dir(f"{memory_root}/CASE_404")
    assert uri.list_names(f"{memory_root}/CASE_404") == []


def test_a_remote_cohort_is_told_to_be_a_store_by_its_entries_names(memory_root: str) -> None:
    """Detection never probes a remote entry as a local path: a bare name would ask the working
    directory, and the answer would change with it."""
    _publish(memory_root, "CASE_000", "CT", np.zeros((1, 4, 4, 4), np.float32))

    assert Dataset._detect_directory_store_format(f"{memory_root}/") == "omezarr"
    assert Dataset(memory_root, "mha").file_format == "omezarr"


def test_rebasing_a_remote_root_is_refused_before_it_stops_looking_remote() -> None:
    """``prefix / 's3://bucket'`` is ``prefix/s3:/bucket``: a local directory named after a scheme."""
    from pathlib import Path

    dataset = Dataset("memory://bucket/out", "mha")
    with pytest.raises(DatasetManagerError, match="writes only to local ones"):
        dataset.rebase(Path("/runs/Predictions/run"))
