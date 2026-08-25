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

"""The TRANSFORM workflow end to end, through its real YAML: parse-time refusals, the plan as the
run's own verdict, the two-branch engine, per-case resume, forced rewrite, and the memory bound."""

import pickle
from pathlib import Path

import numpy as np
import pytest
from konfai.data.materialize import Regime, Verdict
from konfai.data.reduction import Mean
from konfai.data.transform import Reduce
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError, TransformerError

pytest.importorskip("SimpleITK")

_ENV_KEYS = (
    "KONFAI_config_file",
    "KONFAI_ROOT",
    "KONFAI_STATE",
    "KONFAI_CONFIG_MODE",
    "KONFAI_TRANSFORMS_DIRECTORY",
    "KONFAI_OVERWRITE",
)


@pytest.fixture(autouse=True)
def _workflow_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # build_transform writes the workflow environment directly into os.environ; pin every key it
    # touches so the test's values are restored whatever the test does.
    for key in _ENV_KEYS:
        monkeypatch.setenv(key, "sentinel")
        monkeypatch.delenv(key)


def _image_attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attributes


def _write_source(tmp_path: Path, cases: int = 2) -> Path:
    rng = np.random.default_rng(0)
    dataset = Dataset(tmp_path / "source", "mha")
    for index in range(cases):
        volume = (rng.random((1, 12, 10, 8)) * 100).astype(np.float32)
        dataset.write("CT", f"CASE_{index:03d}", volume, _image_attributes())
    return tmp_path / "source"


def _write_config(tmp_path: Path, transforms_yaml: str, header: str = "  on_fallback: warn\n") -> Path:
    source = tmp_path / "source"
    config_path = tmp_path / "Transform.yml"
    config_path.write_text(
        "Transformer:\n"
        "  name: TEST\n"
        f"{header}"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {source}:mha\n"
        "    memory_budget: auto\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          CT_out:\n"
        "            transforms:\n"
        f"{transforms_yaml}"
    )
    return config_path


def _build(tmp_path: Path):
    from konfai.transformer import build_transform

    return build_transform(transform_file=tmp_path / "Transform.yml", transforms_dir=tmp_path / "Transforms")


_STREAMABLE = """\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Write:
                dataset: {out}:h5
"""

_WRITE = """\
              Write:
                dataset: {out}:h5
"""

_UNSTREAMABLE = """\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Standardize:
                inverse: false
              Write:
                dataset: {out}:h5
"""


def test_streamable_chain_plans_streams_and_writes(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    plan = workflow.compute_plan(1, overwrite=False)
    assert [entry.verdict for entry in plan.entries] == ["STREAM", "STREAM"]
    # The probe opened then removed its entry: no probe debris in the output.
    out = Dataset(tmp_path / "out", "h5")
    assert not out.is_dataset_exist("CT_out", "__konfai_plan_probe__")

    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    assert out.is_dataset_exist("CT_out", "CASE_000")
    assert out.is_dataset_exist("CT_out", "CASE_001")
    assert plan.report().count("STREAM")


_REGION_CHAIN = """\
              Resample:
                spacing: [1.0, 1.0, 1.0]
              Write:
                dataset: {out}:h5
"""

_DIRECTORY_CHAIN = """\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Write:
                dataset: {out}:omezarr
"""


def test_plan_leaves_no_probe_case_in_a_directory_store(tmp_path: Path) -> None:
    """A dry run must leave the output directory as it found it.

    Aborting the probe stream drops the entry, but a directory dataset gives every case a directory
    of its own, and that one outlives the stream: left behind, ``__konfai_plan_probe__/`` sits in
    the output root shaped exactly like a case, and ``get_names`` lists it as one.
    """
    _write_source(tmp_path)
    _write_config(tmp_path, _DIRECTORY_CHAIN.format(out=tmp_path / "out"))

    _build(tmp_path).compute_plan(1, overwrite=False)

    assert not (tmp_path / "out" / "__konfai_plan_probe__").exists()
    assert not (tmp_path / "out").exists(), "the probe created the root; a dry run takes it back"


def test_plan_leaves_no_h5_file_behind(tmp_path: Path) -> None:
    """A single-file store is created by the very open the probe makes; --plan must not leave a
    6 KB empty ``out.h5`` where the user asked whether the run would work."""
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    _build(tmp_path).compute_plan(1, overwrite=False)
    assert not (tmp_path / "out.h5").exists()

    # An existing store is left exactly as it was: no entry added, no file removed.
    Dataset(tmp_path / "out", "h5").write("OTHER", "X", np.zeros((1, 2, 2, 2), dtype=np.float32), _image_attributes())
    _build(tmp_path).compute_plan(1, overwrite=False)
    assert Dataset(tmp_path / "out", "h5").get_names("OTHER") == ["X"]
    assert not Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "__konfai_plan_probe__")


def test_plan_leaves_no_single_file_store_behind(tmp_path: Path) -> None:
    """The probe on an h5 destination created the file (a 6 KB store with an empty group): a dry
    run that leaves an output where there was none is not a dry run."""
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))

    _build(tmp_path).compute_plan(1, overwrite=False)

    assert not (tmp_path / "out.h5").exists()


_MHA_CHAIN = """\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Write:
                dataset: {out}:mha
"""


def test_a_bounded_source_streams_to_every_destination_format(tmp_path: Path) -> None:
    """The head segment past the last Save has no reader in a chain ending in a Write, and pricing
    it asked the Write's destination -- not there yet, so 'unbounded' -- and every mha/nii output
    routed to LOAD 'because streaming would re-read the source', which was false."""
    _write_source(tmp_path)
    _write_config(tmp_path, _MHA_CHAIN.format(out=tmp_path / "out"))

    plan = _build(tmp_path).compute_plan(1, overwrite=False)

    assert [entry.verdict for entry in plan.entries] == ["STREAM", "STREAM"]


def test_a_planned_workflow_survives_the_spawn_that_runs_it(tmp_path: Path) -> None:
    """``setup`` runs on the launcher and ``mp.spawn`` then pickles the workflow whole.

    So everything planning leaves behind (the memoized stream sources, the region stages' pull
    maps) has to cross a process boundary, and anything unpicklable in there kills every chain the
    plan just called STREAM before its first byte. The round-trip below is that boundary; the other
    tests call ``run_process`` in-process and cannot see it.
    """
    _write_source(tmp_path)
    _write_config(tmp_path, _REGION_CHAIN.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    plan = workflow.compute_plan(1, overwrite=False)
    assert [entry.verdict for entry in plan.entries] == ["STREAM", "STREAM"]
    workflow.setup(1)

    restored = pickle.loads(pickle.dumps(workflow))  # nosec B301 - our own object, round-tripped
    restored.run_process(1, 0, 0, [])
    assert Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_unstreamable_chain_says_why_and_still_writes(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(tmp_path, _UNSTREAMABLE.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    plan = workflow.compute_plan(1, overwrite=False)
    assert all(entry.verdict == "WHOLE-VOLUME" for entry in plan.entries)
    assert all(entry.reason and "Standardize" in entry.reason for entry in plan.entries)

    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    assert Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_on_fallback_error_refuses_before_any_byte(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(tmp_path, _UNSTREAMABLE.format(out=tmp_path / "out"), header="  on_fallback: error\n")
    workflow = _build(tmp_path)

    with pytest.raises(TransformerError, match="whole-volume"):
        workflow.setup(1)
    assert not Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_budget_is_a_hard_constraint_for_fallback_cases(tmp_path: Path) -> None:
    """A case that cannot stream and does not fit the declared budget refuses globally, before any
    byte, never 40 cases written then a crash at the 41st."""
    _write_source(tmp_path)
    config_path = _write_config(tmp_path, _UNSTREAMABLE.format(out=tmp_path / "out"))
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", "memory_budget: 1b"))
    workflow = _build(tmp_path)

    with pytest.raises(TransformerError, match="budget"):
        workflow.setup(1)
    assert not Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


_PADS_THEN_FALLS_BACK = """\
              Padding:
                padding: [0, 0, 0, 0, 12, 12]
              Standardize:
                inverse: false
              Write:
                dataset: {out}:h5
"""


def test_the_budget_measures_the_largest_intermediate_not_the_stored_case(tmp_path: Path) -> None:
    """A whole-volume fallback holds its BIGGEST tensor, which the chain decides, not the store.

    Padding to twice the depth doubles what has to be resident, and Standardize is what sends the
    case down the fallback path. Sized on the stored extent the case looks half as heavy as it is,
    passes the budget it does not fit, and is OOM-killed with nothing written.
    """
    _write_source(tmp_path)
    config_path = _write_config(tmp_path, _PADS_THEN_FALLS_BACK.format(out=tmp_path / "out"))
    stored = 1 * 12 * 10 * 8 * 4 * 2  # [1, 12, 10, 8] float32, times the in-flight copy
    # A budget the stored case fits and the padded one does not.
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", f"memory_budget: {stored + 1}b"))

    with pytest.raises(TransformerError, match="budget"):
        _build(tmp_path).setup(1)


_RESAMPLES_THEN_FALLS_BACK = """\
              Resample:
                spacing: [0.5, 1.5, 2.0]
              Standardize:
                inverse: false
              Write:
                dataset: {out}:h5
"""


def test_the_budget_counts_a_stage_own_transients(tmp_path: Path) -> None:
    """A whole-volume resample through ITK holds its input image, its output image and their tensor
    copies: 4.5x the case, measured, where an elementwise stage holds 2x. Sized at 2x it passes a
    budget it does not fit. The stage says what it holds (``working_multiple``), the plan sums it."""
    _write_source(tmp_path)
    config_path = _write_config(tmp_path, _RESAMPLES_THEN_FALLS_BACK.format(out=tmp_path / "out"))
    case = 1 * 12 * 10 * 8 * 4  # the resample keeps the grid: same extent in and out
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", f"memory_budget: {case * 3}b"))

    with pytest.raises(TransformerError, match="budget"):
        _build(tmp_path).setup(1)


def test_chain_without_write_is_refused_at_parse(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(tmp_path, "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n")
    with pytest.raises(TransformerError, match="does not end with a 'Write'"):
        _build(tmp_path)


def test_transform_after_the_terminal_write_is_refused(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n"
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n",
    )
    with pytest.raises(TransformerError, match="follows the last Write"):
        _build(tmp_path)


def test_writing_into_the_source_is_refused(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "source"))
    with pytest.raises(TransformerError, match="source"):
        _build(tmp_path)


def test_two_chains_writing_the_same_target_are_refused(tmp_path: Path) -> None:
    _write_source(tmp_path)
    source = tmp_path / "source"
    (tmp_path / "Transform.yml").write_text(
        "Transformer:\n"
        "  name: TEST\n"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {source}:mha\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          A:\n"
        "            transforms:\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n"
        "                group: same\n"
        "          B:\n"
        "            transforms:\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n"
        "                group: same\n"
    )
    with pytest.raises(TransformerError, match="both write"):
        _build(tmp_path)


def test_two_chains_sharing_an_intermediate_save_are_refused(tmp_path: Path) -> None:
    """The terminal Write is not the only boundary two chains can collide on.

    A Save is keyed by (dataset, group, case) and nothing else, so the second chain finds the first
    one's cache already written, adopts it as its own source, and skips its own prefix: producing a
    deliverable computed from another chain's transforms, with no warning and no failed case.
    """
    _write_source(tmp_path)
    source = tmp_path / "source"
    (tmp_path / "Transform.yml").write_text(
        "Transformer:\n"
        "  name: TEST\n"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {source}:mha\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          A:\n"
        "            transforms:\n"
        "              Clip:\n"
        "                min_value: 0.0\n"
        "                max_value: 50.0\n"
        "              Save:\n"
        f"                dataset: {tmp_path / 'work'}:h5\n"
        "                group: shared\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_a'}:h5\n"
        "          B:\n"
        "            transforms:\n"
        "              Clip:\n"
        "                min_value: 0.0\n"
        "                max_value: 10.0\n"
        "              Save:\n"
        f"                dataset: {tmp_path / 'work'}:h5\n"
        "                group: shared\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_b'}:h5\n"
    )
    with pytest.raises(TransformerError, match="both write"):
        _build(tmp_path)


def test_one_chain_may_publish_its_own_save_through_its_write(tmp_path: Path) -> None:
    """The refusal is about two chains, not two stages: a chain's Write over its own Save is legal."""
    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Save:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n"
        "                group: same\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n"
        "                group: same\n",
    )

    assert _build(tmp_path) is not None


_CAST_CHAIN = """\
              Clip:
                min_value: 0.0
                max_value: 50.0
              TensorCast:
                dtype: uint8
              Write:
                dataset: {out}:h5
"""


def test_the_plan_reports_the_dtype_it_probed_the_destinations_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is the probe's own answer, not a constant beside it.

    The write probe must open with the chain's last declared cast: a report that says float32
    whatever the chain casts to describes a run nobody asked for, and a destination that refuses
    uint8 would be green-lit by a probe that still opened float32.
    """
    from konfai.transformer import Transformer

    _write_source(tmp_path)
    _write_config(tmp_path, _CAST_CHAIN.format(out=tmp_path / "out"))

    probed: list[np.dtype] = []
    original = Transformer._probe_destination

    def spy(destination, group, shape, dtype, attributes):
        probed.append(np.dtype(dtype))
        return original(destination, group, shape, dtype, attributes)

    monkeypatch.setattr(Transformer, "_probe_destination", staticmethod(spy))
    plan = _build(tmp_path).compute_plan(1, overwrite=False)

    assert "assumed uint8 / source channels" in plan.report()
    assert probed and all(dtype == np.dtype("uint8") for dtype in probed)


_RESAMPLED_THEN_REFERENCED = """\
              Resample:
                spacing: [2.0, 2.0, 2.0]
                align: origin
              konfai.data.transform:Resample:
                reference: CASE_000
                reference_group: CT
              Write:
                dataset: {out}:h5
"""


def test_a_stage_is_asked_about_its_own_input_not_the_case_as_stored(tmp_path: Path) -> None:
    """The note a stage declares describes the grid it MEETS, which the stages before it decide.

    The resample takes 12x10x8 down to 12x7x2 and keeps voxel zero where it is, so the case's far
    edge falls short of the reference's and part of the output will be fill. Asked about the case as
    STORED it covers all of it and says nothing, and the plan would stay silent about that fill.

    (With ``align: extent``, the default, the box is preserved and the answer is honestly 100%.)
    """
    _write_source(tmp_path)
    _write_config(tmp_path, _RESAMPLED_THEN_REFERENCED.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)
    manager = workflow.dataset.managers["CT_out"][0]
    stored = [int(extent) for extent in manager.base_shape[1:]]
    reference = manager.transforms[1]

    notes = workflow._plan_notes(sub_cap_sweeps=False)

    assert reference.plan_note("CT_out", "CASE_000", stored, manager.stored_attributes) is None
    assert notes and all("covers 67.5%" in note for note in notes)


def test_unknown_key_is_refused_with_its_path(tmp_path: Path) -> None:
    """The strict mode: a typo'd key is a parse error, never a silently-used default."""
    _write_source(tmp_path)
    config_path = _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", "memory_budge: auto"))
    with pytest.raises(ConfigError, match="memory_budge"):
        _build(tmp_path)


def test_second_run_skips_and_overwrite_rewrites(tmp_path: Path) -> None:
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    before, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "CASE_000")

    # Second run, same config: every case is planned SKIP and nothing is recomputed.
    workflow = _build(tmp_path)
    plan = workflow.compute_plan(1, overwrite=False)
    assert [entry.verdict for entry in plan.entries] == ["SKIP", "SKIP"]

    # Forced rewrite: the boundary probes answer "not written", finalize renames over the entries.
    import os

    os.environ["KONFAI_OVERWRITE"] = "True"
    workflow = _build(tmp_path)
    workflow.setup(1)
    stamp = (tmp_path / "out.h5").stat().st_mtime_ns
    workflow.run_process(1, 0, 0, [])
    assert (tmp_path / "out.h5").stat().st_mtime_ns != stamp, "the forced rewrite never wrote"
    after, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "CASE_000")
    np.testing.assert_array_equal(after, before)


def test_multi_rank_accepts_a_directory_store_and_refuses_a_single_file(tmp_path: Path) -> None:
    """Ranks shard by CASE, and a directory dataset gives each case its own store, so their writes
    are disjoint. Only a single-file h5 puts every case in one handle."""
    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_dir'}/:omezarr\n",
    )
    _build(tmp_path).setup(4)  # a directory store must NOT be refused

    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out_file"))
    with pytest.raises(TransformerError, match="single-file store"):
        _build(tmp_path).setup(4)


def test_overwrite_rewrites_a_geometry_changing_chain_correctly(tmp_path: Path) -> None:
    """A forced rewrite replans from the case as STORED, never from the satisfied boundary:
    planning a second run reads the OUTPUT's header, and a rewrite replanned from that geometry
    would resample by a factor of 1 and overwrite the deliverable with untransformed data."""
    import os

    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Resample:\n"
        "                spacing: [1.0, 3.0, 4.0]\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    workflow = _build(tmp_path)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    before, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "CASE_000")

    os.environ["KONFAI_OVERWRITE"] = "True"
    workflow = _build(tmp_path)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    after, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "CASE_000")
    # The rewrite must land on the TARGET grid again, not the source's.
    assert after.shape == before.shape
    np.testing.assert_array_equal(after, before)


def test_plan_flag_stops_before_transforming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--plan`` must SHORT-CIRCUIT in the CLI dispatch.

    The distributed wrapper filters kwargs by the entrypoint's signature, so a 'plan' flag merely
    passed through would be dropped and the run would proceed as if the flag had never been given --
    printing a plan and then transforming anyway.
    """
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "konfai",
            "TRANSFORM",
            "--config",
            str(tmp_path / "Transform.yml"),
            "--plan",
            "--transforms-dir",
            "Transforms",
        ],
    )
    from konfai.main import main

    main()
    # The plan ran; the transform did not.
    assert not Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_streamed_run_never_reads_a_whole_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory bound, asserted as a bound over the WHOLE workflow: plan, setup and run of a
    streamable chain must never assemble a volume. Value equivalence alone would pass even if the
    streamed path loaded everything."""
    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the transform workflow read a whole volume")

    monkeypatch.setattr(Dataset, "read_data", refuse)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    monkeypatch.undo()
    assert Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_the_console_says_the_plan_in_one_line_whatever_the_cohort_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """TRANSFORM is the one workflow that plans, and it printed the whole plan: its per-case notes
    and case lists grow with the cohort, so a startup that should read like every other workflow's
    (a few lines, then the bar) became a page. The console gets a summary that counts what it folds,
    and the full plan opens the run's log.

    Asserted against the cohort rather than against a line count: what must not grow is the
    startup, and a run over a thousand cases says exactly what a run over four says."""
    recorded: list[str] = []

    def startup(root: Path, cases: int) -> list[str]:
        root.mkdir()
        _write_source(root, cases=cases)
        _write_config(root, _STREAMABLE.format(out=root / "out"))
        workflow = _build(root)
        log = root / "Transforms" / "TEST" / "log_0.txt"

        def keep(message: str) -> Path:
            recorded.append(message)
            return log

        monkeypatch.setattr("konfai.transformer.record", keep)
        workflow.setup(1)
        return [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    few = startup(tmp_path / "few", 4)
    many = startup(tmp_path / "many", 12)

    assert len(few) == len(many), (few, many)
    assert many[0].startswith("[KonfAI] listing every case"), "the wait before the plan says what it is"
    assert "12 entr(ies): 12 STREAM" in many[-1]
    assert "log_0.txt" in many[-1]
    # Folded, not dropped: the line points at a plan the log actually holds.
    assert len(recorded) == 2 and recorded[-1].count("STREAM") >= 1


def test_a_reduction_declared_in_yaml_folds_every_case_into_one_output(tmp_path: Path) -> None:
    """The whole point of the wiring: N cases -> 1 entry, declared in YAML, planned and run."""
    _write_source(tmp_path, cases=4)
    _write_config(
        tmp_path,
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Reduce:\n"
        "                operator: Median\n"
        "                output: atlas\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    transformer = _build(tmp_path)
    plan = transformer.compute_plan(1)

    entry = next(entry for entry in plan.entries if entry.reduced)
    assert entry.case == "atlas" and entry.verdict == "REDUCE" and len(entry.reduced) == 4
    assert "REDUCE 4 case(s) -> 1 output 'atlas'" in plan.report()
    assert "cases: " in plan.report()

    transformer.setup(1)
    transformer.run_process(1, 0, 0, None)

    written, attribute = Dataset(tmp_path / "out", "h5").read_data("CT_out", "atlas")
    sources = [
        np.clip(Dataset(tmp_path / "source", "mha").read_data("CT", name)[0].astype(np.float32), 0.0, 50.0)
        for name in sorted(Dataset(tmp_path / "source", "mha").get_names("CT"))
    ]
    np.testing.assert_allclose(written, np.median(np.stack(sources), axis=0), rtol=1e-6)
    assert attribute["konfai_reduce_cases"].count("|") == 3


def test_a_reduction_that_cannot_stream_refuses_whatever_on_fallback_says(tmp_path: Path) -> None:
    """A reduction has no whole-volume path, so `allow` cannot wave it through."""
    _write_source(tmp_path, cases=2)
    _write_config(
        tmp_path,
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Standardize:\n                inverse: false\n"
        "              Reduce:\n                operator: Mean\n                output: atlas\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
        header="  on_fallback: allow\n",
    )
    with pytest.raises(TransformerError, match="no whole-volume path to fall back to"):
        _build(tmp_path).setup(1)


def test_a_finished_reduction_is_skipped_on_the_second_run(tmp_path: Path) -> None:
    _write_source(tmp_path, cases=3)
    _write_config(
        tmp_path,
        "              Reduce:\n                operator: Mean\n                output: atlas\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    transformer = _build(tmp_path)
    transformer.setup(1)
    transformer.run_process(1, 0, 0, None)
    first, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "atlas")

    again = _build(tmp_path)
    plan = again.compute_plan(1)
    assert next(entry for entry in plan.entries if entry.reduced).verdict == "SKIP"
    again.setup(1)
    again.run_process(1, 0, 0, None)
    second, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "atlas")
    np.testing.assert_array_equal(second, first)


# --------------------------------------------------------------- 1-to-N expansion

_EXPAND_HEADER = """\
  on_fallback: warn
  manual_seed: 7
"""


def _write_expand_config(tmp_path: Path, transforms_yaml: str) -> Path:
    source = tmp_path / "source"
    config_path = tmp_path / "Transform.yml"
    config_path.write_text(
        "Transformer:\n"
        "  name: TEST\n"
        f"{_EXPAND_HEADER}"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {source}:mha\n"
        "    memory_budget: auto\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          CT_out:\n"
        "            transforms:\n"
        f"{transforms_yaml}"
    )
    return config_path


_EXPAND_CHAIN = """\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Expand:
                nb: 3
                pattern: "{{name}}_r{{a:02d}}"
              Brightness:
                b_std: 0.3
              Write:
                dataset: {out}:h5
"""


def test_the_run_seed_reaches_the_draws_through_the_real_config(tmp_path: Path) -> None:
    """``manual_seed`` is declared on the workflow and consumed by the draws, four layers down.

    Through the real YAML on purpose: the chains are bound by the workflow (to check cardinality
    before any manager exists) and again by ``Data.prepare``, so a seed stamped on the first set of
    stage objects reaches nothing unless binding is idempotent. Every layer in between has to hold.
    """
    _write_source(tmp_path, cases=1)
    _write_expand_config(tmp_path, _EXPAND_CHAIN.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    manager = workflow.dataset.managers["CT_out"][0]
    assert manager._expand is not None
    assert manager._expand.draw_seed == 7  # _EXPAND_HEADER declares manual_seed: 7


def test_an_expansion_plans_and_writes_one_entry_per_copy(tmp_path: Path) -> None:
    _write_source(tmp_path, cases=2)
    _write_expand_config(tmp_path, _EXPAND_CHAIN.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    plan = workflow.compute_plan(1, overwrite=False)
    # Two cases x three copies: the copies ARE the plan's lines, each with its own resume.
    assert len(plan.entries) == 6
    assert {entry.expanded_from for entry in plan.entries} == {"CASE_000", "CASE_001"}
    assert [entry.case for entry in plan.entries if entry.expanded_from == "CASE_000"] == [
        "CASE_000_r01",
        "CASE_000_r02",
        "CASE_000_r03",
    ]
    assert {entry.verdict for entry in plan.entries} == {"STREAM"}
    # A pointwise draw rides the case's single read pass: the regime the engine exists for.
    assert {entry.regime for entry in plan.entries} == {"shared"}
    assert "EXPAND 2 case(s) -> 6 cop(ies)" in plan.report()
    assert "shared read pass" in plan.report()

    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    out = Dataset(tmp_path / "out", "h5")
    for case in ("CASE_000", "CASE_001"):
        for a in range(1, 4):
            assert out.is_dataset_exist("CT_out", f"{case}_r{a:02d}")
        assert not out.is_dataset_exist("CT_out", case)


def test_an_expanded_copy_carries_its_draw_and_resumes_per_copy(tmp_path: Path) -> None:
    _write_source(tmp_path, cases=1)
    _write_expand_config(tmp_path, _EXPAND_CHAIN.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])

    out = Dataset(tmp_path / "out", "h5")
    source = Dataset(tmp_path / "source", "mha").read_data("CT", "CASE_000")[0]
    clipped = np.clip(source, 0.0, 50.0)
    copies = [out.read_data("CT_out", f"CASE_000_r{a:02d}")[0] for a in range(1, 4)]
    # Each copy is a DIFFERENT draw of the clipped case: identical copies would mean the draw was
    # applied at the wrong place (or not at all).
    assert not any(np.array_equal(copy, clipped) for copy in copies)
    assert not np.array_equal(copies[0], copies[1])

    again = _build(tmp_path)
    plan = again.compute_plan(1)
    assert {entry.verdict for entry in plan.entries} == {"SKIP"}
    stamp = (tmp_path / "out.h5").stat().st_mtime_ns
    again.setup(1)
    again.run_process(1, 0, 0, [])
    assert (tmp_path / "out.h5").stat().st_mtime_ns == stamp


def test_transforms_and_draws_interleave_after_the_marker(tmp_path: Path) -> None:
    """`draw, T, draw` is expressible, which is the point of declaring draws in the chain.

    `Rotate` and not a bare `Flip`: two classes carry that name and the loader resolves the
    TRANSFORM first, so a bare `Flip` in a chain is the deterministic axis flip, not the draw.
    """
    from konfai.data.augmentation import DataAugmentation

    _write_source(tmp_path, cases=1)
    _write_expand_config(
        tmp_path,
        "              Expand:\n                nb: 2\n"
        '                pattern: "{name}_r{a:02d}"\n'
        "              Brightness:\n                b_std: 0.2\n"
        "              TensorCast:\n                dtype: float32\n"
        "              Rotate:\n                is_quarter: true\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    workflow = _build(tmp_path)
    chain = workflow.dataset.groups_src["CT"]["CT_out"].transforms
    assert [type(stage).__name__ for stage in chain] == [
        "Expand",
        "Brightness",
        "TensorCast",
        "Rotate",
        "Write",
    ]
    assert isinstance(chain[1], DataAugmentation) and isinstance(chain[3], DataAugmentation)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    out = Dataset(tmp_path / "out", "h5")
    assert out.is_dataset_exist("CT_out", "CASE_000_r01")
    assert out.is_dataset_exist("CT_out", "CASE_000_r02")


def test_a_draw_before_the_marker_is_refused(tmp_path: Path) -> None:
    _write_source(tmp_path, cases=1)
    _write_expand_config(
        tmp_path,
        "              Brightness:\n                b_std: 0.2\n"
        "              Expand:\n                nb: 2\n"
        '                pattern: "{name}_r{a:02d}"\n'
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    with pytest.raises(TransformerError, match="before the Expand marker"):
        _build(tmp_path)


def test_an_expand_with_no_draw_after_it_is_refused(tmp_path: Path) -> None:
    _write_source(tmp_path, cases=1)
    _write_expand_config(
        tmp_path,
        "              Expand:\n                nb: 2\n"
        '                pattern: "{name}_r{a:02d}"\n'
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    with pytest.raises(TransformerError, match="every copy would be"):
        _build(tmp_path)


def test_expand_and_reduce_in_one_chain_are_refused(tmp_path: Path) -> None:
    _write_source(tmp_path, cases=2)
    _write_expand_config(
        tmp_path,
        "              Expand:\n                nb: 2\n"
        '                pattern: "{name}_r{a:02d}"\n'
        "              Brightness:\n                b_std: 0.2\n"
        "              Reduce:\n                operator: Mean\n                output: atlas\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    with pytest.raises(TransformerError, match="both an Expand and a Reduce"):
        _build(tmp_path)


def test_an_expanded_run_never_reads_a_whole_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory bound, as a bound: a streamed expansion must never assemble a case."""
    _write_source(tmp_path, cases=2)
    _write_expand_config(tmp_path, _EXPAND_CHAIN.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("the expanded run read a whole volume")

    monkeypatch.setattr(Dataset, "read_data", refuse)
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    monkeypatch.undo()

    out = Dataset(tmp_path / "out", "h5")
    assert out.is_dataset_exist("CT_out", "CASE_000_r01")
    assert out.is_dataset_exist("CT_out", "CASE_001_r03")


def test_a_config_without_the_transformer_root_is_refused(tmp_path: Path) -> None:
    """A typo'd root ('Transformr:') must not bind an all-defaults workflow over the user's file."""
    _write_source(tmp_path)
    (tmp_path / "Transform.yml").write_text("Transformr:\n  name: X\n")
    with pytest.raises(ConfigError, match="no 'Transformer' root"):
        _build(tmp_path)


def test_a_typo_stage_argument_is_refused_instead_of_binding_its_default(tmp_path: Path) -> None:
    """A stage argument the signature does not name is a parse error: the binder would otherwise
    materialize the default beside the typo, and `Clip: {min_val: 0}` would clip at -1024, exit 0."""
    _write_source(tmp_path)
    _write_config(
        tmp_path, "              Clip: {min_val: 0.0, max_value: 400.0}\n" + _WRITE.format(out=tmp_path / "out")
    )
    with pytest.raises(ConfigError, match=r"transforms\.Clip\.min_val'.*Did you mean 'min_value'"):
        _build(tmp_path)


def test_a_reduce_mapping_carries_its_operator_arguments_and_refuses_a_typo(tmp_path: Path) -> None:
    """A Reduce's mapping legitimately holds its operator's own parameters (the stage binds them at
    prepare, from the same mapping); only a key neither signature names is refused."""
    _write_source(tmp_path)
    operator = f"{__name__}:_TrimmedMean"
    template = "              Reduce: {{operator: " + operator + ", output: atlas{extra}}}\n" + _WRITE
    _write_config(tmp_path, template.format(extra=", trim: 0.2", out=tmp_path / "out"))
    workflow = _build(tmp_path)
    reduce = next(s for s in workflow.dataset.groups_src["CT"]["CT_out"].transforms if isinstance(s, Reduce))
    assert isinstance(reduce.operator, _TrimmedMean) and reduce.operator.trim == 0.2

    _write_config(tmp_path, template.format(extra=", grib: shape_only", out=tmp_path / "out"))
    with pytest.raises(ConfigError, match=r"transforms\.Reduce\.grib'.*Did you mean 'grid'"):
        _build(tmp_path)


class _TrimmedMean(Mean):
    """A custom operator with a parameter of its own, bound from the Reduce mapping."""

    def __init__(self, trim: float = 0.0) -> None:
        super().__init__()
        self.trim = float(trim)


def test_a_stage_that_resolves_nowhere_is_left_to_the_loader(tmp_path: Path) -> None:
    """An unresolvable stage name must not be reported as a wrong ARGUMENT: the loader owns that
    refusal and names the searched packages."""
    from konfai.utils.errors import TransformError

    _write_source(tmp_path)
    _write_config(tmp_path, "              Clipp: {min_value: 0.0}\n" + _WRITE.format(out=tmp_path / "out"))
    with pytest.raises(TransformError, match="No transform or augmentation is named 'Clipp'"):
        _build(tmp_path)


def test_a_key_of_a_wrapped_foreign_stage_that_it_cannot_take_is_refused(tmp_path: Path) -> None:
    """The binder hands a foreign class only the parameters its signature names, so a key under a
    class taking none (or **kwargs) goes nowhere: refused, not silently dropped."""
    _write_source(tmp_path)
    _write_config(tmp_path, "              torch.nn:Sigmoid: {inplace: true}\n" + _WRITE.format(out=tmp_path / "out"))
    with pytest.raises(ConfigError, match=r"torch\.nn:Sigmoid\.inplace'"):
        _build(tmp_path)


def test_on_fallback_error_holds_at_run_time_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'error' means nothing is quietly written by the whole-volume path: a fallback only discovered
    mid-run (here, a sweep that fails after a green plan) raises at that case instead of costing an
    unannounced volume."""
    from konfai.data.patching import DatasetManager

    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"), header="  on_fallback: error\n")
    workflow = _build(tmp_path)
    workflow.setup(1)
    monkeypatch.setattr(
        DatasetManager,
        "_materialize_save",
        lambda self, sweep: self._sweep_failed_because(sweep, "OSError: no space left on device"),
    )
    with pytest.raises(TransformerError, match="whole-volume path at run time"), pytest.warns(UserWarning):
        workflow.run_process(1, 0, 0, [])
    assert not Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_a_run_time_fallback_refusal_names_the_budget_it_broke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Its remedy is 'raise memory_budget', so the refusal must say what the case needs AND what
    the budget grants: with only the first figure the reader has to go and find the second."""
    from konfai.data.patching import DatasetManager

    _write_source(tmp_path)
    config_path = _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    # Between the smallest region this chain can sweep (1.56 KiB) and the whole volume it falls
    # back to (7.50 KiB): the plan streams, and only the fallback the failure forces is over budget.
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", "memory_budget: 4KiB"))
    workflow = _build(tmp_path)
    workflow.setup(1)
    monkeypatch.setattr(
        DatasetManager,
        "_materialize_save",
        lambda self, sweep: self._sweep_failed_because(sweep, "OSError: no space left on device"),
    )

    with (
        pytest.raises(TransformerError, match=r"exceeds the per-rank budget \(4.00 KiB\)"),
        pytest.warns(UserWarning),
    ):
        workflow.run_process(1, 0, 0, [])


def test_a_failing_case_does_not_stop_the_shard_and_is_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One unreadable case among three: the other two are written, the rank finishes its shard, and
    it exits non-zero naming the failed case. Before, the rank died at the broken case and every
    case sorted after it was silently never written, with no summary of what failed."""
    _write_source(tmp_path, cases=3)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)
    workflow.setup(1)

    # The failure is injected at the read, not filed as a truncated MetaImage: what a short file
    # does is the reader version's business (a warning and a zero fill reads as success), and what
    # this pins is the rank's behaviour around a case it cannot read, whatever made it unreadable.
    def refuse(name: str, method):
        def read(self, groups, entry, *args, **kwargs):
            if entry == name:
                raise OSError("simulated read failure")
            return method(self, groups, entry, *args, **kwargs)

        return read

    # Both doors: the streamed read AND the whole-volume one the rank falls back to, or the case
    # would be written by the fallback and this would test nothing.
    monkeypatch.setattr(Dataset, "read_data_slice", refuse("CASE_001", Dataset.read_data_slice))
    monkeypatch.setattr(Dataset, "read_data", refuse("CASE_001", Dataset.read_data))
    with pytest.raises(TransformerError, match="1 of 3 work item\\(s\\) failed") as raised, pytest.warns(UserWarning):
        workflow.run_process(1, 0, 0, [])
    assert "CASE_001" in str(raised.value)
    out = Dataset(tmp_path / "out", "h5")
    assert out.is_dataset_exist("CT_out", "CASE_000")
    assert out.is_dataset_exist("CT_out", "CASE_002"), "the cases after the broken one must still be written"
    assert not out.is_dataset_exist("CT_out", "CASE_001")


def test_two_ranks_partition_the_cases_and_every_output_is_written_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DDP contract, executed: the shards partition the work items, and running each rank's
    shard writes every case exactly once: a case in no shard (or in two) is a wrong dataset
    delivered with exit code 0."""
    import konfai.transformer as transformer_module
    import torch

    _write_source(tmp_path, cases=3)
    _write_config(
        tmp_path,
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_dir'}/:omezarr\n",
    )
    workflow = _build(tmp_path)
    workflow.setup(2)
    flattened = sorted(index for shard in workflow._shards for index in shard)
    assert flattened == list(range(3)), "the shards must partition the work items exactly"
    assert all(workflow._shards), "three equal cases over two ranks: neither rank idles"
    # Both ranks run in THIS process, without the launcher that narrows CUDA_VISIBLE_DEVICES to
    # one GPU each: on a machine with fewer GPUs than ranks, rank 1 would name cuda:1. The
    # contract under test is the sharding, so the chain stays on CPU.
    monkeypatch.setattr(transformer_module, "get_device", lambda _rank: torch.device("cpu"))
    workflow.run_process(2, 0, 0, [])
    workflow.run_process(2, 1, 1, [])
    out = Dataset(f"{tmp_path / 'out_dir'}/", "omezarr")
    for index in range(3):
        assert out.is_dataset_exist("CT_out", f"CASE_{index:03d}")


def test_the_plan_reads_no_voxel_for_a_global_statistic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Standardize needs the volume's Mean/Std. The plan only checks the source can provide them
    (headers); the rank reads them at first data access, so a 10k-case plan is not 10k full passes
    on the launcher before the first byte. The written result is the standardized volume."""
    from konfai.utils import dataset as dataset_module

    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Standardize:\n                inverse: false\n"
        f"              Write:\n                dataset: {tmp_path / 'out'}:h5\n",
    )
    scans: list[str] = []
    real = dataset_module.Dataset.read_data_statistics

    def counted(self, group, name, channels=None):
        scans.append(name)
        return real(self, group, name, channels)

    monkeypatch.setattr(dataset_module.Dataset, "read_data_statistics", counted)
    workflow = _build(tmp_path)
    plan = workflow.compute_plan(1)
    assert [entry.verdict for entry in plan.entries] == ["STREAM", "STREAM"]
    assert scans == [], "planning must not scan a volume"
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    assert sorted(scans) == ["CASE_000", "CASE_001"], "one scan per case, on the rank"
    source, _ = Dataset(tmp_path / "source", "mha").read_data("CT", "CASE_000")
    written, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "CASE_000")
    np.testing.assert_allclose(written, (source - source.mean()) / source.std(ddof=1), rtol=1e-4, atol=1e-4)


def test_a_loaded_case_takes_its_statistic_from_the_loaded_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOAD reads the source once: a Standardize on that route computes on the volume in hand rather
    than paying a second full read for a disk scan of the same numbers."""
    from konfai.utils import dataset as dataset_module

    rng = np.random.default_rng(3)
    source = Dataset(tmp_path / "source", "nii.gz")
    volume = (rng.random((1, 8, 32, 32)) * 100).astype(np.float32)
    source.write("CT", "CASE_000", volume, _image_attributes())
    config_path = _write_config(
        tmp_path,
        "              Standardize:\n                inverse: false\n"
        f"              Write:\n                dataset: {tmp_path / 'out'}:h5\n",
    )
    config_path.write_text(
        config_path.read_text().replace("memory_budget: auto", f"memory_budget: {3 * volume.nbytes}b")
    )
    monkeypatch.setattr(
        dataset_module.Dataset, "read_data_statistics", lambda *_a, **_k: pytest.fail("no disk scan on the LOAD route")
    )
    workflow = _build(tmp_path)
    plan = workflow.compute_plan(1)
    assert plan.entries[0].verdict == "LOAD"
    workflow.setup(1)
    workflow.run_process(1, 0, 0, [])
    written, _ = Dataset(tmp_path / "out", "h5").read_data("CT_out", "CASE_000")
    np.testing.assert_allclose(written, (volume - volume.mean()) / volume.std(ddof=1), rtol=1e-4, atol=1e-4)


def test_a_chain_through_its_own_save_cache_is_priced_as_bounded(tmp_path: Path) -> None:
    """The segment past a Save the run itself sweeps reads a cache that does not exist at planning
    time; a missing entry answers 'unbounded' to bounded_region_reads. Priced that way, an mha case
    with an h5 Save in the middle was routed LOAD for no reason: the cache lands on a region-write
    store, and every one of those serves bounded reads."""
    rng = np.random.default_rng(3)
    source = Dataset(tmp_path / "source", "mha")
    source.write("CT", "CASE_000", (rng.random((1, 8, 32, 32)) * 100).astype(np.float32), _image_attributes())
    transforms = f"""\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Save:
                dataset: {tmp_path / "cache"}:h5
              TensorCast:
                dtype: float32
              Write:
                dataset: {tmp_path / "out"}:h5
"""
    config_path = _write_config(tmp_path, transforms, header="  on_fallback: error\n")
    budget = 3 * 8 * 32 * 32 * 4
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", f"memory_budget: {budget}b"))
    plan = _build(tmp_path).compute_plan()
    entry = next(entry for entry in plan.entries if entry.case == "CASE_000")
    assert entry.verdict == "STREAM", entry.reason


def test_the_working_set_counts_the_widest_stages_own_buffers(tmp_path: Path) -> None:
    """The estimators guarding the budget were 2 to 4.5x optimistic: a Gradient's whole-volume call
    holds several volumes beside its input and output. A stage declares what it allocates
    (working_multiple) and the plan sizes the fallback working set with it, so this reads the
    declaration rather than a number: re-measuring a stage must not need a test edit."""
    from konfai.data.patching import CASE_ELEMENT_BYTES, FALLBACK_INFLIGHT_FACTOR
    from konfai.data.transform import Gradient

    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Gradient: {}\n"
        "              Standardize:\n                inverse: false\n"
        f"              Write:\n                dataset: {tmp_path / 'out'}:h5\n",
    )
    plan = _build(tmp_path).compute_plan()
    entry = plan.entries[0]
    case = 12 * 10 * 8 * CASE_ELEMENT_BYTES
    assert Gradient.working_multiple > 0.0, "the widest stage of this chain declares its buffers"
    assert entry.working_set_bytes == case * (FALLBACK_INFLIGHT_FACTOR + Gradient.working_multiple)
    assert "widest stage" in plan.report()


def test_the_decomposition_note_is_printed_only_where_it_can_matter(tmp_path: Path) -> None:
    """Sweeping a landing in several blocks changes no byte of a pointwise or separable chain; the
    plan says so only for a chain that interpolates through per-voxel coordinates."""
    rng = np.random.default_rng(3)
    source = Dataset(tmp_path / "source", "mha")
    source.write("CT", "CASE_000", (rng.random((1, 8, 32, 32)) * 100).astype(np.float32), _image_attributes())
    for chain, sensitive in (
        ("              Clip:\n                min_value: 0.0\n                max_value: 50.0\n", False),
        ("              Resample:\n                spacing: [1.0, 1.0, 1.0]\n", False),  # axis-aligned: factorises
        (
            "              Resample:\n                spacing: [1.0, 1.0, 1.0]\n                transforms: {transform: true}\n",
            True,
        ),
    ):
        if sensitive:
            import SimpleITK as sitk

            stored = sitk.Euler3DTransform()
            stored.SetRotation(0.05, -0.03, 0.08)
            (tmp_path / "source" / "CASE_000").mkdir(exist_ok=True)
            sitk.WriteTransform(stored, str(tmp_path / "source" / "CASE_000" / "transform.itk.txt"))
        config_path = _write_config(
            tmp_path, chain + f"              Write:\n                dataset: {tmp_path / 'out'}:h5\n"
        )
        config_path.write_text(
            config_path.read_text().replace("memory_budget: auto", f"memory_budget: {3 * 8 * 32 * 32 * 4}b")
        )
        plan = _build(tmp_path).compute_plan()
        assert plan.entries[0].verdict == "STREAM", plan.entries[0].reason
        assert any("more than one block" in note for note in plan.notes) is sensitive, (chain, plan.notes)


def test_a_bare_name_past_the_marker_is_the_draw(tmp_path: Path) -> None:
    """Flip exists as a transform and as a draw. Before Expand the bare name is the transform;
    after it, the copies' draw: `Flip: {f_prob: ...}` past the marker no longer binds the transform
    and fails on f_prob."""
    from konfai.data.augmentation import Flip as FlipDraw
    from konfai.data.transform import Flip as FlipTransform

    _write_source(tmp_path)
    _write_config(
        tmp_path,
        "              Flip:\n                dims: '0'\n"
        f"              Write:\n                dataset: {tmp_path / 'out'}:h5\n",
    )
    assert isinstance(next(iter(_build(tmp_path).dataset.managers.values()))[0].transforms[0], FlipTransform)
    _write_config(
        tmp_path,
        "              Expand:\n                nb: 2\n"
        "              Flip:\n                f_prob: [1.0, 0.0, 0.0]\n"
        f"              Write:\n                dataset: {tmp_path / 'out2'}:h5\n",
    )
    assert isinstance(next(iter(_build(tmp_path).dataset.managers.values()))[0].transforms[1], FlipDraw)


def test_the_shards_balance_bytes_not_counts(tmp_path: Path) -> None:
    """One 8x larger case among small ones: it takes a rank on its own and the small ones share the
    other, where an index split would pair the large case with half of the small ones."""
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / "source", "mha")
    for index in range(5):
        source.write("CT", f"CASE_{index:03d}", rng.random((1, 4, 8, 8)).astype(np.float32), _image_attributes())
    source.write("CT", "CASE_BIG", rng.random((1, 32, 8, 8)).astype(np.float32), _image_attributes())
    _write_config(tmp_path, _DIRECTORY_CHAIN.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)
    workflow.setup(2)
    big = next(index for index, item in enumerate(workflow._items) if item.manager.name == "CASE_BIG")
    shard_of_big = next(shard for shard in workflow._shards if big in shard)
    assert shard_of_big == [big]
    assert sorted(len(shard) for shard in workflow._shards) == [1, 5]


@pytest.mark.parametrize("destination", ["mha", "omezarr"])
def test_a_bounded_source_streams_whatever_the_destination_format(tmp_path: Path, destination: str) -> None:
    """The route is priced on what the chain READS. A chain ending on a Write leaves nothing to
    read past the boundary, so the destination (which does not exist yet, and cannot answer
    whether it serves bounded reads) must not be priced as a source: an mha case written to
    mha or omezarr streams exactly as it does to h5."""
    rng = np.random.default_rng(3)
    source = Dataset(tmp_path / "source", "mha")
    volume = (rng.random((1, 8, 32, 32)) * 100).astype(np.float32)
    source.write("CT", "CASE_000", volume, _image_attributes())
    out = tmp_path / "out"
    suffix = "/" if destination == "omezarr" else ""
    transforms = f"""\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Write:
                dataset: {out}{suffix}:{destination}
"""
    config_path = _write_config(tmp_path, transforms, header="  on_fallback: error\n")
    budget = 3 * 8 * 32 * 32 * 4  # fits, while the sweep splits the case into several slabs
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", f"memory_budget: {budget}b"))
    workflow = _build(tmp_path)
    plan = workflow.compute_plan()
    entry = next(entry for entry in plan.entries if entry.case == "CASE_000")
    assert entry.verdict == "STREAM", entry.reason


def test_a_case_that_fits_is_loaded_when_streaming_would_reread_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route is chosen from predicted cost against the budget, never the answer.

    A gzipped NIfTI cannot serve bounded region reads, so streaming decodes the whole source once
    per slab; a case whose working set fits the budget is then LOADED (one read), and the plan
    says so with the factor. LOAD is a choice, not a fallback: on_fallback=error must not refuse
    it, and the bytes match the streamed route of the same chain.
    """
    rng = np.random.default_rng(3)
    source = Dataset(tmp_path / "source", "nii.gz")
    volume = (rng.random((1, 8, 32, 32)) * 100).astype(np.float32)
    source.write("CT", "CASE_000", volume, _image_attributes())
    out = tmp_path / "out"
    transforms = f"""\
              Clip:
                min_value: 0.0
                max_value: 50.0
              Write:
                dataset: {out}:h5
"""
    config_path = _write_config(tmp_path, transforms, header="  on_fallback: error\n")
    # 3x the case: fits (working set = 2x), while the sweep rows drop below the case's extent.
    budget = 3 * 8 * 32 * 32 * 4
    config_path.write_text(config_path.read_text().replace("memory_budget: auto", f"memory_budget: {budget}b"))
    workflow = _build(tmp_path)
    plan = workflow.compute_plan()
    entry = next(entry for entry in plan.entries if entry.case == "CASE_000")
    assert entry.verdict == "LOAD"
    assert "x the source" in (entry.reason or "")
    assert not plan.fallback_entries  # a choice, not a fallback: on_fallback has nothing to refuse

    workflow.setup(1)
    # The bytes of a pointwise chain cannot tell the routes apart, so the ROUTE itself is spied:
    # the run must hand materialize the plan's choice, not re-derive its own.
    from konfai.data.materialize import CaseMaterializer

    routes: list[bool] = []
    original = CaseMaterializer.materialize

    def spy(self: CaseMaterializer, a: int = 0, **kwargs) -> bool:
        routes.append(bool(kwargs.get("prefer_whole", False)))
        return original(self, a, **kwargs)

    monkeypatch.setattr(CaseMaterializer, "materialize", spy)
    workflow.run_process(1, 0, 0, None)
    assert routes == [True], "the plan said LOAD and the run must execute it"
    loaded, _ = Dataset(out, "h5").read_data("CT_out", "CASE_000")
    np.testing.assert_array_equal(loaded, np.clip(volume, 0.0, 50.0))


# ------------------------------------------------------------- the plan is the run


def _run_counters(console: str) -> dict[str, int]:
    """The counters of the rank's closing line, keyed like the plan's verdicts."""
    import re

    closing = re.compile(
        r"\[KonfAI\] (?:rank \d+/\d+ )?done in [\d.]+ s: \d+ written"
        r" \((?P<STREAM>\d+) streamed, (?P<LOAD>\d+) loaded, (?P<WHOLE>\d+) whole-volume, (?P<REDUCE>\d+) reduced\)"
        r"(?:, (?P<SKIP>\d+) already written)?"
    )
    found = [match for match in map(closing.search, console.splitlines()) if match is not None]
    assert len(found) == 1, f"expected one closing line, got {len(found)} in:\n{console}"
    groups = found[0].groupdict()
    return {
        "STREAM": int(groups["STREAM"]),
        "LOAD": int(groups["LOAD"]),
        "WHOLE-VOLUME": int(groups["WHOLE"]),
        "REDUCE": int(groups["REDUCE"]),
        "SKIP": int(groups["SKIP"] or 0),
    }


def _plan_counters(plan) -> dict[str, int]:
    counts = dict.fromkeys(("STREAM", "LOAD", "WHOLE-VOLUME", "REDUCE", "SKIP"), 0)
    for entry in plan.entries:
        counts[entry.verdict] += 1
    return counts


def _write_mixed_cohort(tmp_path: Path) -> Path:
    """Two bounded mha cases and one gzipped NIfTI case, read through two chains: one that streams
    (or LOADs, for the source that cannot serve bounded reads) and one that falls back."""
    rng = np.random.default_rng(3)
    bounded = Dataset(tmp_path / "source", "mha")
    for index in range(2):
        volume = (rng.random((1, 8, 32, 32)) * 100).astype(np.float32)
        bounded.write("CT", f"CASE_{index:03d}", volume, _image_attributes())
    gzipped = Dataset(tmp_path / "source_gz", "nii.gz")
    gzipped.write("CT", "CASE_002", (rng.random((1, 8, 32, 32)) * 100).astype(np.float32), _image_attributes())
    budget = 3 * 8 * 32 * 32 * 4  # the case fits (working set 2x), the sweep still slabs it
    config_path = tmp_path / "Transform.yml"
    config_path.write_text(
        "Transformer:\n"
        "  name: TEST\n"
        "  on_fallback: warn\n"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {tmp_path / 'source'}:mha\n"
        f"      - {tmp_path / 'source_gz'}:nii.gz\n"
        f"    memory_budget: {budget}b\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          A:\n"
        "            transforms:\n"
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_a'}:h5\n"
        "          B:\n"
        "            transforms:\n"
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Standardize:\n                inverse: false\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_b'}:h5\n"
    )
    return config_path


def test_the_run_counters_equal_the_plan_verdicts_over_a_mixed_cohort(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan is the run's own verdict, counted: STREAM, LOAD, WHOLE-VOLUME and SKIP together.

    Each verdict is decided by a different question (the patch planner, the read-cost pricing, the
    fallback, the resume), and the run answers them again on its own path; only the totals say the
    two agree for every kind at once, on one cohort, in one run.
    """
    _write_mixed_cohort(tmp_path)
    workflow = _build(tmp_path)
    plan = workflow.compute_plan(1, overwrite=False)
    assert _plan_counters(plan) == {"STREAM": 2, "LOAD": 1, "WHOLE-VOLUME": 3, "REDUCE": 0, "SKIP": 0}

    workflow.setup(1)
    capsys.readouterr()
    workflow.run_process(1, 0, 0, [])
    assert _run_counters(capsys.readouterr().out) == _plan_counters(plan)

    # A resume with one chain's outputs gone: SKIP beside the recomputed fallbacks, still equal.
    (tmp_path / "out_b.h5").unlink()
    again = _build(tmp_path)
    plan = again.compute_plan(1, overwrite=False)
    assert _plan_counters(plan) == {"STREAM": 0, "LOAD": 0, "WHOLE-VOLUME": 3, "REDUCE": 0, "SKIP": 3}
    again.setup(1)
    capsys.readouterr()
    again.run_process(1, 0, 0, [])
    assert _run_counters(capsys.readouterr().out) == _plan_counters(plan)


def test_the_plan_names_the_regime_the_resumed_copies_take(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a resume of an expansion the plan's regime column is the engine's own answer.

    The engine shares one read pass between the copies of a case that are still to write and can
    ride it; a shared pass with ONE member is that copy's own sweep. Two of four copies left share,
    one of four sweeps solo, and the plan says which before the run does it.
    """
    from konfai.data.materialize import CaseMaterializer

    _write_source(tmp_path, cases=2)
    _write_expand_config(tmp_path, _EXPAND_CHAIN.format(out=tmp_path / "out").replace("nb: 3", "nb: 4"))
    workflow = _build(tmp_path)
    workflow.setup(1)
    engines = {manager.name: CaseMaterializer(manager) for manager in workflow.dataset.managers["CT_out"]}
    assert set(engines["CASE_000"].materialize_copies([1, 2]).values()) == {(Verdict.STREAM, Regime.SHARED)}
    assert set(engines["CASE_001"].materialize_copies([1, 2, 3]).values()) == {(Verdict.STREAM, Regime.SHARED)}

    again = _build(tmp_path)
    plan = again.compute_plan(1, overwrite=False)
    planned = {entry.case: (entry.verdict, entry.regime) for entry in plan.entries}
    assert planned == {
        "CASE_000_r01": ("SKIP", None),
        "CASE_000_r02": ("SKIP", None),
        "CASE_000_r03": ("STREAM", "shared"),
        "CASE_000_r04": ("STREAM", "shared"),
        "CASE_001_r01": ("SKIP", None),
        "CASE_001_r02": ("SKIP", None),
        "CASE_001_r03": ("SKIP", None),
        "CASE_001_r04": ("STREAM", "solo"),
    }
    assert "2 STREAM (shared read pass), 1 STREAM (own pass)" in plan.report()
    assert "(1 cop(ies)) own pass: the only copy of this case still to write" in plan.report()

    ran: dict[str, tuple[Verdict, Regime | None]] = {}
    original = CaseMaterializer.materialize_copies

    def spy(self: CaseMaterializer, copies: list[int], **kwargs):
        outcomes = original(self, copies, **kwargs)
        ran.update({self.manager.copy_entry(a): outcome for a, outcome in outcomes.items()})
        return outcomes

    monkeypatch.setattr(CaseMaterializer, "materialize_copies", spy)
    again.setup(1)
    again.run_process(1, 0, 0, [])
    # The run's answer per copy is the plan's line, verdict and regime alike.
    assert ran == {case: outcome for case, outcome in planned.items() if outcome[0] == "STREAM"}


# ---------------------------------------------------------------- refusals and draws


def test_a_save_without_a_dataset_is_refused_at_parse(tmp_path: Path) -> None:
    """A Save that names no dataset would cache next to the source: refused before any byte, with
    the remedy, whether the mapping is empty or carries only a group."""
    _write_source(tmp_path)
    for spelling in ("              Save: {}\n", "              Save:\n                group: cache\n"):
        _write_config(
            tmp_path,
            "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
            f"{spelling}"
            "              Write:\n"
            f"                dataset: {tmp_path / 'out'}:h5\n",
        )
        with pytest.raises(TransformerError, match="'Save' with no dataset"):
            _build(tmp_path)
    assert not (tmp_path / "out.h5").exists()


def _write_mask_file(tmp_path: Path) -> tuple[Path, np.ndarray]:
    import SimpleITK as sitk

    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(tmp_path / "mask.mha"))
    return tmp_path / "mask.mha", mask


def test_a_mask_draw_after_the_marker_lands_every_copy_on_the_masks_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``Mask`` the DRAW (a window of the case under a stored mask, drawn per copy) is reachable
    from a chain by its augmentation classpath, since the bare name is the transform of that name.

    It declares WHOLE_VOLUME, so the plan sends every copy down the fallback and the run agrees;
    what lands is on the mask's grid, zero outside the mask, and a window of the clipped case
    inside it.
    """
    _write_source(tmp_path, cases=2)
    mask_path, mask = _write_mask_file(tmp_path)
    _write_expand_config(
        tmp_path,
        "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
        "              Expand:\n                nb: 2\n"
        '                pattern: "{name}_r{a:02d}"\n'
        "              konfai.data.augmentation:Mask:\n"
        f"                mask: {mask_path}\n"
        "                value: 0.0\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out'}:h5\n",
    )
    workflow = _build(tmp_path)
    plan = workflow.compute_plan(1, overwrite=False)
    assert len(plan.entries) == 4
    assert {entry.verdict for entry in plan.entries} == {"WHOLE-VOLUME"}
    assert all(entry.reason and "'Mask'" in entry.reason for entry in plan.entries)

    workflow.setup(1)
    capsys.readouterr()
    workflow.run_process(1, 0, 0, [])
    assert _run_counters(capsys.readouterr().out) == _plan_counters(plan)

    out = Dataset(tmp_path / "out", "h5")
    source = Dataset(tmp_path / "source", "mha")
    for case in ("CASE_000", "CASE_001"):
        clipped = np.clip(source.read_data("CT", case)[0], 0.0, 50.0)[0]
        for a in (1, 2):
            written = out.read_data("CT_out", f"{case}_r{a:02d}")[0]
            assert written.shape == (1, *mask.shape)
            assert not written[0][mask == 0].any(), "outside the mask the copy must hold the fill value"
            inside = written[0][1:5, 1:5, 1:5]
            windows = (
                clipped[z : z + 4, y : y + 4, x : x + 4]
                for z in range(clipped.shape[0] - 3)
                for y in range(clipped.shape[1] - 3)
                for x in range(clipped.shape[2] - 3)
            )
            assert any(np.array_equal(inside, window) for window in windows), "the copy is not a window of the case"


# ------------------------------------------------------------------ the plan text


def _write_snapshot_cohort(tmp_path: Path) -> Path:
    """Every kind of plan line at once, on two ranks: a chain that streams and LOADs, one that falls
    back, an Expand with copies already written (both regimes, and the solo demotion), a reduction
    that streams and one that is refused, a subset that drops a case, and a resumed plain case."""
    rng = np.random.default_rng(3)
    bounded = Dataset(tmp_path / "source", "mha")
    for name in ("CASE_000", "CASE_001", "CASE_003"):
        bounded.write("CT", name, (rng.random((1, 8, 32, 32)) * 100).astype(np.float32), _image_attributes())
    gzipped = Dataset(tmp_path / "source_gz", "nii.gz")
    gzipped.write("CT", "CASE_002", (rng.random((1, 8, 32, 32)) * 100).astype(np.float32), _image_attributes())
    budget = 3 * 8 * 32 * 32 * 4
    clip = "              Clip:\n                min_value: 0.0\n                max_value: 50.0\n"
    config_path = tmp_path / "Transform.yml"
    config_path.write_text(
        "Transformer:\n"
        "  name: TEST\n"
        "  on_fallback: warn\n"
        "  manual_seed: 7\n"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {tmp_path / 'source'}:mha\n"
        f"      - {tmp_path / 'source_gz'}:nii.gz\n"
        f"    memory_budget: {budget}b\n"
        "    subset: '~CASE_003'\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          A:\n"
        "            transforms:\n"
        f"{clip}"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_a'}:h5\n"
        "          B:\n"
        "            transforms:\n"
        f"{clip}"
        "              Standardize:\n                inverse: false\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_b'}:h5\n"
        "          C:\n"
        "            transforms:\n"
        f"{clip}"
        "              Expand:\n                nb: 3\n                pattern: '{name}_r{a:02d}'\n"
        "              Brightness:\n                b_std: 0.3\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_c'}:h5\n"
        "          D:\n"
        "            transforms:\n"
        f"{clip}"
        "              Reduce:\n                operator: Mean\n                output: atlas\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_d'}:h5\n"
        "          E:\n"
        "            transforms:\n"
        f"{clip}"
        "              Standardize:\n                inverse: false\n"
        "              Reduce:\n                operator: Mean\n                output: atlas\n"
        "              Write:\n"
        f"                dataset: {tmp_path / 'out_e'}:h5\n"
    )
    # Outputs already there: one plain case, one copy of CASE_000, two copies of CASE_001.
    written = (rng.random((1, 8, 32, 32)) * 100).astype(np.float32)
    Dataset(tmp_path / "out_a", "h5").write("A", "CASE_000", written, _image_attributes())
    out_c = Dataset(tmp_path / "out_c", "h5")
    for name in ("CASE_000_r01", "CASE_001_r01", "CASE_001_r02"):
        out_c.write("C", name, written, _image_attributes())
    return config_path


_SNAPSHOT_SUMMARY = (
    "[KonfAI] plan over 2 rank(s) | 17 entr(ies): 1 LOAD, 1 REDUCE, 1 REFUSED, 4 SKIP, 7 STREAM,"
    " 3 WHOLE-VOLUME | per-rank budget 96.00 KiB ('98304b', per rank: x2 = 192.00 KiB on the node)"
    " | 1 case(s) dropped"
)

_SNAPSHOT_REPORT = """\
[KonfAI] plan over 2 rank(s) | per-rank budget 96.00 KiB ('98304b', per rank: x2 = 192.00 KiB on the node) | held beside the regions: engine ~32.00 MiB, decoded-chunk cache up to 32.00 KiB (under the 256.00 MiB floor: a region touching more OME-Zarr chunks than it holds decodes them again) | fallback working set = case x 4 B x (2 + the widest stage's own buffers), headers-only estimate | output dtype/channels assumed float32 / source channels until the first slab
[KonfAI] 1 case(s) of 'CT' are DROPPED: the run keeps the cases every groups_src shares, minus what 'subset' excludes.
  CT -> A (Clip -> Write <tmp>/out_a:h5): 3 case(s) -- 1 STREAM, 1 LOAD, 0 WHOLE-VOLUME, 1 SKIP (output already written)
    (1 case(s)) LOAD: fits the per-rank budget (~64.00 KiB vs 96.00 KiB); streaming would read ~2.0x the source
  CT -> B (Clip -> Standardize -> Write <tmp>/out_b:h5): 3 case(s) -- 0 STREAM, 0 LOAD, 3 WHOLE-VOLUME, 0 SKIP (output already written)
    (3 case(s)) WHOLE-VOLUME: stage 1 'Standardize' needs whole-volume statistics, but an earlier stage changes the values: the stored volume's statistic is not this stage's input.
    worst fallback case ~= 96.00 KiB vs per-rank budget 96.00 KiB
  CT -> C (Clip -> Expand -> Brightness -> Write <tmp>/out_c:h5): EXPAND 3 case(s) -> 9 cop(ies): 5 STREAM (shared read pass), 1 STREAM (own pass), 0 WHOLE-VOLUME, 3 SKIP (copy already written)
    (1 cop(ies)) own pass: the only copy of this case still to write; a shared pass with one member is its own sweep.
  CT -> D (Clip -> Reduce -> Write <tmp>/out_d:h5): REDUCE 3 case(s) -> 1 output 'atlas': REDUCE
    2 resident region(s) of 6 row(s) = 0.00 GiB  (incremental accumulator)
    reads: 1 of 3 member(s) sit on nii.gz, which decodes the whole volume behind every region read: 2 decodes per member (one per region), 2 in all
    put a Save ...:h5 before the Reduce so each member is materialized on a bounded store first
    peak ~= 48.00 KiB vs per-rank budget 96.00 KiB
    cases: CASE_000, CASE_001, CASE_002
  CT -> E (Clip -> Standardize -> Reduce -> Write <tmp>/out_e:h5): REDUCE 3 case(s) -> 1 output 'atlas': REFUSED
    case 'CASE_000': stage 1 'Standardize' needs whole-volume statistics, but an earlier stage changes the values: the stored volume's statistic is not this stage's input.
    cases: CASE_000, CASE_001, CASE_002"""


def test_the_plan_text_is_the_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``report()`` and ``summary()`` byte for byte, over every kind of line the plan can print.

    The plan opens the run's log and is what Studio and the MCP dry-run show; its text is a
    contract, and this pins it across refactors of the code that assembles it.
    """
    monkeypatch.delenv("KONFAI_LOCAL_RANKS", raising=False)
    monkeypatch.setattr(
        "konfai.data.patching._sweep_pipeline_depth", lambda: 1
    )  # the LOAD line's ~4.0x is priced at depth 1
    _write_snapshot_cohort(tmp_path)
    plan = _build(tmp_path).compute_plan(2, overwrite=False)
    assert plan.summary() == _SNAPSHOT_SUMMARY
    assert plan.report().replace(tmp_path.as_posix(), "<tmp>").replace(str(tmp_path), "<tmp>") == _SNAPSHOT_REPORT


def test_the_plan_bounds_the_chunk_cache_by_the_budget_and_says_when_it_is_under_the_floor(tmp_path: Path) -> None:
    """The store's decoded-chunk cache is part of what the process holds: a third of the budget and
    never more, whatever the floor, because the sizing spends the rest on regions. Under the floor
    the cache cannot keep a region's chunks, and the header says that rather than the run paying
    every decode twice in silence."""
    from konfai.utils import ome_zarr

    _write_source(tmp_path)
    config_path = _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"))
    text = config_path.read_text()
    try:
        config_path.write_text(text.replace("memory_budget: auto", "memory_budget: 98304b"))
        report = _build(tmp_path).compute_plan().report()
        assert ome_zarr._chunk_cache().capacity == 98304 // 3
        assert "decoded-chunk cache up to 32.00 KiB (under the 256.00 MiB floor:" in report

        config_path.write_text(text.replace("memory_budget: auto", "memory_budget: 3GiB"))
        report = _build(tmp_path).compute_plan().report()
        assert ome_zarr._chunk_cache().capacity == 1 << 30
        assert "decoded-chunk cache up to 1.00 GiB" in report
    finally:
        ome_zarr.set_chunk_cache_budget(None)


@pytest.mark.parametrize(
    ("on_fallback", "file_format"), [("allow", "omezarr"), ("warn", "omezarr"), ("error", "omezarr"), ("warn", "h5")]
)
def test_a_remote_write_destination_is_refused_by_the_plan_whatever_on_fallback_says(
    tmp_path: Path, on_fallback: str, file_format: str
) -> None:
    """A remote root is read-only, and no route writes it: the streamed sweep refuses at its first
    slab and the whole-volume path at its write, after reading and transforming a case. The plan
    says REFUSED, not WHOLE-VOLUME, setup raises before a byte is read whatever on_fallback says,
    and the run directory names no output: a Path of a URI resolves to a local directory named
    after the scheme, which never existed."""
    pytest.importorskip("fsspec")
    _write_source(tmp_path)
    _write_config(
        tmp_path, _WRITE.format(out=f"memory://bucket/out:{file_format}")[:-4], f"  on_fallback: {on_fallback}\n"
    )
    workflow = _build(tmp_path)

    plan = workflow.compute_plan()
    assert [entry.verdict for entry in plan.entries] == ["REFUSED", "REFUSED"]
    assert "remote root" in (plan.entries[0].reason or "")
    assert not plan.fallback_entries, "no route writes there: nothing to fall back to"
    assert "2 REFUSED" in plan.summary()
    assert "0 SKIP (output already written), 2 REFUSED" in plan.report()
    assert "REFUSED: 'memory://bucket/out" in plan.report()

    with pytest.raises(TransformerError, match="local path"):
        workflow.setup(1)
    assert not (tmp_path / "Transforms" / "TEST" / "outputs.json").exists()
