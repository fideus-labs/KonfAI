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
    assert (tmp_path / "Transforms" / "TEST" / "plan.txt").read_text().count("STREAM")


_REGION_CHAIN = """\
              ResampleToResolution:
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
    assert Dataset(tmp_path / "out", "omezarr").get_names("CT_out") == []


def test_a_planned_workflow_survives_the_spawn_that_runs_it(tmp_path: Path) -> None:
    """``setup`` runs on the launcher and ``mp.spawn`` then pickles the workflow whole.

    So everything planning leaves behind — the memoized stream sources, the region stages' pull
    maps — has to cross a process boundary, and anything unpicklable in there kills every chain the
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
    byte — never 40 cases written then a crash at the 41st."""
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
    one's cache already written, adopts it as its own source, and skips its own prefix -- producing a
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
              ResampleToReference:
                entry: CASE_000
                group: CT
              Write:
                dataset: {out}:h5
"""


def test_a_stage_is_asked_about_its_own_input_not_the_case_as_stored(tmp_path: Path) -> None:
    """The note a stage declares describes the grid it MEETS, which the stages before it decide.

    The resample takes 12x10x8 down to 12x7x2 and keeps voxel zero where it is, so the case's far
    edge falls short of the reference's and part of the output will be fill. Asked about the case as
    STORED it covers all of it and says nothing -- and the plan would stay silent about that fill.

    (With ``align: extent``, the default, the box is preserved and the answer is honestly 100%.
    67.5% is also exactly what the old ``ResampleToResolution`` recorded here -- because its header
    said origin-aligned while its data was extent-aligned, which is the mismatch this stage no
    longer has.)
    """
    _write_source(tmp_path)
    _write_config(tmp_path, _RESAMPLED_THEN_REFERENCED.format(out=tmp_path / "out"))
    workflow = _build(tmp_path)
    manager = workflow._managers()["CT_out"][0]
    stored = [int(extent) for extent in manager.base_shape[1:]]
    reference = manager.transforms[1]

    notes = workflow._plan_notes()

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
        "              ResampleToResolution:\n"
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

    manager = workflow.dataset._prepared_data["CT_out"][0]
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
    # A pointwise draw rides the case's single read pass -- the regime the engine exists for.
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
    from konfai.transformer import _reject_unknown_keys

    config = tmp_path / "Transform.yml"
    config.write_text("Transformr:\n  name: X\n")
    with pytest.raises(ConfigError, match="no 'Transformer' root"):
        _reject_unknown_keys(config)


def test_a_typo_stage_argument_is_refused_instead_of_binding_its_default(tmp_path: Path) -> None:
    """A stage argument the signature does not name is a parse error: the binder would otherwise
    materialize the default beside the typo, and `Clip: {min_val: 0}` would clip at -1024, exit 0."""
    from konfai.transformer import _reject_unknown_keys

    config = tmp_path / "Transform.yml"
    config.write_text(
        "Transformer:\n  Dataset:\n    groups_src:\n      CT:\n        groups_dest:\n"
        "          CT_out:\n            transforms:\n"
        "              Clip: {min_val: 0.0, max_value: 400.0}\n"
        "              Write: {dataset: ./Out:h5}\n"
    )
    with pytest.raises(ConfigError, match=r"Unknown argument 'min_val' for 'Clip'"):
        _reject_unknown_keys(config)


def test_a_reduce_mapping_carries_its_operator_arguments_and_refuses_a_typo(tmp_path: Path) -> None:
    """A Reduce's mapping legitimately holds its operator's own parameters (resolve_operator binds
    them from the same mapping); only a key neither signature names is refused."""
    from konfai.transformer import _reject_unknown_keys

    config = tmp_path / "Transform.yml"
    template = (
        "Transformer:\n  Dataset:\n    groups_src:\n      CT:\n        groups_dest:\n"
        "          CT_out:\n            transforms:\n"
        "              Reduce: {{operator: Mean, output: atlas{extra}}}\n"
        "              Write: {{dataset: ./Out:h5}}\n"
    )
    config.write_text(template.format(extra=", grid: shape_only"))
    _reject_unknown_keys(config)
    config.write_text(template.format(extra=", grib: shape_only"))
    with pytest.raises(ConfigError, match=r"Unknown argument 'grib' for 'Reduce'"):
        _reject_unknown_keys(config)


def test_a_stage_that_resolves_nowhere_is_left_to_the_loader(tmp_path: Path) -> None:
    """An unresolvable stage name must not be reported as a wrong ARGUMENT: the loader owns that
    refusal and names the searched packages."""
    from konfai.transformer import _reject_unknown_keys

    config = tmp_path / "Transform.yml"
    config.write_text(
        "Transformer:\n  Dataset:\n    groups_src:\n      CT:\n        groups_dest:\n"
        "          CT_out:\n            transforms:\n"
        "              Clipp: {min_value: 0.0}\n"
        "              Write: {dataset: ./Out:h5}\n"
    )
    _reject_unknown_keys(config)


def test_on_fallback_error_holds_at_run_time_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'error' means nothing is quietly written by the whole-volume path: a fallback only discovered
    mid-run (here, a sweep that fails after a green plan) raises at that case instead of costing an
    unannounced volume."""
    from konfai.data.patching import DatasetManager
    from konfai.utils.errors import PatchError

    _write_source(tmp_path)
    _write_config(tmp_path, _STREAMABLE.format(out=tmp_path / "out"), header="  on_fallback: error\n")
    workflow = _build(tmp_path)
    workflow.setup(1)
    monkeypatch.setattr(
        DatasetManager,
        "_materialize_save",
        lambda self, sweep: self._sweep_failed_because(sweep, "OSError: no space left on device"),
    )
    with pytest.raises(PatchError, match="whole-volume path at run time"), pytest.warns(UserWarning):
        workflow.run_process(1, 0, 0, [])
    assert not Dataset(tmp_path / "out", "h5").is_dataset_exist("CT_out", "CASE_000")


def test_the_strict_grammar_knows_every_key_the_binder_can_read() -> None:
    """The grammar mirrors the two __init__ signatures: a parameter added to either without a
    grammar entry would be refused in every config that sets it."""
    import inspect

    from konfai.data.data_manager import DataTransform
    from konfai.transformer import _STRICT_GRAMMAR, Transformer

    workflow_params = set(inspect.signature(Transformer.__init__).parameters) - {"self", "dataset"}
    assert workflow_params | {"Dataset"} == set(_STRICT_GRAMMAR)
    dataset_grammar = _STRICT_GRAMMAR["Dataset"]
    assert isinstance(dataset_grammar, dict)
    dataset_params = set(inspect.signature(DataTransform.__init__).parameters) - {"self"}
    assert dataset_params == set(dataset_grammar)


def test_two_ranks_partition_the_cases_and_every_output_is_written_once(tmp_path: Path) -> None:
    """The DDP contract, executed: the shards partition the work items, and running each rank's
    shard writes every case exactly once -- a case in no shard (or in two) is a wrong dataset
    delivered with exit code 0."""
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
    workflow.run_process(2, 0, 0, [])
    workflow.run_process(2, 1, 1, [])
    out = Dataset(f"{tmp_path / 'out_dir'}/", "omezarr")
    for index in range(3):
        assert out.is_dataset_exist("CT_out", f"CASE_{index:03d}")


def test_a_case_that_fits_is_loaded_when_streaming_would_reread_the_source(tmp_path: Path) -> None:
    """The route is chosen from predicted cost against the budget — never the answer.

    A gzipped NIfTI cannot serve bounded region reads, so streaming decodes the whole source once
    per slab; a case whose working set fits the budget is then LOADED — one read — and the plan
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
    workflow.run_process(1, 0, 0, None)
    loaded, _ = Dataset(out, "h5").read_data("CT_out", "CASE_000")
    np.testing.assert_array_equal(loaded, np.clip(volume, 0.0, 50.0))
