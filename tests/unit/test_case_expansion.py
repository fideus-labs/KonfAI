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

"""The 1-to-N cardinality: ``Expand`` as a declared point of the chain, and the engine behind it.

A draw is a stage. It is declared in the chain, where it applies, and it composes with the
transforms and the other draws around it — ``T, draw, T, draw`` means exactly what it reads like.
The properties that would fail without any of this: the copies carry their draw and land under
their own names; a streamed copy equals the whole-volume one; and the copies that can share a read
pass do share it — the optimisation the engine exists for — while the ones that cannot say why.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.augmentation import Brightness, Flip, Permute, Scale
from konfai.data.patching import DatasetManager
from konfai.data.transform import Clip, Expand, Save, TensorCast, Transform, Write, split_expand
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PatchError, TransformError

pytest.importorskip("SimpleITK")


def _image_attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attributes


def _source(tmp_path: Path, name: str = "CASE_000") -> Dataset:
    rng = np.random.default_rng(0)
    dataset = Dataset(tmp_path / "source", "mha")
    dataset.write("CT", name, (rng.random((1, 12, 10, 8)) * 100).astype(np.float32), _image_attributes())
    return dataset


def _draw(augmentation):
    """A draw as the chain declares it: an ordinary stage, loaded so it applies to every copy."""
    augmentation.load(1.0)
    return augmentation


def _drawn_scales(augmentation: Scale, index: int = 0) -> list[torch.Tensor]:
    """The matrices a ``Scale`` actually drew for one case's copies — the draw itself, not its
    effect: a scale keeps the grid, so comparing shapes would compare nothing."""
    return list(augmentation.matrix[index])


def _manager(source: Dataset, transforms: list[Transform], name: str = "CASE_000") -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name=name,
        dataset=source,
        patch=None,
        transforms=transforms,
        data_augmentations_list=[],
    )


# --------------------------------------------------------------------- the marker


def test_expand_refuses_a_pattern_that_cannot_separate_its_entries() -> None:
    with pytest.raises(TransformError, match=r"\{a\}"):
        Expand(pattern="{name}_fixed")
    with pytest.raises(TransformError, match=r"\{name\}"):
        Expand(pattern="copy_{a:02d}")
    with pytest.raises(TransformError, match="cannot format"):
        Expand(pattern="{name}_{nope}")


def test_expand_refuses_a_cardinality_below_one() -> None:
    with pytest.raises(TransformError, match="at least one"):
        Expand(nb=0)


def test_expand_is_never_applied_as_an_ordinary_transform() -> None:
    with pytest.raises(TransformError, match="expands nothing"):
        Expand()("CASE_000", torch.zeros(1, 2, 2, 2), Attribute())


def test_split_expand_is_the_chain_around_its_marker() -> None:
    clip, expand, write = Clip(0.0, 50.0), Expand(), Write("./out:h5")
    assert split_expand([clip, expand, write]) == ([clip], expand, [write])
    # No marker: everything is the shared part, which is what keeps a 1-to-1 chain untouched.
    assert split_expand([clip, write]) == ([clip, write], None, [])


# ------------------------------------------------------- the declared position


def test_copies_are_written_under_their_own_names_and_carry_their_draw(tmp_path: Path) -> None:
    """With the draw declared after the marker, what lands on disk is AUGMENTED.

    Before this feature the draws lived in a separate section and were applied after the whole
    chain — that is, after the Write — so every copy's entry held the un-augmented volume.
    """
    source = _source(tmp_path)
    out = Dataset(tmp_path / "out", "h5")
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=2, pattern="{name}_r{a:02d}"),
            _draw(Flip(f_prob=[0.0, 1.0, 1.0])),
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )

    assert set(manager.materialize_copies([1, 2])) == {1, 2}
    assert out.is_dataset_exist("CT", "CASE_000_r01")
    assert out.is_dataset_exist("CT", "CASE_000_r02")
    # The case's own name is NOT written: a copy is not the case.
    assert not out.is_dataset_exist("CT", "CASE_000")

    clipped = np.clip(source.read_data("CT", "CASE_000")[0], 0.0, 50.0)
    written = out.read_data("CT", "CASE_000_r01")[0]
    assert not np.array_equal(written, clipped), "the copy carries no draw"
    np.testing.assert_array_equal(written, clipped[:, :, ::-1, ::-1])


def test_a_chain_without_expand_is_untouched(tmp_path: Path) -> None:
    """The regression guard: with no marker there are no copies at all."""
    source = _source(tmp_path)
    manager = _manager(source, [Clip(0.0, 50.0)])
    assert manager._expand is None
    assert manager.copy_entry(1) == "CASE_000"
    assert manager.shapes == [[12, 10, 8]]
    with pytest.raises(PatchError, match="declares no Expand"):
        manager.materialize_copies([1])


def test_two_chains_of_one_case_draw_the_same_copies(tmp_path: Path) -> None:
    """Augmenting an image and its mask coherently.

    Every ``groups_dest`` builds its OWN stage objects, so an image chain and a mask chain hold two
    different ``Scale`` instances and cannot agree on the order they would consume a shared random
    generator in. They derive from the seed instead. A mask scaled differently from its image is a
    silently ruined dataset: both outputs look entirely normal on their own.
    """
    source = _source(tmp_path)
    image_draw, mask_draw = _draw(Scale(0.2)), _draw(Scale(0.2))
    _manager(source, [Expand(nb=4), image_draw], name="CASE_000")
    _manager(source, [Expand(nb=4), mask_draw], name="CASE_000")

    drawn = _drawn_scales(image_draw)
    assert len(drawn) == 4 and len({tuple(row.flatten().tolist()) for row in drawn}) == 4
    for mine, theirs in zip(drawn, _drawn_scales(mask_draw), strict=True):
        assert torch.equal(mine, theirs)

    # A different seed is a different set of copies -- otherwise "the same" would prove nothing.
    other_draw = _draw(Scale(0.2))
    _manager(source, [Expand(nb=4, seed=1), other_draw], name="CASE_000")
    assert not torch.equal(_drawn_scales(other_draw)[0], drawn[0])


def test_an_unrelated_draw_in_one_chain_does_not_shift_the_shared_ones(tmp_path: Path) -> None:
    """A mask chain has no use for an intensity draw, and dropping it must not desynchronise the
    geometry. Each draw is keyed on its own class and rank, not on its position in the tail."""
    source = _source(tmp_path)
    with_intensity, alone = _draw(Scale(0.2)), _draw(Scale(0.2))
    _manager(source, [Expand(nb=3), _draw(Brightness(0.2)), with_intensity])
    _manager(source, [Expand(nb=3), alone])

    for mine, theirs in zip(_drawn_scales(with_intensity), _drawn_scales(alone), strict=True):
        assert torch.equal(mine, theirs)


# --------------------------------------------------- draws chain like transforms


def test_a_draw_is_a_stage_that_chains_with_the_transforms_around_it(tmp_path: Path) -> None:
    """``T, draw, T, draw`` — the order is the declared one, and it is planned as ONE chain.

    The streamed result must equal applying exactly that order by hand on the whole volume.
    """
    source = _source(tmp_path)
    first, second = _draw(Brightness(b_std=0.3)), _draw(Flip(f_prob=[0.0, 1.0, 0.0]))
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=2, pattern="{name}_r{a:02d}"),
            first,
            TensorCast(dtype="float32"),
            second,
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    tail = manager._expand_tail(1)
    assert [type(getattr(stage, "augmentation", stage)).__name__ for stage in tail] == [
        "Brightness",
        "TensorCast",
        "Flip",
        "Write",
    ]

    assert set(manager.materialize_copies([1, 2]).values()) <= {"stream", "stream-shared"}

    expected = Clip(0.0, 50.0)(
        "CASE_000", torch.from_numpy(source.read_data("CT", "CASE_000")[0]), Attribute(_image_attributes())
    )
    expected = first.compute("CASE_000", 0, 0, expected)
    expected = TensorCast(dtype="float32")("CASE_000", expected, Attribute())
    expected = second.compute("CASE_000", 0, 0, expected)

    got = Dataset(tmp_path / "out", "h5").read_data("CT", "CASE_000_r01")[0]
    np.testing.assert_allclose(got, expected.numpy(), rtol=0, atol=1e-6)


def test_a_transform_after_a_shape_changing_draw_folds_on_the_copys_grid(tmp_path: Path) -> None:
    """A draw that reorders axes hands the NEXT stage its own extent — the chain, both ways."""
    source = _source(tmp_path)
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=2, pattern="{name}_r{a:02d}"),
            _draw(Permute(prob_permute=[1.0, 0.0])),
            TensorCast(dtype="float32"),
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    assert manager.shapes[0] == [12, 10, 8]
    assert manager.shapes[1] != manager.shapes[0], "the copy's grid did not follow its draw"

    assert set(manager.materialize_copies([1, 2]).values()) <= {"stream", "stream-shared", "whole-volume"}
    written = Dataset(tmp_path / "out", "h5").read_data("CT", "CASE_000_r01")[0]
    assert list(written.shape[1:]) == manager.shapes[1]


def test_chain_stages_is_the_one_definition_of_a_copy(tmp_path: Path) -> None:
    source = _source(tmp_path)
    clip, write = Clip(0.0, 50.0), Write(f"{tmp_path / 'out'}:h5")
    manager = _manager(source, [clip, Expand(nb=2), _draw(Brightness(b_std=0.2)), write])
    # The marker itself is never a stage: it is where the per-copy tail begins.
    assert manager.chain_stages(1) == [clip, *manager._expand_tail(1)]
    assert manager.chain_stages(1)[-1] == write
    # Copy 0 is the case: no draw, so the tail is its transforms alone.
    assert manager.chain_stages(0) == [clip, write]


# ------------------------------------------------- streamed equals whole-volume


def test_a_streamed_copy_equals_the_whole_volume_copy(tmp_path: Path) -> None:
    source = _source(tmp_path)
    # ONE draw instance in both chains: a draw caches its parameters per case index, so the second
    # manager reuses the first one's rather than redrawing.
    draw = _draw(Brightness(b_std=0.3))
    streamed = _manager(
        source,
        [Clip(0.0, 50.0), Expand(nb=3, pattern="{name}_r{a:02d}"), draw, Write(f"{tmp_path / 'streamed'}:h5")],
    )
    assert streamed.materialize_copies([1, 2, 3]) == {1: "stream-shared", 2: "stream-shared", 3: "stream-shared"}

    classic = _manager(
        source,
        [Clip(0.0, 50.0), Expand(nb=3, pattern="{name}_r{a:02d}"), draw, Write(f"{tmp_path / 'classic'}:h5")],
    )
    for a in (1, 2, 3):
        classic._assemble_and_write(a)

    for a in (1, 2, 3):
        entry = f"CASE_000_r{a:02d}"
        got, got_attributes = Dataset(tmp_path / "streamed", "h5").read_data("CT", entry)
        expected, expected_attributes = Dataset(tmp_path / "classic", "h5").read_data("CT", entry)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
        for key in ("Origin", "Spacing", "Direction"):
            np.testing.assert_allclose(
                got_attributes.get_np_array(key), expected_attributes.get_np_array(key), rtol=0, atol=1e-9
            )


# --------------------------------------------------------------- the regimes


def test_the_shared_pass_reads_the_source_once_for_every_copy(tmp_path: Path) -> None:
    """The optimisation, asserted as a bound rather than inferred from a verdict.

    Reads are decompression-bound, so N copies must not cost N reads. Counting them is the only way
    to prove it: a per-copy loop would show three times as many.
    """
    source = _source(tmp_path)
    reads: list[tuple] = []
    original = Dataset.read_data_slice

    def counted(self, groups, name, slices):
        reads.append((groups, name))
        return original(self, groups, name, slices)

    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=3, pattern="{name}_r{a:02d}"),
            _draw(Brightness(b_std=0.3)),
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    Dataset.read_data_slice = counted  # type: ignore[method-assign]
    try:
        regimes = manager.materialize_copies([1, 2, 3])
    finally:
        Dataset.read_data_slice = original  # type: ignore[method-assign]

    assert set(regimes.values()) == {"stream-shared"}
    source_reads = [entry for entry in reads if entry == ("CT", "CASE_000")]
    assert len(source_reads) == 1, f"the copies did not share the read pass: {len(source_reads)} reads"


def test_a_region_draw_takes_its_own_pass_and_the_plan_names_the_draw(tmp_path: Path) -> None:
    """A draw reading elsewhere than its target slab cannot ride the shared slab — and says so."""
    source = _source(tmp_path)
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=2, pattern="{name}_r{a:02d}"),
            _draw(Flip(f_prob=[0.0, 1.0, 1.0])),
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    reason = manager.expansion_solo_reason(1)
    assert reason is not None and "Flip" in reason and "ORIENTATION" in reason

    assert manager.materialize_copies([1, 2]) == {1: "stream", 2: "stream"}
    out = Dataset(tmp_path / "out", "h5")
    assert out.is_dataset_exist("CT", "CASE_000_r01") and out.is_dataset_exist("CT", "CASE_000_r02")


def test_a_whole_volume_draw_falls_back_per_copy_and_still_writes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=2, pattern="{name}_r{a:02d}"),
            _draw(Scale(s_std=0.1)),  # Scale declares WHOLE_VOLUME on purpose
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    assert manager.stream_refusal(1, apply_augmentations=True) is not None
    assert manager.materialize_copies([1, 2]) == {1: "whole-volume", 2: "whole-volume"}
    out = Dataset(tmp_path / "out", "h5")
    assert out.is_dataset_exist("CT", "CASE_000_r01") and out.is_dataset_exist("CT", "CASE_000_r02")
    assert not manager.loaded and not manager.data


def test_the_write_probe_targets_the_copys_grid_not_the_cases(tmp_path: Path) -> None:
    """The plan measures the RUN, so it must probe the extent the copies really write."""
    source = _source(tmp_path)
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Expand(nb=2, pattern="{name}_r{a:02d}"),
            _draw(Permute(prob_permute=[1.0, 0.0])),
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    case_targets = manager.write_targets(0)
    copy_targets = manager.write_targets(1)
    assert [shape for _stage, shape, _attributes in copy_targets] == [manager.shapes[1]]
    assert copy_targets[0][1] != case_targets[0][1], "the probe would validate the pre-draw extent"


# --------------------------------------------------------------------- resume


def test_a_written_copy_is_not_rewritten_and_overwrite_forces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is per copy: the entry already on disk is the one that is skipped.

    Counted at the store's own door rather than read off the file, because both of those proxies
    lie in one direction each -- a timestamp two writes share within a tick reads as 'skipped', and
    so do identical bytes for a draw that is reproducible, which is what this codebase aims at.
    """
    source = _source(tmp_path)
    opened: list[str] = []
    real_open = Dataset.open_data_stream

    def spy(self: Dataset, group: str, entry: str, *args, **kwargs):
        opened.append(entry)
        return real_open(self, group, entry, *args, **kwargs)

    monkeypatch.setattr(Dataset, "open_data_stream", spy)

    def build() -> DatasetManager:
        return _manager(
            source,
            [
                Clip(0.0, 50.0),
                Expand(nb=2, pattern="{name}_r{a:02d}"),
                _draw(Brightness(b_std=0.3)),
                Write(f"{tmp_path / 'out'}:h5"),
            ],
        )

    build().materialize_copies([1, 2])
    written = list(opened)
    assert len(written) == 2, written

    opened.clear()
    build().materialize_copies([1, 2])
    assert opened == [], "a copy already on disk was written again"

    opened.clear()
    build().materialize_copies([1, 2], rewrite=True)
    assert opened == written, "rewrite=True did not force both copies"


def test_a_shared_cache_before_the_marker_is_swept_once_for_every_copy(tmp_path: Path) -> None:
    """A Save before the Expand is the copies' shared work: materialized exactly once."""
    source = _source(tmp_path)
    manager = _manager(
        source,
        [
            Clip(0.0, 50.0),
            Save(f"{tmp_path / 'work'}:h5"),
            Expand(nb=3, pattern="{name}_r{a:02d}"),
            _draw(Brightness(b_std=0.3)),
            Write(f"{tmp_path / 'out'}:h5"),
        ],
    )
    manager.materialize_copies([1, 2, 3], rewrite=True)

    work = Dataset(tmp_path / "work", "h5")
    # The shared cache holds the CASE (no draw), under the case's own name, once.
    assert work.is_dataset_exist("CT", "CASE_000")
    assert not work.is_dataset_exist("CT", "CASE_000_r01")
    clipped = np.clip(source.read_data("CT", "CASE_000")[0], 0.0, 50.0)
    np.testing.assert_allclose(work.read_data("CT", "CASE_000")[0], clipped, rtol=0, atol=1e-6)

    out = Dataset(tmp_path / "out", "h5")
    for a in (1, 2, 3):
        assert out.is_dataset_exist("CT", f"CASE_000_r{a:02d}")
