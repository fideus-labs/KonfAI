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

"""The checkpoint budget bounds retained allocations, without thrashing a cyclic ensemble."""

import gc
import os
import sys
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.reduction import Mean
from konfai.network.network import Network
from konfai.predictor.ensemble import ModelComposite, _checkpoint_bytes
from konfai.utils.errors import PredictorError


class CacheNet(Network):
    def __init__(self) -> None:
        super().__init__(in_channels=1, dim=2)
        self.add_module("Conv", torch.nn.Conv2d(1, 1, 1))


class PayloadNet(Network):
    def __init__(self) -> None:
        super().__init__(in_channels=1, dim=2)
        self.scale = 1.0

    def load(self, state_dict, init=True, ema=False):  # type: ignore[override]
        self.scale = float(state_dict["scale"])


def _payload(scale: float, elements: int = 4096) -> dict:
    return {"scale": scale, "aux": torch.zeros(elements)}


def _paths(tmp_path: Path, count: int) -> list[Path]:
    paths = [tmp_path / f"fold-{index}.pt" for index in range(count)]
    for index, path in enumerate(paths):
        torch.save(_payload(float(index + 1)), path)
    return paths


def _count_reads(composite, monkeypatch) -> Counter:
    reads = Counter()
    read = composite._read_state_source

    def counted(source):
        reads[str(source)] += 1
        return read(source)

    monkeypatch.setattr(composite, "_read_state_source", counted)
    return reads


def test_storage_accounting_charges_the_allocation_behind_views_once() -> None:
    storage_owner = torch.zeros(16384)
    small, other = storage_owner[:1], storage_owner[1:2]
    one = _checkpoint_bytes({"first": small})
    two = _checkpoint_bytes({"first": small, "second": other})
    assert one is not None and two is not None
    assert one >= storage_owner.untyped_storage().nbytes()
    assert two - one < 1024  # another view and key, not another 64 KiB allocation


def test_five_members_below_the_budget_deserialize_once(tmp_path, monkeypatch) -> None:
    composite = ModelComposite(PayloadNet(), Mean())
    paths = _paths(tmp_path, 5)
    reads = _count_reads(composite, monkeypatch)
    composite.load(paths)
    for _ in range(4):
        for index in range(5):
            assert composite._model_for_index(index).scale == index + 1
            assert composite._state_cache_bytes <= composite._cache_limit_bytes
    assert list(reads.values()) == [1] * 5
    assert set(composite._state_cache) == set(range(5))


def test_oversubscribed_cycles_keep_hits_instead_of_thrashing(tmp_path, monkeypatch) -> None:
    composite = ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=40000 / 1024**3)
    paths = _paths(tmp_path, 5)
    reads = _count_reads(composite, monkeypatch)
    composite.load(paths)
    for _ in range(4):
        for index in range(5):
            assert composite._model_for_index(index).scale == index + 1
            assert composite._state_cache_bytes <= 40000
    assert list(composite._state_cache) == [0, 1]
    assert [reads[str(path)] for path in paths] == [1, 1, 4, 4, 4]


@pytest.mark.parametrize("budget", [0, 100])
def test_disabled_or_oversized_path_cache_loads_without_retaining(tmp_path, monkeypatch, budget) -> None:
    composite = ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=budget / 1024**3)
    paths = _paths(tmp_path, 2)
    reads = _count_reads(composite, monkeypatch)
    composite.load(paths)
    for _ in range(3):
        for index in range(2):
            assert composite._model_for_index(index).scale == index + 1
    assert list(reads.values()) == [3, 3]
    assert composite._state_cache_bytes == 0 and not composite._state_cache


@pytest.mark.parametrize(
    "replacement",
    [
        "in_place",
        "atomic",
        pytest.param(
            "preserved_mtime",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="Windows st_ctime is the creation time: the rewrite keeps every stat field",
            ),
        ),
    ],
)
def test_single_member_reloads_when_its_file_changes(tmp_path, monkeypatch, replacement) -> None:
    composite = ModelComposite(PayloadNet(), Mean())
    paths = _paths(tmp_path, 1)
    path = paths[0]
    reads = _count_reads(composite, monkeypatch)
    composite.load(paths)
    assert composite._model_for_index(0).scale == 1
    before = path.stat()
    if replacement == "atomic":
        temporary = tmp_path / "replacement.pt"
        torch.save(_payload(9.0), temporary)
        temporary.replace(path)
    else:
        torch.save(_payload(9.0), path)
        if replacement == "preserved_mtime":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif path.stat().st_mtime_ns == before.st_mtime_ns:
            # A rewrite inside the file system's timestamp tick (Windows): the next tick tells it apart.
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    assert composite._model_for_index(0).scale == 9
    assert reads[str(path)] == 2


def test_a_growing_cached_member_evicts_the_least_recently_used_reloadable_member(tmp_path) -> None:
    paths = _paths(tmp_path, 3)
    composite = ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=54000 / 1024**3)
    composite.load(paths)
    for index in range(3):
        composite._model_for_index(index)
    composite._model_for_index(0)  # refresh 0: 2 is older when 1 grows
    torch.save(_payload(8.0, elements=8192), paths[1])
    assert composite._model_for_index(1).scale == 8
    assert list(composite._state_cache) == [0, 1]
    assert composite._state_cache_bytes <= 54000


def test_a_file_modified_during_deserialization_is_refused(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path, 1)
    composite = ModelComposite(PayloadNet(), Mean())
    read = composite._read_state_source

    def racing(source):
        state = read(source)
        torch.save(_payload(7.0), source)
        return state

    monkeypatch.setattr(composite, "_read_state_source", racing)
    with pytest.raises(PredictorError, match="changed while being read"):
        composite.load(paths)


def test_dict_sources_drop_optimizer_ownership_and_reserve_their_storage() -> None:
    weights = CacheNet().network_states()
    optimizer = torch.zeros(65536)
    optimizer_ref = weakref.ref(optimizer)
    source = {"Model": weights, "optimizer": optimizer}
    composite = ModelComposite(CacheNet(), Mean())
    composite.load([source])
    del source, optimizer
    gc.collect()
    assert optimizer_ref() is None
    assert set(composite._state_sources[0]) == {"Model"}
    assert composite._state_cache_bytes == _checkpoint_bytes({"Model": weights})


def test_stock_loader_releases_unused_ema_weights_and_keeps_declared_model() -> None:
    source, ema = CacheNet(), CacheNet()
    with torch.no_grad():
        source["Conv"].weight.fill_(2)
        ema["Conv"].weight.fill_(9)
    ema_states = ema.network_states()
    ema_tensor_ref = weakref.ref(ema_states["CacheNet"]["Conv.weight"])
    composite = ModelComposite(CacheNet(), Mean())
    composite.load([{"Model": source.network_states(), "Model_EMA": ema_states}])
    del ema_states, ema
    gc.collect()
    assert ema_tensor_ref() is None
    assert float(composite._model_for_index(0)["Conv"].weight.sum()) == 2
    assert set(composite._state_cache[0]) == {"Model"}


def test_a_custom_container_cannot_hide_allocations_from_the_budget() -> None:
    class HiddenPayload(list):
        pass

    hidden = HiddenPayload()
    hidden.payload = torch.zeros(65536)
    assert _checkpoint_bytes({"scale": 1.0, "custom": hidden}) is None
    with pytest.raises(PredictorError, match="file paths"):
        ModelComposite(PayloadNet(), Mean()).load([{"scale": 1.0, "custom": hidden}])


def test_retained_gradients_are_charged_and_unknown_autograd_graphs_bypass_cache() -> None:
    leaf = torch.zeros(16384, requires_grad=True)
    leaf.grad = torch.ones_like(leaf)
    size = _checkpoint_bytes(leaf)
    assert size is not None and size >= leaf.untyped_storage().nbytes() * 2
    assert _checkpoint_bytes(leaf * 2) is None


def test_dict_sources_cannot_bypass_the_budget_through_tiny_views() -> None:
    source = {"scale": 2.0, "aux": torch.zeros(65536)[:1]}
    composite = ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=10000 / 1024**3)
    with pytest.raises(PredictorError, match="file paths"):
        composite.load([source])
    with pytest.raises(PredictorError, match="In-memory"):
        ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=0).load([{"scale": 2.0}])


def test_resident_dicts_and_reloadable_files_share_one_budget(tmp_path) -> None:
    paths = _paths(tmp_path, 2)
    composite = ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=40000 / 1024**3)
    composite.load([_payload(4.0), *paths])
    for index in range(3):
        composite._model_for_index(index)
        assert composite._state_cache_bytes <= 40000
    assert list(composite._state_cache) == [0, 1]


def test_uncached_custom_payloads_are_released_and_reload_drops_the_old_cache(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path, 2)
    composite = ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=20000 / 1024**3)
    read = composite._read_state_source
    references = []

    def recorded(source):
        state = read(source)
        references.append(weakref.ref(state["aux"]))
        return state

    monkeypatch.setattr(composite, "_read_state_source", recorded)
    composite.load(paths)
    composite._model_for_index(0)
    composite._model_for_index(1)
    gc.collect()
    assert references[0]() is not None and references[1]() is None
    composite.load([])  # weightless custom loader
    gc.collect()
    assert references[0]() is None and composite._state_cache_bytes == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_custom_objects_with_unknown_size_load_from_file_without_caching(tmp_path, legacy) -> None:
    path = tmp_path / "numpy-custom.pt"
    torch.save({"scale": 6.0, "custom": np.ones(4)}, path, _use_new_zipfile_serialization=not legacy)
    composite = ModelComposite(PayloadNet(), Mean())
    composite.load([path])
    assert composite._model_for_index(0).scale == 6
    assert not composite._state_cache and composite._state_cache_bytes == 0


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), 1e308])
def test_invalid_cache_budget_is_refused(budget) -> None:
    with pytest.raises(PredictorError, match="finite and non-negative"):
        ModelComposite(PayloadNet(), Mean(), checkpoint_cache_gib=budget)


def test_predictor_keeps_urls_reloadable_instead_of_retaining_downloaded_payloads(monkeypatch) -> None:
    from konfai.predictor.workflow import Predictor

    predictor = object.__new__(Predictor)
    predictor.path_to_models = ["https://example.invalid/fold.pt"]
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda **kwargs: pytest.fail("eager download"))
    assert predictor._load() == predictor.path_to_models


def test_checkpoint_cache_budget_binds_from_real_prediction_config_and_reaches_composite(tmp_path, monkeypatch) -> None:
    import ruamel.yaml
    from konfai.predictor.workflow import build_predict
    from konfai.utils.dataset import Attribute, Dataset

    # A real source, model, config binder and setup; no mocked Predictor constructor.
    for key in (
        "KONFAI_config_file",
        "KONFAI_ROOT",
        "KONFAI_STATE",
        "KONFAI_CONFIG_MODE",
        "KONFAI_PREDICTIONS_DIRECTORY",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.chdir(tmp_path)
    Dataset(str(tmp_path / "Dataset"), "mha").write("CT", "CASE", np.ones((1, 1, 4, 4), dtype=np.float32), Attribute())
    model_path = tmp_path / "fold.pt"
    torch.save({"Model": CacheNet().network_states()}, model_path)
    config_path = tmp_path / "Prediction.yml"
    config_path.write_text(
        "Predictor:\n"
        "  checkpoint_cache_gib: 0.125\n"
        "  check_training_transforms: false\n"
        "  Model:\n"
        "    classpath: test_predictor_checkpoint_cache:CacheNet\n"
        "  Dataset:\n"
        "    dataset_filenames: [./Dataset:a:mha]\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          CT: {is_input: true}\n"
        "    Patch: {patch_size: [1, 4, 4], overlap: 0}\n"
        "    num_workers: 0\n"
        "  outputs_dataset:\n"
        "    Conv:\n"
        "      OutputDataset: {same_as_group: 'CT:CT', group: OUT, dataset_filename: 'Dataset:mha'}\n"
    )
    predictor = build_predict([model_path], config_path, tmp_path / "Predictions")
    predictor.setup(1)
    assert predictor.checkpoint_cache_gib == 0.125
    assert predictor.model_composite._cache_limit_bytes == 128 * 1024**2
    assert ruamel.yaml.YAML().load(config_path.read_text())["Predictor"]["checkpoint_cache_gib"] == 0.125
