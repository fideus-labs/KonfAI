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

"""The IMPACT loss settings the three engines share: per-layer expansion, per-level models and what is refused."""

import pytest
from impact_reg_konfai.models.impact_loss import LevelSpec, ModelSpec, check_models, layer_weights, level_models


def test_a_single_weight_spreads_over_every_kept_layer() -> None:
    specs = [
        ModelSpec(ref="a.pt", layers_mask="0110"),
        ModelSpec(ref="b.pt", layers_mask="011", layers_weight=[0.5]),
        ModelSpec(ref="c.pt", layers_mask="11", layers_weight=[2.0, 3.0]),
    ]
    assert layer_weights(specs) == [1.0, 1.0, 0.5, 0.5, 2.0, 3.0]


def test_a_number_weighs_every_kept_layer_as_a_one_value_list_does() -> None:
    """The published presets write layers_weight: 1.0, as its help allows; a list-only annotation refused them."""
    from impact_reg_konfai.models.fireants import RegistrationNet
    from konfai.utils.config import _coerce_config_value
    from konfai_apps.app_repository import apply_overrides

    assert ModelSpec(ref="a.pt", layers_mask="011", layers_weight=0.5).layers_weight == [0.5]
    annotation = ModelSpec.__dataclass_fields__["layers_weight"].type
    assert _coerce_config_value(0.8, annotation.__origin__, "layers_weight") == 0.8
    block = {"models": {"0": {"ref": "a.pt", "layers_weight": [1.0]}}}
    config = {"Predictor": {"Model": {"classpath": "m:RegistrationNet", "RegistrationNet": block}}}
    apply_overrides(config, ["Predictor.Model.RegistrationNet.models.0.layers_weight=0.8"], lambda: RegistrationNet)
    assert block["models"]["0"]["layers_weight"] == 0.8


@pytest.mark.parametrize(
    ("spec", "dense", "message"),
    [
        (ModelSpec(ref="a.pt", layers_mask="000"), True, "keeps no layer"),
        (ModelSpec(ref="a.pt", layers_mask="1x"), True, "other than 0 and 1"),
        (ModelSpec(ref="a.pt", layers_mask="011", layers_weight=[1.0, 2.0, 3.0]), True, "3 values for 2 kept layers"),
        (ModelSpec(ref="a.pt", layers_weight=[-1.0]), True, "non-negative"),
        (ModelSpec(ref="a.pt", voxel_size=[2.0, 0.0, 2.0]), True, "voxel_size must be positive"),
        (ModelSpec(ref="a.pt", distance="LNCC"), False, "does not have"),
    ],
)
def test_what_no_engine_can_run_fails_before_it_starts(spec: ModelSpec, dense: bool, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_models([spec], "engine", dense=dense)


def test_the_dense_engines_take_lncc() -> None:
    check_models([ModelSpec(ref="a.pt", distance="LNCC")], "engine", dense=True)


def test_levels_replace_models_level_by_level() -> None:
    mind, other = ModelSpec(ref="mind.pt"), ModelSpec(ref="other.pt")
    assert level_models({"0": mind}, {}, 3, "engine") == [[mind]] * 3
    # Keyed '0', '1', ... and read in numeric order, as 'models'.
    levels = {"1": LevelSpec(models={"1": other, "0": mind}), "0": LevelSpec(models={"0": other})}
    assert level_models({"0": mind}, levels, 2, "engine") == [[other], [mind, other]]
    with pytest.raises(ValueError, match="2 entries for 3 levels"):
        level_models({"0": mind}, levels, 3, "engine")
