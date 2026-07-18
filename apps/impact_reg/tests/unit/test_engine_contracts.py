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

"""Build-time contracts of the registration engines: promises the parameter annotations make must hold
at construction, not surface minutes later as a cryptic subprocess or autograd failure."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from impact_reg_konfai.models import elastix_engine as elastix_engine_module
from impact_reg_konfai.models.elastix_engine import ElastixEngine


def test_download_models_accepts_a_local_file_beside_hf_refs(tmp_path: Path, monkeypatch) -> None:
    # The docs promise "a user may still point ``ref`` at a local model (path)"; an unconditional
    # 'repo:filename' split crashed on it. A local ref stages under the very name the map references
    # (_model_key(ref) == the ref), an HF ref under its repo-relative filename.
    local = tmp_path / "custom.pt"
    local.write_bytes(b"jit")
    fetched = tmp_path / "fetched.pt"
    monkeypatch.setattr(elastix_engine_module, "hf_hub_download", lambda repo_id, filename, repo_type: str(fetched))
    engine = SimpleNamespace(_models=[str(local), "org/repo:MIND/R1D2.pt"])

    staged = ElastixEngine._download_models(engine)

    assert staged == [(str(local), local.resolve()), ("MIND/R1D2.pt", fetched)]


def test_elastix_engine_refuses_an_empty_parameter_map_list() -> None:
    # 'resolutions' rewrites a template's resolution-dependent lines; it never creates one. Without a
    # map elastix would launch with no -p and die in a cryptic subprocess error.
    with pytest.raises(ValueError, match="parameter-map template"):
        ElastixEngine(parameter_maps=[])


def test_fireants_impact_metric_requires_a_feature_model() -> None:
    # With no feature model the IMPACT loss returns a None total deep in the deformable stage, after
    # the rigid/affine stages already burned minutes; the net must refuse at build time instead.
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="requires at least one feature model"):
        RegistrationNet(deformable_metric="impact", models={})
