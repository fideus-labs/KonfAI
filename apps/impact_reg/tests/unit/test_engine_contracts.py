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

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from impact_reg_konfai.models import elastix_engine as elastix_engine_module
from impact_reg_konfai.models.elastix_engine import ElastixEngine
from konfai.utils.dataset import Attribute


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


def test_download_models_refuses_a_missing_local_file_at_build(tmp_path: Path) -> None:
    # A typo'd local path must fail HERE: staged later it would plant a dangling symlink at the
    # user-supplied location and crash the SECOND case with an unrelated FileExistsError.
    engine = SimpleNamespace(_models=[str(tmp_path / "typo.pt")])
    with pytest.raises(ValueError, match="does not exist"):
        ElastixEngine._download_models(engine)


def test_a_windows_drive_letter_is_a_local_path_not_an_hf_repo() -> None:
    # 'C:/models/m.pt' contains ':' but is a path: splitting it as repo 'C' would send a Windows
    # user's local ref to Hugging Face. The same rule keys the registry/staged name.
    from impact_reg_konfai.models.elastix import _is_local_ref, _model_key

    assert _is_local_ref("C:/models/m.pt") and _is_local_ref(r"D:\models\m.pt")
    assert _is_local_ref("/abs/model.pt") and _is_local_ref("relative/model.pt")
    assert not _is_local_ref("org/repo:MIND/R1D2.pt")
    assert _model_key("C:/models/m.pt") == "C:/models/m.pt"
    assert _model_key("org/repo:MIND/R1D2.pt") == "MIND/R1D2.pt"


def test_an_empty_fixed_mask_is_a_zero_field_and_never_reaches_elastix(monkeypatch) -> None:
    # A patch the fixed mask does not reach has nothing to register. Read as 'no mask', it had elastix
    # fit the whole patch, background included, and a tiled run dragged the tissue edge by millimetres.
    def elastix_must_not_run(*args, **kwargs):
        raise AssertionError("elastix was launched on an empty fixed mask")

    monkeypatch.setattr(elastix_engine_module.subprocess, "Popen", elastix_must_not_run)
    fixed = sitk.Image([6, 5, 4], sitk.sitkFloat32)
    fixed.SetSpacing((0.5, 0.5, 2.0))
    fixed.SetOrigin((3.0, -1.0, 7.0))
    empty = sitk.Image([6, 5, 4], sitk.sitkUInt8)

    field = ElastixEngine.register(SimpleNamespace(), fixed, sitk.Image(fixed), 0, fixed_mask=empty)

    assert field.shape == (3, 4, 5, 6)
    assert not field.any()


def _elastix_run(monkeypatch, lines: list[str], code: int):
    """``ElastixEngine.register`` over a subprocess that prints ``lines`` and exits with ``code``."""

    class Process:
        stdout = iter(lines)

        def wait(self) -> int:
            return code

    monkeypatch.setattr(elastix_engine_module.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(elastix_engine_module, "loader_env", lambda root: {})
    engine = SimpleNamespace(
        _local_models=[],
        _elastix_bin="elastix",
        _elastix_root=Path("/opt/elastix-impact"),
        _stage_parameter_maps=lambda work, device_index: [],
        _max_iterations=0,
        _iterations=None,
    )
    fixed = sitk.Image([6, 5, 4], sitk.sitkFloat32)
    return ElastixEngine.register(engine, fixed, sitk.Image(fixed), 0)


def test_what_impact_says_about_device_memory_is_shown_once(monkeypatch, capsys) -> None:
    # IMPACT goes on with smaller patches when the device runs out of memory: the run is slower for it, and
    # elastix's output is otherwise shown on a failure only.
    retry = "IMPACT: the model ran out of device memory on the whole image; retrying with a patch of (96 160 192).\n"
    with pytest.raises(FileNotFoundError, match="no composite transform"):
        _elastix_run(monkeypatch, ["Resolution: 0\n", retry, retry, "1 -0.25 3.0\n"], 0)

    assert capsys.readouterr().out.count("retrying with a patch of (96 160 192)") == 1


def test_a_gpu_the_install_cannot_see_says_what_to_do(monkeypatch) -> None:
    # A CPU build answers `-h` and passes for a valid install: the failure only comes mid-registration.
    unseen = "Description: ITK ERROR: ImpactMetric(0x5e): CUDA is not available. Please check your CUDA installation.\n"
    with pytest.raises(RuntimeError, match="KONFAI_ELASTIX_DIR") as raised:
        _elastix_run(monkeypatch, [unseen], 1)
    assert str(Path("/opt/elastix-impact")) in str(raised.value)

    with pytest.raises(RuntimeError) as other:
        _elastix_run(monkeypatch, ["no such parameter file\n"], 1)
    assert "KONFAI_ELASTIX_DIR" not in str(other.value)


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


def test_fireants_linear_method_reaches_the_engine() -> None:
    # The knob is only useful if it survives the config binding: a value that stops at RegistrationNet
    # leaves the engine on its default, and a tiled pass would run the per-patch linear it asked to
    # skip: silently, because the run still produces a plausible field.
    from impact_reg_konfai.models.fireants import RegistrationNet

    for method in ("rigid_affine", "rigid", "none"):
        net = RegistrationNet(linear_method=method)
        assert net["Registration"]._engine._linear_method == method


def test_fireants_moments_init_reaches_the_engine() -> None:
    # The seed decides where the rigid starts, and a value that stops at RegistrationNet leaves the
    # engine on 'cof', which aligns frames: a pair the caller centred is then pulled apart by the
    # difference between the two subjects' offsets from their own frames.
    from impact_reg_konfai.models.fireants import RegistrationNet

    for seed in ("cof", "com", "none"):
        net = RegistrationNet(moments_init=seed)
        assert net["Registration"]._engine._moments_init == seed


def _plain_inputs(tensor, attribute):
    """What ``ImpactFeatureModel.inputs`` returns, for a stub model that has no registry entry."""
    import torch

    return [tensor, torch.tensor([1]), torch.tensor([0.0, 1.0, 0.5, 0.2])]


def test_fireants_refuses_an_unknown_linear_method() -> None:
    # Every unrecognised value would otherwise fall through to the rigid-then-affine branch, so a
    # typo registers with a stage the caller did not ask for and returns a plausible result. The
    # Literal annotation only guards a config-driven call; a direct Python one reaches the engine.
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="Unknown linear_method 'affine'"):
        RegistrationNet(linear_method="affine")


def test_fireants_refuses_a_registration_with_no_stage_at_all() -> None:
    # linear_method='none' and deformable_method='none' together optimise nothing. Left to run it
    # would return the identity: a Moved equal to the moving image and a zero field, which no
    # downstream check tells apart from a pair that needed no moving. Refused at build time, like the
    # missing-feature-model case, rather than after the stages have burned minutes.
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="leaves nothing to optimise"):
        RegistrationNet(linear_method="none", deformable_method="none")


def test_an_exact_mixed_precision_override_stays_off_on_the_cpu() -> None:
    # An exact override of the key used to be appended behind the forced line, so the map carried a
    # second, "true" entry and elastix ran the half precision the CPU cannot.
    text = '(ImpactGPU 0)\n(ImpactUseMixedPrecision "true" "true")\n(Metric "Impact")'

    cpu = ElastixEngine._apply_map_overrides(text, {}, [("ImpactUseMixedPrecision", '"true"')], -1)

    assert cpu.count("ImpactUseMixedPrecision") == 1
    assert '(ImpactUseMixedPrecision "false")' in cpu


def test_mixed_precision_is_off_on_the_cpu_and_kept_on_a_gpu() -> None:
    # Every shipped IMPACT preset turns half precision on; on the CPU the feature model's pooling has no
    # half-precision kernel, so a run placed there with --cpu died in the first layer.
    text = '(ImpactGPU 0)\n(ImpactUseMixedPrecision "true" "true")\n(Metric "Impact")'

    cpu = ElastixEngine._apply_map_overrides(text, {}, [], -1)
    gpu = ElastixEngine._apply_map_overrides(text, {}, [], 1)

    assert '(ImpactUseMixedPrecision "false")' in cpu and "(ImpactGPU -1)" in cpu
    assert '(ImpactUseMixedPrecision "true" "true")' in gpu and "(ImpactGPU 1)" in gpu


def _registration_inputs(device: str = "cpu") -> tuple[torch.Tensor, list[list[Attribute]]]:
    geometry = Attribute()
    geometry["Origin"] = np.zeros(3)
    geometry["Spacing"] = np.ones(3)
    geometry["Direction"] = np.eye(3).flatten()
    return torch.zeros(1, 1, 4, 4, 4, device=device), [[geometry] for _ in range(4)]


@pytest.mark.parametrize(
    ("on_cuda", "message", "translated"),
    [
        (True, "CUDA out of memory. Tried to allocate 224.00 MiB.", True),
        (True, "elastix failed (code 1):\nDescription: ITK ERROR\nCUDA error: out of memory\n", True),
        (False, "CUDA out of memory. Tried to allocate 224.00 MiB.", False),
        (True, "elastix failed (code 1):\nno such parameter file", False),
        (True, "std::bad_alloc: out of memory", False),
    ],
)
def test_only_an_engine_s_cuda_out_of_memory_becomes_torch_s_class(
    on_cuda: bool, message: str, translated: bool
) -> None:
    # An engine that allocates outside PyTorch (libtorch in itk-impact, the elastix subprocess) reports a
    # plain RuntimeError, and konfai shrinks a free patch axis on torch.cuda.OutOfMemoryError only. A CPU
    # run, a host allocation or any other failure keeps its class: none is a reason to cut a patch.
    from impact_reg_konfai.models.engine_errors import out_of_memory_as_torch

    with pytest.raises(RuntimeError) as raised, out_of_memory_as_torch(on_cuda):
        raise RuntimeError(message)
    assert isinstance(raised.value, torch.cuda.OutOfMemoryError) is translated


_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the translation is for a CUDA run")


@_needs_cuda
def test_an_out_of_memory_inside_the_engine_reaches_konfai_as_torch_s_class() -> None:
    pytest.importorskip("itk")
    from impact_reg_konfai.models.convexadam import ConvexAdamRegistration

    class Engine:
        def register(self, fixed, moving, device):
            raise RuntimeError("CUDA out of memory. Tried to allocate 224.00 MiB.")

    image, attributes = _registration_inputs("cuda")
    with pytest.raises(torch.cuda.OutOfMemoryError, match="out of memory"):
        ConvexAdamRegistration(Engine())(image, image, image, image, attributes)


@_needs_cuda
def test_an_out_of_memory_in_the_elastix_subprocess_reaches_konfai_as_torch_s_class() -> None:
    from impact_reg_konfai.models.elastix_engine import ElastixRegistration

    class Engine:
        def register(self, *pair):
            raise RuntimeError(
                "elastix failed (code 1):\nDescription: ITK ERROR\nCUDA out of memory. Tried to allocate 2.00 GiB.\n"
            )

    image, attributes = _registration_inputs("cuda")
    with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory"):
        ElastixRegistration.forward(SimpleNamespace(_engine=Engine()), image, image, image, image, attributes)


def test_fireants_static_mode_reaches_the_engine() -> None:
    # Dropped at RegistrationNet, the deformable stage would keep extracting inside the loss: the run
    # still succeeds, on a card it may not fit, so nothing points at the setting having been ignored.
    from impact_reg_konfai.models.fireants import RegistrationNet

    for mode, patch in (("Jacobian", 0), ("Static", 128)):
        engine = RegistrationNet(mode=mode, feature_patch=patch)["Registration"]._engine
        assert (engine._mode, engine._feature_patch) == (mode, patch)


def test_fireants_refuses_an_unknown_mode() -> None:
    # An unrecognised value would otherwise fall through to Jacobian, which is the mode that does not fit
    # the volume the caller asked Static for. The elastix engine spells these two the same way.
    import pytest
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="mode"):
        RegistrationNet(mode="static")


def test_fireants_tiled_extraction_leaves_no_seam() -> None:
    # The tiles are blended by KonfAI's own cosine window, which sums to one over the overlap: features
    # constant over the image must come back constant, or a seam shows as a band of half-weight voxels.
    import torch
    from impact_reg_konfai.models.fireants import _one_volume

    class Constant(torch.nn.Module):
        def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
            return [torch.full((tile.shape[0], 2, *tile.shape[2:]), 3.0)]

    model = SimpleNamespace(model=Constant(), weights=[1.0], in_channels=1, model_path="", inputs=_plain_inputs, dim=3)
    image = torch.rand(1, 1, 40, 24, 70)
    volume = _one_volume(
        SimpleNamespace(model=model, _stats=lambda t: {}), 1.0, image, patch=32, overlap=0.25, normalization="none"
    )
    assert volume.shape == (1, 2, 40, 24, 70)
    assert torch.allclose(volume, torch.full_like(volume, 3.0), atol=1e-5)


def test_fireants_rounds_the_extraction_up_for_a_model_that_needs_it() -> None:
    # anatomix halves its input four times and its skip connections only meet on a multiple of 16: handed
    # a 40-voxel image it raises inside the network. The padding must not reach the result.
    import pytest
    import torch
    from impact_reg_konfai.models.fireants import RegistrationNet, _one_volume

    class NeedsSixteen(torch.nn.Module):
        def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
            if any(size % 16 for size in tile.shape[2:]):
                raise RuntimeError(f"sizes of tensors must match: {tuple(tile.shape[2:])}")
            return [tile.repeat(1, 4, 1, 1, 1)]

    model = SimpleNamespace(
        model=NeedsSixteen(), weights=[1.0], in_channels=1, model_path="", inputs=_plain_inputs, dim=3
    )
    core = SimpleNamespace(model=model, _stats=lambda t: {})
    image = torch.rand(1, 1, 40, 40, 40)
    with pytest.raises(RuntimeError, match="sizes of tensors"):
        _one_volume(core, 1.0, image, patch=0, overlap=0.25, normalization="none")
    volume = _one_volume(core, 1.0, image, patch=0, overlap=0.25, normalization="none", multiple=16)
    assert volume.shape == (1, 4, 40, 40, 40)
    assert RegistrationNet(feature_multiple=16)["Registration"]._engine._feature_multiple == 16


def test_fireants_static_settings_reach_the_engine() -> None:
    # Dropped at RegistrationNet, each of these would silently keep its default: the run still produces a
    # field, computed with settings the caller did not ask for.
    from impact_reg_konfai.models.fireants import RegistrationNet

    engine = RegistrationNet(
        mode="Static", feature_overlap=0.5, feature_normalization="standardized", feature_metric="mi"
    )["Registration"]._engine
    assert (engine._feature_overlap, engine._feature_normalization, engine._feature_metric) == (
        0.5,
        "standardized",
        "mi",
    )


def test_fireants_refuses_unknown_static_settings() -> None:
    import pytest
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="feature_normalization"):
        RegistrationNet(feature_normalization="zscore")
    with pytest.raises(ValueError, match="feature_metric"):
        RegistrationNet(feature_metric="ncc")


def test_fireants_refuses_an_even_correlation_window() -> None:
    # FireANTs' own cross-correlation raises on an even window; the chunked one would instead pool a
    # voxel wider than the image, which crashes on a masked pair and shifts the correlation by half a
    # voxel on an unmasked one. Both paths have to refuse it, and before the run starts.
    import pytest
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="cc_kernel"):
        RegistrationNet(cc_kernel=4)


def test_fireants_static_puts_every_feature_layer_on_the_image_grid() -> None:
    # A segmentation network hands back coarser deeper layers (M730: 64/32/16 voxels for a 64-voxel
    # tile). Concatenated as they come, the second layer would fail torch.cat outright.
    import torch
    from impact_reg_konfai.models.fireants import _one_volume

    class TwoResolutions(torch.nn.Module):
        def forward(self, tile: torch.Tensor, nb_layers: torch.Tensor, stats: torch.Tensor) -> list[torch.Tensor]:
            coarse = torch.nn.functional.avg_pool3d(tile, 2)
            return [tile.repeat(1, 3, 1, 1, 1), coarse.repeat(1, 5, 1, 1, 1)]

    model = SimpleNamespace(
        model=TwoResolutions(), weights=[1.0, 1.0], in_channels=1, model_path="", inputs=_plain_inputs, dim=3
    )
    core = SimpleNamespace(model=model, _stats=lambda t: {})
    volume = _one_volume(core, 1.0, torch.rand(1, 1, 16, 16, 16), patch=0, overlap=0.25, normalization="none")
    assert volume.shape == (1, 8, 16, 16, 16)


def test_the_jacobian_patch_covers_the_deepest_selected_layer() -> None:
    # A patch smaller than the receptive field crops the context the feature was trained to see. Measured
    # on TS/M730, one output voxel's sensitivity to the input stays above 1 % of its peak over 5 voxels for
    # the first layer and 11 for the second -- the study's own recommended map sets 11 for this mask.
    from impact_reg_konfai.models.elastix import _fov_value

    fov = {"formula": "2^l+3"}
    assert _fov_value(fov, "1") == 5
    assert _fov_value(fov, "01") == 11
    assert _fov_value(fov, "001") == 23
    assert _fov_value({"formula": "2*r*d+1", "r": 1, "d": 2}, "1") == 5
