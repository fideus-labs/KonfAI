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
from typing import Optional

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from impact_reg_konfai.models import elastix_engine as elastix_engine_module
from impact_reg_konfai.models.elastix_engine import ElastixEngine
from konfai.metric.measure.impact import _statistics
from konfai.utils.dataset import Attribute
from konfai.utils.errors import MeasureError


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


def _elastix_run(monkeypatch, lines: list[str], code: int, parameter_map: str = "", fixed: sitk.Image | None = None):
    """``ElastixEngine.register`` over a subprocess that prints ``lines`` and exits with ``code``, the run staging
    ``parameter_map`` (if any) as its map."""

    class Process:
        stdout = iter(lines)

        def wait(self) -> int:
            return code

    def stage(work: Path, device_index: int, native_voxel_size: tuple, voxels: int | None = None) -> list[Path]:
        if not parameter_map:
            return []
        (work / "map.txt").write_text(parameter_map)
        return [work / "map.txt"]

    monkeypatch.setattr(elastix_engine_module.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(elastix_engine_module, "loader_env", lambda root: {})
    engine = SimpleNamespace(
        _feature_models={},
        _elastix_bin="elastix",
        _elastix_root=Path("/opt/elastix-impact"),
        _stage_parameter_maps=stage,
        _max_iterations=0,
        _iterations=None,
    )
    fixed = fixed if fixed is not None else sitk.Image([6, 5, 4], sitk.sitkFloat32)
    return ElastixEngine.register(engine, fixed, sitk.Image(fixed), 0)


@pytest.mark.parametrize("impact", [False, True])
def test_an_intensity_run_hands_elastix_its_images_winsorised(monkeypatch, impact: bool) -> None:
    # Mutual information bins each image between its extremes: lone hot voxels put the tissue of two ExaSPIM brains in
    # the first of 32 bins, and the rigid stage aligned noise. An IMPACT run keeps the intensities its models expect.
    written = {}
    monkeypatch.setattr(
        elastix_engine_module.sitk,
        "WriteImage",
        lambda image, path, *args: written.__setitem__(Path(path).name, sitk.GetArrayFromImage(image)),
    )

    class Process:
        stdout = iter([])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(elastix_engine_module.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(elastix_engine_module, "loader_env", lambda root: {})
    engine = SimpleNamespace(
        _feature_models={("TS/M730.pt", "1"): object()} if impact else {},
        _unchecked=False,
        _elastix_bin="elastix",
        _elastix_root=Path("/opt/elastix-impact"),
        _stage_parameter_maps=lambda work, device_index, native_voxel_size, voxels=None: [],
        _iterations=None,
    )
    tissue = np.zeros((30, 30, 30), np.uint16)
    tissue[5:25, 5:25, 5:25] = 30
    tissue[15, 15, 15] = 21668
    fixed = sitk.GetImageFromArray(tissue)

    with pytest.raises(FileNotFoundError, match="no composite transform"):
        ElastixEngine.register(engine, fixed, sitk.Image(fixed), 0)

    assert written["Fixed.mha"].max() == (21668 if impact else 30)
    assert written["Moving.mha"].max() == (21668 if impact else 30)
    assert written["Fixed.mha"].dtype == np.uint16


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

    # A plugin-based build passes -h without loading IMPACT, whose LibTorch may not be the one it was built against.
    plugin = "ERROR: IMPACT requested but could not be loaded: libImpactMetric.so: undefined symbol: _ZN3c10\n"
    with pytest.raises(RuntimeError, match="KONFAI_ELASTIX_EXTRA_LIB"):
        _elastix_run(monkeypatch, [plugin], 1)


def test_a_mask_too_thin_for_the_random_sampler_says_what_to_do(monkeypatch) -> None:
    # The sampler draws in the mask's bounding box and gives up after ten times the samples it needs.
    thin = "Description: itk::ERROR: ImageRandomCoordinateSampler: Could not find enough image samples within 10 x\n"
    with pytest.raises(RuntimeError, match="RandomSparseMask"):
        _elastix_run(monkeypatch, [thin], 1)


def test_an_image_too_small_for_the_impact_grid_says_so(monkeypatch) -> None:
    # An ExaSPIM brain at 160 um (53 x 40 x 21 mm) is a few voxels on the 6 mm grid of the IMPACT presets' coarsest
    # level: elastix said only that no sample mapped inside the moving image, or that a model rejected its input.
    brain = sitk.Image([443, 332, 129], sitk.sitkFloat32)
    brain.SetSpacing((0.12032, 0.12032, 0.16))
    impact_map = (
        "(ImpactPatchSize0 11 11 11)\n(ImpactVoxelSize0 6 6 6)\n(ImpactPatchSize1 11 11 11)\n(ImpactVoxelSize1 3 3 3)\n"
    )
    no_sample = (
        "Description: ITK ERROR: ImpactMetric(0x5f): Too many samples map outside moving image buffer: 0 / 2000\n"
    )
    with pytest.raises(RuntimeError) as raised:
        _elastix_run(monkeypatch, [no_sample], 1, impact_map, brain)
    assert "8 x 6 x 3 voxels at 6 x 6 x 6 mm (level 0), where its 11-voxel patch spans 66 mm" in str(raised.value)

    # Static mode: whole images (patch 0), one voxel size per model on a line.
    static_map = "(ImpactPatchSize0 0 0 0 0 0 0)\n(ImpactVoxelSize0 6 6 6 6 6 6)\n"
    rejected = "Description: ITK ERROR: IMPACT: the model TS/M730.pt rejected its input. Check the number of channels\n"
    with pytest.raises(RuntimeError, match="FireANTs_SyN") as raised:
        _elastix_run(monkeypatch, [rejected], 1, static_map, brain)
    assert "patch spans" not in str(raised.value)
    # A TotalSegmentator encoder's normalisation, handed a slab one voxel thick at 6 mm (an APEX brain section).
    slab = "builtins.ValueError: Expected more than 1 spatial element when training, got input size [1, 320, 1, 1, 1]\n"
    with pytest.raises(RuntimeError, match="too small for the grid"):
        _elastix_run(monkeypatch, [slab], 1, static_map, brain)

    # A CT the grid suits failed for another reason: nothing to say about its size.
    ct = sitk.Image([300, 300, 200], sitk.sitkFloat32)
    with pytest.raises(RuntimeError) as other:
        _elastix_run(monkeypatch, [no_sample], 1, impact_map, ct)
    assert "too small" not in str(other.value)


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


def test_fireants_deformable_masked_reaches_the_engine() -> None:
    # Dropped at RegistrationNet, the deformable stage would stay masked: a silent no-op.
    from impact_reg_konfai.models.fireants import RegistrationNet

    for value in (True, False):
        assert RegistrationNet(deformable_masked=value)["Registration"]._engine._deformable_masked is value


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

    cpu, _ = ElastixEngine._apply_map_overrides(text, {}, [("ImpactUseMixedPrecision", '"true"')], -1)

    assert cpu.count("ImpactUseMixedPrecision") == 1
    assert '(ImpactUseMixedPrecision "false")' in cpu


def test_mixed_precision_is_off_on_the_cpu_and_kept_on_a_gpu() -> None:
    # Every shipped IMPACT preset turns half precision on; on the CPU the feature model's pooling has no
    # half-precision kernel, so a run placed there with --cpu died in the first layer.
    text = '(ImpactGPU 0)\n(ImpactUseMixedPrecision "true" "true")\n(Metric "Impact")'

    cpu, _ = ElastixEngine._apply_map_overrides(text, {}, [], -1)
    gpu, _ = ElastixEngine._apply_map_overrides(text, {}, [], 1)

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
        # cuBLAS and cuDNN report a failed device allocation in their own words, as itk-impact reads them too.
        (
            True,
            "elastix failed (code 1):\nCUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
            True,
        ),
        (True, "cuDNN error: CUDNN_STATUS_ALLOC_FAILED", True),
    ],
)
def test_only_an_engine_s_cuda_out_of_memory_becomes_torch_s_class(
    on_cuda: bool, message: str, translated: bool
) -> None:
    # An engine that allocates outside PyTorch (libtorch in itk-impact, the elastix subprocess) reports a
    # plain RuntimeError, and konfai shrinks a free patch axis on torch.cuda.OutOfMemoryError only. A CPU
    # run, a host allocation or any other failure keeps its class: none is a reason to cut a patch.
    from konfai.utils.vram import out_of_memory_as_torch

    with pytest.raises(RuntimeError) as raised, out_of_memory_as_torch(on_cuda):
        raise RuntimeError(message)
    assert isinstance(raised.value, torch.cuda.OutOfMemoryError) is translated


_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the translation is for a CUDA run")


@_needs_cuda
def test_an_out_of_memory_inside_the_engine_reaches_konfai_as_torch_s_class() -> None:
    pytest.importorskip("itk")
    from impact_reg_konfai.models.intensity import EngineRegistration

    class Engine:
        def register(self, fixed, moving, device, fixed_mask, moving_mask):
            raise RuntimeError("CUDA out of memory. Tried to allocate 224.00 MiB.")

    image, attributes = _registration_inputs("cuda")
    with pytest.raises(torch.cuda.OutOfMemoryError, match="out of memory"):
        EngineRegistration(Engine(), fuse_texpr=False)(image, image, image, image, attributes)


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


def test_fireants_takes_a_models_input_multiple_from_the_registry(monkeypatch, tmp_path: Path) -> None:
    # The size an encoder-decoder's input must divide by is the model's own, read off the registry.
    from impact_reg_konfai.models import fireants
    from konfai.metric.measure import impact

    registry = {"m.pt": {"dimension": "3", "numberofchannels": "1", "fov": [3], "multiple": 16}}
    monkeypatch.setattr(impact, "models_registry", lambda: registry)
    monkeypatch.setattr(impact, "fetch_model", lambda ref: Path(_local_model(tmp_path)))
    loss = fireants.ImpactFeatureLoss([[fireants.ModelSpec(ref="org/repo:m.pt")]], "Static", False, 5, 0, 0, False)
    assert loss.cores[0].model.multiple == 16


def test_fireants_impact_settings_reach_the_engine() -> None:
    # Dropped at RegistrationNet, each of these would silently keep its default: the run still produces a
    # field, computed with settings the caller did not ask for.
    from impact_reg_konfai.models.fireants import RegistrationNet

    engine = RegistrationNet(mode="Static", feature_overlap=0.5, normalize=False, lncc_kernel=7, mixed_precision=True)[
        "Registration"
    ]._engine
    assert (engine._feature_overlap, engine._normalize, engine._lncc_kernel, engine._mixed_precision) == (
        0.5,
        False,
        7,
        True,
    )


def test_fireants_refuses_what_its_loss_cannot_honour(tmp_path: Path) -> None:
    # Jacobian mode extracts the features at every step: a refresh interval means nothing there and is refused, as is
    # an even LNCC window.
    import pytest
    from impact_reg_konfai.models.fireants import ModelSpec, RegistrationNet

    impact = {"deformable_metric": "impact", "models": {"0": ModelSpec(ref=str(tmp_path / "m.pt"))}}
    with pytest.raises(ValueError, match="feature_map_update_interval"):
        RegistrationNet(**impact, mode="Jacobian", feature_map_update_interval=10)
    with pytest.raises(ValueError, match="lncc_kernel"):
        RegistrationNet(**impact, lncc_kernel=4)
    with pytest.raises(ValueError, match="one per level"):
        RegistrationNet(**impact, levels={"0": {"models": {}}})


def test_fireants_refuses_an_even_correlation_window() -> None:
    # FireANTs' own cross-correlation raises on an even window; the chunked one would instead pool a
    # voxel wider than the image, which crashes on a masked pair and shifts the correlation by half a
    # voxel on an unmasked one. Both paths have to refuse it, and before the run starts.
    import pytest
    from impact_reg_konfai.models.fireants import RegistrationNet

    with pytest.raises(ValueError, match="cc_kernel"):
        RegistrationNet(cc_kernel=4)


def test_fireants_jacobian_mode_refuses_a_segmentation_head(tmp_path: Path) -> None:
    # Jacobian mode differentiates the metric through the network, and a one-hot head carries no gradient: selected
    # alone, the deformable stage did not move and the run ended as if it had registered.
    import torch
    from impact_reg_konfai.models.fireants import ImpactFeatureLoss, ModelSpec

    class WithHead(torch.nn.Module):
        def forward(self, x: torch.Tensor, nb_layers: torch.Tensor) -> list[torch.Tensor]:
            return [x.repeat(1, 2, 1, 1, 1), (x > 0).long()]

    path = tmp_path / "with_head.pt"
    torch.jit.script(WithHead()).save(str(path))

    def loss(layers_mask: str, mode: str) -> ImpactFeatureLoss:
        return ImpactFeatureLoss([[ModelSpec(ref=str(path), layers_mask=layers_mask)]], mode, True, 5, 0, 0, False)

    with pytest.raises(MeasureError, match="layer 2 carries no gradient"):
        loss("01", "Jacobian")
    loss("10", "Jacobian")
    loss("01", "Static")  # Static mode registers the maps themselves


@pytest.mark.parametrize(
    ("module_name", "class_name"), [("fireants", "FireANTsEngine"), ("convexadam", "ConvexAdamEngine")]
)
def test_a_tile_the_fixed_mask_does_not_reach_gets_a_zero_field(module_name: str, class_name: str) -> None:
    """As the elastix engine: an empty fixed mask leaves nothing to register, and nothing runs."""
    if module_name == "convexadam":
        pytest.importorskip("itk")
    import importlib

    engine_class = getattr(importlib.import_module(f"impact_reg_konfai.models.{module_name}"), class_name)
    engine = engine_class.__new__(engine_class)  # unconfigured: any step past the check would raise
    engine._linear_method = "none"  # as a tile: the tile pass runs no linear stage
    image = sitk.GetImageFromArray(np.random.rand(4, 5, 6).astype(np.float32))
    empty = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8))

    field = engine.register(image, image, -1, fixed_mask=empty)

    assert field.shape == (3, 4, 5, 6) and not field.any()


def test_elastix_stages_the_pair_under_tmpdir(monkeypatch, tmp_path: Path) -> None:
    # impact-reg-konfai points it at its work dir: not the system temp, RAM itself where /tmp is a tmpfs.
    monkeypatch.setattr(elastix_engine_module.tempfile, "tempdir", str(tmp_path))
    staged = []
    monkeypatch.setattr(elastix_engine_module.sitk, "WriteImage", lambda image, path: staged.append(Path(path)))
    with pytest.raises(FileNotFoundError):
        _elastix_run(monkeypatch, [], 0)
    assert staged and all(tmp_path in path.parents for path in staged)


class _LocalFeatures(torch.nn.Module):
    """A feature model with a 3-voxel receptive field and two channels, taking the inputs the IMPACT metric gives."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layers: torch.Tensor,
        stats: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
        direction: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
    ) -> list[torch.Tensor]:
        mean = torch.nn.functional.avg_pool3d(x, 3, 1, 1, count_include_pad=False)
        return [torch.cat([mean, torch.nn.functional.avg_pool3d(x * x, 3, 1, 1, count_include_pad=False)], dim=1)]


def _local_model(tmp_path: Path) -> str:
    path = tmp_path / "local.pt"
    torch.jit.script(_LocalFeatures()).save(str(path))
    return str(path)


def test_fireants_sampled_jacobian_compares_the_patch_centres(tmp_path: Path) -> None:
    # elastix's Jacobian scheme: the network runs on the patch of its receptive field around each drawn point and only
    # the centre voxel is compared. For a local model that is the whole-image feature at the point, whatever the
    # batches, and the gradient reaches the image only inside the patches.
    from impact_reg_konfai.models.fireants import ModelSpec, _ImpactCore

    core = _ImpactCore(ModelSpec(ref=_local_model(tmp_path)), False)
    moved = torch.rand(1, 1, 20, 20, 20, requires_grad=True)
    fixed = torch.rand(1, 1, 20, 20, 20)
    centres = torch.tensor([[5, 6, 7], [10, 10, 10], [14, 3, 12]])
    value = core.sampled_distances(moved, fixed, centres, 5, [("L2", 0)], 0, 5)
    network = core.model.network(torch.device("cpu"))
    whole = [network(*core.model.inputs(image, _statistics(image)[0]))[0][0] for image in (moved, fixed)]
    at = (slice(None), centres[:, 0], centres[:, 1], centres[:, 2])
    assert torch.allclose(value[0], (whole[0][at] - whole[1][at]).pow(2).mean())
    core.model.batch = 2  # two batches, the same points
    assert torch.allclose(core.sampled_distances(moved, fixed, centres, 5, [("L2", 0)], 0, 5), value)
    value.sum().backward()
    touched = torch.nonzero(moved.grad[0, 0])
    assert len(touched) and all(bool(((point - centres).abs().max(1).values <= 2).any()) for point in touched)


def test_fireants_sampled_static_reads_a_share_of_the_voxels(tmp_path: Path) -> None:
    # Static compares the warped feature volumes at the drawn voxels only: the value of a uniform difference is kept,
    # and the gradient reaches no more voxels than were drawn.
    from impact_reg_konfai.models.fireants import ImpactFeatureLoss, ModelSpec

    loss = ImpactFeatureLoss(
        [[ModelSpec(ref=_local_model(tmp_path))]], "Static", False, 5, 0, 0, False, voxel_sampling=0.01
    )
    loss._channels = [[2]]
    moved = torch.rand(1, 2, 20, 20, 20, requires_grad=True)
    value = loss(moved, moved.detach() + 1.0)
    assert torch.allclose(value, torch.tensor(1.0))
    value.backward()
    assert 0 < int((moved.grad.abs().sum(1) > 0).sum()) <= round(0.01 * 20**3)


def test_fireants_voxel_sampling_reaches_the_engine_and_is_refused_where_meaningless(tmp_path: Path) -> None:
    # Dropped at RegistrationNet, the loss would read every voxel. A share out of (0, 1], an LNCC on drawn points and
    # a Jacobian patch that cannot be sized (a local model has no registry FOV) are refused before the run.
    from impact_reg_konfai.models.fireants import ImpactFeatureLoss, ModelSpec, RegistrationNet

    assert RegistrationNet(voxel_sampling=0.25)["Registration"]._engine._voxel_sampling == 0.25
    impact = {"deformable_metric": "impact", "models": {"0": ModelSpec(ref="m.pt", distance="LNCC")}}
    with pytest.raises(ValueError, match="LNCC"):
        RegistrationNet(**impact, voxel_sampling=0.5)
    with pytest.raises(ValueError, match="share of the voxels"):
        RegistrationNet(voxel_sampling=1.5)
    with pytest.raises(ValueError, match="cannot be sized"):
        ImpactFeatureLoss(
            [[ModelSpec(ref=_local_model(tmp_path))]], "Jacobian", True, 5, 0, 0, False, voxel_sampling=0.1
        )
