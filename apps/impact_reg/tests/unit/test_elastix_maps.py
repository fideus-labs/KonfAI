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

"""The parameter maps the elastix engine hands to elastix, as elastix reads them."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai.models import elastix_engine
from impact_reg_konfai.models.elastix import ElastixLevelSpec, RegistrationNet, generate_impact_parameter_map
from impact_reg_konfai.models.elastix_engine import ElastixEngine
from impact_reg_konfai.models.impact_loss import ModelSpec
from konfai.metric.measure import ImpactFeatureModel

RIGID = """(Transform "EulerTransform")
(Metric "AdvancedMattesMutualInformation")
(NumberOfResolutions 4)
(MaximumNumberOfIterations 250)
(FixedImagePyramid "FixedRecursiveImagePyramid")
"""

IMPACT = """(MaximumNumberOfIterations 400 300)
(NumberOfResolutions 2)
(FixedImagePyramidRescaleSchedule 1 1 1 1 1 1)
(MovingImagePyramidRescaleSchedule 1 1 1 1 1 1)
(ImpactModelsPath0 "TS/M852.pt")
(ImpactDimension0 3)
(ImpactNumberOfChannels0 1)
(ImpactPatchSize0 0 0 0)
(ImpactVoxelSize0 6 6 6)
(ImpactLayersMask0 "1")
(ImpactSubsetFeatures0 64)
(ImpactPCA0 0)
(ImpactDistance0 "L1")
(ImpactLayersWeight0 1)
(ImpactMode "Static")
(ImpactGPU 0)
(Metric "Impact" "AdvancedMattesMutualInformation")
"""

# As models.json on VBoussot/impact-torchscript-models spells them.
REGISTRY = {
    "MIND/R1D2_3D.pt": {"dimension": "3", "numberofchannels": "1", "fov": [5]},
    "TS/M730.pt": {"dimension": "3", "numberofchannels": "1", "fov": [5, 11, 23, 47, 95, 191, 191, 191]},
}


def _levels(*levels: tuple[int, list[ModelSpec]]) -> dict[str, ElastixLevelSpec]:
    return {
        str(k): ElastixLevelSpec(max_iterations=iterations, models={str(m): spec for m, spec in enumerate(models)})
        for k, (iterations, models) in enumerate(levels)
    }


def _feature_model(spec: ModelSpec) -> ImpactFeatureModel:
    """KonfAI's feature model of ``spec`` as the registry above shapes it, named by its key, without a download."""
    key = spec.ref.split(":", 1)[-1]
    entry = REGISTRY[key]
    model = ImpactFeatureModel(
        key, int(entry["numberofchannels"]), [float(b == "1") for b in spec.layers_mask], None, 3
    )
    model.dim, model.fov = int(entry["dimension"]), entry["fov"]
    return model


def _generate(template: str, levels: dict | None = None, models: list[ModelSpec] = (), **settings) -> str:
    """The map generated for a fixed image at 0.8 mm."""
    specs = [*models, *(m for level in (levels or {}).values() for m in level.models.values())]
    return generate_impact_parameter_map(
        template,
        {str(m): spec for m, spec in enumerate(models)},
        levels or {},
        {(m.ref, m.layers_mask): _feature_model(m) for m in specs},
        (0.8, 0.8, 0.8),
        **settings,
    )


def _ts(voxel: float, **fields) -> ModelSpec:
    fields.setdefault("layers_mask", "0000001")
    return ModelSpec(ref="VBoussot/impact-torchscript-models:TS/M730.pt", voxel_size=[voxel] * 3, **fields)


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, maps: dict[str, str], **knobs) -> ElastixEngine:
    """An engine built over ``maps`` written in the bundle directory, without the binary or the model downloads."""
    for name, text in maps.items():
        (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(elastix_engine, "feature_model", _feature_model)
    monkeypatch.setattr(ElastixEngine, "_ensure_binary", lambda self: Path("elastix"))
    return ElastixEngine(**{"parameter_maps": list(maps), **knobs})


def _staged(
    engine: ElastixEngine, tmp_path: Path, device_index: int = 0, spacing: tuple = (0.8, 0.8, 0.8)
) -> list[str]:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return [path.read_text() for path in engine._stage_parameter_maps(work, device_index, spacing)]


def test_a_rigid_stage_before_the_impact_map_keeps_its_own_schedule() -> None:
    # The levels describe the IMPACT metric's resolutions. Applied to every map, they rewrote a rigid Mattes MI stage
    # run first into a 3-level full-resolution one, so no preset could align the pair before IMPACT.
    levels = _levels((400, [_ts(6.0)]), (300, [_ts(3.0)]), (200, [_ts(2.0)]))

    assert _generate(RIGID, levels) == RIGID
    generated = _generate(IMPACT, levels)
    assert "(NumberOfResolutions 3)" in generated
    assert "(MaximumNumberOfIterations 400 300 200)" in generated
    assert '(ImpactModelsPath2 "TS/M730.pt")' in generated


def test_an_impact_map_without_models_is_refused() -> None:
    """elastix would read the map's own ImpactModelsPath entries, bare names it finds nowhere."""
    with pytest.raises(ValueError, match="declares no 'models'"):
        _generate(IMPACT)


def test_models_without_an_impact_map_are_refused_at_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "Parameters_Rigid.txt").write_text(RIGID)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="no parameter map has an IMPACT block"):
        ElastixEngine(["Parameters_Rigid.txt"], levels=_levels((400, [_ts(6.0)])))


def test_the_global_iteration_override_reaches_a_generated_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The annotation promises a global override, but a map generated from the levels silently kept their
    # iterations, and max_iterations even dropped the progress total.
    levels = _levels((400, [_ts(6.0, subset_features=64)]), (300, [_ts(3.0, subset_features=64)]))
    engine = _engine(tmp_path, monkeypatch, {"ParameterMap.txt": IMPACT}, levels=levels, max_iterations=7)

    (staged,) = _staged(engine, tmp_path)

    assert "(MaximumNumberOfIterations 7 7)" in staged
    assert "(ImpactSubsetFeatures0 64)" in staged and "(ImpactSubsetFeatures1 64)" in staged


def test_an_override_one_map_takes_is_not_reported_missing_and_can_target_one_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # final_grid_spacing lives in the B-spline map only: the rigid map printed a false note on every case and
    # tile. And an exact override reached every map, so the rigid stage in front of a B-spline could not be left
    # alone.
    bspline = '(Transform "BSplineTransform")\n(FinalGridSpacingInPhysicalUnits 16)\n(NumberOfResolutions 4)\n'
    engine = _engine(
        tmp_path,
        monkeypatch,
        {"Parameters_Rigid.txt": RIGID, "Parameters_BSpline.txt": bspline},
        final_grid_spacing=8.0,
        parameter_overrides=["Parameters_BSpline.txt:NumberOfResolutions=2", "Missing.txt:NumberOfResolutions=1"],
    )

    rigid, deformable = _staged(engine, tmp_path)

    assert "(NumberOfResolutions 4)" in rigid and "(NumberOfResolutions 2)" in deformable
    assert "(FinalGridSpacingInPhysicalUnits 8.0)" in deformable
    notes = capsys.readouterr().out
    assert "FinalGridSpacingInPhysicalUnits" not in notes and "'Missing.txt:NumberOfResolutions'" in notes


def test_the_seed_reaches_every_map_and_a_randomseed_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # elastix's samplers drew from 121212 and the IMPACT metric its channels, patches and 2D planes from the clock
    # whatever the seed the other engines take.
    maps = {"Rigid.txt": RIGID, "ParameterMap.txt": IMPACT}
    engine = _engine(tmp_path, monkeypatch, maps, levels=_levels((400, [_ts(6.0)])), seed=7)
    for staged in _staged(engine, tmp_path):
        assert staged.count("RandomSeed") == 1 and "(RandomSeed 7)" in staged
    net = RegistrationNet(parameter_maps=list(maps), levels=_levels((400, [_ts(6.0)])), seed=7)  # the presets' path
    assert net["Registration"]._engine._seed == 7
    engine = _engine(
        tmp_path, monkeypatch, maps, levels=_levels((400, [_ts(6.0)])), parameter_overrides=["RandomSeed=3"]
    )
    for staged in _staged(engine, tmp_path):
        assert staged.count("RandomSeed") == 1 and "(RandomSeed 3)" in staged


def test_the_progress_total_is_what_elastix_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # elastix repeats a single MaximumNumberOfIterations for every level: Generic_Rigid_BSpline declared 750 of the
    # 3000 iterations it runs, and the bar Slicer parses ran past 100 %. A rigid stage before an IMPACT map counted
    # nothing, and max_iterations dropped the total altogether.
    bspline = RIGID.replace("250", "500")
    assert _engine(tmp_path, monkeypatch, {"Rigid.txt": RIGID, "BSpline.txt": bspline})._iterations == 3000
    assert _engine(tmp_path, monkeypatch, {"Rigid.txt": RIGID}, max_iterations=10)._iterations == 40

    levels = _levels((400, [_ts(6.0)]), (300, [_ts(3.0)]))
    staged = _engine(tmp_path, monkeypatch, {"Rigid.txt": RIGID, "Impact.txt": IMPACT}, levels=levels)
    assert staged._iterations == 1000 + 700


def test_a_mask_reaches_elastix_binarised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Cast to UInt8, a soft mask's 0.5 became 0 and a label map's 256 or 1024 wrapped to 0, while the empty and
    # partial checks read every nonzero value as inside: part of the region, or all of it, left the metric.
    engine = _engine(tmp_path, monkeypatch, {"Rigid.txt": RIGID})
    values = np.zeros((4, 5, 6), np.float32)
    values[0, 0, :3] = [0.5, 256, 1024]
    mask = sitk.GetImageFromArray(values)
    written: dict[str, np.ndarray] = {}

    class Process:
        stdout = iter(())

        def __init__(self, args, **kwargs) -> None:
            written["mask"] = sitk.GetArrayFromImage(sitk.ReadImage(args[args.index("-fMask") + 1]))
            written["threads"] = kwargs["env"]["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]
            written["flags"] = args

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(elastix_engine.subprocess, "Popen", Process)
    monkeypatch.setattr(elastix_engine, "loader_env", lambda root: {})
    engine._elastix_root = tmp_path
    fixed = sitk.Image([6, 5, 4], sitk.sitkFloat32)

    with pytest.raises(RuntimeError, match="elastix failed"):
        engine.register(fixed, sitk.Image(fixed), -1, fixed_mask=mask)

    assert written["mask"][0, 0, :4].tolist() == [1, 1, 1, 0] and written["mask"].sum() == 3
    # konfai's per-rank ITK share: elastix at full core count in every rank oversubscribed the node. Not as -threads,
    # which crashes the IMPACT plugin of the published elastix.
    assert written["threads"] == str(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
    assert "-threads" not in written["flags"]


def test_the_per_layer_rows_hold_one_entry_per_selected_layer() -> None:
    # elastix-IMPACT reads SubsetFeatures, PCA, Distance and LayersWeight once per selected layer, flat across a
    # level's models: written once per model, a two-layer mask aborted elastix on the missing entry.
    mind = ModelSpec(ref="VBoussot/impact-torchscript-models:MIND/R1D2_3D.pt", voxel_size=[6.0] * 3, layers_mask="1")
    two = _ts(6.0, layers_mask="0101", layers_weight=[0.5, 0.25], subset_features=16, distance="Dice")

    generated = _generate(IMPACT, _levels((400, [mind, two])))

    assert "(ImpactSubsetFeatures0 100000 16 16)" in generated
    assert '(ImpactDistance0 "L2" "Dice" "Dice")' in generated
    assert "(ImpactLayersWeight0 1 0.5 0.25)" in generated


def test_models_alone_apply_to_every_resolution_the_map_runs() -> None:
    # Without levels, the map's own resolutions and iterations stand and every one compares the same models.
    generated = _generate(IMPACT, models=[_ts(2.0)])

    assert "(NumberOfResolutions 2)" in generated and "(MaximumNumberOfIterations 400 300)" in generated
    assert "(ImpactVoxelSize0 2 2 2)" in generated and "(ImpactVoxelSize1 2 2 2)" in generated


def test_the_loss_settings_reach_the_map() -> None:
    # The same IMPACT loss in every engine: the settings elastix reads beside the models.
    spec = _ts(2.0, feature_normalization="l2")

    generated = _generate(
        IMPACT, models=[spec], mode="Jacobian", normalize=False, feature_map_update_interval=25, mixed_precision=True
    )

    assert '(ImpactMode "Jacobian")' in generated and '(ImpactNormalizeLosses "false")' in generated
    assert "(ImpactFeaturesMapUpdateInterval 25)" in generated and '(ImpactUseMixedPrecision "true")' in generated
    assert '(ImpactFeatureNormalization0 "l2")' in generated
    assert '(ImpactNormalizeLosses "true")' in _generate(IMPACT, models=[spec])


def test_the_pyramid_is_the_map_s_own_unless_the_levels_change_its_length() -> None:
    # IMPACT reads the original images, so the pyramid only reaches the map's other metrics: it is no longer forced
    # to full resolution. A schedule written for another number of resolutions would stop elastix: dropped, elastix's
    # default pyramid runs.
    kept = _generate(IMPACT.replace("1 1 1 1 1 1)", "4 4 4 1 1 1)"), _levels((400, [_ts(6.0)]), (300, [_ts(3.0)])))
    dropped = _generate(IMPACT, _levels((400, [_ts(6.0)]), (300, [_ts(3.0)]), (200, [_ts(2.0)])))

    assert "(FixedImagePyramidRescaleSchedule 4 4 4 1 1 1)" in kept
    assert "PyramidRescaleSchedule" not in dropped


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"layers_mask": "0000"}, "keeps no layer"),
        ({"layers_mask": "01", "layers_weight": [0.5, 0.5]}, "2 values for 1 kept layers"),
        ({"voxel_size": [2.0, 2.0]}, "voxel_size needs 3 values"),
        ({"distance": "LNCC"}, "does not have"),
    ],
)
def test_a_cell_elastix_cannot_read_is_refused_at_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict, message: str
) -> None:
    # A long layers_weight silently weighted the next model's layers; the others aborted elastix mid-run, after
    # the binary install and the model downloads. elastix draws points, where LNCC's window means nothing.
    spec = ModelSpec(
        ref="VBoussot/impact-torchscript-models:TS/M730.pt", **{"voxel_size": [6.0] * 3, "layers_mask": "1", **fields}
    )

    with pytest.raises(ValueError, match=message):
        _engine(tmp_path, monkeypatch, {"Impact.txt": IMPACT}, levels=_levels((400, [spec])))


def test_a_model_spelled_with_its_defaults_gives_a_map_elastix_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The defaults wrote an empty voxel size and layer mask (elastix aborted) and subset_features 0, which the C++
    # clamps to ONE random channel although the annotation promises all of them. Without a voxel_size the model
    # sees the fixed image as elastix gets it, on this case's own grid.
    spec = ModelSpec(ref="VBoussot/impact-torchscript-models:TS/M730.pt")

    generated = _generate(IMPACT, _levels((400, [spec])))

    assert '(ImpactLayersMask0 "1")' in generated and "(ImpactLayersWeight0 1)" in generated
    assert "(ImpactSubsetFeatures0 100000)" in generated and "(ImpactVoxelSize0 0.8 0.8 0.8)" in generated
    engine = _engine(tmp_path, monkeypatch, {"Impact.txt": IMPACT}, levels=_levels((400, [spec])))
    (staged,) = _staged(engine, tmp_path, spacing=(0.5, 0.6, 2.5))
    assert "(ImpactVoxelSize0 0.5 0.6 2.5)" in staged


def test_the_generated_map_names_each_model_by_its_downloaded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The models were symlinked under the run's directory to match a relative ImpactModelsPath: on Windows without
    # Developer Mode a symlink needs admin rights (WinError 1314), and nothing fell back.
    downloaded = tmp_path / "hub" / "TS" / "M730.pt"
    engine = _engine(tmp_path, monkeypatch, {"Impact.txt": IMPACT}, levels=_levels((400, [_ts(6.0)])))
    for model in engine._feature_models.values():
        model.model_path = str(downloaded)

    (staged,) = _staged(engine, tmp_path)

    assert f'(ImpactModelsPath0 "{downloaded}")' in staged


def test_a_map_of_one_s_own_reaches_elastix_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A custom map lost its directory, ran without the composite transform the engine reads (failing only after the
    # whole registration) or IMPACT on the CPU without ImpactGPU, and a '// comment' tail turned an exact override
    # into a second entry elastix refused.
    own = tmp_path / "maps" / "Mine.txt"
    own.parent.mkdir()
    own.write_text(
        IMPACT.replace("(ImpactGPU 0)\n", "").replace('(ImpactMode "Static")\n', "")
        + "(NumberOfSpatialSamples 2000) // per level\n"
    )
    engine = _engine(
        tmp_path,
        monkeypatch,
        {},
        parameter_maps=[str(own)],
        levels=_levels((400, [_ts(6.0)])),
        parameter_overrides=["NumberOfSpatialSamples=500"],
    )

    (staged,) = _staged(engine, tmp_path, device_index=1)

    assert staged.count("NumberOfSpatialSamples") == 1 and "(NumberOfSpatialSamples 500)" in staged
    assert (
        '(WriteITKCompositeTransform "true")' in staged and '(ITKTransformOutputFileNameExtension "itk.txt")' in staged
    )
    assert "(ImpactGPU 1)" in staged and '(ImpactMode "Static")' in staged


def test_a_cuda_out_of_memory_under_a_long_backtrace_still_reaches_konfai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # elastix logs a c10 out-of-memory as 'std: <what()>' and then up to 64 frames of backtrace: only the last 40
    # lines reached the error, so the out-of-memory was never recognised and konfai never re-planned. The log went
    # with the run's directory.
    import torch
    from konfai.utils.vram import out_of_memory_as_torch

    engine = _engine(tmp_path, monkeypatch, {"Rigid.txt": RIGID})
    output = ["Resolution: 0\n", "std: CUDA out of memory. Tried to allocate 2.00 GiB.\n"]
    output += [f"frame #{i}: 0x7f elastix\n" for i in range(64)]

    class Process:
        stdout = iter(output)

        def __init__(self, args, cwd: str, **kwargs) -> None:
            (Path(cwd) / "elastix.log").write_text("".join(output))

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(elastix_engine.subprocess, "Popen", Process)
    monkeypatch.setattr(elastix_engine, "loader_env", lambda root: {})
    monkeypatch.setattr(elastix_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    engine._elastix_root = tmp_path
    fixed = sitk.Image([6, 5, 4], sitk.sitkFloat32)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="Tried to allocate 2") as raised:
        with out_of_memory_as_torch(True):
            engine.register(fixed, sitk.Image(fixed), 0)

    kept = str(raised.value.__cause__).split("(log: ", 1)[1].split(")", 1)[0]
    assert Path(kept).read_text().count("frame #") == 64


def test_the_field_maps_a_fixed_point_to_the_moving_one_channel_first() -> None:
    # The convention every consumer of Transform.h5 relies on: T(p) = p + u(p) takes a fixed point to the moving
    # one, components in world x, y, z, channel-first on the fixed grid [3, Z, Y, X], also for an oblique grid.
    from impact_reg_konfai.models.elastix_engine import _displacement_on

    fixed = sitk.Image([6, 5, 4], sitk.sitkFloat32)
    fixed.SetSpacing((2.0, 1.0, 3.0))
    fixed.SetDirection(sitk.VersorTransform((0, 0, 1), 0.3).GetMatrix())

    field = _displacement_on(fixed, sitk.TranslationTransform(3, (4.0, -1.5, 2.0)))

    assert field.shape == (3, 4, 5, 6)
    assert np.allclose(field.reshape(3, -1).mean(1), [4.0, -1.5, 2.0]) and np.ptp(field.reshape(3, -1), 1).max() < 1e-9


def test_two_maps_of_one_name_from_two_folders_both_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Staged under their base name, the second overwrote the first and elastix ran it twice: a rigid pre-alignment
    silently dropped."""
    for folder, transform in (("rigid", "EulerTransform"), ("bspline", "BSplineTransform")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "params.txt").write_text(f'(Transform "{transform}")\n(MaximumNumberOfIterations 10)\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(elastix_engine, "feature_model", _feature_model)
    monkeypatch.setattr(ElastixEngine, "_ensure_binary", lambda self: Path("elastix"))
    engine = ElastixEngine(
        parameter_maps=[str(tmp_path / "rigid" / "params.txt"), str(tmp_path / "bspline" / "params.txt")]
    )

    staged = _staged(engine, tmp_path)
    assert ['"EulerTransform"' in staged[0], '"BSplineTransform"' in staged[1]] == [True, True]


def test_voxel_sampling_gives_each_level_its_share_of_the_voxels_the_pyramid_leaves() -> None:
    # voxel_sampling is a share of the voxels in every engine, and elastix reads a count: each level gets that share
    # of the voxels its pyramid leaves, a recursive pyramid shrinking them, a generic one at rescale 1 keeping them.
    from impact_reg_konfai.models.elastix import sampled_spatial_samples

    recursive = """(ImageSampler "Random")
(NumberOfResolutions 4)
(FixedImagePyramid "FixedRecursiveImagePyramid")
(NumberOfSpatialSamples 2048)"""
    assert "(NumberOfSpatialSamples 16 125 1000 8000)" in sampled_spatial_samples(recursive, 0.001, 8_000_000)
    scheduled = recursive + "\n(FixedImagePyramidSchedule 4 4 2 2 2 1 1 1 1 1 1 1)"
    assert "(NumberOfSpatialSamples 250 2000 8000 8000)" in sampled_spatial_samples(scheduled, 0.001, 8_000_000)
    generic = """(ImageSampler "RandomCoordinate")
(NumberOfResolutions 2)
(FixedImagePyramid "FixedGenericImagePyramid")
(FixedImagePyramidRescaleSchedule 1 1 1 1 1 1)"""
    assert sampled_spatial_samples(generic, 0.001, 8_000_000).endswith("(NumberOfSpatialSamples 8000 8000)")
    with pytest.raises(ValueError, match="random one"):
        sampled_spatial_samples('(ImageSampler "Full")', 0.1, 1000)


def test_voxel_sampling_counts_the_fixed_mask_at_run_time_and_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The count is written when elastix is called, from the voxels its sampler draws from (the fixed mask's), and
    # replaces the IMPACT map's own and spatial_samples; the rigid map in front keeps what spatial_samples gives it.
    maps = {
        "Rigid.txt": RIGID + "(NumberOfSpatialSamples 2048)\n",
        "Impact.txt": IMPACT + '(ImageSampler "Random")\n(NumberOfSpatialSamples 2048)\n',
    }
    engine = _engine(tmp_path, monkeypatch, maps, models={"0": _ts(2.0)}, voxel_sampling=0.5, spatial_samples=500)
    staged: dict[str, str] = {}

    class Process:
        stdout = iter(())

        def __init__(self, args, **kwargs) -> None:
            staged.update(
                {Path(args[i + 1]).name: Path(args[i + 1]).read_text() for i, a in enumerate(args) if a == "-p"}
            )

        def wait(self) -> int:
            return 1

    monkeypatch.setattr(elastix_engine.subprocess, "Popen", Process)
    monkeypatch.setattr(elastix_engine, "loader_env", lambda root: {})
    engine._elastix_root = tmp_path
    fixed = sitk.Image([6, 5, 4], sitk.sitkFloat32)
    values = np.zeros((4, 5, 6), np.uint8)
    values[:2, :4, :5] = 1  # 40 voxels
    mask = sitk.GetImageFromArray(values)

    with pytest.raises(RuntimeError, match="elastix failed"):
        engine.register(fixed, sitk.Image(fixed), -1, fixed_mask=mask)

    assert "(NumberOfSpatialSamples 500)" in staged["0_Rigid.txt"]
    assert "(NumberOfSpatialSamples 20 20)" in staged["1_Impact.txt"]
