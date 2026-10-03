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

"""Elastix-IMPACT runtime for the registration bundle.

``ElastixEngine`` installs the elastix-IMPACT binary, downloads the TorchScript feature models, stages the
parameter maps (their IMPACT lines generated from the models, or copied + overridden), runs the subprocess, and
resamples.
``ElastixRegistration`` is the graph module ``RegistrationNet`` wires, it bridges KonfAI tensors <-> SITK
images. The config -> parameter-map MAPPING lives in ``elastix.py`` and is imported here.
"""

import math
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tqdm
from konfai.utils.dataset import image_to_data
from konfai.utils.vram import device_out_of_memory

from .elastix import generate_impact_parameter_map, has_impact_block, sampled_spatial_samples
from .elastix_install import cuda_upgrade_available, get_elastix_bin, install_elastix_impact, loader_env, try_elastix
from .impact_loss import check_models, feature_model, sorted_specs
from .intensity import EngineRegistration, is_partial_mask, winsorized
from .orientation import world_aligned_pair

# Elastix + IMPACT binary is cached once here (with its own LibTorch when the environment's torch is another
# version) and reused across runs.
# Set KONFAI_ELASTIX_DIR to point at an existing install and skip the download.
ELASTIX_CACHE = Path.home() / ".cache" / "konfai" / "elastix-impact"


def _displacement_on(fixed: sitk.Image, transform: sitk.Transform) -> np.ndarray:
    """``transform`` sampled as a displacement field on the grid of ``fixed``, channel-first."""
    dvf = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection(),
    )
    dvf_np, _ = image_to_data(dvf)
    return dvf_np


def _iterations_of(text: str) -> int:
    """The iterations elastix runs for a parameter map: ``MaximumNumberOfIterations`` over its
    ``NumberOfResolutions`` levels, a level without an entry of its own reading the first one (elastix's
    default entry), so a single value counts once per level."""
    budget = re.search(r"^\s*\(MaximumNumberOfIterations\s+([^)]*)\)", text, re.MULTILINE)
    if budget is None:
        return 0
    tokens = [int(float(token)) for token in budget.group(1).split()]
    levels = re.search(r"^\s*\(NumberOfResolutions\s+(\d+)", text, re.MULTILINE)
    count = int(levels.group(1)) if levels else len(tokens)
    return sum(tokens[k] if k < len(tokens) else tokens[0] for k in range(count))


def _kept_log(work: Path) -> str:
    """Copy elastix.log out of the run's directory, which is removed, and say where it went."""
    log = work / "elastix.log"
    if not log.is_file():
        return ""
    kept = Path(tempfile.gettempdir()) / f"{work.name}-elastix.log"
    shutil.copyfile(log, kept)
    return f" (log: {kept})"


def _cuda_hint(captured: list[str], root: Path) -> str:
    """What to do about an install that cannot run IMPACT as asked. A CPU build answers ``-h`` and passes for a
    valid install, and so does a build whose IMPACT plugin is loaded only once a map asks for it (the
    plugin-based elastix-IMPACT): the failure only comes mid-registration."""
    if any("IMPACT" in line and "could not be loaded" in line for line in captured):
        return (
            f"\nThe IMPACT plugin of the elastix-IMPACT install at '{root}' could not be loaded: it was built "
            "against another LibTorch than the one on its loader path. Point KONFAI_ELASTIX_EXTRA_LIB at the "
            "LibTorch it was built against, or KONFAI_ELASTIX_DIR at a build for this machine."
        )
    if not any("CUDA is not available" in line for line in captured):
        return ""
    return (
        f"\nThe elastix-IMPACT install at '{root}' cannot use CUDA: it is a CPU build. The installer takes the "
        "CUDA one for a CUDA torch and an NVIDIA driver of 570.26 or later (delete the install to have it made "
        "again); otherwise point KONFAI_ELASTIX_DIR at a CUDA build, or run on the CPU (--cpu)."
    )


def _mask_hint(captured: list[str]) -> str:
    """What to do when the random sampler gives up on a thin fixed mask: it draws points in the mask's bounding
    box and aborts after ten times the samples it needs."""
    if not any("Could not find enough image samples" in line for line in captured):
        return ""
    return (
        "\nThe fixed mask fills too little of its bounding box for elastix's random sampler: set "
        "parameter_overrides to ['ImageSampler=\"RandomSparseMask\"'], which draws inside the mask."
    )


def _extent_hint(captured: list[str], maps: list[Path], image: sitk.Image) -> str:
    """What to do when the image is too small for the grid IMPACT's models see it on (their ImpactVoxelSize): elastix
    names neither the image nor the grid in the errors ``signs`` lists."""
    signs = (
        "rejected its input",
        "Too many samples map outside moving image buffer",
        "Expected more than 1 spatial element",
    )
    if not any(sign in line for sign in signs for line in captured):
        return ""
    dim = image.GetDimension()
    extent = [size * spacing for size, spacing in zip(image.GetSize(), image.GetSpacing(), strict=True)]
    coarsest = None  # (voxel, patch, level) of the coarsest model grid over every map and level
    for pmap in maps:
        text = pmap.read_text(encoding="utf-8")
        for level, values in re.findall(r"^\(ImpactVoxelSize(\d+)((?:\s+[-\d.eE]+)+)\)", text, re.MULTILINE):
            patches = re.search(rf"^\(ImpactPatchSize{level}((?:\s+\d+)+)\)", text, re.MULTILINE)
            voxels = [float(v) for v in values.split()]
            sizes = [int(v) for v in patches.group(1).split()] if patches else [0] * len(voxels)
            for start in range(0, len(voxels) - dim + 1, dim):  # one voxel size (and patch) per model
                voxel, patch = voxels[start : start + dim], sizes[start : start + dim]
                if coarsest is None or max(voxel) > max(coarsest[0]):
                    coarsest = (voxel, patch, level)
    if coarsest is None:
        return ""
    voxel, patch, level = coarsest
    grid = [e / v for e, v in zip(extent, voxel, strict=True)]
    wider = [p * v for p, v, e in zip(patch, voxel, extent, strict=True) if p * v > e]
    if min(grid) >= 16 and not wider:
        return ""
    spans = f", where its {max(patch)}-voxel patch spans {max(wider):g} mm" if wider else ""
    return (
        f"\nThe image, {' x '.join(f'{e:.1f}' for e in extent)} mm, is too small for the grid IMPACT's models see it "
        f"on: {' x '.join(str(int(g)) for g in grid)} voxels at {' x '.join(f'{v:g}' for v in voxel)} mm (level "
        f"{level}){spans}. The IMPACT presets are sized for CT and MR at millimetre scale: a small field of view "
        "(microscopy, a crop) registers on its intensities (Generic_Rigid_BSpline, FireANTs_SyN), or on features at "
        "a finer grid, with a smaller voxel_size for each of the preset's models."
    )


class ElastixEngine:
    """Run the elastix-IMPACT binary on a fixed/moving pair; return the displacement field on the fixed grid.

    NOTE: the elastix-IMPACT metric lives only in the custom ``elastix-impact`` binary (SimpleElastix does
    NOT ship it), so registration is a subprocess call, not ``sitk.ElastixImageFilter``.
    """

    def __init__(
        self,
        parameter_maps: list[str],
        max_iterations: int = 0,
        final_grid_spacing: float = 0.0,
        spatial_samples: int = 0,
        parameter_overrides: list[str] = [],
        models: dict = {},
        levels: dict = {},
        mode: str = "Static",
        normalize: bool = True,
        feature_map_update_interval: int = -1,
        mixed_precision: bool = False,
        voxel_sampling: float = 1.0,
        seed: int = 42,
    ) -> None:
        # The parameter-map .txt files are per-preset config staged into the run's working directory (KonfAIApp
        # chdir's into the app workspace before building the model), so resolve them against cwd, not this
        # module's directory, which is the installed package, not next to the .txt.
        # An absolute path to a map of one's own is honoured.
        self._bundle_dir = Path.cwd()
        self._parameter_maps = [
            Path(p) if Path(p).is_absolute() and Path(p).is_file() else self._bundle_dir / Path(p).name
            for p in parameter_maps
        ]
        # The models rewrite a template's IMPACT lines; they never create one. Without a
        # map, elastix would launch with no -p and die in a cryptic subprocess error: fail here instead.
        if not self._parameter_maps:
            raise ValueError(
                "at least one parameter-map template is required; 'models' and 'levels' rewrite a template's "
                "IMPACT lines, they do not replace it."
            )
        self._max_iterations = max_iterations
        self._final_grid_spacing = final_grid_spacing
        self._spatial_samples = spatial_samples
        if not 0 < voxel_sampling <= 1:
            raise ValueError(f"voxel_sampling is a share of the voxels, in (0, 1]: got {voxel_sampling}.")
        self._voxel_sampling = float(voxel_sampling)
        self._seed = int(seed)
        self._parameter_overrides = list(parameter_overrides)
        # ImpactMode: Static computes features once per level (PatchSize 0 0 0 = whole image); Jacobian
        # samples random FOV-sized patches each iteration. One mode per preset.
        self._mode = mode
        self._loss = {
            "normalize": normalize,
            "feature_map_update_interval": feature_map_update_interval,
            "mixed_precision": mixed_precision,
        }
        # With ``models`` / ``levels`` each map's IMPACT block is GENERATED from them. Neither = an intensity
        # preset (no IMPACT models): the fixed maps are staged with only the global overrides.
        self._impact_models, self._levels = models, levels
        if (models or levels) and not any(
            has_impact_block(p.read_text(encoding="utf-8")) for p in self._parameter_maps
        ):
            raise ValueError(
                "'models' and 'levels' set the IMPACT metric, but no parameter map has an IMPACT block "
                "((ImpactModelsPath0 ...) and its siblings) for them to rewrite."
            )
        # Each model fetched and shaped by the registry, by (ref, layers_mask).
        specs = sorted_specs(models) + [m for level in sorted_specs(levels) for m in sorted_specs(level.models)]
        if specs:
            check_models(specs, "elastix", dense=False)  # before any download
        unique = {(m.ref, m.layers_mask): m for m in specs}
        self._feature_models = {key: feature_model(m) for key, m in unique.items()}
        # Jacobian mode differentiates the metric through the networks: probed at the first run, when the models are
        # loaded, so a layer without a gradient is refused.
        self._unchecked = mode.strip().strip('"').lower() == "jacobian"
        # The maps are generated here once, so the progress-bar total counts the iterations elastix will run, over
        # every map, overrides included. Any positive spacing: the build-time maps only count the iterations.
        maps, unused = self._map_texts(
            -1, (1.0, 1.0, 1.0), voxels=1
        )  # voxels: a sampler voxel_sampling can't set fails here
        if self._voxel_sampling < 1 and spatial_samples > 0:
            print("[ImpactReg] note: voxel_sampling sets the IMPACT map's NumberOfSpatialSamples, not spatial_samples.")
        for key in unused:
            print(f"[ImpactReg] note: override '{key}' matched no entry in the preset's parameter maps.")
        self._iterations = sum(_iterations_of(text) for _, text in maps)
        self._elastix_bin = self._ensure_binary()

    def _ensure_binary(self) -> Path:
        # Optional override: point at an existing elastix-IMPACT install (skips the download).
        override = os.environ.get("KONFAI_ELASTIX_DIR", "")
        # Absolute: a registration runs from a temporary directory, and loader_env spells its search
        # paths from this root.
        self._elastix_root = Path(override).expanduser().resolve() if override else ELASTIX_CACHE
        if override:
            try_elastix(self._elastix_root)
            return get_elastix_bin(self._elastix_root).resolve()
        ELASTIX_CACHE.mkdir(parents=True, exist_ok=True)
        try:
            try_elastix(ELASTIX_CACHE)
        except RuntimeError:
            # Staged and probed before it replaces the cache: it raises, once, when nothing runs.
            install_elastix_impact(ELASTIX_CACHE, force_cuda=False, force_cpu=False)
        else:
            # A working install can be the CPU build of an environment that could use the CUDA one.
            if cuda_upgrade_available(ELASTIX_CACHE):
                try:
                    install_elastix_impact(ELASTIX_CACHE, force_cuda=False, force_cpu=False)
                except Exception as failure:
                    print(f"[ImpactReg] note: keeping the CPU elastix-IMPACT install, the CUDA one failed: {failure}")
        return get_elastix_bin(ELASTIX_CACHE).resolve()

    def _parameter_map_overrides(self) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """The tuned knobs as parameter-map overrides: ``(per_token, exact)``.

        ``per_token`` maps an elastix key to a value replacing **each** existing token, preserving
        per-resolution multiplicity: ``max_iterations`` replaces every level's budget, the global override its
        annotation promises. ``exact`` entries (from
        ``parameter_overrides``, ``Key=value text``) replace the whole value verbatim and win over the named
        knobs; ``Map.txt:Key=value text`` touches the map of that name only (the IMPACT stage and not the rigid
        one before it). A named knob only REPLACES a key the map already has; an ``exact`` entry the map lacks
        is appended (``_apply_map_overrides``).
        """
        per_token: dict[str, str] = {}
        if self._max_iterations > 0:
            per_token["MaximumNumberOfIterations"] = str(int(self._max_iterations))
        if self._final_grid_spacing > 0:
            per_token["FinalGridSpacingInPhysicalUnits"] = str(float(self._final_grid_spacing))
        if self._spatial_samples > 0:
            per_token["NumberOfSpatialSamples"] = str(int(self._spatial_samples))
        exact: list[tuple[str, str]] = []
        for entry in self._parameter_overrides:
            key, sep, value = entry.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"Invalid parameter_overrides entry '{entry}': expected 'Key=value text'.")
            exact.append((key.strip(), value.strip()))
        return per_token, exact

    @staticmethod
    def _apply_map_overrides(
        text: str, per_token: dict[str, str], exact: list[tuple[str, str]], device_index: int
    ) -> tuple[str, set[str]]:
        """Patch a parameter map: set ImpactGPU to the device, apply exact key overrides, replace each token
        of a per-token knob (preserving multiplicity); return the map and the override keys it took.

        On the CPU (``device_index`` below 0) ``ImpactUseMixedPrecision`` is forced to ``"false"``: half
        precision has no 3-D pooling on the CPU.
        """
        # An entry may be followed by a '// comment'.
        entry_pattern = re.compile(r"^(\s*)\((\S+)((?:\s+[^)]*)?)\)\s*(?://.*)?$")
        seen: set[str] = set()
        lines = []
        for line in text.splitlines():
            match = entry_pattern.match(line)
            if match:
                indent, key, values = match.group(1), match.group(2), match.group(3)
                if key == "ImpactGPU":
                    seen.add(key)
                    line = f"{indent}(ImpactGPU {device_index})"
                elif key == "ImpactUseMixedPrecision" and device_index < 0:
                    # Handled here, so an exact override of the key is not appended behind it.
                    seen.add(key)
                    line = f'{indent}(ImpactUseMixedPrecision "false")'
                else:
                    exact_value = next((value for k, value in exact if k == key), None)
                    if exact_value is not None:
                        seen.add(key)
                        line = f"{indent}({key} {exact_value})"
                    else:
                        if key in per_token:
                            seen.add(key)
                            replaced = " ".join(per_token[key] for _ in values.split())
                            line = f"{indent}({key} {replaced})"
            lines.append(line)
        # A raw ``parameter_overrides`` entry is the escape hatch for ANY elastix parameter, including one
        # the preset's map never mentions (an ITK default such as RequiredRatioOfValidSamples): an absent exact
        # override is APPENDED. The named knobs (final_grid_spacing, spatial_samples, ...) keep replacing what exists and only
        # warn: injecting them into a map that does not use them is meaningless (e.g. a B-spline grid
        # spacing in a rigid-only preset).
        missing_exact = [(key, value) for key, value in exact if key not in seen]
        if missing_exact:
            lines.append("// appended by impact_reg_konfai parameter_overrides")
            lines += [f"({key} {value})" for key, value in missing_exact]
            seen.update(key for key, _ in missing_exact)
        return "\n".join(lines), seen

    def _map_texts(
        self, device_index: int, native_voxel_size: tuple[float, ...], voxels: int | None = None
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Each parameter map as elastix reads it on ``device_index``, as ``(name, text)``, and the overrides no
        map took.

        With models, each map carrying an IMPACT block is GENERATED from them + the registry, a model without a
        voxel_size seeing the image at ``native_voxel_size``; without, the preset's maps are copied. Both then
        apply every per-token / exact override and set the ImpactGPU device. With ``voxels`` (the fixed image's,
        or its mask's) and a voxel_sampling below 1, the IMPACT maps then draw that share of them at each level.
        """
        maps = []
        per_token, exact = self._parameter_map_overrides()
        names = {src.name for src in self._parameter_maps}
        consumed: set[str] = set()
        for src in self._parameter_maps:
            text = src.read_text(encoding="utf-8")
            if self._impact_models or self._levels:
                text = generate_impact_parameter_map(
                    text,
                    self._impact_models,
                    self._levels,
                    self._feature_models,
                    native_voxel_size,
                    self._mode,
                    **self._loss,
                )
            mine = [(key.rpartition(":")[2], value) for key, value in exact if key.rpartition(":")[0] in ("", src.name)]
            # What the engine reads back is not the map's to choose: the composite transform file it samples,
            # and for IMPACT the device (without ImpactGPU, elastix-IMPACT ran on the CPU).
            owned = [("WriteITKCompositeTransform", '"true"'), ("ITKTransformOutputFileNameExtension", '"itk.txt"')]
            if has_impact_block(text):
                owned.append(("ImpactGPU", str(device_index)))
            # The seed, in every map: elastix's generator, which the RandomCoordinate and RandomSparseMask samplers
            # draw their points from (121212 without it), and the IMPACT metric's, which draws its channels, patches
            # and 2D planes (from the clock without it). A RandomSeed override wins.
            if "RandomSeed" not in dict(mine):
                mine.append(("RandomSeed", str(self._seed)))
            mine = owned + [(key, value) for key, value in mine if key not in dict(owned)]
            text, seen = self._apply_map_overrides(text, per_token, mine, device_index)
            if voxels is not None and self._voxel_sampling < 1 and has_impact_block(text):
                text = sampled_spatial_samples(text, self._voxel_sampling, voxels)
            consumed |= seen
            maps.append((src.name, text))
        # A knob the rigid map lacks and the B-spline map takes is no news: only one no map took is.
        unused = sorted(set(per_token) - consumed) + [
            key for key, _ in exact if key.rpartition(":")[0] not in {"", *names}
        ]
        return maps, unused

    def _stage_parameter_maps(
        self, work: Path, device_index: int, native_voxel_size: tuple[float, ...], voxels: int | None = None
    ) -> list[Path]:
        """Stage the parameter maps, as elastix reads them on ``device_index``, into ``work``, each under its rank, so
        two maps of one name from two folders (``rigid/params.txt``, ``bspline/params.txt``) stay apart."""
        staged = []
        for index, (name, text) in enumerate(self._map_texts(device_index, native_voxel_size, voxels)[0]):
            dst = work / f"{index}_{name}"
            dst.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            staged.append(dst)
        return staged

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> np.ndarray:
        """Register ``moving`` onto ``fixed``; return the displacement field, channel-first, on the fixed grid.

        Optional ``fixed_mask`` / ``moving_mask`` restrict the similarity metric to a region (elastix
        ``-fMask`` / ``-mMask``); a mask covering the whole image is equivalent to passing none, and a
        fixed mask with no voxel in it leaves nothing to register, so the field is zero.
        """
        if fixed_mask is not None and not sitk.GetArrayViewFromImage(fixed_mask).any():
            # Nothing to register: in a tiled run, a patch the tissue does not reach.
            return _displacement_on(fixed, sitk.Transform(fixed.GetDimension(), sitk.sitkIdentity))
        grid = fixed  # the field is sampled on the fixed image as it came
        if self._feature_models:
            for model in self._feature_models.values() if self._unchecked else ():
                model.check(gradient=True)
            self._unchecked = False
            # IMPACT's features are computed along each image's voxel axes: both images go in with their voxel axes
            # in LPS order, the transform comes back physical. A residual oblique rotation is sampled physically by
            # elastix-IMPACT, which is built on ITKIMPACT.
            fixed, moving, fixed_mask, moving_mask = world_aligned_pair(fixed, moving, fixed_mask, moving_mask)
        else:
            # Grey values alone, compared through mutual information: a few hot voxels no longer squeeze the tissue
            # into one bin. IMPACT's models read the intensities they were trained on, so an IMPACT run keeps them.
            fixed, moving = winsorized(fixed), winsorized(moving)
        work = Path(tempfile.mkdtemp(prefix="konfai_reg_"))
        try:
            fixed_path, moving_path = work / "Fixed.mha", work / "Moving.mha"
            sitk.WriteImage(fixed, str(fixed_path))
            sitk.WriteImage(moving, str(moving_path))

            args = [str(self._elastix_bin), "-f", str(fixed_path), "-m", str(moving_path)]
            for flag, mask, name in (
                ("-fMask", fixed_mask, "FixedMask.mha"),
                ("-mMask", moving_mask, "MovingMask.mha"),
            ):
                if is_partial_mask(mask):
                    mask_path = work / name
                    # Binarised: a cast would truncate a soft mask and wrap labels.
                    sitk.WriteImage(sitk.Cast(mask != 0, sitk.sitkUInt8), str(mask_path))
                    args += [flag, str(mask_path)]
            args += ["-out", str(work)]
            # A model without a voxel_size sees the fixed image's grid, as elastix gets it.
            # voxel_sampling counts the voxels the sampler draws from: the fixed mask's, when there is one.
            voxels = math.prod(fixed.GetSize())
            if is_partial_mask(fixed_mask):
                voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(fixed_mask)))
            staged = self._stage_parameter_maps(work, device_index, fixed.GetSpacing(), voxels)
            for pmap in staged:
                args += ["-p", str(pmap)]

            env = loader_env(self._elastix_root)
            # The ITK thread share konfai gave this rank, so N ranks do not oversubscribe the node. Through the
            # environment, which every copy of ITK in the process reads: the published elastix links ITK statically
            # into the IMPACT plugin too, and its -threads crashes that plugin.
            env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
            proc = subprocess.Popen(  # nosec B603
                args,
                cwd=str(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            # Drive a tqdm bar over elastix's iteration lines so SlicerKonfAI (which parses the "N% done"
            # progress line) shows real progress. The description mirrors KonfAI's bars: resolution level + the
            # metric value.
            captured: list[str] = []
            told: set[str] = set()
            iteration_line = re.compile(r"^\d+\s")
            budget = self._iterations or None
            progress = tqdm.tqdm(total=budget, desc="Registration", ncols=0, leave=True)
            assert proc.stdout is not None
            resolution = 0
            for line in proc.stdout:
                captured.append(line)
                stripped = line.strip()
                if stripped.startswith("Resolution:"):
                    try:
                        resolution = int(stripped.split(":", 1)[1])
                    except ValueError:
                        pass
                elif stripped.startswith("IMPACT:") and stripped not in told:
                    # IMPACT says when it ran out of device memory and went on with smaller patches: the run
                    # is slower for it, and nothing else tells why.
                    told.add(stripped)
                    progress.write(stripped)
                elif iteration_line.match(line):
                    progress.update(1)
                    columns = line.split()  # column 2 is the metric (header "1:ItNr 2:Metric ...")
                    if len(columns) > 1:
                        try:
                            progress.set_description(
                                f"Registration : res {resolution} | metric {float(columns[1]):.4f}"
                            )
                        except ValueError:
                            pass
            progress.close()
            returncode = proc.wait()
            if returncode != 0:
                # elastix follows a CUDA out-of-memory with its C++ backtrace: the line leads the message, where
                # out_of_memory_as_torch looks for it, so konfai re-plans the patch.
                oom = [line for line in captured if device_out_of_memory(line)][:1]
                raise RuntimeError(
                    f"elastix failed (code {returncode}){_kept_log(work)}:\n{''.join(oom + captured[-40:])}"
                    f"{_cuda_hint(captured, self._elastix_root)}{_mask_hint(captured)}"
                    f"{_extent_hint(captured, staged, moving)}"
                )

            transforms = sorted(
                work.glob("TransformParameters.*-Composite.itk.txt"),
                key=lambda p: int(p.name.split(".")[1].split("-")[0]),
            )
            if not transforms:
                raise FileNotFoundError(f"elastix produced no composite transform file{_kept_log(work)}.")
            return _displacement_on(grid, sitk.ReadTransform(str(transforms[-1])))
        finally:
            shutil.rmtree(work, ignore_errors=True)


class ElastixRegistration(EngineRegistration):
    """The elastix engine built from its settings, run as every engine is (``EngineRegistration``)."""

    def __init__(
        self,
        engine: str,
        parameter_maps: list[str],
        max_iterations: int = 0,
        final_grid_spacing: float = 0.0,
        spatial_samples: int = 0,
        parameter_overrides: list[str] = [],
        models: dict = {},
        levels: dict = {},
        mode: str = "Static",
        normalize: bool = True,
        feature_map_update_interval: int = -1,
        mixed_precision: bool = False,
        voxel_sampling: float = 1.0,
        seed: int = 42,
    ) -> None:
        if engine != "elastix":
            raise NotImplementedError(f"ElastixRegistration engine '{engine}' is not implemented yet.")
        super().__init__(
            ElastixEngine(
                parameter_maps,
                max_iterations,
                final_grid_spacing,
                spatial_samples,
                parameter_overrides,
                models,
                levels,
                mode,
                normalize,
                feature_map_update_interval,
                mixed_precision,
                voxel_sampling,
                seed,
            )
        )
