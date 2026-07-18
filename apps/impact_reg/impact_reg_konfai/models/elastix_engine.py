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
parameter maps (generated from the model matrix or copied + overridden), runs the subprocess, and resamples.
``ElastixRegistration`` is the graph module ``RegistrationNet`` wires — it bridges KonfAI tensors <-> SITK
images. The config -> parameter-map MAPPING lives in ``elastix.py`` and is imported here.
"""

import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import tqdm
from huggingface_hub import hf_hub_download
from .elastix_install import get_elastix_bin, install_elastix_impact, try_elastix
from konfai.utils.dataset import Attribute, data_to_image, image_to_data

from .elastix import _sorted_specs, generate_impact_parameter_map, load_models_registry

# Elastix + IMPACT binary is cached once here (heavy: binary + LibTorch) and reused across runs.
# Set KONFAI_ELASTIX_DIR to point at an existing install and skip the download.
ELASTIX_CACHE = Path.home() / ".cache" / "konfai" / "elastix-impact"


def _is_partial_mask(mask: "sitk.Image | None") -> bool:
    """True only for a mask that actually restricts the metric region — some voxels in, some out. An
    absent optional mask arrives as a whole-image (all-ones) default from KonfAI, and an all-zero mask
    is degenerate; both are treated as no mask, so elastix runs without ``-fMask`` / ``-mMask`` (i.e.
    the whole image) instead of paying for a mask that restricts nothing."""
    if mask is None:
        return False
    arr = sitk.GetArrayViewFromImage(mask)
    return bool((arr > 0).any()) and bool((arr == 0).any())


class ElastixEngine:
    """Run the elastix-IMPACT binary on a fixed/moving pair; return (moved, dvf) on the fixed grid.

    NOTE: the elastix-IMPACT metric lives only in the custom ``elastix-impact`` binary (SimpleElastix does
    NOT ship it), so registration is a subprocess call, not ``sitk.ElastixImageFilter``.
    """

    def __init__(
        self,
        parameter_maps: list[str],
        max_iterations: int = 0,
        final_grid_spacing: float = 0.0,
        subset_features: int = 0,
        spatial_samples: int = 0,
        parameter_overrides: list[str] = [],
        resolutions: dict = {},
        mode: str = "Static",
    ) -> None:
        # The parameter-map .txt files are per-preset config staged into the run's working directory (KonfAIApp
        # chdir's into the app workspace before building the model), so resolve them against cwd -- not this
        # module's directory, which is the installed package, not next to the .txt.
        self._bundle_dir = Path.cwd()
        self._parameter_maps = [self._bundle_dir / Path(p).name for p in parameter_maps]
        # Matrix mode rewrites a template's resolution-dependent lines; it never creates one. Without a
        # map, elastix would launch with no -p and die in a cryptic subprocess error — fail here instead.
        if not self._parameter_maps:
            raise ValueError(
                "at least one parameter-map template is required; 'resolutions' rewrites a template, "
                "it does not replace it."
            )
        self._max_iterations = max_iterations
        self._final_grid_spacing = final_grid_spacing
        self._subset_features = subset_features
        self._spatial_samples = spatial_samples
        self._parameter_overrides = list(parameter_overrides)
        # ImpactMode: Static computes features once per level (PatchSize 0 0 0 = whole image); Jacobian
        # samples random FOV-sized patches each iteration. One mode per preset.
        self._mode = mode
        # Matrix mode: with ``resolutions`` the map is GENERATED from it. Empty ``resolutions`` = an
        # intensity preset (no IMPACT models): the fixed maps are staged with only the global overrides.
        self._resolutions = resolutions
        self._registry = load_models_registry() if resolutions else {}
        # Feature models are DERIVED — the unique refs across the matrix cells (no flat ``models`` param).
        models: list[str] = []
        for res in _sorted_specs(resolutions):
            for model in _sorted_specs(res.models):
                if model.ref not in models:
                    models.append(model.ref)
        self._models = models
        # ``iterations`` (the progress-bar total) is DERIVED: the sum of per-resolution iteration budgets.
        self._iterations = self._total_iterations()
        self._elastix_bin = self._ensure_binary()
        self._local_models = self._download_models()

    def _total_iterations(self) -> int:
        """Total iterations across resolutions — the progress-bar budget, from the config (or the maps)."""
        if self._resolutions:
            return sum(int(res.max_iterations) for res in _sorted_specs(self._resolutions))
        total = 0
        for src in self._parameter_maps:
            match = re.search(r"\(MaximumNumberOfIterations\s+([^)]*)\)", src.read_text(encoding="utf-8"))
            if match:
                total += sum(int(token) for token in match.group(1).split())
        return total

    def _ensure_binary(self) -> Path:
        # Optional override: point at an existing elastix-IMPACT install (skips the download).
        override = os.environ.get("KONFAI_ELASTIX_DIR", "")
        if override:
            try_elastix(Path(override))
            return get_elastix_bin(Path(override)).resolve()
        ELASTIX_CACHE.mkdir(parents=True, exist_ok=True)
        try:
            try_elastix(ELASTIX_CACHE)
        except Exception:
            install_elastix_impact(ELASTIX_CACHE, force_cuda=False, force_cpu=False)
            try_elastix(ELASTIX_CACHE)
        return get_elastix_bin(ELASTIX_CACHE).resolve()

    def _download_models(self) -> list[tuple[str, Path]]:
        """Fetch the TorchScript feature models (``repo:filename``, or a local file); keep
        ``(staged_name, local_path)``. The staged name equals ``_model_key(ref)`` -- the path the
        generated/preset map references -- so a local ref stages under the very name the map resolves."""
        models = []
        for ref in self._models:
            if ":" in ref:
                repo, filename = ref.split(":", 1)
                local = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="model"))  # nosec B615
                models.append((filename, local))
            else:
                models.append((ref, Path(ref).expanduser().resolve()))
        return models

    def _parameter_map_overrides(self, global_only: bool = False) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """The tuned knobs as parameter-map overrides: ``(per_token, exact)``.

        ``per_token`` maps an elastix key (or the ``ImpactSubsetFeatures`` prefix) to a value replacing
        **each** existing token, preserving per-resolution / per-model multiplicity. ``exact`` entries (from
        ``parameter_overrides``, ``Key=value text``) replace the whole value verbatim and win over the named
        knobs. Overrides only REPLACE keys already present — never inject. ``global_only`` (matrix mode) drops
        ``max_iterations`` / ``subset_features`` (the matrix already sets those per cell).
        """
        per_token: dict[str, str] = {}
        if not global_only and self._max_iterations > 0:
            per_token["MaximumNumberOfIterations"] = str(int(self._max_iterations))
        if self._final_grid_spacing > 0:
            per_token["FinalGridSpacingInPhysicalUnits"] = str(float(self._final_grid_spacing))
        if not global_only and self._subset_features > 0:
            per_token["ImpactSubsetFeatures"] = str(int(self._subset_features))  # prefix: indexed per metric
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
    ) -> str:
        """Patch a parameter map: set ImpactGPU to the device, apply exact key overrides, replace each token
        of a per-token knob (preserving multiplicity), and warn for a requested key absent from the map.
        """
        entry_pattern = re.compile(r"^(\s*)\((\S+)((?:\s+[^)]*)?)\)\s*$")
        requested = set(per_token) | {key for key, _ in exact}
        seen: set[str] = set()
        lines = []
        for line in text.splitlines():
            match = entry_pattern.match(line)
            if match:
                indent, key, values = match.group(1), match.group(2), match.group(3)
                if key == "ImpactGPU":
                    line = f"{indent}(ImpactGPU {device_index})"
                else:
                    exact_value = next((value for k, value in exact if k == key), None)
                    if exact_value is not None:
                        seen.add(key)
                        line = f"{indent}({key} {exact_value})"
                    else:
                        token_key = "ImpactSubsetFeatures" if key.startswith("ImpactSubsetFeatures") else key
                        if token_key in per_token:
                            seen.add(token_key)
                            replaced = " ".join(per_token[token_key] for _ in values.split())
                            line = f"{indent}({key} {replaced})"
            lines.append(line)
        # Overrides never inject keys, so a knob set for a key absent from every map silently does nothing —
        # surface it (e.g. final_grid_spacing on a rigid-only preset).
        for key in sorted(requested - seen):
            print(f"[ImpactReg] note: override '{key}' matched no entry in the preset's parameter maps.")
        return "\n".join(lines)

    def _stage_parameter_maps(self, work: Path, device_index: int) -> list[Path]:
        """Stage the parameter maps into ``work``.

        Matrix mode GENERATES each map from ``resolutions`` + the registry, then applies only the map-wide
        knobs (the matrix already sets iterations/features per cell). Legacy mode copies the preset's maps and
        applies every per-token / exact override. Both set the ImpactGPU device.
        """
        staged = []
        for src in self._parameter_maps:
            if self._resolutions:
                text = generate_impact_parameter_map(
                    src.read_text(encoding="utf-8"), self._resolutions, self._registry, self._mode
                )
                per_token, exact = self._parameter_map_overrides(global_only=True)
            else:
                text = src.read_text(encoding="utf-8")
                per_token, exact = self._parameter_map_overrides()
            text = self._apply_map_overrides(text, per_token, exact, device_index)
            dst = work / src.name
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Register ``moving`` onto ``fixed``; return (moved, dvf) as channel-first arrays on the fixed grid.

        Optional ``fixed_mask`` / ``moving_mask`` restrict the similarity metric to a region (elastix
        ``-fMask`` / ``-mMask``); a mask covering the whole image is equivalent to passing none.
        """
        work = Path(tempfile.mkdtemp(prefix="konfai_reg_"))
        try:
            fixed_path, moving_path = work / "Fixed.mha", work / "Moving.mha"
            sitk.WriteImage(fixed, str(fixed_path))
            sitk.WriteImage(moving, str(moving_path))

            # Stage the feature models at the relative path the maps reference (e.g. ImpactModelsPath0
            # "MIND/R1D2_3D.pt"), resolved from the elastix working directory.
            for rel_name, model_path in self._local_models:
                dst = work / rel_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(model_path)

            args = [str(self._elastix_bin), "-f", str(fixed_path), "-m", str(moving_path)]
            for flag, mask, name in (("-fMask", fixed_mask, "FixedMask.mha"), ("-mMask", moving_mask, "MovingMask.mha")):
                if _is_partial_mask(mask):
                    mask_path = work / name
                    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(mask_path))
                    args += [flag, str(mask_path)]
            args += ["-out", str(work)]
            for pmap in self._stage_parameter_maps(work, device_index):
                args += ["-p", str(pmap)]

            # The IMPACT metric plugin links LibTorch from the environment's pip ``torch`` (its ``lib/`` dir) --
            # the same LibTorch the elastix asset is built against in CI. ``<install>/lib`` (the elastix runtime)
            # and any extra dirs (KONFAI_ELASTIX_EXTRA_LIB) are also searched; on Windows the loader reads PATH.
            import torch

            env = os.environ.copy()
            torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
            extra_libs = [
                str(self._elastix_bin.parent.parent / "lib"),
                torch_lib,
                os.environ.get("KONFAI_ELASTIX_EXTRA_LIB", ""),
            ]
            lib_var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[lib_var] = os.pathsep.join(p for p in [*extra_libs, env.get(lib_var, "")] if p)
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
            # progress line) shows real progress. A tuned max_iterations makes the declared budget stale ->
            # open-ended bar. The description mirrors KonfAI's bars: resolution level + the metric value.
            captured: list[str] = []
            iteration_line = re.compile(r"^\d+\s")
            budget = None if self._max_iterations > 0 else (self._iterations or None)
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
                raise RuntimeError(f"elastix failed (code {returncode}):\n{''.join(captured[-40:])}")

            transforms = sorted(
                work.glob("TransformParameters.*-Composite.itk.txt"),
                key=lambda p: int(p.name.split(".")[1].split("-")[0]),
            )
            if not transforms:
                raise FileNotFoundError("elastix produced no composite transform file.")
            transform = sitk.ReadTransform(str(transforms[-1]))

            moved = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
            dvf = sitk.TransformToDisplacementField(
                transform,
                sitk.sitkVectorFloat64,
                fixed.GetSize(),
                fixed.GetOrigin(),
                fixed.GetSpacing(),
                fixed.GetDirection(),
            )
            moved_np, _ = image_to_data(moved)
            dvf_np, _ = image_to_data(dvf)
            return moved_np, dvf_np
        finally:
            shutil.rmtree(work, ignore_errors=True)


class ElastixRegistration(torch.nn.Module):
    """Custom graph module: (fixed, moving) tensors + their geometry -> moved image on the fixed grid.

    ``accepts_attributes = True`` opts this module into receiving, from the KonfAI graph, the per-branch
    ``Attribute`` list alongside the tensors (same convention as ``CriterionWithAttribute``). elastix needs
    the physical geometry (Origin/Spacing/Direction), which raw tensors do not carry.
    """

    accepts_attributes = True

    def __init__(
        self,
        engine: str,
        parameter_maps: list[str],
        max_iterations: int = 0,
        final_grid_spacing: float = 0.0,
        subset_features: int = 0,
        spatial_samples: int = 0,
        parameter_overrides: list[str] = [],
        resolutions: dict = {},
        mode: str = "Static",
    ) -> None:
        super().__init__()
        if engine != "elastix":
            raise NotImplementedError(f"ElastixRegistration engine '{engine}' is not implemented yet.")
        self._engine = ElastixEngine(
            parameter_maps,
            max_iterations,
            final_grid_spacing,
            subset_features,
            spatial_samples,
            parameter_overrides,
            resolutions,
            mode,
        )

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        attributes: list[list[Attribute]],
    ) -> torch.Tensor:
        # attributes = [fixed, moving, fixed_mask, moving_mask] branch attrs; each a list[Attribute] over the
        # batch. Returns, per sample, the moved image (1 channel) stacked with the DVF (dim channels), both on
        # the fixed grid; downstream ChannelSelect splits them. A whole-image mask (the default) restricts nothing.
        fixed_attrs, moving_attrs, fmask_attrs, mmask_attrs = attributes
        device_index = fixed.device.index if fixed.device.type == "cuda" else -1
        combined = []
        for b in range(fixed.shape[0]):
            fixed_img = data_to_image(fixed[b].detach().cpu().numpy(), fixed_attrs[b])
            moving_img = data_to_image(moving[b].detach().cpu().numpy(), moving_attrs[b])
            fixed_mask_img = data_to_image(fixed_mask[b].detach().cpu().numpy(), fmask_attrs[b])
            moving_mask_img = data_to_image(moving_mask[b].detach().cpu().numpy(), mmask_attrs[b])
            moved_np, dvf_np = self._engine.register(
                fixed_img, moving_img, device_index, fixed_mask_img, moving_mask_img
            )
            combined.append(torch.from_numpy(np.concatenate([moved_np, dvf_np], axis=0)))
        return torch.stack(combined, dim=0).to(fixed.device)
