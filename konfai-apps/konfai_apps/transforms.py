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


"""A KonfAI app run as a chain stage.

Published configs spell the stage by its bare name (``KonfAIInference:``); core's transform package
resolves that name to this class when ``konfai-apps`` is installed.
"""

import os
import tempfile
from multiprocessing import current_process, get_context
from pathlib import Path

import SimpleITK as sitk
import torch
from konfai import cuda_visible_devices
from konfai.data.transform import Transform
from konfai.utils.dataset import Attribute, data_to_image, image_to_data

# Published app used by KonfAIInference when the configuration leaves repo/model unset.
DEFAULT_INFERENCE_REPO_ID = "VBoussot/MRSegmentator-KonfAI"
DEFAULT_INFERENCE_MODEL_NAME = "MRSegmentator"


class KonfAIInference(Transform):
    """Run a nested KonfAI app inference on the case, as one stage of a chain.

    Whole-volume by construction: the tensor is written to a temporary ``.mha``, a spawned process
    resolves the app (``konfai-apps``, a HuggingFace repo by default) and loads the model, and the
    output is read back. That happens once per case, so a cohort loads the model as many times as it
    has cases; its GPU/RAM usage lives outside the ``memory_budget`` a TRANSFORM plan bounds. Inference
    over a cohort is PREDICTION's job; this stage is for an inference that feeds a later stage.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_INFERENCE_REPO_ID,
        model_name: str = DEFAULT_INFERENCE_MODEL_NAME,
        checkpoints_name: list[str] = ["fold_0"],
        number_of_tta: int = 0,
        number_of_mc: int = 0,
        per_channel: bool = False,
        config_overrides: list[str] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.model_name = model_name
        self.checkpoints_name = checkpoints_name
        self.number_of_tta = number_of_tta
        self.number_of_mc = number_of_mc
        self.per_channel = per_channel
        # Generic 'NAME=VALUE' overrides for the nested run's own config (the --set mechanism), so a caller
        # can tune it without editing the bundle, not a memory workaround (never shrink a trained
        # segmentation's patch_size: it degrades the result; the allocator hint below keeps memory in check).
        self.config_overrides = config_overrides

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        del name, shape, cache_attribute
        return (
            f"chain '{group_dest}' runs a NESTED KonfAI inference: its GPU and RAM usage live"
            " outside the declared memory_budget, and the plan cannot bound them"
        )

    def infer_entry(self, dataset_path: Path, output_path: Path, gpu: list[int]):
        # Defragment the nested run's CUDA allocator: a heavy model (e.g. a 3D segmentation a metric relies
        # on) can OOM on a large volume purely from reserved-but-unallocated blocks even though the live
        # footprint fits, so it runs at its trained patch_size, not a shrunk one. setdefault so an
        # explicit caller setting still wins.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from konfai_apps import KonfAIApp

        # Nested KonfAI runs must choose their own rendezvous ports instead of
        # inheriting the parent's already-bound distributed settings.
        os.environ.pop("KONFAI_MASTER_PORT", None)
        os.environ.pop("KONFAI_TENSORBOARD_PORT", None)

        konfai_app = KonfAIApp(f"{self.repo_id}:{self.model_name}", False, False)
        konfai_app.infer(
            [[dataset_path]],
            output_path,
            0,
            self.checkpoints_name,
            self.number_of_tta,
            mc=0,
            config_overrides=self.config_overrides,
            uncertainty=False,
            gpu=gpu,
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if current_process().daemon:
            raise RuntimeError(
                "KonfAIInference cannot run inside daemon DataLoader workers. "
                "Use 'Dataset.num_workers: 0' for pipelines that include this transform."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "Dataset"
            if self.per_channel:
                for i, channel in enumerate(tensor):
                    image = data_to_image(channel.unsqueeze(0), cache_attribute)
                    (dataset_path / f"P{i:03d}").mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(image, str(dataset_path / f"P{i:03d}" / "Volume.mha"))
            else:
                image = data_to_image(tensor, cache_attribute)

                (dataset_path / "P000").mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(image, str(dataset_path / "P000" / "Volume.mha"))

            ctx = get_context("spawn")

            # Release the caller's cached GPU blocks so the nested run (its own process, same physical
            # device) is not squeezed by memory this process is only holding in reserve.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # The nested run gets THIS rank's device, not every device the launch was given: under
            # --gpu 0 1 each rank would otherwise spawn a two-GPU prediction of its own. The ordinal
            # is the one the tensor sits on: a TRANSFORM chain routes its tensors to the rank's
            # device without moving the process's own default, so current_device() can read 0 there.
            visible = cuda_visible_devices()
            devices = visible
            if visible and torch.cuda.is_available():
                ordinal = tensor.device.index if tensor.device.type == "cuda" else torch.cuda.current_device()
                if ordinal is not None and 0 <= ordinal < len(visible):
                    devices = [visible[ordinal]]
            p = ctx.Process(target=self.infer_entry, args=(dataset_path, Path(tmpdir) / "Output", devices))
            p.start()
            p.join()

            if p.exitcode != 0:
                raise RuntimeError("Inference process failed")

            return self._reassemble_output(Path(tmpdir) / "Output")

    @staticmethod
    def _reassemble_output(output_dir: Path) -> torch.Tensor:
        result = []
        for file in sorted(output_dir.rglob("*.mha")):
            if file.name != "InferenceStack.mha":
                result.append(torch.from_numpy(image_to_data(sitk.ReadImage(str(file)))[0]))
        return torch.stack(result, dim=1).squeeze(0)
