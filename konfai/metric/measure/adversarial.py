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


"""Adversarial, style and feature-distribution criteria."""

import copy
import os

import numpy as np
import torch

from konfai.metric.measure.base import Criterion, _require_optional, models_register
from konfai.network.network import ModelLoader, Network
from konfai.utils.config import apply_config
from konfai.utils.utils import get_module


class PatchGanLoss(Criterion):
    def __init__(self, target: float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.register_buffer("target", torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, (torch.ones_like(output) * target).to(output.device))


class WGP(Criterion):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((output - 1) ** 2)


class Gram(Criterion):
    @staticmethod
    def compute_gram(tensor: torch.Tensor):
        (_b, ch, w) = tensor.size()
        with torch.amp.autocast("cuda", enabled=False):
            return tensor.bmm(tensor.transpose(1, 2)).div(ch * w)

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction="sum")

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = targets[0]
        if len(output.shape) > 3:
            output = output.view(output.shape[0], output.shape[1], int(np.prod(output.shape[2:])))
        if len(target.shape) > 3:
            target = target.view(target.shape[0], target.shape[1], int(np.prod(target.shape[2:])))
        return self.loss(Gram.compute_gram(output), Gram.compute_gram(target))


class PerceptualLoss(Criterion):
    class Module:
        def __init__(self, losses: dict[str, float] = {"Gram": 1, "torch:nn:L1Loss": 1}) -> None:
            self.losses = losses
            self.konfai_args = os.environ["KONFAI_CONFIG_PATH"] if "KONFAI_CONFIG_PATH" in os.environ else ""

        def get_loss(self) -> dict[torch.nn.Module, float]:
            result: dict[torch.nn.Module, float] = {}
            for loss, loss_value in self.losses.items():
                module, name = get_module(loss, "konfai.metric.measure")
                result[apply_config(self.konfai_args)(getattr(module, name))()] = loss_value
            return result

    def __init__(
        self,
        model_loader: ModelLoader = ModelLoader(),
        path_model: str = "name",
        modules: dict[str, Module] = {
            "UNetBlock_0.DownConvBlock.Activation_1": Module({"Gram": 1, "torch:nn:L1Loss": 1})
        },
        shape: list[int] = [128, 128, 128],
    ) -> None:
        super().__init__()
        self.path_model = path_model
        if self.path_model not in models_register:
            self.model = model_loader.get_model(
                train=False,
                konfai_args=os.environ["KONFAI_CONFIG_PATH"].split("PerceptualLoss")[0] + "PerceptualLoss.Model",
                konfai_without=[
                    "optimizer",
                    "schedulers",
                    "nb_batch_per_step",
                    "init_type",
                    "init_gain",
                    "outputs_criterions",
                    "drop_p",
                ],
            )
            if path_model.startswith("https"):
                state_dict = torch.hub.load_state_dict_from_url(path_model)
                state_dict = {"Model": {self.model.get_name(): state_dict["model"]}}
            else:
                state_dict = torch.load(path_model, weights_only=True)
            self.model.load(state_dict)
            models_register[self.path_model] = self.model
        else:
            self.model = models_register[self.path_model]

        self.shape = shape
        self.mode = "trilinear" if len(shape) == 3 else "bilinear"
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}
        for name, losses in modules.items():
            self.modules_loss[name.replace(":", ".")] = losses.get_loss()

        self.model.eval()
        self.model.requires_grad_(False)
        self.models: dict[int, torch.nn.Module] = {}

    def preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _compute(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        output_preprocessing = self.preprocessing(output)
        targets_preprocessing = [self.preprocessing(target) for target in targets]
        for zipped_output in zip([output_preprocessing], *[[target] for target in targets_preprocessing], strict=False):
            output = zipped_output[0]
            targets = zipped_output[1:]

            for zipped_layers in list(
                zip(
                    self.models[output.device.index].get_layers([output], set(self.modules_loss.keys()).copy()),
                    *[
                        self.models[output.device.index].get_layers([target], set(self.modules_loss.keys()).copy())
                        for target in targets
                    ],
                    strict=False,
                )
            ):
                output_layer = zipped_layers[0][1].view(
                    zipped_layers[0][1].shape[0],
                    zipped_layers[0][1].shape[1],
                    int(np.prod(zipped_layers[0][1].shape[2:])),
                )
                # Apply every configured loss to every target layer. Zipping the losses against the
                # targets instead drops losses whenever there are fewer targets than losses: the
                # default {Gram, L1Loss} on a single reference would silently use only Gram.
                for target_entry in zipped_layers[1:]:
                    target_layer = target_entry[1].view(
                        target_entry[1].shape[0],
                        target_entry[1].shape[1],
                        int(np.prod(target_entry[1].shape[2:])),
                    )
                    for loss_function, loss_value in self.modules_loss[zipped_layers[0][0]].items():
                        loss = (
                            loss
                            + loss_value
                            * loss_function(output_layer.float(), target_layer.float())
                            / output_layer.shape[0]
                        )
        return loss

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        if output.device.index not in self.models:
            # `Network.to` resets its GPU-index counter per call, so the perceptual model is
            # placed starting at this device.
            self.models[output.device.index] = Network.to(copy.deepcopy(self.model).eval(), output.device.index).eval()
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and len(self.shape) == 2:
            for i in range(output.shape[2]):
                loss = loss + self._compute(output[:, :, i, ...], *[t[:, :, i, ...] for t in targets]) / output.shape[2]
        else:
            loss = self._compute(output, *targets)
        return loss.to(output)


class FID(Criterion):
    class InceptionV3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            torchvision_models = _require_optional("torchvision.models", criterion="FID", extra="fid")
            inception_v3 = torchvision_models.inception_v3
            Inception_V3_Weights = torchvision_models.Inception_V3_Weights

            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            self.model.fc = torch.nn.Identity()
            self.model.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    def __init__(self) -> None:
        super().__init__()
        _require_optional("scipy.linalg", criterion="FID", extra="fid")
        # Built on the CPU and moved to the evaluated tensor's device in forward: a hardcoded .cuda()
        # crashes CPU-only hosts and pins every DDP rank to the same GPU.
        self.inception_model = FID.InceptionV3()

    @staticmethod
    def preprocess_images(image: torch.Tensor) -> torch.Tensor:
        # resize/normalise-with-mean-std live in torchvision.transforms.functional, not torch.nn.functional
        # (which has no ``resize`` and whose ``normalize`` takes no mean/std).
        tvf = _require_optional("torchvision.transforms.functional", criterion="FID", extra="fid")
        resized = tvf.resize(image, [299, 299]).repeat((1, 3, 1, 1))
        return tvf.normalize(resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def get_features(images: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
        with torch.no_grad():
            features = model(images).cpu().numpy()
        return features

    @staticmethod
    def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        mu2 = np.mean(generated_features, axis=0)
        sigma2 = np.cov(generated_features, rowvar=False)

        diff = mu1 - mu2
        linalg = _require_optional("scipy.linalg", criterion="FID", extra="fid")

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        self.inception_model.to(output.device)
        real_images = FID.preprocess_images(targets[0].to(output.device).squeeze(0).permute([1, 0, 2, 3]))
        generated_images = FID.preprocess_images(output.squeeze(0).permute([1, 0, 2, 3]))

        real_features = FID.get_features(real_images, self.inception_model)
        generated_features = FID.get_features(generated_images, self.inception_model)

        return FID.calculate_fid(real_features, generated_features)
