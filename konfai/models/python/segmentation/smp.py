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

"""Encoder-backed 2D segmentation models from ``segmentation_models_pytorch`` (SMP).

One thin KonfAI wrapper over SMP's whole architecture/encoder zoo: any SMP architecture
(Unet, UnetPlusPlus, MAnet, FPN, DeepLabV3Plus, PSPNet, ...) combined with any of its
encoders (resnet*, efficientnet*, timm-*, ...), optionally with pretrained encoder
weights (``encoder_weights: imagenet``). Reference it from a config as
``classpath: segmentation.smp.SMP``.
"""

from konfai.data.patching import ModelPatch
from konfai.network import network
from konfai.utils.config import config
from konfai.utils.errors import ConfigError


def _require_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ConfigError(
            "segmentation_models_pytorch is required for the SMP model wrapper.",
            "Install it with `pip install konfai[smp]`.",
        ) from None
    return smp


@config()
class SMP(network.Network):
    """Any ``segmentation_models_pytorch`` architecture/encoder pair as a KonfAI model.

    ``load`` never re-initialises: SMP performs its own initialisation and the encoder may
    carry pretrained weights (``encoder_weights='imagenet'``) — the trainer's
    ``load(init=True)`` would silently destroy them. Checkpoint loading is unaffected.
    """

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        patch: ModelPatch | None = None,
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        nb_batch_per_step: int = 1,
        dim: int = 2,
        arch: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        in_channels: int = 1,
        classes: int = 2,
    ) -> None:
        if dim != 2:
            raise ConfigError(
                f"SMP architectures are 2D-only (got dim={dim}).",
                "Use dim: 2 (slice-wise or 2.5D via extra input channels), or pick a 3D model "
                "(e.g. 'classpath: default|DynUNet.yml').",
            )
        super().__init__(
            in_channels=in_channels,
            optimizer=optimizer,
            schedulers=schedulers,
            patch=patch,
            outputs_criterions=outputs_criterions,
            nb_batch_per_step=nb_batch_per_step,
            dim=dim,
        )
        smp = _require_smp()
        self.add_module(
            "Model",
            smp.create_model(
                arch=arch,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            ),
        )

    def load(
        self,
        state_dict: dict,
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ):
        del init  # SMP owns its initialisation (possibly pretrained encoder)
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)
