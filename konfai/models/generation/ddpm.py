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

from collections.abc import Callable

import numpy as np
import torch
import tqdm
from konfai.data.patching import ModelPatch
from konfai.metric.measure import Criterion
from konfai.network import blocks, network
from konfai.utils.runtime import gpu_info


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPM(network.Network):
    class DDPMTE(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear_0 = torch.nn.Linear(in_channels, out_channels)
            self.siLU = torch.nn.SiLU()
            self.linear_1 = torch.nn.Linear(out_channels, out_channels)

        def forward(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return tensor + self.linear_1(self.siLU(self.linear_0(t))).reshape(
                tensor.shape[0], -1, *[1 for _ in range(len(tensor.shape) - 2)]
            )

    class DDPMUNetBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int],
            nb_conv_per_stage: int,
            block_config: blocks.BlockConfig,
            downsample_mode: blocks.DownsampleMode,
            upsample_mode: blocks.UpsampleMode,
            attention: bool,
            time_embedding_dim: int,
            dim: int,
            i: int = 0,
        ) -> None:
            super().__init__()
            if i > 0:
                self.add_module(
                    downsample_mode.name,
                    blocks.downsample(
                        in_channels=channels[0],
                        out_channels=channels[1],
                        downsample_mode=downsample_mode,
                        dim=dim,
                    ),
                )
            self.add_module(
                "Te_down",
                DDPM.DDPMTE(
                    time_embedding_dim,
                    channels[(1 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i > 0 else 0)],
                ),
                in_branch=[0, 1],
            )
            self.add_module(
                "DownConvBlock",
                blocks.ResBlock(
                    in_channels=channels[(1 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i > 0 else 0)],
                    out_channels=channels[1],
                    block_configs=[block_config] * nb_conv_per_stage,
                    dim=dim,
                ),
            )
            if len(channels) > 2:
                self.add_module(
                    f"UNetBlock_{i + 1}",
                    DDPM.DDPMUNetBlock(
                        channels[1:],
                        nb_conv_per_stage,
                        block_config,
                        downsample_mode,
                        upsample_mode,
                        attention,
                        time_embedding_dim,
                        dim,
                        i + 1,
                    ),
                    in_branch=[0, 1],
                )
                self.add_module(
                    "Te_up",
                    DDPM.DDPMTE(
                        time_embedding_dim,
                        (
                            (channels[1] + channels[2])
                            if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE
                            else channels[1] * 2
                        ),
                    ),
                    in_branch=[0, 1],
                )
                self.add_module(
                    "UpConvBlock",
                    blocks.ResBlock(
                        in_channels=(
                            (channels[1] + channels[2])
                            if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE
                            else channels[1] * 2
                        ),
                        out_channels=channels[1],
                        block_configs=[block_config] * nb_conv_per_stage,
                        dim=dim,
                    ),
                )
            if i > 0:
                if attention:
                    self.add_module(
                        "Attention",
                        blocks.Attention(f_g=channels[1], f_l=channels[0], f_int=channels[0], dim=dim),
                        in_branch=[2, 0],
                        out_branch=[2],
                    )
                self.add_module(
                    upsample_mode.name,
                    blocks.upsample(
                        in_channels=channels[1],
                        out_channels=channels[0],
                        upsample_mode=upsample_mode,
                        dim=dim,
                    ),
                )
                self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, 2])

    class DDPMUNetHead(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

    class DDPMForwardProcess(torch.nn.Module):
        def __init__(
            self,
            noise_step: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
        ) -> None:
            super().__init__()
            self.betas = torch.linspace(beta_start, beta_end, noise_step)
            self.betas = DDPM.DDPMForwardProcess.enforce_zero_terminal_snr(self.betas)
            self.alphas = 1 - self.betas
            self.alpha_hat = torch.concat((torch.ones(1), torch.cumprod(self.alphas, dim=0)))

        @staticmethod
        def enforce_zero_terminal_snr(betas: torch.Tensor):
            alphas = 1 - betas
            alphas_bar = alphas.cumprod(0)
            alphas_bar_sqrt = alphas_bar.sqrt()
            alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
            alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
            alphas_bar_sqrt -= alphas_bar_sqrt_t
            alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)
            alphas_bar = alphas_bar_sqrt**2
            alphas = alphas_bar[1:] / alphas_bar[:-1]
            alphas = torch.cat([alphas_bar[0:1], alphas])
            betas = 1 - alphas
            return betas

        def forward(self, tensor: torch.Tensor, t: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
            alpha_hat_t = (
                self.alpha_hat[t.cpu()]
                .to(tensor.device)
                .reshape(tensor.shape[0], *[1 for _ in range(len(tensor.shape) - 1)])
            )
            result = alpha_hat_t.sqrt() * tensor + (1 - alpha_hat_t).sqrt() * eta
            return result

    class DDPMSampleT(torch.nn.Module):
        def __init__(self, noise_step: int) -> None:
            super().__init__()
            self.noise_step = noise_step

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.randint(0, self.noise_step, (tensor.shape[0],)).to(tensor.device)

    class DDPMTimeEmbedding(torch.nn.Module):
        @staticmethod
        def sinusoidal_embedding(noise_step: int, time_embedding_dim: int):
            noise_step += 1
            embedding = torch.zeros(noise_step, time_embedding_dim)
            wk = torch.tensor([1 / 10_000 ** (2 * j / time_embedding_dim) for j in range(time_embedding_dim)])
            wk = wk.reshape((1, time_embedding_dim))
            t = torch.arange(noise_step).reshape((noise_step, 1))
            embedding[:, ::2] = torch.sin(t * wk[:, ::2])
            embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
            return embedding

        def __init__(self, noise_step: int = 1000, time_embedding_dim: int = 100) -> None:
            super().__init__()
            self.time_embed = torch.nn.Embedding(noise_step, time_embedding_dim)
            self.time_embed.weight.data = DDPM.DDPMTimeEmbedding.sinusoidal_embedding(noise_step, time_embedding_dim)
            self.time_embed.requires_grad_(False)
            self.noise_step = noise_step

        def forward(self, tensor: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return self.time_embed((p * self.noise_step).long().repeat(tensor.shape[0]))

    class DDPMUNet(network.ModuleArgsDict):
        def __init__(
            self,
            noise_step: int,
            channels: list[int],
            block_config: blocks.BlockConfig,
            nb_conv_per_stage: int,
            downsample_mode: str,
            upsample_mode: str,
            attention: bool,
            time_embedding_dim: int,
            dim: int,
        ) -> None:
            super().__init__()
            self.add_module(
                "t",
                DDPM.DDPMTimeEmbedding(noise_step, time_embedding_dim),
                in_branch=[1],
                out_branch=["te"],
            )
            self.add_module(
                "UNetBlock_0",
                DDPM.DDPMUNetBlock(
                    channels,
                    nb_conv_per_stage,
                    block_config,
                    downsample_mode=blocks.DownsampleMode[downsample_mode],
                    upsample_mode=blocks.UpsampleMode[upsample_mode],
                    attention=attention,
                    time_embedding_dim=time_embedding_dim,
                    dim=dim,
                ),
                in_branch=[0, "te"],
            )
            self.add_module(
                "Head",
                DDPM.DDPMUNetHead(in_channels=channels[1], out_channels=1, dim=dim),
            )

    class DDPMInference(torch.nn.Module):
        def __init__(
            self,
            model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            train_noise_step: int,
            inference_noise_step: int,
            beta_start: float,
            beta_end: float,
        ) -> None:
            super().__init__()
            self.model = model
            self.train_noise_step = train_noise_step
            self.inference_noise_step = inference_noise_step
            self.forwardProcess = DDPM.DDPMForwardProcess(train_noise_step, beta_start, beta_end)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            x = torch.randn_like(tensor).to(tensor.device)
            result = []
            result.append(x.unsqueeze(1))
            offset = self.train_noise_step // self.inference_noise_step
            t_list = np.round(np.arange(self.train_noise_step - 1, 0, -offset)).astype(int).tolist()
            with tqdm.tqdm(
                iterable=enumerate(t_list),
                desc="Inference : " + gpu_info(),
                total=len(t_list),
                leave=False,
                disable=True,
            ) as batch_iter:
                for _, t in batch_iter:
                    alpha_t_hat = self.forwardProcess.alpha_hat[t]
                    alpha_t_hat_prev = (
                        self.forwardProcess.alpha_hat[t - offset]
                        if t - offset >= 0
                        else self.forwardProcess.alpha_hat[0]
                    )
                    eta_theta = self.model(
                        torch.concat((x, tensor), dim=1),
                        (torch.ones(tensor.shape[0], 1) * t).to(tensor.device).long(),
                    )

                    predicted = (
                        (x - (1 - alpha_t_hat).sqrt() * eta_theta) / alpha_t_hat.sqrt() * alpha_t_hat_prev.sqrt()
                    )

                    variance = ((1 - alpha_t_hat_prev) / (1 - alpha_t_hat)) * (1 - alpha_t_hat / alpha_t_hat_prev)

                    direction = (1 - alpha_t_hat_prev - variance).sqrt() * eta_theta

                    x = predicted + direction
                    x += variance.sqrt() * torch.randn_like(tensor).to(tensor.device)

                    result.append(x.unsqueeze(1))

                    batch_iter.set_description("Inference : " + gpu_info())
            result[-1] = torch.clip(result[-1], -1, 1)
            return torch.concat(result, dim=1)

    class DDPMVLoss(torch.nn.Module):
        def __init__(self, alpha_hat: torch.Tensor) -> None:
            super().__init__()
            self.alpha_hat = alpha_hat

        def v(self, tensor: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
            alpha_hat_t = (
                self.alpha_hat[t.cpu()]
                .to(tensor.device)
                .reshape(tensor.shape[0], *[1 for _ in range(len(tensor.shape) - 1)])
            )
            return alpha_hat_t.sqrt() * tensor - (1 - alpha_hat_t).sqrt() * noise

        def forward(self, tensor: torch.Tensor, eta: torch.Tensor, eta_hat: torch.Tensor, t: int) -> torch.Tensor:
            return torch.concat((self.v(tensor, eta, t), self.v(tensor, eta_hat, t)), dim=1)

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        train_noise_step: int = 1000,
        inference_noise_step: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_embedding_dim: int = 100,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        dim: int = 3,
    ) -> None:
        # Experimental / not currently runnable: the time-embedding module is wired with a single
        # input but its forward needs two, and it expects a [0, 1] fraction while the graph feeds
        # integer timesteps — so no forward pass succeeds. Fail fast with a clear message instead of
        # an opaque crash deep in the graph, until the time-conditioning is fixed and validated.
        raise NotImplementedError(
            "DDPM is experimental and not currently runnable (its time-embedding wiring is broken and "
            "no forward pass succeeds). Do not select it in a config until it is fixed."
        )
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module("Noise", blocks.NormalNoise(), out_branch=["eta"], training=True)
        self.add_module(
            "Sample",
            DDPM.DDPMSampleT(train_noise_step),
            out_branch=["t"],
            training=True,
        )
        self.add_module(
            "Forward",
            DDPM.DDPMForwardProcess(train_noise_step, beta_start, beta_end),
            in_branch=[0, "t", "eta"],
            out_branch=["x_t"],
            training=True,
        )
        self.add_module(
            "Concat",
            blocks.Concat(),
            in_branch=["x_t", 1],
            out_branch=["xy_t"],
            training=True,
        )
        self.add_module(
            "UNet",
            DDPM.DDPMUNet(
                train_noise_step,
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                time_embedding_dim,
                dim,
            ),
            in_branch=["xy_t", "t"],
            out_branch=["eta_hat"],
            training=True,
        )

        self.add_module(
            "Noise_optim",
            DDPM.DDPMVLoss(self["Forward"].alpha_hat),
            in_branch=[0, "eta", "eta_hat", "t"],
            out_branch=["noise"],
            training=True,
        )

        self.add_module(
            "Inference",
            DDPM.DDPMInference(
                self.inference,
                train_noise_step,
                inference_noise_step,
                beta_start,
                beta_end,
            ),
            in_branch=[1],
            training=False,
        )
        self.add_module(
            "LastImage",
            blocks.Select([slice(None, None), slice(-1, None)]),
            training=False,
        )

    def inference(self, tensor: torch.Tensor, t: torch.Tensor):
        return self["UNet"](tensor, t)


class MSE(Criterion):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, tensor: torch.Tensor, *targets: list[torch.Tensor]) -> torch.Tensor:
        return self.loss(tensor[:, 0, ...], tensor[:, 1, ...])
