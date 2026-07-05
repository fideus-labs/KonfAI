import torch
from konfai.network import network


class Head(network.ModuleArgsDict):
    def __init__(self):
        super().__init__()
        self.add_module("Tanh", torch.nn.Tanh())


class TinySynthNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {"default|ConstantLR": network.LRSchedulersLoader(0)},
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"Head:Tanh": network.TargetCriterionsLoader()},
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=2,
        )
        self.add_module(
            "Projection",
            torch.nn.Conv2d(1, 1, kernel_size=1, bias=True),
        )
        self.add_module("Head", Head())
