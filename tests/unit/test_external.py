"""An app that runs an external tool: what reaches the tool's engine, and what comes back."""

import numpy as np
import SimpleITK as sitk
import torch
from konfai.network import network
from konfai.network.external import ExternalNet, ExternalTool, scalar
from konfai.utils.dataset import Attribute


def _geometry() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.array([3.0, -1.0, 7.0])
    attribute["Spacing"] = np.array([0.5, 0.5, 2.0])
    attribute["Direction"] = np.eye(3).flatten()
    return attribute


def test_the_engine_gets_each_branch_as_an_image_with_its_geometry_and_a_mask_of_ones_as_none() -> None:
    seen = []

    def engine(image, mask, other_mask, device):
        seen.append((image, mask, other_mask, device))
        return sitk.GetArrayFromImage(image)[None] * 2

    volume = torch.rand(2, 1, 4, 5, 6)
    partial = (volume > 0.5).float()
    out = ExternalTool(engine, masks=(1, 2))(
        volume, torch.ones_like(volume), partial, attributes=[[_geometry()] * 2] * 3
    )

    image, mask, other_mask, device = seen[0]
    assert len(seen) == 2 and device == -1 and mask is None and other_mask is not None
    assert image.GetSpacing() == (0.5, 0.5, 2.0) and image.GetOrigin() == (3.0, -1.0, 7.0)
    assert torch.allclose(out, volume * 2)


def test_the_app_s_one_output_carries_the_task_s_name() -> None:
    net = ExternalNet(
        lambda image, mask, device: None,
        2,
        (1,),
        "Segmentation",
        3,
        network.OptimizerLoader(),
        {"default:ReduceLROnPlateau": network.LRSchedulersLoader(0)},
        {"default": network.TargetCriterionsLoader()},
    )
    assert isinstance(net["Segmentation"], ExternalTool) and net.full_precision_outputs


def test_a_colour_image_is_read_as_the_mean_of_its_components() -> None:
    colour = sitk.Compose([sitk.Image([3, 2], sitk.sitkUInt8) + v for v in (0, 30, 60)])
    assert (sitk.GetArrayFromImage(scalar(colour)) == 30).all()
