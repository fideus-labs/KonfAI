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

"""Point sets (landmarks, ``.fcsv``) moved by the transforms stored beside them."""

import torch

from konfai.data.transform.base import LocalityKind, PatchLocality, Transform
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError
from konfai.utils.ITK import displace_points


class Displace(Transform):
    """Points carried by the case's displacement field: each point ``x`` becomes ``x + u(x)``, ``u`` the case's entry
    in ``field_group`` (one component per channel, in the points' units along the world axes), read at ``x`` as
    SimpleITK's ``DisplacementFieldTransform`` reads it, from the voxels around each point alone
    (:func:`konfai.utils.ITK.displace_points`). A 2D field moves the first two coordinates."""

    def __init__(self, field_group: str = "DVF") -> None:
        super().__init__()
        self.field_group = field_group

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME, reason="a point set has no grid to cut into patches")

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        dataset = self.dataset_holding(self.field_group, name)
        if dataset is None:
            raise TransformError(
                f"'Displace' found no field of group '{self.field_group}' for case '{name}'.",
                "Name the group the fields are in: Displace: {field_group: DVF}.",
            )
        points = tensor.detach().cpu().numpy().reshape(-1, tensor.shape[-1])
        moved = displace_points(points, dataset, self.field_group, name)
        return torch.from_numpy(moved.reshape(tensor.shape)).to(device=tensor.device, dtype=tensor.dtype)
