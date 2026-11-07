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

"""The stack of an ensemble: its members' channels member-major, its member count in its header, whatever a member's
channel count (a displacement field has three)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("SimpleITK")

from konfai.data.transform import InferenceStack, Magnitude, SegmentationDisagreement, StandardDeviation
from konfai.data.transform.ensemble import MEMBERS_ATTRIBUTE, stack_members
from konfai.utils.dataset import Attribute, Dataset


def _attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.array([1.0, 2.0, 3.0])
    attributes["Spacing"] = np.array([1.5, 1.0, 2.0])
    attributes["Direction"] = np.eye(3).ravel()
    return attributes


@pytest.mark.parametrize("fmt", ["mha", "nrrd"])
def test_a_stack_of_fields_keeps_each_member_s_components(tmp_path: Path, fmt: str) -> None:
    """Three members of a three-component field: nine channels, member-major, and the count in the header, so the
    reader splits members from components; the output (their mean) carries no count."""
    members = torch.randn(3, 3, 4, 5, 6)
    attributes = _attributes()
    mean = InferenceStack(dataset=f"{tmp_path}:{fmt}")("P000", members, attributes)

    torch.testing.assert_close(mean, members.mean(0))
    assert MEMBERS_ATTRIBUTE not in attributes
    data, header = Dataset(str(tmp_path), fmt).read_data("InferenceStack", "P000")
    assert list(data.shape) == [9, 4, 5, 6]
    torch.testing.assert_close(stack_members(torch.from_numpy(data), header), members, rtol=0, atol=1e-6)


def test_the_spread_of_a_stack_of_fields_is_the_spread_of_its_vectors() -> None:
    """StandardDeviation then Magnitude: the per-component standard deviation across the members, then its norm, the
    root mean square distance of the members to their mean vector (N - 1). Two members that agree in length but not in
    direction spread; the standard deviation of their lengths read 0."""
    members = torch.zeros(2, 3, 2, 2, 2)
    members[0, 0], members[1, 1] = 5.0, 5.0  # same length, perpendicular
    stack = members.reshape(6, 2, 2, 2)
    header = Attribute()
    header[MEMBERS_ATTRIBUTE] = 2
    spread = Magnitude()("P000", StandardDeviation()("P000", stack, header), header)

    assert list(spread.shape) == [1, 2, 2, 2]
    torch.testing.assert_close(spread, torch.full((1, 2, 2, 2), float(np.hypot(*members[:, :2, 0, 0, 0].std(0)))))
    assert MEMBERS_ATTRIBUTE not in header


def test_a_stack_without_a_count_is_one_member_a_channel() -> None:
    """A stack written before the count (or by hand) keeps its meaning: every channel a member."""
    stack = torch.tensor([[[1.0]], [[3.0]]])[:, :, :, None]  # two one-channel members
    torch.testing.assert_close(StandardDeviation()("P000", stack, Attribute()), stack.std(0)[None])
    labels = torch.tensor([1, 1, 2])[:, None, None, None]
    assert float(SegmentationDisagreement()("P000", labels, Attribute()).flatten()[0]) == pytest.approx(1 / 3)


def test_copies_folded_one_at_a_time_average_their_members(tmp_path: Path) -> None:
    """Two TTA copies of two members of three channels: the stack holds the four members, and the mean divides by
    four, not by the twelve channel rows written."""
    copies = [torch.randn(2, 3, 4, 5, 6), torch.randn(2, 3, 4, 5, 6)]
    stage = InferenceStack(dataset=f"{tmp_path}:mha")
    mean = stage.fold_copies("P000", iter(copies), 2, _attributes())

    torch.testing.assert_close(mean, torch.cat(copies).mean(0))
    data, header = Dataset(str(tmp_path), "mha").read_data("InferenceStack", "P000")
    assert list(data.shape) == [12, 4, 5, 6] and int(float(header[MEMBERS_ATTRIBUTE])) == 4


def test_a_stack_of_fields_is_refused_where_its_count_is_lost(tmp_path: Path) -> None:
    """NIfTI keeps no free header: nine channels would read as nine members, so a stack of fields is refused there."""
    from konfai.utils.errors import TransformError

    with pytest.raises(TransformError, match="NIfTI"):
        InferenceStack(dataset=f"{tmp_path}:nii.gz")("P000", torch.randn(2, 3, 4, 5, 6), _attributes())
