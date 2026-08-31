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

"""What a stage HOLDS against what it DECLARES, for every built-in transform.

``Transform.working_multiple`` is the one figure every sizing route reads: the sweep prices a region
with it, the reduction charges the member chain by it, and the whole-volume fallback is sized against
it. A stage that holds more than it declares is a region sized for less than it takes, on every route
at once -- and nothing checked it. An audit of the 39 built-ins found nine wrong, including the two
that declared a non-zero figure at all: ``Resample`` at 3.0 holding 21, ``Dilate`` at 3.0 holding 15.

The allocator reports the peak exactly, so the declaration is checked rather than argued. The
configurations come from the same registry the locality contract enumerates, so a stage is covered
the day it lands.
"""

import contextlib

import numpy as np
import pytest
import torch
from konfai.data.transform import Resample, Transform
from konfai.utils.budget import set_per_rank_budget
from konfai.utils.dataset import Attribute
from konfai.utils.errors import KonfAIError
from oracle_support import CASE_NAME, FIXED_GEOMETRY, attributes, builtin_transforms, cases_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the allocator that reports a peak exactly is CUDA's"
)

#: Big enough that a fixed-size scratch buffer does not dominate the ratio (a stage reads 1.50 at
#: 8^3 and 1.00 at 24^3 and above), small enough that the sweep of every configuration is quick.
_SPATIAL = (32, 96, 96)
#: What a declaration may be under by, in volumes-worth, before it stops being a bound. Allocator
#: rounding and a stage's own constants live in here; a real under-declaration is far larger than it
#: (the audit's smallest was 1.00 against 0.0).
_SLACK = 0.25
#: A budget large enough not to clamp any stage at this size: the slope is what is being checked.
_BUDGET = 16 << 30

#: Configurations whose hold is bounded by the BUDGET rather than by the region, so no multiple of
#: the region describes them. ``Resample`` through a map that does not factorise is walked coordinate
#: by coordinate in float64 (21.4 volumes-worth measured, against 0.19 to 2.85 when it factorises),
#: and that walk slabs itself against the declared budget: charging the region for it as well would
#: shrink every resampling chain sevenfold for memory the walk does not take.
_SELF_BOUNDED = {(Resample, "Oblique")}


#: What one payload and a seeded ``Attribute`` cannot drive, and what each asks for that is not
#: here. Named rather than caught: a blanket ``except Exception`` read a stage that broke on its
#: own payload as a covered one, and five stages were being skipped without anyone choosing it.
_NEEDS_MORE_THAN_A_PAYLOAD = {
    "HistogramMatching": "a reference dataset to match the histogram against",
    "KonfAIInference": "a model to run",
    "Resample": "a reference dataset to resample onto",
}


#: Groups whose values are labels whatever the store: a small integer range, never CT units.
_LABELISH = ("Label", "Labels", "Mask", "Segmentation", "Ensemble")


def _payload(group: str, stored: torch.dtype) -> torch.Tensor:
    """One case's tensor, as ``stored`` serves it.

    Measured on BOTH float32 and the integer a real store holds, because a stage widens an integer
    input and then works in float: measured on float32 alone, eleven stages certified a figure they
    hold twice over on a CT. A CT and an MR are int16, a label map uint8, and neither is exotic.
    """
    channels = 3 if group in ("Ensemble", "Field", "Vector") else 1
    device = torch.device("cuda:0")
    if stored is torch.float32:
        if group in ("Label", "Labels", "Mask", "Segmentation"):
            return torch.randint(0, 4, (channels, *_SPATIAL), dtype=torch.uint8, device=device)
        return torch.rand((channels, *_SPATIAL), device=device)
    low, high = (0, 5) if group in _LABELISH else (-1024, 3072)
    return torch.randint(low, high, (channels, *_SPATIAL), dtype=torch.int16, device=device)


#: The dtypes a store serves, both measured: the declaration is the worst of them.
_STORED_DTYPES = (torch.float32, torch.int16)


def _seeded(group: str, channels: int) -> Attribute:
    """The group's metadata, plus the whole-volume statistics a GLOBAL_STAT stage reads."""
    attribute = Attribute(attributes(FIXED_GEOMETRY, group))
    # Each in the form its producers write, because its readers parse that form and no other:
    # Mean and Std per channel (``case_reduction`` and ``Standardize`` write a one-element array
    # even at one channel, and ``get_tensor`` reads it), Min and Max as bare scalars (both
    # ``case_reduction`` and ``Normalize`` itself write one, and ``float()`` reads it).
    for key, value in (("Mean", 0.5), ("Std", 0.25)):
        attribute[key] = np.asarray([value] * channels)
    for key, value in (("Min", 0.0), ("Max", 1.0)):
        attribute[key] = np.float32(value)
    return attribute


def _held(stage: Transform, group: str, stored: torch.dtype) -> float | None:
    """Volumes-worth held above the input and the output, or None when the stage cannot run here.

    Two ways it cannot: a DESIGNED refusal, and one of the stages :data:`_NEEDS_MORE_THAN_A_PAYLOAD`
    names. Anything else -- an allocation that failed, a stage that broke on its own payload -- is
    the finding this test exists for, so it is left to escape.
    """
    tensor = _payload(group, stored)
    scope = _seeded(group, int(tensor.shape[0]))
    with contextlib.suppress(KonfAIError):
        stage.transform_shape("CT", CASE_NAME, list(_SPATIAL), Attribute(scope))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        out = stage(CASE_NAME, tensor, scope)
    except KonfAIError:
        return None
    except Exception:
        if type(stage).__name__ in _NEEDS_MORE_THAN_A_PAYLOAD:
            return None
        raise
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    given = tensor.numel() * tensor.element_size()
    # An in-place stage hands back what it was given: there is no second buffer to discount.
    same = isinstance(out, torch.Tensor) and out.data_ptr() == tensor.data_ptr()
    landed = 0 if same else (out.numel() * out.element_size() if isinstance(out, torch.Tensor) else 0)
    return max(0.0, (peak - base - landed) / max(given, landed or given))


@pytest.mark.parametrize("cls", builtin_transforms(), ids=lambda cls: cls.__name__)
def test_a_stage_holds_no_more_than_it_declares(cls: type[Transform]) -> None:
    set_per_rank_budget(_BUDGET)
    try:
        exercised = []
        for case in cases_of(cls):
            if (cls, case.group) in _SELF_BOUNDED:
                continue
            for stored in _STORED_DTYPES:
                held = _held(case.transform, case.group, stored)
                if held is None:
                    continue
                exercised.append((case.group, held))
                declared = float(case.transform.case_working_multiple(CASE_NAME))
                assert held <= declared + _SLACK, (
                    f"{cls.__name__} on '{case.group}' stored as {stored} holds {held:.2f}"
                    f" volumes-worth and declares {declared:.2f}: every route sizes its regions from"
                    f" the declaration, so a region is being cut for less than this stage takes"
                )
        if not exercised:
            pytest.skip(f"{cls.__name__} refuses every configuration standalone")
    finally:
        set_per_rank_budget(None)


def test_a_stage_that_declares_nothing_is_assumed_to_hold_something() -> None:
    """Silence must read as "one working copy", never as "free".

    A stage is written by whoever needs it, and the memory contract is not the reason they are
    writing it. A default of zero meant a stage whose author never heard of this attribute was
    priced as costing nothing, and every route sized its regions on that: of the 39 stages KonfAI
    itself ships, 33 declared nothing and nine of those held something -- up to fifteen
    volumes-worth. The framework's own authors did not get it right, so a user will not either.

    One working copy is what an out-of-place operation costs, and being wrong in that direction
    costs a shorter region where being wrong in the other costs the run.
    """

    class Invented(Transform):
        """What someone writing their own stage produces on the first try."""

        def transform_shape(self, group: str, name: str, shape: list[int], attribute) -> list[int]:
            return shape

        def __call__(self, name: str, tensor: torch.Tensor, cache_attribute) -> torch.Tensor:
            return tensor * 2.0

    # Two, because a store serves int16 and a stage cannot work in it: it materialises a float
    # copy before its own. Measured on this shape: 1.00 on float32 and 2.00 on int16.
    assert Invented().working_multiple == 2.0
    assert Invented().case_working_multiple("CASE_000") == 2.0

    # And a stage that measured free stays free: the declaration is what the allocator said, not a
    # habit. Flip is a view onto what it was handed, on every dtype a store serves.
    from konfai.data.transform import Flip

    assert Flip().working_multiple == 0.0
