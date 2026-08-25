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

"""The memory budget as the kernel enforces it: a streamed route that outgrows its price is killed.

Each route runs in a child that caps its own ``RLIMIT_AS`` before importing torch, so an allocation
past the cap fails where the process runs, not in a reader's judgement. The cap is the interpreter
floor measured here (torch + SimpleITK + the workflow modules, in an identically pinned
environment) plus what the route is allowed above it, and the child reports the peak RSS it reached.

Two ceilings, because they bound two different things. The address space carries the maps a region
read and a region write hold on top of the working set, so it is the looser of the two; the resident
set is the working set itself, and it is what a budget is about. A route that read a case whole
instead of streaming it breaks both.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SimpleITK")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_AS accounting and /proc/self/status are Linux's"
)

#: Address space a route may map above the interpreter floor, as a multiple of its declared budget.
#: Measured minima on the cohort below: 1.25x (PREDICTION), 1.9x (EVALUATION), 2.0x (TRANSFORM).
_ADDRESS_SPACE_MULTIPLE = 3
#: Resident bytes a route may hold above the interpreter floor, as a multiple of its declared budget
#: and over the floor below. Measured over a 667 MiB interpreter floor: 0.41x (PREDICTION at 128
#: MiB), 0.33x / 1.05x / 1.17x (TRANSFORM at 512, 128, 64), 0.55x / 1.42x (EVALUATION at 512, 128).
#: A whole-volume read of the cohort's case is 2 x 78 MiB before its working copies: over both.
_RESIDENT_MULTIPLE = 2
#: What a route holds that no budget shrinks: the workflow's objects, the chain's stages, the store
#: handles, the allocator's slack. Measured at 25 to 27 MiB on this cohort, which is what
#: ``konfai.data.patching.SWEEP_ENGINE_FLOOR_BYTES`` declares and the TRANSFORM plan's header prints.
_ENGINE_FLOOR = 32 * (1 << 20)

_MIB = 1 << 20
#: 2 groups x 20.5 M voxels of float32: over the 17.9 M the 512 MiB sizing allows, so both budgets
#: cut the case, and 78 MiB per group, so a whole-volume read of it breaks either ceiling.
_CASE_SHAPE = (200, 320, 320)

# The child's environment, pinned so the floor it is measured against is reproducible: the CUDA
# libraries reserve address space at import, an OpenMP pool and a glibc arena reserve more per
# thread, and none of that is what a budget is meant to bound.
_CHILD_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "1"}

_PREAMBLE = '''\
"""One route under a hard cap. argv: <cap bytes> <budget> <cohort root>."""

import json
import resource
import sys
from pathlib import Path

CAP, BUDGET, ROOT = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if CAP:
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))


def _kib(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1])
    return 0

'''

_EPILOGUE = """

outcome, detail = "ok", ""
try:
    route()
except BaseException as failure:  # the finding is which failure, so none of them escapes
    outcome, detail = type(failure).__name__, str(failure)
print("@@" + json.dumps({
    "outcome": outcome,
    "detail": detail,
    "address_space_kib": _kib("VmPeak"),
    "resident_kib": _kib("VmHWM"),
}))
"""

_FLOOR = """
def route():
    import SimpleITK  # noqa: F401
    import torch  # noqa: F401

    import konfai  # noqa: F401
    import konfai.evaluator  # noqa: F401
    import konfai.predictor  # noqa: F401
    import konfai.transformer  # noqa: F401
"""

_EVALUATION = """
def route():
    import konfai
    from konfai.metric.measure import MAE, SSIM

    konfai.evaluate(
        "MEMORY_LIMIT",
        f"{ROOT / 'Dataset'}:mha",
        {"sCT": {"CT": [MAE(), SSIM(dynamic_range=1.0)]}},
        dataset_options={"memory_budget": BUDGET, "num_workers": 0},
        cpu=1,
        quiet=True,
        overwrite=True,
        evaluations_dir=ROOT / "Evaluations",
    )
"""

_TRANSFORM = """
def route():
    import konfai
    from konfai.data.transform import Gradient, Resample, Standardize, Write

    whole = BUDGET.endswith("!whole")
    chain = [Resample(spacing=[2.0, 2.0, 2.0]), Gradient()]
    if whole:
        chain.append(Standardize())  # GLOBAL_STAT: no region serves it, the case goes whole-volume
    # One output per run: an existing one is a resume, and a run that resumed would measure nothing.
    out = ROOT / f"Out_{BUDGET.replace('!', '_')}"
    konfai.transform(
        "MEMORY_LIMIT",
        f"{ROOT / 'Dataset'}:mha",
        {"CT": {"CT_out": [*chain, Write(dataset=f"{out}:mha")]}},
        memory_budget=BUDGET.removesuffix("!whole"),
        cpu=1,
        quiet=True,
        overwrite=True,
        transforms_dir=ROOT / "Transforms",
    )
"""

_PREDICTION = """
def route():
    sys.path.insert(0, str(ROOT))
    import konfai

    konfai.predict(
        models=[],  # TinyWeightless has no parameter: the route is the patch grid, not the weights
        config={"Predictor": {
            "Model": {
                "classpath": "tiny_weightless:TinyWeightlessNet",
                "TinyWeightlessNet": {"outputs_criterions": "None"},
            },
            "Dataset": {
                "dataset_filenames": [f"{ROOT / 'Dataset'}:mha"],
                "groups_src": {"CT": {"groups_dest": {"CT": {
                    "transforms": "None", "patch_transforms": "None", "is_input": True,
                }}}},
                "augmentations": "None",
                "Patch": {"patch_size": [32, 128, 128], "overlap": "None", "pad_value": 0, "extend_slice": 0},
                "memory_budget": BUDGET,
                "subset": "None",
                "batch_size": 1,
                "num_workers": 0,
            },
            "outputs_dataset": {"Head:Scale": {"OutputDataset": {
                "name_class": "OutSameAsGroupDataset",
                "before_reduction_transforms": "None",
                "after_reduction_transforms": "None",
                "final_transforms": "None",
                "dataset_filename": f"{ROOT / 'Out'}:mha",
                "group": "pCT",
                "same_as_group": "CT:CT",
                "patch_combine": "None",
                "reduction": "Mean",
            }}},
            "train_name": "MEMORY_LIMIT",
            "manual_seed": 0,
            "gpu_checkpoints": "None",
            "autocast": False,
            "combine": "Mean",
            "data_log": "None",
        }},
        cpu=1,
        quiet=True,
        overwrite=True,
        predictions_dir=ROOT / "Predictions",
    )
"""

_WEIGHTLESS_MODEL = '''\
"""A model with no parameter: prediction runs it as constructed, with no checkpoint to load."""

import torch
from konfai.network import network


class Scale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5


class Head(network.ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Scale", Scale())


class TinyWeightlessNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {"default|ConstantLR": network.LRSchedulersLoader(0)},
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {
            "Head:Scale": network.TargetCriterionsLoader()
        },
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=3,
        )
        self.add_module("Head", Head())
'''


def _cap(floor: dict, allowance_bytes: int) -> int:
    """A child's address-space cap: the interpreter floor plus what this run is allowed above it."""
    return floor["address_space_kib"] * 1024 + allowance_bytes


def _run(source: str, cap_bytes: int, budget: str, root: Path) -> dict:
    """Run one route in a child capped at ``cap_bytes`` of address space (``0`` leaves it uncapped,
    which is how the floor itself is measured); return what it reported."""
    script = root / "route.py"
    script.write_text(_PREAMBLE + source + _EPILOGUE, encoding="utf-8")
    # Every KONFAI_* key the caller happens to carry is dropped: each workflow sets its own, and a
    # leftover would make the measurement the developer's environment's rather than the route's.
    inherited = {key: value for key, value in os.environ.items() if not key.startswith("KONFAI")}
    environment = inherited | _CHILD_ENVIRONMENT | {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    completed = subprocess.run(
        [sys.executable, str(script), str(cap_bytes), budget, str(root)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )
    reported = [line for line in completed.stdout.splitlines() if line.startswith("@@")]
    assert reported, f"the child reported nothing:\n{completed.stdout}\n{completed.stderr}"
    return json.loads(reported[-1][2:]) | {"log": completed.stdout + completed.stderr}


@pytest.fixture(scope="module")
def cohort(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One synthetic case of a ground truth and a prediction of it, too big to fit either budget."""
    import SimpleITK as sitk

    root = tmp_path_factory.mktemp("memory_limit")
    case = root / "Dataset" / "CASE_000"
    case.mkdir(parents=True)
    truth = np.random.default_rng(0).random(_CASE_SHAPE, dtype=np.float32)
    for group, values in (("CT", truth), ("sCT", (0.9 * truth + 0.05).astype(np.float32))):
        image = sitk.GetImageFromArray(values)
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(case / f"{group}.mha"))
    (root / "tiny_weightless.py").write_text(_WEIGHTLESS_MODEL, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def floor(cohort: Path) -> dict:
    """The interpreter's own address space and resident set, in the child's pinned environment."""
    measured = _run(_FLOOR, 0, "None", cohort)
    assert measured["outcome"] == "ok", measured
    return measured


def _within_its_budget(source: str, budget_bytes: int, floor: dict, cohort: Path) -> dict:
    """Run one route under the cap its budget earns it, and hold it to the resident ceiling."""
    measured = _run(
        source, _cap(floor, _ENGINE_FLOOR + _ADDRESS_SPACE_MULTIPLE * budget_bytes), f"{budget_bytes}b", cohort
    )
    assert measured["outcome"] == "ok", measured["detail"]
    held = (measured["resident_kib"] - floor["resident_kib"]) * 1024
    assert held <= _ENGINE_FLOOR + _RESIDENT_MULTIPLE * budget_bytes, (
        f"held {held / _MIB:.0f} MiB above the interpreter floor for a budget of {budget_bytes / _MIB:.0f} MiB"
    )
    return measured


@pytest.mark.parametrize("budget_mib", [512, 128])
def test_a_streamed_evaluation_holds_the_budget_it_declared(budget_mib: int, floor: dict, cohort: Path) -> None:
    """The case exceeds both budgets, so both runs cut it into disjoint patches read with SSIM's
    halo, and neither may hold more than the budget that sized those patches."""
    measured = _within_its_budget(_EVALUATION, budget_mib * _MIB, floor, cohort)

    assert "disjoint patches" in measured["log"] and "halo of 3" in measured["log"]


def test_a_streamed_transform_tracks_the_budget_it_declared(floor: dict, cohort: Path) -> None:
    """Resample (REGRID) then Gradient (HALO): a chain that streams, so the sweep sizes its regions
    from the budget and the case is never resident.

    Not only inside the budget but FOLLOWING it. A sizing that priced the landed rows alone held the
    same 173 MiB at 512, 128 and 32 MiB alike, because what this chain holds is what its regions pull
    (four source voxels per landed one) and what Gradient allocates over them (eight volumes-worth),
    neither of which the landed rows say anything about.
    """
    held = {}
    for budget_mib in (512, 128, 64):
        measured = _within_its_budget(_TRANSFORM, budget_mib * _MIB, floor, cohort)
        held[budget_mib] = (measured["resident_kib"] - floor["resident_kib"]) * 1024

    assert held[64] < held[128] < held[512], f"flat against the budget: {held}"
    plan = (cohort / "Transforms" / "MEMORY_LIMIT" / "log_0.txt").read_text(encoding="utf-8")
    assert "1 STREAM" in plan and "0 WHOLE-VOLUME" in plan


def test_a_transform_refuses_a_budget_no_region_of_its_chain_fits(floor: dict, cohort: Path) -> None:
    """The floor of the sizing is a refusal, not a one-row sweep: one row of this landing pulls four
    rows of the source and Gradient allocates eight of them over it, so a 1 MiB budget buys nothing
    and the message says what the smallest region holds."""
    measured = _run(_TRANSFORM, _cap(floor, 256 * _MIB), "1MiB", cohort)

    assert measured["outcome"] == "TransformerError", measured
    assert "1.00 MiB" in measured["detail"] and "no region of 'CT' fits" in measured["detail"]
    assert not (cohort / "Out_1MiB").exists(), "refused before a byte"


def test_a_patched_prediction_holds_the_budget_it_declared(floor: dict, cohort: Path) -> None:
    """A [32, 128, 128] grid over a 200x320x320 case, its output written slab by slab because the
    assembled volume is worth streaming against the budget."""
    _within_its_budget(_PREDICTION, 128 * _MIB, floor, cohort)


def test_an_evaluation_refuses_a_budget_no_haloed_patch_fits(floor: dict, cohort: Path) -> None:
    """A refusal, not a MemoryError and not a kill: the budget cannot hold one patch with the halo
    SSIM scores through, and the message carries both figures."""
    measured = _run(_EVALUATION, _cap(floor, 256 * _MIB), "4096b", cohort)

    assert measured["outcome"] == "DatasetManagerError", measured
    assert "4.00 KiB" in measured["detail"] and "halo" in measured["detail"]


def test_a_transform_refuses_a_case_its_budget_cannot_hold_whole(floor: dict, cohort: Path) -> None:
    """Standardize needs the stored volume's statistics, so no region serves the chain and the case
    is priced against the budget whole: refused before a byte is written."""
    measured = _run(_TRANSFORM, _cap(floor, 256 * _MIB), "1MiB!whole", cohort)

    assert measured["outcome"] == "TransformerError", measured
    assert "1.00 MiB" in measured["detail"]
    assert not (cohort / "Out_1MiB_whole").exists(), "refused before a byte"
