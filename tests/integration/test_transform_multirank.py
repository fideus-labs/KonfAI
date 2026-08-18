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

"""``konfai TRANSFORM --cpu 2``, as the CLI runs it: two ranks spawned by ``mp.spawn``.

The unit tests call ``run_process`` for each rank inside one process, which exercises the sharding
but not the launch: the pickle of the planned workflow across the spawn, one log per rank, and
the absence of any rendezvous (TRANSFORM declares ``uses_collectives = False``, so no process
group, no port, no gloo).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from harness import konfai_cli_command, subprocess_env
from konfai.utils.dataset import Attribute, Dataset

_CASES = 4
_DONE = re.compile(r"\[KonfAI\] rank (?P<rank>\d)/2 done in [\d.]+ s: (?P<written>\d+) written")


def _cohort(workdir: Path) -> dict[str, np.ndarray]:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(0)
    source = Dataset(workdir / "Raw", "mha")
    volumes: dict[str, np.ndarray] = {}
    for index in range(_CASES):
        volumes[f"CASE_{index:03d}"] = (rng.random((1, 12, 10, 8)) * 100).astype(np.float32)
        source.write("CT", f"CASE_{index:03d}", volumes[f"CASE_{index:03d}"], attributes)
    (workdir / "Transform.yml").write_text(
        "Transformer:\n"
        "  name: TWO_RANKS\n"
        "  Dataset:\n"
        "    dataset_filenames:\n"
        f"      - {workdir / 'Raw'}:mha\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          CT_out:\n"
        "            transforms:\n"
        "              Clip:\n"
        "                min_value: 0.0\n"
        "                max_value: 50.0\n"
        "              Write:\n"
        f"                dataset: {workdir / 'Out'}/:omezarr\n",
        encoding="utf-8",
    )
    return volumes


@pytest.mark.integration
def test_two_spawned_ranks_write_every_case_once_without_a_process_group(tmp_path: Path) -> None:
    volumes = _cohort(tmp_path)
    env = subprocess_env()
    # A rendezvous nobody can reach: a run that initialised a process group would refuse it
    # ("Port could not be cast to integer"), where a run without one never reads it.
    env["KONFAI_MASTER_PORT"] = "no-rendezvous"
    completed = subprocess.run(
        [*konfai_cli_command(), "TRANSFORM", "--config", "Transform.yml", "--cpu", "2"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
    )
    assert completed.returncode == 0, f"the two-rank run failed:\n{completed.stdout}{completed.stderr}"

    logs = {rank: tmp_path / "Transforms" / "TWO_RANKS" / f"log_{rank}.txt" for rank in (0, 1)}
    assert all(log.is_file() for log in logs.values()), sorted(p.name for p in (tmp_path / "Transforms").rglob("*"))
    written_by_rank: dict[int, int] = {}
    for rank, log in logs.items():
        closing = [match for match in map(_DONE.search, log.read_text(encoding="utf-8").splitlines()) if match]
        assert len(closing) == 1, f"rank {rank} did not close its shard exactly once:\n{log.read_text()}"
        assert int(closing[0]["rank"]) == rank
        written_by_rank[rank] = int(closing[0]["written"])
    # Every case written exactly once, and both ranks did their share of it.
    assert sum(written_by_rank.values()) == _CASES, written_by_rank
    assert all(count > 0 for count in written_by_rank.values()), written_by_rank

    output = Dataset(f"{tmp_path / 'Out'}/", "omezarr")
    assert sorted(output.get_names("CT_out")) == sorted(volumes)
    for case, volume in volumes.items():
        np.testing.assert_array_equal(output.read_data("CT_out", case)[0], np.clip(volume, 0.0, 50.0))

    everything = completed.stdout + completed.stderr + "".join(log.read_text(encoding="utf-8") for log in logs.values())
    for rendezvous in ("init_process_group", "gloo", "MASTER"):
        assert rendezvous not in everything, f"a rank spoke of a rendezvous ({rendezvous!r}):\n{everything}"
