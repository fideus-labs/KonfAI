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

"""Write a small cohort that does NOT share a grid, which is what makes the example worth running.

Six volumes, each a smooth blob on its own extent, spacing and origin, with its own intensity
dynamics. That is the ordinary state of a cohort as acquired: nothing about it is wrong, and nothing
about it lets you average two members together. Bringing them onto one grid is the work.

    python make_dataset.py            # writes ./Raw/<case>/CT.mha

The whole cohort is about 3 MB and takes a second to generate; there is nothing to download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

#: Each case gets its own geometry. The spread is deliberate and mild: extents differ by a few
#: voxels, spacings by up to 30%, and origins by more than one voxel -- enough that `grid: strict`
#: refuses the cohort as stored, which is the point the example is making.
_CASES: tuple[tuple[str, tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("CASE_000", (48, 56, 56), (1.00, 1.00, 1.00), (0.0, 0.0, 0.0)),
    ("CASE_001", (44, 60, 52), (1.15, 0.95, 1.05), (2.5, -1.0, 0.5)),
    ("CASE_002", (52, 52, 60), (0.90, 1.10, 0.95), (-1.5, 3.0, -2.0)),
    ("CASE_003", (46, 58, 54), (1.05, 1.00, 1.20), (1.0, 1.5, 2.5)),
    ("CASE_004", (50, 54, 58), (0.95, 1.20, 1.00), (-2.0, 0.5, 1.0)),
    ("CASE_005", (45, 59, 53), (1.10, 1.05, 0.90), (3.0, -2.5, -1.5)),
)


def _volume(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """A blob with a soft rim and a little noise, in roughly Hounsfield-looking numbers."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.linspace(-1.0, 1.0, extent) for extent in shape], indexing="ij")
    # A shifted, slightly anisotropic ellipsoid, so the members overlap without coinciding.
    centre = rng.uniform(-0.15, 0.15, size=3)
    radii = rng.uniform(0.55, 0.75, size=3)
    distance = sum(((axis - centre[i]) / radii[i]) ** 2 for i, axis in enumerate(grids))
    blob = 1.0 / (1.0 + np.exp((distance - 1.0) * 8.0))
    intensity = rng.uniform(280.0, 360.0)
    return (blob * intensity + rng.normal(0.0, 6.0, size=shape)).astype(np.float32)


def write(root: Path, group: str = "CT") -> None:
    for seed, (name, shape, spacing, origin) in enumerate(_CASES):
        image = sitk.GetImageFromArray(_volume(shape, seed))
        # sitk takes geometry in (x, y, z) where the array is (z, y, x).
        image.SetSpacing(tuple(reversed(spacing)))
        image.SetOrigin(tuple(reversed(origin)))
        case = root / name
        case.mkdir(parents=True, exist_ok=True)
        # Uncompressed on purpose: a compressed .mha cannot serve a disk region, so every slab would
        # decode the whole volume again and the example would demonstrate the opposite of streaming.
        sitk.WriteImage(image, str(case / f"{group}.mha"), useCompression=False)
    print(f"Wrote {len(_CASES)} cases under {root}")
    print("No two share an extent, a spacing or an origin -- 'grid: strict' refuses them as stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./Raw"), help="where to write the cohort")
    parser.add_argument("--group", default="CT", help="group name each case is stored under")
    write(**vars(parser.parse_args()))
