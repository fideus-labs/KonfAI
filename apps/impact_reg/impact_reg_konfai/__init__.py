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

"""Python package for the IMPACT-Reg KonfAI app wrapper."""

import os

#: The revision of ``VBoussot/ImpactReg`` this package resolves its presets at. A release pins it here, to the
#: Hugging Face tag of the presets it was tested against, so that a later edit of the presets never reaches an
#: install that does not know their keys; "main" follows the repository as it moves.
PRESETS_REVISION = "7d0971d48946106f5b907312d209632ef0f2fcba"

#: Where the presets are resolved from: ``KONFAI_IMPACTREG_REPO`` when set (a local directory of preset folders,
#: for development and offline use, or a Hugging Face ``<repo>[@<revision>]``), else ``PRESETS_REVISION`` of
#: ``VBoussot/ImpactReg``. Here rather than in ``impact_reg`` so that reading it imports no torch.
#: How a case registers (``--out-of-core``): auto, whole where it fits the device and sampled where it does not;
#: whole, in one piece on its native grid; tiled, a global pass on the pair resampled to fit, then the deformable
#: stage on native tiles; sampled, at native resolution, by boxes.
ROUTES = ("auto", "whole", "tiled", "sampled")

PRESETS_REPO = os.environ.get("KONFAI_IMPACTREG_REPO", f"VBoussot/ImpactReg@{PRESETS_REVISION}")
