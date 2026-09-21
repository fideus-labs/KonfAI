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

"""What a registration engine's failure looks like to konfai."""

import contextlib
from collections.abc import Iterator

import torch


@contextlib.contextmanager
def out_of_memory_as_torch(on_cuda: bool) -> Iterator[None]:
    """Re-raise an engine's CUDA out-of-memory as ``torch.cuda.OutOfMemoryError``.

    An engine that allocates outside PyTorch (libtorch inside itk-impact, the elastix subprocess) reports it
    as a plain ``RuntimeError``; konfai shrinks a free patch axis on torch's own class only. Only a CUDA run
    is translated, on a line naming both CUDA and the out-of-memory: a host allocation that fails is no
    reason to cut a patch.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as error:
        lines = (line.strip() for line in str(error).splitlines())
        line = next((line for line in lines if "cuda" in line.lower() and "out of memory" in line.lower()), None)
        if not on_cuda or line is None:
            raise
        raise torch.cuda.OutOfMemoryError(line) from error
