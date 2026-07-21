# SPDX-License-Identifier: Apache-2.0
"""A user's custom POINTWISE torch transform, for the auto-fold test (imported by classpath)."""

from konfai.data.transform import LocalityKind, PatchLocality, Transform


class DoubleThenBias(Transform):
    """Pointwise `2*x + bias` -- the kind of custom intensity transform that folds into the ONNX."""

    def __init__(self, bias: float = 1.0) -> None:
        super().__init__()
        self.bias = bias

    def patch_locality(self, cache_attribute):
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name, tensor, cache_attribute):
        return tensor * 2.0 + self.bias
