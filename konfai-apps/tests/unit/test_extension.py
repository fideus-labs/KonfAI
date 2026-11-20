# SPDX-License-Identifier: Apache-2.0
"""An app's extension of KonfAI Edge is compared with the Python class it stands for."""

from pathlib import Path

import pytest
from konfai_apps.errors import AppMetadataError
from konfai_apps.extension import check_extension

#: An extension holding one transform, whatever its name: each value doubled, and halved back.
DOUBLE = """
(module
  (memory (export "memory") 1)
  (global $top (mut i32) (i32.const 1024))
  (func (export "ext_buffer") (param $len i32) (result i32)
    (local $at i32)
    (local.set $at (global.get $top))
    (global.set $top (i32.add (global.get $top) (i32.mul (local.get $len) (i32.const 4))))
    (local.get $at))
  (func (export "ext_open") (param $kind i32) (param i32 i32 i32 i32) (result i32)
    (select (i32.const 0) (i32.const 2) (i32.eq (local.get $kind) (i32.const 1))))
  (func (export "ext_output") (param $channels i32) (param i32) (result i32) (local.get $channels))
  (func (export "ext_apply")
    (param $in i32) (param $channels i32) (param $out i32) (param $pixels i32) (param $inverse i32) (result i32)
    (local $i i32) (local $bytes i32)
    (local.set $bytes (i32.mul (i32.mul (local.get $channels) (local.get $pixels)) (i32.const 4)))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $bytes)))
        (f32.store (i32.add (local.get $out) (local.get $i))
          (f32.mul (f32.load (i32.add (local.get $in) (local.get $i)))
                   (select (f32.const 0.5) (f32.const 2) (local.get $inverse))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $next)))
    (i32.const 1)))
"""

OWN = """
from konfai.data.transform import LocalityKind, PatchLocality, TransformInverse


class Scale(TransformInverse):
    def __init__(self, scale: float = 2.0, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.scale = scale

    def patch_locality(self, cache_attribute):
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name, tensor, cache_attribute):
        return tensor * self.scale

    def inverse(self, name, tensor, cache_attribute):
        return tensor / self.scale
"""


def test_an_extension_is_compared_with_the_class_it_stands_for(tmp_path: Path, monkeypatch) -> None:
    wasmtime = pytest.importorskip("wasmtime")
    (tmp_path / "OwnScale.py").write_text(OWN)
    monkeypatch.syspath_prepend(str(tmp_path))
    extension = tmp_path / "own.wasm"
    extension.write_bytes(wasmtime.wat2wasm(DOUBLE))
    config = tmp_path / "Prediction.yml"
    chain = "Predictor:\n  Model:\n    classpath: OwnScale:Net\n  Dataset:\n    transforms:\n      OwnScale:Scale: {scale: %s}\n"

    # The model is no class an extension implements, and a class that does as the extension does passes.
    config.write_text(chain % 2.0)
    assert check_extension(extension, [config]) == ["OwnScale:Scale"]
    config.write_text(chain % 3.0)
    with pytest.raises(AppMetadataError, match="'OwnScale:Scale' does not transform a volume as the Python class does"):
        check_extension(extension, [config])
