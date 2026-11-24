# SPDX-License-Identifier: Apache-2.0
"""An app's extension of KonfAI Edge is compared with the Python class it stands for."""

from pathlib import Path

import pytest
from konfai_apps.errors import AppMetadataError
from konfai_apps.extension import check_extension, classes_run

#: An extension holding one transform, whatever its name: each value doubled, and halved back.
DOUBLE = """
(module
  (memory (export "memory") 1)
  (global $top (mut i32) (i32.const 1024))
  (func (export "ext_abi") (result i32) (i32.const 2))
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
    # A class the compiled program runs otherwise (folded into a graph) is left out; one it runs by the extension
    # is compared.
    program = {"steps": [{"op": "extension", "class": "OwnScale:Scale", "params": {"scale": 3.0}}]}
    assert classes_run(program) == {"OwnScale:Scale": [{"scale": 3.0}]}
    assert check_extension(extension, [config], runs={}) == []
    # Only the binding the program runs: the config's scale: 3.0 is not it.
    assert check_extension(extension, [config], runs={"OwnScale:Scale": [{"scale": 2.0}]}) == []
    with pytest.raises(AppMetadataError, match="does not transform a volume"):
        check_extension(extension, [config], runs=classes_run(program))
    # A pointwise class the program runs by the extension must be there, though a graph could carry it.
    (tmp_path / "missing.wasm").write_bytes(
        wasmtime.wat2wasm(DOUBLE.replace("(i32.const 0) (i32.const 2)", "(i32.const 1) (i32.const 1)"))
    )
    with pytest.raises(AppMetadataError, match="implements no such class"):
        check_extension(tmp_path / "missing.wasm", [config], runs=classes_run(program))


#: An extension holding one patch combine, whatever its name: the patch replaces what is held, from the first one on,
#: or from the second one on with ``(global $from (mut i32) (i32.const 1))``.
LAST = """
(module
  (memory (export "memory") 1)
  (global $top (mut i32) (i32.const 1024))
  (global $from (mut i32) (i32.const 0))
  (func (export "ext_abi") (result i32) (i32.const 2))
  (func (export "ext_buffer") (param $len i32) (result i32)
    (local $at i32)
    (local.set $at (global.get $top))
    (global.set $top (i32.add (global.get $top) (i32.mul (local.get $len) (i32.const 4))))
    (local.get $at))
  (func (export "ext_open") (param $kind i32) (param i32 i32 i32 i32) (result i32)
    (select (i32.const 0) (i32.const 2) (i32.eqz (local.get $kind))))
  (func (export "ext_output") (param $channels i32) (param i32) (result i32) (local.get $channels))
  (func $copy (param $from i32) (param $to i32) (param $bytes i32)
    (local $i i32)
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $i) (local.get $bytes)))
        (f32.store (i32.add (local.get $to) (local.get $i)) (f32.load (i32.add (local.get $from) (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $next))))
  (func (export "ext_fold")
    (param $held i32) (param i32) (param $patch i32) (param i32) (param $channels i32) (param i32) (param i32)
    (param $pixels i32)
    (if (global.get $from)
      (then (global.set $from (i32.sub (global.get $from) (i32.const 1))) (return)))
    (call $copy (local.get $patch) (local.get $held)
      (i32.mul (i32.mul (local.get $channels) (local.get $pixels)) (i32.const 4))))
  (func (export "ext_leave")
    (param $held i32) (param i32) (param $out i32) (param i32) (param $channels i32) (param $pixels i32)
    (call $copy (local.get $held) (local.get $out)
      (i32.mul (i32.mul (local.get $channels) (local.get $pixels)) (i32.const 4)))))
"""


def _app(tmp_path: Path, monkeypatch, module: str, source: str, wat: str, config: str) -> tuple[Path, Path]:
    """The app's own module, ``OWN`` and ``source``, its extension built from ``wat``, and its prediction config."""
    wasmtime = pytest.importorskip("wasmtime")
    (tmp_path / f"{module}.py").write_text(OWN + source)
    monkeypatch.syspath_prepend(str(tmp_path))
    extension = tmp_path / f"{module}.wasm"
    extension.write_bytes(wasmtime.wat2wasm(wat))
    (tmp_path / "Prediction.yml").write_text(config)
    return extension, tmp_path / "Prediction.yml"


def _transform(module: str, name: str) -> str:
    return (
        f"Predictor:\n  Model:\n    classpath: {module}:Net\n  Dataset:\n    transforms:\n      {module}:{name}: {{}}\n"
    )


def test_a_class_taking_keywords_it_forwards_is_built_as_konfai_builds_it(tmp_path: Path, monkeypatch) -> None:
    # Its `**options` is no parameter: none is passed to it, and none is handed to the extension.
    from konfai.data.transform import Transform
    from konfai_apps.extension import _built

    forwarded = """

class Forwarded(Scale):
    def __init__(self, scale: float = 2.0, **options) -> None:
        super().__init__(scale, **options)
"""
    module = "OwnForwarded"
    extension, config = _app(tmp_path, monkeypatch, module, forwarded, DOUBLE, _transform(module, "Forwarded"))

    built = _built(f"{module}:Forwarded", {"scale": 2.0}, (Transform,))
    assert built is not None and built[0].scale == 2.0 and built[1] == {"scale": 2.0}
    assert check_extension(extension, [config]) == [f"{module}:Forwarded"]


def test_a_pointwise_class_is_compared_on_the_case_it_planned(tmp_path: Path, monkeypatch) -> None:
    # What the class settles at plan time (transform_shape) is on the attribute its call and its inverse read.
    planned = """

class Planned(Scale):
    def transform_shape(self, group_src, name, shape, cache_attribute):
        cache_attribute["Factor"] = 2.0
        return shape

    def __call__(self, name, tensor, cache_attribute):
        return tensor * float(cache_attribute["Factor"])

    def inverse(self, name, tensor, cache_attribute):
        return tensor / float(cache_attribute["Factor"])
"""
    module = "OwnPlanned"
    extension, config = _app(tmp_path, monkeypatch, module, planned, DOUBLE, _transform(module, "Planned"))

    assert check_extension(extension, [config]) == [f"{module}:Planned"]


def test_a_class_taking_labels_only_is_compared_on_labels(tmp_path: Path, monkeypatch) -> None:
    # Past an Argmax a class is handed int64 labels, the only dtype it takes: it is compared on them.
    labels = """

import torch


class Labels(Scale):
    def __call__(self, name, tensor, cache_attribute):
        if tensor.dtype != torch.int64:
            raise TypeError(f"labels, not {tensor.dtype}")
        return tensor * 2

    def inverse(self, name, tensor, cache_attribute):
        return tensor // 2
"""
    module = "OwnLabels"
    extension, config = _app(tmp_path, monkeypatch, module, labels, DOUBLE, _transform(module, "Labels"))

    assert check_extension(extension, [config]) == [f"{module}:Labels"]


def test_a_patch_combine_is_compared_on_one_patch_as_on_several(tmp_path: Path, monkeypatch) -> None:
    # One that leaves the first patch out returns the last of three as the class does, and nothing of a lone one.
    last = """

from konfai.data.patching import Fold


class Last(Fold):
    def fold(self, held, patch):
        return patch
"""
    config = "Predictor:\n  outputs_dataset:\n    H:\n      OutputDataset:\n        patch_combine: OwnLast:Last\n"
    extension, config_file = _app(tmp_path, monkeypatch, "OwnLast", last, LAST, config)

    assert check_extension(extension, [config_file]) == ["OwnLast:Last"]
    late = tmp_path / "late.wasm"
    late.write_bytes(
        pytest.importorskip("wasmtime").wat2wasm(
            LAST.replace("$from (mut i32) (i32.const 0)", "$from (mut i32) (i32.const 1)")
        )
    )
    with pytest.raises(AppMetadataError, match="'OwnLast:Last' does not combine one patch"):
        check_extension(late, [config_file])


@pytest.mark.parametrize(
    ("module", "refused"),
    [
        (b"\0asm, and no module", "is no WebAssembly module"),
        (DOUBLE.replace("(i32.const 2))", "(i32.const 1))", 1), "interface 1"),
        (DOUBLE.replace("(module", '(module (import "konfai" "missing" (func))', 1), "does not start"),
    ],
    ids=["no module", "another interface", "an import nothing gives"],
)
def test_the_module_is_checked_when_the_configs_name_no_class_of_the_app(tmp_path: Path, module, refused) -> None:
    wasmtime = pytest.importorskip("wasmtime")
    extension = tmp_path / "own.wasm"
    extension.write_bytes(module if isinstance(module, bytes) else wasmtime.wat2wasm(module))
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor:\n  Dataset:\n    transforms:\n      Normalize: {}\n")

    with pytest.raises(AppMetadataError, match=refused):
        check_extension(extension, [config])
    extension.write_bytes(wasmtime.wat2wasm(DOUBLE))
    assert check_extension(extension, [config]) == []
