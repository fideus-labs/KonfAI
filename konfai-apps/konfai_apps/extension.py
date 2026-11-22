# SPDX-License-Identifier: Apache-2.0
"""Check an app's extension of KonfAI Edge against the Python classes it stands for.

An app's own classes run from its ``.py`` files under konfai and from its extension, a WebAssembly module, under
KonfAI Edge, each under the classpath the app's configs name it by. :func:`check_extension` runs both on the same
random voxels and refuses an extension that returns something else. It needs ``wasmtime`` (the ``extension``
extra).
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

from .errors import AppMetadataError

#: The kinds an extension opens a class as, numbered as the extension numbers them.
PATCH_COMBINE, TRANSFORM, REDUCTION, CRITERION, VOLUME_TRANSFORM = range(5)
#: The dtypes a grid's code names, in the order of their codes, as KonfAI Edge lists them.
_DTYPES = (
    "bool",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float16",
    "float32",
    "float64",
)
#: The spatial extents of the volumes compared, the last ones for a case of fewer axes.
_EXTENTS = (3, 4, 5)


class _Opened:
    """One class of an extension, opened in an instance of its own: the calls of the extension's interface."""

    def __init__(self, extension: Path, kind: int, name: str, params: dict[str, Any]) -> None:
        import wasmtime

        engine = wasmtime.Engine()
        self.store = wasmtime.Store(engine)
        module = wasmtime.Module.from_file(engine, str(extension))
        # The functions a class reading by regions is lent, over the volumes `lent` holds for the call.
        self.lent: tuple[list[Any], Any] | None = None
        linker = wasmtime.Linker(engine)
        word = wasmtime.ValType.i32()
        read, write = wasmtime.FuncType([word] * 5, [word]), wasmtime.FuncType([word] * 4, [word])
        linker.define_func("konfai", "read", read, self._read, access_caller=True)
        linker.define_func("konfai", "write", write, self._write, access_caller=True)
        self.exports: Any = linker.instantiate(self.store, module).exports(self.store)
        refused = self.call("ext_open", kind, *self._text(name), *self._text(json.dumps(params)))
        if refused:
            reason = "implements no such class" if refused == 1 else "implements it as another kind"
            raise AppMetadataError(f"The extension {extension.name} {reason}: '{name}'.")

    def call(self, name: str, *arguments: int) -> int:
        return int(self.exports[name](self.store, *arguments) or 0)

    def _box(self, caller: Any, start: int, extents: int, rank: int) -> tuple[slice, ...]:
        """The box at the addresses `start` and `extents` of the caller's memory, as slices."""
        import numpy as np

        memory = caller.get("memory")
        read = [
            np.frombuffer(bytes(memory.read(caller, at & 0xFFFFFFFF, (at & 0xFFFFFFFF) + 4 * rank)), np.uint32)
            for at in (start, extents)
        ]
        return tuple(slice(int(first), int(first) + int(extent)) for first, extent in zip(*read, strict=True))

    def _read(self, caller: Any, source: int, channel: int, start: int, extents: int, out: int) -> int:
        import numpy as np

        sources, _ = self.lent or ([], None)
        if not 0 <= source < len(sources) or not 0 <= channel < sources[source].shape[0]:
            return 0
        volume = sources[source]
        box = self._box(caller, start, extents, volume.ndim - 1)
        values = np.ascontiguousarray(volume[(channel, *box)], np.float32)
        if values.shape != tuple(part.stop - part.start for part in box):
            return 0
        caller.get("memory").write(caller, values.tobytes(), out & 0xFFFFFFFF)
        return 1

    def _write(self, caller: Any, channel: int, start: int, extents: int, at: int) -> int:
        import numpy as np

        _, output = self.lent or ([], None)
        if output is None or not 0 <= channel < output.shape[0]:
            return 0
        box = self._box(caller, start, extents, output.ndim - 1)
        shape = tuple(part.stop - part.start for part in box)
        if output[(channel, *box)].shape != shape:
            return 0
        at &= 0xFFFFFFFF
        read = bytes(caller.get("memory").read(caller, at, at + 4 * math.prod(shape)))
        output[(channel, *box)] = np.frombuffer(read, np.float32).reshape(shape)
        return 1

    def has(self, name: str) -> bool:
        """Whether the extension exports ``name``."""
        return self.exports.get(name) is not None

    def buffer(self, count: int) -> int:
        """``count`` float32 at zero in the extension's memory."""
        return self.call("ext_buffer", count) & 0xFFFFFFFF

    def _text(self, text: str) -> tuple[int, int]:
        data = text.encode()
        at = self.buffer(-(-len(data) // 4))
        self.exports["memory"].write(self.store, data, at)
        return at, len(data)

    def put(self, values: Any, dtype: str = "float32") -> int:
        """``values`` in a buffer of the extension's memory."""
        import numpy as np

        array = np.ascontiguousarray(values, dtype=dtype)
        at = self.buffer(-(-array.nbytes // 4))
        self.exports["memory"].write(self.store, array.tobytes(), at)
        return at

    def get(self, at: int, shape: tuple[int, ...], dtype: str = "float32") -> Any:
        """The volume of ``shape`` at ``at``."""
        import numpy as np

        data = self.exports["memory"].read(self.store, at, at + np.dtype(dtype).itemsize * math.prod(shape))
        return np.frombuffer(bytes(data), dtype).reshape(shape)


def own_classes(config: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Every classpath outside konfai a config names, with the mapping it is bound from: its own for a key
    (a transform, a criterion, a draw), the one that holds it for a value (a ``patch_combine``, a ``reduction``)."""
    from konfai.portable import own_class

    if isinstance(config, list):
        for item in config:
            yield from own_classes(item)
    if not isinstance(config, dict):
        return
    for key, value in config.items():
        if isinstance(key, str) and own_class(key):
            yield key.split("#", 1)[0], value if isinstance(value, dict) else {}
        elif isinstance(value, str) and own_class(value):
            bound = config.get(value)
            yield value, bound if isinstance(bound, dict) else config
        yield from own_classes(value)


def _built(classpath: str, given: dict[str, Any], kinds: tuple[type, ...]) -> tuple[Any, dict[str, Any]] | None:
    """The class a classpath names, built with the keys of ``given`` its constructor takes, and every argument it
    was built with, defaults included: what the compiler hands the extension. ``None`` for a class of no kind an
    extension implements (the model), or for no class at all."""
    from konfai.portable import clean
    from konfai.utils.errors import ConfigError
    from konfai.utils.utils import get_module

    try:
        module, name = get_module(classpath, "konfai")
        cls = getattr(module, name)
        signature = inspect.signature(cls)
    except ImportError as error:  # the module is there and does not import: a class nothing would check
        raise AppMetadataError(f"'{classpath}' does not import, so the extension is not checked: {error}") from error
    except (ConfigError, AttributeError, TypeError, ValueError):  # a path, a format, a module's output: no class
        return None
    if not (isinstance(cls, type) and issubclass(cls, kinds)):
        return None
    bound = signature.bind_partial(**{key: clean(value) for key, value in given.items() if key in signature.parameters})
    bound.apply_defaults()
    return cls(**bound.arguments), dict(bound.arguments)


def _close(name: str, what: str, konfai: Any, extension: Any) -> None:
    import numpy as np

    konfai = np.asarray(konfai, dtype=np.float32)
    if konfai.shape != extension.shape or not np.allclose(konfai, extension, rtol=1e-5, atol=1e-6, equal_nan=True):
        raise AppMetadataError(
            f"The extension's '{name}' does not {what} as the Python class does.",
            "Both must compute the same values: KonfAI Edge runs the extension where konfai runs the class.",
        )


class _Served:
    """The run's datasets as a checked class reads them (``read_companion``): each group of the case it reads, a
    random volume of one channel over ``shape``, the same each time it is read, recorded in the order it was."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.read: list[str] = []
        self.volumes: dict[str, Any] = {}
        #: What a class memoizes per case, as on a dataset (konfai's ``Dataset.case_facts``).
        self.case_facts: dict[Any, Any] = {}

    def own(self, volume: Any) -> None:
        """``volume`` as the checked group stored: what a class reads of its own case (``transform_shape``), what it
        memoized of another forgotten."""
        self.volumes["check"] = volume.numpy()
        self.case_facts.clear()

    def is_dataset_exist(self, group: str, name: str) -> bool:
        import numpy as np

        if group not in self.volumes:
            self.read.append(group)
            generator = np.random.default_rng(len(self.read))
            self.volumes[group] = (generator.random((1, *self.shape)) * 100).astype(np.float32)
        return True

    def read_data(self, group: str, name: str) -> tuple[Any, None]:
        return self.volumes[group], None


def _geometry(rank: int, identity: bool = False) -> tuple[dict[str, Any], Any]:
    """A grid of ``rank`` axes, and the attribute konfai's class reads it from: a spacing and a direction no axis
    shares, so a class that reads the wrong one is told apart, or, with ``identity``, the identity grid."""
    import numpy as np
    from konfai.utils.dataset import Attribute

    geometry = {
        "Spacing": np.ones(rank) if identity else np.array([0.5, 1.0, 2.0][:rank]),
        "Origin": np.zeros(rank) if identity else np.array([10.0, 20.0, 30.0][:rank]),
        "Direction": np.eye(rank) if identity else np.diag([1.0, -1.0, 1.0][:rank]),
    }
    attribute = Attribute()
    for key, value in geometry.items():
        attribute[key] = value.flatten()
    return geometry, attribute


def _check_volume_transform(extension: Path, name: str, stage: Any, params: dict[str, Any], volume: Any) -> None:
    """A class that reads past the voxel: the extension is handed the volume with its grid, and says the grid it
    writes, where konfai's class records it in the attribute."""
    import numpy as np
    from konfai.data.transform import TransformInverse

    channels, *shape = volume.shape
    rank = len(shape)
    geometry, attribute = _geometry(rank)
    # Its plan-time facts first, as konfai settles them, then the class.
    stage.datasets[0].own(volume)
    stage.transform_shape("check", "check", list(shape), attribute)
    transformed = stage("check", volume.clone(), attribute)

    def grid(shape: Any, read: Any) -> Any:
        return np.concatenate([np.asarray(shape, float), *(np.asarray(read(key), float).flatten() for key in geometry)])

    opened = _Opened(extension, VOLUME_TRANSFORM, name, params)
    source = grid(shape, geometry.__getitem__)
    _check_halo(opened, name, stage, source, rank, attribute)

    def stated(export: str, *notes: int) -> Any:
        """The grid the extension states for the grid compared, given the notes at `notes` where there are."""
        at = opened.put(source, "float64")
        opened.call(export, at, rank, 1, *notes)
        return opened.get(at, (source.size,), "float64")

    # The groups of the case the class read, in that order, each in a buffer listed with its channels.
    served = stage.datasets[0]
    groups = [served.volumes[group] for group in served.read]
    # The forward into the grid stated before the values are read, which bounds one the values settle.
    bound = stated("ext_output_grid")

    def typed(values: Any) -> tuple[int, int]:
        """The grid compared with the code of the dtype of ``values`` after it, and the flags saying both follow."""
        code = _DTYPES.index(str(values.dtype).removeprefix("torch."))
        return opened.put(np.append(source, code), "float64"), 1 | 2

    at, flags = typed(volume)
    regions = opened.has("ext_reads_regions") and opened.call("ext_reads_regions") == 1
    if not regions:
        # The volume and what the class writes in its own memory.
        out = opened.buffer(transformed.shape[0] * math.prod(int(extent) for extent in bound[:rank]))
        read = (opened.put(volume), channels, at, rank, flags)
    if regions:
        # Lent the volumes, held here, and writing to one held here too.
        held = np.zeros((transformed.shape[0], *(int(extent) for extent in bound[:rank])), np.float32)
        opened.lent = ([volume.numpy(), *groups], held)
        counts = opened.put(np.array([channels, *(group.shape[0] for group in groups)], np.uint32), "uint32")
        try:
            opened.call("ext_apply_regions", at, rank, flags, counts, len(groups))
        finally:
            opened.lent = None
    elif groups:
        table = opened.put(np.array([[opened.put(group), group.shape[0]] for group in groups], np.uint32), "uint32")
        opened.call("ext_apply_volume_reading", *read, table, len(groups), out)
    else:
        opened.call("ext_apply_volume", *read, out, 0)
    # What it measured on the volume, and the grid it settled on.
    count = opened.call("ext_notes_len") if opened.has("ext_notes_len") else 0
    notes = opened.buffer(2 * count)
    if count:
        opened.call("ext_notes", notes)
    written = stated("ext_output_grid_noted", notes, count) if count else bound
    _close(name, "write the grid", grid(transformed.shape[1:], attribute.get_np_array), written.astype(np.float32))
    # By regions, the grid the values settle is the first box of the one bounding it.
    landed = (
        held[(slice(None), *(slice(0, int(extent)) for extent in written[:rank]))]
        if regions
        else opened.get(out, tuple(transformed.shape))
    )
    _close(name, "transform a volume", transformed, landed)
    if isinstance(stage, TransformInverse) and stage.apply_inverse:
        undone = stage.inverse("check", transformed.clone(), attribute)
        back = opened.buffer(channels * math.prod(shape))
        at, flags = typed(transformed)
        read = (opened.put(transformed), transformed.shape[0], at, rank, flags)
        done = (
            opened.call("ext_inverse_noted", *read, notes, count, back)
            if count
            else opened.call("ext_apply_volume", *read, back, 1)
        )
        if not done:
            raise AppMetadataError(f"The extension's '{name}' has no inverse, and the Python class is undone.")
        _close(name, "undo itself", undone, opened.get(back, (channels, *shape)))


def _check_halo(opened: _Opened, name: str, stage: Any, source: Any, rank: int, attribute: Any) -> None:
    """A class that reads its neighbours: the halo the extension states for the grid compared, or, where it states
    none, one the class keeps whatever the grid (the halo its step states)."""
    from konfai.data.transform import LocalityKind

    locality = stage.patch_locality(attribute)
    if locality.kind is not LocalityKind.HALO:
        return
    halo = max(locality.halo or (0,))
    # An int32 to wasmtime: `None`, the class's u32::MAX, is -1.
    stated = opened.call("ext_halo", opened.put(source, "float64"), rank, 1) if opened.has("ext_halo") else -1
    if stated == -1:
        identity = stage.patch_locality(_geometry(rank, identity=True)[1])
        if max(identity.halo or (0,)) != halo:
            raise AppMetadataError(
                f"'{name}' reads {halo} pixels around each on the grid compared and another count on another grid,"
                " and its extension states no halo for the grid (VolumeTransform::halo)."
            )
    elif stated != halo:
        raise AppMetadataError(f"The extension's '{name}' reads {stated} pixels around each, the Python class {halo}.")


def _taken(name: str, stage: Any, shape: tuple[int, ...]) -> Any:
    """A random volume of ``shape`` holding the first count of channels, up to 1024, the class takes: what its place
    in a chain hands it, which no config states."""
    import torch

    generator = torch.Generator().manual_seed(0)
    refused: Exception | None = None
    for channels in range(1, 1025):
        volume = torch.rand((channels, *shape), generator=generator) * 100
        # A first slab of zeros along each axis: a class that crops to the content crops.
        for axis in range(1, volume.dim()):
            volume.narrow(axis, 0, 1).zero_()
        attribute = _geometry(len(shape))[1]
        try:
            stage.datasets[0].own(volume)
            stage.transform_shape("check", "check", list(shape), attribute)
            stage("check", volume.clone(), attribute)
        except Exception as error:  # a count the class refuses is not the one its chain hands it
            refused = refused or error
            continue
        return volume
    raise AppMetadataError(
        f"'{name}' takes no volume of 1 to 1024 channels over {list(shape)}, so it is not checked: {refused}"
    )


def _check_transform(extension: Path, name: str, stage: Any, params: dict[str, Any], shape: tuple[int, ...]) -> None:
    from konfai.data.transform import LocalityKind, TransformInverse
    from konfai.utils.dataset import Attribute

    stage.set_datasets([_Served(shape)])
    volume = _taken(name, stage, shape)
    pixels = math.prod(shape)
    if stage.patch_locality(Attribute()).kind is not LocalityKind.POINTWISE:
        _check_volume_transform(extension, name, stage, params, volume)
        return
    opened = _Opened(extension, TRANSFORM, name, params)

    def run(values: Any, inverse: bool) -> Any:
        channels = opened.call("ext_output", values.shape[0], 1)
        out = opened.buffer(channels * pixels)
        if not opened.call("ext_apply", opened.put(values), values.shape[0], out, pixels, int(inverse)):
            raise AppMetadataError(f"The extension's '{name}' has no inverse, and the Python class is undone.")
        return opened.get(out, (channels, *shape))

    attribute = Attribute()  # one for both ways: a stage may record on it what its inverse reads back
    transformed = stage("check", volume.clone(), attribute)
    _close(name, "transform a volume", transformed, run(volume, False))
    if isinstance(stage, TransformInverse) and stage.apply_inverse and transformed.shape == volume.shape:
        _close(name, "undo itself", stage.inverse("check", transformed.clone(), attribute), run(transformed, True))


def _check_reduction(extension: Path, name: str, fold: Any, params: dict[str, Any], shape: tuple[int, ...]) -> None:
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    copies = [torch.rand((1, *shape), generator=generator) for _ in range(3)]
    opened = _Opened(extension, REDUCTION, name, params)
    pixels, channels = math.prod(shape), opened.call("ext_output", 1, len(copies))
    out = opened.buffer(channels * pixels)
    table = opened.put(np.array([opened.put(copy) for copy in copies]), "uint32")
    opened.call("ext_reduce", table, len(copies), 1, out, pixels)
    _close(name, "fold the copies", fold([copy[None] for copy in copies])[0], opened.get(out, (channels, *shape)))


def _check_fold(extension: Path, name: str, fold: Any, params: dict[str, Any], shape: tuple[int, ...]) -> None:
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    # What a patch enters the fold as depends on the head's channels, which no config states: the first count the
    # class takes.
    for head in (2, 3, 8, 32, 256, 1024):
        try:
            patches = [fold.enter(torch.rand((head, *shape), generator=generator)) for _ in range(3)]
            break
        except (IndexError, RuntimeError):
            continue
    else:
        raise AppMetadataError(f"'{name}' enters no patch of 2 to 1024 channels, so its extension is not checked.")
    held = torch.zeros_like(patches[0])
    opened = _Opened(extension, PATCH_COMBINE, name, params)
    pixels, channels = math.prod(shape), int(held.shape[0])
    held_at, runs = opened.buffer(channels * pixels), opened.put(np.zeros(2), "uint32")
    for patch in patches:
        held = fold.fold(held, patch)
        opened.call("ext_fold", held_at, pixels, opened.put(patch.float()), pixels, channels, runs, 1, pixels)
    written = opened.call("ext_output", channels, 1)
    out = opened.buffer(written * pixels)
    opened.call("ext_leave", held_at, pixels, out, pixels, channels, pixels)
    _close(name, "combine the patches", fold.leave(held), opened.get(out, (written, *shape)))


def _check_criterion(
    extension: Path, name: str, criterion: Any, params: dict[str, Any], shape: tuple[int, ...]
) -> None:
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    # As many targets as the class's forward names, one for a variadic `*targets`.
    named = [
        parameter
        for parameter in inspect.signature(criterion.forward).parameters.values()
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD and parameter.default is parameter.empty
    ]
    output, *targets = (torch.rand((1, *shape), generator=generator) for _ in range(max(len(named), 2)))
    opened = _Opened(extension, CRITERION, name, params)
    table = opened.put(np.array([[opened.put(target), 1] for target in targets]), "uint32")
    opened.call("ext_measure", opened.put(output), 1, table, len(targets), math.prod(shape))
    value = opened.buffer(2)
    opened.call("ext_value", value)
    measured = np.frombuffer(bytes(opened.exports["memory"].read(opened.store, value, value + 8)), np.float64)
    scored = criterion(output[None], *(target[None] for target in targets))
    scored = scored[0] if isinstance(scored, tuple) else scored
    _close(name, "score a case", [float(scored)], measured.astype(np.float32))


def check_extension(
    extension: str | Path, configs: Collection[str | Path], runs: dict[str, list[dict[str, Any]] | None] | None = None
) -> list[str]:
    """Run every class of the app's own that ``configs`` name, in Python and from ``extension``, on the same random
    voxels: a transform and its inverse, a reduction, a ``patch_combine`` and a criterion. Returns the classpaths
    checked. A test-time draw is opened and not compared: the extension draws from the seed its step carries,
    where konfai draws from torch's generator. ``runs``, the classes a compiled program runs by the extension
    (:func:`classes_run`), leaves out those it folds into a graph.

    Raises ``AppMetadataError`` when the extension lacks a class, or returns other values than the Python class.
    """
    import yaml
    from konfai.data.augmentation import DataAugmentation
    from konfai.data.patching import Fold
    from konfai.data.reduction import Reduction
    from konfai.data.transform import Transform
    from konfai.metric.measure import Criterion
    from konfai.portable import clean, inference_patch

    checks = {
        Fold: _check_fold,
        Transform: _check_transform,
        Reduction: _check_reduction,
        Criterion: _check_criterion,
        # Opened and not compared: its values are the extension's own draw.
        DataAugmentation: lambda extension, name, _draw, params, _shape: _Opened(
            extension, TRANSFORM, name, params | {"seed": 0}
        ),
    }
    extension = Path(extension)
    checked: dict[tuple[str, str], None] = {}
    for config in configs:
        loaded = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        # The axes of the config's patch, which are the case's.
        root = next(iter(loaded or {}), "")
        rank = len(inference_patch(loaded, root).get("patch_size") or []) if isinstance(loaded, dict) else 0
        shape = _EXTENTS[-(rank or 3) :]
        for name, given in own_classes(loaded):
            if (built := _built(name, given, tuple(checks))) is None:
                continue
            instance, params = built
            # A class, or a binding of it, the program does not run by the extension.
            if runs is not None:
                bindings = runs.get(name, [])
                if bindings is not None and {k: clean(v) for k, v in given.items()} not in bindings:
                    continue
            # A class bound twice with other parameters is two things to check.
            bound = (name, json.dumps(params, sort_keys=True, default=str))
            if bound in checked:
                continue
            next(check for base, check in checks.items() if isinstance(instance, base))(
                extension, name, instance, params, shape
            )
            checked[bound] = None
    return list(dict.fromkeys(name for name, _ in checked))


def classes_run(program: Any) -> dict[str, list[dict[str, Any]] | None]:
    """The classes of the app's own a compiled program (a prediction or an evaluation program, a manifest) runs by its
    extension, each with the parameters each of its transform steps binds it with, or ``None`` for a class run with the
    parameters its config binds: a draw, a reduction, a manifest's ``blend``, a criterion."""
    from konfai.portable import own_class

    found: dict[str, list[dict[str, Any]] | None] = {}
    if isinstance(program, list):
        for item in program:
            for name, bound in classes_run(item).items():
                _merge(found, name, bound)
    if isinstance(program, dict):
        for key, value in program.items():
            if key in ("class", "blend") and isinstance(value, str) and own_class(value):
                # A draw's step carries its own seed and no `prob`: any binding of it.
                params = program.get("params") or {}
                step = key == "class" and program.get("op") == "extension" and "seed" not in params
                _merge(found, value, [params] if step else None)
            for name, bound in classes_run(value).items():
                _merge(found, name, bound)
    return found


def _merge(found: dict[str, list[dict[str, Any]] | None], name: str, bound: list[dict[str, Any]] | None) -> None:
    """``name`` run with the bindings ``bound`` too (``None``: any)."""
    if (name in found and found[name] is None) or bound is None:
        found[name] = None
    else:
        found[name] = [*(found.get(name) or []), *bound]
