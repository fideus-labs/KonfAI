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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .errors import AppMetadataError

#: The kinds an extension opens a class as, numbered as the extension numbers them.
PATCH_COMBINE, TRANSFORM, REDUCTION, CRITERION = range(4)
#: The spatial extent of the volumes compared.
_SHAPE = (3, 4, 5)


class _Opened:
    """One class of an extension, opened in an instance of its own: the calls of the extension's interface."""

    def __init__(self, extension: Path, kind: int, name: str, params: dict[str, Any]) -> None:
        import wasmtime

        engine = wasmtime.Engine()
        self.store = wasmtime.Store(engine)
        module = wasmtime.Module.from_file(engine, str(extension))
        self.exports: Any = wasmtime.Instance(self.store, module, []).exports(self.store)
        refused = self.call("ext_open", kind, *self._text(name), *self._text(json.dumps(params)))
        if refused:
            reason = "implements no such class" if refused == 1 else "implements it as another kind"
            raise AppMetadataError(f"The extension {extension.name} {reason}: '{name}'.")

    def call(self, name: str, *arguments: int) -> int:
        return int(self.exports[name](self.store, *arguments) or 0)

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
        at = self.buffer(array.size)
        self.exports["memory"].write(self.store, array.tobytes(), at)
        return at

    def get(self, at: int, shape: tuple[int, ...]) -> Any:
        """The float32 volume of ``shape`` at ``at``."""
        import numpy as np

        data = self.exports["memory"].read(self.store, at, at + 4 * math.prod(shape))
        return np.frombuffer(bytes(data), np.float32).reshape(shape)


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


def _check_transform(extension: Path, name: str, stage: Any, params: dict[str, Any]) -> None:
    import torch
    from konfai.data.transform import TransformInverse
    from konfai.utils.dataset import Attribute

    volume = torch.rand((1, *_SHAPE), generator=torch.Generator().manual_seed(0)) * 100
    opened = _Opened(extension, TRANSFORM, name, params)
    pixels = math.prod(_SHAPE)

    def run(values: Any, inverse: bool) -> Any:
        channels = opened.call("ext_output", 1, 1)
        out = opened.buffer(channels * pixels)
        if not opened.call("ext_apply", opened.put(values), 1, out, pixels, int(inverse)):
            raise AppMetadataError(f"The extension's '{name}' has no inverse, and the Python class is undone.")
        return opened.get(out, (channels, *_SHAPE))

    attribute = Attribute()  # one for both ways: a stage may record on it what its inverse reads back
    transformed = stage("check", volume.clone(), attribute)
    _close(name, "transform a volume", transformed, run(volume, False))
    if isinstance(stage, TransformInverse) and stage.apply_inverse and transformed.shape == volume.shape:
        _close(name, "undo itself", stage.inverse("check", transformed.clone(), attribute), run(transformed, True))


def _check_reduction(extension: Path, name: str, fold: Any, params: dict[str, Any]) -> None:
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    copies = [torch.rand((1, *_SHAPE), generator=generator) for _ in range(3)]
    opened = _Opened(extension, REDUCTION, name, params)
    pixels, channels = math.prod(_SHAPE), opened.call("ext_output", 1, len(copies))
    out = opened.buffer(channels * pixels)
    table = opened.put(np.array([opened.put(copy) for copy in copies]), "uint32")
    opened.call("ext_reduce", table, len(copies), 1, out, pixels)
    _close(name, "fold the copies", fold([copy[None] for copy in copies])[0], opened.get(out, (channels, *_SHAPE)))


def _check_fold(extension: Path, name: str, fold: Any, params: dict[str, Any]) -> None:
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    # What a patch enters the fold as depends on the head's channels, which no config states: the first count the
    # class takes.
    for head in (2, 3, 8, 32, 256, 1024):
        try:
            patches = [fold.enter(torch.rand((head, *_SHAPE), generator=generator)) for _ in range(3)]
            break
        except (IndexError, RuntimeError):
            continue
    else:
        raise AppMetadataError(f"'{name}' enters no patch of 2 to 1024 channels, so its extension is not checked.")
    held = torch.zeros_like(patches[0])
    opened = _Opened(extension, PATCH_COMBINE, name, params)
    pixels, channels = math.prod(_SHAPE), int(held.shape[0])
    held_at, runs = opened.buffer(channels * pixels), opened.put(np.zeros(2), "uint32")
    for patch in patches:
        held = fold.fold(held, patch)
        opened.call("ext_fold", held_at, pixels, opened.put(patch.float()), pixels, channels, runs, 1, pixels)
    written = opened.call("ext_output", channels, 1)
    out = opened.buffer(written * pixels)
    opened.call("ext_leave", held_at, pixels, out, pixels, channels, pixels)
    _close(name, "combine the patches", fold.leave(held), opened.get(out, (written, *_SHAPE)))


def _check_criterion(extension: Path, name: str, criterion: Any, params: dict[str, Any]) -> None:
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    # As many targets as the class's forward names, one for a variadic `*targets`.
    named = [
        parameter
        for parameter in inspect.signature(criterion.forward).parameters.values()
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD and parameter.default is parameter.empty
    ]
    output, *targets = (torch.rand((1, *_SHAPE), generator=generator) for _ in range(max(len(named), 2)))
    opened = _Opened(extension, CRITERION, name, params)
    table = opened.put(np.array([[opened.put(target), 1] for target in targets]), "uint32")
    opened.call("ext_measure", opened.put(output), 1, table, len(targets), math.prod(_SHAPE))
    value = opened.buffer(2)
    opened.call("ext_value", value)
    measured = np.frombuffer(bytes(opened.exports["memory"].read(opened.store, value, value + 8)), np.float64)
    scored = criterion(output[None], *(target[None] for target in targets))
    scored = scored[0] if isinstance(scored, tuple) else scored
    _close(name, "score a case", [float(scored)], measured.astype(np.float32))


def check_extension(extension: str | Path, configs: list[str | Path]) -> list[str]:
    """Run every class of the app's own that ``configs`` name, in Python and from ``extension``, on the same random
    voxels: a transform and its inverse, a reduction, a ``patch_combine`` and a criterion. Returns the classpaths
    checked. A test-time draw is opened and not compared: the extension draws from the seed its step carries,
    where konfai draws from torch's generator.

    Raises ``AppMetadataError`` when the extension lacks a class, or returns other values than the Python class.
    """
    import yaml
    from konfai.data.augmentation import DataAugmentation
    from konfai.data.patching import Fold
    from konfai.data.reduction import Reduction
    from konfai.data.transform import Transform
    from konfai.metric.measure import Criterion

    checks = {
        Fold: _check_fold,
        Transform: _check_transform,
        Reduction: _check_reduction,
        Criterion: _check_criterion,
        # Opened and not compared: its values are the extension's own draw.
        DataAugmentation: lambda extension, name, _draw, params: _Opened(
            extension, TRANSFORM, name, params | {"seed": 0}
        ),
    }
    extension = Path(extension)
    checked: dict[tuple[str, str], None] = {}
    for config in configs:
        for name, given in own_classes(yaml.safe_load(Path(config).read_text(encoding="utf-8"))):
            if (built := _built(name, given, tuple(checks))) is None:
                continue
            instance, params = built
            # A class bound twice with other parameters is two things to check.
            bound = (name, json.dumps(params, sort_keys=True, default=str))
            if bound in checked:
                continue
            next(check for base, check in checks.items() if isinstance(instance, base))(
                extension, name, instance, params
            )
            checked[bound] = None
    return list(dict.fromkeys(name for name, _ in checked))
