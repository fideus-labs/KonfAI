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

"""``impact_reg_sampled``, the konfai-apps runner a server runs for a client's ``--host`` with ``--out-of-core
sampled`` (``POST /apps/{app}/run``): the sampled registration of the cases it was sent, their fields and compact
transforms in ``--output``, the moved images left to the client, as on its own machine."""

import argparse
from pathlib import Path

#: What a client's ``--arg NAME=VALUE`` may set, and how it is read: never a path.
_ARGS = {"set": str, "samples": int, "epochs": int, "region_size": int, "groups": str, "tta": int}
_GROUPS = ("Fixed", "Moving", "FixedMask", "MovingMask")


def main(argv: list[str]) -> None:
    from konfai_apps.cli import app_id

    from .impact_reg import IMPACT_REG_KONFAI_REPO, ImpactRegKonfAIApp

    parser = argparse.ArgumentParser(prog="impact_reg_sampled")
    parser.add_argument("app")
    parser.add_argument("--inputs", action="append", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, nargs="*", default=[])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--arg", action="append", default=[])
    args = parser.parse_args(argv)
    named: dict[str, list] = {}
    for arg in args.arg:
        name, _, value = arg.partition("=")
        if name not in _ARGS:
            raise SystemExit(f"impact_reg_sampled: no argument '{name}'.")
        named.setdefault(name, []).append(_ARGS[name](value))
    # The app the server checked against its list, registered as it names it: a preset of this server's repository.
    preset = Path(args.app).name if Path(args.app).exists() else args.app.rpartition(":")[2]
    if app_id(IMPACT_REG_KONFAI_REPO, preset) != args.app:
        raise SystemExit(f"impact_reg_sampled: {args.app} is no preset of {IMPACT_REG_KONFAI_REPO}, this server's.")
    names = named.get("groups", ["Fixed,Moving"])[-1].split(",")
    if not set(names) <= set(_GROUPS) or len(names) != len(args.inputs):
        raise SystemExit(f"impact_reg_sampled: groups {names} for {len(args.inputs)} input groups.")
    groups = {name: [Path(path) for path in paths] for name, paths in zip(names, args.inputs, strict=True)}

    def last(name: str):
        return named[name][-1] if name in named else None

    ImpactRegKonfAIApp().register(
        [preset],
        *(groups.get(group, []) for group in _GROUPS),
        output=args.output,
        gpu=args.gpu,
        cpu=args.cpu,
        quiet=args.quiet,
        config_overrides=named.get("set"),
        fields_only=True,
        out_of_core="sampled",
        region_size=last("region_size") or 64,
        samples=last("samples"),
        epochs=last("epochs"),
        tta=last("tta") or 0,
    )
