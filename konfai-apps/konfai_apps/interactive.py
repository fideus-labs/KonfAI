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

"""An interactive session: one app predicting again and again, its model kept loaded, for a front end that predicts
after each prompt (3D Slicer's interactive mode, a server's session).

``python -m konfai_apps.interactive <app> [--gpu 0] [--host <server> --port ... --token ... --https --ca-cert ...]``
prints ``konfai-interactive: ready`` once the app is resolved, then reads one JSON request a line on its standard
input, ``{"id": <n>, "inputs": [[<file>], ...], "output": "<folder>"}``, the inputs in the app's order (the image
first), and answers ``konfai-interactive: done <n> <seconds>`` once the prediction is in ``output``, or
``konfai-interactive: error <n> <why>``. Anything else it prints is the runs' log.

Without ``--host`` the app runs in this process, its model kept from one prediction to the next (``KONFAI_KEEP_MODEL``).
With it, the session is a konfai-apps server's: the image is sent once (again when it changes) and each request sends
the other inputs and receives the prediction.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from konfai import RemoteServer

from .cli import add_remote_arguments, remote_server

PREFIX = "konfai-interactive:"
IMAGE_SUFFIXES = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd")
SESSION_PREDICT_SECONDS = 3600.0  # a server's deadline for one prediction, the first building and loading the model


def _compressed(path: Path, target: Path) -> Path:
    """``path`` as a compressed MetaImage at ``target`` when it is an image (a mask is mostly zeros), else itself."""
    if not path.name.lower().endswith(IMAGE_SUFFIXES):
        return path
    import SimpleITK as sitk

    sitk.WriteImage(sitk.ReadImage(str(path)), str(target), True)
    return target


class SessionGone(RuntimeError):
    """The server no longer holds the session: closed idle, its worker stopped, the server restarted."""


class RemoteSession:
    """A session on a konfai-apps server: ``POST /apps/<app>/sessions`` with the image, ``POST
    /sessions/<id>/predict`` with the other inputs (answered with the prediction), ``DELETE /sessions/<id>``."""

    def __init__(self, server: RemoteServer, app: str, gpu: list[int]) -> None:
        self.server, self.app, self.gpu = server, app, gpu
        self.session_id: str | None = None
        self.image: tuple[str, int, int] | None = None  # (path, size, mtime) of the image the session holds

    def _call(self, method: str, path: str, timeout: float, **kwargs: Any):
        import requests

        response = requests.request(
            method,
            f"{self.server.get_url()}{path}",
            headers=self.server.get_headers(),
            verify=self.server.verify,
            timeout=(30, timeout),
            allow_redirects=False,  # the token is not sent elsewhere
            **kwargs,
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            gone = path.startswith("/sessions/") and response.status_code in (404, 410)
            raise (SessionGone if gone else RuntimeError)(f"{method} {path}: {response.status_code} {detail}")
        return response

    def _open(self, image: Path, signature: tuple[str, int, int] | None, scratch: Path) -> None:
        from urllib.parse import quote

        self.close()
        upload = _compressed(image, scratch / "image.mha")
        with open(upload, "rb") as file:
            answer = self._call(
                "POST",
                f"/apps/{quote(self.app, safe='/:@')}/sessions",  # an app#preset's preset is no URL fragment
                3600,  # the server resolves the app first: a download
                files={"image": (upload.name, file)},
                data={"gpu": ",".join(str(g) for g in self.gpu)},
            ).json()
        self.session_id, self.image = answer["session_id"], signature

    def predict(self, inputs: list[list[str]], output: Path) -> None:
        image = Path(inputs[0][0])
        stat = image.stat()  # before any upload: a file changed meanwhile is sent again at the next prediction
        # A .mhd's voxels are in a file of their own, which its signature does not follow: it is sent each time.
        signature = None if image.name.lower().endswith(".mhd") else (str(image), stat.st_size, stat.st_mtime_ns)
        with tempfile.TemporaryDirectory() as scratch:
            if self.session_id is None or signature is None or self.image != signature:
                self._open(image, signature, Path(scratch))
            uploads = [
                _compressed(Path(group[0]), Path(scratch) / f"input{index}.mha")
                for index, group in enumerate(inputs[1:])
            ]
            files = [("inputs", (path.name, path.read_bytes())) for path in uploads]
            try:
                answer = self._call(
                    "POST", f"/sessions/{self.session_id}/predict", SESSION_PREDICT_SECONDS + 120, files=files
                )
            except SessionGone:  # opened again on the image, and asked again once
                self._open(image, signature, Path(scratch))
                answer = self._call(
                    "POST", f"/sessions/{self.session_id}/predict", SESSION_PREDICT_SECONDS + 120, files=files
                )
        output.mkdir(parents=True, exist_ok=True)
        (output / "Prediction.mha").write_bytes(answer.content)

    def close(self) -> None:
        if self.session_id is not None:
            with suppress(Exception):  # a session the server no longer holds is closed already
                self._call("DELETE", f"/sessions/{self.session_id}", 60)
        self.session_id = self.image = None


def serve(app: str, gpu: list[int], server: RemoteServer | None) -> None:
    """Answer the requests read on the standard input until it closes."""
    if server is None:
        os.environ.setdefault("KONFAI_KEEP_MODEL", "1")  # built and loaded at the first prediction, kept for the next
        from .app import KonfAIApp

        local = KonfAIApp(app, False, False)
        device: dict[str, Any] = {"gpu": gpu} if gpu else {"cpu": 1}

        def predict(inputs: list[list[str]], output: Path) -> None:
            local.infer(inputs, output, quiet=True, **device)

    else:
        remote = RemoteSession(server, app, gpu)
        predict = remote.predict
    print(PREFIX, "ready", flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            start = time.perf_counter()
            try:
                predict(request["inputs"], Path(request["output"]))
            except (Exception, SystemExit) as error:  # one prediction that fails leaves the session to the next
                print(PREFIX, "error", request["id"], " ".join(str(error).split()) or type(error).__name__, flush=True)
                continue
            print(PREFIX, "done", request["id"], f"{time.perf_counter() - start:.2f}", flush=True)
    finally:
        if server is not None:
            remote.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m konfai_apps.interactive", description=__doc__.split("\n\n")[0])
    parser.add_argument("app", help="KonfAI App: repo_id:app_name or a local app folder")
    parser.add_argument("--gpu", type=int, nargs="*", default=[], help="GPU indices (none: the CPU)")
    add_remote_arguments(parser)  # with --host, the session is that server's
    arguments = vars(parser.parse_args())
    server = remote_server(arguments)
    serve(arguments["app"], arguments["gpu"], server)


if __name__ == "__main__":
    main()
