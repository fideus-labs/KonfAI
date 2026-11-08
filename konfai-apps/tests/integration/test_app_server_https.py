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
"""A server started with a certificate answers over HTTPS, to a client that trusts that certificate only."""

import datetime
import ipaddress
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from konfai import RemoteServer, check_server

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
x509 = pytest.importorskip("cryptography.x509")
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from konfai_apps.app_repository import get_available_apps_on_remote_server  # noqa: E402

_SERVE = "import sys; from konfai_apps import main_apps_server; sys.argv = sys.argv[1:]; main_apps_server()"


def _self_signed(directory: Path) -> tuple[Path, Path]:
    """A certificate for 127.0.0.1 that no system authority signed, and its key."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
        .sign(key, hashes.SHA256())
    )
    cert, key_file = directory / "cert.pem", directory / "key.pem"
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return cert, key_file


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_an_https_server_answers_the_client_that_trusts_its_certificate(tmp_path: Path) -> None:
    cert, key = _self_signed(tmp_path)
    apps = tmp_path / "apps.json"
    apps.write_text(json.dumps({"apps": []}))
    port = _free_port()
    argv = ["konfai-apps-server", "--host", "127.0.0.1", "--port", str(port), "--auth", "off", "--apps", str(apps)]
    process = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", _SERVE, *argv, "--ssl-certfile", str(cert), "--ssl-keyfile", str(key)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    trusting = RemoteServer("127.0.0.1", port, None, ca_cert=str(cert))
    try:
        deadline = time.monotonic() + 60
        while not check_server(trusting, timeout_s=0.5)[0]:
            assert process.poll() is None and time.monotonic() < deadline, "the server never answered over HTTPS"
            time.sleep(0.2)

        assert get_available_apps_on_remote_server(trusting) == []
        ok, message = check_server(RemoteServer("127.0.0.1", port, None, https=True))
        assert not ok and "CERTIFICATE_VERIFY_FAILED" in message
        assert not check_server(RemoteServer("127.0.0.1", port, None))[0]
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_a_certificate_without_its_key_is_refused(tmp_path: Path) -> None:
    cert, _ = _self_signed(tmp_path)
    apps = tmp_path / "apps.json"
    apps.write_text(json.dumps({"apps": []}))
    argv = ["konfai-apps-server", "--apps", str(apps), "--ssl-certfile", str(cert)]
    run = subprocess.run([sys.executable, "-c", _SERVE, *argv], capture_output=True, text=True)  # nosec B603
    assert run.returncode != 0 and "--ssl-keyfile" in run.stderr
