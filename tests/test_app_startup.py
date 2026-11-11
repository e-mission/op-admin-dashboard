import http.client
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

APP_DIR = pathlib.Path(__file__).resolve().parent.parent
STARTUP_TIMEOUT_SECS = 60


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_app(tmp_path):
    port = _free_port()
    log_path = tmp_path / "app.log"
    env = dict(os.environ, DASH_SERVER_PORT=str(port), DASH_DEBUG_MODE="False")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "app_sidebar_collapsible.py"],
            cwd=APP_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        yield proc, port, log_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_app_starts_and_serves_index(running_app):
    proc, port, log_path = running_app
    deadline = time.time() + STARTUP_TIMEOUT_SECS
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"App exited with code {proc.returncode}:\n{log_path.read_text()}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            # Talisman redirects plain HTTP to HTTPS unless the request looks proxied over HTTPS
            conn.request("GET", "/", headers={"X-Forwarded-Proto": "https"})
            status = conn.getresponse().status
            conn.close()
            assert status == 200
            return
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    pytest.fail(f"App did not respond on port {port} within {STARTUP_TIMEOUT_SECS}s")
