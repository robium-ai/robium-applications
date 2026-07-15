"""Demo gateway smoke — the live-demo skill's bar for shipping a demo:
gateway boots to DEMO READY, /start claims, intruder sessions are rejected
(409/403), one scripted (oracle) episode succeeds end-to-end THROUGH the
Gradio API, and /shutdown exits the process.

Marked slow (module-wide): boots the real gateway subprocess including the
SmolVLA checkpoint load. Run via `make demo-smoke`, not the default suite.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.slow

PORT = 8799  # not 8765: must not collide with a dev gateway the user left up
BASE = f"http://127.0.0.1:{PORT}"
BOOT_TIMEOUT_S = 300  # container CPU model load is the slow case
EPISODE_TIMEOUT_S = 240


def _http(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        f"{BASE}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("demo") / "gateway.log"
    env = {**os.environ, "PORT": str(PORT)}
    # macOS: cgl (thread-affine — see demo/episode_runner.py's docstring);
    # the demo container sets its own MUJOCO_GL in the Dockerfile.
    env.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "vla_trial.demo.gateway"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        text = log_path.read_text()
        if "DEMO READY" in text:
            break
        if "BOOT FAILED" in text or proc.poll() is not None:
            break
        time.sleep(2)
    else:
        proc.kill()
        pytest.fail(f"gateway never reached DEMO READY in {BOOT_TIMEOUT_S}s:\n{log_path.read_text()[-3000:]}")
    if "DEMO READY" not in log_path.read_text():
        proc.kill()
        pytest.fail(f"gateway boot failed:\n{log_path.read_text()[-3000:]}")
    yield proc
    if proc.poll() is None:
        proc.kill()


def test_status_ready_then_claim(gateway):
    code, st = _http("GET", "/status?session=alice")
    assert code == 200
    assert st["ready"] is True
    assert st["claimed"] is False
    assert st["fleet"]["budget"] >= 1

    code, body = _http("POST", "/start?session=alice")
    assert code == 200 and body["ok"] is True

    code, st = _http("GET", "/status?session=alice")
    assert code == 200 and st["claimed"] is True


def test_intruder_session_rejected(gateway):
    code, _ = _http("GET", "/status?session=bob")
    assert code == 409
    code, _ = _http("POST", "/shutdown?session=bob")
    assert code == 403


def test_oracle_episode_succeeds_via_gradio_api(gateway):
    code, sub = _http(
        "POST",
        "/ui/gradio_api/call/run_episode",
        {"data": ["oracle", "put the green cube in the bin"]},
    )
    assert code == 200 and "event_id" in sub, sub

    req = urllib.request.Request(f"{BASE}/ui/gradio_api/call/run_episode/{sub['event_id']}")
    final_status = None
    deadline = time.time() + EPISODE_TIMEOUT_S
    with urllib.request.urlopen(req, timeout=EPISODE_TIMEOUT_S) as r:
        for raw in r:
            if time.time() > deadline:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: null":
                continue
            payload = json.loads(line[len("data:"):])
            if isinstance(payload, list) and payload and isinstance(payload[-1], str):
                final_status = payload[-1]
                if "finished at step" in final_status:
                    break
    assert final_status is not None, "no status updates arrived on the SSE stream"
    assert "✅" in final_status, f"oracle episode did not succeed: {final_status!r}"


def test_shutdown_exits_process(gateway):
    code, body = _http("POST", "/shutdown?session=alice")
    assert code == 200 and body["bye"] is True
    deadline = time.time() + 10
    while time.time() < deadline and gateway.poll() is None:
        time.sleep(0.2)
    assert gateway.poll() is not None, "gateway process still alive after /shutdown"
