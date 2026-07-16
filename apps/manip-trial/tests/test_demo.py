"""Demo tests. The EpisodeRunner test and the gateway smoke are both marked
slow (real checkpoint loads); run via `make demo-smoke`, not the default suite.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest
import rerun as rr

pytestmark = pytest.mark.slow

PORT = 8798  # NOT 8765 (a dev gateway may be up) and NOT 8799 (vla-trial's test port)
BASE = f"http://127.0.0.1:{PORT}"
BOOT_TIMEOUT_S = 180
EPISODE_TIMEOUT_S = 300


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
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "manip_trial.demo.gateway"],
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


def test_refresh_reclaims_claim(gateway):
    code, body = _http("POST", "/start?session=carol")
    assert code == 200 and body["ok"] is True
    code, st = _http("GET", "/status?session=carol")
    assert code == 200 and st["claimed"] is True
    code, _ = _http("GET", "/status?session=alice")
    assert code == 409


def test_episode_runner_completes_episode():
    from manip_trial.demo.episode_runner import EpisodeRunner

    runner = EpisodeRunner()
    assert "10k" in runner.rungs and len(runner.rungs) == 4

    rec = rr.RecordingStream(application_id="manip_trial_test", recording_id="t0")
    events = list(runner.run("1k", rec))

    assert events, "no StepEvents yielded"
    last = events[-1]
    assert last.done is True
    assert last.aborted is False
    assert 0.0 <= last.max_reward <= 1.0
    assert last.step <= last.total <= 300
    assert not runner.busy
