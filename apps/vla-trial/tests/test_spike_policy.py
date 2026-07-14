import json

from vla_trial.config import POLICY_SPIKE_JSON
from vla_trial.spike.bench_policy import bench_policy


def test_bench_policy_cpu_produces_latency(tmp_path):
    """CPU is the number that decides the architecture (Docker/macOS has no MPS).

    Writes to a tmp file, NOT to POLICY_SPIKE_JSON: this 3-pass smoke run must
    never overwrite the canonical 20-pass M0 benchmark artifact.
    """
    out = tmp_path / "policy.json"
    result = bench_policy(device="cpu", n_passes=3, output_json=out)

    assert result["device"] == "cpu"
    assert result["mean_s"] > 0
    assert result["n_passes"] == 3

    # A real forward pass, not a cached action-chunk lookup. SmolVLA serves ~50
    # select_action() calls from one chunk; without policy.reset() before each
    # timed call we would be timing a deque pop (sub-millisecond).
    assert result["mean_s"] > 0.01, "suspiciously fast — are we timing a cache hit?"

    # Results are keyed by device AND runtime, so a container run cannot clobber
    # the native one in the shared bind-mounted outputs/ dir.
    written = json.loads(out.read_text())
    assert result["runtime"] in ("native", "container")
    assert f"cpu-{result['runtime']}" in written


def test_bench_policy_does_not_touch_the_canonical_artifact(tmp_path):
    """Guards the bug this file's tmp_path usage exists to prevent: a throwaway
    test run silently overwriting the real M0 benchmark in outputs/spike/."""
    before = POLICY_SPIKE_JSON.read_text() if POLICY_SPIKE_JSON.is_file() else None
    bench_policy(device="cpu", n_passes=1, output_json=tmp_path / "throwaway.json")
    after = POLICY_SPIKE_JSON.read_text() if POLICY_SPIKE_JSON.is_file() else None
    assert before == after, "the test run overwrote the canonical M0 benchmark artifact"
