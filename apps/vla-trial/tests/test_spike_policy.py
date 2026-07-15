"""M0 spike guards. Marked `slow` — deselected from the default run.

Each test here does REAL SmolVLA forward passes at ~9 s apiece on CPU: together
they were 268 s, 56% of the whole suite, re-deriving numbers already committed
in docs/architecture-brief.md. They are a benchmark, not a regression test.

Run them deliberately: `make spike-test`.
"""

import json

import pytest

from vla_trial.config import POLICY_SPIKE_JSON
from vla_trial.spike.bench_policy import bench_policy

pytestmark = pytest.mark.slow


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


def test_cold_call_is_within_factor_of_2_of_steady_state(tmp_path):
    """Verifies, from shipped code rather than an out-of-band claim in the
    report, that the steady-state timed passes are real forward passes and
    not chunk-cache hits.

    A cache hit (``self._queues[ACTION].popleft()``) is sub-millisecond; a
    real forward pass on this checkpoint is single-digit seconds on CPU. If
    the "steady state" numbers were secretly serving cached actions, a truly
    cold call (zero warm-up) would be dramatically slower than them — not
    within 2x.
    """
    steady = bench_policy(
        device="cpu", n_passes=2, n_warmup_passes=1, output_json=tmp_path / "steady.json"
    )
    cold = bench_policy(
        device="cpu", n_passes=1, n_warmup_passes=0, output_json=tmp_path / "cold.json"
    )

    ratio = cold["mean_s"] / steady["mean_s"]
    assert 0.5 <= ratio <= 2.0, (
        f"cold call ({cold['mean_s']:.2f}s) vs steady-state mean "
        f"({steady['mean_s']:.2f}s) differ by {ratio:.2f}x — steady-state timings "
        "may be serving cached action-chunk lookups rather than real forward passes"
    )
