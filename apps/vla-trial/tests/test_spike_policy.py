from vla_trial.spike.bench_policy import bench_policy


def test_bench_policy_cpu_produces_latency():
    """CPU is the number that decides the architecture (Docker/macOS has no MPS)."""
    result = bench_policy(device="cpu", n_passes=3)
    assert result["device"] == "cpu"
    assert result["mean_s"] > 0
    assert result["n_passes"] == 3
