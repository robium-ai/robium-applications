"""Guard the training command builders and prove the loop assembles.

The command-shape tests are fast and stay in the default suite — they lock down
the flags a real spend depends on (a typo'd `--job.target` would silently run a
paid job locally, or a local smoke run would try to reach a GPU). The actual
5-step CPU loop-start is `slow` (it loads the 450M base policy) and runs via
`make train-smoke`, not in CI.
"""

import subprocess

import pytest

from vla_trial.config import (
    BASE_POLICY_ID,
    DATASET_REPO_ID,
    PIPE_TEST_JOB_TARGET,
    POLICY_REPO_ID,
    train_remote_cmd,
    train_smoke_cmd,
)


def test_smoke_cmd_is_local_and_never_pushes():
    """The pre-spend smoke run must be CPU-local and must not touch the Hub."""
    cmd = " ".join(train_smoke_cmd())
    assert "--policy.device=cpu" in cmd
    assert "--policy.push_to_hub=false" in cmd
    # A local smoke run must NOT carry a remote job target.
    assert "--job.target=" not in cmd


def test_remote_pipe_test_targets_hf_jobs_and_pushes():
    cmd = " ".join(train_remote_cmd(pipe_test=True))
    assert f"--job.target={PIPE_TEST_JOB_TARGET}" in cmd
    assert "--policy.push_to_hub=true" in cmd
    assert f"--policy.repo_id={POLICY_REPO_ID}" in cmd
    assert f"--dataset.repo_id={DATASET_REPO_ID}" in cmd
    assert f"--policy.path={BASE_POLICY_ID}" in cmd


def test_remote_never_targets_a_local_device():
    """The remote job picks its own GPU; it must not inherit mps/cpu from us."""
    for pipe_test in (True, False):
        cmd = " ".join(train_remote_cmd(pipe_test=pipe_test))
        assert "--policy.device=mps" not in cmd
        assert "--policy.device=cpu" not in cmd


def test_pipe_test_and_full_differ_in_steps():
    """The cheap run and the real run must not be accidentally identical."""
    pipe = " ".join(train_remote_cmd(pipe_test=True))
    full = " ".join(train_remote_cmd(pipe_test=False))
    assert pipe != full
    assert "--steps=2000" in pipe
    assert "--steps=20000" in full


@pytest.mark.slow
def test_train_loop_starts():
    """A handful of CPU steps — proves config/shape/dtype before any GPU spend."""
    result = subprocess.run(
        train_smoke_cmd(), capture_output=True, text=True, timeout=1800
    )
    assert result.returncode == 0, result.stderr[-4000:]
