"""Demo tests. The EpisodeRunner test and the gateway smoke are both marked
slow (real checkpoint loads); run via `make demo-smoke`, not the default suite.
"""

import pytest
import rerun as rr

pytestmark = pytest.mark.slow


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
