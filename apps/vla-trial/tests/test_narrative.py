"""Task 10 — the narrative comparison, reframed as a HARNESS test, not a claim.

The demo's eventual claim is "base SmolVLA flails (<20%), the fine-tuned
checkpoint clears 60%." Asserting that today would be FALSE: the real
20k-step fine-tune is deferred (user directive 2026-07-15, no further spend),
and the only checkpoints that exist are a 5-step local CPU smoke checkpoint
and a 100-step remote GPU pipe-test checkpoint — both essentially untrained,
both scoring ~0% (see tests/test_smoke.py and .superpowers/sdd/progress.md).

So this test does NOT assert the claim. It proves the 3-way comparison
MACHINERY is ready: `evaluate()` can run two DIFFERENT checkpoints and
produce two independent, well-formed results in two distinct output
directories, without one run clobbering the other. That is everything the
eventual narrative script needs from the eval layer; only the checkpoint
quality is missing.

Both stand-in checkpoints are real, distinct, HF-Jobs-trained artifacts
(config.NARRATIVE_FINETUNED_STANDIN is genuinely 20x more steps than
SMOKE_CHECKPOINT_PATH) — this is not the same checkpoint evaluated twice.
What it is NOT: the raw, un-fine-tuned base model. See config.py's
NARRATIVE_FINETUNED_STANDIN comment for why `BASE_POLICY_ID` can't stand in
for "base" through this same `evaluate()` path today (empty_cameras mismatch).

Kept CHEAP on purpose: n_episodes=2 (config.NARRATIVE_EPISODES), not the real
SMOKE_EVAL_EPISODES=10 — loading the 450M model twice is already the
expensive part; this is a harness check, not a scored eval run.

Marked `slow`: it loads the 450M SmolVLA checkpoint (twice). Not run by
default `make test`.

TODO(full-training): once the 20k-step checkpoint exists (POLICY_REPO_ID),
repoint `finetuned` below to it, add a real base-model eval path (see
config.py's NARRATIVE_FINETUNED_STANDIN note — BASE_POLICY_ID needs its own
empty_cameras=0-compatible eval, not this same evaluate() call), and assert
the real claim:
    assert base["success_rate"] < 0.2 <= finetuned["success_rate"]
    assert finetuned["success_rate"] >= SUCCESS_RATE_FLOOR
"""

import json

import pytest

from vla_trial.config import (
    NARRATIVE_BASE_OUTPUT_DIR,
    NARRATIVE_EPISODES,
    NARRATIVE_FINETUNED_OUTPUT_DIR,
    NARRATIVE_FINETUNED_STANDIN,
    SEED,
    SMOKE_CHECKPOINT_PATH,
)
from vla_trial.policy.evaluate import evaluate

pytestmark = pytest.mark.slow


def test_comparison_harness_evaluates_two_checkpoints_to_distinct_dirs():
    base = evaluate(
        policy_path=str(SMOKE_CHECKPOINT_PATH),
        n_episodes=NARRATIVE_EPISODES,
        seed=SEED,
        output_dir=NARRATIVE_BASE_OUTPUT_DIR,
    )
    finetuned = evaluate(
        policy_path=NARRATIVE_FINETUNED_STANDIN,
        n_episodes=NARRATIVE_EPISODES,
        seed=SEED,
        output_dir=NARRATIVE_FINETUNED_OUTPUT_DIR,
    )

    # The harness contract: both runs produce a well-formed numeric rate...
    for result, n in ((base, NARRATIVE_EPISODES), (finetuned, NARRATIVE_EPISODES)):
        assert result["n_episodes"] == n
        assert result["successes"] == sum(ep["success"] for ep in result["episodes"])
        assert 0.0 <= result["success_rate"] <= 1.0

    # ...and each landed in ITS OWN eval_info.json — no clobbering between the
    # two legs of the comparison, the exact bug a shared output dir would hide.
    assert NARRATIVE_BASE_OUTPUT_DIR != NARRATIVE_FINETUNED_OUTPUT_DIR
    base_written = json.loads((NARRATIVE_BASE_OUTPUT_DIR / "eval_info.json").read_text())
    finetuned_written = json.loads(
        (NARRATIVE_FINETUNED_OUTPUT_DIR / "eval_info.json").read_text()
    )
    assert base_written["success_rate"] == base["success_rate"]
    assert finetuned_written["success_rate"] == finetuned["success_rate"]
    assert base_written["policy"] != finetuned_written["policy"]
