.PHONY: sync info train-baseline baseline-eval train-smoke eval-trained smoke clean train-ladder eval-ladder demo demo-image demo-smoke

sync:
	uv sync

info:
	uv run lerobot-info

# Meaningful-metrics baseline: 10k-step ACT train (~15 min on M2 Pro MPS)…
train-baseline:
	uv run python -m manip_trial.run train-baseline

# …then eval it over 10 episodes.
baseline-eval:
	uv run python -m manip_trial.run baseline-eval

# 200-step ACT train on lerobot/pusht (MPS).
train-smoke:
	uv run python -m manip_trial.run train-smoke

# Eval the freshly trained ACT checkpoint, 2 episodes.
eval-trained:
	uv run python -m manip_trial.run eval-trained

# Demo ladder: one 5k-step run saving every 1k, pruned to rungs 1k/3k/5k
# (~8 min on M2 Pro MPS). The 10k baseline run is reused as the top rung.
train-ladder:
	uv run python -m manip_trial.run train-ladder

# Eval every rung (10 seeded episodes each) + write outputs/demo/ladder.json.
eval-ladder:
	uv run python -m manip_trial.run eval-ladder

# --- demo page (spec: docs/superpowers/specs/2026-07-15-manip-trial-demo-page-design.md)

# The demo gateway, NATIVE (MPS inference). Pair with the website's dev
# backend dropdown: "direct localhost:8765". Needs the ladder built first
# (make train-ladder eval-ladder).
demo:
	uv run python -m manip_trial.demo.gateway

# Build the demo container (CPU inference). Checkpoints + ladder.json are
# baked from LOCAL outputs/ — no Hub access, no token.
demo-image:
	docker build -f docker/demo.Dockerfile -t manip-trial:latest .

# The live-demo skill's ship bar: gateway boots to DEMO READY, session guards
# hold (409/403), one episode COMPLETES through the Gradio API (completion is
# the assertion — success at this scale would be theater), /shutdown exits.
demo-smoke:
	uv run pytest tests/test_demo.py -v -m slow

# The pass bar: train + eval at tiny scale, asserted.
smoke:
	uv run pytest tests/test_smoke.py -v

clean:
	rm -rf outputs
