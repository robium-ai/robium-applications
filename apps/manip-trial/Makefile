.PHONY: sync info train-baseline baseline-eval train-smoke eval-trained smoke clean

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

# The pass bar: train + eval at tiny scale, asserted.
smoke:
	uv run pytest tests/test_smoke.py -v

clean:
	rm -rf outputs
