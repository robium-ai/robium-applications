.PHONY: sync assets test test-all spike-test clean spike-render spike-policy spike-policy-container oracle viz-oracle record push-dataset spot-check train-smoke train train-full

sync:
	uv sync

assets:
	bash scripts/fetch_assets.sh

# The regression suite. Excludes the `slow` M0 benchmarks — they re-derive
# already-committed numbers at ~9 s per SmolVLA forward pass and were 56% of
# the runtime. Benchmarks are not regression tests.
test:
	uv run pytest tests/ -v

# The M0 benchmark guards, on demand. Slow by design (real SmolVLA inference).
spike-test:
	uv run pytest tests/ -v -m slow

test-all:
	uv run pytest tests/ -v -m ""

clean:
	rm -rf outputs

# M0: MuJoCo offscreen render throughput (spec risk 2).
spike-render:
	uv run python -m vla_trial.run spike-render

# M0: SmolVLA forward-pass latency, native (spec risk 1).
spike-policy:
	uv run python -m vla_trial.run spike-policy cpu mps

# M0: the decisive number — CPU inside linux/arm64, because Docker on macOS
# cannot see MPS, so the demo container is CPU here AND on Cloud Run.
spike-policy-container:
	docker build -f docker/spike.Dockerfile -t vla-trial-spike:latest .
	docker run --rm -v $(PWD)/outputs:/app/outputs vla-trial-spike:latest

# The day-1 canary. The oracle sees ground truth; anything but N/N is a bug.
# Currently 10/10. Three things are load-bearing and coupled — the pedestal
# (scene_pick.xml), the wrist-roll solve (so the jaws pinch HORIZONTALLY), and
# the calibrated grasp offset. Changing any one without re-running the others'
# sweeps quietly degrades the pick. See .superpowers/sdd/task-5-report.md.
oracle:
	uv run python -m vla_trial.run oracle 10

# Task 6: one oracle episode (seed 0), logged to a Rerun .rrd for eyeballing.
# open with: uv run rerun outputs/viz/oracle.rrd
viz-oracle:
	uv run python -m vla_trial.run viz-oracle

# Task 7: record N clean episodes LOCALLY (discard/retry on oracle misses).
# No Hub push — the user reviews the dataset first.
record:
	uv run python -m vla_trial.run record

# Task 7: push the already-recorded LOCAL dataset to the Hub (private) — run
# ONLY after reviewing it. Not invoked by `record` or by any test.
push-dataset:
	uv run python -m vla_trial.run push-dataset

# Task 7: re-roll a handful of kept seeds through the oracle with RerunLogger
# attached, one .rrd per episode under outputs/viz/, for visual spot-checking.
spot-check:
	uv run python -m vla_trial.run spot-check

# Task 8: prove the training loop assembles — a few steps on CPU, free, ~2 min.
# NOT training (MPS is ~2 h per 20 steps). Catches config/shape errors before
# any GPU spend.
train-smoke:
	uv run python -m vla_trial.run train-smoke

# Task 8: submit the fine-tune to HF Jobs. Default is the CHEAP PIPE-TEST run
# (2k steps, a10g-small, ~$2-4) that proves the remote loop end-to-end. The real
# 20k-step run is `make train-full` — a separate, deliberate spend.
train:
	uv run python -m vla_trial.run train

train-full:
	uv run python -m vla_trial.run train full
