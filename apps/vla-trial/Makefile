.PHONY: sync assets test test-all spike-test clean spike-render spike-policy spike-policy-container oracle

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
