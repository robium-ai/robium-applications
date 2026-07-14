.PHONY: sync assets test clean spike-render spike-policy spike-policy-container oracle

sync:
	uv sync

assets:
	bash scripts/fetch_assets.sh

test:
	uv run pytest tests/ -v

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
# CURRENTLY RED (0/10) — the SO-101 pincer cannot grasp menagerie's 4x6cm box.
# See .superpowers/sdd/task-5-report.md before touching this.
oracle:
	uv run python -m vla_trial.run oracle 10
