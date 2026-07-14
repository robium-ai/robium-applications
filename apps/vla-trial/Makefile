.PHONY: sync assets test clean spike-render

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
