.PHONY: sync assets test clean

sync:
	uv sync

assets:
	bash scripts/fetch_assets.sh

test:
	uv run pytest tests/ -v

clean:
	rm -rf outputs
