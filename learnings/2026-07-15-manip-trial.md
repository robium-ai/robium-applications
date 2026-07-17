# Learnings — 2026-07-15 (manip-trial demo page build)

- [lerobot] `lerobot[viz]==0.6.0` pins `rerun-sdk>=0.24.0,<0.34.0`, which is
  unsatisfiable next to `gradio_rerun==0.34.1` (needs rerun-sdk==0.34.1).
  Exact error: `Because lerobot[viz]==0.6.0 depends on rerun-sdk>=0.24.0,<0.34.0
  … your project's requirements are unsatisfiable.` Fix that passed: drop the
  `viz` extra and pin `rerun-sdk==0.34.1` explicitly (vla-trial's pyproject
  already encodes this — the extra-less shape is why its resolve works).
  Dead-end ruled out: keeping `viz` and downgrading gradio_rerun — there is no
  gradio_rerun release for rerun <0.34 with the streaming API the demo uses.
  Verified by: `uv sync` succeeding after the change (and the demo tests later
  in the session). Candidate for the lerobot skill's platform gotchas
  (seen 1x; vla-trial hit the adjacent shape on 07-14, so effectively 2x).

- [integration] In a demo Dockerfile, `uv pip install --system -e .` — the
  `-e` is load-bearing, not style: an app whose config derives APP_ROOT from
  `config.py.__file__` (manip-trial, vla-trial both do) resolves to
  site-packages under a non-editable install, and every baked `outputs/` path
  silently misses. Caught at plan-review time by reading vla-trial's working
  Dockerfile; no failing build needed. Worth a line in the integration (or
  live-demo) skill's container patterns.

- [live-demo] CPU-only demo images built with plain `uv pip install` pull the
  default linux torch wheel with the full CUDA dependency train
  (nvidia-cudnn/cusolver/... — multi-GB) that a CPU container never uses.
  vla-trial's demo image has the same trait, so this is seen 2x. Candidate
  fix for the skill's container patterns: install torch from the
  `https://download.pytorch.org/whl/cpu` index in demo Dockerfiles. Not
  applied this session (build already green; size, not correctness).

- [environments] `pymunk` (via `lerobot[pusht]`) ships no linux/arm64 wheel —
  a python:3.12-slim image fails at `uv pip install` with
  `error: command 'gcc' failed: No such file or directory`. Fix that passed:
  `apt-get install build-essential` in the image (verified: rebuild
  completed). Dead-end ruled out: pinning older pymunk (6.x line has no
  arm64 manylinux wheels either).
