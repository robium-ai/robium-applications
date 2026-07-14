# vla-trial

Language-conditioned VLA control of a simulated SO-101 arm: type an instruction
("put the red cube in the box") and a SmolVLA policy drives a simulated
SO-101 robot arm in MuJoCo to do it.

**Task 1 (this state):** pure scaffolding — the uv project and the vendored
SO-101 asset (menagerie's `robotstudio_so101`) proven to load under our
MuJoCo. Later tasks build the environment, an oracle, a dataset, and the
policy.

## Setup

```bash
make sync    # uv sync
make assets  # vendor the menagerie SO-101 MJCF (sparse-checkout, not committed)
make test    # uv run pytest tests/ -v
```

On macOS (Apple Silicon), headless MuJoCo rendering requires:

```bash
export MUJOCO_GL=cgl
```

(`osmesa`/`egl` are Linux-only.)

See `docs/architecture-brief.md` for the stack decision.
