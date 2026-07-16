"""Single source of truth for manip-trial run parameters.

The smoke test (tests/test_smoke.py) and the Makefile's manual targets both
build their CLI invocations from these values, so the pass-bar run and the
hand-run stages can never drift apart.
"""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]

DATASET_REPO_ID = "lerobot/pusht"
# No usable pretrained Hub baseline exists for lerobot 0.6.0: the official
# lerobot/diffusion_pusht predates the processor-pipeline format and cannot
# load, and the community-migrated copy evals at chance level (see
# learnings/2026-07-12.md). The baseline is therefore our own 10k-step ACT.
ENV_TYPE = "pusht"

DEVICE = "mps"  # documented fallback: cpu (see docs/architecture-brief.md §8.2)
SEED = 1000

# Smoke train: enough steps to prove the loop, not to learn the task.
SMOKE_STEPS = 200
SMOKE_BATCH_SIZE = 8
SMOKE_SAVE_FREQ = 200  # == SMOKE_STEPS so an end-of-run checkpoint is guaranteed
SMOKE_LOG_FREQ = 50

# Smoke eval: tiny but real rollouts with metrics.
SMOKE_EVAL_EPISODES = 2
SMOKE_EVAL_BATCH_SIZE = 2

# Baseline: a real (still small) ACT run whose eval metrics mean something.
BASELINE_STEPS = 10_000
BASELINE_SAVE_FREQ = 10_000
BASELINE_EVAL_EPISODES = 10
BASELINE_EVAL_BATCH_SIZE = 5

TRAIN_OUTPUT_DIR = APP_ROOT / "outputs" / "train" / "act_pusht_smoke"
BASELINE_TRAIN_OUTPUT_DIR = APP_ROOT / "outputs" / "train" / "act_pusht_10k"
SMOKE_EVAL_OUTPUT_DIR = APP_ROOT / "outputs" / "eval" / "smoke"
BASELINE_EVAL_OUTPUT_DIR = APP_ROOT / "outputs" / "eval" / "baseline"


def _train_cmd(steps: int, save_freq: int, log_freq: int, output_dir: Path) -> list[str]:
    return [
        "lerobot-train",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        "--policy.type=act",
        f"--policy.device={DEVICE}",
        "--policy.push_to_hub=false",
        f"--steps={steps}",
        f"--batch_size={SMOKE_BATCH_SIZE}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        f"--seed={SEED}",
        f"--output_dir={output_dir}",
    ]


def train_smoke_cmd() -> list[str]:
    return _train_cmd(SMOKE_STEPS, SMOKE_SAVE_FREQ, SMOKE_LOG_FREQ, TRAIN_OUTPUT_DIR)


def train_baseline_cmd() -> list[str]:
    return _train_cmd(BASELINE_STEPS, BASELINE_SAVE_FREQ, 500, BASELINE_TRAIN_OUTPUT_DIR)


def eval_cmd(policy_path: str, n_episodes: int, batch_size: int, output_dir: Path) -> list[str]:
    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        f"--env.type={ENV_TYPE}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        # async default is broken: forkserver workers never import gym_pusht
        # (NamespaceNotFound -> BrokenPipeError). Sync is fine at smoke scale.
        "--eval.use_async_envs=false",
        f"--seed={SEED}",
        f"--policy.device={DEVICE}",
        "--policy.use_amp=false",
        f"--output_dir={output_dir}",
    ]


def latest_checkpoint(train_dir: Path = TRAIN_OUTPUT_DIR) -> Path:
    """Resolve the trained policy dir, preferring the 'last' pointer if present."""
    ckpt_root = train_dir / "checkpoints"
    last = ckpt_root / "last" / "pretrained_model"
    if last.is_dir():
        return last
    candidates = sorted(ckpt_root.glob("*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {ckpt_root}")
    return candidates[-1]


# --- checkpoint ladder (demo spec: docs/superpowers/specs/2026-07-15-manip-trial-demo-page-design.md)
# One 5k run saving every 1k, pruned to 3 rungs; the 10k baseline (a separate,
# earlier run — labeled as such in the UI) is the free top rung.
LADDER_STEPS = 5_000
LADDER_SAVE_FREQ = 1_000
LADDER_KEEP_STEPS = (1_000, 3_000, 5_000)
LADDER_EVAL_EPISODES = 10
LADDER_EVAL_BATCH_SIZE = 5
LADDER_TRAIN_OUTPUT_DIR = APP_ROOT / "outputs" / "train" / "act_pusht_ladder"
LADDER_EVAL_OUTPUT_DIR = APP_ROOT / "outputs" / "eval" / "ladder"
DEMO_LADDER_MANIFEST = APP_ROOT / "outputs" / "demo" / "ladder.json"


def train_ladder_cmd() -> list[str]:
    return _train_cmd(LADDER_STEPS, LADDER_SAVE_FREQ, 500, LADDER_TRAIN_OUTPUT_DIR)


def ladder_rungs() -> list[dict]:
    """The demo's checkpoint ladder, weakest to strongest."""
    rungs = [
        {
            "name": f"{s // 1000}k",
            "steps": s,
            "run": "ladder run (5k steps, 2026-07-15)",
            "checkpoint": LADDER_TRAIN_OUTPUT_DIR / "checkpoints" / f"{s:06d}" / "pretrained_model",
        }
        for s in LADDER_KEEP_STEPS
    ]
    rungs.append(
        {
            "name": "10k",
            "steps": BASELINE_STEPS,
            "run": "baseline run (10k steps, 2026-07-12)",
            "checkpoint": BASELINE_TRAIN_OUTPUT_DIR / "checkpoints" / f"{BASELINE_STEPS:06d}" / "pretrained_model",
        }
    )
    return rungs


# --- demo gateway (spec: docs/superpowers/specs/2026-07-15-manip-trial-demo-page-design.md)
DEMO_PORT = int(os.environ.get("PORT", "8765"))
DEMO_SESSION_SECONDS = 1800  # mirrors demos.json sessionSeconds
DEMO_FLEET_BUDGET = 2  # mirrors demos.json maxInstances (ACT-CPU is light)
DEMO_DEFAULT_RUNG = "10k"  # strongest rung loads at boot; others load lazily


def demo_device() -> str:
    """mps native, cpu in the container — resolved at runtime, not import."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
