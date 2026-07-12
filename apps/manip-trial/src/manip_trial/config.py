"""Single source of truth for manip-trial run parameters.

The smoke test (tests/test_smoke.py) and the Makefile's manual targets both
build their CLI invocations from these values, so the pass-bar run and the
hand-run stages can never drift apart.
"""

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
