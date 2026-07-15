# manip-trial

Manipulation-vertical MVP trial for the [robium](https://github.com/jazarium/robium-plugin)
plugin: a small-scale imitation-learning pipeline — ACT trained on the
`lerobot/pusht` dataset, evaluated in the PushT sim — running entirely on a
GPU-less Mac (Apple Silicon, MPS). See `docs/architecture-brief.md` for the
stack decision and open risks.

**Pass bar (repo README, trial 2):** a training run completes; eval produces
metrics; smoke-scale, not SOTA.

## Setup

One manual host step (video decode for the dataset's AV1 streams):

```bash
brew install ffmpeg
```

Everything else is uv-managed and pinned (`lerobot==0.6.0`, Python 3.12):

```bash
uv sync
uv run lerobot-info   # sanity check
```

## Run

| Command | What it does |
|---|---|
| `make smoke` | **The pass bar.** 200-step ACT train on MPS + 2-episode seeded eval, asserted via pytest (~40 s warm, plus ~200 MB of downloads cold). |
| `make train-smoke` | The training stage alone → `outputs/train/act_pusht_smoke/`. |
| `make eval-trained` | Eval the freshly trained checkpoint → `outputs/eval/smoke/eval_info.json` + rollout MP4s. |
| `make train-baseline` | 10k-step ACT train (~15 min on M2 Pro MPS) → `outputs/train/act_pusht_10k/`. |
| `make baseline-eval` | Meaningful-metrics run: eval the 10k checkpoint over 10 episodes → `outputs/eval/baseline/`. |

Measured on the M2 Pro (2026-07-12): smoke test green twice consecutively
(39.5 s / 43.0 s); 10k-step baseline trained in 14.5 min (loss 14.4 → 0.33)
and evaluated at `avg_max_reward` 0.283 over 10 episodes (`pc_success` 0% —
expected: PushT success needs ≥95% coverage, beyond ACT at this scale).

All run parameters (steps, episodes, seed, device) live in one place:
`src/manip_trial/config.py`. Device is `mps`; set `DEVICE = "cpu"` there as
the fallback if MPS misbehaves.

## Gotchas encountered (details in `learnings/2026-07-12.md`)

- `lerobot-eval` defaults to async vector envs whose forkserver workers never
  import `gym_pusht` → `NamespaceNotFound`/`BrokenPipeError`. All eval
  commands here pass `--eval.use_async_envs=false`.
- No usable pretrained PushT baseline exists on the Hub for lerobot 0.6.0:
  the official `lerobot/diffusion_pusht` predates the processor-pipeline
  format and cannot load (`policy_preprocessor.json` missing), and the
  community-migrated copy loads but evals at chance level. The baseline here
  is our own 10k-step ACT run instead.
- Evaluating any diffusion checkpoint needs the `diffusion` extra (installed
  here); ACT needs none.
