# manip-trial

Manipulation-vertical MVP trial for the [robium](https://github.com/robium-ai/robium)
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
| `make train-ladder` | 5k-step ACT run saving every 1k, pruned to rungs 1k/3k/5k → `outputs/train/act_pusht_ladder/` (~8 min MPS). |
| `make eval-ladder` | 10-episode seeded eval of every rung (incl. the reused 10k baseline) → `outputs/eval/ladder/` + `outputs/demo/ladder.json`. |
| `make demo` | The live-demo gateway, native/MPS, on :8765 (needs the ladder built). |
| `make demo-image` | Build `manip-trial:latest` (CPU) — checkpoints baked from local outputs, no HF token. |
| `make demo-smoke` | The demo ship bar: boot → ready, session guards, one episode completes via the Gradio API, shutdown. |

Measured on the M2 Pro (2026-07-12): smoke test green twice consecutively
(39.5 s / 43.0 s); 10k-step baseline trained in 14.5 min (loss 14.4 → 0.33)
and evaluated at `avg_max_reward` 0.283 over 10 episodes (`pc_success` 0% —
expected: PushT success needs ≥95% coverage, beyond ACT at this scale).

All run parameters (steps, episodes, seed, device) live in one place:
`src/manip_trial/config.py`. Device is `mps`; set `DEVICE = "cpu"` there as
the fallback if MPS misbehaves.

## Live demo

The demo page (spec: `docs/superpowers/specs/2026-07-15-manip-trial-demo-page-design.md`
at the repo root) serves a checkpoint-ladder story: pick a rung (1k/3k/5k from one
training run + the 10k baseline), watch ACT push the T-block live on an embedded
Rerun timeline, and browse each rung's real eval videos in the Gallery tab.
v1 is local-only — the robium-website page's Start button spawns
`manip-trial:latest` via the local orchestrator (`npm run dev` there), or run
`make demo` and use the page's direct-host mode. Honest numbers: `pc_success`
is 0% at every rung (PushT success needs ≥95% coverage), and the ladder is not
monotonic — the 5k rung out-evals the older 10k baseline run (0.474 vs 0.283
avg_max_reward); the UI shows the real numbers per rung.

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
