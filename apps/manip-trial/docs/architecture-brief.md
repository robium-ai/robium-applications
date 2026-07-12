# Architecture Brief — manip-trial

**Date:** 2026-07-12   **Status:** built — pass bar met (smoke test green); §8 updated with build outcomes
**Author:** robium-architect subagent; refined during the build (main session)

## 1. Requirements

All inputs confirmed by the operator at kickoff (none assumed):

- **Robot type:** simulated arm/manipulator only. No physical robot exists or is planned.
- **Task:** learn a manipulation policy — a small-scale train (or fine-tune) from an existing dataset, then evaluate in sim producing metrics (success rate / avg reward over N episodes). Smoke-scale: minutes to tens of minutes on this host, explicitly not SOTA.
- **Hardware:** sim only; sim-to-real out of scope.
- **GPU:** none (no NVIDIA). Host is a MacBook Pro, Apple M2 Pro, 12 cores, 32 GB RAM, macOS 26.5, arm64. PyTorch MPS backend available. Isaac Sim / Isaac Lab are hard-gated out (NVIDIA-RTX floor unmet, no macOS support).
- **Local vs remote:** local Mac is the primary and only confirmed target. No remote GPU server exists — remote training is recorded as an optional path in §8, not a dependency.
- **Env tooling:** uv 0.11.8 on host; Docker Desktop available but not needed (pure-Python stack, no ROS).
- **Pass bar (repo README, trial 2):** a training run completes; eval produces metrics; smoke-scale. Test-driven (smoke test is the done bar); virtual-environment-first with identical local/remote reproduction.

## 2. Chosen stack + reasoning

| Layer | Choice | Version | Why (and what was rejected) |
|---|---|---|---|
| Middleware | none | — | Stack-selection Decision 1, "No" branch: pure learning/policy problem with no runtime robot middleware — LeRobot owns the whole loop. ROS 2 rejected for the MVP: it adds a Docker requirement on macOS and buys nothing for a train+eval pipeline. Can be added later if a hardware phase ever appears. |
| Simulator / env | `gym-pusht` (LeRobot-shipped sim, `--env.type=pusht`) | gym-pusht >=0.1.5,<0.2.0 (pulled by the `lerobot[pusht]` extra; pin verified in lerobot v0.6.0 pyproject) | PushT is 2D pymunk physics + pygame `rgb_array` rendering — no MuJoCo, no GL backend gymnastics, macOS-arm64-clean, and the fastest env LeRobot ships (96×96 images, 10 fps, 2-dim action). Rejected: **Isaac Sim/Lab** (GPU floor unmet, no macOS — hard gate); **LIBERO** (`hf-libero` dependency carries `sys_platform == 'linux'` marker in lerobot 0.6.0's pyproject — literally uninstallable here); **gym-aloha** (MuJoCo, 50 fps, 14-dim bimanual, 480×640 obs — viable on macOS but ~an order of magnitude more compute per step; wrong choice for a smoke-scale MPS run). |
| Learning framework | LeRobot | **0.6.0, pinned** (PyPI latest, released 2026-07-06; verified against PyPI JSON + simple index 2026-07-12 — note: the robium `lerobot` skill says 0.6.1, which does not exist on PyPI) | The architect skill's manipulation golden path. Supports ACT/Diffusion/VLA policies, ships `lerobot-train`/`lerobot-eval` CLIs, sim envs, Hub integration. Requires Python >=3.12, torch >=2.7,<2.12 (MPS-capable wheels on macOS by default). Rejected: Isaac Lab (GPU gate); hand-rolled gym+BC script (re-invents dataset/train/eval plumbing the pass bar needs). |
| Policies | Eval baseline: **Diffusion Policy** (pretrained `lerobot/diffusion_pusht`). Smoke train: **ACT** from scratch (`--policy.type=act`) on `lerobot/pusht` | shipped with lerobot 0.6.0 | Follows the lerobot skill's "start from pretrained" directive: the pretrained diffusion checkpoint validates the eval pipeline with zero training risk; ACT is the smallest/fastest policy family (constant LR, no scheduler to rescale when shrinking `--steps`) so it's the right smoke-train choice. Rejected for the smoke train: diffusion from scratch (slower per step, LR-scheduler-coupled), VLA families (VRAM/compute far beyond this host). |
| Visualization | rerun (via `lerobot[viz]` extra → `rerun-sdk`, `lerobot-dataset-viz`) + eval MP4s written by `lerobot-eval` | rerun-sdk >=0.24,<0.34 (lerobot 0.6.0 pin) | Rerun is what current LeRobot itself wraps for dataset/episode inspection (`lerobot-dataset-viz`). Eval rollout videos need no display at all: `lerobot-eval` renders `rgb_array` frames and writes MP4s to `<output_dir>/videos` (up to 10 episodes), verified in `lerobot_eval.py` at the v0.6.0 tag. Rejected: foxglove (aimed at ROS/remote-headless robots — no ROS here); rviz2 (ROS-only, Linux). |
| Environment | uv | uv 0.11.8 on host; Python pinned 3.12 | `environments` decision tree: pure-Python ML stack, no ROS, no apt deps → uv, not Docker. One system-level exception: `ffmpeg` via Homebrew for video decode (see §5, risk in §8). Docker rejected: nothing needs it, and MPS is not usable from inside Docker on macOS anyway. |

## 3. Module breakdown

Pruned LeRobot scaffold (per architect `references/scaffold-patterns.md`; LeRobot's CLIs do the heavy lifting, so `src/` stays thin):

```
apps/manip-trial/
├── docs/architecture-brief.md      # this file (fixed location)
├── pyproject.toml                  # uv project; lerobot[pusht,training,viz]==0.6.0
├── uv.lock                         # committed
├── Makefile                        # thin targets: baseline-eval, train-smoke, eval-trained, smoke
├── src/manip_trial/
│   └── configs/                    # smoke-run parameters (steps, episodes, seeds) in one place
├── tests/
│   └── test_smoke.py               # the pass-bar test (see §7 phase "Testing" and the shape below)
├── outputs/                        # checkpoints, eval_info.json, videos (gitignored)
├── data/                           # only if anything is materialized outside the HF cache (gitignored)
└── README.md
```

Pipeline stages (each a Make target wrapping one LeRobot CLI call):

| Stage | Command | Input → Output |
|---|---|---|
| Env setup | `uv sync` | pyproject/uv.lock → `.venv` |
| Sanity check | `uv run lerobot-info` | env → stdout |
| Baseline eval (pipeline validation, no training risk) | `uv run lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht ...` | Hub checkpoint (~1.0 GB) → `outputs/eval/baseline/eval_info.json` + videos |
| Smoke train | `uv run lerobot-train --dataset.repo_id=lerobot/pusht --policy.type=act --policy.device=mps --steps=<small> ...` | Hub dataset (~186 MB) → `outputs/train/act_pusht_smoke/checkpoints/...` |
| Eval trained checkpoint | `uv run lerobot-eval --policy.path=outputs/train/.../pretrained_model ...` | checkpoint → `outputs/eval/smoke/eval_info.json` + videos |
| Smoke test | `uv run pytest tests/test_smoke.py` | runs train+eval at tiny scale, asserts exit codes + metrics file |

## 4. Comms plan

Trivial, and deliberately so: single host, no middleware, no networked components. Each stage is one CLI process; hand-offs are files on disk — the HF cache (`~/.cache/huggingface/lerobot/`) for datasets/models, checkpoint directories under `outputs/train/`, and `eval_info.json` + MP4s under `outputs/eval/`. No topics, services, or RPC of any kind. If a remote training path is ever added (§8), the interface stays file/Hub-based (push checkpoint to Hub, pull for local eval).

## 5. Environment strategy

- **uv project** (`environments` skill's pure-Python branch). `uv python pin 3.12` (LeRobot requires >=3.12; 3.12 is the documented floor — don't chase 3.13 for the trial). Install: `uv add "lerobot[pusht,training,viz,diffusion]==0.6.0"` (the `diffusion` extra was added during the build: evaluating any diffusion checkpoint hard-fails without `diffusers`). Extras verified against the v0.6.0 pyproject: `pusht` → gym-pusht + pymunk; `training` → accelerate/wandb; `viz` → rerun-sdk (for `lerobot-dataset-viz`). `core_scripts` is NOT needed (it serves record/replay/calibrate — teleop/recording is out of scope). Add `pytest` as a dev-group dependency for the smoke test.
- **Version pin discipline:** `lerobot==0.6.0` exact-pinned in pyproject; `uv.lock` committed. Torch resolves to >=2.7,<2.12 macOS-arm64 wheels (MPS included by default — no index gymnastics needed off-CUDA).
- **One system dep:** `ffmpeg` (`brew install ffmpeg`) for video decode. LeRobot 0.6.0's `dataset` extra pulls `torchcodec >=0.3,<0.12` on darwin/arm64 plus `av` (PyAV 15.x) — torchcodec links against system ffmpeg libs, PyAV bundles its own. Documented in README as the single manual host step; does not justify Docker. See §8 for the ffmpeg-major-version pairing risk.
- **Device flag:** `--policy.device=mps` everywhere by default; `cpu` is the documented fallback (smoke-scale stays feasible on CPU, just slower). `--policy.use_amp=false` on MPS.
- **Local == remote parity:** local Mac is the only confirmed target. The parity story is `uv.lock` — the same `uv sync && uv run ...` commands reproduce on any future Linux box, with two declared deltas: `--policy.device` (mps → cuda) and torchcodec's platform markers (handled automatically by uv's universal resolution in the lockfile). No hardcoded paths; HF cache location is the library default on both.

## 6. Data plan

- **Training dataset:** `lerobot/pusht` (Hub). Verified 2026-07-12 via its `meta/info.json`: LeRobotDataset **v3.0** (compatible with lerobot 0.6.0 — no migration needed), 206 episodes / 25,650 frames, 10 fps, 96×96 RGB video (AV1 codec) + 2-dim state/action. Download ≈ **186 MB** (Hub API `usedStorage`). Cached at `~/.cache/huggingface/lerobot/`. Note: the dataset *card* metadata still says v2.0 — the card is stale; `meta/info.json` is authoritative.
- **Pretrained baseline:** ~~`lerobot/diffusion_pusht`~~ → **`ebenl08/diffusion_pusht_migrated`** (build outcome: the official repo lacks 0.6-format processor files and cannot load — see §8 item 3; the migrated community copy carries the same weights plus `policy_pre/postprocessor.json`). Original verification note kept below:
  `lerobot/diffusion_pusht` (Hub). Verified to exist; `model.safetensors` ≈ **1.0 GB** + `config.json` (the two files `--policy.path` needs). It is the checkpoint used in `lerobot_eval.py`'s own docstring at v0.6.0, so upstream treats it as current — but it was last modified 2025-03, see §8.
- **No pushes:** `--policy.push_to_hub=false` on all training runs (also avoids the `repo_id`-required validation in `TrainPipelineConfig`, verified at v0.6.0). No HF auth required for any of the above (public repos).
- **No teleop/recording, no sim-generated data** — offline Hub dataset only (`data` skill decision: the smallest sourcing story that meets the pass bar).
- **Gitignore boundary:** `outputs/`, `data/`, `.venv/`, `wandb/` ignored; `pyproject.toml`, `uv.lock`, configs, tests committed.

## 7. Robium skills per build phase

| Phase | Skill(s) | Notes |
|---|---|---|
| Env setup (uv project, Python pin, ffmpeg) | `environments` | uv-patterns reference; record any friction with the pinned install |
| Framework install + baseline eval + smoke train + eval | `lerobot` | Quick start maps 1:1 to this app's stages; its `examples/train-act-command.md` is the template for the smoke train (adapted: device=mps, smaller steps) |
| Hub pulls (dataset + checkpoint) | `huggingface` | delegation to `hf-cli@huggingface-skills`; public repos, no auth needed |
| Dataset sourcing decision | `data` | already made in this brief (offline Hub dataset); load only if the plan changes |
| Dataset/episode inspection | `visualization` → `rerun` | `lerobot-dataset-viz --repo-id=lerobot/pusht --episode-index=0`; eval videos are plain MP4s, no tool needed |
| Testing (smoke test = done bar) | `testing` | policy-eval layer of the test pyramid; deterministic seeds |

**Smoke-test shape** (the pass bar, one command: `uv run pytest tests/test_smoke.py`):

1. **Tiny train:** `lerobot-train --dataset.repo_id=lerobot/pusht --policy.type=act --policy.device=mps --steps=200 --batch_size=8 --save_freq=200 --log_freq=50 --policy.push_to_hub=false --seed=1000 --output_dir=outputs/train/act_pusht_smoke` → assert exit 0 and the checkpoint's `pretrained_model/` dir (with `config.json` + `model.safetensors`) exists. (Defaults verified at v0.6.0: steps=100k, batch=8, save_freq=20k — all overridden; `save_freq == steps` guarantees an end-of-run checkpoint.)
2. **Tiny eval:** `lerobot-eval --policy.path=<that checkpoint> --env.type=pusht --eval.n_episodes=2 --eval.batch_size=2 --seed=1000 --policy.device=mps --policy.use_amp=false --output_dir=outputs/eval/smoke` → assert exit 0, `eval_info.json` exists and contains numeric `pc_success` and `avg_sum_reward` (metric names verified in `lerobot_eval.py` at v0.6.0).
3. **No success-rate threshold** — a 200-step ACT policy is not expected to succeed; the smoke bar is *pipeline completes + metrics produced*, exactly the trial pass bar. The meaningful-metrics run is the manual baseline eval of `lerobot/diffusion_pusht` (10 episodes).

**Wall-clock budget (estimate, unverified on this host — see §8):** first run ≈ 10–25 min including the ~1.2 GB of downloads; warm runs ≈ 5–10 min (ACT 200 steps on MPS estimated low single-digit minutes; 2 eval episodes ≈ 1–3 min).

## 8. Open risks — build outcomes (2026-07-12)

The build resolved or confirmed every risk below; kept for the record with outcomes inline:

1. **Confirmed** — 0.6.0 pinned; skill's 0.6.1 claim logged as a learning.
2. **Resolved, opposite direction** — MPS ran ACT at 11.6 steps/s (200-step train in 31 s); no float64 issue at eval; the skill's MPS-slowness expectation is stale, not optimistic.
3. **Confirmed exactly, then escalated** — `lerobot/diffusion_pusht` cannot load on 0.6.0 (no `policy_preprocessor.json`; no fallback in `make_pre_post_processors`). The community-migrated copy (`ebenl08/diffusion_pusht_migrated`) loads with sane normalizer stats but evals at **chance level** (pc_success 0%, trajectories bit-identical across cpu/mps — the pipeline itself was verified healthy by step-probing our own ACT checkpoint). Conclusion: no working pretrained PushT baseline exists on the Hub for 0.6.0; the baseline stage was repointed at our own 10k-step ACT run (`make train-baseline` + `make baseline-eval`).
4. **Resolved** — torchcodec 0.11.1 + Homebrew ffmpeg 8.1.2 decoded the AV1 dataset with zero errors.
5. **Resolved** — layout is `checkpoints/000200/pretrained_model/` + `checkpoints/last` pointer; `config.latest_checkpoint()` handles both.
6. **Resolved** — measured: smoke test 39.5 s warm end-to-end; cold adds ~200 MB dataset + model downloads. Estimates were ~10× conservative.
7. **Unchanged** — still no remote GPU; HF Jobs remains the documented option.
8. **Resolved at smoke scale** — batch 2 eval + ACT train fit trivially in 32 GB.

**New finding (not a §8 risk):** `lerobot-eval`'s async-vector-env default is broken for pusht (forkserver workers never import `gym_pusht` → `NamespaceNotFound` → `BrokenPipeError`); all eval invocations set `--eval.use_async_envs=false`. Also, `eval_info.json`'s 0.6.0 schema nests metrics under `overall` (not the older `aggregated`).

## 8a. Original open risks (pre-build)

1. **Version-fact drift between the robium `lerobot` skill and PyPI.** The skill states lerobot 0.6.1; PyPI's index (checked 2026-07-12) tops out at **0.6.0** (2026-07-06). Pinning 0.6.0. Blocks nothing; resolve by re-checking PyPI at build time and logging the learning for skill-hardening. All CLI/flag facts in this brief were re-verified at the v0.6.0 tag, not taken from the skill.
2. **MPS backend gaps.** Historical LeRobot issue #143 (diffusion_pusht eval on Apple Silicon hit the MPS-no-float64 wall) predates the current CLI and is likely fixed, but MPS op coverage is not execution-verified here for lerobot 0.6.0. Blocks: train/eval commands as written. De-risk: `--policy.device=cpu` fallback is documented in every Make target; smoke scale keeps CPU viable. Community reports of extreme MPS slowness exist for diffusion-policy *training* — mitigated by training ACT, not diffusion.
3. **Pretrained checkpoint age.** `lerobot/diffusion_pusht` was last modified 2025-03 — before the 0.6.0 breaking changes. Upstream's own v0.6.0 eval docstring still uses it (strong signal it loads), but this is not execution-verified. Blocks: baseline-eval stage only. De-risk: the pass bar deliberately rests on evaluating *our own* freshly-trained ACT checkpoint; the baseline eval is validation sugar and can be dropped without failing the trial.
4. **ffmpeg/torchcodec pairing on macOS.** torchcodec dynamically links system ffmpeg and is strict about supported ffmpeg major versions; Homebrew currently ships a new ffmpeg major. lerobot 0.6.0 allows torchcodec up to <0.12 and also carries PyAV as a fallback decoder, but the exact working combination on this host is unverified. Blocks: dataset video decode during training. De-risk: try default `brew install ffmpeg` first; fall back to `ffmpeg@7` or rely on the PyAV path; the dataset's AV1 codec decodes via libdav1d in both. Log whatever combination works as a learning.
5. **Checkpoint directory layout unverified.** The `outputs/train/.../checkpoints/last/pretrained_model` path comes from the robium skill's unverified example; the `last` pointer's exact name at 0.6.0 was not confirmed. Blocks: smoke-test assertion path. De-risk: assert on a glob (`checkpoints/*/pretrained_model`) in the test, confirm the real layout on first run.
6. **Wall-clock estimates are paper numbers.** Nothing has been timed on this M2 Pro. Blocks: the "minutes-to-tens-of-minutes" promise. De-risk: `--steps` and `--eval.n_episodes` are the two knobs, both in one config place; first real run calibrates them.
7. **No remote GPU exists.** Any "train longer/bigger" ambition beyond smoke scale has no confirmed home. Optional paths, in order: Hugging Face Jobs (`lerobot-train --job.target=<flavor>`, paid), or a future Linux CUDA box reusing the same uv.lock with `--policy.device=cuda`. Neither is a dependency of the trial; recorded here so nobody designs against phantom hardware.
8. **Memory headroom on MPS.** 32 GB unified memory is shared with the OS; eval `--eval.batch_size=10` (upstream default example) spawns 10 parallel envs plus a ~1 GB diffusion model. Blocks: baseline eval at upstream-default sizes. De-risk: smoke uses batch_size=2; scale up only after watching memory on the first baseline run.
