# vla-trial

Language-conditioned VLA control of a simulated SO-101 arm: type an instruction
("put the green cube in the bin") and a fine-tuned SmolVLA policy drives a
MuJoCo SO-101 robot arm to do it.

**Current state (2026-07-15): the pipeline is validated end-to-end. There is
no working policy yet.** Everything from scene to scripted oracle to dataset
recording to remote fine-tuning to evaluation has run for real and produced
correct, expected output at every stage — but the fine-tune has only run for
100 steps (a pipe-test, not training), so it scores ~0%, exactly as a
100-step checkpoint should. The real 20k-step run (the one expected to clear
the 60% bar) is deferred, not done — see "What's actually proven" below.

## Quick "is it alive" path

```bash
make sync    # uv sync
make assets  # vendor the menagerie SO-101 MJCF (sparse-checkout, not committed)
make oracle  # scripted IK oracle, 10 episodes — the day-1 canary
```

`make oracle` is the fastest real signal in this repo: a hand-written
inverse-kinematics controller (no learning involved) that picks the cube and
drops it in the bin using ground-truth state. It is **10/10 on tuned seeds**
(seeds 0-9; ~85-92% on unseen seeds — see Battle scars). If this ever drops
below 10/10 on seeds 0-9, something in the scene/physics/success-predicate
broke, independent of any policy question.

```bash
export MUJOCO_GL=cgl   # required for headless MuJoCo rendering on Apple Silicon;
                        # osmesa/egl are Linux-only and will not work here.
make test    # the default regression suite (excludes `slow`)
```

## The full pipeline

Each stage is a `Makefile` target wrapping `python -m vla_trial.run <cmd>`;
`src/vla_trial/config.py` is the single source of truth for every run
parameter (steps, batch size, paths, seeds) — the Makefile and the tests both
build their invocations from it, so a hand-run stage and the pass-bar test
can never drift apart.

| Stage | Command | What it does |
| --- | --- | --- |
| Oracle canary | `make oracle` | scripted IK picks + drops, 10 episodes, ground-truth state — the smoke test for the *scene*, not the policy |
| Visual spot-check | `make viz-oracle` | one oracle episode logged to Rerun (`outputs/viz/oracle.rrd`) |
| Record | `make record` | run the oracle 75x, discard/retry episodes it fails, save a `LeRobotDataset` locally (no Hub push) |
| Push | `make push-dataset` | push the reviewed local dataset to the Hub (private) — a separate, deliberate step from `record` |
| Train (pipe-test) | `make train` | submit a cheap, deliberately under-trained fine-tune to HF Jobs (100 steps, a10g-small, ~$1-2) — proves the remote loop, not a real policy |
| Train (real) | `make train-full` | the actual 20k-step fine-tune (~4h on an A100-class GPU, ~$20-40) — **not yet run** |
| Eval | `make eval CKPT=<repo_id_or_path>` | roll a checkpoint out in sim, N seeded episodes, write `outputs/eval/*/eval_info.json` with a numeric success rate |
| Pipe-test pass bar | `make smoke` | asserts the eval PIPELINE runs end-to-end on the local 5-step checkpoint (loads, rolls out, writes JSON) — NOT a `>=60%` score bar |
| Narrative harness | `MUJOCO_GL=cgl uv run pytest tests/test_narrative.py -m slow` | asserts the base-vs-fine-tuned COMPARISON MACHINERY works (two checkpoints, two distinct output dirs, both produce numeric rates) — not the demo's eventual claim; see the file's `TODO(full-training)` |

There is a free, local, CPU-only gate in front of every paid step:
`train-smoke` (a handful of CPU steps, ~2 min, catches config/shape errors)
must pass before `make train` ever touches HF Jobs money. This caught a real
bug for free once already (see Battle scars).

## What's actually proven vs. what's deferred

**Proven, for real, no shortcuts:**
- The scene, IK, and success predicate are correct: the oracle picks and
  releases the cube into the bin 10/10 on tuned seeds using real MuJoCo
  physics, not a scripted animation.
- The dataset is real: 75 clean episodes (9 discarded for oracle misses),
  recorded with real physics and pushed to the Hub.
- The **entire remote training loop** is real and has run to completion on
  GPU: submit to HF Jobs -> train -> save checkpoint -> push to Hub -> pull
  -> eval in sim. The checkpoint at `jazarium/train_2026-07-15_08-09-36`
  is a genuine artifact of that loop, not a placeholder.
- The eval pipeline is real: camera renaming, action un-normalization, and
  the sim rollout loop are all exercised against a real trained checkpoint,
  not mocked.

**Deferred, honestly:**
- **No checkpoint has been trained long enough to succeed.** The pipe-test
  checkpoint is 100 steps (a10g-small, ~$1-2) — essentially the base model —
  and scores 0/10 = 0%, which is the *correct* result for 100 steps, not a
  bug. The real fine-tune is `make train-full` (20k steps, ~$20-40,
  reference success rate 60-80%). The user made a deliberate cost decision
  to wrap up here rather than spend on the full run.
- The narrative test (`tests/test_narrative.py`) proves the comparison
  *harness* works, not the demo's eventual "base flails, fine-tuned clears
  60%" claim — that assertion is commented out with a grep-able
  `TODO(full-training)` until a real 20k-step checkpoint exists.

## Hard-won facts (read before you touch this app)

- **Never train on macOS.** MPS fine-tuning is ~2 hours per 20 steps — CPU is
  even worse. All training happens on HF Jobs (remote GPU); local MPS is for
  *inference/eval only*, where SmolVLA is fast (~0.55 s/forward-pass on
  Apple Silicon, ~17x faster than CPU on the same machine — see the M0 spike
  in `docs/architecture-brief.md`).
- **The pedestal + wrist-roll grasp is a matched, load-bearing pair.** The
  cube sits on a 0.06 m pedestal (`scene_pick.xml`) — without it, the arm's
  reach geometry forces a downward finger pitch that makes grasping
  structurally impossible, independent of any IK tuning. On top of that,
  *position-only IK leaves the wrist roll free*, and a free roll can leave
  the gripper's pinch axis vertical (trying to span the cube's 6cm height
  with a 4.2cm aperture) instead of horizontal (spanning its 4cm width).
  Both the pedestal height (0.06) and the grasp offset
  (`ORACLE_GRASP_LOCAL` in `config.py`) were swept end-to-end together — do
  not change one without re-sweeping the other.
- **Camera renaming is required at both train AND eval.** SmolVLA's base
  checkpoint expects exactly three cameras named `observation.images.camera1/
  2/3`; our env has two (`wrist`, `scene`). Fix:
  `--rename_map={"observation.images.wrist":"observation.images.camera1",
  "observation.images.scene":"observation.images.camera2"}` plus
  `--policy.empty_cameras=1` for a masked placeholder covering the missing
  third camera. The fine-tuned checkpoint's own saved
  `policy_preprocessor.json` bakes this rename in — at eval time you feed
  the env's raw `wrist`/`scene` keys and the checkpoint renames them
  internally; renaming yourself first would break it (see
  `src/vla_trial/policy/evaluate.py`'s module docstring).
- **Remote `--output_dir` must be a container path, never a local Mac
  path.** `--output_dir` is passed verbatim to the HF Jobs container. A
  local absolute path trains to completion and then crashes at checkpoint
  save with `PermissionError: '/Users'` — a full paid run for zero artifact.
  `REMOTE_OUTPUT_DIR` in `config.py` is a `/tmp/...` container path for
  exactly this reason.
- **HF Jobs needs prepaid credits, separately from being logged in**, and a
  correctly-formed submission fails with `402 Payment Required` if the
  account has none — a safe, free way to validate the whole submission
  (auth, dataset, command shape) before adding credits.
- **HF Jobs ignores `--policy.repo_id` for the final push.** The trained
  model lands at an auto-generated `<user>/train_<timestamp>` repo instead —
  you have to read the "Model pushed to `<url>`" line in the job log to find
  the real checkpoint; you cannot assume it landed where you asked.

## Full docs

- `docs/architecture-brief.md` — stack rationale, M0 spike results
  (render throughput + SmolVLA latency), pipe-test results, and the full
  `Battle scars` section.
- `../../REGISTRY.md` — this app's registry card (bootstrap-for list,
  battle scars index).
- `../../learnings/2026-07-14-vla-trial.md`,
  `../../learnings/2026-07-15-vla-trial.md` — the session-by-session
  friction log this README and the brief are distilled from.
