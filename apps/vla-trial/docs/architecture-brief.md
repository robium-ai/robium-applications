# vla-trial — architecture brief

Language-conditioned VLA control of a simulated SO-101 arm: type an instruction
and a SmolVLA policy drives a MuJoCo SO-101 to carry it out.

This file is written incrementally as the milestones land. Task 11 adds the full
stack rationale and the `§8 Battle scars` section; Task 10 adds `## Results`.

## M0 spike results

**Date: 2026-07-13.** Both M0 spikes are now measured. Every later task's
assumption about *where inference runs* traces back to this section.

### Spike 2 — MuJoCo offscreen render throughput (spec risk 2): PASS

| Metric | Value |
| --- | --- |
| Offscreen render, 256x256, `MUJOCO_GL=cgl`, Apple Silicon | **84.0–84.9 fps** |
| Floor (`RENDER_FPS_FLOOR`, 2 cams x 30 FPS) | 60 fps |
| Verdict | **Clears with ~40% margin** |

`mj_step` costs 0.012 ms vs ~11.8 ms/render, so physics is free on this scene —
rendering is the cost, and it is affordable. ~42 control-steps/sec achievable at
2 cameras/step vs the 30 Hz target. Caveat: scene-specific; a heavier scene
should re-measure. **Rendering is not the bottleneck.**

### Spike 1 — SmolVLA forward-pass latency (spec risk 1): FAIL on CPU

The demo's core unverified assumption. HuggingFace's "runs on a MacBook" is a
marketing claim, not a measurement; no published SmolVLA-on-Apple-Silicon
benchmark exists. It does now.

`lerobot/smolvla_base` (450M), 20 timed passes after 5 warm-up passes, one
`policy.reset()` before every timed call so each call is a genuine forward pass
and not a cached action-chunk lookup. Reproduce with `make spike-policy` and
`make spike-policy-container`; raw data in `outputs/spike/policy.json`.

**MPS number is sync-corrected (2026-07-13 fix pass).** MPS ops are
asynchronous and nothing in the LeRobot `select_action` call chain forces a
wait, so a bare `time.perf_counter()` was measuring dispatch time, not
compute time. `bench_policy.py` now calls `torch.mps.synchronize()` (device-
agnostic `_sync()` helper, no-op on CPU) both immediately before starting the
timer and immediately before stopping it, around every warm-up and timed
pass. The corrected MPS mean moved from 0.536 s to 0.549 s (+2.4%) — a small,
expected shift, not a qualitative change to the verdict.

| Config | mean | median | p95 | vs 1 s ceiling |
| --- | --- | --- | --- | --- |
| **MPS, native** (uv, Apple Silicon, sync-corrected) | **0.549 s** | 0.547 s | 0.571 s | **PASS** (~1.8x headroom) |
| **CPU, native** (uv, Apple Silicon) | **9.004 s** | 8.953 s | 9.244 s | FAIL (9.0x over) |
| **CPU, linux/arm64 container** | **9.318 s** | 9.258 s | 9.621 s | **FAIL (9.3x over)** |

Native CPU was measured three independent times (9.159 / 9.186 / 9.004 s mean) —
the result is stable to within ~2%, so the verdict does not hinge on one run.
(CPU is synchronous already, so the sync fix does not change the CPU numbers;
a fourth CPU run during the fix pass measured 9.648 s, consistent with normal
run-to-run variance from other load on the machine, not a fix effect.)

Bar is `POLICY_LATENCY_CEILING_S = 1.0` s, not 30 Hz: action chunking means one
forward pass yields ~50 actions ≈ 1.7 s of robot motion at 30 FPS.

**The container number is the decisive one.** Docker on macOS cannot see MPS
(the container logs `No accelerated backend detected. Using default cpu`), and
Cloud Run has no GPU — so the deployed demo is CPU-in-a-container *both* locally
and in production. Containerising costs almost nothing over native CPU
(9.32 s vs ~9.0–9.2 s, a few percent): the penalty is **not** the container, it
is the **absence of MPS**. MPS is ~17x faster than CPU on the same machine.

### M0 verdict: local CPU inference is dead

Against the gate in the Task 3 brief:

| Container-CPU mean | Band | Measured |
| --- | --- | --- |
| ≤ 1.0 s | In-process everywhere, no GPU | — |
| 1–3 s | Borderline; async policy-server split | — |
| **> 3 s** | **Local inference dead; remote GPU policy server needed** | **9.318 s** |

We land in the **third band, by a factor of 3.1x** — and async chunk-ahead does
not rescue it. One chunk buys ~1.7 s of robot motion but costs 9.3 s to compute,
so the executor starves ~5.5x faster than the producer can refill it. No amount
of overlapping hides a deficit that large; the demo would stutter, not stream.

**Consequences — these are decisions, not suggestions, and Part 2 must not be
built until they are confirmed with the user:**

1. **The deployed demo needs a remote GPU policy server.** CPU-only Cloud Run
   cannot serve this policy at interactive rates. This is materially more
   expensive per visitor than the plan assumed and the plan's costing should be
   revisited.
2. **A laptop demo is still viable — but only natively (uv), never in Docker.**
   MPS at 0.549 s (sync-corrected) clears the ceiling with ~1.8x headroom. This
   mirrors `apps/manip-trial`, which chose uv over Docker for exactly this
   reason. So local dev and the deployed demo now have *genuinely different*
   inference paths, and the code must abstract over that seam from the start.
3. **The 1 s ceiling itself deserves a second look.** It came from the chunking
   argument, and MPS meets it — but 0.549 s per chunk still means a visible
   half-second pause at every chunk boundary unless chunk N+1 is computed while
   chunk N executes. Async is worth building even on the passing path.

**Open question for the user (blocks Part 2):** accept the cost of GPU instances
in the deployed demo, or change the deal — e.g. serve pre-recorded rollouts to
web visitors and reserve live inference for local runs. That is a product call,
not an engineering one.

## Stack rationale

| Layer | Choice | Why (and what was rejected) |
| --- | --- | --- |
| Policy | SmolVLA 450M (`lerobot/smolvla_base`) | The only VLA family that plausibly runs without a GPU — 450M parameters vs. the multi-billion-parameter OpenVLA/pi0-class models. Its pretraining corpus is SO-100 data exclusively, so an SO-101 fine-tune is in-embodiment (same robot family, not a cross-embodiment transfer). No published Apple-Silicon benchmark existed before this app's M0 spike — HuggingFace's "runs on a MacBook" claim was marketing, not measurement, until Task 3 measured it directly (0.549 s/pass on MPS). |
| Robot | SO-101 (`mujoco_menagerie`'s `robotstudio_so101`) | Cheap ($~100s), open-hardware, 6-DOF (5 arm joints + 1 gripper) arm with an actively maintained MuJoCo MJCF and a real teleop/community dataset ecosystem on the Hub. Chosen specifically because SmolVLA's pretraining corpus is SO-100/SO-101 data — picking a different arm would have made every fine-tune an out-of-embodiment transfer, a much harder and less demo-honest claim. |
| Simulator | MuJoCo 3.10+, `MUJOCO_GL=cgl` offscreen rendering | menagerie's SO-101 asset requires mujoco>=3.10 (the floor is enforced in `pyproject.toml` with an explicit comment: LeRobot's other bundled sim envs — gym-hil, gym-aloha, gym-xarm — pin OLDER mujoco floors and would silently downgrade the install if pulled in as a dependency). `cgl` is the macOS-native offscreen GL backend; `osmesa`/`egl` are Linux-only and do not work on Apple Silicon. M0 spike 2 measured 84-85 fps at 256x256 (2 cameras), clearing the 60 fps floor with ~40% margin — rendering was never the bottleneck. |
| Visualization | Rerun 0.34.1 (exact-pinned) | Matches robium's `rerun` skill pin and the `gradio_rerun` component Part 2 (the deferred web demo) depends on — pinning early avoids a version-drift surprise later. File-based recording (`rr.save()` right after `rr.init(spawn=False)`) needs no GUI/display, which matters for a headless macOS dev loop and for CI. |
| Training | Hugging Face Jobs (remote GPU, `a10g-small`) | M0's decisive finding: MPS fine-tuning is ~2h/20 steps (untimed directly but implied by the ~0.55s/forward-pass inference number plus backward-pass overhead — confirmed empirically painful enough that no local training run was ever attempted), and CPU is 17x slower again. There is no local GPU on this Apple Silicon host. HF Jobs was chosen over standing up a cloud VM by hand because `lerobot-train --job.target=<flavor>` is a single CLI flag on top of the exact same command used for local smoke runs — zero additional tooling, and the Hub is already the dataset/checkpoint hand-off point. |
| Env tooling | uv, Python 3.12 (pinned, `>=3.12` hard-required by LeRobot 0.6.0) | Same reasoning as `manip-trial`: a pure-Python ML stack with no ROS/system deps belongs in uv, not Docker — and here it is a documented *requirement*, not just a preference, because Docker on macOS cannot see MPS at all (`No accelerated backend detected. Using default cpu` in the spike container logs). Containerizing costs ~nothing over native CPU (+3%); the entire 17x penalty is losing MPS. So local dev must run natively; only the *deployed* (Cloud Run) path is container-based, and it is CPU-bound there by definition (no Cloud Run GPU in this deal), which is exactly the M0 gate that pushed real inference to HF Jobs / a remote GPU policy server for Part 2. |

## Pipe-test results (2026-07-15)

The full 20k-step fine-tune was deferred (user cost decision — no further
spend after the pipe validated). What *did* run, for real, on billed GPU
hardware:

| Run | Steps | Target | Result |
| --- | --- | --- | --- |
| First remote submit | 2,000 | a10g-small | Trained to completion, **crashed at checkpoint save** (`REMOTE_OUTPUT_DIR` bug — see Battle scars #7). ~$2-4 spent, no artifact. |
| Second remote submit | 100 | a10g-small | **Full loop green.** Checkpoint pushed to `jazarium/train_2026-07-15_08-09-36` (not `POLICY_REPO_ID` — see Battle scars #8). Eval: 10 seeded episodes, **0/10 = 0% success** on MPS locally. |

0% is the *correct* result for a 100-step checkpoint (effectively the base
model plus 100 gradient steps) — the deliverable of this run was proving the
whole `record -> push -> HF-Jobs-submit -> train -> save -> push -> pull ->
eval` loop end-to-end with a genuine GPU-trained artifact, not a score. The
reference expectation (`SUCCESS_RATE_FLOOR = 0.60`) applies to the deferred
20k-step run, not to either pipe-test checkpoint; `tests/test_smoke.py` and
`tests/test_narrative.py` both assert pipeline *mechanics* against these
checkpoints, never the score bar (see each file's docstring/`TODO`).

## Battle scars

Gotchas hit during the build, each with its fix. Full detail (verbatim
errors, dead ends ruled out) is in `learnings/2026-07-14-vla-trial.md` and
`learnings/2026-07-15-vla-trial.md`; this section is the index.

1. **The end-effector site is not the grasp point.** `gripperframe` (the
   MJCF site an obvious IK target) sits near the fingertips; the arm
   actually holds a 4x4x6cm cube in the jaw *throat*, ~7cm away. Found by
   brute-force calibration (teleport the object to a grid of candidate
   gripper-local offsets, close the gripper, lift, keep the offsets that
   hold). Fixed via `ORACLE_GRASP_LOCAL` in `config.py`, empirically
   calibrated, not assumed.
2. **Position-only IK leaves the wrist roll free, and a free roll can make
   grasping geometrically impossible.** On this 5-DOF arm the roll settled
   near vertical, making the gripper's pinch axis 91% vertical — trying to
   span the cube's 6cm *height* with a 4.2cm aperture instead of its 4cm
   *width*. Diagnostic: print the pinch axis in world coordinates and check
   it's perpendicular to the dimension you intend to grasp; position-only
   IK "converging" (residual 1e-5) tells you nothing about orientation.
   Fixed by solving roll as a 1-D root-find (folding it into the DLS
   objective diverges), then re-solving position with roll pinned. Took the
   oracle 0/10 -> 8/10 in one change.
3. **The floor sits at the arm's own base level; a real SO-101 rig doesn't.**
   menagerie's stock scene forced the fingers to a ~32deg down-pitch to
   reach the workspace at all, which drove the fixed jaw's shaft through
   the cube. Fixed with a 0.06m pedestal under the cube (`scene_pick.xml`)
   restoring the raised-work-surface geometry the arm was designed for.
   Pedestal height and grasp offset are a **matched, coupled pair** — both
   were swept end-to-end together (0.06 -> 10/10, 0.09 -> 3/10, 0.12+ ->
   0/10; higher clearance is not monotonically better). Change one without
   re-sweeping the other and the dataset degrades quietly.
4. **Unnamed MJCF collision geoms are invisible to name-based contact
   checks.** The geom that actually blocks the grasp
   (`wrist_roll_follower_so101_gripper_part0_v1`) has no `name` attribute in
   the vendored MJCF, so the original `GRIPPER_GEOMS` name-list config could
   never see it — a latent false-success bug (a still-held cube could read
   as "released"). Fixed by resolving gripper geoms via `model.geom_bodyid`
   membership (`GRIPPER_BODIES` in `config.py`) instead of a name list —
   robust to unnamed meshes by construction.
5. **`qfrc_actuator` saturated against `forcerange` means BLOCKED, not
   slowly converging.** "Raise settle steps" (the brief's own debug
   suggestion) was swept 12 -> 60 with zero effect, because the arm was
   stalled against a solid object with two joints at their force limits,
   not still settling. The decisive diagnostic that separates the two
   hypotheses: teleport the obstructing object out of the scene and re-run
   the identical control — `arm_err` collapsing from 0.18 to 0.0006 with
   the object removed proves the object is the obstruction, instantly
   ruling out IK error, gravity, and torque limits.
6. **Rerun 0.34.1 renamed the timeline setter, not just the archetypes.**
   `rr.Scalar` -> `rr.Scalars` is documented; `rr.set_time_sequence` being
   replaced by a unified `rr.set_time(timeline, sequence=...)` is not
   covered by the obvious migration checklist and throws
   `AttributeError: module 'rerun' has no attribute 'set_time_sequence'`.
7. **A remote `--output_dir` must be a container path.** `--output_dir` is
   passed verbatim to the HF Jobs container. The first real submit
   (2,000 steps) trained to completion and only then crashed at checkpoint
   save with `PermissionError: '/Users'` — a full paid run, zero artifact,
   because the config held this Mac's local `TRAIN_OUTPUT_DIR`. Fixed with
   a dedicated `REMOTE_OUTPUT_DIR = "/tmp/vla_train/..."` constant plus a
   regression test asserting no remote command string contains `~` or
   `/Users/`.
8. **HF Jobs ignores `--policy.repo_id` for the final push.** The trained
   model landed at an auto-generated `jazarium/train_2026-07-15_08-09-36`
   repo, not the `jazarium/smolvla_so101_pick` that `--policy.repo_id`
   requested. The only way to find the real checkpoint is to read the
   "Model pushed to `<url>`" line in the job log — never assume
   `POLICY_REPO_ID` is where a remote run actually landed.
9. **HF Jobs requires prepaid credits, separate from being authenticated.**
   A correctly-formed, fully-validated `--job.target` submission fails with
   `402 Payment Required` if the account has none. Useful as a free,
   zero-risk way to validate an entire submission (auth, dataset-on-Hub,
   command shape) before adding money — a rejected job never bills.
10. **A test asserting an exact rendered config value can go stale
    silently.** `test_pipe_test_and_full_differ_in_steps` hardcoded
    `"--steps=2000"`; when `PIPE_TEST_STEPS` dropped to 100 under a later
    cost-minimization directive, the command builder updated correctly but
    the test's literal didn't, and nothing caught it until the next full
    `make test` run. Fixed by asserting against the config constant, not a
    re-typed literal — see `learnings/2026-07-15-vla-trial.md`.
11. **The `.gitignore` negation pattern around `scene_pick.xml` is a
    maintainer trap.** The vendored menagerie dump is bulk-ignored
    (`src/vla_trial/env/assets/*`) and re-fetched by `make assets`, but
    `scene_pick.xml` — this app's *own* file, tracked in git, not vendored
    — lives inside that same ignored directory and is pulled back in with
    an explicit negation line (`!src/vla_trial/env/assets/scene_pick.xml`)
    because MuJoCo resolves `<include>`/mesh paths relative to the
    including file's own directory, so it cannot live anywhere else. A
    maintainer skimming the directory and seeing "vendored, gitignored,
    regenerated by a script" could delete `scene_pick.xml` as assumed
    cruft — it is the file that adds the "scene" overview camera
    (menagerie ships only a wrist camera) and is NOT regenerated by
    `fetch_assets.sh`. If it's ever missing, `make assets` will not bring
    it back; restore it from git history.
