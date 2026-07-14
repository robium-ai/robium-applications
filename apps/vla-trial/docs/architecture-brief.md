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
