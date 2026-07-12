# Application registry

The index of every app in this repo — what exists, what stack it proves, and
what a new build can bootstrap from — so neither humans nor agents have to
re-research the apps in detail. **Rule: an app is not done until its card
here is added/updated** (same commit as the app change). Newest facts win;
each card's `verified` date says when its smoke test last passed.

Read this file first when: starting an app that resembles an existing one
(bootstrap from the closest card's app), looking for a canonical sample of a
stack combination, or checking which combinations are already battle-tested.
Details live in each app's `docs/architecture-brief.md` — the card is the
map, not the territory.

## Quick index

| App | Vertical | Stack | Sim | Env | Viz | Smoke | Verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [nav-trial](#nav-trial) | Classical ROS navigation | ROS 2 Jazzy + Nav2 + slam_toolbox | Gazebo Harmonic (headless) | Docker (arm64) | Foxglove (browser) | `make smoke` | 2026-07-11 |
| [manip-trial](#manip-trial) | Physical AI / ML manipulation | LeRobot 0.6.0 (ACT policy) | gym-pusht (pygame/pymunk) | uv + Python 3.12 (MPS) | rerun / MP4s | `make smoke` | 2026-07-12 |

## Cards

### nav-trial

**One-liner:** TurtleBot 3 Burger navigates autonomously in sim — SLAM builds
the map, Nav2 drives goals on it — fully headless in Docker on a macOS host.

- **Stack:** ROS 2 Jazzy · Nav2 (direct server launch, own lifecycle
  manager, `bond_timeout: 0`) · slam_toolbox (online_async) · AMCL on saved
  map · Gazebo Harmonic via ros_gz (headless, software rendering, RTF≈1.0
  without GPU) · TB3 burger (TwistStamped cmd_vel).
- **Env:** one Docker image (`ros:jazzy-ros-base-noble`, arm64), compose
  profiles = scenarios (sim / slam / nav / test), all nodes of a scenario in
  ONE container (macOS DDS-multicast constraint).
- **Viz:** foxglove_bridge → browser app.foxglove.dev; committed layout at
  `foxglove/nav-trial-layout.json` (import once: map/scan/plan/costmap +
  goal publishing on /goal_pose).
- **Pass bar:** `make smoke` — compose test profile, nav-on-saved-map, two
  map-frame goals SUCCEEDED, exit-code chain through make. ~90 s warm.
- **Bootstrap for:** any ROS 2 + Nav2 + Gazebo mobile-robot app; headless
  Gazebo in Docker (esp. Apple Silicon); compose-profile scenario layouts;
  Foxglove remote viz with a committed layout; SLAM→saved-map→AMCL flows.
- **Battle scars encoded:** TwistStamped alignment, collision_monitor
  source_timeout, SLAM map-origin-at-start, ParameterFile(allow_substs),
  bringup-abort recovery — see `learnings/2026-07-10.md` and the brief §8.

### manip-trial

**One-liner:** small-scale imitation-learning pipeline — ACT trained on
`lerobot/pusht`, evaluated in the PushT sim with metrics — entirely on a
GPU-less Mac via MPS.

- **Stack:** LeRobot 0.6.0 (pinned) · ACT policy (`--policy.type=act`) ·
  `lerobot/pusht` dataset (Hub, v3.0, ~186 MB) · gym-pusht sim (no MuJoCo,
  macOS-clean) · torchcodec + Homebrew ffmpeg for AV1 decode.
- **Env:** uv project, Python 3.12 pinned, `uv.lock` committed; one host dep
  (`brew install ffmpeg`); device `mps` with `cpu` fallback — a documented
  exception to graduate-to-Docker (container would lose MPS).
- **Pass bar:** `make smoke` — pytest wraps 200-step train + 2-episode
  seeded eval, asserts exit codes + checkpoint + numeric metrics
  (`eval_info.json` → `overall`). ~40 s warm. `make train-baseline` (10k
  steps, ~15 min) + `make baseline-eval` for meaningful metrics.
- **Bootstrap for:** any LeRobot train/eval pipeline; uv-based ML robotics
  envs on Apple Silicon; policy-eval-as-smoke-test shape (config.py as
  single source of run params shared by Makefile and pytest).
- **Battle scars encoded:** `--eval.use_async_envs=false` (forkserver
  crash), pre-0.6 Hub checkpoints unloadable (no working pretrained PushT
  baseline exists — train your own), `diffusion` extra needed for diffusion
  checkpoints — see `learnings/2026-07-12.md` and the brief §8.

## Adding a card (checklist for new apps)

1. Copy a card's shape: one-liner, Stack, Env, Pass bar, Bootstrap for,
   Battle scars encoded (link the learnings file + brief).
2. Add the Quick index row (keep columns: vertical, stack, sim, env, viz,
   smoke command, verified date).
3. Set `verified` to the date the smoke test last passed; update it on
   re-verification, not on unrelated edits.
4. Land the card in the same commit that makes the app pass its bar.
