# Architecture Brief — nav-trial

**Date:** 2026-07-10   **Status:** implemented (2026-07-11)
**Author:** robium-architect subagent

## 1. Requirements

All confirmed with the user (design spec: `docs/superpowers/specs/2026-07-10-nav-trial-design.md`);
no silent assumptions were needed.

- **Robot type:** mobile base, TurtleBot-class off-the-shelf with existing Gazebo model and
  Nav2 params. Final pick delegated to this brief → **TurtleBot 3 Burger** (see §2).
- **Task:** autonomous navigation in simulation, SLAM-first pipeline: drive → build map →
  save map → Nav2 localizes on the saved map → programmatically-sent goals reached.
- **Sim vs real:** simulation only. No real robot, no sim-to-real.
- **Hardware/host:** Apple Silicon Mac (macOS/Darwin), no NVIDIA GPU → Isaac path gated off
  by both the GPU floor and the no-macOS rule. Everything runs in Docker (arm64 images).
- **Local vs remote:** local Docker, headless (no X display) → browser-based Foxglove viz.
- **Testing bar:** one-command headless smoke test (launch stack, load saved map, send nav
  goal(s), assert arrival within timeout) + lighter SLAM check (map file produced, non-trivial).
- **Layout:** self-contained app at `apps/nav-trial/` inside the robium-applications repo.

## 2. Chosen stack + reasoning

This is the architect skill's **navigation golden path** (`ros2` + `nav2` + `gazebo` +
`visualization`), with the headless/macOS constraints steering the leaf choices.

| Layer | Choice | Version | Why (and what was rejected) |
|---|---|---|---|
| Middleware | ROS 2 | **Jazzy Jalisco** (LTS, EOL May 2029) | Decision-tree 1: mobile base + standard drivers + package ecosystem → ROS 2. Distro: Lyrical Luth is the newer LTS, but **Nav2 has no Lyrical binaries yet** (verified open: [ros-navigation/navigation2#6123](https://github.com/ros-navigation/navigation2/issues/6123), and no Lyrical rows on the ROS package index as of 2026-07-10) — the skill's stated exception applies, so the Nav2 vertical stays on Jazzy. Kilted rejected (non-LTS, EOL ~Dec 2026). |
| Simulator | Gazebo | **Harmonic** (LTS, paired with Jazzy) | Decision-tree 2: no NVIDIA GPU + macOS host → Gazebo is the only viable branch; Isaac Sim is hard-gated (RTX floor, Linux-only). Harmonic is the officially paired sim for Jazzy via `ros_gz`; **arm64 debs confirmed** on packages.osrfoundation.org for Ubuntu Noble. Jetty rejected (pairs with Lyrical, which Nav2 blocks). Run mode: server-only, headless rendering (see §5). |
| Robot | **TurtleBot 3 Burger** | `turtlebot3_gazebo` 2.3.7 (Jazzy binary, released 2025-06-27) | Off-the-shelf model + worlds + per-model Nav2 params (`turtlebot3_navigation2`), all as Jazzy apt binaries; `turtlebot3_world` is purpose-built for exactly this SLAM→nav demo; it is also the robot the Nav2 docs use. **TurtleBot 4 rejected**: also Jazzy/Harmonic-supported (`ros-jazzy-turtlebot4-simulator`), but the Create 3 base pulls in a much heavier node/sim stack — needless load when every frame is software-rendered (no GPU). **Burger over Waffle**: lidar-only; SLAM + Nav2 need only lidar + odom, and skipping the camera removes an extra render-based sensor under llvmpipe. Switching model later is one env var (`TURTLEBOT3_MODEL`). |
| SLAM | **slam_toolbox** | Jazzy binary (`ros-jazzy-slam-toolbox`, 2.8.x) | The Nav2-recommended and only officially supported ROS 2 SLAM library; online-async mode + `map_saver_cli` covers the drive→map→save milestone. Cartographer (the old TB3 default) rejected: effectively unmaintained upstream. |
| Navigation | **Nav2** (AMCL + planner/controller servers, `nav2_simple_commander` for goals) | Jazzy binaries | The classical nav stack for a mobile base; no learning component → no training framework (decision-tree 3, "No" branch). Goals sent programmatically via the `nav2_simple_commander` Python API (`NavigateToPose`), which is also what the smoke test drives. |
| Visualization | **Foxglove** (browser) via `foxglove_bridge` | `ros-jazzy-foxglove-bridge` (3.x, MIT) | Headless Docker + macOS host → no X display → RViz2 rejected per the skill's headless rule. Bridge runs in-container (WebSocket :8765); the Foxglove web app connects from the Mac's browser. Fallback if Foxglove's account/free-tier terms bite: Lichtblick (open-source fork), same WebSocket protocol. |
| Environment | **Docker** (arm64, compose) | `ros:jazzy` base (multi-arch, arm64 confirmed) | Mandated twice: macOS cannot run ROS 2 + Gazebo natively (platform gotcha), and the requirement says Docker. uv/venv rejected — not viable for a full ROS 2 + Gazebo stack on macOS. Native `linux/arm64` images throughout; no amd64 emulation. |

## 3. Module breakdown

Scaffold-pattern ROS 2 layout, pruned hard for an MVP: TurtleBot 3 upstream packages already
provide the description, sim models, worlds, and baseline params, so the app carries **one
colcon package** plus docker and tests.

```
apps/nav-trial/
├── docs/architecture-brief.md        # this file
├── docker/
│   ├── Dockerfile                    # ros:jazzy + turtlebot3*, nav2, slam_toolbox, foxglove_bridge, gz-harmonic (via ros_gz)
│   ├── compose.yaml                  # profiles: sim | slam | nav | test (one container per scenario)
│   └── entrypoint.sh                 # sources ROS + workspace, then exec
├── scripts/
│   ├── run_slam.sh                   # slam scenario wrapper: launch in bg + outer `timeout` around the driver
│   └── run_smoke.sh                  # smoke wrapper: same shape, outer `timeout -k 10 ${SMOKE_TIMEOUT:-180}`
├── src/
│   └── nav_trial_bringup/            # the one app package
│       ├── launch/
│       │   ├── sim.launch.py         # ros_gz_sim gz_sim.launch.py (headless args) + upstream spawn/rsp includes + foxglove_bridge
│       │   ├── slam.launch.py        # sim + slam_toolbox (online_async) + Nav2 servers launched DIRECTLY
│       │   └── nav.launch.py         # sim + map_server(saved map) + AMCL + Nav2 servers launched DIRECTLY
│       ├── config/                   # nav2_params.yaml (TB3 burger.yaml base + grafted smoother_server), slam_params.yaml
│       ├── maps/                     # saved map (map.pgm + map.yaml) — committed
│       └── nav_trial_bringup/        # package module (ros2 run entry points)
│           ├── drive_mapping_route.py  # scripted waypoint route for the SLAM run
│           └── send_goals.py           # nav2_simple_commander goal sender (shared with smoke test)
├── tests/
│   ├── smoke_nav.py                  # the pass bar (see §6/§7)
│   ├── check_map.py                  # SLAM check: map exists + non-trivial
│   └── check_scan.py                 # R1 de-risk probe: /scan publishing
├── Makefile                          # build | sim | slam | nav | smoke | check-map | down
└── README.md
```

**Launch composition as built (changed from draft).** Upstream
`turtlebot3_world.launch.py` is NOT headless-usable as an include — it
hardcodes a gz GUI client and non-overridable server `gz_args` — so
`sim.launch.py` composes `ros_gz_sim/gz_sim.launch.py` directly with headless
args (`-r -s --headless-rendering`) plus the upstream `spawn_turtlebot3` /
`robot_state_publisher` sub-launch includes, which are cleanly reusable.
`slam.launch.py`/`nav.launch.py` launch the Nav2 servers DIRECTLY
(`ParameterFile(allow_substs=True)`, own lifecycle manager with
`bond_timeout: 0.0`, standalone nodes + respawn) instead of including
nav2_bringup — three live-hit reasons: Jazzy `slam:=True` starts a duplicate
(sync) slam_toolbox; `navigation_launch.py` hard-codes lifecycle-manager
params so `bond_timeout` can't be set (Docker stall spikes then kill the
stack); and TB3's `$(find-pkg-share)` param substitutions break without
`allow_substs=True`. Scenario scripts `run_slam.sh`/`run_smoke.sh` wrap each
run with an outer `timeout` so no failure mode is an unbounded hang.
Params: `config/nav2_params.yaml` base is the shipped TB3
`turtlebot3_navigation2` burger.yaml with `smoother_server` grafted in and
`enable_stamped_cmd_vel: true` in all five cmd_vel-publishing sections.

Responsibilities:
- **Upstream (apt):** robot model/URDF (`turtlebot3_description`), sim worlds/models
  (`turtlebot3_gazebo`), baseline Nav2/SLAM params (`turtlebot3_navigation2`).
- **`nav_trial_bringup`:** composition only — launch files, tuned param copies, the saved
  map, and the two thin scripts. No custom nodes unless a milestone forces one.
- **`docker/`:** one image, several compose profiles; the `environments` skill owns detail.
- **`tests/`:** the definition of done; the `testing` skill owns detail.

## 4. Comms plan

Single robot, single host — short plan. Key ROS 2 interfaces (all bridged from Gazebo via
`ros_gz_bridge` where sim-sourced):

| Interface | Type | Rate | Notes |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | ~5 Hz (TB3 LDS, 360 samples) | From gz `gpu_lidar` → needs a render engine even headless (see §8 R1) |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | controller rate | **Confirmed at runtime** (bridge yaml bridges TwistStamped; manual odom drive test + full Nav2 run): TB3 Jazzy sim takes TwistStamped only; Nav2 aligned via `enable_stamped_cmd_vel: true` — §8 R3 closed |
| `/odom`, `/tf`, `/tf_static` | nav_msgs/Odometry, tf2 | ~30 Hz | odom + `map→odom` from slam_toolbox (mapping) or AMCL (nav) |
| `/map` | `nav_msgs/OccupancyGrid` | latched/1 Hz | slam_toolbox during mapping; `map_server` during nav |
| `navigate_to_pose` | Nav2 action | on demand | driven by `send_goals.py` / smoke test via `nav2_simple_commander` |
| Foxglove WebSocket | `foxglove_bridge` | — | container port 8765 published to host; browser connects to `ws://localhost:8765` |

Frames: standard `map → odom → base_link → base_scan`. Transport is plain DDS topics inside
one compose network — no zenoh/gRPC needed at this scale (`integration` skill owns any change).

## 5. Environment strategy

- **Docker only** (macOS host makes native impossible; requirement mandates it anyway).
  One image: `ros:jazzy` (arm64) + apt layers: `ros-jazzy-turtlebot3*`,
  `ros-jazzy-navigation2` + `ros-jazzy-nav2-bringup`, `ros-jazzy-slam-toolbox`,
  `ros-jazzy-foxglove-bridge`, `ros-jazzy-ros-gz` (pulls Gazebo Harmonic from the OSRF repo).
  Colcon-build the one app package on top. Pin the base image digest once bringup works.
- **All images native `linux/arm64`** — no Rosetta/QEMU emulation. Gazebo Harmonic and the
  ROS 2 Jazzy buildfarm both publish arm64/Noble debs; individual package availability is
  confirmed at env-setup time (§8 R4).
- **Headless rendering:** run `gz sim -s` (server only) with headless rendering
  (EGL / mesa-llvmpipe software rendering inside the container — no host GPU is exposed).
  Required because the lidar is a render-based sensor. This is the top de-risk item (§8 R1).
- **No X anywhere.** Visualization is entirely the browser → foxglove_bridge WebSocket.
- **Reproducibility:** compose is the single entry point; `docker compose --profile <x> up`
  behaves identically anywhere Docker runs arm64 (remote Linux/arm64 reproduces; amd64 would
  need a multi-arch build — out of scope per the spec).

## 6. Data plan

Non-learning app — short plan.

- **The saved map is the app's one data artifact**: produced by the SLAM milestone
  (`map_saver_cli`), committed at `src/nav_trial_bringup/maps/` (pgm+yaml, ~KBs), consumed
  by the nav milestone and the smoke test. Regenerating it is a documented one-command run.
- **Map-frame convention:** slam_toolbox sets the `map` origin at the robot's *starting pose*,
  not the Gazebo world origin — spawn at world (-2.0, -0.5) means world (-2.0, -0.5) = map
  (0, 0). The saved map inherits this origin, so all nav goals (send_goals.py, smoke test)
  are map-frame coordinates: `goal_map = goal_world - spawn_pose`.
- **Rosbags/logs** (debug recordings, Foxglove captures): gitignored under `apps/nav-trial/bags/`.
- No datasets, no Hub pulls/pushes — `data`/`huggingface` skills not in the routing table.

## 7. Robium skills per build phase

Ordered by the design spec's milestones; testing is planned in, not bolted on.

| Phase | Skill(s) | Exit criterion |
|---|---|---|
| Env setup | `environments` | image builds arm64; `ros2 topic list` works in container |
| Bringup (M1) | `ros2`, `gazebo` | TB3 spawns in `turtlebot3_world` headless; `/scan` publishes (kills risk R1); drivable |
| Visualization (M1) | `visualization` → `foxglove` | live `/scan` + TF in browser Foxglove |
| SLAM (M2) | `nav2` (slam_toolbox is in its orbit), `ros2` | scripted drive → map saved + committed; `tests/check_map.py` passes |
| Navigation (M3) | `nav2` | AMCL localizes on saved map; `send_goals.py` reaches goals |
| Wiring/compose | `integration` | profiles compose cleanly; cmd_vel stamping aligned (R3) |
| Testing (gate) | `testing` | one-command smoke test green headless |

**Smoke test shape** (pass bar): one command (`docker compose --profile test up
--exit-code-from smoke` or a `make smoke` wrapper) → launches sim + nav stack headless, waits
for Nav2 active, sends goal(s) via `nav2_simple_commander`, asserts `SUCCEEDED` within a
timeout sized to the measured real-time factor (R2), exits nonzero on failure.
**SLAM check** (lighter): map yaml+pgm exist; occupied and free cell counts above thresholds
(rejects an empty/degenerate map).

## 8. Risks (with outcomes at implementation, 2026-07-11)

1. **R1 — Headless GPU-less lidar rendering (top risk).** Gazebo's lidar is a `gpu_lidar`
   render-based sensor; even `gz sim -s` needs a working render engine (OGRE2 via
   EGL/llvmpipe software rendering) inside an arm64 container with no GPU. This is
   known-workable but the least-paved part of the stack. *Blocks:* everything (no `/scan`,
   no SLAM, no nav). *De-risk:* first task of M1 — assert `/scan` publishes; TB3's lidar is
   tiny (360 samples @ 5 Hz) so llvmpipe should cope; fallbacks: reduce samples/rate, pin
   mesa version, or (last resort) a remote Linux box.
   **CLOSED:** llvmpipe/EGL headless lidar rendered on the first try (`/scan` publishing,
   zero OGRE2/EGL errors); none of the fallback rungs were needed.
2. **R2 — Real-time factor under software rendering + Docker-on-macOS VM.** Sim may run
   well below 1.0 RTF. *Blocks:* smoke-test reliability (flaky timeouts). *De-risk:* measure
   RTF at M1; size all test timeouts from sim time, not wall time, where possible.
   **CLOSED:** measured RTF ≈ 0.99 in Docker-on-macOS (essentially real time); smoke
   timeout sized 180 s = ~90 s sim × 2 margin (`SMOKE_TIMEOUT` overridable).
3. **R3 — `/cmd_vel` Twist vs TwistStamped mismatch.** TB3's Jazzy Gazebo integration
   expects `TwistStamped`; Nav2's stamped-output default on Jazzy must be checked and
   aligned (`enable_stamped_cmd_vel` or a relay). *Blocks:* M3 (robot ignores commands).
   *De-risk:* verify at integration; it's a one-param or one-relay fix.
   **CLOSED:** TwistStamped confirmed at runtime (bridge type + odom drive test);
   `enable_stamped_cmd_vel: true` set in all five cmd_vel sections; robot navigates
   end-to-end under Nav2.
4. **R4 — arm64 apt coverage assumed per-package.** Buildfarm arm64/Noble coverage is the
   norm and `turtlebot3_gazebo`/`slam_toolbox`/`foxglove_bridge` Jazzy releases are
   confirmed, but each package's arm64 build wasn't individually verified. *Blocks:* env
   setup. *De-risk:* the Dockerfile build itself is the check; source-build any straggler.
   **CLOSED:** arm64 apt coverage complete — every package installed from stock
   packages.ros.org/ports.ubuntu.com; no OSRF repo or source builds needed.
5. **R5 — Foxglove web-app account/terms.** `foxglove_bridge` is MIT, but the browser app
   requires a Foxglove account (free individual tier). *Blocks:* nothing functional (viz
   only). *De-risk:* Lichtblick (open-source, same WebSocket protocol) as drop-in fallback.
   **MITIGATED:** user verified app.foxglove.dev with a free account against the live
   bridge; Lichtblick remains the untested fallback.
6. **R6 — Jazzy is a dead end for Nav2's next major.** Nav2's Lyrical release lands with a
   large refactor/user migration; this app pins Jazzy (supported to May 2029) and does not
   attempt forward-compat. *Blocks:* nothing now; a future distro bump is a real migration.
   **OPEN (unchanged):** accepted; a future migration when Nav2's Lyrical line stabilizes.
