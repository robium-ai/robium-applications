# nav-trial

TurtleBot 3 Burger, simulation-only: SLAM builds the map, Nav2 navigates it.
Runs entirely in Docker (arm64) on a macOS host, headless; visualization is
browser Foxglove at ws://localhost:8765.

- Architecture: `docs/architecture-brief.md`
- `make build` — build the image (`docker compose build sim`; the explicit
  service name matters — a bare `compose build` builds nothing when every
  service is behind a profile)
- `make sim` — headless sim bringup (M1); foreground, Ctrl-C to stop
- `make slam` — SLAM run: drives the waypoint route, saves the map (M2)
- `make nav` — navigation on the saved map (M3); waits for goals from
  `send_goals.py` / Foxglove
- `make smoke` — the one-command pass bar (rebuilds the image via `--build`,
  runs the nav scenario + goal client, exits 0 on success)
- `make check-map` — host-side map sanity check (`tests/check_map.py`)
- `make down` — tear down all profiles' containers
- `make demo` — the live-demo scenario (nav stack + auto initial pose +
  Foxglove bridge); `make demo-smoke` gates it. `make demo-image` +
  `make demo-deploy` push it to Cloud Run (`demo-nav-trial`, robium-prod,
  per-visitor instances via concurrency=1, GZ_RELAY unicast discovery)
  where robium.org/demos/nav-trial hands each visitor a private instance.

Map regeneration: `make slam` rewrites `src/nav_trial_bringup/maps/`
(map.pgm + map.yaml) via the compose volume mount; the next image build
(`make build`, or `make smoke`'s `--build`) bakes the new map in for the
nav/smoke scenarios.

Timeouts: the smoke run is bounded by `SMOKE_TIMEOUT` (seconds, default 180
≈ 90 s sim × 2 at RTF ≈ 1.0) — override with e.g. `SMOKE_TIMEOUT=300 make
smoke`. The SLAM run has an analogous `SLAM_TIMEOUT` (default 900) inside
the container.

## Visualization

With any profile running: open https://app.foxglove.dev (or Lichtblick),
"Open connection" → `ws://localhost:8765`. Live topics: /scan, /tf, /map
(during SLAM/nav), /plan, costmaps. Note: nav goals are map-frame — the
SLAM map origin is the robot's start pose, so world (-2.0, -0.5) = map (0, 0).

### Preconfigured layout

Import `foxglove/nav-trial-layout.json` once (Foxglove: **Layout menu → Import from file…**).
It sets display frame `map`, shows /map, /scan, /plan and the global costmap, and points the
Publish tool at Nav2's `/goal_pose` (initial pose → `/initialpose`). After importing it's saved
in your Foxglove account. Use Chrome — Safari blocks ws://localhost from the https app.
