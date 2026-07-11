# nav-trial

TurtleBot 3 Burger, simulation-only: SLAM builds the map, Nav2 navigates it.
Runs entirely in Docker (arm64) on a macOS host, headless; visualization is
browser Foxglove at ws://localhost:8765.

- Architecture: `docs/architecture-brief.md`
- `make build` — build the image
- `make sim` — headless sim bringup (M1)
- `make slam` — SLAM run: drives a route, saves the map (M2)
- `make nav` — navigation on the saved map (M3)
- `make smoke` — the one-command pass bar
