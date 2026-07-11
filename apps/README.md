# apps/

One self-contained application per directory. Each app carries its own environment
(uv project or Docker, per the robium `environments` skill), its own tests, and its own
`docs/architecture-brief.md` written by the `robium-architect` agent at kickoff.

Planned MVP trials: `nav-trial/` (ROS 2 navigation in sim) and `manip-trial/`
(LeRobot manipulation policy, train + eval). See the root README for pass bars.
