# robium-applications

Proving ground **and reference library** for the [robium](https://github.com/robium-ai/robium)
Claude Code plugin. Real robotics applications are built here **using robium's skills**,
serving four purposes:

1. **Trial runs** that harden the skills — stumbles are captured, not papered over.
2. A living, test-driven regression suite: every app stays maintained and running.
3. Canonical samples the robium skills reference (verified examples get promoted from
   `status: unverified` after passing here).
4. **Bootstrap material**: when a future application resembles an existing app here,
   start from that app — its structure, env, tests, and encoded battle scars — instead
   of from scratch. Over time this repo grows toward covering the common *combinations*
   of robotics applications (nav × sim × viz × learning stacks), each battle-tested.

**Start at [REGISTRY.md](REGISTRY.md)** — the index of what exists, what stack each app
proves, and what each can bootstrap. An app is not done until its registry card is
added/updated in the same commit.

## How to build an app here

Start a **fresh Claude Code session** with the robium plugin enabled, from the new app's
directory under `apps/`. State the application goal and let the skills do the routing —
the `robium-architect` agent researches the stack and writes `docs/architecture-brief.md`
into the app repo; refinement then happens with the `architect` skill in the main session.

Rules of the game:

- Don't steer around skill gaps. If a skill misroutes, gives stale commands, or is missing
  guidance, let it happen, note it, and move on — the failure is the data.
- Every app is test-driven: it is not done until its smoke test passes (see the robium
  `testing` skill's pyramid).
- Environments are virtual-first and must reproduce local == remote (robium `environments`).

## MVP trial runs

| Trial | Vertical | Target | Pass bar |
| --- | --- | --- | --- |
| `apps/nav-trial` | Classical ROS | Autonomous mobile-robot navigation in simulation (expected: ROS 2 + Nav2 + Gazebo, dockerized, live viz) | Robot navigates to goals in sim; smoke test passes; skills visibly drove the stack decisions |
| `apps/manip-trial` | Physical AI / ML | Manipulation policy small-scale train or fine-tune + eval in sim (expected: LeRobot, uv env) | Training run completes; eval produces metrics; smoke-scale, not SOTA |

MVP is done when **both trials pass and one absorption cycle has run** (learnings folded
back into the robium skills).

## Learnings loop

While building, capture friction in `learnings/YYYY-MM-DD.md` as it happens — one bullet
per stumble, each naming the robium skill involved (or "no skill fired"):

```markdown
# 2026-07-12 — nav-trial

- [nav2] bringup snippet's lifecycle_manager node list missing smoother_server → robot froze at first goal
- [no skill fired] asked "how do I record a rosbag" — nothing triggered; candidate: ros2 skill description
- [gazebo] worked as documented (bridge YAML pattern) ✓
```

Beyond frictions, also capture *better methods found*, *noise/verbosity*, and
*worked-as-documented ✓* entries (✓ is what promotes a skill example from
`status: unverified` to verified), plus a short end-of-block retro line per skill used —
see CLAUDE.md for the full taxonomy.

Periodically, a hardening session in the **robium repo** (with the `skill-author` skill)
absorbs these files: twice-seen learnings become skill edits + changelog lines; knowledge
goes to the lowest skill that can hold it; absorbed entries get an inline
`<!-- absorbed: YYYY-MM-DD -->` marker appended to their line.

## Layout

- `apps/<app-name>/` — one self-contained application per directory (own env, own tests,
  own `docs/architecture-brief.md`).
- `learnings/` — dated friction notes feeding the robium hardening loop.
