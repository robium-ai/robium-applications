# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The proving ground for the **robium** Claude Code plugin. Apps here are built *using* robium's skills, and the friction encountered while building them is the primary product — it feeds robium's skill-hardening loop. See README.md for trial pass bars and layout.

## Capture learnings as you work (mandatory)

Whenever you hit friction involving a robium skill, **append a bullet to `learnings/YYYY-MM-DD.md` immediately** (create the file if it's the first note today; use today's real date). Friction means any of:

- A robium skill gave a command/config/fact that turned out wrong, stale, or incomplete.
- A skill misrouted you, or the skill you needed never fired ("no skill fired" counts — name the question you asked).
- You needed knowledge no skill carried and had to research it yourself.
- A skill worked exactly as documented on something non-trivial (mark ✓ — positive signal is data too).

Format, one bullet per event, tagged with the skill name:

```markdown
# 2026-07-12 — nav-trial

- [nav2] bringup snippet's lifecycle node list missing smoother_server → robot froze at first goal
- [no skill fired] "how do I record a rosbag" triggered nothing; candidate: ros2 description
- [gazebo] bridge YAML pattern worked as documented ✓
```

Write the bullet at the moment it happens, not in an end-of-session summary — details (exact command, exact error) are the valuable part and they evaporate. Do NOT fix robium's skills from this repo; capture here, absorb later in the robium repo with its `skill-author` skill (absorbed files move to `learnings/absorbed/`).

## Building apps here

- One app per `apps/<name>/` directory: own env, own tests, own `docs/architecture-brief.md` (written by the `robium-architect` agent at kickoff; refined afterward with the `architect` skill in the main conversation).
- Let robium's skills drive stack decisions — don't steer around a skill gap or quietly substitute your own knowledge without logging the learning first.
- Test-driven: an app is not done until its smoke test passes (robium `testing` skill's bar).
- Environment-first: uv or Docker per the robium `environments` skill; local and remote runs must reproduce identically.
