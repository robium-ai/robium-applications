# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The proving ground for the **robium** Claude Code plugin. Apps here are built *using* robium's skills — but the operator is robium's developer, not a client. Every session wears two hats: build the app honestly, and treat every skill interaction as QA data for the plugin. The learnings captured here are the primary product; the apps are the instrument. See README.md for trial pass bars and layout.

## Capture learnings as you work (mandatory)

Append a bullet to `learnings/YYYY-MM-DD.md` **at the moment an event happens** (create the file on first note; use today's real date; append `-<app>` to the filename if two apps run the same day). Details — exact command, exact error, exact phrasing — are the valuable part and they evaporate by end of session. Capture ALL of these signal types, tagged `[skill-name]` or `[none]`:

- **Wrong/stale guidance** — a skill's command/config/fact failed or is outdated.
- **No skill fired** — you asked something a skill should cover and nothing triggered. Record the exact phrasing you used; it becomes an eval case.
- **Figured out from scratch** — trial-and-error, source-reading, or web research that a skill should have spared you. Highest-value entries.
- **Better method found** — the skill's way worked, but you found a superior approach (simpler command, newer API, cleaner pattern). Robium's bar is best-known-method; capture upgrades even when nothing broke.
- **Noise/verbosity** — the answer existed but was buried; prose that should be a table; duplication. Feeds the hardening prune pass.
- **Worked as documented ✓** — a non-trivial snippet/example ran exactly as written. Name the file/section; ✓ entries are the only evidence that promotes `status: unverified` examples to verified.

Good entry: names the skill (or "none"), what was expected, what happened, and — if known — the fix. "nav2 was confusing" is useless; "nav2 Quick start costmap YAML omits the inflation_layer block → robot hugged obstacles" is actionable.

## End-of-block retro (mandatory)

At the end of each work block (milestone or session), add one line per robium skill that loaded during the block, scoring: **fired** (triggered when it should, quiet when it shouldn't), **accurate**, **complete**, **lean**. A clean score still gets a line — "no findings under real load" is evidence too.

## Two hats, one rule

- **During a build**: use the skills as a client would. Do NOT edit robium's skills mid-build and do NOT quietly substitute your own knowledge — log the learning first, then proceed however the build needs.
- **At session end**: when the user invokes the robium `skill-updater` skill (or asks to "update my skills" / "absorb these learnings"), harvest this session's learnings plus unabsorbed files and apply them to the robium source checkout (`~/repos/robium`) per that skill's workflow — this is the sanctioned exception to "don't edit mid-build," gated on explicit user invocation.
- **Between builds**: fuller hardening sessions run in the **robium repo** with its `skill-author` skill (see its learnings-loop reference — it consumes these files, edits skills, promotes ✓-verified examples, prunes noise, then marks entries here with `<!-- absorbed: YYYY-MM-DD -->`).

## Building apps here

- One app per `apps/<name>/` directory: own env, own tests, own `docs/architecture-brief.md` (written by the `robium-architect` agent at kickoff; refined afterward with the `architect` skill in the main conversation).
- Test-driven: an app is not done until its smoke test passes (robium `testing` skill's bar).
- Environment-first: uv or Docker per the robium `environments` skill; local and remote runs must reproduce identically.
