# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The proving ground **and reference library** for the **robium** Claude Code plugin. Apps here are built *using* robium's skills — but the operator is robium's developer, not a client. Every session wears two hats: build the app honestly, and treat every skill interaction as QA data for the plugin. The learnings captured here are a primary product; the apps themselves are the second product — canonical, battle-tested samples that future applications reference or **bootstrap from**. Over time the repo grows toward covering the common combinations of robotics stacks. See README.md for trial pass bars and layout.

## Sibling repos — anchor the session in the repo that owns the output

Three repos are worked on together and sit side by side under `~/repos/`:
**robium** (the plugin: skills, agents, its own STRICT skill-update policy),
**robium-applications** (this one: apps + learnings), and **robium.org** (landing
site + live-demo orchestrator). `.claude/settings.json` here puts the other two on
`additionalDirectories`, so they are readable and writable from this session — but
**launch Claude in the repo whose output you are producing**, because the launch
directory is what selects the operating mode (this file's two-hats rule only loads
when this repo is the anchor, and git branch/status in the prompt track it too).

- Building or QA'ing an app → anchor here.
- Hardening/absorbing skills → anchor in `robium` (that is where `skill-author` and
  the archive/version rules live).
- Site or demo infrastructure → anchor in `robium.org`.

Writes to `robium/skills/**` from this repo are gated by an `ask` permission rule —
a deliberate speed bump for the two-hats rule below. It is not a hard block: an
explicit, user-invoked `skill-updater` run still works, it just has to be confirmed.

## Registry (mandatory)

`REGISTRY.md` at the repo root is the index of every app — stack, pass bar, what it can bootstrap, battle scars. Two rules:

- **Read it first** when starting any new app: if an existing app resembles the target, bootstrap from it (copy its structure/env/test shape, then diverge) instead of scaffolding from scratch.
- **Keep it current**: an app is not done until its registry card is added/updated (quick-index row + card, `verified` date = last smoke pass), in the same commit as the app change.

## Capture learnings as you work (mandatory)

Append a bullet to `learnings/YYYY-MM-DD.md` **at the moment an event happens** (create the file on first note; use today's real date; append `-<app>` to the filename if two apps run the same day). Details — exact command, exact error, exact phrasing — are the valuable part and they evaporate by end of session. Capture ALL of these signal types, tagged `[skill-name]` or `[none]`:

- **Wrong/stale guidance** — a skill's command/config/fact failed or is outdated.
- **No skill fired** — you asked something a skill should cover and nothing triggered. Record the exact phrasing you used; it becomes an eval case.
- **Figured out from scratch** — trial-and-error, source-reading, or web research that a skill should have spared you. Highest-value entries.
- **Better method found** — the skill's way worked, but you found a superior approach (simpler command, newer API, cleaner pattern). Robium's bar is best-known-method; capture upgrades even when nothing broke.
- **Noise/verbosity** — the answer existed but was buried; prose that should be a table; duplication. Feeds the hardening prune pass.
- **Worked as documented ✓** — a non-trivial snippet/example ran exactly as written. Name the file/section; ✓ entries are the only evidence that promotes `status: unverified` examples to verified.
- **User-corrected approach** — the user overrode or corrected a skill-guided approach mid-session. Record the exact correction and what the skill had suggested; a correction is the strongest single-observation signal that guidance and reality disagree.

Good entry: names the skill (or "none"), what was expected, what happened, and — if known — the fix. "nav2 was confusing" is useless; "nav2 Quick start costmap YAML omits the inflation_layer block → robot hugged obstacles" is actionable.

**Evidence bar (write entries that can be absorbed):** where they exist, capture (1) the passing check that verified the fix, (2) the exact error/symptom verbatim, and (3) the dead-ends ruled out and why — absorption holds new knowledge to this three-part bar; an entry missing a part waits in learnings/ as tentative until the evidence shows up. Append a `(seen 2x)` count when the same friction re-hits — recurrence is the strongest promotion signal. Project-local facts (this app's port, this repo's path) go to the app's README/brief, not learnings/.

## End-of-block retro (mandatory)

At the end of each work block (milestone or session), add one line per robium skill that loaded during the block, scoring: **fired** (triggered when it should, quiet when it shouldn't), **accurate**, **complete**, **lean**. A clean score still gets a line — "no findings under real load" is evidence too.

## Two hats, one rule

- **During a build**: use the skills as a client would. Do NOT edit robium's skills mid-build and do NOT quietly substitute your own knowledge — log the learning first, then proceed however the build needs.
- **Capture is automatic; absorption is never automatic.** NEVER edit robium's skills or invoke `skill-updater` on your own initiative. Instead, when a work block ends (or notable learnings have piled up), OFFER: present the skill-worthy candidates as a short list — `[target-skill] finding → smallest intended edit` — and ask "want me to run skill-updater with these?" The user is the editorial gate against skill bloat; items they don't approve stay in `learnings/` as notes, which is a fine permanent home.
- **On explicit invocation only**: `skill-updater` (or "update my skills" / "absorb these learnings") applies the user-approved items to the robium source checkout (`~/repos/robium`) per that skill's workflow. Even then there is a second gate: before ANY commit, the per-skill change summary (skill, old → new version, concrete diff-level changes) is presented for explicit approval. Every skill edit bumps the skill's `version:` and snapshots the prior version to robium's `archive/<name>/<old-version>/` in the same commit. These rules hold in autonomous mode too — autonomy never extends to skill edits.
- **Between builds**: fuller hardening sessions run in the **robium repo** with its `skill-author` skill (see its learnings-loop reference — it consumes these files, edits skills, promotes ✓-verified examples, prunes noise, then marks entries here with `<!-- absorbed: YYYY-MM-DD -->`).

## Building apps here

- One app per `apps/<name>/` directory: own env, own tests, own `docs/architecture-brief.md` (written by the `robium-architect` agent at kickoff; refined afterward with the `architect` skill in the main conversation).
- Test-driven: an app is not done until its smoke test passes (robium `testing` skill's bar).
- Environment-first: uv or Docker per the robium `environments` skill; local and remote runs must reproduce identically.
