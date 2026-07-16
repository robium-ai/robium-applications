# Learnings — 2026-07-15 (manip-trial demo page build)

- [lerobot] `lerobot[viz]==0.6.0` pins `rerun-sdk>=0.24.0,<0.34.0`, which is
  unsatisfiable next to `gradio_rerun==0.34.1` (needs rerun-sdk==0.34.1).
  Exact error: `Because lerobot[viz]==0.6.0 depends on rerun-sdk>=0.24.0,<0.34.0
  … your project's requirements are unsatisfiable.` Fix that passed: drop the
  `viz` extra and pin `rerun-sdk==0.34.1` explicitly (vla-trial's pyproject
  already encodes this — the extra-less shape is why its resolve works).
  Dead-end ruled out: keeping `viz` and downgrading gradio_rerun — there is no
  gradio_rerun release for rerun <0.34 with the streaming API the demo uses.
  Verified by: `uv sync` succeeding after the change (and the demo tests later
  in the session). Candidate for the lerobot skill's platform gotchas
  (seen 1x; vla-trial hit the adjacent shape on 07-14, so effectively 2x).
