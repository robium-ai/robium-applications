"""Ladder eval + manifest: eval every rung, write outputs/demo/ladder.json.

The manifest is the single source both the demo's checkpoint radio and its
gallery tab read — the eval target GENERATES it (never hand-edit), so every
number the page shows traces to a real eval_info.json.
"""

import json
import shutil
import subprocess
from pathlib import Path

from manip_trial import config

_METRIC_KEYS = ("avg_max_reward", "avg_sum_reward", "pc_success", "n_episodes")


def build_manifest(rungs: list[dict], eval_root: Path, app_root: Path) -> dict:
    entries = []
    for r in rungs:
        eval_dir = eval_root / r["name"]
        overall = json.loads((eval_dir / "eval_info.json").read_text())["overall"]
        videos = sorted(
            str(p.relative_to(app_root)) for p in eval_dir.glob("videos/**/*.mp4")
        )
        entries.append(
            {
                "name": r["name"],
                "steps": r["steps"],
                "run": r["run"],
                "checkpoint": str(r["checkpoint"].relative_to(app_root)),
                "metrics": {k: overall[k] for k in _METRIC_KEYS},
                "videos": videos,
            }
        )
    return {"seed": config.SEED, "rungs": entries}


def eval_ladder() -> int:
    rungs = config.ladder_rungs()
    for r in rungs:
        out = config.LADDER_EVAL_OUTPUT_DIR / r["name"]
        shutil.rmtree(out, ignore_errors=True)
        cmd = config.eval_cmd(
            str(r["checkpoint"]), config.LADDER_EVAL_EPISODES, config.LADDER_EVAL_BATCH_SIZE, out
        )
        print(f"$ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc
    manifest = build_manifest(rungs, config.LADDER_EVAL_OUTPUT_DIR, config.APP_ROOT)
    config.DEMO_LADDER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    config.DEMO_LADDER_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {config.DEMO_LADDER_MANIFEST}")
    return 0
