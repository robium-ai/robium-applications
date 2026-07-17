"""The demo's Gradio app: pick a rung on the training ladder -> Run ->
embedded Rerun. Plus a Gallery tab with every rung's REAL eval videos and
metrics (from outputs/demo/ladder.json — generated, never hand-edited).

Mounted at /ui by gateway.py; the website's Robot pane iframes it. Streaming
pattern (fresh RecordingStream + recording_id per Run, yield stream.read())
is vla-trial's, including the merge-on-same-id gotcha its docstring records.

Honesty is part of the layout: pc_success is 0% at every rung and the intro
says so — and the ladder is NOT monotonic (the 5k rung out-evals the older
10k baseline run); the real numbers are shown per rung, noise and all.
"""

import json
import uuid

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun

from manip_trial import config

APP_ID = "manip_trial_demo"


def _manifest() -> dict:
    return json.loads(config.DEMO_LADDER_MANIFEST.read_text())


INTRO_MD = """\
**Pick a checkpoint from the training ladder, hit Run.** The ACT policy pushes
the gray T-block toward the green target zone; every control step streams onto
the Rerun timeline below (the 96×96 frame the policy sees, its actions, and
the coverage reward) — scrub it when the episode ends.

- The 1k/3k/5k rungs are one training run frozen at increasing steps; 10k is
  an earlier baseline run — **watch what more training buys (and what it
  doesn't: the ladder isn't monotonic, and the numbers shown are the real
  ones).**
- PushT counts "success" only at ≥95% target coverage, which ACT at this
  scale never reaches — `pc_success` is 0% at every rung. The max-coverage
  reward is the honest metric.
"""


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="sim", name="what the policy sees"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="reward", name="coverage reward"),
                rrb.TimeSeriesView(origin="action", name="action (target xy)"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def _rung_choices(manifest: dict) -> list[tuple[str, str]]:
    return [
        (
            f"{r['name']} — {r['steps']:,} steps · avg_max_reward "
            f"{r['metrics']['avg_max_reward']:.3f} · {r['run']}",
            r["name"],
        )
        for r in manifest["rungs"]
    ]


def _gallery_md(manifest: dict) -> str:
    rows = [
        "| rung | steps | run | avg_max_reward | avg_sum_reward | pc_success |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in manifest["rungs"]:
        m = r["metrics"]
        rows.append(
            f"| {r['name']} | {r['steps']:,} | {r['run']} "
            f"| {m['avg_max_reward']:.3f} | {m['avg_sum_reward']:.1f} | {m['pc_success']:.0f}% |"
        )
    rows.append("")
    rows.append(
        f"Every row: a real {manifest['rungs'][0]['metrics']['n_episodes']}-episode "
        f"seeded eval (seed {manifest['seed']}) of that exact checkpoint."
    )
    return "\n".join(rows)


def build_ui(get_runner) -> gr.Blocks:
    """`get_runner` -> EpisodeRunner | None (None while the gateway boots)."""
    manifest = _manifest()

    def run_episode(rung: str):
        runner = get_runner()
        if runner is None:
            raise gr.Error("Still booting — checkpoints are loading (see the page's status pill).")

        # Fresh recording id per Run — same-id recordings MERGE in the viewer
        # (vla-trial learned this the hard way; see its ui.py docstring).
        rec = rr.RecordingStream(application_id=APP_ID, recording_id=str(uuid.uuid4()))
        stream = rec.binary_stream()
        rec.send_blueprint(_blueprint())
        yield stream.read(), f"resetting env — {rung} rung episode starting…"

        print(f"[demo] run_episode start: rung={rung}", flush=True)
        try:
            for ev in runner.run(rung, rec):
                if ev.step % 50 == 0 or ev.done:
                    print(f"[demo] step {ev.step} done={ev.done} success={ev.success}", flush=True)
                if ev.done:
                    if ev.aborted:
                        verdict = "⏹ aborted — the instance was reclaimed (page refresh or new visitor)"
                    elif ev.success:
                        verdict = f"✅ ≥95% coverage — solved (max reward {ev.max_reward:.2f})"
                    else:
                        verdict = (
                            f"❌ no success — max coverage reward {ev.max_reward:.2f} "
                            "(success needs ≥95% coverage; expected at this training scale)"
                        )
                    yield stream.read(), f"finished at step {ev.step + 1}: {verdict}"
                else:
                    yield stream.read(), f"step {ev.step + 1}/{ev.total} · max reward {ev.max_reward:.2f}"
        except RuntimeError as e:  # run lock held — another episode is executing
            raise gr.Error(str(e))

    with gr.Blocks(title="manip-trial — robium live demo") as blocks:
        with gr.Tab("Run"):
            gr.Markdown(INTRO_MD)
            rung = gr.Radio(
                choices=_rung_choices(manifest),
                value=config.DEMO_DEFAULT_RUNG,
                label="checkpoint (the training ladder — real eval numbers, noise and all)",
            )
            run_btn = gr.Button("Run episode", variant="primary")
            status = gr.Textbox(value="idle", label="status", interactive=False)
            viewer = Rerun(
                streaming=True,
                height=560,
                panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"},
            )
            run_btn.click(run_episode, inputs=[rung], outputs=[viewer, status], api_name="run_episode")

        with gr.Tab("Gallery — the ladder, evaluated"):
            gr.Markdown(_gallery_md(manifest))
            with gr.Row():
                for r in manifest["rungs"]:
                    video = config.APP_ROOT / r["videos"][0]
                    gr.Video(
                        value=str(video),
                        label=f"{r['name']} ({r['steps']:,} steps) — eval episode 0",
                        interactive=False,
                    )

    return blocks
