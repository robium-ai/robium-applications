"""The demo's Gradio app: instruction -> controller -> Run -> embedded Rerun.

Mounted at /ui by gateway.py (gr.mount_gradio_app); the website's Robot pane
iframes it. Built against gradio_rerun 0.34.1's streaming pattern (verified
2026-07-15 from rerun-io/gradio-rerun-viewer's README): each Run constructs a
fresh RecordingStream, logs into it, and yields `stream.read()` bytes to the
`Rerun(streaming=True)` component.

Honesty is part of the layout: the trained controller is labeled as the
100-step pipe-test checkpoint that does NOT complete the task yet, and the
oracle is labeled scripted-and-blind. No success theater.
"""

import uuid

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun

from vla_trial.config import DEMO_CHECKPOINT, TASK

APP_ID = "vla_trial_demo"

CONTROLLER_CHOICES = [
    ("oracle — scripted pick, completes the task (blind: uses ground-truth cube pose, ignores your text)", "oracle"),
    ("trained — SmolVLA fine-tune-in-progress, acts from pixels + your text (currently flails: 100-step pipe-test checkpoint)", "trained"),
]

INTRO_MD = f"""\
**Type an instruction, pick a controller, hit Run.** The arm acts in MuJoCo;
every step streams onto the Rerun timeline below (cameras, joint state,
actions) — scrub it when the episode ends.

- The policy was trained on episodes of *"{TASK}"* — off-distribution
  instructions produce honest flailing, not magic.
- Trained checkpoint: `{DEMO_CHECKPOINT}` (a pipe-test artifact; the real
  20k-step fine-tune is a deliberate later spend and will slot in here).
"""


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="camera/scene", name="scene camera"),
                rrb.Spatial2DView(origin="camera/wrist", name="wrist camera"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="state", name="joint state"),
                rrb.TimeSeriesView(origin="action", name="action sent"),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def build_ui(get_runner) -> gr.Blocks:
    """`get_runner` -> EpisodeRunner | None (None while the gateway boots)."""

    def run_episode(recording_id: str, controller: str, instruction: str):
        runner = get_runner()
        if runner is None:
            raise gr.Error("Still booting — the model is loading (see the page's status pill).")
        instruction = (instruction or "").strip() or TASK

        rec = rr.RecordingStream(application_id=APP_ID, recording_id=recording_id)
        stream = rec.binary_stream()
        rec.send_blueprint(_blueprint())
        yield stream.read(), f"resetting env — {controller} episode starting…"

        print(f"[demo] run_episode start: controller={controller} instruction={instruction!r}", flush=True)
        try:
            for ev in runner.run(controller, instruction, rec):
                if ev.step % 50 == 0 or ev.done:
                    print(f"[demo] step {ev.step} done={ev.done} success={ev.success}", flush=True)
                if ev.done:
                    if ev.aborted:
                        verdict = "⏹ aborted — the instance was reclaimed (page refresh or new visitor)"
                    elif ev.success:
                        verdict = "✅ cube in the bin"
                    elif controller == "trained":
                        verdict = "❌ no success — expected for the pipe-test checkpoint (see the note above)"
                    else:
                        verdict = "❌ oracle miss — off its tuned seed band; run it again"
                    yield stream.read(), f"finished at step {ev.step + 1}: {verdict}"
                else:
                    yield stream.read(), f"step {ev.step + 1}/{ev.total}"
        except RuntimeError as e:  # run lock held — another episode is executing
            raise gr.Error(str(e))

    with gr.Blocks(title="vla-trial — robium live demo") as blocks:
        gr.Markdown(INTRO_MD)
        with gr.Row():
            instruction = gr.Textbox(
                value=TASK, label="instruction (the policy's language condition, verbatim)", scale=3
            )
            controller = gr.Radio(
                choices=CONTROLLER_CHOICES, value="oracle", label="controller", scale=2
            )
        with gr.Row():
            run_btn = gr.Button("Run episode", variant="primary")
        status = gr.Textbox(value="idle", label="status", interactive=False)
        viewer = Rerun(
            streaming=True,
            height=560,
            panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"},
        )

        # Per-browser-session recording id: reruns from the same visitor merge
        # into one viewer timeline history, a reload starts clean.
        recording_id = gr.State(lambda: str(uuid.uuid4()))

        run_btn.click(
            run_episode,
            inputs=[recording_id, controller, instruction],
            outputs=[viewer, status],
            api_name="run_episode",
        )

    return blocks
