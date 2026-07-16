"""Demo Gradio app — full implementation lands with the UI task."""

import gradio as gr


def build_ui(get_runner) -> gr.Blocks:
    with gr.Blocks(title="manip-trial — robium live demo") as blocks:
        gr.Markdown("UI under construction.")
    return blocks
