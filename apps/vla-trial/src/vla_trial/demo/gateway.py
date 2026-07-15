"""Demo session gateway — one process, one port (8765), per the demo spec.

FastAPI implementing nav-trial's session contract (so robium-website's
Controls/demoClient/orchestrator reuse unchanged) + the Gradio app mounted at
/ui. Unlike nav-trial's gateway there is no WebSocket tunnel: the "viewer" is
the Gradio app itself, and "busy" means an episode is executing.

Contract (mirrors apps/nav-trial/scripts/demo_gateway.py):
  POST /start?session=U    -> claim; foreign session while a run executes -> 503
  GET  /status?session=U   -> nav-trial's JSON shape; foreign session -> 409
  POST /shutdown?session=U -> foreign -> 403; own -> exit the process
  /ui                      -> the Gradio app (iframed by the website)

Runs identically native (uv, MPS) and in the demo container (CPU) — no
container-only assumptions: shutdown exits THIS process (not PID 1 blindly),
readiness is "checkpoint + env loaded" printed as DEMO READY (the
orchestrator's readyLog), device comes from config.INFERENCE_DEVICE.
"""

import os
import threading
import time
from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vla_trial.config import (
    DEMO_CHECKPOINT,
    DEMO_FLEET_BUDGET,
    DEMO_PORT,
    DEMO_SESSION_SECONDS,
    INFERENCE_DEVICE,
)
from vla_trial.demo.ui import build_ui

state = {
    "session": None,
    "claimed_at": None,
    "ready": False,
    "runner": None,
    "start": time.time(),
    "log": ["gateway up — loading model + env…"],
}


def _boot() -> None:
    """Heavy load in a thread so /status answers from the first second."""
    try:
        state["log"].append(f"loading {DEMO_CHECKPOINT} on {INFERENCE_DEVICE}…")
        from vla_trial.demo.episode_runner import EpisodeRunner

        runner = EpisodeRunner()
        state["runner"] = runner
        state["ready"] = True
        state["log"].append(f"ready — {runner.device} inference, checkpoint {runner.checkpoint}")
        print("DEMO READY", flush=True)  # the orchestrator's readyLog line
    except Exception as e:  # surface boot failures in the page's log pane
        state["log"].append(f"BOOT FAILED: {e}")
        print(f"BOOT FAILED: {e}", flush=True)
        raise


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=_boot, daemon=True).start()
    yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    # Exact-origin reflect (ACAO:* is invalid with credentials): prod site +
    # localhost dev, same shape as nav-trial's gateway.
    allow_origin_regex=r"^https://(www\.)?robium\.(ai|org)$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _busy() -> bool:
    return state["runner"] is not None and state["runner"].busy


@app.post("/start")
def start(session: str | None = None):
    # A claim is sacred only while a run executes; an idle claim is takeable
    # (page-reload semantics, same as nav-trial).
    if _busy() and session != state["session"]:
        return JSONResponse({"error": "busy"}, status_code=503)
    if session != state["session"]:
        state["claimed_at"] = time.time()
    state["session"] = session or "anonymous"
    state["claimed_at"] = state["claimed_at"] or time.time()
    return {"ok": True}


@app.get("/status")
def status(session: str | None = None):
    if state["session"] and session != state["session"]:
        return JSONResponse({"error": "not your instance"}, status_code=409)
    up = int(time.time() - (state["claimed_at"] or state["start"]))
    return {
        "claimed": state["session"] is not None,
        "ready": state["ready"],
        "rtf": None,  # kept for the shared Status shape; meaningless here
        "nodes": 0,
        "uptime_s": up,
        "remaining_s": max(0, DEMO_SESSION_SECONDS - up),
        "fleet": {"running": None, "budget": DEMO_FLEET_BUDGET},
        "log": state["log"],
    }


@app.post("/shutdown")
def shutdown(session: str | None = None):
    if state["session"] is None or session != state["session"]:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    # Answer first, then exit THIS process: PID 1 in the container (AutoRemove
    # reaps it), a plain uv-run process natively.
    threading.Timer(0.2, os._exit, args=(0,)).start()
    return {"bye": True}


@app.get("/")
def root():
    return {"service": "robium demo gateway (vla-trial)"}


app = gr.mount_gradio_app(app, build_ui(lambda: state["runner"]), path="/ui")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="info")


if __name__ == "__main__":
    main()
