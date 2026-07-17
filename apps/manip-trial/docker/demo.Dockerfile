# The demo container: the session gateway + Gradio/Rerun UI on :8765.
# CPU-only by design — Docker on macOS cannot see MPS; native MPS runs use
# `make demo` instead. Unlike vla-trial there is NO Hub fetch and NO token:
# gym-pusht renders with pygame (no GL stack), and every artifact the demo
# needs (rung checkpoints, ladder.json, eval videos) is COPY'd from local
# outputs/ — build after `make train-ladder eval-ladder`.
FROM python:3.12-slim

# ffmpeg: lerobot's video stack imports torchcodec, which needs the ffmpeg
# shared libraries present even though the demo never decodes a dataset.
# build-essential: pymunk (gym-pusht's physics) ships no linux/arm64 wheel
# and compiles from source at install time.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# SDL dummy driver: pygame without a display (belt-and-braces — rgb_array
# rendering is offscreen already).
ENV SDL_VIDEODRIVER=dummy \
    PORT=8765 \
    HF_HOME=/opt/hf

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# -e is load-bearing: config.APP_ROOT resolves from config.py's __file__, so
# the module must live at /app/src (editable), not in site-packages — or the
# baked outputs/ tree below would never be found.
RUN pip install --no-cache-dir uv && uv pip install --system -e .

# Bake the ladder: manifest, rung checkpoints (pretrained_model only — the
# .dockerignore drops training_state), and the gallery's eval videos.
COPY outputs/demo ./outputs/demo
COPY outputs/eval/ladder ./outputs/eval/ladder
COPY outputs/train/act_pusht_ladder ./outputs/train/act_pusht_ladder
COPY outputs/train/act_pusht_10k/checkpoints/010000/pretrained_model ./outputs/train/act_pusht_10k/checkpoints/010000/pretrained_model

# Boot probe at BUILD time: loads the default rung + constructs/renders the
# env — a broken bake fails the build, not a visitor's session.
RUN python -c "from manip_trial.demo.episode_runner import EpisodeRunner; EpisodeRunner()"

# Runtime never touches the Hub.
ENV HF_HUB_OFFLINE=1

EXPOSE 8765
CMD ["python", "-m", "manip_trial.demo.gateway"]
