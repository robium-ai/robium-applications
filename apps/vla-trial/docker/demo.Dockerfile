# The demo container: the session gateway + Gradio/Rerun UI on :8765.
# CPU-only by design — Docker on macOS cannot see MPS and the (future) cloud
# target has no GPU either; native MPS runs use `make demo` instead.
#
# Build via `make demo-image` — the checkpoint repos are private, so the bake
# step needs an HF token as a BuildKit secret (never an ENV/ARG: those persist
# in image history).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libosmesa6 git \
    && rm -rf /var/lib/apt/lists/*

# osmesa: pure-CPU software GL — the only MuJoCo backend that needs no GPU or
# display. The bake step below actually renders, so a broken GL fails the
# BUILD, not a visitor's session.
ENV MUJOCO_GL=osmesa \
    VLA_DEVICE=cpu \
    HF_HOME=/opt/hf \
    PORT=8765

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir uv && uv pip install --system -e .

# Hermetic assets: re-vendor menagerie in the image rather than trusting
# whatever the local checkout happens to contain.
RUN bash scripts/fetch_assets.sh

# Bake the model into the image by running the real boot path once:
# downloads the private fine-tune checkpoint + the public SmolVLM2 backbone
# into HF_HOME, loads them, constructs the env, and renders a frame.
RUN --mount=type=secret,id=hf_token \
    sh -c 'HF_TOKEN=$(cat /run/secrets/hf_token) python -c "from vla_trial.demo.episode_runner import EpisodeRunner; EpisodeRunner()"'

# Runtime never touches the Hub: everything it needs was baked above, and a
# 401 on a private-repo etag check must not be able to break a session.
ENV HF_HUB_OFFLINE=1

EXPOSE 8765
CMD ["python", "-m", "vla_trial.demo.gateway"]
