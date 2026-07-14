# The M0 container spike. Deliberately CPU-only: Docker on macOS cannot see
# MPS, and Cloud Run has no GPU — so this measures what the deployed demo will
# actually experience.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libosmesa6 git \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv pip install --system -e .

CMD ["python", "-m", "vla_trial.run", "spike-policy", "cpu"]
