"""M0 spike: SmolVLA forward-pass latency on this machine.

Spec risk 1 — the demo's core unverified assumption. HF's "runs on a MacBook"
is a marketing claim, not a measurement, and no published benchmark exists.

Measure THREE numbers:
  make spike-policy            # MPS native + CPU native
  make spike-policy-container  # CPU inside linux/arm64 — the one that decides

The container number is decisive because Docker on macOS cannot see MPS, so the
demo container is CPU on the Mac *and* CPU on Cloud Run.

The bar is POLICY_LATENCY_CEILING_S (~1 s), not 30 Hz: action chunking means one
forward pass yields ~50 actions ≈ 1.7 s of robot motion at 30 FPS.
"""

import json
import statistics
import time
from pathlib import Path

import torch

from vla_trial.config import (
    BASE_POLICY_ID,
    IMG_H,
    IMG_W,
    N_JOINTS,
    POLICY_SPIKE_JSON,
    SPIKE_OUTPUT_DIR,
    TASK,
)

# First timed pass pays lazy-init/kernel-compile costs far beyond the rest —
# 2 warm-up passes, as sketched in the brief, left no headroom; use 5.
N_WARMUP_PASSES = 5


def _runtime() -> str:
    """"native" on the host, "container" inside Docker.

    The container spike bind-mounts the host's ``outputs/`` and also runs on
    device "cpu", so keying policy.json by device ALONE (as the brief's
    sketch did) makes the container run silently OVERWRITE the native CPU
    result — the two numbers we most need to compare. Key on device+runtime.
    """
    return "container" if Path("/.dockerenv").exists() else "native"


def _dummy_batch(policy, device: str) -> dict:
    """One synthetic observation, shaped like the loaded checkpoint expects.

    Ground truth (established by running this against the real
    `lerobot/smolvla_base` checkpoint, not assumed — the brief's sketch used
    different key names):
      - the pretrained config declares three GENERIC image keys,
        ``observation.images.camera{1,2,3}``, each ``(3, 256, 256)`` — not
        the env's own "wrist"/"scene" camera names. Env-specific camera
        naming/remapping is a Task 4+ concern, not this latency
        measurement's, so keys are taken from
        ``policy.config.image_features`` rather than hardcoded.
      - a 6-dim ``observation.state`` (matches ``N_JOINTS``), from
        ``policy.config.robot_state_feature``.
      - ``select_action`` itself does NOT tokenize "task" — it requires the
        batch to already carry ``observation.language.tokens`` /
        ``observation.language.attention_mask``. That only happens by
        running the raw batch through the policy's own pre-processor
        pipeline (``make_pre_post_processors``, see ``bench_policy``) before
        calling ``select_action`` — the real inference contract for this
        policy family, not a detail the brief's sketch mentioned.

    No batch dimension here: the preprocessor's ``AddBatchDimensionProcessorStep``
    adds it.
    """
    batch = {}
    for key, feat in policy.config.image_features.items():
        c, h, w = feat.shape
        assert (h, w) == (IMG_H, IMG_W), f"{key} shape {feat.shape} != config IMG_H/IMG_W"
        batch[key] = torch.rand(c, h, w, device=device)

    state_dim = policy.config.robot_state_feature.shape[0]
    assert state_dim == N_JOINTS, f"robot_state_feature dim {state_dim} != N_JOINTS"
    batch["observation.state"] = torch.rand(state_dim, device=device)
    batch["task"] = TASK
    return batch


def bench_policy(device: str = "cpu", n_passes: int = 20) -> dict:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(BASE_POLICY_ID)
    policy.to(device)
    policy.eval()

    # The real select_action contract: tokenization/normalization/batching
    # happen in this pre-processor pipeline, not inside select_action.
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=BASE_POLICY_ID,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    batch = preprocessor(_dummy_batch(policy, device))

    # Warm-up: first passes pay lazy-init and kernel-compile costs.
    with torch.inference_mode():
        for _ in range(N_WARMUP_PASSES):
            policy.reset()
            policy.select_action(batch)

    timings = []
    with torch.inference_mode():
        for _ in range(n_passes):
            policy.reset()  # force a real forward pass, not a cached chunk step
            start = time.perf_counter()
            policy.select_action(batch)
            timings.append(time.perf_counter() - start)

    runtime = _runtime()
    result = {
        "device": device,
        "runtime": runtime,
        "policy": BASE_POLICY_ID,
        "n_passes": n_passes,
        "n_warmup_passes": N_WARMUP_PASSES,
        "mean_s": statistics.mean(timings),
        # Median is the honest central estimate here: the container run shows
        # occasional multi-second stalls (Docker Desktop VM scheduling) that
        # drag mean/p95 far above the typical pass.
        "median_s": statistics.median(timings),
        "p95_s": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "min_s": min(timings),
        "max_s": max(timings),
        "timings_s": timings,
    }

    SPIKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if POLICY_SPIKE_JSON.is_file():
        existing = json.loads(POLICY_SPIKE_JSON.read_text())
    existing[f"{device}-{runtime}"] = result
    POLICY_SPIKE_JSON.write_text(json.dumps(existing, indent=2))
    return result
