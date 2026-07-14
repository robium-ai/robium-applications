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
    POLICY_SPIKE_N_PASSES,
    POLICY_SPIKE_WARMUP_PASSES,
    TASK,
)


def _sync(device: str) -> None:
    """Block until all queued async device ops finish.

    MPS (and CUDA) ops are dispatched asynchronously: ``select_action()``
    returns as soon as the work is *queued*, not once it is *computed*.
    Nothing downstream in the LeRobot call chain (``select_action`` ->
    ``_get_action_chunk`` -> ``sample_actions`` -> ``vlm_with_expert.forward``)
    calls ``.cpu()``/``.item()``/``.numpy()`` to force a wait, so timing with
    a bare ``time.perf_counter()`` around it measures dispatch time, not
    compute time — silently flattering MPS. CPU is synchronous already; this
    is a no-op there.
    """
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


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


def bench_policy(
    device: str = "cpu",
    n_passes: int = POLICY_SPIKE_N_PASSES,
    output_json: Path | None = None,
    n_warmup_passes: int = POLICY_SPIKE_WARMUP_PASSES,
) -> dict:
    """Time ``n_passes`` real SmolVLA forward passes on ``device``.

    ``output_json`` defaults to ``POLICY_SPIKE_JSON`` (the canonical M0
    artifact). The TEST must pass a tmp path: a 3-pass unit-test run writing
    into the canonical file would silently overwrite the real 20-pass
    benchmark with a throwaway number — which it did, once, before this
    parameter existed.

    ``n_warmup_passes=0`` turns this into a cold-call probe: one timed pass
    with no warm-up at all, used to check the "is the steady-state timing a
    real forward pass and not a warmed-up cache artifact" claim from code
    rather than an out-of-band, unverifiable number (see
    ``test_cold_call_is_within_factor_of_2_of_steady_state``).
    """
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

    # Warm-up: first passes pay lazy-init and kernel-compile costs. Synced on
    # both sides of each call so the LAST warm-up pass's queued MPS work is
    # fully drained before the timed loop's first pass starts — otherwise the
    # first timed measurement would silently include leftover warm-up work.
    with torch.inference_mode():
        for _ in range(n_warmup_passes):
            policy.reset()
            _sync(device)
            policy.select_action(batch)
            _sync(device)

    timings = []
    with torch.inference_mode():
        for _ in range(n_passes):
            policy.reset()  # force a real forward pass, not a cached chunk step
            _sync(device)  # drain anything still queued before starting the clock
            start = time.perf_counter()
            policy.select_action(batch)
            _sync(device)  # block until the forward pass actually finishes
            timings.append(time.perf_counter() - start)

    runtime = _runtime()
    result = {
        "device": device,
        "runtime": runtime,
        "policy": BASE_POLICY_ID,
        "n_passes": n_passes,
        "n_warmup_passes": n_warmup_passes,
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

    out = POLICY_SPIKE_JSON if output_json is None else output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.is_file():
        existing = json.loads(out.read_text())
    existing[f"{device}-{runtime}"] = result
    out.write_text(json.dumps(existing, indent=2))
    return result
