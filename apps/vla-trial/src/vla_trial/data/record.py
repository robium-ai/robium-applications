"""Roll the oracle out N times and write a LeRobotDataset v3.

Scripted-oracle collection, not teleop: we own the MJCF, so the cube's pose is
known, and 75 episodes take minutes unattended and are regenerable after any
scene tweak. Teleop would be an hour of tedium with variable quality.

API note (Task 7 Step 1, installed lerobot==0.6.0): the brief's sample code
called ``dataset.add_frame(frame, task=TASK)`` — that signature does not
exist. The installed ``LeRobotDataset.add_frame(self, frame: dict) -> None``
takes ONLY the frame dict, and its docstring says the frame "must include a
'task' key"; the writer does ``frame.pop("task")`` internally. So the task
string goes INSIDE the per-frame dict, not as a call kwarg. ``save_episode``
similarly takes no ``task`` argument in this release.
"""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_trial.config import (
    CONTROL_FPS,
    DATASET_REPO_ID,
    IMG_H,
    IMG_W,
    N_EPISODES,
    N_JOINTS,
    TASK,
)
from vla_trial.env.so101_pick import SO101PickEnv
from vla_trial.oracle.scripted_pick import rollout

FEATURES = {
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.scene": {
        "dtype": "video",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (N_JOINTS,),
        "names": [f"joint_{i}" for i in range(N_JOINTS)],
    },
    "action": {
        "dtype": "float32",
        "shape": (N_JOINTS,),
        "names": [f"joint_{i}" for i in range(N_JOINTS)],
    },
}


def record(
    n_episodes: int = N_EPISODES,
    repo_id: str = DATASET_REPO_ID,
    push: bool = False,
) -> Path:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=CONTROL_FPS,
        features=FEATURES,
        robot_type="so101",
        use_videos=True,
    )

    env = SO101PickEnv()
    kept = 0
    seed = 0
    discarded = 0
    # DISCARD/RETRY, not "skip". The oracle succeeds ~85-92% on UNSEEN spawns
    # (Task 5 review measured 9/10 on seeds outside 0-9), so a ~10% discard rate
    # here is EXPECTED, not a bug — a discarded episode is one where the cube was
    # ejected off the pedestal on an untuned corner, and it must never enter the
    # training data (that is the poison the canary exists to keep out). We keep
    # drawing fresh seeds until N clean episodes exist.
    #
    # But guard against a real regression masquerading as bad luck: if the
    # success rate collapses, something broke (a scene edit, a detuned grasp),
    # and we must fail loudly rather than loop forever writing nothing.
    max_attempts = n_episodes * 3
    while kept < n_episodes:
        attempts = kept + discarded
        if attempts >= max_attempts:
            env.close()
            raise RuntimeError(
                f"oracle success rate collapsed: {kept} kept / {attempts} attempts "
                f"({kept / attempts:.0%}) after {max_attempts} tries. Expected ~85-92%. "
                "Something regressed — re-run `make oracle` before recording."
            )
        result = rollout(env, seed=seed)
        seed += 1
        if not result["success"]:
            discarded += 1
            print(f"seed {seed - 1}: discarded (oracle miss, expected ~10%)")
            continue

        for frame in result["frames"]:
            dataset.add_frame({**frame, "task": TASK})
        dataset.save_episode()
        kept += 1
        print(f"episode {kept}/{n_episodes} (seed {seed - 1}, {result['n_steps']} steps)")

    env.close()
    rate = kept / (kept + discarded)
    print(f"recorded {kept} clean episodes; discarded {discarded} "
          f"(oracle success {rate:.0%} on unseen seeds)")

    if push:
        dataset.push_to_hub()

    return dataset.root
