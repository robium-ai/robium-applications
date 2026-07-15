"""CLI dispatcher: python -m vla_trial.run <subcommand>."""

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m vla_trial.run <subcommand>", file=sys.stderr)
        return 2

    cmd, *rest = argv
    if cmd == "spike-render":
        from vla_trial.spike.bench_render import bench_render
        print(json.dumps(bench_render(), indent=2))
        return 0

    if cmd == "spike-policy":
        from vla_trial.config import POLICY_SPIKE_N_PASSES
        from vla_trial.spike.bench_policy import bench_policy

        devices = rest or ["cpu", "mps"]
        measured = 0
        for device in devices:
            try:
                print(json.dumps(bench_policy(device=device, n_passes=POLICY_SPIKE_N_PASSES), indent=2))
                measured += 1
            except Exception as exc:  # a device may genuinely be unavailable
                print(f"{device}: unavailable ({exc})", file=sys.stderr)
        # Fail loudly if NOTHING was measured. The whole point of this
        # subcommand is to produce a number; a run that measured nothing
        # (e.g. the container hitting an HF Hub 401 while fetching weights)
        # must not exit 0 and look like a successful benchmark.
        if measured == 0:
            print(
                f"spike-policy: measured 0 of {len(devices)} device(s) — no benchmark produced",
                file=sys.stderr,
            )
            return 1
        return 0

    if cmd == "oracle":
        from vla_trial.env.so101_pick import SO101PickEnv
        from vla_trial.oracle.scripted_pick import rollout

        env = SO101PickEnv()
        n = int(rest[0]) if rest else 5
        wins = sum(rollout(env, seed=s)["success"] for s in range(n))
        env.close()
        print(f"oracle: {wins}/{n} succeeded")
        return 0 if wins == n else 1

    if cmd == "viz-oracle":
        from vla_trial.config import VIZ_DIR
        from vla_trial.env.so101_pick import SO101PickEnv
        from vla_trial.oracle.scripted_pick import rollout
        from vla_trial.viz.rerun_logger import RerunLogger

        env = SO101PickEnv()
        logger = RerunLogger(app_id="vla_trial_oracle", save_path=VIZ_DIR / "oracle.rrd")
        result = rollout(env, seed=0, logger=logger)
        logger.close()
        env.close()
        print(f"oracle seed 0: success={result['success']} steps={result['n_steps']}")
        print(f"open with: rerun {VIZ_DIR / 'oracle.rrd'}")
        return 0

    if cmd == "record":
        from vla_trial.data.record import record

        # No push here by design — pushing is `make push-dataset`, gated on review.
        root = record(push=False)
        print(f"dataset written to {root}")
        print("NOT pushed. Review it, then `make push-dataset` when ready.")
        return 0

    if cmd == "spot-check":
        from vla_trial.config import VIZ_DIR
        from vla_trial.env.so101_pick import SO101PickEnv
        from vla_trial.oracle.scripted_pick import rollout
        from vla_trial.viz.rerun_logger import RerunLogger

        env = SO101PickEnv()
        seeds = [int(s) for s in rest] or [0, 1, 2, 3, 4]
        for s in seeds:
            logger = RerunLogger(
                app_id=f"vla_spotcheck_{s}", save_path=VIZ_DIR / f"episode_{s}.rrd"
            )
            r = rollout(env, seed=s, logger=logger)
            logger.close()
            print(
                f"seed {s}: success={r['success']} steps={r['n_steps']} "
                f"-> {VIZ_DIR / f'episode_{s}.rrd'}"
            )
        env.close()
        return 0

    if cmd == "push-dataset":
        # DEFERRED, GATED: exists as code but is run only by the user, after
        # reviewing the locally recorded dataset. Task 7 must never call this.
        from vla_trial.config import DATASET_REPO_ID
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(DATASET_REPO_ID)  # loads the already-recorded local copy
        ds.push_to_hub(private=True)  # PRIVATE by default
        print(f"pushed {DATASET_REPO_ID} to the Hub (private)")
        return 0

    if cmd == "train-smoke":
        import subprocess

        from vla_trial.config import train_smoke_cmd

        return subprocess.run(train_smoke_cmd()).returncode

    if cmd == "train":
        import subprocess

        from vla_trial.config import train_remote_cmd

        # Default is the cheap pipe-test run; `train full` is the real 20k spend.
        pipe_test = "full" not in rest
        cmd_argv = train_remote_cmd(pipe_test=pipe_test)
        kind = "PIPE-TEST" if pipe_test else "FULL 20k"
        print(f"submitting {kind} fine-tune to HF Jobs:\n  {' '.join(cmd_argv)}")
        return subprocess.run(cmd_argv).returncode

    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
