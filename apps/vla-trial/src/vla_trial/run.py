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

    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
