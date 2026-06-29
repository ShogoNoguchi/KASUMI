"""Standalone one-condition entrypoint for manual Shachi runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shachi.env.japan_policy_scientist import PolicyConfig, run_policy_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--policy_json", default=None)
    args = parser.parse_args()
    policy = PolicyConfig.baseline()
    if args.policy_json:
        policy = PolicyConfig.model_validate(
            json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
        )
    metrics = run_policy_experiment(
        output_dir=args.out_dir,
        baseline_policy=PolicyConfig.baseline(),
        intervention_policy=policy,
        config_path=args.config,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
