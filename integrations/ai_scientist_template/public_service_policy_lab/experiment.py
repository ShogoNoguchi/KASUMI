"""AI Scientist v1 entry point for one matched Shachi policy condition.

SCIENTIST-ONLY CONTRACT
=======================
- run_0 is the human-authored baseline.
- 120 slots, 48 months, months 1-12 warm-up, policy begins month 13.
- Exits, hiring, transfers, and support institutions operate from month 1; only
  PolicyConfig changes at month 13.
- Exactly four intervention runs are required: run_1 through run_4.
- The AI Scientist may edit only ``candidate_policy.json``.
- ``experiment.py`` and ``plot.py`` are immutable and verified by SHA-256.
- Every intervention must remain within the fixed 35-point synthetic policy budget.
- Prompts, schemas, transition equations, metrics, cache identity, seed, and
  model contract are protected.
- Python calculates absolute work, deadline-aware task ledgers, quality/rework,
  modeled work strain, staffing, and simultaneous transfers.
- Employee self-reports are sealed evaluation outputs: they are not written to
  future prompt memory and do not drive exit, management priority, or transitions.
- A deterministic finite management gate allocates support/triage/reform envelopes;
  optional LLM management is an ablation, and employees do not communicate directly.
- Candidate selection rules and current public release development-only evidence rules are frozen outside the editable JSON.
- The LLM returns bounded actions only.

Command: python experiment.py --out_dir=run_i
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shachi.env.japan_policy_scientist import run_policy_experiment

from template_contract import (
    BASELINE_POLICY,
    MAX_INTERVENTION_RUNS,
    load_candidate_policy,
    preflight_run,
    validate_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.name.startswith("run_"):
        raise ValueError("out_dir must be named run_i")
    try:
        run_num = int(out_dir.name.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError("out_dir must be named run_i with integer i") from exc
    if run_num < 0 or run_num > MAX_INTERVENTION_RUNS:
        raise ValueError("This PoC permits run_0 plus exactly four intervention runs")
    intervention = (
        BASELINE_POLICY
        if run_num == 0
        else load_candidate_policy(Path(__file__).with_name("candidate_policy.json"), run_name=out_dir.name)
    )
    # This is deliberately before run_policy_experiment: immutable-source,
    # strict-JSON, schema, budget, and duplicate-policy failures are rejected
    # before any provider call.
    preflight_run(out_dir=out_dir, run_num=run_num, policy=intervention)
    metrics = run_policy_experiment(
        output_dir=out_dir,
        baseline_policy=BASELINE_POLICY,
        intervention_policy=intervention,
        config_path=Path(__file__).with_name("japan_policy_scientist.yaml"),
    )
    validate_run(out_dir, run_num=run_num)
    print(json.dumps({"policy_lab": {"means": metrics}}, indent=2))


if __name__ == "__main__":
    main()
