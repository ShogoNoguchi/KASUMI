#!/usr/bin/env python3
"""Run a small paid, paired behavioral pilot before the expensive baseline.

No generic LLM smoke call is made. The same synthetic profile receives a normal
and a high-stress realized-event observation with the same provider seed. The
mechanical gate checks whether the selected model reacts directionally and can
use the newly available health-protecting actions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from shachi.agent.japan_policy_bureaucrat import (
    JapanPolicyBureaucratAgent,
    SharedLLMRuntime,
    UsageLedger,
)
from shachi.env.japan_policy_scientist.behavioral_gate import (
    evaluate_paired_behavioral_pilot,
)
from shachi.env.japan_policy_scientist.fingerprints import (
    behavioral_pilot_config_payload,
    canonical_json,
    runtime_fingerprints,
    sha256_text,
)
from shachi.env.japan_policy_scientist.memory import BureaucratMemory
from shachi.env.japan_policy_scientist.population import generate_initial_profiles
from shachi.env.japan_policy_scientist.schemas import (
    BureaucracyMessage,
    BureaucracyObservation,
    BureaucratMonthlyAction,
    DepartmentState,
    WorkEvent,
)


def make_observation(profile, *, stress: bool) -> BureaucracyObservation:
    condition = "stress" if stress else "normal"
    workload = 1.52 if stress else 0.86
    severity = 3 if stress else 0
    department = DepartmentState(
        department=profile.department,
        active_headcount=100,
        workload_ratio=workload,
        backlog_units=18.0 if stress else 1.5,
        completion_ratio=0.72 if stress else 0.98,
    )
    events = [
        WorkEvent(
            event_id=f"pilot-{profile.slot_id:03d}-{condition}-workload",
            month=1,
            event_type="monthly_workload",
            description=(
                "Required work substantially exceeds available departmental capacity, and several deadlines overlap."
                if stress
                else "Required work is within available departmental capacity and deadlines are predictable."
            ),
            objective_workload_ratio=workload,
            after_hours_severity=severity,
        )
    ]
    if stress:
        events.extend(
            [
                WorkEvent(
                    event_id=f"pilot-{profile.slot_id:03d}-{condition}-afterhours",
                    month=1,
                    event_type="after_hours_shock",
                    description=(
                        "Unplanned late-night work occurred on several days while existing family or caregiving responsibilities remained unchanged."
                    ),
                    objective_workload_ratio=workload,
                    after_hours_severity=severity,
                ),
                WorkEvent(
                    event_id=f"pilot-{profile.slot_id:03d}-{condition}-support",
                    month=1,
                    event_type="support_request_outcome",
                    description=(
                        "A staffing-relief request received no substantive response before the deadline."
                    ),
                    objective_workload_ratio=workload,
                    after_hours_severity=severity,
                ),
            ]
        )
    else:
        events.append(
            WorkEvent(
                event_id=f"pilot-{profile.slot_id:03d}-{condition}-support",
                month=1,
                event_type="management_response",
                description=(
                    "A routine workload question received a clear response within three business days."
                ),
                objective_workload_ratio=workload,
            )
        )
    return BureaucracyObservation(
        agent_id=profile.slot_id,
        messages=[
            BureaucracyMessage(
                time=1,
                src_agent_id=None,
                dst_agent_id=profile.slot_id,
                content="Paired workplace observation",
            )
        ],
        reward=None,
        response_type=BureaucratMonthlyAction,
        tools=[],
        phase="monthly_action",
        month=1,
        identity_epoch=profile.identity_epoch,
        profile_summary=profile.prompt_summary(),
        department_summary=department.prompt_summary(),
        personal_objective_summary=(
            "No prior-month subjective score is supplied. Base the decision on the realized events above."
        ),
        recent_events=events,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm_cfg = config["llm"]
    pilot_cfg = config["behavioral_pilot"]
    pair_count = int(args.pairs or pilot_cfg["paired_profiles"])
    model = os.environ.get("POLICYLAB_SHACHI_MODEL", llm_cfg["model"])
    evidence_mode = "mock" if os.environ.get("POLICYLAB_MOCK_LLM") == "1" else "live"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operator_audit_dir = args.output_dir / ".operator_audit"
    operator_audit_dir.mkdir(parents=True, exist_ok=True)
    ledger = UsageLedger(
        operator_audit_dir / "llm_usage.jsonl",
        max_cost_usd=float(
            os.environ.get(
                "POLICYLAB_OPERATOR_HARD_STOP_USD",
                os.environ.get(
                    "POLICYLAB_PILOT_MAX_LLM_COST_USD",
                    pilot_cfg.get("max_cost_usd", 2.0),
                ),
            )
        ),
        campaign_log_path=os.environ.get("POLICYLAB_CAMPAIGN_LEDGER"),
        campaign_id=os.environ.get("POLICYLAB_CAMPAIGN_ID"),
    )
    runtime = SharedLLMRuntime(
        model=model,
        temperature=float(llm_cfg["temperature"]),
        max_concurrency=min(int(llm_cfg["max_concurrency"]), 8),
        network_retries=int(llm_cfg["network_retries"]),
        validation_retries=int(llm_cfg["validation_retries"]),
        retry_sleep_seconds=float(llm_cfg["retry_sleep_seconds"]),
        reasoning_effort=str(llm_cfg["reasoning_effort"]),
        ledger=ledger,
        base_seed=int(config["experiment"]["fixed_seed"]),
    )
    profiles = generate_initial_profiles(
        num_agents=pair_count,
        seed=int(config["experiment"]["fixed_seed"]),
    )

    async def one(profile, stress: bool):
        # Fresh memory for each arm prevents carry-over and preserves pairing.
        agent = JapanPolicyBureaucratAgent(
            slot_id=profile.slot_id,
            runtime=runtime,
            monthly_output_token_estimate=int(llm_cfg.get("monthly_output_token_estimate", 900)),
            quarterly_output_token_estimate=int(llm_cfg.get("quarterly_output_token_estimate", 900)),
            memory=BureaucratMemory(monthly_window=0, quarterly_window=0),
        )
        observation = make_observation(profile, stress=stress)
        response = await agent.step(observation)
        return BureaucratMonthlyAction.model_validate(response)

    started = time.time()
    pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for profile in profiles:
        normal_result, stress_result = await asyncio.gather(
            one(profile, False), one(profile, True), return_exceptions=True
        )
        if isinstance(normal_result, BaseException) or isinstance(stress_result, BaseException):
            failures.append(
                {
                    "slot_id": profile.slot_id,
                    "normal_error": (
                        f"{type(normal_result).__name__}: {normal_result}"
                        if isinstance(normal_result, BaseException)
                        else None
                    ),
                    "stress_error": (
                        f"{type(stress_result).__name__}: {stress_result}"
                        if isinstance(stress_result, BaseException)
                        else None
                    ),
                }
            )
            continue
        pair_rows.append(
            {
                "slot_id": profile.slot_id,
                "person_id": profile.person_id,
                "profile": profile.model_dump(mode="json"),
                "normal_action": normal_result.model_dump(mode="json"),
                "stress_action": stress_result.model_dump(mode="json"),
            }
        )

    pairs_path = args.output_dir / "paired_behavioral_pilot_pairs.jsonl"
    failures_path = args.output_dir / "paired_behavioral_pilot_failures.jsonl"
    pairs_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    failures_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )

    thresholds = {
        key: pilot_cfg[key]
        for key in (
            "min_valid_pair_rate",
            "min_stress_fatigue_increase_share",
            "min_stress_turnover_increase_share",
            "min_stress_effort_decrease_share",
            "min_stress_effort_le_85_share",
            "min_health_protecting_action_share_under_stress",
            "min_distinct_work_responses_under_stress",
            "max_identical_action_pair_share",
        )
    }
    gate = evaluate_paired_behavioral_pilot(
        pair_rows=pair_rows,
        failures=failures,
        thresholds=thresholds,
    )
    report = {
        "package_version": runtime_fingerprints()["package_version"],
        "model": model,
        "evidence_mode": evidence_mode,
        "temperature": float(llm_cfg["temperature"]),
        "fixed_seed": int(config["experiment"]["fixed_seed"]),
        "config_hash": sha256_text(canonical_json(behavioral_pilot_config_payload(config))),
        "fingerprints": runtime_fingerprints(),
        "started_unix_time": started,
        "completed_unix_time": time.time(),
        "gate": gate,
        "operator_accounting_hidden_from_scientist": True,
        "pairs_file": pairs_path.name,
        "failures_file": failures_path.name,
    }
    report_path = args.output_dir / "paired_behavioral_pilot_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (operator_audit_dir / "operator_audit.json").write_text(
        json.dumps(
            {
                "operator_only": True,
                "visibility_contract": (
                    "Not passed to bureaucrat agents, idea generation, Aider, policy ranking, notes, plots, or papers."
                ),
                "usage": ledger.snapshot(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    marker = args.output_dir / "paired_behavioral_pilot.pass.json"
    if gate["passed"]:
        marker.write_text(
            json.dumps(
                {
                    "passed": True,
                    "report": report_path.name,
                    "fingerprints": report["fingerprints"],
                    "config_hash": report["config_hash"],
                    "model": model,
                    "evidence_mode": evidence_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        marker.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=None)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(
            "Paired behavioral pilot failed the mechanical gate. Inspect the report; "
            "do not start the 160-slot baseline with this provider/model contract."
        )


if __name__ == "__main__":
    main()
