"""Survivor-aware current public release metrics and auditable artifact writing."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .schemas import MonthlyAgentRecord, PolicyLabResult
from .dynamics import TransitionParameters, welfare_from_strain_pressure

MetricValue = float | None


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return float(mean(rows)) if rows else None


def _mean0(values: Iterable[float]) -> float:
    value = _mean(values)
    return 0.0 if value is None else value


def _quantile(values: Iterable[float], q: float) -> float | None:
    rows = sorted(float(value) for value in values)
    if not rows:
        return None
    position = (len(rows) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    weight = position - lower
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _event_rates_by_group(
    result: PolicyLabResult,
    post_records: list[MonthlyAgentRecord],
    intervention_start_month: int,
) -> dict[str, MetricValue]:
    profiles = {profile.person_id: profile for profile in result.profiles}
    person_months: Counter[str] = Counter()
    for row in post_records:
        profile = profiles.get(row.person_id)
        if profile is None:
            continue
        person_months["high_family" if profile.family_constraint >= 0.65 else "lower_family"] += 1
        person_months["junior" if profile.rank == "junior" else "nonjunior"] += 1
    counts: Counter[str] = Counter()
    for event in result.staffing_events:
        if event.month < intervention_start_month:
            continue
        profile = profiles.get(event.person_id)
        if profile is None:
            continue
        family = "high_family" if profile.family_constraint >= 0.65 else "lower_family"
        career = "junior" if profile.rank == "junior" else "nonjunior"
        counts[f"{event.event_type}_{family}"] += 1
        counts[f"{event.event_type}_{career}"] += 1
        if event.exit_reason:
            counts[f"exit_reason_{event.exit_reason}"] += 1
    result_rates: dict[str, MetricValue] = {
        key: float(value) for key, value in counts.items()
    }
    for event_type in ("exit", "voluntary_transfer", "involuntary_transfer", "hire"):
        for group in ("high_family", "lower_family", "junior", "nonjunior"):
            result_rates[f"{event_type}_rate_per_100_person_months__{group}"] = (
                100.0 * counts[f"{event_type}_{group}"] / person_months[group]
                if person_months[group] > 0
                else None
            )
    return result_rates


def aggregate_metrics(
    *,
    result: PolicyLabResult,
    intervention_start_month: int,
    initial_slots: int,
    runtime_seconds: float,
    usage: dict[str, float],
    cache: dict[str, float],
) -> dict[str, MetricValue]:
    post_records = [
        row for row in result.monthly_records if row.month >= intervention_start_month
    ]
    post_departments = [
        row for row in result.department_rows if int(row["month"]) >= intervention_start_month
    ]
    post_tasks = [
        row for row in result.task_rows if row.month >= intervention_start_month
    ]
    profiles = {profile.person_id: profile for profile in result.profiles}
    initial_profiles = [profile for profile in result.profiles if profile.identity_epoch == 0]
    initial_ids = {profile.person_id for profile in initial_profiles}
    post_exit_events = [
        event
        for event in result.staffing_events
        if event.event_type == "exit" and event.month >= intervention_start_month
    ]
    all_exit_ids = {
        event.person_id
        for event in result.staffing_events
        if event.event_type == "exit"
    }
    initial_exit_ids = initial_ids & all_exit_ids
    initial_survivor_ids = initial_ids - initial_exit_ids

    initial_records = [row for row in post_records if row.initial_cohort]
    replacement_records = [row for row in post_records if not row.initial_cohort]
    initial_survivor_records = [
        row for row in initial_records if row.person_id in initial_survivor_ids
    ]
    high_family_records = [
        row
        for row in post_records
        if row.person_id in profiles and profiles[row.person_id].family_constraint >= 0.65
    ]
    lower_family_records = [
        row
        for row in post_records
        if row.person_id in profiles and profiles[row.person_id].family_constraint < 0.65
    ]
    junior_records = [
        row
        for row in post_records
        if row.person_id in profiles and profiles[row.person_id].rank == "junior"
    ]
    nonjunior_records = [
        row
        for row in post_records
        if row.person_id in profiles and profiles[row.person_id].rank != "junior"
    ]

    action_counts = Counter(row.action.work_response for row in post_records)
    voice_counts = Counter(row.action.voice_action for row in post_records)
    career_counts = Counter(row.action.career_action for row in post_records)
    person_months = len(post_records)
    post_staffing = [
        event for event in result.staffing_events if event.month >= intervention_start_month
    ]
    departures = sum(event.event_type == "exit" for event in post_staffing)
    hires = sum(event.event_type == "hire" for event in post_staffing)
    voluntary_transfers = sum(
        event.event_type == "voluntary_transfer" for event in post_staffing
    )
    involuntary_transfers = sum(
        event.event_type == "involuntary_transfer" for event in post_staffing
    )
    active_end = float(result.summary.get("final_active_headcount", 0))
    final_month = int(result.summary.get("months", 0) or 0)
    active_end_reconciled = sum(
        float(row["active_headcount_end"])
        for row in result.department_rows
        if int(row["month"]) == final_month
    )

    support_outcomes = [
        outcome
        for outcome in result.management_outcomes
        if outcome.month >= intervention_start_month
        and outcome.request_kind in {"operational_support", "staffing_relief"}
    ]
    reform_outcomes = [
        outcome
        for outcome in result.management_outcomes
        if outcome.month >= intervention_start_month
        and outcome.request_kind == "process_reform"
    ]
    risk_outcomes = [
        outcome
        for outcome in result.management_outcomes
        if outcome.month >= intervention_start_month
        and outcome.request_kind == "operational_risk"
    ]
    post_exposures = [
        exposure
        for exposure in result.exposure_events
        if exposure.effective_month >= intervention_start_month
    ]
    post_management_outcomes = [
        outcome for outcome in result.management_outcomes
        if outcome.month >= intervention_start_month
    ]
    outside_docket_deferred = [
        outcome for outcome in post_management_outcomes
        if outcome.decision_status == "deferred"
        and "did not enter" in outcome.public_message
    ]
    identity_invalidated_outcomes = [
        outcome for outcome in post_management_outcomes
        if outcome.decision_status == "identity_invalidated"
    ]

    mean_fatigue = _mean0(
        row.action.self_report.fatigue_pct / 100.0 for row in post_records
    )
    mean_turnover = _mean0(
        row.action.self_report.turnover_intent_pct / 100.0 for row in post_records
    )
    mean_strain = _mean0(row.work_strain_pressure_after for row in post_records)
    mean_trust = _mean0(
        row.action.self_report.organizational_trust_pct / 100.0 for row in post_records
    )
    mean_fairness = _mean0(
        row.action.self_report.procedural_fairness_pct / 100.0 for row in post_records
    )
    initial_exit_rate = _rate(len(initial_exit_ids), len(initial_ids)) or 0.0
    post_initial_exit_ids = {
        event.person_id
        for event in result.staffing_events
        if event.event_type == "exit"
        and event.month >= intervention_start_month
        and event.person_id in initial_ids
    }
    initial_exit_rate_post = _rate(len(post_initial_exit_ids), len(initial_ids)) or 0.0
    initial_cohort_retention = 1.0 - initial_exit_rate
    strain_welfare = (
        welfare_from_strain_pressure(mean_strain, TransitionParameters())
        if post_records
        else 0.0
    )
    mechanical_welfare_anchor = (
        mean([strain_welfare, initial_cohort_retention]) if post_records else 0.0
    )
    sealed_survey_welfare = (
        mean([1.0 - mean_fatigue, 1.0 - mean_turnover, mean_trust, mean_fairness])
        if post_records
        else 0.0
    )
    # Predeclared dual-channel endpoint. Survey values are useful ABM outputs,
    # but receive a minority weight and are causally sealed from future actions
    # and transitions. Service/fairness guardrails are applied before ranking.
    primary_welfare = 0.75 * mechanical_welfare_anchor + 0.25 * sealed_survey_welfare

    high_family_initial_ids = {
        profile.person_id for profile in initial_profiles if profile.family_constraint >= 0.65
    }
    high_family_involuntary = sum(
        event.event_type == "involuntary_transfer"
        and event.person_id in high_family_initial_ids
        and event.month >= intervention_start_month
        for event in result.staffing_events
    )

    time_errors = [
        abs(
            row.effective_core_share
            + row.effective_coordination_share
            + row.effective_learning_share
            + row.effective_process_share
            - 1.0
        )
        for row in post_records
    ]
    metrics: dict[str, MetricValue] = {
        "active_headcount_end": active_end,
        "active_headcount_end_reconciled": active_end_reconciled,
        "vacancy_rate_end": max(0.0, (initial_slots - active_end) / max(initial_slots, 1)),
        "cumulative_departures": float(departures),
        "cumulative_hires": float(hires),
        "cumulative_voluntary_transfers": float(voluntary_transfers),
        "cumulative_involuntary_transfers": float(involuntary_transfers),
        "initial_cohort_cumulative_exit_rate": initial_exit_rate,
        "initial_cohort_post_intervention_exit_rate": initial_exit_rate_post,
        "initial_cohort_retention_rate": initial_cohort_retention,
        "initial_high_family_involuntary_transfer_rate": _rate(
            high_family_involuntary, len(high_family_initial_ids)
        ),
        "departure_events_per_initial_slot": departures / max(initial_slots, 1),
        "ever_active_person_count": float(len(result.profiles)),
        "primary_staff_welfare_composite_post": float(primary_welfare),
        "mechanical_welfare_anchor_post": float(mechanical_welfare_anchor),
        "sealed_survey_welfare_composite_post": float(sealed_survey_welfare),
        "sealed_survey_weight_in_primary": 0.25,
        "mean_reported_fatigue_post_person_month": mean_fatigue,
        "mean_turnover_intent_post_person_month": mean_turnover,
        "mean_modeled_work_strain_post_person_month": mean_strain,
        "mean_modeled_work_strain_pressure_post_person_month": mean_strain,
        "strain_welfare_transform_post": float(strain_welfare),
        "p90_modeled_work_strain_pressure_post_person_month": _quantile([row.work_strain_pressure_after for row in post_records], 0.90),
        "p95_modeled_work_strain_pressure_post_person_month": _quantile([row.work_strain_pressure_after for row in post_records], 0.95),
        "p99_modeled_work_strain_pressure_post_person_month": _quantile([row.work_strain_pressure_after for row in post_records], 0.99),
        "max_modeled_work_strain_pressure_post_person_month": max((row.work_strain_pressure_after for row in post_records), default=None),
        "share_person_months_strain_pressure_ge_1_post": _mean(float(row.work_strain_pressure_after >= 1.0) for row in post_records),
        "share_person_months_strain_pressure_ge_2_post": _mean(float(row.work_strain_pressure_after >= 2.0) for row in post_records),
        "mean_organizational_trust_post": mean_trust,
        "mean_procedural_fairness_post": mean_fairness,
        "mean_exploratory_ebpm_interest_post": _mean(
            row.action.self_report.ebpm_interest_pct / 100.0 for row in post_records
        ),
        "mean_exploratory_dx_improvement_interest_post": _mean(
            row.action.self_report.dx_improvement_interest_pct / 100.0
            for row in post_records
        ),
        "initial_cohort_mean_reported_fatigue_post_person_month": _mean(
            row.action.self_report.fatigue_pct / 100.0 for row in initial_records
        ),
        "initial_cohort_mean_turnover_intent_post_person_month": _mean(
            row.action.self_report.turnover_intent_pct / 100.0 for row in initial_records
        ),
        "initial_cohort_survivor_only_mean_reported_fatigue_post": _mean(
            row.action.self_report.fatigue_pct / 100.0
            for row in initial_survivor_records
        ),
        "initial_cohort_survivor_only_mean_turnover_intent_post": _mean(
            row.action.self_report.turnover_intent_pct / 100.0
            for row in initial_survivor_records
        ),
        "replacement_mean_reported_fatigue_post_person_month": _mean(
            row.action.self_report.fatigue_pct / 100.0 for row in replacement_records
        ),
        "mean_turnover_intent_high_family_constraint_post": _mean(
            row.action.self_report.turnover_intent_pct / 100.0
            for row in high_family_records
        ),
        "mean_turnover_intent_lower_family_constraint_post": _mean(
            row.action.self_report.turnover_intent_pct / 100.0
            for row in lower_family_records
        ),
        "mean_turnover_intent_junior_post": _mean(
            row.action.self_report.turnover_intent_pct / 100.0 for row in junior_records
        ),
        "mean_turnover_intent_nonjunior_post": _mean(
            row.action.self_report.turnover_intent_pct / 100.0
            for row in nonjunior_records
        ),
        "relative_effort_mean_post": _mean(
            row.action.relative_effort_pct / 100.0 for row in post_records
        ),
        "relative_effort_p50_post": _quantile(
            (row.action.relative_effort_pct / 100.0 for row in post_records), 0.50
        ),
        "relative_effort_p90_post": _quantile(
            (row.action.relative_effort_pct / 100.0 for row in post_records), 0.90
        ),
        "relative_effort_p99_post": _quantile(
            (row.action.relative_effort_pct / 100.0 for row in post_records), 0.99
        ),
        "share_relative_effort_gt_120_post": _mean(
            float(row.action.relative_effort_pct > 120) for row in post_records
        ),
        "share_relative_effort_gt_160_post": _mean(
            float(row.action.relative_effort_pct > 160) for row in post_records
        ),
        "health_protecting_response_share_post": _mean(
            float(
                row.action.work_response
                in {
                    "protect_health_capacity",
                    "take_health_leave",
                    "caregiving_leave",
                    "refuse_overtime",
                }
            )
            for row in post_records
        ),
        "mean_personal_completion_ratio_post": _mean(
            row.actual_completion_ratio for row in post_records
        ),
        "mean_department_workload_ratio_post": _mean(
            float(row["workload_ratio"]) for row in post_departments
        ),
        "mean_department_completion_ratio_post": _mean(
            float(row["completion_ratio"]) for row in post_departments
        ),
        "mean_department_effective_demand_served_ratio_post": _mean(
            float(row["effective_demand_served_ratio"]) for row in post_departments
        ),
        "mean_department_backlog_units_post": _mean(
            float(row["backlog_units"]) for row in post_departments
        ),
        "mean_critical_overdue_units_post": _mean(
            float(row["critical_overdue_units"]) for row in post_departments
        ),
        "cumulative_deferred_work_units_post": sum(
            float(row["deferred_work_units"]) for row in post_departments
        ),
        "cumulative_outsourced_work_units_post": sum(
            float(row["outsourced_work_units"]) for row in post_departments
        ),
        "cumulative_service_harm_points_post": sum(
            float(row["service_harm_points"]) for row in post_departments
        ),
        "cumulative_quality_error_units_post": sum(
            float(row["quality_error_units"]) for row in post_departments
        ),
        "cumulative_rework_generated_units_post": sum(
            float(row["rework_generated_units"]) for row in post_departments
        ),
        "terminal_liability_points": sum(
            float(row["terminal_liability_points"])
            for row in result.department_rows
            if int(row["month"]) == int(result.summary.get("months", 0))
        ),
        "cumulative_recipient_coordination_cost_units_post": sum(
            float(row["recipient_coordination_cost_units"]) for row in post_departments
        ),
        "task_ledger_row_count_post": float(len(post_tasks)),
        "management_request_count": float(len(post_management_outcomes)),
        "management_outside_docket_deferred_count": float(len(outside_docket_deferred)),
        "management_identity_invalidated_count": float(len(identity_invalidated_outcomes)),
        "management_approval_rate_all_requests": _rate(
            sum(outcome.approved for outcome in post_management_outcomes), len(post_management_outcomes)
        ),
        "management_outside_docket_deferred_share": _rate(
            len(outside_docket_deferred), len(post_management_outcomes)
        ),
        "support_request_count": float(len(support_outcomes)),
        "support_request_approval_rate": _rate(
            sum(outcome.approved for outcome in support_outcomes), len(support_outcomes)
        ),
        "approved_temporary_support_units_total": sum(
            outcome.approved_support_units for outcome in support_outcomes if outcome.approved
        ),
        "process_reform_request_count": float(len(reform_outcomes)),
        "process_reform_approval_rate": _rate(
            sum(outcome.approved for outcome in reform_outcomes), len(reform_outcomes)
        ),
        "operational_risk_request_count": float(len(risk_outcomes)),
        "operational_risk_approval_rate": _rate(
            sum(outcome.approved for outcome in risk_outcomes), len(risk_outcomes)
        ),
        "realized_exposure_count_post": float(len(post_exposures)),
        "implemented_process_reform_exposure_count_post": float(
            sum(
                exposure.exposure_type == "implemented_process_reform"
                for exposure in post_exposures
            )
        ),
        "approved_triage_exposure_units_post": sum(
            exposure.units
            for exposure in post_exposures
            if exposure.exposure_type == "approved_triage"
        ),
        "individual_support_exposure_units_post": sum(
            exposure.units
            for exposure in post_exposures
            if exposure.exposure_type == "individual_support"
        ),
        "formal_event_id_reference_validity_rate": 1.0 if post_records else None,
        "accepted_response_valid_event_reference_rate": 1.0 if post_records else None,
        "max_time_conservation_absolute_error": max(time_errors, default=0.0),
        "runtime_seconds": runtime_seconds,
        "initial_cohort_person_month_count_post": float(len(initial_records)),
        "replacement_person_month_count_post": float(len(replacement_records)),
        "high_family_person_month_count_post": float(len(high_family_records)),
        "lower_family_person_month_count_post": float(len(lower_family_records)),
        "junior_person_month_count_post": float(len(junior_records)),
        "nonjunior_person_month_count_post": float(len(nonjunior_records)),
    }

    denominator = max(person_months, 1)
    for response in (
        "deliver_normally",
        "work_overtime",
        "prioritize_core_work",
        "request_support",
        "defer_low_priority_work",
        "protect_health_capacity",
        "take_health_leave",
        "caregiving_leave",
        "refuse_overtime",
    ):
        metrics[f"work_response_share__{response}"] = action_counts[response] / denominator
    for action in (
        "none",
        "ask_for_explanation",
        "request_staffing_relief",
        "propose_process_reform",
        "raise_operational_risk",
    ):
        metrics[f"voice_action_share__{action}"] = voice_counts[action] / denominator
    for action in (
        "stay",
        "request_transfer",
        "request_specialist_track",
        "explore_external_exit",
    ):
        metrics[f"career_action_share__{action}"] = career_counts[action] / denominator

    metrics.update(_event_rates_by_group(result, post_records, intervention_start_month))
    metrics.update(usage)
    metrics.update(cache)
    provider_attempts = max(float(metrics.get("llm_provider_attempts") or 0.0), 1.0)
    metrics["structured_validation_failure_rate_per_provider_attempt"] = (
        float(metrics.get("llm_validation_failures") or 0.0) / provider_attempts
    )
    metrics["network_failure_rate_per_provider_attempt"] = (
        float(metrics.get("llm_network_failures") or 0.0) / provider_attempts
    )
    # Compatibility aliases, with explicit modeled/self-report terminology.
    metrics["mean_reported_fatigue_post"] = metrics[
        "mean_reported_fatigue_post_person_month"
    ]
    metrics["mean_turnover_intent_post"] = metrics[
        "mean_turnover_intent_post_person_month"
    ]
    metrics["mean_objective_fatigue_index_post"] = None
    metrics["event_grounding_rate"] = metrics[
        "formal_event_id_reference_validity_rate"
    ]
    metrics["structured_output_violation_rate"] = metrics[
        "structured_validation_failure_rate_per_provider_attempt"
    ]
    return metrics


def add_baseline_deltas(
    metrics: dict[str, MetricValue], baseline_final_info: Path | None
) -> dict[str, MetricValue]:
    if baseline_final_info is None or not baseline_final_info.exists():
        return metrics
    baseline_payload = json.loads(baseline_final_info.read_text(encoding="utf-8"))
    baseline = baseline_payload["policy_lab"]["means"]
    for key, value in list(metrics.items()):
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key in baseline
            and isinstance(baseline[key], (int, float))
            and not isinstance(baseline[key], bool)
        ):
            metrics[f"delta_vs_baseline__{key}"] = float(value) - float(baseline[key])
    cost = float(metrics.get("policy_implementation_cost_points") or 0.0)
    avoided_departures = float(baseline.get("cumulative_departures", 0.0) or 0.0) - float(
        metrics.get("cumulative_departures") or 0.0
    )
    service_loss_reduction = float(
        baseline.get("cumulative_service_harm_points_post", 0.0) or 0.0
    ) - float(metrics.get("cumulative_service_harm_points_post") or 0.0)
    strain_reduction = float(
        baseline.get("mean_modeled_work_strain_post_person_month", 0.0) or 0.0
    ) - float(metrics.get("mean_modeled_work_strain_post_person_month") or 0.0)
    metrics["policy_cost_points_per_departure_avoided"] = (
        cost / avoided_departures if avoided_departures > 0 else None
    )
    metrics["policy_cost_points_per_service_harm_point_reduced"] = (
        cost / service_loss_reduction if service_loss_reduction > 0 else None
    )
    metrics["policy_cost_points_per_work_strain_unit_reduced"] = (
        cost / strain_reduction if strain_reduction > 0 else None
    )
    return metrics


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_model_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    payloads = [row.model_dump(mode="json") for row in rows]
    columns = list(payloads[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(payloads)


def _write_survival_curve(output_dir: Path, result: PolicyLabResult) -> None:
    initial_ids = {
        profile.person_id for profile in result.profiles if profile.identity_epoch == 0
    }
    exits_by_month: Counter[int] = Counter(
        event.month
        for event in result.staffing_events
        if event.event_type == "exit" and event.person_id in initial_ids
    )
    at_risk = len(initial_ids)
    survival = 1.0
    rows: list[dict[str, float | int]] = []
    max_month = int(result.summary.get("months", 0) or 0)
    for month in range(1, max_month + 1):
        exits = exits_by_month[month]
        if at_risk > 0:
            survival *= 1.0 - exits / at_risk
        rows.append(
            {
                "month": month,
                "initial_cohort_at_risk_start": at_risk,
                "initial_cohort_exits": exits,
                "kaplan_meier_survival": survival,
            }
        )
        at_risk -= exits
    if rows:
        with (output_dir / "initial_cohort_survival_curve.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_artifacts(
    *,
    output_dir: Path,
    result: PolicyLabResult,
    metrics: dict[str, MetricValue],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "population_profiles.jsonl", result.profiles)
    _write_jsonl(output_dir / "monthly_agent_records.jsonl", result.monthly_records)
    _write_jsonl(output_dir / "staffing_events.jsonl", result.staffing_events)
    _write_jsonl(output_dir / "transfer_plan.jsonl", result.transfer_plans)
    _write_jsonl(output_dir / "transfer_requests.jsonl", result.transfer_requests)
    _write_jsonl(output_dir / "management_outcomes.jsonl", result.management_outcomes)
    _write_jsonl(output_dir / "exposure_events.jsonl", result.exposure_events)
    _write_jsonl(output_dir / "task_ledger.jsonl", result.task_rows)
    _write_jsonl(output_dir / "quarterly_reflections.jsonl", result.reflections)
    _write_model_csv(output_dir / "transfer_requests.csv", result.transfer_requests)
    _write_model_csv(output_dir / "exposure_events.csv", result.exposure_events)
    _write_model_csv(output_dir / "task_ledger.csv", result.task_rows)
    if result.department_rows:
        columns = list(result.department_rows[0].keys())
        with (output_dir / "department_monthly.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(result.department_rows)
    _write_survival_curve(output_dir, result)
    (output_dir / "result_summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "final_info.json").write_text(
        json.dumps({"policy_lab": {"means": metrics}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
