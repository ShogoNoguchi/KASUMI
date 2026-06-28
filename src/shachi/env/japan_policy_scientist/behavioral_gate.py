"""Mechanical acceptance gate for the paired behavioral pilot.

The gate does not claim human validity. It rejects a provider/model configuration
that is mechanically unresponsive to paired workplace stress, produces no
health-protecting variation, never reduces effort under stress, or fails
structured validation.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import BureaucratMonthlyAction

HEALTH_RESPONSES = {
    "protect_health_capacity",
    "take_health_leave",
    "caregiving_leave",
    "refuse_overtime",
}


def evaluate_paired_behavioral_pilot(
    *,
    pair_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    expected_pairs = len(pair_rows) + len(failures)
    valid_pair_rate = len(pair_rows) / max(expected_pairs, 1)
    stress_actions: list[BureaucratMonthlyAction] = []
    normal_actions: list[BureaucratMonthlyAction] = []
    fatigue_increases = 0
    turnover_increases = 0
    effort_decreases = 0
    identical_action_pairs = 0
    paired_effort_drops: list[int] = []

    for row in pair_rows:
        normal = BureaucratMonthlyAction.model_validate(row["normal_action"])
        stress = BureaucratMonthlyAction.model_validate(row["stress_action"])
        normal_actions.append(normal)
        stress_actions.append(stress)
        fatigue_increases += int(
            stress.self_report.fatigue_pct > normal.self_report.fatigue_pct
        )
        turnover_increases += int(
            stress.self_report.turnover_intent_pct
            > normal.self_report.turnover_intent_pct
        )
        effort_decreases += int(
            stress.relative_effort_pct < normal.relative_effort_pct
        )
        paired_effort_drops.append(
            normal.relative_effort_pct - stress.relative_effort_pct
        )
        identical_action_pairs += int(
            stress.relative_effort_pct == normal.relative_effort_pct
            and stress.work_response == normal.work_response
            and stress.voice_action == normal.voice_action
            and stress.career_action == normal.career_action
        )

    denominator = max(len(pair_rows), 1)
    stress_counts = Counter(action.work_response for action in stress_actions)
    metrics = {
        "expected_pair_count": expected_pairs,
        "valid_pair_count": len(pair_rows),
        "failed_pair_count": len(failures),
        "valid_pair_rate": valid_pair_rate,
        "stress_fatigue_increase_share": fatigue_increases / denominator,
        "stress_turnover_increase_share": turnover_increases / denominator,
        "stress_effort_decrease_share": effort_decreases / denominator,
        "mean_paired_effort_drop_pct_points": (
            sum(paired_effort_drops) / denominator
        ),
        "health_protecting_action_share_under_stress": sum(
            action.work_response in HEALTH_RESPONSES for action in stress_actions
        )
        / denominator,
        "distinct_work_responses_under_stress": len(stress_counts),
        "identical_action_pair_share": identical_action_pairs / denominator,
        "stress_work_response_counts": dict(sorted(stress_counts.items())),
        "normal_effort_median": _median(
            [action.relative_effort_pct for action in normal_actions]
        ),
        "stress_effort_min": min(
            (action.relative_effort_pct for action in stress_actions), default=0
        ),
        "stress_effort_median": _median(
            [action.relative_effort_pct for action in stress_actions]
        ),
        "stress_effort_le_85_share": sum(
            action.relative_effort_pct <= 85 for action in stress_actions
        )
        / denominator,
    }
    checks = {
        "valid_pair_rate": metrics["valid_pair_rate"]
        >= float(thresholds["min_valid_pair_rate"]),
        "stress_fatigue_response": metrics["stress_fatigue_increase_share"]
        >= float(thresholds["min_stress_fatigue_increase_share"]),
        "stress_turnover_response": metrics["stress_turnover_increase_share"]
        >= float(thresholds["min_stress_turnover_increase_share"]),
        "paired_effort_reduction": metrics["stress_effort_decrease_share"]
        >= float(thresholds["min_stress_effort_decrease_share"]),
        "low_effort_capacity_protection": metrics["stress_effort_le_85_share"]
        >= float(thresholds["min_stress_effort_le_85_share"]),
        "health_protecting_behavior": metrics[
            "health_protecting_action_share_under_stress"
        ]
        >= float(thresholds["min_health_protecting_action_share_under_stress"]),
        "behavioral_diversity": metrics["distinct_work_responses_under_stress"]
        >= int(thresholds["min_distinct_work_responses_under_stress"]),
        "paired_nonidentity": metrics["identical_action_pair_share"]
        <= float(thresholds["max_identical_action_pair_share"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "claim_boundary": (
            "This gate checks mechanical behavioral responsiveness and schema reliability only; "
            "it does not validate synthetic agents against real civil servants."
        ),
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    middle = len(rows) // 2
    if len(rows) % 2:
        return float(rows[middle])
    return (rows[middle - 1] + rows[middle]) / 2.0
