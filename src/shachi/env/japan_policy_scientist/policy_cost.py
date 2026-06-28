"""Versioned synthetic policy-implementation burden accounting.

Points are not yen, API expenditure, or empirical cost estimates. This release
exposes them to The AI Scientist and enforces the same 35-point ceiling for all
interventions so policies are compared under an equal synthetic resource rule.
Simulation compute accounting is handled separately in a human-only audit.
"""
from __future__ import annotations

from .schemas import PolicyConfig

# Burden points per unit change from the human-authored baseline. Values are
# deliberately simple and versioned. They do not affect Environment transitions
# or bureaucrat-agent prompts, but they do constrain AI-Scientist policy preflight.
POSITIVE_DELTA_COSTS: dict[str, float] = {
    "staffing_buffer": 180.0,
    "workload_triage_support": 12.0,
    "manager_support": 8.0,
    "management_case_capacity": 4.0,
    "explanation_quality": 5.0,
    "preference_matching": 12.0,
    "learning_access_probability": 8.0,
    "protected_learning_share": 20.0,
    "digital_support": 10.0,
    "process_reform_support": 8.0,
    "specialist_track_access": 6.0,
    "appeal_channel": 4.0,
    "hiring_fill_rate": 3.0,
    "hiring_capacity_rate": 100.0,
    "midcareer_hire_share": 2.0,
    "transfer_capacity_rate": 120.0,
    "transfer_explanation_quality": 4.0,
    "productivity_savings_capture": 1.0,
}


def policy_cost_breakdown(
    policy: PolicyConfig,
    baseline: PolicyConfig | None = None,
) -> dict[str, float]:
    baseline = baseline or PolicyConfig.baseline()
    costs: dict[str, float] = {}
    for field, unit_cost in POSITIVE_DELTA_COSTS.items():
        delta = max(0.0, float(getattr(policy, field)) - float(getattr(baseline, field)))
        costs[field] = delta * unit_cost
    # Reducing forced movement requires extra matching, consultation, and appeal
    # capacity, so this direction also consumes synthetic resources.
    costs["reduced_involuntary_transfer_share"] = (
        max(0.0, baseline.involuntary_transfer_share - policy.involuntary_transfer_share) * 20.0
    )
    return costs


def policy_implementation_cost_points(
    policy: PolicyConfig,
    baseline: PolicyConfig | None = None,
) -> float:
    return float(sum(policy_cost_breakdown(policy, baseline).values()))


def validate_policy_budget(
    policy: PolicyConfig,
    *,
    max_points: float,
    baseline: PolicyConfig | None = None,
) -> float:
    cost = policy_implementation_cost_points(policy, baseline)
    if cost > max_points + 1e-9:
        breakdown = policy_cost_breakdown(policy, baseline)
        largest = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)[:5]
        raise ValueError(
            f"Policy implementation cost {cost:.3f} exceeds the fixed budget {max_points:.3f}; "
            f"largest components={largest}"
        )
    return cost
