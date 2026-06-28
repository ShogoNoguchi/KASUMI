"""Transparent, versioned transition functions for current public release.

current public release removes hard clipping from the causal work-strain path. Strain is a
latent pressure, not a bounded probability or percentage. Bounded quantities
are produced only by explicit probability/link functions or by schema-level
validation. No LLM is called here; coefficients are synthetic PoC assumptions
and must not be interpreted as estimates for real civil servants or ministries.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .schemas import (
    AgentProfile,
    BureaucratState,
    CareerAction,
    ExitReason,
    WorkMix,
    WorkResponse,
)


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _nonnegative(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _unit_interval(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return value


def logistic(value: float) -> float:
    """Smooth probability link. This is a model link, not a hard cap."""
    value = _finite(value, name="logistic_input")
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softplus(value: float) -> float:
    """Smooth positive transform without an upper bound or hard floor."""
    value = _finite(value, name="softplus_input")
    if value > 40.0:
        return value
    if value < -40.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def smooth_positive(value: float, *, sharpness: float = 4.0) -> float:
    """Differentiable positive-part analogue used instead of hard thresholds."""
    sharpness = _nonnegative(sharpness, name="sharpness")
    if sharpness == 0.0:
        raise ValueError("sharpness must be positive")
    return softplus(sharpness * _finite(value, name="smooth_positive_input")) / sharpness


@dataclass(frozen=True)
class TransitionParameters:
    work_strain_persistence: float = 0.78
    overload_gain: float = 0.24
    overexertion_gain: float = 0.12
    after_hours_gain: float = 0.045
    family_after_hours_interaction_gain: float = 0.040
    forced_transfer_gain: float = 0.13
    support_recovery_gain: float = 0.080
    protected_learning_recovery_gain: float = 0.030
    health_capacity_recovery_gain: float = 0.13
    natural_recovery: float = 0.022

    strain_productivity_penalty: float = 0.32
    exit_strain_pressure_gain: float = 1.65
    quality_strain_pressure_penalty: float = 0.55
    welfare_strain_pressure_penalty: float = 0.90
    skill_productivity_gain: float = 0.20
    onboarding_months: int = 6
    transfer_disruption_penalty: float = 0.10
    recipient_coordination_penalty: float = 0.035

    skill_learning_gain: float = 0.020
    skill_depreciation: float = 0.0015
    modernization_gain: float = 0.022
    modernization_depreciation: float = 0.0020
    codified_knowledge_depreciation: float = 0.0025
    team_routine_depreciation: float = 0.0030
    exit_routine_damage: float = 0.010
    transfer_routine_damage: float = 0.006
    department_tacit_productivity_gain: float = 0.080
    department_codified_productivity_gain: float = 0.018
    department_routine_productivity_gain: float = 0.050
    department_modernization_productivity_gain: float = 0.030

    high_exit_strain_pressure_threshold: float = 0.90
    high_exit_streak_required: int = 2

    staffing_implementation_lag_months: int = 2
    recipient_coordination_cost_rate: float = 0.08
    transfer_request_ttl_months: int = 12
    specialist_track_duration_months: int = 24

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def onboarding_multiplier(months_since_hire: int, onboarding_months: int) -> float:
    if months_since_hire >= onboarding_months:
        return 1.0
    if months_since_hire < 0:
        raise ValueError("months_since_hire must be non-negative")
    return 0.60 + 0.40 * months_since_hire / max(1, onboarding_months)


def compute_capacity_multiplier(
    state: BureaucratState,
    params: TransitionParameters,
) -> float:
    strain_burden = smooth_positive(state.work_strain_pressure)
    strain_multiplier = math.exp(-params.strain_productivity_penalty * strain_burden)
    skill_multiplier = 1.0 + params.skill_productivity_gain * (state.skill_stock - 1.0)
    if skill_multiplier <= 0.0:
        raise ValueError(f"skill multiplier became non-positive: {skill_multiplier}")
    onboarding = onboarding_multiplier(state.months_since_hire, params.onboarding_months)
    disruption = math.exp(
        -params.transfer_disruption_penalty
        * (1.0 if state.transfer_disruption_months > 0 else 0.0)
    )
    coordination = math.exp(
        -params.recipient_coordination_penalty
        * (1.0 if state.recipient_coordination_months > 0 else 0.0)
    )
    return strain_multiplier * skill_multiplier * onboarding * disruption * coordination


def effective_time_shares(
    *, work_mix: WorkMix, protected_learning_share: float
) -> dict[str, float]:
    """Reserve protected time first, then allocate the remainder.

    Protected learning share is validated as an input invariant. It is not
    silently forced into range by the transition function.
    """
    protected = _unit_interval(protected_learning_share, name="protected_learning_share")
    remaining = 1.0 - protected
    shares = {
        "core": remaining * work_mix.core_delivery_pct / 100.0,
        "coordination": remaining * work_mix.coordination_pct / 100.0,
        "learning": protected + remaining * work_mix.learning_pct / 100.0,
        "process": remaining * work_mix.process_improvement_pct / 100.0,
        "protected_learning": protected,
    }
    total = shares["core"] + shares["coordination"] + shares["learning"] + shares["process"]
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"effective time shares violate conservation: {shares}")
    return shares


def update_work_strain_pressure(
    *,
    current_pressure: float,
    assigned_work_ratio: float,
    relative_effort_pct: int,
    after_hours_severity: int,
    family_constraint: float,
    individual_support_ratio: float,
    forced_transfer: bool,
    transfer_explanation_received: bool,
    appeal_relief_realized: bool,
    work_response: WorkResponse,
    protected_learning_time: float,
    params: TransitionParameters,
) -> float:
    """Update latent strain pressure without upper hard clipping.

    Under-load and health-protective behavior may reduce pressure; overload,
    excess effort, after-hours severity, family conflict, and forced transfer may
    raise it. The pressure itself may exceed one, making extreme overwork visible
    to downstream dynamics and metrics.
    """
    current_pressure = _finite(current_pressure, name="current_pressure")
    assigned_work_ratio = _finite(assigned_work_ratio, name="assigned_work_ratio")
    effort_ratio = _finite(relative_effort_pct / 100.0, name="effort_ratio")
    after_hours_severity = int(after_hours_severity)
    if after_hours_severity < 0:
        raise ValueError("after_hours_severity must be non-negative")
    family_constraint = _unit_interval(family_constraint, name="family_constraint")
    individual_support_ratio = _nonnegative(individual_support_ratio, name="individual_support_ratio")
    protected_learning_time = _unit_interval(protected_learning_time, name="protected_learning_time")

    overload = assigned_work_ratio - 1.0
    overexertion = effort_ratio - 1.0
    family_after_hours = family_constraint * after_hours_severity / 4.0
    health_action = work_response in {
        "protect_health_capacity",
        "take_health_leave",
        "caregiving_leave",
        "refuse_overtime",
    }
    reduced_effort = (1.0 - effort_ratio) if health_action else 0.0
    support_recovery = params.support_recovery_gain * math.sqrt(individual_support_ratio)
    forced_transfer_multiplier = 1.0
    if transfer_explanation_received:
        forced_transfer_multiplier *= 0.65
    if appeal_relief_realized:
        forced_transfer_multiplier *= 0.70
    forced_transfer_load = (
        params.forced_transfer_gain * forced_transfer_multiplier if forced_transfer else 0.0
    )
    return (
        params.work_strain_persistence * current_pressure
        + params.overload_gain * overload
        + params.overexertion_gain * overexertion
        + params.after_hours_gain * after_hours_severity
        + params.family_after_hours_interaction_gain * family_after_hours
        + forced_transfer_load
        - support_recovery
        - params.protected_learning_recovery_gain * protected_learning_time
        - params.health_capacity_recovery_gain * reduced_effort
        - params.natural_recovery
    )


def update_work_strain(
    *,
    current: float,
    assigned_work_ratio: float,
    relative_effort_pct: int,
    after_hours_severity: int,
    family_constraint: float,
    individual_support_ratio: float,
    forced_transfer: bool,
    transfer_explanation_received: bool,
    appeal_relief_realized: bool,
    work_response: WorkResponse,
    protected_learning_time: float,
    params: TransitionParameters,
) -> float:
    """Compatibility alias for older no-API tests.

    V1.3.7+ stores the causal state as unbounded work_strain_pressure;
    this wrapper preserves the old function name without restoring hard clipping.
    """
    return update_work_strain_pressure(
        current_pressure=current,
        assigned_work_ratio=assigned_work_ratio,
        relative_effort_pct=relative_effort_pct,
        after_hours_severity=after_hours_severity,
        family_constraint=family_constraint,
        individual_support_ratio=individual_support_ratio,
        forced_transfer=forced_transfer,
        transfer_explanation_received=transfer_explanation_received,
        appeal_relief_realized=appeal_relief_realized,
        work_response=work_response,
        protected_learning_time=protected_learning_time,
        params=params,
    )


def update_skill(
    *,
    current: float,
    learning_investment_ratio: float,
    specialist_track_active: bool,
    params: TransitionParameters,
) -> float:
    """Update tacit individual skill from effort-scaled learning investment."""
    current = _finite(current, name="skill_current")
    learning_investment_ratio = _finite(learning_investment_ratio, name="learning_investment_ratio")
    specialist_multiplier = 1.25 if specialist_track_active else 1.0
    return (
        current * (1.0 - params.skill_depreciation)
        + params.skill_learning_gain
        * learning_investment_ratio
        * specialist_multiplier
    )


def update_modernization(
    *,
    current: float,
    process_investment_ratio: float,
    digital_exposure_intensity: float,
    implemented_reform_intensity: float,
    params: TransitionParameters,
) -> float:
    """Update only from realized digital/reform exposure ledger entries."""
    current = _finite(current, name="modernization_current")
    process_investment_ratio = _finite(process_investment_ratio, name="process_investment_ratio")
    digital_exposure_intensity = _unit_interval(
        digital_exposure_intensity, name="digital_exposure_intensity"
    )
    implemented_reform_intensity = _unit_interval(
        implemented_reform_intensity, name="implemented_reform_intensity"
    )
    realized_support = 0.5 * digital_exposure_intensity + 0.5 * implemented_reform_intensity
    return (
        current * (1.0 - params.modernization_depreciation)
        + params.modernization_gain
        * process_investment_ratio
        * realized_support
    )


def modeled_exit_pressure_probability(
    *,
    work_strain_pressure: float,
    external_market_pull: float,
    public_service_motivation: float,
    career_action: CareerAction,
    params: TransitionParameters,
) -> float:
    """Mechanical resignation pressure, independent of the sealed survey."""
    action_bonus = {
        "stay": 0.0,
        "request_transfer": 0.55,
        "request_specialist_track": 0.20,
        "explore_external_exit": 1.25,
    }[career_action]
    linear = (
        -5.4
        + params.exit_strain_pressure_gain * smooth_positive(work_strain_pressure)
        + 0.9 * _unit_interval(external_market_pull, name="external_market_pull")
        - 1.0 * _unit_interval(public_service_motivation, name="public_service_motivation")
        + action_bonus
    )
    return logistic(linear)


def baseline_voluntary_attrition_probability(profile: AgentProfile) -> float:
    rank_component = {"junior": 0.0008, "mid": 0.0006, "senior": 0.0003}[profile.rank]
    return 0.00035 + rank_component + 0.00035 * profile.external_market_pull


def external_offer_probability(profile: AgentProfile) -> float:
    specialist_bonus = 0.0007 if profile.field in {"data", "engineering"} else 0.0
    return 0.00015 + 0.0016 * profile.external_market_pull + specialist_bonus


def retirement_probability(profile: AgentProfile) -> float:
    if profile.rank != "senior" or profile.years_service < 25:
        return 0.0
    years_over = max(0, profile.years_service - 25)
    return 0.0010 + 0.00035 * years_over


def _probability_to_rate(probability: float, *, name: str) -> float:
    probability = _finite(probability, name=name)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {probability!r}")
    return -math.log1p(-probability)


def draw_competing_exit(
    *,
    draw: float,
    hazards: dict[ExitReason, float],
) -> tuple[ExitReason, float] | None:
    """Single-draw discrete competing-risks attribution from hazard rates.

    Individual hazards are interpreted as one-month event probabilities and are
    converted to rates. Total event probability follows the continuous-time
    competing-risk identity rather than a capped sum.
    """
    draw = _unit_interval(draw, name="exit_draw")
    order: tuple[ExitReason, ...] = (
        "retirement",
        "external_offer",
        "modeled_resignation_pressure",
        "baseline_voluntary_attrition",
    )
    rates = {
        cause: _probability_to_rate(float(hazards.get(cause, 0.0)), name=str(cause))
        for cause in order
    }
    total_rate = sum(rates.values())
    if total_rate <= 0.0:
        return None
    event_probability = 1.0 - math.exp(-total_rate)
    if draw >= event_probability:
        return None
    within_event = draw / event_probability
    cursor = 0.0
    for cause in order:
        probability_mass = rates[cause] / total_rate
        cursor += probability_mass
        if within_event < cursor:
            return cause, 1.0 - math.exp(-rates[cause])
    return order[-1], 1.0 - math.exp(-rates[order[-1]])


def quality_probability(
    *,
    mean_skill: float,
    modernization_stock: float,
    team_routine_stock: float,
    mean_strain_pressure: float,
    params: TransitionParameters,
) -> float:
    score = (
        0.58
        + 0.18 * (_finite(mean_skill, name="mean_skill") - 0.8)
        + 0.08 * _finite(modernization_stock, name="modernization_stock")
        + 0.08 * (_finite(team_routine_stock, name="team_routine_stock") - 0.7)
        - params.quality_strain_pressure_penalty * smooth_positive(mean_strain_pressure)
    )
    return logistic(score)


def welfare_from_strain_pressure(mean_strain_pressure: float, params: TransitionParameters) -> float:
    score = -params.welfare_strain_pressure_penalty * smooth_positive(mean_strain_pressure)
    return math.exp(score)
