"""Deterministic correlated synthetic populations for current public release.

The distributions are deliberately synthetic and are not estimates of Japanese
civil-service demographics.  current public release replaces the faulty earlier release slot-to-
department formula with an exact-quota assignment: largest-remainder quotas are
computed from the declared scenario weights, then assigned to a keyed
permutation of slots.  Every declared department is therefore populated when
``num_agents >= len(DEPARTMENTS)`` and the department counts are reproducible.

"Scenario" means a frozen robustness condition.  It is not Shachi's
cross-environment "living in multiple worlds" construct.
"""
from __future__ import annotations

import math

from .schemas import (
    AgentProfile,
    DEPARTMENTS,
    DepartmentName,
    FieldName,
    SyntheticScenarioName,
)
from .shock_tape import keyed_uniform

CAREERS = ("generalist", "specialist", "managerial", "mixed")
TRACKS = ("policy_generalist", "administrative_generalist", "specialist")

DEPARTMENT_FIELD_WEIGHTS: dict[DepartmentName, tuple[FieldName, ...]] = {
    "policy_planning": ("economics", "public_management", "social_policy", "law"),
    "budget_coordination": ("economics", "public_management", "law", "data"),
    "regulatory_affairs": ("law", "economics", "social_policy", "public_management"),
    "digital_transformation": ("data", "engineering", "public_management", "economics"),
    "public_service_operations": ("social_policy", "public_management", "data", "law"),
}

SCENARIO_DEPARTMENT_WEIGHTS: dict[SyntheticScenarioName, tuple[float, ...]] = {
    "reference_stressed": (0.19, 0.21, 0.18, 0.17, 0.25),
    "junior_pipeline_shortage": (0.18, 0.22, 0.18, 0.19, 0.23),
    "caregiving_pressure": (0.19, 0.20, 0.18, 0.17, 0.26),
}

# Backward-compatible input spelling accepted only at the configuration edge.
SCENARIO_ALIASES = {"young_staff_shortage": "junior_pipeline_shortage"}


def normalize_scenario(value: str) -> SyntheticScenarioName:
    normalized = SCENARIO_ALIASES.get(value, value)
    if normalized not in SCENARIO_DEPARTMENT_WEIGHTS:
        raise ValueError(
            f"unknown synthetic scenario {value!r}; expected one of "
            f"{sorted(SCENARIO_DEPARTMENT_WEIGHTS)}"
        )
    return normalized  # type: ignore[return-value]


def _choice(seed: int, key: str, values: tuple[str, ...]) -> str:
    idx = min(int(keyed_uniform(seed, key) * len(values)), len(values) - 1)
    return values[idx]


def department_quotas(
    num_agents: int,
    scenario: SyntheticScenarioName = "reference_stressed",
) -> dict[DepartmentName, int]:
    """Return exact largest-remainder department quotas summing to ``num_agents``.

    Ties in the fractional remainder are resolved by the stable department
    order.  With at least five agents and strictly positive weights, each
    department receives at least one slot; this is asserted rather than silently
    accepted because an empty department invalidates the organizational model.
    """

    if num_agents < len(DEPARTMENTS):
        raise ValueError("num_agents must be at least the number of departments")
    weights = SCENARIO_DEPARTMENT_WEIGHTS[scenario]
    total = sum(weights)
    exact = [num_agents * weight / total for weight in weights]
    quotas = [int(math.floor(value)) for value in exact]
    remainder = num_agents - sum(quotas)
    order = sorted(
        range(len(DEPARTMENTS)),
        key=lambda index: (-(exact[index] - quotas[index]), index),
    )
    for index in order[:remainder]:
        quotas[index] += 1
    result = {
        department: quotas[index]
        for index, department in enumerate(DEPARTMENTS)
    }
    if sum(result.values()) != num_agents:
        raise AssertionError("department quotas did not reconcile")
    if any(count <= 0 for count in result.values()):
        raise AssertionError(f"every department must be populated, got {result}")
    return result


def department_assignments(
    num_agents: int,
    seed: int,
    scenario: SyntheticScenarioName = "reference_stressed",
) -> tuple[DepartmentName, ...]:
    """Assign exact quotas to a deterministic keyed permutation of slots."""

    quotas = department_quotas(num_agents, scenario)
    labels: list[DepartmentName] = []
    for department in DEPARTMENTS:
        labels.extend([department] * quotas[department])
    slot_order = sorted(
        range(num_agents),
        key=lambda slot_id: (
            keyed_uniform(seed, "department-assignment", scenario, slot_id),
            slot_id,
        ),
    )
    assignments: list[DepartmentName | None] = [None] * num_agents
    for slot_id, department in zip(slot_order, labels, strict=True):
        assignments[slot_id] = department
    if any(department is None for department in assignments):
        raise AssertionError("department assignment left an unassigned slot")
    return tuple(assignments)  # type: ignore[return-value]


def _rank_and_years(
    *,
    seed: int,
    prefix: str,
    scenario: SyntheticScenarioName,
    new_hire: bool,
    midcareer: bool,
) -> tuple[str, int]:
    if new_hire and not midcareer:
        return "junior", int(keyed_uniform(seed, prefix, "years") * 3)
    if new_hire and midcareer:
        return "mid", 4 + int(keyed_uniform(seed, prefix, "years") * 9)
    r = keyed_uniform(seed, prefix, "rank")
    if scenario == "junior_pipeline_shortage":
        junior_cut, mid_cut = 0.25, 0.76
    else:
        junior_cut, mid_cut = 0.39, 0.83
    rank = "junior" if r < junior_cut else "mid" if r < mid_cut else "senior"
    years = (
        int(keyed_uniform(seed, prefix, "years") * 5)
        if rank == "junior"
        else 5 + int(keyed_uniform(seed, prefix, "years") * 10)
        if rank == "mid"
        else 15 + int(keyed_uniform(seed, prefix, "years") * 16)
    )
    return rank, years


def _profile(
    *,
    seed: int,
    slot_id: int,
    person_id: str,
    identity_epoch: int,
    department: DepartmentName,
    scenario: SyntheticScenarioName,
    new_hire: bool,
    midcareer: bool = False,
    target_field: FieldName | None = None,
) -> AgentProfile:
    prefix = f"profile:{scenario}:{person_id}:{identity_epoch}"
    rank, years = _rank_and_years(
        seed=seed,
        prefix=prefix,
        scenario=scenario,
        new_hire=new_hire,
        midcareer=midcareer,
    )

    field = target_field or _choice(
        seed, prefix + ":field", DEPARTMENT_FIELD_WEIGHTS[department]
    )
    specialist_field = field in {"data", "engineering", "law"}
    track_draw = keyed_uniform(seed, prefix, "track")
    if specialist_field and track_draw < 0.55:
        track = "specialist"
    elif track_draw < 0.70:
        track = "policy_generalist"
    else:
        track = "administrative_generalist"
    if track == "specialist":
        career = "specialist" if keyed_uniform(seed, prefix, "career") < 0.70 else "mixed"
    elif rank == "senior" and keyed_uniform(seed, prefix, "career") < 0.55:
        career = "managerial"
    else:
        career = _choice(seed, prefix + ":career", CAREERS)

    family_base = 0.14 + 0.018 * min(years, 20)
    if scenario == "caregiving_pressure":
        family_base += 0.25
    family = max(
        0.02,
        min(0.98, family_base + 0.42 * keyed_uniform(seed, prefix, "family-noise")),
    )
    autonomy = 0.22 + 0.62 * keyed_uniform(seed, prefix, "autonomy")
    external = 0.08 + 0.26 * autonomy + (0.28 if specialist_field else 0.08)
    external += 0.12 if rank == "mid" else 0.03
    external += 0.18 * keyed_uniform(seed, prefix, "market-noise")
    external = max(0.02, min(0.95, external))
    transfer_aversion = 0.10 + 0.55 * family + 0.25 * keyed_uniform(
        seed, prefix, "transfer-aversion-noise"
    )
    psm = 0.48 + 0.28 * (1.0 - external) + 0.20 * keyed_uniform(seed, prefix, "psm")

    rank_capacity = {"junior": 0.82, "mid": 1.04, "senior": 1.16}[rank]
    baseline_capacity = rank_capacity * (
        0.91 + 0.16 * keyed_uniform(seed, prefix, "capacity")
    )
    initial_skill = (
        0.80
        + min(years, 20) * 0.016
        + 0.13 * keyed_uniform(seed, prefix, "skill")
    )
    authority = (
        "section_support"
        if rank == "senior"
        else "team_lead"
        if rank == "mid" and years >= 10
        else "staff"
    )

    return AgentProfile(
        slot_id=slot_id,
        person_id=person_id,
        identity_epoch=identity_epoch,
        department=department,
        rank=rank,  # type: ignore[arg-type]
        years_service=years,
        field=field,  # type: ignore[arg-type]
        employment_track=track,  # type: ignore[arg-type]
        career_orientation=career,  # type: ignore[arg-type]
        authority_scope=authority,  # type: ignore[arg-type]
        family_constraint=family,
        public_service_motivation=max(0.05, min(0.98, psm)),
        autonomy_need=max(0.05, min(0.98, autonomy)),
        external_market_pull=external,
        transfer_aversion=max(0.05, min(0.98, transfer_aversion)),
        baseline_capacity_units=baseline_capacity,
        initial_skill_stock=min(1.5, initial_skill),
    )


def generate_initial_profiles(
    num_agents: int,
    seed: int,
    scenario: SyntheticScenarioName = "reference_stressed",
) -> list[AgentProfile]:
    assignments = department_assignments(num_agents, seed, scenario)
    return [
        _profile(
            seed=seed,
            slot_id=slot_id,
            person_id=f"initial-{slot_id:04d}",
            identity_epoch=0,
            department=assignments[slot_id],
            scenario=scenario,
            new_hire=False,
        )
        for slot_id in range(num_agents)
    ]


def generate_replacement_profile(
    *,
    seed: int,
    slot_id: int,
    identity_epoch: int,
    month: int,
    department: DepartmentName,
    midcareer: bool,
    scenario: SyntheticScenarioName = "reference_stressed",
    target_field: FieldName | None = None,
) -> AgentProfile:
    # ``month`` is accepted for API compatibility but deliberately excluded from
    # the person identity and random key. Composition therefore does not change
    # merely because two otherwise identical policies hire in different months.
    del month
    person_id = f"hire-s{slot_id:04d}-e{identity_epoch}"
    return _profile(
        seed=seed,
        slot_id=slot_id,
        person_id=person_id,
        identity_epoch=identity_epoch,
        department=department,
        scenario=scenario,
        new_hire=True,
        midcareer=midcareer,
        target_field=target_field,
    )
