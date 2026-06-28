"""Simultaneous capacity-constrained transfer planning.

The plan is computed from one frozen pre-transfer snapshot and then applied as a
batch. An optional maximum-weight min-cost-flow formulation prevents sequential moves
from changing later candidates' targets or donor capacity. A zero-cost no-move
path means the optimizer never fills a quota merely because capacity exists.
"""
from __future__ import annotations

import math
from collections import Counter
from statistics import median

import networkx as nx

from .schemas import (
    AgentProfile,
    BureaucratMonthlyAction,
    BureaucratState,
    DEPARTMENTS,
    DepartmentName,
    PolicyConfig,
    TransferPlanRecord,
    TransferRequestRecord,
)
from .shock_tape import ShockTape

FIELD_FIT: dict[DepartmentName, set[str]] = {
    "policy_planning": {"economics", "public_management", "social_policy"},
    "budget_coordination": {"economics", "public_management", "law"},
    "regulatory_affairs": {"law", "economics", "social_policy"},
    "digital_transformation": {"data", "engineering", "public_management"},
    "public_service_operations": {"social_policy", "public_management", "data"},
}


def _field_fit(
    profile: AgentProfile, target: DepartmentName, preferred_fields: list[str] | None
) -> float:
    fit = 1.0 if profile.field in FIELD_FIT[target] else 0.35
    if preferred_fields and profile.field in preferred_fields:
        fit = min(1.0, fit + 0.20)
    return fit


def _candidate_score(
    *,
    month: int,
    slot_id: int,
    target: DepartmentName,
    state: BureaucratState,
    profile: AgentProfile,
    action: BureaucratMonthlyAction | None,
    request: TransferRequestRecord | None,
    pressures: dict[DepartmentName, float],
    policy: PolicyConfig,
    shock_tape: ShockTape,
) -> tuple[float, float, float, float, float, float, bool]:
    source = state.department
    pressure_gain = pressures[target] - pressures[source]
    pref = request.preference if request is not None else (
        action.transfer_preference
        if action is not None and action.career_action == "request_transfer"
        else None
    )
    voluntary = pref is not None
    preference_fit = 0.0
    preferred_fields: list[str] | None = None
    if pref is not None:
        preferred_fields = list(pref.preferred_fields)
        preference_fit = 1.0 if target == pref.preferred_department else 0.60
    field_fit = _field_fit(profile, target, preferred_fields)
    family_cost = profile.family_constraint * (0.35 if voluntary else 1.00)
    aversion_cost = profile.transfer_aversion * (0.25 if voluntary else 1.00)
    tie = shock_tape.transfer_draw(month, state.person_id, f"plan:{target}") * 1e-4
    source_pressure_cost = max(0.0, pressures[source] - median(pressures.values()))
    transition_cost = 0.12 * profile.baseline_capacity_units
    if voluntary:
        score = (
            0.35
            + 3.50 * policy.preference_matching * preference_fit
            + 1.50 * field_fit
            + 0.90 * max(0.0, pressure_gain)
            - 1.10 * source_pressure_cost
            - 0.45 * family_cost
            - 0.35 * aversion_cost
            - transition_cost
            + tie
        )
    else:
        score = (
            0.10
            + 2.40 * max(0.0, pressure_gain)
            + 1.00 * field_fit
            - 0.70 * source_pressure_cost
            - 2.00 * family_cost
            - 2.40 * aversion_cost
            - transition_cost
            + tie
        )
    return score, pressure_gain, field_fit, preference_fit, family_cost, aversion_cost, voluntary


def build_transfer_plan(
    *,
    month: int,
    states: dict[int, BureaucratState],
    profiles: dict[int, AgentProfile],
    actions: dict[int, BureaucratMonthlyAction],
    transfer_requests: dict[int, TransferRequestRecord] | None = None,
    pressures: dict[DepartmentName, float],
    policy: PolicyConfig,
    shock_tape: ShockTape,
) -> list[TransferPlanRecord]:
    active_ids = sorted(slot_id for slot_id, state in states.items() if state.active)
    transfer_requests = transfer_requests or {}
    max_transfers = int(math.floor(len(active_ids) * policy.transfer_capacity_rate))
    if max_transfers <= 0 or len(active_ids) < 2:
        return []

    headcounts = Counter(states[slot_id].department for slot_id in active_ids)
    pressure_median = median(pressures.values())
    preference_demand: Counter[DepartmentName] = Counter()
    for slot_id in active_ids:
        action = actions.get(slot_id)
        request = transfer_requests.get(slot_id)
        pref = request.preference if request is not None else (action.transfer_preference if action and action.career_action == "request_transfer" else None)
        if pref is not None:
            for target in pref.acceptable_departments:
                if target != states[slot_id].department:
                    preference_demand[target] += 1

    donor_caps: dict[DepartmentName, int] = {}
    target_caps: dict[DepartmentName, int] = {}
    for department in DEPARTMENTS:
        count = headcounts[department]
        source_pressure = pressures[department]
        extra_protection = min(0.12, 0.06 * max(0.0, source_pressure - pressure_median))
        protected_floor = max(2, int(math.ceil(count * (0.85 + extra_protection))))
        donor_caps[department] = max(0, count - protected_floor)
        absolute_need = max(0.0, pressures[department] - 1.0)
        relative_need = max(0.0, pressures[department] - pressure_median)
        workload_need = int(
            math.ceil((0.70 * absolute_need + 0.30 * relative_need) * max(1, count) * 0.45)
        )
        preference_need = min(preference_demand[department], max(1, int(math.ceil(count * 0.05))))
        target_caps[department] = min(max_transfers, max(workload_need, preference_need))

    graph = nx.DiGraph()
    source = "source"
    quota = "quota"
    sink = "sink"
    graph.add_edge(source, quota, capacity=max_transfers, weight=0)
    # max_flow_min_cost otherwise maximizes the number of transfers before it
    # minimizes cost. The direct zero-cost path represents "do not transfer".
    # Beneficial transfer paths have negative cost and replace no-op flow only
    # when they improve the objective.
    graph.add_edge(quota, sink, capacity=max_transfers, weight=0)
    edge_meta: dict[tuple[int, DepartmentName], tuple[float, float, float, float, float, float, bool]] = {}

    for department in DEPARTMENTS:
        donor_node = f"donor::{department}"
        target_node = f"target::{department}"
        if donor_caps[department] > 0:
            graph.add_edge(quota, donor_node, capacity=donor_caps[department], weight=0)
        if target_caps[department] > 0:
            graph.add_edge(target_node, sink, capacity=target_caps[department], weight=0)

    for slot_id in active_ids:
        state = states[slot_id]
        profile = profiles[slot_id]
        action = actions.get(slot_id)
        request = transfer_requests.get(slot_id)
        pref = request.preference if request is not None else (
            action.transfer_preference
            if action is not None and action.career_action == "request_transfer"
            else None
        )
        valid_voluntary = pref is not None and state.months_in_department >= 6
        involuntary_candidate = (
            not valid_voluntary
            and state.months_in_department >= 12
            and shock_tape.transfer_draw(month, state.person_id, "involuntary_eligibility") <= policy.involuntary_transfer_share
        )
        candidate_targets: list[tuple[DepartmentName, bool]] = []
        if valid_voluntary and pref is not None:
            candidate_targets.extend(
                (target, True)
                for target in pref.acceptable_departments
                if target != state.department
            )
        if involuntary_candidate:
            candidate_targets.extend(
                (target, False) for target in DEPARTMENTS if target != state.department
            )
        if not candidate_targets:
            continue

        donor_node = f"donor::{state.department}"
        if not graph.has_node(donor_node):
            continue
        agent_node = f"agent::{slot_id}"
        added_edge = False
        for target, forced_voluntary in candidate_targets:
            target_node = f"target::{target}"
            if not graph.has_node(target_node):
                continue
            meta = _candidate_score(
                month=month,
                slot_id=slot_id,
                target=target,
                state=state,
                profile=profile,
                action=action,
                request=(request if forced_voluntary else None),
                pressures=pressures,
                policy=policy,
                shock_tape=shock_tape,
            )
            score, pressure_gain, *_rest, candidate_voluntary = meta
            # Involuntary movement is a workload-balancing instrument. Never
            # move somebody into an equally or less pressured department.
            if not candidate_voluntary and pressure_gain <= 0.0:
                continue
            threshold = 1.0 if candidate_voluntary else 0.75
            if score < threshold:
                continue
            graph.add_edge(agent_node, target_node, capacity=1, weight=-int(round(score * 10_000)))
            edge_meta[(slot_id, target)] = meta
            added_edge = True
        if added_edge:
            graph.add_edge(donor_node, agent_node, capacity=1, weight=0)

    if not graph.has_node(sink) or graph.out_degree(source) == 0:
        return []
    try:
        flow = nx.max_flow_min_cost(graph, source, sink, capacity="capacity", weight="weight")
    except (nx.NetworkXError, nx.NetworkXUnfeasible) as exc:
        raise RuntimeError("transfer planning solver failed") from exc

    plan: list[TransferPlanRecord] = []
    for (slot_id, target), meta in edge_meta.items():
        agent_node = f"agent::{slot_id}"
        target_node = f"target::{target}"
        if flow.get(agent_node, {}).get(target_node, 0) != 1:
            continue
        score, pressure_gain, field_fit, preference_fit, family_cost, aversion_cost, voluntary = meta
        state = states[slot_id]
        selected_request = transfer_requests.get(slot_id)
        plan.append(
            TransferPlanRecord(
                month=month,
                slot_id=slot_id,
                person_id=state.person_id,
                from_department=state.department,
                to_department=target,
                voluntary=voluntary,
                request_id=(selected_request.request_id if selected_request is not None else None),
                total_score=score,
                pressure_gain=pressure_gain,
                field_fit=field_fit,
                preference_fit=preference_fit,
                family_cost=family_cost,
                transfer_aversion_cost=aversion_cost,
            )
        )
    return sorted(plan, key=lambda item: (-item.total_score, item.slot_id, item.to_department))
