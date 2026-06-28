"""Dynamic Shachi environment for Japan Policy Scientist current public release.

Employees interact indirectly through shared organizational state. Management
requests pass through an environment-mediated finite-allocation gate. The
default gate is deterministic and auditable; an LLM manager remains an optional
exploratory ablation. This module does not claim Shachi Level III, because it
does not model rich peer communication, negotiation, or coalition formation.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any

import pydantic

from shachi import Environment

from .dynamics import (
    TransitionParameters,
    baseline_voluntary_attrition_probability,
    compute_capacity_multiplier,
    draw_competing_exit,
    effective_time_shares,
    external_offer_probability,
    logistic,
    modeled_exit_pressure_probability,
    quality_probability,
    smooth_positive,
    retirement_probability,
    update_modernization,
    update_skill,
    update_work_strain_pressure,
)
from .population import (
    department_quotas,
    generate_initial_profiles,
    generate_replacement_profile,
    normalize_scenario,
)
from .schemas import (
    AgentProfile,
    BureaucracyMessage,
    BureaucracyObservation,
    BureaucratMonthlyAction,
    BureaucratQuarterlyReflection,
    BureaucratState,
    DEPARTMENTS,
    DepartmentName,
    DepartmentState,
    ExitReason,
    ExposureEvent,
    FieldName,
    ManagementRequest,
    ManagerDecision,
    ManagerMode,
    ManagerObservation,
    ManagerRequestDecision,
    MonthlyAgentRecord,
    PolicyConfig,
    PolicyLabResult,
    RealizedManagementOutcome,
    StaffingEvent,
    SyntheticScenarioName,
    TransferPlanRecord,
    TransferRequestRecord,
    WorkEvent,
)
from .shock_tape import ShockTape
from .task_queue import TaskQueue
from .transfer_planner import build_transfer_plan




def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _unit_interval_input(value: float, *, name: str, tolerance: float = 1e-9) -> float:
    value = _finite(value, name=name)
    if value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    if abs(value) <= tolerance:
        return 0.0
    if abs(value - 1.0) <= tolerance:
        return 1.0
    return value


def _positive_fraction(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value

DEPARTMENT_DEMAND_PER_AUTHORIZED_SLOT: dict[DepartmentName, float] = {
    # Exogenous public obligations per authorized slot. These values scale with
    # the declared organization size, not with realized worker capacity, rank,
    # skill, vacancies, or policy outcomes. A staffing shortage therefore does
    # not make the public workload disappear.
    "policy_planning": 0.90,
    "budget_coordination": 0.95,
    "regulatory_affairs": 0.86,
    "digital_transformation": 0.92,
    "public_service_operations": 1.00,
}


class JapanPolicyLabEnv(Environment[PolicyLabResult]):
    """Fixed employee slots with dynamic identities, tasks, and sparse managers."""

    def __init__(
        self,
        *,
        num_agents: int,
        months: int,
        warmup_months: int,
        intervention_start_month: int,
        seed: int,
        baseline_policy: PolicyConfig,
        intervention_policy: PolicyConfig,
        transition_parameters: TransitionParameters | None = None,
        hiring_interval_months: int = 6,
        transfer_interval_months: int = 6,
        enable_feedback: bool = True,
        scenario: SyntheticScenarioName = "reference_stressed",
        manager_mode: ManagerMode = "deterministic_priority",
    ):
        if intervention_start_month != warmup_months + 1:
            raise ValueError("intervention_start_month must equal warmup_months + 1")
        if num_agents < len(DEPARTMENTS):
            raise ValueError("num_agents must be at least the number of departments")
        self._num_agents = num_agents
        self.months = months
        self.warmup_months = warmup_months
        self.intervention_start_month = intervention_start_month
        self.seed = seed
        self.scenario = normalize_scenario(str(scenario))
        if manager_mode not in {"deterministic_priority", "llm"}:
            raise ValueError(f"unsupported manager_mode={manager_mode!r}")
        self.manager_mode: ManagerMode = manager_mode
        self.baseline_policy = baseline_policy
        self.intervention_policy = intervention_policy
        self.params = transition_parameters or TransitionParameters()
        self.hiring_interval_months = hiring_interval_months
        self.transfer_interval_months = transfer_interval_months
        self.enable_feedback = enable_feedback
        self.enable_quarterly_reflections = os.environ.get("POLICYLAB_ENABLE_QUARTERLY_REFLECTIONS", "0") == "1"
        self.shock_tape = ShockTape(seed)
        self.task_queue = TaskQueue(seed=seed, horizon_months=months)

        self.initial_profiles = generate_initial_profiles(
            num_agents=num_agents, seed=seed, scenario=self.scenario
        )
        self._initial_capacity_by_department: dict[DepartmentName, float] = {
            department: sum(
                profile.baseline_capacity_units
                for profile in self.initial_profiles
                if profile.department == department
            )
            for department in DEPARTMENTS
        }
        self._authorized_slots_by_department = department_quotas(
            num_agents, self.scenario
        )
        self._base_monthly_demand: dict[DepartmentName, float] = {
            department: (
                self._authorized_slots_by_department[department]
                * DEPARTMENT_DEMAND_PER_AUTHORIZED_SLOT[department]
            )
            for department in DEPARTMENTS
        }
        self.manager_ids: dict[DepartmentName, int] = {
            department: num_agents + index
            for index, department in enumerate(DEPARTMENTS)
        }
        self.manager_departments = {value: key for key, value in self.manager_ids.items()}

        self.month = 1
        self.phase = "monthly_action"
        self.profiles: dict[int, AgentProfile] = {}
        self.profile_history: dict[str, AgentProfile] = {}
        self.states: dict[int, BureaucratState] = {}
        self.departments: dict[DepartmentName, DepartmentState] = {}
        self.current_events: dict[int, list[WorkEvent]] = {}
        self.pending_events: dict[tuple[int, str, int], list[WorkEvent]] = defaultdict(list)
        self.fact_history: dict[tuple[int, str, int], list[WorkEvent]] = defaultdict(list)
        self.monthly_records: list[MonthlyAgentRecord] = []
        self.department_rows: list[dict[str, float | int | str | None]] = []
        self.staffing_events: list[StaffingEvent] = []
        self.transfer_plans: list[TransferPlanRecord] = []
        self.transfer_requests: list[TransferRequestRecord] = []
        self.management_outcomes: list[RealizedManagementOutcome] = []
        self.exposure_events: list[ExposureEvent] = []
        self.reflections: list[dict[str, object]] = []
        self.last_actions: dict[int, BureaucratMonthlyAction] = {}
        self._last_assigned_ratio: dict[int, float] = {}
        self._last_actual_units: dict[int, float] = {}
        self._headcount_start: dict[DepartmentName, int] = {}
        self._current_external_support_capacity: dict[DepartmentName, float] = {}
        self._temporary_support_next: dict[DepartmentName, float] = {}
        self._recipient_coordination_cost_next: dict[DepartmentName, float] = {}
        self._current_recipient_coordination_cost: dict[DepartmentName, float] = {}
        self._scheduled_triage_next: dict[DepartmentName, float] = {}
        self._realized_learning_time: dict[int, float] = {}
        self._realized_digital_support: dict[int, bool] = {}
        self._current_management_requests: dict[str, ManagementRequest] = {}
        self._manager_dockets: dict[DepartmentName, list[ManagementRequest]] = {}

    def num_agents(self) -> int:
        # Employee count is the scientific population size. Five managers are
        # non-capacity actors and are exposed separately via ``manager_ids``.
        return self._num_agents

    def get_default_agent_configs(self) -> list[dict] | None:
        return [
            {
                "slot_id": profile.slot_id,
                "person_id": profile.person_id,
                "identity_epoch": profile.identity_epoch,
            }
            for profile in self.initial_profiles
        ]

    def done(self) -> bool:
        return self.month > self.months

    def current_policy(self) -> PolicyConfig:
        return (
            self.baseline_policy
            if self.month < self.intervention_start_month
            else self.intervention_policy
        )

    def _effective_staffing_buffer(self) -> float:
        if self.month < self.intervention_start_month + self.params.staffing_implementation_lag_months:
            return self.baseline_policy.staffing_buffer
        return self.current_policy().staffing_buffer

    @staticmethod
    def _identity_key(slot_id: int, person_id: str, identity_epoch: int) -> tuple[int, str, int]:
        return slot_id, person_id, identity_epoch

    def _state_identity_key(self, slot_id: int) -> tuple[int, str, int]:
        state = self.states[slot_id]
        return self._identity_key(slot_id, state.person_id, state.identity_epoch)

    def _queue_event(self, slot_id: int, event: WorkEvent) -> None:
        state = self.states.get(slot_id)
        if state is None:
            return
        self.pending_events[self._state_identity_key(slot_id)].append(event)

    def _active_ids(self) -> list[int]:
        return sorted(slot_id for slot_id, state in self.states.items() if state.active)

    def _active_ids_in_department(self, department: DepartmentName) -> list[int]:
        return [
            slot_id
            for slot_id in self._active_ids()
            if self.states[slot_id].department == department
        ]

    async def reset(self) -> dict[int, BureaucracyObservation]:
        self.month = 1
        self.phase = "monthly_action"
        self.profiles = {
            profile.slot_id: profile.model_copy(deep=True) for profile in self.initial_profiles
        }
        self.profile_history = {
            profile.person_id: profile.model_copy(deep=True) for profile in self.initial_profiles
        }
        self.states = {
            profile.slot_id: BureaucratState(
                slot_id=profile.slot_id,
                person_id=profile.person_id,
                identity_epoch=profile.identity_epoch,
                active=True,
                initial_cohort=True,
                department=profile.department,
                work_strain_pressure=(
                    0.24 + 0.18 * profile.family_constraint + 0.05 * profile.external_market_pull
                ),
                skill_stock=profile.initial_skill_stock,
                modernization_stock=0.08,
                months_in_department=12,
                months_since_hire=12,
            )
            for profile in self.initial_profiles
        }
        self.departments = {}
        for department in DEPARTMENTS:
            active = [
                state for state in self.states.values() if state.department == department
            ]
            mean_skill = sum(state.skill_stock for state in active) / max(1, len(active))
            self.departments[department] = DepartmentState(
                department=department,
                active_headcount=len(active),
                knowledge_stock=mean_skill,
                codified_knowledge_stock=0.35 * len(active),
                team_routine_stock=0.85,
                modernization_stock=0.08,
            )
        self.task_queue.reset()
        self.current_events = {}
        self.pending_events = defaultdict(list)
        self.fact_history = defaultdict(list)
        self.monthly_records = []
        self.department_rows = []
        self.staffing_events = []
        self.transfer_plans = []
        self.transfer_requests = []
        self.management_outcomes = []
        self.exposure_events = []
        self.reflections = []
        self.last_actions = {}
        self._last_assigned_ratio = {}
        self._last_actual_units = {}
        self._headcount_start = {department: 0 for department in DEPARTMENTS}
        self._current_external_support_capacity = {
            department: 0.0 for department in DEPARTMENTS
        }
        self._temporary_support_next = {department: 0.0 for department in DEPARTMENTS}
        self._recipient_coordination_cost_next = {
            department: 0.0 for department in DEPARTMENTS
        }
        self._current_recipient_coordination_cost = {
            department: 0.0 for department in DEPARTMENTS
        }
        self._scheduled_triage_next = {department: 0.0 for department in DEPARTMENTS}
        self._realized_learning_time = {}
        self._realized_digital_support = {}
        self._current_management_requests = {}
        self._manager_dockets = {}
        return self._prepare_monthly_observations()

    def _record_exposure(self, exposure: ExposureEvent) -> None:
        self.exposure_events.append(exposure)

    def _current_individual_support_ratio(self, slot_id: int) -> float:
        state = self.states[slot_id]
        profile = self.profiles[slot_id]
        units = sum(
            event.units
            for event in self.exposure_events
            if event.effective_month == self.month
            and event.exposure_type == "individual_support"
            and event.slot_id == slot_id
            and event.person_id == state.person_id
            and event.identity_epoch == state.identity_epoch
            and event.realized
        )
        return _positive_fraction(units / max(profile.baseline_capacity_units, 1e-9), name="individual_support_ratio")

    def _current_department_exposure_intensity(
        self, department: DepartmentName, exposure_type: str
    ) -> float:
        values = [
            event.intensity
            for event in self.exposure_events
            if event.effective_month == self.month
            and event.department == department
            and event.exposure_type == exposure_type
            and event.realized
        ]
        return max(values, default=0.0)

    def _refresh_department_inputs(self) -> None:
        policy = self.current_policy()
        staffing_buffer = self._effective_staffing_buffer()
        for department in DEPARTMENTS:
            active_ids = self._active_ids_in_department(department)
            self._headcount_start[department] = len(active_ids)
            baseline_capacity = sum(
                self.profiles[slot_id].baseline_capacity_units for slot_id in active_ids
            )
            effective_capacity = sum(
                self.profiles[slot_id].baseline_capacity_units
                * compute_capacity_multiplier(self.states[slot_id], self.params)
                for slot_id in active_ids
            )
            direct_buffer = self._initial_capacity_by_department[department] * staffing_buffer
            manager_support = self._temporary_support_next[department]
            support_capacity = direct_buffer + manager_support
            if direct_buffer > 0:
                self._record_exposure(
                    ExposureEvent(
                        exposure_id=f"m{self.month:02d}-{department}-external-capacity",
                        month_decided=self.month,
                        effective_month=self.month,
                        department=department,
                        exposure_type="department_external_capacity",
                        units=direct_buffer,
                        intensity=_unit_interval_input(
                            direct_buffer
                            / max(self._initial_capacity_by_department[department], 1e-9),
                            name="department_external_capacity_intensity",
                        ),
                        description=(
                            "A department-level external capacity buffer was realized. It is not "
                            "individualized managerial support and does not directly reduce strain."
                        ),
                    )
                )
            coordination_cost = min(
                baseline_capacity + support_capacity,
                self._recipient_coordination_cost_next[department],
            )
            self._current_external_support_capacity[department] = support_capacity
            self._current_recipient_coordination_cost[department] = coordination_cost

            shifted = self.task_queue.apply_approved_triage(
                month=self.month,
                department=department,
                units=self._scheduled_triage_next[department],
            )
            raw_new_demand = self._base_monthly_demand[department] * self.shock_tape.demand_factor(
                self.month, department
            )
            created = self.task_queue.add_monthly_arrivals(
                month=self.month,
                department=department,
                total_units=raw_new_demand,
            )
            incoming = sum(cohort.units_total for cohort in created)
            state = self.departments[department]
            state.active_headcount = len(active_ids)
            state.raw_incoming_work_units = raw_new_demand
            state.incoming_work_units = incoming
            state.deferred_work_due_units = 0.0
            state.deferred_work_units = shifted
            state.cumulative_deferred_work_units += shifted
            state.outsourced_work_units = 0.0
            state.service_harm_points = 0.0
            state.critical_overdue_units = 0.0
            state.quality_error_units = 0.0
            state.rework_generated_units = 0.0
            state.terminal_liability_points = 0.0
            state.required_work_units = self.task_queue.open_units(department)
            state.recipient_coordination_cost_units = coordination_cost
            state.available_baseline_capacity = max(
                0.0, baseline_capacity + support_capacity - coordination_cost
            )
            state.available_effective_capacity = max(
                0.0, effective_capacity + support_capacity - coordination_cost
            )
            state.workload_ratio = state.required_work_units / max(
                state.available_effective_capacity, 1e-9
            )
            state.completed_work_units = 0.0
            state.backlog_units = state.required_work_units

        self._temporary_support_next = {department: 0.0 for department in DEPARTMENTS}
        self._recipient_coordination_cost_next = {
            department: 0.0 for department in DEPARTMENTS
        }
        self._scheduled_triage_next = {department: 0.0 for department in DEPARTMENTS}

    def _realize_experience_events(
        self,
        *,
        slot_id: int,
        policy: PolicyConfig,
        department_state: DepartmentState,
        severity: int,
    ) -> list[WorkEvent]:
        state = self.states[slot_id]
        person_id = state.person_id
        events = [
            WorkEvent(
                event_id=f"m{self.month:02d}-s{slot_id:04d}-workload",
                month=self.month,
                event_type="monthly_workload",
                description=(
                    "This month's open administrative obligations are "
                    f"{department_state.workload_ratio:.2f} times estimated effective capacity."
                ),
                objective_workload_ratio=department_state.workload_ratio,
                after_hours_severity=severity,
            )
        ]
        urgent = self.task_queue.urgent_units(state.department, self.month)
        if urgent > 0:
            events.append(
                WorkEvent(
                    event_id=f"m{self.month:02d}-s{slot_id:04d}-deadline",
                    month=self.month,
                    event_type="service_deadline_notice",
                    description=(
                        f"The department has {urgent:.2f} high-criticality units due now or next month."
                    ),
                    objective_workload_ratio=department_state.workload_ratio,
                )
            )
        if severity > 0:
            events.append(
                WorkEvent(
                    event_id=f"m{self.month:02d}-s{slot_id:04d}-afterhours",
                    month=self.month,
                    event_type="after_hours_shock",
                    description=(
                        f"An unplanned parliamentary, ministerial, or deadline-driven after-hours "
                        f"episode occurred at severity {severity}/4."
                    ),
                    objective_workload_ratio=department_state.workload_ratio,
                    after_hours_severity=severity,
                )
            )

        learning_granted = (
            self.shock_tape.realized_policy_draw(self.month, person_id, "protected_learning")
            < policy.learning_access_probability
        )
        realized_learning_time = policy.protected_learning_share if learning_granted else 0.0
        digital_granted = (
            self.shock_tape.realized_policy_draw(self.month, person_id, "digital_support")
            < policy.digital_support
        )
        self._realized_learning_time[slot_id] = realized_learning_time
        self._realized_digital_support[slot_id] = digital_granted
        if learning_granted:
            exposure_id = f"m{self.month:02d}-s{slot_id:04d}-protected-learning"
            self._record_exposure(
                ExposureEvent(
                    exposure_id=exposure_id,
                    month_decided=self.month,
                    effective_month=self.month,
                    department=state.department,
                    exposure_type="protected_learning",
                    slot_id=slot_id,
                    person_id=person_id,
                    identity_epoch=state.identity_epoch,
                    units=realized_learning_time * self.profiles[slot_id].baseline_capacity_units,
                    intensity=realized_learning_time,
                    description="A protected learning block was actually placed on this employee's schedule.",
                )
            )
            events.append(
                WorkEvent(
                    event_id=f"m{self.month:02d}-s{slot_id:04d}-learning-time",
                    month=self.month,
                    event_type="learning_time_outcome",
                    description=(
                        f"A protected learning block covering {100.0 * realized_learning_time:.1f}% "
                        "of normal monthly capacity was placed on your schedule."
                    ),
                    objective_workload_ratio=department_state.workload_ratio,
                )
            )
        if digital_granted:
            exposure_id = f"m{self.month:02d}-s{slot_id:04d}-digital-workflow"
            self._record_exposure(
                ExposureEvent(
                    exposure_id=exposure_id,
                    month_decided=self.month,
                    effective_month=self.month,
                    department=state.department,
                    exposure_type="digital_workflow",
                    slot_id=slot_id,
                    person_id=person_id,
                    identity_epoch=state.identity_epoch,
                    intensity=1.0,
                    description="A checked digital workflow was realized for this employee's current work.",
                )
            )
            events.append(
                WorkEvent(
                    event_id=f"m{self.month:02d}-s{slot_id:04d}-digital-support",
                    month=self.month,
                    event_type="digital_support_outcome",
                    description=(
                        "A supported digital workflow is available for your current tasks after implementation checks."
                    ),
                    objective_workload_ratio=department_state.workload_ratio,
                )
            )
        if self._current_external_support_capacity[state.department] > 0:
            events.append(
                WorkEvent(
                    event_id=f"m{self.month:02d}-s{slot_id:04d}-department-capacity",
                    month=self.month,
                    event_type="management_response",
                    description=(
                        f"The department has {self._current_external_support_capacity[state.department]:.2f} "
                        "temporary external capacity units this month. This is a department resource, "
                        "not an approval of your personal request."
                    ),
                    objective_workload_ratio=department_state.workload_ratio,
                )
            )
        if department_state.deferred_work_units > 0:
            events.append(
                WorkEvent(
                    event_id=f"m{self.month:02d}-s{slot_id:04d}-triage-ledger",
                    month=self.month,
                    event_type="work_deferral_notice",
                    description=(
                        f"An approved triage decision extended deadlines for {department_state.deferred_work_units:.2f} "
                        "low-criticality internal-reform units; no obligation was deleted."
                    ),
                    objective_workload_ratio=department_state.workload_ratio,
                )
            )
        key = self._state_identity_key(slot_id)
        events.extend(self.pending_events.pop(key, []))
        return events

    def _prepare_monthly_observations(self) -> dict[int, BureaucracyObservation]:
        if self.done():
            return {}
        self.phase = "monthly_action"
        self._refresh_department_inputs()
        policy = self.current_policy()
        observations: dict[int, BureaucracyObservation] = {}
        self.current_events = {}
        self._realized_learning_time = {}
        self._realized_digital_support = {}
        for slot_id in self._active_ids():
            profile = self.profiles[slot_id]
            state = self.states[slot_id]
            department_state = self.departments[state.department]
            severity = self.shock_tape.after_hours_severity(self.month, state.department)
            events = self._realize_experience_events(
                slot_id=slot_id,
                policy=policy,
                department_state=department_state,
                severity=severity,
            )
            self.current_events[slot_id] = events
            key = self._state_identity_key(slot_id)
            self.fact_history[key].extend(events)
            self.fact_history[key] = [
                event for event in self.fact_history[key] if event.month >= self.month - 2
            ][-30:]
            objective_summary = (
                f"last_relative_effort={state.last_relative_effort_pct}%; "
                f"last_personal_completion_ratio={state.last_completion_ratio:.2f}; "
                f"months_in_department={state.months_in_department}; "
                f"months_since_hire={state.months_since_hire}; "
                f"transfer_adjustment_months_remaining={state.transfer_disruption_months}; "
                f"specialist_track_months_remaining={state.specialist_track_months_remaining}."
            )
            observations[slot_id] = BureaucracyObservation(
                agent_id=slot_id,
                messages=[
                    BureaucracyMessage(
                        time=self.month,
                        src_agent_id=None,
                        dst_agent_id=slot_id,
                        content="Monthly workplace observation",
                    )
                ],
                reward=None,
                response_type=BureaucratMonthlyAction,
                tools=[],
                phase="monthly_action",
                month=self.month,
                identity_epoch=state.identity_epoch,
                profile_summary=profile.prompt_summary(),
                department_summary=department_state.prompt_summary(),
                personal_objective_summary=objective_summary,
                recent_events=events,
            )
        return observations

    def _prepare_quarterly_observations(self) -> dict[int, BureaucracyObservation]:
        self.phase = "quarterly_reflection"
        observations: dict[int, BureaucracyObservation] = {}
        for slot_id in self._active_ids():
            state = self.states[slot_id]
            profile = self.profiles[slot_id]
            events = self.fact_history.get(self._state_identity_key(slot_id), [])
            if not events:
                continue
            department_state = self.departments[state.department]
            observations[slot_id] = BureaucracyObservation(
                agent_id=slot_id,
                messages=[
                    BureaucracyMessage(
                        time=self.month,
                        src_agent_id=None,
                        dst_agent_id=slot_id,
                        content="Quarterly reflection request",
                    )
                ],
                reward=None,
                response_type=BureaucratQuarterlyReflection,
                tools=[],
                phase="quarterly_reflection",
                month=self.month,
                identity_epoch=state.identity_epoch,
                profile_summary=profile.prompt_summary(),
                department_summary=department_state.prompt_summary(),
                personal_objective_summary=(
                    f"last_relative_effort={state.last_relative_effort_pct}%; "
                    f"last_completion_ratio={state.last_completion_ratio:.2f}; "
                    f"months_in_department={state.months_in_department}."
                ),
                recent_events=events,
            )
        return observations

    @staticmethod
    def _parse_response(
        response: str | pydantic.BaseModel | None,
        response_type: type[pydantic.BaseModel],
    ) -> pydantic.BaseModel:
        if response is None:
            raise ValueError("agent returned None")
        if isinstance(response, pydantic.BaseModel):
            return response_type.model_validate(response.model_dump())
        return response_type.model_validate_json(response)

    def _validate_event_refs(self, slot_id: int, refs: list[str], quarterly: bool = False) -> None:
        if quarterly:
            allowed = {
                event.event_id
                for event in self.fact_history.get(self._state_identity_key(slot_id), [])
            }
        else:
            allowed = {event.event_id for event in self.current_events.get(slot_id, [])}
        if not refs or not set(refs).issubset(allowed):
            raise ValueError(
                f"slot {slot_id} used invalid event_refs={refs}; allowed={sorted(allowed)}"
            )

    def _validate_action_context(self, slot_id: int, action: BureaucratMonthlyAction) -> None:
        if action.transfer_preference is None:
            return
        current = self.states[slot_id].department
        if action.transfer_preference.preferred_department == current:
            raise ValueError("preferred transfer department must differ from current department")
        if current in action.transfer_preference.acceptable_departments:
            raise ValueError("acceptable transfer departments must exclude current department")

    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, BureaucracyObservation | ManagerObservation]:
        if self.phase == "monthly_action":
            return self._step_monthly(responses)
        if self.phase == "manager_decision":
            return self._step_manager(responses)
        return self._step_quarterly(responses)

    def _transfer_context(self, slot_id: int) -> tuple[bool, bool, bool]:
        events = self.current_events.get(slot_id, [])
        personal = [event for event in events if event.event_type == "personal_transfer"]
        if not personal:
            return False, False, False
        event = personal[-1]
        return event.involuntary, event.explanation_received, event.appeal_relief_realized

    def _queue_transfer_request(self, slot_id: int, action: BureaucratMonthlyAction) -> None:
        if action.career_action != "request_transfer" or action.transfer_preference is None:
            return
        state = self.states[slot_id]
        existing = next(
            (
                request
                for request in self.transfer_requests
                if request.status == "pending"
                and request.slot_id == slot_id
                and request.person_id == state.person_id
                and request.identity_epoch == state.identity_epoch
            ),
            None,
        )
        if existing is not None:
            existing.preference = action.transfer_preference.model_copy(deep=True)
            return
        self.transfer_requests.append(
            TransferRequestRecord(
                request_id=f"tr-m{self.month:02d}-s{slot_id:04d}-e{state.identity_epoch}",
                created_month=self.month,
                expires_after_month=self.month + self.params.transfer_request_ttl_months - 1,
                slot_id=slot_id,
                person_id=state.person_id,
                identity_epoch=state.identity_epoch,
                from_department=state.department,
                preference=action.transfer_preference.model_copy(deep=True),
            )
        )

    def _management_request_from_action(
        self, slot_id: int, action: BureaucratMonthlyAction
    ) -> ManagementRequest | None:
        kind: str | None = None
        if action.voice_action == "request_staffing_relief":
            kind = "staffing_relief"
        elif action.work_response == "request_support":
            kind = "operational_support"
        elif action.voice_action == "ask_for_explanation":
            kind = "explanation"
        elif action.voice_action == "propose_process_reform":
            kind = "process_reform"
        elif action.voice_action == "raise_operational_risk":
            kind = "operational_risk"
        elif action.career_action == "request_specialist_track":
            kind = "specialist_track"
        if kind is None:
            return None
        state = self.states[slot_id]
        action_priority = {
            "staffing_relief": 2.0,
            "operational_support": 1.7,
            "explanation": 0.8,
            "process_reform": 1.0,
            "operational_risk": 2.2,
            "specialist_track": 0.6,
        }[kind]
        # Survey firewall: management priority uses objective load/strain and
        # the observable career action, never sealed self-report values.
        career_signal = {
            "stay": 0.0,
            "request_specialist_track": 0.25,
            "request_transfer": 0.60,
            "explore_external_exit": 1.0,
        }[action.career_action]
        # current public release: keep management priority as an unbounded ranking score.
        # The docket is finite, but severe overload must not be saturated away
        # before ranking.  Sealed survey fields remain excluded.
        priority = (
            4.5 * self._last_assigned_ratio.get(slot_id, 1.0)
            + 3.5 * smooth_positive(state.work_strain_pressure)
            + 2.0 * career_signal
        )
        requested_units = (
            self.profiles[slot_id].baseline_capacity_units * 0.12
            if kind in {"staffing_relief", "operational_support", "operational_risk"}
            else 0.0
        )
        return ManagementRequest(
            request_id=f"mr-m{self.month:02d}-s{slot_id:04d}-e{state.identity_epoch}-{kind}",
            month=self.month,
            slot_id=slot_id,
            person_id=state.person_id,
            identity_epoch=state.identity_epoch,
            department=state.department,
            request_kind=kind,  # type: ignore[arg-type]
            priority_score=priority + action_priority,
            requested_units=requested_units,
            fact_summary=(
                f"workload_ratio={self._last_assigned_ratio.get(slot_id, 1.0):.2f}; "
                f"completion_ratio={state.last_completion_ratio:.2f}; "
                f"request grounded in event refs {','.join(action.event_refs[:3])}"
            ),
        )

    def _exit_cause(
        self,
        *,
        slot_id: int,
        action: BureaucratMonthlyAction,
    ) -> tuple[ExitReason, float] | None:
        if not self.enable_feedback:
            return None
        state = self.states[slot_id]
        profile = self.profiles[slot_id]
        high_p = 0.0
        if state.exit_pressure_streak >= self.params.high_exit_streak_required:
            high_p = modeled_exit_pressure_probability(
                work_strain_pressure=state.work_strain_pressure,
                external_market_pull=profile.external_market_pull,
                public_service_motivation=profile.public_service_motivation,
                career_action=action.career_action,
                params=self.params,
            )
        hazards: dict[ExitReason, float] = {
            "retirement": retirement_probability(profile),
            "external_offer": external_offer_probability(profile),
            "modeled_resignation_pressure": high_p,
            "baseline_voluntary_attrition": baseline_voluntary_attrition_probability(profile),
        }
        draw = self.shock_tape.exit_draw(self.month, state.person_id, "competing-risk")
        return draw_competing_exit(draw=draw, hazards=hazards)

    def _field_coverage(self, department: DepartmentName) -> dict[FieldName, float]:
        active_ids = self._active_ids_in_department(department)
        total = sum(self.profiles[slot].baseline_capacity_units for slot in active_ids)
        result: dict[FieldName, float] = {}
        for field in (
            "economics", "law", "public_management", "data", "engineering", "social_policy"
        ):
            coverage_pressure = (
                sum(
                    self.profiles[slot].baseline_capacity_units
                    for slot in active_ids
                    if self.profiles[slot].field == field
                )
                / max(total * 0.35, 1e-9)
            )
            # Return raw coverage pressure. TaskQueue owns the smooth link from
            # coverage pressure to quality; applying logistic here as well would
            # double-compress field mismatch and hide specialized-skill shortages.
            result[field] = coverage_pressure
        return result

    def _step_monthly(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, BureaucracyObservation | ManagerObservation]:
        active_ids = self._active_ids()
        if set(responses) != set(active_ids):
            missing = set(active_ids) - set(responses)
            unexpected = set(responses) - set(active_ids)
            raise ValueError(
                f"monthly response IDs mismatch; missing={missing}, unexpected={unexpected}"
            )
        actions: dict[int, BureaucratMonthlyAction] = {}
        for slot_id in active_ids:
            action = BureaucratMonthlyAction.model_validate(
                self._parse_response(responses[slot_id], BureaucratMonthlyAction).model_dump()
            )
            self._validate_event_refs(slot_id, action.event_refs)
            self._validate_action_context(slot_id, action)
            actions[slot_id] = action
        self.last_actions = actions
        policy = self.current_policy()

        active_capacity_snapshot: dict[DepartmentName, float] = {
            department: sum(
                self.profiles[slot_id].baseline_capacity_units
                for slot_id in active_ids
                if self.states[slot_id].department == department
            )
            for department in DEPARTMENTS
        }
        active_count_snapshot: dict[DepartmentName, int] = {
            department: sum(
                1 for slot_id in active_ids if self.states[slot_id].department == department
            )
            for department in DEPARTMENTS
        }
        outputs_by_department: dict[DepartmentName, float] = defaultdict(float)
        learning_by_department: dict[DepartmentName, float] = defaultdict(float)
        process_by_department: dict[DepartmentName, float] = defaultdict(float)
        records_by_slot: dict[int, MonthlyAgentRecord] = {}
        pending_exits: list[tuple[int, ExitReason, float]] = []
        requests: list[ManagementRequest] = []

        for slot_id in active_ids:
            profile = self.profiles[slot_id]
            state = self.states[slot_id]
            department = state.department
            department_state = self.departments[department]
            action = actions[slot_id]
            active_capacity = active_capacity_snapshot[department]
            external_support = self._current_external_support_capacity[department]
            work_for_agents = max(0.0, department_state.required_work_units - external_support)
            assigned_units = (
                work_for_agents
                * profile.baseline_capacity_units
                / max(active_capacity, 1e-9)
            )
            assigned_ratio = assigned_units / max(profile.baseline_capacity_units, 1e-9)
            self._last_assigned_ratio[slot_id] = assigned_ratio

            shares = effective_time_shares(
                work_mix=action.work_mix,
                protected_learning_share=self._realized_learning_time.get(slot_id, 0.0),
            )
            delivery_share = shares["core"] + 0.65 * shares["coordination"] + 0.10 * shares["process"]
            capacity_multiplier = compute_capacity_multiplier(state, self.params)
            response_multiplier = {
                "deliver_normally": 1.00,
                "work_overtime": 1.04,
                "prioritize_core_work": 1.04,
                "request_support": 0.95,
                "defer_low_priority_work": 0.90,
                "protect_health_capacity": 0.95,
                "take_health_leave": 0.75,
                "caregiving_leave": 0.82,
                "refuse_overtime": 0.98,
            }[action.work_response]
            department_spillover = (
                1.0
                + self.params.department_tacit_productivity_gain
                * max(0.0, self._mean_skill(department) - 1.0)
                + self.params.department_codified_productivity_gain
                * math.log1p(department_state.codified_knowledge_stock)
                / max(1.0, math.log1p(self._initial_capacity_by_department[department] * 2.0))
                + self.params.department_routine_productivity_gain
                * math.log1p(max(0.0, department_state.team_routine_stock - 0.8))
                + self.params.department_modernization_productivity_gain
                * department_state.modernization_stock
            )
            digital_intensity = 1.0 if self._realized_digital_support.get(slot_id, False) else 0.0
            reform_intensity = self._current_department_exposure_intensity(
                department, "implemented_process_reform"
            )
            digital_efficiency = 1.0 + (
                0.06
                * digital_intensity
                * (0.5 * state.modernization_stock + 0.5 * department_state.modernization_stock)
                * policy.productivity_savings_capture
            )
            potential_output = (
                profile.baseline_capacity_units
                * action.relative_effort_pct
                / 100.0
                * capacity_multiplier
                * response_multiplier
                * delivery_share
                * department_spillover
                * digital_efficiency
            )
            actual_units = max(0.0, potential_output)
            completion_ratio = _positive_fraction(actual_units / max(assigned_units, 1e-9), name="agent_completion_ratio")
            self._last_actual_units[slot_id] = actual_units
            outputs_by_department[department] += actual_units

            effort_ratio = action.relative_effort_pct / 100.0
            learning_investment = effort_ratio * shares["learning"]
            process_investment = effort_ratio * shares["process"]
            learning_by_department[department] += (
                learning_investment * profile.baseline_capacity_units
            )
            process_by_department[department] += (
                process_investment * profile.baseline_capacity_units
            )

            involuntary_transfer, explanation_received, appeal_relief = self._transfer_context(slot_id)
            individual_support_ratio = self._current_individual_support_ratio(slot_id)
            severity = max(
                (event.after_hours_severity for event in self.current_events[slot_id]),
                default=0,
            )
            strain_before = state.work_strain_pressure
            state.work_strain_pressure = update_work_strain_pressure(
                current_pressure=state.work_strain_pressure,
                assigned_work_ratio=assigned_ratio,
                relative_effort_pct=action.relative_effort_pct,
                after_hours_severity=severity,
                family_constraint=profile.family_constraint,
                individual_support_ratio=individual_support_ratio,
                forced_transfer=involuntary_transfer,
                transfer_explanation_received=explanation_received,
                appeal_relief_realized=appeal_relief,
                work_response=action.work_response,
                protected_learning_time=shares["protected_learning"],
                params=self.params,
            )
            state.skill_stock = update_skill(
                current=state.skill_stock,
                learning_investment_ratio=learning_investment,
                specialist_track_active=state.specialist_track_active,
                params=self.params,
            )
            state.modernization_stock = update_modernization(
                current=state.modernization_stock,
                process_investment_ratio=process_investment,
                digital_exposure_intensity=digital_intensity,
                implemented_reform_intensity=reform_intensity,
                params=self.params,
            )
            state.last_relative_effort_pct = action.relative_effort_pct
            state.last_completion_ratio = completion_ratio
            state.last_career_action = action.career_action
            state.months_in_department += 1
            state.months_since_hire += 1
            if state.transfer_disruption_months > 0:
                state.transfer_disruption_months -= 1
            if state.recipient_coordination_months > 0:
                state.recipient_coordination_months -= 1
            if state.specialist_track_months_remaining > 0:
                state.specialist_track_months_remaining -= 1
                if state.specialist_track_months_remaining == 0:
                    state.specialist_track_active = False

            self._queue_transfer_request(slot_id, action)
            request = self._management_request_from_action(slot_id, action)
            if request is not None:
                requests.append(request)

            # Survey firewall: resignation pressure requires an observable
            # career action plus mechanical strain; a questionnaire answer alone
            # can never create an exit transition.
            exit_pressure_signal = (
                action.career_action == "explore_external_exit"
                or (
                    action.career_action == "request_transfer"
                    and state.work_strain_pressure >= self.params.high_exit_strain_pressure_threshold
                )
            )
            if exit_pressure_signal:
                state.exit_pressure_streak += 1
            else:
                state.exit_pressure_streak = 0
            exit_cause = self._exit_cause(slot_id=slot_id, action=action)
            if exit_cause is not None:
                pending_exits.append((slot_id, exit_cause[0], exit_cause[1]))

            records_by_slot[slot_id] = MonthlyAgentRecord(
                month=self.month,
                slot_id=slot_id,
                person_id=state.person_id,
                identity_epoch=state.identity_epoch,
                initial_cohort=state.initial_cohort,
                department=department,
                action=action,
                assigned_work_ratio=assigned_ratio,
                actual_completed_units=actual_units,
                actual_completion_ratio=completion_ratio,
                work_strain_pressure_before=strain_before,
                work_strain_pressure_after=state.work_strain_pressure,
                skill_after=state.skill_stock,
                modernization_after=state.modernization_stock,
                effective_core_share=shares["core"],
                effective_coordination_share=shares["coordination"],
                effective_learning_share=shares["learning"],
                effective_process_share=shares["process"],
                protected_learning_share=shares["protected_learning"],
                individual_support_ratio=individual_support_ratio,
                exited_this_month=False,
            )

        # Service is closed on the frozen start-of-month population. Staffing
        # changes below affect future months only.
        for department in DEPARTMENTS:
            state = self.departments[department]
            support_capacity = self._current_external_support_capacity[department]
            coordination_cost = self._current_recipient_coordination_cost[department]
            agent_output = max(0.0, outputs_by_department[department] - coordination_cost)
            total_gross_output = min(
                state.required_work_units, agent_output + support_capacity
            )
            active_ids_dept = [
                slot_id
                for slot_id in active_ids
                if self.states[slot_id].department == department
            ]
            mean_strain = sum(
                self.states[slot].work_strain_pressure for slot in active_ids_dept
            ) / max(1, len(active_ids_dept))
            quality_index = quality_probability(
                mean_skill=self._mean_skill(department),
                modernization_stock=state.modernization_stock,
                team_routine_stock=state.team_routine_stock,
                mean_strain_pressure=mean_strain,
                params=self.params,
            )
            task_result = self.task_queue.allocate_and_close_month(
                month=self.month,
                department=department,
                completed_units=total_gross_output,
                quality_index=quality_index,
                field_coverage=self._field_coverage(department),
                terminal=self.month == self.months,
            )
            state.completed_work_units = task_result["completed_units"]
            state.outsourced_work_units = min(
                support_capacity, max(0.0, total_gross_output - agent_output)
            )
            state.cumulative_outsourced_work_units += state.outsourced_work_units
            state.service_harm_points = task_result["service_harm_points"]
            state.cumulative_service_harm_points += state.service_harm_points
            state.critical_overdue_units = task_result["critical_overdue_units"]
            state.quality_error_units = task_result["quality_error_units"]
            state.cumulative_quality_error_units += state.quality_error_units
            state.rework_generated_units = task_result["rework_generated_units"]
            state.cumulative_rework_units += state.rework_generated_units
            state.backlog_units = task_result["backlog_units"]
            state.terminal_liability_points = task_result[
                "terminal_liability_points"
            ]
            state.completion_ratio = _unit_interval_input(
                state.completed_work_units / max(state.required_work_units, 1e-9),
                name="department_completion_ratio",
            )
            # Unit-consistent ratio: both numerator and denominator are work
            # units. Weighted public-harm points remain a separate guardrail.
            state.effective_demand_served_ratio = _unit_interval_input(
                state.completed_work_units / max(state.required_work_units, 1e-9),
                name="effective_demand_served_ratio",
            )
            state.codified_knowledge_stock = max(
                0.0,
                state.codified_knowledge_stock
                * (1.0 - self.params.codified_knowledge_depreciation)
                + 0.012 * learning_by_department[department],
            )
            state.team_routine_stock = (
                state.team_routine_stock
                * math.exp(-self.params.team_routine_depreciation)
                + 0.005
                * process_by_department[department]
                / max(self._initial_capacity_by_department[department], 1e-9)
            )
            state.modernization_stock = max(
                0.0,
                sum(self.states[slot].modernization_stock for slot in active_ids_dept)
                / max(1, len(active_ids_dept)),
            )
            state.knowledge_stock = self._mean_skill(department)

        for slot_id, reason, probability in pending_exits:
            self._exit_person(slot_id, reason=reason, probability=probability)
            records_by_slot[slot_id].exited_this_month = True
            records_by_slot[slot_id].exit_reason = reason

        self.monthly_records.extend(records_by_slot[slot_id] for slot_id in active_ids)
        if self.enable_feedback:
            if self.month % self.transfer_interval_months == 0:
                self._perform_transfers(policy)
            if self.month % self.hiring_interval_months == 0:
                self._perform_hiring(policy)

        for department in DEPARTMENTS:
            state = self.departments[department]
            end_headcount = len(self._active_ids_in_department(department))
            state.active_headcount = end_headcount
            self.department_rows.append(
                {
                    "month": self.month,
                    "department": department,
                    "active_headcount_start": self._headcount_start[department],
                    "active_headcount_end": end_headcount,
                    "raw_incoming_work_units": state.raw_incoming_work_units,
                    "incoming_work_units": state.incoming_work_units,
                    "required_work_units": state.required_work_units,
                    "completed_work_units": state.completed_work_units,
                    "outsourced_work_units": state.outsourced_work_units,
                    "deferred_work_units": state.deferred_work_units,
                    "service_harm_points": state.service_harm_points,
                    "critical_overdue_units": state.critical_overdue_units,
                    "quality_error_units": state.quality_error_units,
                    "rework_generated_units": state.rework_generated_units,
                    "terminal_liability_points": state.terminal_liability_points,
                    "backlog_units": state.backlog_units,
                    "workload_ratio": state.workload_ratio,
                    "completion_ratio": state.completion_ratio,
                    "effective_demand_served_ratio": state.effective_demand_served_ratio,
                    "available_baseline_capacity": state.available_baseline_capacity,
                    "available_effective_capacity": state.available_effective_capacity,
                    "recipient_coordination_cost_units": state.recipient_coordination_cost_units,
                    "mean_tacit_skill": self._mean_skill(department),
                    "codified_knowledge_stock": state.codified_knowledge_stock,
                    "team_routine_stock": state.team_routine_stock,
                    "modernization_stock": state.modernization_stock,
                }
            )

        self._current_management_requests = {request.request_id: request for request in requests}
        if self.enable_feedback:
            manager_observations = self._prepare_manager_observations(requests)
            if self.manager_mode == "llm":
                return manager_observations
            deterministic = self._deterministic_manager_responses(manager_observations)
            return self._step_manager(deterministic)
        return self._advance_after_manager()

    def _mean_skill(self, department: DepartmentName) -> float:
        active = self._active_ids_in_department(department)
        return sum(self.states[slot].skill_stock for slot in active) / max(1, len(active))

    def _prepare_manager_observations(
        self, requests: list[ManagementRequest]
    ) -> dict[int, ManagerObservation]:
        self.phase = "manager_decision"
        policy = self.current_policy()
        by_department: dict[DepartmentName, list[ManagementRequest]] = defaultdict(list)
        for request in requests:
            by_department[request.department].append(request)
        observations: dict[int, ManagerObservation] = {}
        self._manager_dockets = {}
        for department in DEPARTMENTS:
            # Deterministic representative docket: highest priority per kind,
            # then remaining global priorities, capped by policy.management_case_capacity.
            candidates = sorted(
                by_department[department],
                key=lambda request: (-request.priority_score, request.request_kind, request.request_id),
            )
            selected: list[ManagementRequest] = []
            seen_kinds: set[str] = set()
            docket_capacity = int(policy.management_case_capacity)
            for request in candidates:
                if request.request_kind not in seen_kinds and len(selected) < docket_capacity:
                    selected.append(request)
                    seen_kinds.add(request.request_kind)
            for request in candidates:
                if request not in selected and len(selected) < docket_capacity:
                    selected.append(request)
            self._manager_dockets[department] = selected
            support_envelope = (
                self._initial_capacity_by_department[department]
                * 0.06
                * policy.manager_support
            )
            triage_envelope = (
                self._base_monthly_demand[department]
                * 0.10
                * policy.workload_triage_support
            )
            reform_slots = min(3, int(math.floor(3.0 * policy.process_reform_support + 0.25)))
            explanation_slots = min(3, int(math.floor(3.0 * policy.explanation_quality + 0.25)))
            specialist_slots = min(
                3,
                int(
                    math.floor(
                        max(1, len(self._active_ids_in_department(department)))
                        * 0.04
                        * policy.specialist_track_access
                        + 0.25
                    )
                ),
            )
            manager_id = self.manager_ids[department]
            observations[manager_id] = ManagerObservation(
                agent_id=manager_id,
                messages=[
                    BureaucracyMessage(
                        time=self.month,
                        src_agent_id=None,
                        dst_agent_id=manager_id,
                        content="Privacy-minimized monthly management docket",
                    )
                ],
                reward=None,
                response_type=ManagerDecision,
                tools=[],
                month=self.month,
                identity_epoch=0,
                department=department,
                department_fact_summary=(
                    f"active_headcount={self.departments[department].active_headcount}; "
                    f"workload_ratio={self.departments[department].workload_ratio:.2f}; "
                    f"critical_overdue_units={self.departments[department].critical_overdue_units:.2f}; "
                    f"service_harm_points={self.departments[department].service_harm_points:.2f}"
                ),
                request_docket=selected,
                support_envelope_units=support_envelope,
                triage_envelope_units=triage_envelope,
                reform_slots=reform_slots,
                explanation_slots=explanation_slots,
                specialist_slots=specialist_slots,
            )
        return observations

    def _deterministic_manager_responses(
        self,
        observations: dict[int, ManagerObservation],
    ) -> dict[int, ManagerDecision]:
        """Allocate finite management resources by a published priority rule.

        The docket is already deterministically ordered by priority and request
        type.  This baseline approves each request only while the corresponding
        finite envelope remains. It makes the management gate auditable and
        keeps an optional LLM-manager ablation from being confused with a
        necessary part of the core PoC.
        """

        responses: dict[int, ManagerDecision] = {}
        for manager_id in sorted(observations):
            observation = observations[manager_id]
            support_remaining = observation.support_envelope_units
            triage_remaining = observation.triage_envelope_units
            reform_remaining = observation.reform_slots
            explanation_remaining = observation.explanation_slots
            specialist_remaining = observation.specialist_slots
            decisions: list[ManagerRequestDecision] = []
            for request in observation.request_docket:
                committed = 0.0
                decision = "defer"
                if request.request_kind in {"operational_support", "staffing_relief"}:
                    committed = min(request.requested_units, support_remaining)
                    support_remaining -= committed
                    if committed > 0:
                        decision = (
                            "approve"
                            if committed + 1e-12 >= request.requested_units
                            else "partially_approve"
                        )
                elif request.request_kind == "operational_risk":
                    committed = min(request.requested_units, triage_remaining)
                    triage_remaining -= committed
                    if committed > 0:
                        decision = (
                            "approve"
                            if committed + 1e-12 >= request.requested_units
                            else "partially_approve"
                        )
                elif request.request_kind == "process_reform" and reform_remaining > 0:
                    reform_remaining -= 1
                    committed = 1.0
                    decision = "approve"
                elif request.request_kind == "explanation" and explanation_remaining > 0:
                    explanation_remaining -= 1
                    committed = 1.0
                    decision = "approve"
                elif request.request_kind == "specialist_track" and specialist_remaining > 0:
                    specialist_remaining -= 1
                    committed = 1.0
                    decision = "approve"
                public_message = (
                    "The request was approved within the published finite allocation rule."
                    if decision == "approve"
                    else "The request was partially approved within the remaining published envelope."
                    if decision == "partially_approve"
                    else "The request is deferred because the published monthly envelope is exhausted."
                )
                decisions.append(
                    ManagerRequestDecision(
                        request_id=request.request_id,
                        decision=decision,  # type: ignore[arg-type]
                        committed_units=committed,
                        public_message=public_message,
                    )
                )
            responses[manager_id] = ManagerDecision(
                department_message=(
                    "Requests were processed by the deterministic priority-and-envelope rule; "
                    "no discretionary language-model judgment was used."
                ),
                decisions=decisions,
                confidence=1.0,
            )
        return responses

    def _request_identity_is_current(self, request: ManagementRequest) -> bool:
        """Require complete identity continuity at the moment of allocation."""
        state = self.states.get(request.slot_id)
        return bool(
            state is not None
            and state.active
            and state.slot_id == request.slot_id
            and state.person_id == request.person_id
            and state.identity_epoch == request.identity_epoch
            and state.department == request.department
        )

    def _record_management_outcome(
        self,
        *,
        request: ManagementRequest,
        decision_status: str,
        committed_units: float,
        public_message: str,
        effective: bool,
    ) -> None:
        approved = effective and decision_status in {"approved", "partially_approved"}
        event_id = f"m{self.month + 1:02d}-s{request.slot_id:04d}-management-{request.request_kind}"
        description = (
            f"Management {decision_status.replace('_', ' ')} the {request.request_kind} request. "
            f"Committed units={committed_units:.2f}. Message: {public_message}"
        )
        outcome = RealizedManagementOutcome(
            month=self.month,
            effective_month=self.month + 1,
            slot_id=request.slot_id,
            person_id=request.person_id,
            identity_epoch=request.identity_epoch,
            department=request.department,
            request_id=request.request_id,
            request_kind=request.request_kind,
            approved=approved,
            decision_status=decision_status,  # type: ignore[arg-type]
            response_delay_days=(
                0
                if decision_status == "identity_invalidated"
                else 5
                if approved
                else 15
                if decision_status == "deferred"
                else 20
            ),
            approved_support_units=committed_units if request.request_kind in {"operational_support", "staffing_relief"} else 0.0,
            event_id=event_id,
            public_message=public_message,
            description=description,
        )
        self.management_outcomes.append(outcome)
        if self._request_identity_is_current(request):
            self._queue_event(
                request.slot_id,
                WorkEvent(
                    event_id=event_id,
                    month=self.month + 1,
                    event_type=(
                        "support_request_outcome"
                        if request.request_kind in {"operational_support", "staffing_relief"}
                        else "specialist_track_outcome"
                        if request.request_kind == "specialist_track"
                        else "management_response"
                    ),
                    description=description,
                    decision_status=decision_status,  # type: ignore[arg-type]
                    response_delay_days=outcome.response_delay_days,
                    realized_support_units=outcome.approved_support_units,
                    source_request_id=request.request_id,
                ),
            )

    def _step_manager(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, BureaucracyObservation | ManagerObservation]:
        expected = set(self.manager_departments)
        if set(responses) != expected:
            raise ValueError(
                f"manager response IDs mismatch; missing={expected-set(responses)}, unexpected={set(responses)-expected}"
            )
        policy = self.current_policy()
        docket_request_ids = {
            request.request_id
            for requests in self._manager_dockets.values()
            for request in requests
        }
        processed: set[str] = set()
        for manager_id in sorted(expected):
            department = self.manager_departments[manager_id]
            observation_docket = self._manager_dockets[department]
            decision = ManagerDecision.model_validate(
                self._parse_response(responses[manager_id], ManagerDecision).model_dump()
            )
            expected_ids = {request.request_id for request in observation_docket}
            actual_ids = {item.request_id for item in decision.decisions}
            if actual_ids != expected_ids:
                raise ValueError(
                    f"manager {manager_id} decisions must exactly match docket IDs; expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
                )
            support_remaining = (
                self._initial_capacity_by_department[department]
                * 0.06
                * policy.manager_support
            )
            triage_remaining = (
                self._base_monthly_demand[department]
                * 0.10
                * policy.workload_triage_support
            )
            reform_remaining = min(3, int(math.floor(3.0 * policy.process_reform_support + 0.25)))
            explanation_remaining = min(3, int(math.floor(3.0 * policy.explanation_quality + 0.25)))
            specialist_remaining = min(
                3,
                int(
                    math.floor(
                        max(1, len(self._active_ids_in_department(department)))
                        * 0.04
                        * policy.specialist_track_access
                        + 0.25
                    )
                ),
            )
            for item in decision.decisions:
                request = self._current_management_requests[item.request_id]
                processed.add(request.request_id)
                if not self._request_identity_is_current(request):
                    self._record_management_outcome(
                        request=request,
                        decision_status="identity_invalidated",
                        committed_units=0.0,
                        public_message=(
                            "The request was invalidated because the requesting identity is no "
                            "longer active in the recorded department."
                        ),
                        effective=False,
                    )
                    continue
                requested_approval = item.decision in {"approve", "partially_approve"}
                effective = False
                committed = 0.0
                status = {
                    "approve": "approved",
                    "partially_approve": "partially_approved",
                    "reject": "rejected",
                    "defer": "deferred",
                }[item.decision]
                if requested_approval and request.request_kind in {"operational_support", "staffing_relief"}:
                    committed = min(item.committed_units, request.requested_units, support_remaining)
                    effective = committed > 0
                    support_remaining -= committed
                    if effective:
                        self._temporary_support_next[department] += committed
                        self._record_exposure(
                            ExposureEvent(
                                exposure_id=f"x-{request.request_id}-support",
                                month_decided=self.month,
                                effective_month=self.month + 1,
                                department=department,
                                exposure_type="individual_support",
                                source_request_id=request.request_id,
                                slot_id=request.slot_id,
                                person_id=request.person_id,
                                identity_epoch=request.identity_epoch,
                                units=committed,
                                intensity=_unit_interval_input(
                                    committed
                                    / max(self.profiles[request.slot_id].baseline_capacity_units, 1e-9),
                                    name="individual_support_intensity",
                                ),
                                description="Named individual support approved within a finite manager envelope.",
                            )
                        )
                elif requested_approval and request.request_kind == "operational_risk":
                    committed = min(item.committed_units, request.requested_units, triage_remaining)
                    effective = committed > 0
                    triage_remaining -= committed
                    if effective:
                        self._scheduled_triage_next[department] += committed
                        self._record_exposure(
                            ExposureEvent(
                                exposure_id=f"x-{request.request_id}-triage",
                                month_decided=self.month,
                                effective_month=self.month + 1,
                                department=department,
                                exposure_type="approved_triage",
                                source_request_id=request.request_id,
                                units=committed,
                                intensity=_unit_interval_input(
                                    committed / max(self._base_monthly_demand[department], 1e-9),
                                    name="triage_intensity",
                                ),
                                description="Low-criticality deadline triage approved within a finite envelope.",
                            )
                        )
                elif requested_approval and request.request_kind == "process_reform":
                    effective = reform_remaining > 0
                    if effective:
                        reform_remaining -= 1
                        committed = 1.0
                        self._record_exposure(
                            ExposureEvent(
                                exposure_id=f"x-{request.request_id}-reform",
                                month_decided=self.month,
                                effective_month=self.month + 1,
                                department=department,
                                exposure_type="implemented_process_reform",
                                source_request_id=request.request_id,
                                intensity=1.0,
                                description="A manager-approved reform occupied a finite implementation slot.",
                            )
                        )
                        for colleague_id in self._active_ids_in_department(department):
                            self._queue_event(
                                colleague_id,
                                WorkEvent(
                                    event_id=f"m{self.month+1:02d}-s{colleague_id:04d}-implemented-reform-{request.slot_id}",
                                    month=self.month + 1,
                                    event_type="organizational_learning",
                                    description="A manager-approved process reform entered the department workflow.",
                                    decision_status="approved",
                                    source_request_id=request.request_id,
                                ),
                            )
                elif requested_approval and request.request_kind == "explanation":
                    effective = explanation_remaining > 0
                    if effective:
                        explanation_remaining -= 1
                        committed = 1.0
                elif requested_approval and request.request_kind == "specialist_track":
                    effective = specialist_remaining > 0
                    state = self.states.get(request.slot_id)
                    if effective and state is not None and not state.specialist_track_active:
                        specialist_remaining -= 1
                        committed = 1.0
                        state.specialist_track_active = True
                        state.specialist_track_months_remaining = self.params.specialist_track_duration_months
                        self._record_exposure(
                            ExposureEvent(
                                exposure_id=f"x-{request.request_id}-specialist",
                                month_decided=self.month,
                                effective_month=self.month + 1,
                                department=department,
                                exposure_type="specialist_track",
                                source_request_id=request.request_id,
                                slot_id=request.slot_id,
                                person_id=request.person_id,
                                identity_epoch=request.identity_epoch,
                                intensity=1.0,
                                description="A finite-duration specialist post was actually assigned.",
                            )
                        )
                if requested_approval and not effective:
                    status = "deferred" if item.decision == "partially_approve" else "rejected"
                self._record_management_outcome(
                    request=request,
                    decision_status=status,
                    committed_units=committed,
                    public_message=item.public_message,
                    effective=effective,
                )

        # Requests outside the configured finite dockets are explicitly deferred with no
        # effect. This prevents invisible approvals and preserves a complete ledger.
        for request_id, request in self._current_management_requests.items():
            if request_id in processed or request_id in docket_request_ids:
                continue
            identity_current = self._request_identity_is_current(request)
            self._record_management_outcome(
                request=request,
                decision_status="deferred" if identity_current else "identity_invalidated",
                committed_units=0.0,
                public_message=(
                    "The request did not enter this month's privacy-minimized representative docket."
                    if identity_current
                    else "The request was invalidated because the requesting identity is no longer current."
                ),
                effective=False,
            )
        return self._advance_after_manager()

    def _advance_after_manager(self) -> dict[int, BureaucracyObservation | ManagerObservation]:
        if self.enable_quarterly_reflections and self.month % 3 == 0 and self._active_ids():
            return self._prepare_quarterly_observations()
        self.month += 1
        return self._prepare_monthly_observations()

    def _step_quarterly(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, BureaucracyObservation | ManagerObservation]:
        reflection_ids = {
            slot_id
            for slot_id in self._active_ids()
            if self.fact_history.get(self._state_identity_key(slot_id))
        }
        if set(responses) != reflection_ids:
            raise ValueError(
                f"quarterly response IDs mismatch; missing={reflection_ids-set(responses)}, unexpected={set(responses)-reflection_ids}"
            )
        for slot_id in sorted(reflection_ids):
            reflection = BureaucratQuarterlyReflection.model_validate(
                self._parse_response(responses[slot_id], BureaucratQuarterlyReflection).model_dump()
            )
            self._validate_event_refs(slot_id, reflection.event_refs, quarterly=True)
            state = self.states[slot_id]
            self.reflections.append(
                {
                    "month": self.month,
                    "slot_id": slot_id,
                    "person_id": state.person_id,
                    "identity_epoch": state.identity_epoch,
                    **reflection.model_dump(mode="json"),
                }
            )
        self.month += 1
        return self._prepare_monthly_observations()

    def _exit_person(self, slot_id: int, *, reason: ExitReason, probability: float) -> None:
        state = self.states[slot_id]
        if not state.active:
            return
        department = state.department
        person_id = state.person_id
        state.active = False
        self.departments[department].cumulative_departures += 1
        self.departments[department].team_routine_stock *= math.exp(
            -self.params.exit_routine_damage
        )
        for request in self.transfer_requests:
            if (
                request.status == "pending"
                and request.slot_id == slot_id
                and request.person_id == person_id
                and request.identity_epoch == state.identity_epoch
            ):
                request.status = "withdrawn"
        self.staffing_events.append(
            StaffingEvent(
                month=self.month,
                event_type="exit",
                slot_id=slot_id,
                person_id=person_id,
                from_department=department,
                exit_reason=reason,
                details=(
                    f"Single-draw competing-risk exit; selected cause hazard={probability:.6f}."
                ),
            )
        )
        for colleague_id in self._active_ids_in_department(department):
            self._queue_event(
                colleague_id,
                WorkEvent(
                    event_id=f"m{self.month+1:02d}-s{colleague_id:04d}-colleague-exit-{slot_id}",
                    month=self.month + 1,
                    event_type="colleague_exit",
                    description="A colleague left the department; tacit skill left with the person and team routines were disrupted.",
                ),
            )

    def _pending_transfer_map(self) -> dict[int, TransferRequestRecord]:
        result: dict[int, TransferRequestRecord] = {}
        for request in self.transfer_requests:
            if request.status != "pending":
                continue
            if self.month > request.expires_after_month:
                request.status = "expired"
                state = self.states.get(request.slot_id)
                if (
                    state is not None
                    and state.active
                    and state.person_id == request.person_id
                    and state.identity_epoch == request.identity_epoch
                ):
                    self._queue_event(
                        request.slot_id,
                        WorkEvent(
                            event_id=f"m{self.month+1:02d}-s{request.slot_id:04d}-transfer-expired",
                            month=self.month + 1,
                            event_type="transfer_application_outcome",
                            description="Your transfer request expired after twelve months without a feasible match.",
                            decision_status="rejected",
                        ),
                    )
                continue
            state = self.states.get(request.slot_id)
            if (
                state is not None
                and state.active
                and state.person_id == request.person_id
                and state.identity_epoch == request.identity_epoch
            ):
                request.last_considered_month = self.month
                result[request.slot_id] = request
        return result

    def _perform_transfers(self, policy: PolicyConfig) -> None:
        pressures = {
            department: self.departments[department].workload_ratio
            + 0.20 * self.departments[department].critical_overdue_units
            / max(self._base_monthly_demand[department], 1e-9)
            for department in DEPARTMENTS
        }
        pending = self._pending_transfer_map()
        plan = build_transfer_plan(
            month=self.month,
            states=self.states,
            profiles=self.profiles,
            actions=self.last_actions,
            transfer_requests=pending,
            pressures=pressures,
            policy=policy,
            shock_tape=self.shock_tape,
        )
        moved_ids: set[int] = set()
        for item in plan:
            state = self.states[item.slot_id]
            if not state.active:
                continue
            moved_ids.add(item.slot_id)
            old_department = state.department
            state.department = item.to_department
            state.months_in_department = 0
            state.transfer_disruption_months = 2 if item.voluntary else 3
            state.recipient_coordination_months = 2
            self.departments[old_department].cumulative_transfers_out += 1
            self.departments[item.to_department].cumulative_transfers_in += 1
            self.departments[old_department].team_routine_stock *= math.exp(
                -self.params.transfer_routine_damage
            )
            self.departments[item.to_department].team_routine_stock *= math.exp(
                -0.5 * self.params.transfer_routine_damage
            )
            self._recipient_coordination_cost_next[item.to_department] += (
                self.profiles[item.slot_id].baseline_capacity_units
                * self.params.recipient_coordination_cost_rate
            )
            if item.request_id:
                request = next(
                    req for req in self.transfer_requests if req.request_id == item.request_id
                )
                request.status = "selected"
            event_type = "voluntary_transfer" if item.voluntary else "involuntary_transfer"
            self.staffing_events.append(
                StaffingEvent(
                    month=self.month,
                    event_type=event_type,
                    slot_id=item.slot_id,
                    person_id=state.person_id,
                    from_department=old_department,
                    to_department=item.to_department,
                    details=(
                        "Simultaneous capacity-constrained transfer; tacit skill moved with the person, "
                        "codified knowledge remained in the source department."
                    ),
                )
            )
            self.transfer_plans.append(item)
            explained = (
                self.shock_tape.realized_policy_draw(
                    self.month, state.person_id, "transfer-explanation"
                )
                < policy.transfer_explanation_quality
            )
            appeal_available = (
                self.shock_tape.realized_policy_draw(
                    self.month, state.person_id, "transfer-appeal-available"
                )
                < policy.appeal_channel
            )
            appeal_relief = (
                (not item.voluntary)
                and appeal_available
                and self.shock_tape.realized_policy_draw(
                    self.month, state.person_id, "transfer-appeal-relief"
                )
                < 0.50
            )
            if explained:
                self._record_exposure(
                    ExposureEvent(
                        exposure_id=f"x-m{self.month:02d}-s{item.slot_id:04d}-transfer-explanation",
                        month_decided=self.month,
                        effective_month=self.month + 1,
                        department=item.to_department,
                        exposure_type="transfer_explanation",
                        slot_id=item.slot_id,
                        person_id=state.person_id,
                        identity_epoch=state.identity_epoch,
                        intensity=1.0,
                        description="A transfer explanation was actually delivered.",
                    )
                )
            if appeal_relief:
                self._record_exposure(
                    ExposureEvent(
                        exposure_id=f"x-m{self.month:02d}-s{item.slot_id:04d}-appeal-relief",
                        month_decided=self.month,
                        effective_month=self.month + 1,
                        department=item.to_department,
                        exposure_type="appeal_relief",
                        slot_id=item.slot_id,
                        person_id=state.person_id,
                        identity_epoch=state.identity_epoch,
                        intensity=1.0,
                        description="A transfer appeal produced a concrete scheduling/accommodation relief.",
                    )
                )
            self._queue_event(
                item.slot_id,
                WorkEvent(
                    event_id=f"m{self.month+1:02d}-s{item.slot_id:04d}-personal-transfer",
                    month=self.month + 1,
                    event_type="personal_transfer",
                    description=(
                        f"You moved from {old_department} to {item.to_department}. "
                        f"Explanation delivered={str(explained).lower()}; appeal relief realized={str(appeal_relief).lower()}."
                    ),
                    involuntary=not item.voluntary,
                    decision_status="approved",
                    explanation_received=explained,
                    appeal_available=appeal_available,
                    appeal_relief_realized=appeal_relief,
                ),
            )
            for colleague_id in self._active_ids_in_department(old_department):
                self._queue_event(
                    colleague_id,
                    WorkEvent(
                        event_id=f"m{self.month+1:02d}-s{colleague_id:04d}-transfer-out-{item.slot_id}",
                        month=self.month + 1,
                        event_type="colleague_transfer_out",
                        description="A colleague transferred out; tacit capacity left and team routines were disrupted.",
                    ),
                )
            for colleague_id in self._active_ids_in_department(item.to_department):
                if colleague_id == item.slot_id:
                    continue
                self._queue_event(
                    colleague_id,
                    WorkEvent(
                        event_id=f"m{self.month+1:02d}-s{colleague_id:04d}-transfer-in-{item.slot_id}",
                        month=self.month + 1,
                        event_type="colleague_transfer_in",
                        description="A colleague transferred in; future capacity rose while onboarding coordination consumes capacity next month.",
                    ),
                )
        for slot_id, request in pending.items():
            if slot_id in moved_ids:
                continue
            self._queue_event(
                slot_id,
                WorkEvent(
                    event_id=f"m{self.month+1:02d}-s{slot_id:04d}-transfer-pending",
                    month=self.month + 1,
                    event_type="transfer_application_outcome",
                    description=(
                        "Your transfer request remains queued because no simultaneous donor-capacity, "
                        "recipient-capacity, and field-fit match was available this cycle."
                    ),
                    decision_status="deferred",
                ),
            )

    def _perform_hiring(self, policy: PolicyConfig) -> None:
        open_slots = sorted(slot_id for slot_id, state in self.states.items() if not state.active)
        capacity = int(math.floor(self._num_agents * policy.hiring_capacity_rate))
        if not open_slots or capacity <= 0:
            return
        fill_cap = min(capacity, int(math.ceil(len(open_slots) * policy.hiring_fill_rate)))
        # Allocate the batch against projected post-hire capacity. Without this
        # update, every vacancy in a hiring round sees the same pre-batch state
        # and can collapse into one department.
        projected_capacity = {
            department: sum(
                self.profiles[slot_id].baseline_capacity_units
                for slot_id in self._active_ids_in_department(department)
            )
            for department in DEPARTMENTS
        }
        for slot_id in open_slots[:fill_cap]:
            target = max(
                DEPARTMENTS,
                key=lambda department: (
                    self.departments[department].required_work_units
                    / max(projected_capacity[department], 1e-9)
                    + self.departments[department].critical_overdue_units
                    / max(self._base_monthly_demand[department], 1e-9),
                    department,
                ),
            )
            old_state = self.states[slot_id]
            identity_epoch = old_state.identity_epoch + 1
            midcareer = (
                self.shock_tape.hire_type_draw(self.month, slot_id, identity_epoch)
                < policy.midcareer_hire_share
            )
            target_field = self.task_queue.dominant_required_field(target)
            profile = generate_replacement_profile(
                seed=self.seed,
                slot_id=slot_id,
                identity_epoch=identity_epoch,
                month=self.month,
                department=target,
                midcareer=midcareer,
                scenario=self.scenario,
                target_field=target_field,
            )
            self.profiles[slot_id] = profile
            projected_capacity[target] += profile.baseline_capacity_units
            self.profile_history[profile.person_id] = profile.model_copy(deep=True)
            self.states[slot_id] = BureaucratState(
                slot_id=slot_id,
                person_id=profile.person_id,
                identity_epoch=identity_epoch,
                active=True,
                initial_cohort=False,
                department=target,
                work_strain_pressure=0.18,
                skill_stock=profile.initial_skill_stock,
                modernization_stock=self.departments[target].modernization_stock,
                months_in_department=0,
                months_since_hire=0,
                recipient_coordination_months=2,
            )
            self.departments[target].cumulative_hires += 1
            self.departments[target].team_routine_stock *= math.exp(-0.003)
            self._recipient_coordination_cost_next[target] += (
                profile.baseline_capacity_units * self.params.recipient_coordination_cost_rate
            )
            self.staffing_events.append(
                StaffingEvent(
                    month=self.month,
                    event_type="hire",
                    slot_id=slot_id,
                    person_id=profile.person_id,
                    to_department=target,
                    details=(
                        f"Replacement hire into an existing vacant slot; target_field={target_field}; "
                        f"midcareer={midcareer}. Profile identity excludes hire month."
                    ),
                )
            )
            self._queue_event(
                slot_id,
                WorkEvent(
                    event_id=f"m{self.month+1:02d}-s{slot_id:04d}-new-hire",
                    month=self.month + 1,
                    event_type="new_hire_arrival",
                    description=(
                        f"You joined {target} in a vacant slot with a {target_field} assignment and a six-month onboarding ramp."
                    ),
                ),
            )
            for colleague_id in self._active_ids_in_department(target):
                if colleague_id == slot_id:
                    continue
                self._queue_event(
                    colleague_id,
                    WorkEvent(
                        event_id=f"m{self.month+1:02d}-s{colleague_id:04d}-hire-{slot_id}",
                        month=self.month + 1,
                        event_type="new_hire_arrival",
                        description="A replacement hire arrived; mentoring and onboarding consume coordination capacity before full productivity.",
                    ),
                )

    def get_result(self) -> PolicyLabResult:
        total_service_loss = sum(
            state.cumulative_service_harm_points for state in self.departments.values()
        )
        terminal_liability = sum(
            state.terminal_liability_points for state in self.departments.values()
        )
        summary: dict[str, float | int | str | None] = {
            "version": "1.4.5",
            "scenario": self.scenario,
            "manager_mode": self.manager_mode,
            "seed": self.seed,
            "months": self.months,
            "initial_slots": self._num_agents,
            "final_active_headcount": len(self._active_ids()),
            "departures": sum(
                1 for event in self.staffing_events if event.event_type == "exit"
            ),
            "hires": sum(
                1 for event in self.staffing_events if event.event_type == "hire"
            ),
            "voluntary_transfers": sum(
                1
                for event in self.staffing_events
                if event.event_type == "voluntary_transfer"
            ),
            "involuntary_transfers": sum(
                1
                for event in self.staffing_events
                if event.event_type == "involuntary_transfer"
            ),
            "cumulative_service_harm_points": total_service_loss,
            "terminal_liability_points": terminal_liability,
            "management_outcomes": len(self.management_outcomes),
            "realized_exposures": len(self.exposure_events),
            "task_ledger_rows": len(self.task_queue.rows),
        }
        return PolicyLabResult(
            summary=summary,
            profiles=list(self.profile_history.values()),
            monthly_records=self.monthly_records,
            department_rows=self.department_rows,
            staffing_events=self.staffing_events,
            transfer_plans=self.transfer_plans,
            transfer_requests=self.transfer_requests,
            management_outcomes=self.management_outcomes,
            exposure_events=self.exposure_events,
            task_rows=self.task_queue.rows,
            reflections=self.reflections,
        )
