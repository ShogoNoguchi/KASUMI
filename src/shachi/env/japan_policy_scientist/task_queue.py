"""Deadline-, quality-, and public-harm-aware administrative task queue.

This is a synthetic mechanism test, not a calibrated model of Japanese case
volumes. It records service harm as weighted points, separate from physical work units,
in an observable stock-flow ledger: arrivals -> completion -> quality error/rework ->
overdue public harm -> terminal liability.
"""
from __future__ import annotations

from collections import defaultdict
import math
from dataclasses import dataclass, field

from .dynamics import logistic, smooth_positive
from .schemas import (
    DEPARTMENTS,
    DepartmentName,
    FieldName,
    TaskCohort,
    TaskLedgerRow,
    TaskType,
)
from .shock_tape import keyed_uniform


@dataclass(frozen=True)
class TaskSpec:
    task_type: TaskType
    share: float
    deadline_lag: int
    criticality: int
    public_harm_weight: float
    quality_threshold: float
    preferred_fields: tuple[FieldName, ...]


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        "statutory_deadline", 0.23, 1, 5, 1.70, 0.82,
        ("law", "public_management", "social_policy"),
    ),
    TaskSpec(
        "parliamentary_ministerial", 0.18, 0, 5, 1.35, 0.86,
        ("public_management", "law", "economics"),
    ),
    TaskSpec(
        "budget_legislation", 0.22, 2, 4, 1.15, 0.84,
        ("economics", "law", "public_management"),
    ),
    TaskSpec(
        "interministerial_coordination", 0.22, 2, 3, 0.85, 0.78,
        ("public_management", "economics", "social_policy"),
    ),
    TaskSpec(
        "internal_reform", 0.15, 4, 2, 0.40, 0.74,
        ("data", "engineering", "public_management"),
    ),
)


@dataclass
class TaskQueue:
    seed: int
    horizon_months: int
    cohorts: dict[str, TaskCohort] = field(default_factory=dict)
    rows: list[TaskLedgerRow] = field(default_factory=list)
    _rework_due: dict[tuple[int, DepartmentName], list[tuple[str, float, FieldName]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def reset(self) -> None:
        self.cohorts.clear()
        self.rows.clear()
        self._rework_due.clear()

    @staticmethod
    def _field_for(
        *, seed: int, month: int, department: DepartmentName, spec: TaskSpec
    ) -> FieldName:
        draw = keyed_uniform(seed, "task-field", month, department, spec.task_type)
        index = min(int(draw * len(spec.preferred_fields)), len(spec.preferred_fields) - 1)
        return spec.preferred_fields[index]

    def add_monthly_arrivals(
        self,
        *,
        month: int,
        department: DepartmentName,
        total_units: float,
    ) -> list[TaskCohort]:
        created: list[TaskCohort] = []
        if total_units > 0:
            # Jitter changes only the composition of demand. The declared external
            # demand mass is conserved exactly; it is not a second hidden demand shock.
            raw_weights = [
                spec.share
                * (
                    0.94
                    + 0.12
                    * keyed_uniform(
                        self.seed,
                        "task-share-jitter",
                        month,
                        department,
                        spec.task_type,
                    )
                )
                for spec in TASK_SPECS
            ]
            weight_total = sum(raw_weights)
            assigned = 0.0
            for index, (spec, raw_weight) in enumerate(zip(TASK_SPECS, raw_weights, strict=True)):
                units = (
                    max(0.0, total_units - assigned)
                    if index == len(TASK_SPECS) - 1
                    else total_units * raw_weight / weight_total
                )
                assigned += units
                task_id = f"m{month:02d}-{department}-{index}-{spec.task_type}"
                cohort = TaskCohort(
                    task_id=task_id,
                    department=department,
                    task_type=spec.task_type,
                    arrival_month=month,
                    deadline_month=month + spec.deadline_lag,
                    required_field=self._field_for(
                        seed=self.seed, month=month, department=department, spec=spec
                    ),
                    criticality=spec.criticality,
                    public_harm_weight=spec.public_harm_weight,
                    quality_threshold=spec.quality_threshold,
                    units_total=units,
                    units_remaining=units,
                )
                self.cohorts[task_id] = cohort
                created.append(cohort)

        for parent_id, units, required_field in self._rework_due.pop((month, department), []):
            task_id = f"m{month:02d}-{department}-rework-{parent_id}"
            cohort = TaskCohort(
                task_id=task_id,
                department=department,
                task_type="statutory_deadline",
                arrival_month=month,
                deadline_month=month + 1,
                required_field=required_field,
                criticality=4,
                public_harm_weight=1.25,
                quality_threshold=0.84,
                units_total=units,
                units_remaining=units,
                rework_parent_id=parent_id,
            )
            self.cohorts[task_id] = cohort
            created.append(cohort)
        return created

    def open_cohorts(self, department: DepartmentName) -> list[TaskCohort]:
        return [
            cohort
            for cohort in self.cohorts.values()
            if cohort.department == department and cohort.units_remaining > 1e-12
        ]

    def open_units(self, department: DepartmentName) -> float:
        return sum(cohort.units_remaining for cohort in self.open_cohorts(department))

    def urgent_units(self, department: DepartmentName, month: int) -> float:
        return sum(
            cohort.units_remaining
            for cohort in self.open_cohorts(department)
            if cohort.deadline_month <= month + 1 and cohort.criticality >= 4
        )

    def dominant_required_field(self, department: DepartmentName) -> FieldName:
        by_field: dict[FieldName, float] = defaultdict(float)
        for cohort in self.open_cohorts(department):
            by_field[cohort.required_field] += cohort.units_remaining * cohort.criticality
        if not by_field:
            return "public_management"
        return max(sorted(by_field), key=lambda field: by_field[field])

    def apply_approved_triage(
        self,
        *,
        month: int,
        department: DepartmentName,
        units: float,
    ) -> float:
        """Extend only low-criticality internal work; never erase obligations."""
        remaining = max(0.0, units)
        shifted = 0.0
        candidates = sorted(
            (
                cohort
                for cohort in self.open_cohorts(department)
                if cohort.task_type == "internal_reform" and cohort.deadline_month >= month
            ),
            key=lambda cohort: (cohort.deadline_month, cohort.task_id),
            reverse=True,
        )
        for cohort in candidates:
            if remaining <= 1e-12:
                break
            amount = min(remaining, cohort.units_remaining)
            # current public release: without cohort splitting, a full-cohort deadline extension
            # requires a full-cohort envelope.  The old half-coverage rule moved
            # entire cohorts while recording only the covered amount.
            if remaining + 1e-12 >= cohort.units_remaining:
                amount = cohort.units_remaining
                cohort.deadline_month += 1
                shifted += amount
                remaining -= amount
        return shifted

    def allocate_and_close_month(
        self,
        *,
        month: int,
        department: DepartmentName,
        completed_units: float,
        quality_index: float,
        field_coverage: dict[FieldName, float],
        terminal: bool,
    ) -> dict[str, float]:
        remaining_output = max(0.0, completed_units)
        service_loss = 0.0
        quality_error = 0.0
        rework_generated = 0.0
        completed_total = 0.0
        critical_overdue = 0.0
        terminal_liability = 0.0

        cohorts = sorted(
            self.open_cohorts(department),
            key=lambda c: (c.deadline_month, -c.criticality, -c.public_harm_weight, c.task_id),
        )
        snapshots = {cohort.task_id: cohort.units_remaining for cohort in cohorts}
        month_completed: dict[str, float] = defaultdict(float)
        month_errors: dict[str, float] = defaultdict(float)
        month_rework: dict[str, float] = defaultdict(float)

        for cohort in cohorts:
            if remaining_output <= 1e-12:
                break
            gross = min(remaining_output, cohort.units_remaining)
            coverage_pressure = float(field_coverage.get(cohort.required_field, 0.0))
            coverage = logistic(3.0 * (coverage_pressure - 0.5))
            field_sensitivity = {
                "statutory_deadline": 0.24,
                "parliamentary_ministerial": 0.22,
                "budget_legislation": 0.24,
                "interministerial_coordination": 0.16,
                "internal_reform": 0.12,
            }[cohort.task_type]
            effective_quality = quality_index - field_sensitivity * (1.0 - coverage)
            shortfall = cohort.quality_threshold - effective_quality
            error_sensitivity = {
                "statutory_deadline": 5.0,
                "parliamentary_ministerial": 4.6,
                "budget_legislation": 4.2,
                "interministerial_coordination": 3.4,
                "internal_reform": 2.6,
            }[cohort.task_type]
            error_rate = 1.0 - math.exp(-error_sensitivity * smooth_positive(shortfall, sharpness=8.0))
            errors = gross * error_rate
            net = gross - errors
            cohort.units_remaining = max(0.0, cohort.units_remaining - net)
            cohort.completed_units += net
            if cohort.units_remaining <= 1e-9:
                cohort.units_remaining = 0.0
                cohort.closed_month = month
            remaining_output -= gross
            completed_total += net
            quality_error += errors
            month_completed[cohort.task_id] += net
            month_errors[cohort.task_id] += errors
            if errors > 0:
                rework = errors * 0.85
                self._rework_due[(month + 1, department)].append(
                    (cohort.task_id, rework, cohort.required_field)
                )
                rework_generated += rework
                month_rework[cohort.task_id] += rework

        # Public harm is an incremental monthly flow from overdue open work.
        for cohort in self.open_cohorts(department):
            overdue_units = cohort.units_remaining if month > cohort.deadline_month else 0.0
            months_overdue = max(0, month - cohort.deadline_month)
            aging_multiplier = 1.0 + 0.10 * months_overdue
            loss = overdue_units * cohort.public_harm_weight * (0.06 + 0.02 * cohort.criticality) * aging_multiplier
            cohort.overdue_harm_accrued += loss
            service_loss += loss
            if cohort.criticality >= 4:
                critical_overdue += overdue_units
            if terminal:
                months_overdue_or_deferred = max(1, month - cohort.deadline_month + 1)
                liability = (
                    cohort.units_remaining
                    * cohort.public_harm_weight
                    * (1.0 + 0.15 * months_overdue_or_deferred)
                )
                terminal_liability += liability

        for cohort in cohorts:
            overdue_units = cohort.units_remaining if month > cohort.deadline_month else 0.0
            months_overdue = max(0, month - cohort.deadline_month)
            aging_multiplier = 1.0 + 0.10 * months_overdue
            loss = overdue_units * cohort.public_harm_weight * (0.06 + 0.02 * cohort.criticality) * aging_multiplier
            terminal_row = 0.0
            if terminal and cohort.units_remaining > 0:
                terminal_row = (
                    cohort.units_remaining
                    * cohort.public_harm_weight
                    * (1.0 + 0.15 * max(1, month - cohort.deadline_month + 1))
                )
            self.rows.append(
                TaskLedgerRow(
                    month=month,
                    task_id=cohort.task_id,
                    department=department,
                    task_type=cohort.task_type,
                    arrival_month=cohort.arrival_month,
                    deadline_month=cohort.deadline_month,
                    required_field=cohort.required_field,
                    criticality=cohort.criticality,
                    public_harm_weight=cohort.public_harm_weight,
                    quality_threshold=cohort.quality_threshold,
                    opening_units=snapshots[cohort.task_id],
                    completed_units=month_completed[cohort.task_id],
                    quality_error_units=month_errors[cohort.task_id],
                    rework_generated_units=month_rework[cohort.task_id],
                    closing_units=cohort.units_remaining,
                    overdue_units=overdue_units,
                    service_harm_points=loss,
                    terminal_liability_points=terminal_row,
                )
            )

        return {
            "completed_units": completed_total,
            "service_harm_points": service_loss,
            "critical_overdue_units": critical_overdue,
            "quality_error_units": quality_error,
            "rework_generated_units": rework_generated,
            "backlog_units": self.open_units(department),
            "terminal_liability_points": terminal_liability,
        }
