"""Branch-independent common random numbers for matched policy comparisons."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .schemas import DEPARTMENTS, DepartmentName

# The first post-intervention stress episode is deliberately separated from the
# intervention start (month 13). This avoids a five-way structural break while
# preserving an identical post-policy resilience test for every arm.
COMMON_STRESS_MONTHS = (15, 16, 25, 37)

DEPARTMENT_SHOCK_STD: dict[DepartmentName, float] = {
    "policy_planning": 0.10,
    "budget_coordination": 0.12,
    "regulatory_affairs": 0.08,
    "digital_transformation": 0.09,
    "public_service_operations": 0.11,
}

DEPARTMENT_AFTER_HOURS_SHIFT: dict[DepartmentName, float] = {
    "policy_planning": 0.03,
    "budget_coordination": 0.05,
    "regulatory_affairs": -0.02,
    "digital_transformation": 0.00,
    "public_service_operations": 0.01,
}


def _unit(seed: int, *keys: object) -> float:
    payload = ":".join([str(seed), *(str(key) for key in keys)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def keyed_uniform(seed: int, *keys: object) -> float:
    return _unit(seed, *keys)


def keyed_normal(seed: int, *keys: object) -> float:
    u1 = max(_unit(seed, *keys, "u1"), 1e-12)
    u2 = _unit(seed, *keys, "u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@dataclass(frozen=True)
class ShockTape:
    seed: int

    def demand_factor(self, month: int, department: DepartmentName) -> float:
        dept_index = DEPARTMENTS.index(department)
        seasonal = 0.08 * math.sin((month + dept_index * 2) * math.pi / 6.0)
        shock = DEPARTMENT_SHOCK_STD[department] * keyed_normal(
            self.seed, "demand", month, department
        )
        common_crisis = 0.0
        if month in COMMON_STRESS_MONTHS:
            common_crisis = 0.18 + 0.06 * keyed_uniform(
                self.seed, "crisis", month, department
            )
        return max(0.65, min(1.55, 1.0 + seasonal + shock + common_crisis))

    def after_hours_severity(self, month: int, department: DepartmentName) -> int:
        u = keyed_uniform(self.seed, "after_hours", month, department)
        crisis_bonus = 0.16 if month in COMMON_STRESS_MONTHS else 0.0
        shift = DEPARTMENT_AFTER_HOURS_SHIFT[department]
        if u < max(0.0, 0.04 + crisis_bonus + shift):
            return 4
        if u < max(0.0, 0.12 + crisis_bonus + shift):
            return 3
        if u < max(0.0, 0.28 + crisis_bonus + shift):
            return 2
        if u < max(0.0, 0.52 + crisis_bonus + shift):
            return 1
        return 0

    def person_draw(self, month: int, person_id: str, purpose: str) -> float:
        return keyed_uniform(self.seed, "person", purpose, month, person_id)

    def exit_draw(self, month: int, person_id: str, reason: str = "high_intent") -> float:
        return self.person_draw(month, person_id, f"exit:{reason}")

    def hire_type_draw(self, month: int, slot_id: int, identity_epoch: int) -> float:
        return keyed_uniform(self.seed, "hire_type", month, slot_id, identity_epoch)

    def transfer_draw(self, month: int, person_id: str, purpose: str = "priority") -> float:
        return keyed_uniform(self.seed, "transfer", purpose, month, person_id)

    def specialist_track_draw(self, month: int, person_id: str) -> float:
        return keyed_uniform(self.seed, "specialist_track", month, person_id)

    def realized_policy_draw(self, month: int, person_id: str, purpose: str) -> float:
        return keyed_uniform(self.seed, "policy_realization", purpose, month, person_id)

    def manifest_hash(self, months: int) -> str:
        rows: list[str] = []
        for month in range(1, months + 1):
            for department in DEPARTMENTS:
                rows.append(
                    f"{month}:{department}:{self.demand_factor(month, department):.10f}:"
                    f"{self.after_hours_severity(month, department)}"
                )
        return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
