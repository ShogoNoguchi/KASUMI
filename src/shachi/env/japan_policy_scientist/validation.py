"""Application-level semantic validation, separate from network retry logic."""
from __future__ import annotations

import pydantic

from .schemas import (
    BureaucracyObservation,
    BureaucratMonthlyAction,
    BureaucratQuarterlyReflection,
    ManagerDecision,
    ManagerObservation,
)


def validate_response_against_observation(
    observation: BureaucracyObservation | ManagerObservation,
    response: pydantic.BaseModel,
) -> None:
    if isinstance(observation, ManagerObservation):
        if not isinstance(response, ManagerDecision):
            raise TypeError(f"manager observation requires ManagerDecision, got {type(response).__name__}")
        expected = set(observation.allowed_request_ids())
        actual = {item.request_id for item in response.decisions}
        if actual != expected:
            raise ValueError(
                f"manager decisions must exactly match docket request IDs; expected={sorted(expected)}, actual={sorted(actual)}"
            )
        request_by_id = {item.request_id: item for item in observation.request_docket}
        approved = {"approve", "partially_approve"}
        for item in response.decisions:
            if item.decision not in approved and item.committed_units > 1e-9:
                raise ValueError("rejected or deferred manager decisions must commit zero units")
        support = sum(
            item.committed_units
            for item in response.decisions
            if item.decision in approved
            and request_by_id[item.request_id].request_kind
            in {"operational_support", "staffing_relief"}
        )
        triage = sum(
            item.committed_units
            for item in response.decisions
            if item.decision in approved
            and request_by_id[item.request_id].request_kind == "operational_risk"
        )
        if support > observation.support_envelope_units + 1e-6:
            raise ValueError("manager support commitments exceed envelope")
        if triage > observation.triage_envelope_units + 1e-6:
            raise ValueError("manager triage commitments exceed envelope")
        slot_limits = {
            "process_reform": observation.reform_slots,
            "explanation": observation.explanation_slots,
            "specialist_track": observation.specialist_slots,
        }
        for request_kind, limit in slot_limits.items():
            committed_count = sum(
                1
                for item in response.decisions
                if item.decision in approved
                and request_by_id[item.request_id].request_kind == request_kind
            )
            if committed_count > limit:
                raise ValueError(f"manager {request_kind} approvals exceed finite slot envelope")
        return

    refs = getattr(response, "event_refs", None)
    allowed = set(observation.allowed_event_ids())
    if not isinstance(refs, list) or not refs or not set(refs).issubset(allowed):
        raise ValueError(f"invalid event_refs={refs}; allowed={sorted(allowed)}")
    if isinstance(response, BureaucratMonthlyAction):
        preference = response.transfer_preference
        if preference is not None:
            current = None
            marker = "department="
            if marker in observation.profile_summary:
                current = observation.profile_summary.split(marker, 1)[1].split(";", 1)[0]
            if current and preference.preferred_department == current:
                raise ValueError("preferred transfer department must differ from current department")
            if current and current in preference.acceptable_departments:
                raise ValueError("acceptable transfer departments must exclude current department")
    elif not isinstance(response, BureaucratQuarterlyReflection):
        raise TypeError(f"unexpected response model: {type(response).__name__}")
