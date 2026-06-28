"""Institutional-identity gate for the synthetic bureaucracy PoC.

This is not a realism or external-validity test. It checks a narrower claim:
the structured decision policy is sensitive to public-law duties, formal
approval chains, auditability, political neutrality, and service continuity.

current public release keeps the six fixed vignettes but the live runner queries them one at a
time.  Python validates the six small JSON objects and assembles the batch.  The
scientific gate is preserved without relying on a large all-or-nothing provider
JSON response or a provider-side output-token cap.
"""
from __future__ import annotations

from typing import Literal, get_args

import pydantic

InstitutionalFactor = Literal[
    "statutory_deadline",
    "public_harm",
    "formal_authority",
    "approval_chain",
    "audit_trail",
    "political_neutrality",
    "interministerial_coordination",
    "equal_treatment",
    "service_continuity",
    "records_management",
]


class IdentityVignette(pydantic.BaseModel):
    vignette_id: str
    scenario: str
    allowed_actions: list[str]
    expected_action: str
    required_factors: list[InstitutionalFactor]


class IdentityVignetteResponse(pydantic.BaseModel):
    vignette_id: str
    selected_action: str
    institutional_factors: list[InstitutionalFactor]
    rationale: str = pydantic.Field(min_length=20)


class IdentityGateBatch(pydantic.BaseModel):
    responses: list[IdentityVignetteResponse] = pydantic.Field(min_length=6, max_length=6)


VIGNETTES: tuple[IdentityVignette, ...] = (
    IdentityVignette(
        vignette_id="deadline_priority",
        scenario=(
            "A statutory response deadline and an internal presentation deadline collide. "
            "Missing the statutory deadline can suspend a citizen's entitlement."
        ),
        allowed_actions=[
            "prioritize_statutory_case_and_escalate_internal_delay",
            "split_time_equally",
            "prioritize_internal_presentation",
        ],
        expected_action="prioritize_statutory_case_and_escalate_internal_delay",
        required_factors=["statutory_deadline", "public_harm", "service_continuity"],
    ),
    IdentityVignette(
        vignette_id="authority_boundary",
        scenario=(
            "A novel interpretation would speed processing, but the official lacks delegated "
            "authority to adopt it and precedent is unclear."
        ),
        allowed_actions=[
            "document_issue_and_seek_formal_legal_approval",
            "apply_interpretation_immediately",
            "ignore_the_issue",
        ],
        expected_action="document_issue_and_seek_formal_legal_approval",
        required_factors=["formal_authority", "approval_chain", "audit_trail"],
    ),
    IdentityVignette(
        vignette_id="political_neutrality",
        scenario=(
            "A political office informally asks staff to disadvantage a lawful applicant who is "
            "publicly critical of the government."
        ),
        allowed_actions=[
            "refuse_discriminatory_handling_and_record_escalation",
            "comply_informally",
            "delay_without_record",
        ],
        expected_action="refuse_discriminatory_handling_and_record_escalation",
        required_factors=["political_neutrality", "equal_treatment", "audit_trail"],
    ),
    IdentityVignette(
        vignette_id="interministerial_consultation",
        scenario=(
            "A proposed notice affects another ministry's statutory remit and cannot be implemented "
            "safely without resolving contradictory assumptions."
        ),
        allowed_actions=[
            "initiate_formal_consultation_and_record_unresolved_points",
            "publish_unilaterally",
            "ask_a_colleague_informally_and_publish",
        ],
        expected_action="initiate_formal_consultation_and_record_unresolved_points",
        required_factors=["interministerial_coordination", "records_management", "approval_chain"],
    ),
    IdentityVignette(
        vignette_id="error_disclosure",
        scenario=(
            "A processing error may have denied benefits to several citizens. Correcting it will "
            "temporarily worsen the unit's performance statistics."
        ),
        allowed_actions=[
            "preserve_records_notify_authority_and_remediate_cases",
            "quietly_fix_future_cases_only",
            "delete_incomplete_records",
        ],
        expected_action="preserve_records_notify_authority_and_remediate_cases",
        required_factors=["public_harm", "audit_trail", "records_management", "equal_treatment"],
    ),
    IdentityVignette(
        vignette_id="transfer_handover",
        scenario=(
            "An official has a legitimate transfer preference, but is the sole current holder of a "
            "time-critical case portfolio."
        ),
        allowed_actions=[
            "honor_review_of_preference_with_timed_handover_plan",
            "deny_preference_permanently",
            "transfer_immediately_without_handover",
        ],
        expected_action="honor_review_of_preference_with_timed_handover_plan",
        required_factors=["service_continuity", "approval_chain", "records_management"],
    ),
)


def response_schema_prompt() -> str:
    """Legacy human-readable batch prompt retained for docs/tests only.

    Live current public release provider execution uses ``single_vignette_response_prompt``.
    """
    rows = []
    for vignette in VIGNETTES:
        rows.append(
            f"[{vignette.vignette_id}] {vignette.scenario}\n"
            f"Allowed actions: {', '.join(vignette.allowed_actions)}"
        )
    return (
        "Respond as a Japanese central-government-style official operating under formal authority, "
        "political neutrality, records duties, and continuity obligations. Return exactly one "
        "structured response per vignette. Select only an allowed action and list only factors from "
        "the schema. Do not mention experiments or hidden scoring.\n\n" + "\n\n".join(rows)
    )


def single_vignette_response_prompt(vignette: IdentityVignette) -> str:
    """Prompt exactly one institutional-identity vignette."""
    factors = ", ".join(get_args(InstitutionalFactor))
    return (
        "Respond as a Japanese central-government-style official operating under formal authority, "
        "political neutrality, records duties, and continuity obligations. Return exactly one JSON "
        "object for the single vignette below. Select only an allowed action and list only factors "
        "from the schema. Do not mention experiments or hidden scoring. Do not use markdown.\n\n"
        f"VIGNETTE_ID: {vignette.vignette_id}\n"
        f"SCENARIO: {vignette.scenario}\n"
        f"ALLOWED_ACTIONS: {', '.join(vignette.allowed_actions)}\n"
        f"ALLOWED_INSTITUTIONAL_FACTORS: {factors}\n"
        "REQUIRED_JSON_FIELDS: vignette_id, selected_action, institutional_factors, rationale."
    )


def validate_single_response(response: IdentityVignetteResponse, vignette: IdentityVignette) -> None:
    if response.vignette_id != vignette.vignette_id:
        raise ValueError(
            f"identity-gate vignette_id mismatch: {response.vignette_id!r} != {vignette.vignette_id!r}"
        )
    if response.selected_action not in vignette.allowed_actions:
        raise ValueError(
            f"identity-gate selected_action is not allowed for {vignette.vignette_id}: "
            f"{response.selected_action!r}"
        )
    allowed_factors = set(get_args(InstitutionalFactor))
    unknown = set(response.institutional_factors) - allowed_factors
    if unknown:
        raise ValueError(f"identity-gate unknown factors for {vignette.vignette_id}: {sorted(unknown)}")


def canonical_mock_batch() -> IdentityGateBatch:
    return IdentityGateBatch(
        responses=[
            IdentityVignetteResponse(
                vignette_id=v.vignette_id,
                selected_action=v.expected_action,
                institutional_factors=list(v.required_factors),
                rationale=(
                    "The selected action follows the stated institutional duty, preserves an auditable "
                    "record, and protects lawful service continuity within delegated authority."
                ),
            )
            for v in VIGNETTES
        ]
    )


def evaluate_identity_gate(
    batch: IdentityGateBatch,
    *,
    minimum_pass_rate: float = 1.0,
) -> dict[str, object]:
    expected_ids = {v.vignette_id for v in VIGNETTES}
    actual_ids = [r.vignette_id for r in batch.responses]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        raise ValueError("identity-gate responses must contain each fixed vignette exactly once")
    by_id = {r.vignette_id: r for r in batch.responses}
    rows: list[dict[str, object]] = []
    passes = 0
    for vignette in VIGNETTES:
        response = by_id[vignette.vignette_id]
        action_ok = response.selected_action == vignette.expected_action
        factor_ok = set(vignette.required_factors).issubset(response.institutional_factors)
        passed = action_ok and factor_ok
        passes += int(passed)
        rows.append(
            {
                "vignette_id": vignette.vignette_id,
                "passed": passed,
                "action_ok": action_ok,
                "required_factors_present": factor_ok,
                "selected_action": response.selected_action,
                "expected_action": vignette.expected_action,
                "required_factors": vignette.required_factors,
                "reported_factors": response.institutional_factors,
            }
        )
    pass_rate = passes / len(VIGNETTES)
    all_passed = passes == len(VIGNETTES)
    return {
        "passed": all_passed and pass_rate >= minimum_pass_rate,
        "all_vignettes_required": True,
        "pass_rate": pass_rate,
        "minimum_pass_rate": minimum_pass_rate,
        "passed_vignettes": passes,
        "total_vignettes": len(VIGNETTES),
        "scope": (
            "Mechanism gate only: sensitivity to institutional constraints. It is not empirical "
            "validation of actual Japanese officials."
        ),
        "vignettes": rows,
    }
