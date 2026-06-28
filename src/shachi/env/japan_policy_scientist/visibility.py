"""Science–operator visibility boundary for Japan Policy Scientist current public release.

The AI Scientist must see synthetic policy implementation burden because it is
part of the scientific comparison. It must not see simulation-compute spending,
token volume, cache volume, or runtime, because those operator facts can induce
cheap-but-uninformative experiment selection.

This is an information-flow boundary, not a security sandbox. The protected
experiment contract also fixes scale and duration, so The AI Scientist cannot
reduce fidelity in response to hidden operator accounting.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

# Exact scalar metrics withheld from The AI Scientist. Policy implementation
# cost is deliberately NOT in this set.
_OPERATOR_ONLY_EXACT_KEYS = {
    "runtime_seconds",
    "simulation_compute_cost_usd",
    "estimated_llm_cost_usd",  # legacy internal name
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_provider_attempts",
    "llm_valid_responses",
    "llm_calls",
    "llm_failed_calls",
    "llm_reserved_cost_usd_end",
}
_OPERATOR_ONLY_PREFIXES = (
    "response_cache_",
)


def is_operator_only_metric(key: str) -> bool:
    """Return whether a metric is execution accounting rather than science."""
    if key in _OPERATOR_ONLY_EXACT_KEYS:
        return True
    if key.startswith(_OPERATOR_ONLY_PREFIXES):
        return True
    if key.startswith("delta_vs_baseline__"):
        base = key.removeprefix("delta_vs_baseline__")
        return is_operator_only_metric(base)
    return False


def scientist_visible_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return policy and reliability evidence safe for The AI Scientist.

    Visible examples include policy_implementation_cost_points, service,
    staffing, behavior, validation/network failure rates, and cohort outcomes.
    """
    return {
        str(key): value
        for key, value in metrics.items()
        if not is_operator_only_metric(str(key))
    }


def operator_only_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return compute/runtime accounting withheld from The AI Scientist."""
    result = {
        str(key): value
        for key, value in metrics.items()
        if is_operator_only_metric(str(key))
    }
    if "estimated_llm_cost_usd" in result:
        result["simulation_compute_cost_usd"] = result.pop(
            "estimated_llm_cost_usd"
        )
    return result


def sanitize_scientist_payload(payload: Any) -> Any:
    """Recursively remove operator-only keys from nested scientist payloads."""
    if isinstance(payload, dict):
        return {
            str(key): sanitize_scientist_payload(value)
            for key, value in payload.items()
            if not is_operator_only_metric(str(key))
        }
    if isinstance(payload, list):
        return [sanitize_scientist_payload(value) for value in payload]
    return payload


def operator_audit_dir(output_dir: str | Path) -> Path:
    """Return a hidden sibling directory for human-only execution accounting."""
    output = Path(output_dir).resolve()
    explicit = os.environ.get("POLICYLAB_OPERATOR_AUDIT_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        return root / output.parent.name / output.name
    return output.parent / ".operator_audit" / output.name
