from __future__ import annotations

from typing import Any


def select_candidate(development_summary: dict[str, Any]) -> dict[str, Any]:
    """Recompute the predeclared development-stage selection from public evidence."""
    eligible = [
        row for row in development_summary["candidates"]
        if row["guardrails_passed"] and row["delta_vs_reference"] > 0
    ]
    if not eligible:
        raise ValueError("no guardrail-passing positive candidate")
    return max(eligible, key=lambda row: row["delta_vs_reference"])


def assert_selection_matches(development_summary: dict[str, Any]) -> None:
    selected = select_candidate(development_summary)
    if selected["run"] != development_summary["selected_run"]:
        raise AssertionError(
            f"selection mismatch: recomputed {selected['run']} vs recorded "
            f"{development_summary['selected_run']}"
        )
