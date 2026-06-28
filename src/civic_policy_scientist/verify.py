from __future__ import annotations

from pathlib import Path
from .evidence import load_evidence
from .selection import assert_selection_matches


def verify_public_bundle(evidence_dir: str | Path) -> dict[str, object]:
    bundle = load_evidence(evidence_dir)
    assert_selection_matches(bundle.development)
    if not bundle.holdout["all_holdout_guardrails_passed"]:
        raise AssertionError("holdout guardrails did not all pass")
    if bundle.holdout["n_holdout_cells"] < 3:
        raise AssertionError("expected at least three holdout cells")
    if not bundle.verification["claim_verification_passed"]:
        raise AssertionError("claim verification did not pass")
    if not bundle.verification["latex_gate_passed"]:
        raise AssertionError("LaTeX gate did not pass")
    if not bundle.verification["text_gate_passed"]:
        raise AssertionError("text gate did not pass")
    if any(r != "accept_poc" for r in bundle.verification["review_recommendations"]):
        raise AssertionError("automated review did not accept the proof of concept")
    return {
        "selected_run": bundle.development["selected_run"],
        "selected_label": bundle.development["selected_label"],
        "holdout_cells": bundle.holdout["n_holdout_cells"],
        "primary_holdout_delta_mean": bundle.holdout["aggregate"]["primary_delta_mean"],
        "verified_claim_count": bundle.verification["verified_claim_count"],
        "review_recommendations": bundle.verification["review_recommendations"],
    }
