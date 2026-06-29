from pathlib import Path


def test_e2e_public_surface_is_packaged() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "scripts/e2e/bootstrap_workspace.sh",
        "scripts/e2e/apply_overlays.sh",
        "scripts/e2e/run_baseline.sh",
        "scripts/e2e/run_ai_scientist_development.sh",
        "scripts/e2e/run_selection_and_holdout.sh",
        "scripts/e2e/run_final_paper.sh",
        "scripts/e2e/run_domain_review.sh",
        "third_party/ai_scientist/policy_context.py",
        "third_party/ai_scientist/ai_scientist_v1_policy.patch",
        "docs/E2E_REPRODUCTION.md",
        "configs/operator_budget_plan.template.json",
    ]
    missing = [path for path in required if not (root / path).is_file()]
    assert not missing


def test_third_party_notices_explain_upstream_integration() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "The AI Scientist v1" in text
    assert "Shachi" in text
    assert "not vendored" in text
