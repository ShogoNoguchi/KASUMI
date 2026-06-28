"""Two-perspective review for synthetic public-administration research papers.

Reviewer A checks computational/scientific rigor. Reviewer B checks public-
administration feasibility, rights, service continuity, and extrapolation. This
is an adaptation layer; it is not represented as a calibrated Japanese review
dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import pydantic


class DomainReview(pydantic.BaseModel):
    reviewer_role: Literal["computational_science", "public_administration"]
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    required_revisions: list[str]
    causal_validity_score: int = pydantic.Field(ge=1, le=5)
    evidence_traceability_score: int = pydantic.Field(ge=1, le=5)
    service_continuity_score: int = pydantic.Field(ge=1, le=5)
    fairness_rights_score: int = pydantic.Field(ge=1, le=5)
    claim_boundary_score: int = pydantic.Field(ge=1, le=5)
    recommendation: Literal["accept_poc", "revise", "reject_claims"]
    confidence: int = pydantic.Field(ge=1, le=5)


def read_paper(path: Path) -> str:
    if path.suffix.lower() in {".tex", ".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to review a PDF") from exc
        document = fitz.open(path)
        return "\n".join(page.get_text() for page in document)
    raise ValueError(f"Unsupported paper format: {path.suffix}")


def _read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if fenced:
        return json.loads(fenced.group(1))
    if stripped.startswith("{"):
        return json.loads(stripped)
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        return json.loads(stripped[first : last + 1])
    raise ValueError("Model did not return a JSON object")


def prompt(
    *,
    role: str,
    paper: str,
    verification: dict[str, Any],
    final_manifest: dict[str, Any],
    evidence: str,
) -> str:
    schema = DomainReview.model_json_schema()
    common = (
        "Review this as a proof-of-concept synthetic research harness, not as an empirical causal "
        "estimate or digital twin. Check whether every numerical claim is traceable, whether the "
        "matched design and holdout contract are respected, whether Shachi and the policy environment "
        "are described from the provided model/evidence cards rather than release-history jargon, and "
        "whether limitations are explicit. Return only one JSON object that validates against the schema.\n\n"
    )
    if role == "computational_science":
        focus = (
            "Focus on transition invariants, common random numbers, selection/holdout leakage, "
            "independent experimental units, implementation correctness, sensitivity, claim verification, "
            "and whether the final manuscript correctly separates development and frozen holdout evidence."
        )
    else:
        focus = (
            "Focus on lawful authority, procedural fairness, staff welfare, involuntary transfer, "
            "service continuity, operational feasibility, synthetic-to-real extrapolation, Japanese "
            "administrative specificity, and whether the Wakate source is used only as problem context."
        )
    return (
        common
        + focus
        + "\n\nJSON SCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nCLAIM VERIFICATION:\n"
        + json.dumps(verification, ensure_ascii=False)
        + "\n\nFINAL PAPER MANIFEST:\n"
        + json.dumps(final_manifest, ensure_ascii=False)
        + "\n\nPAPER EVIDENCE:\n"
        + evidence
        + "\n\nPAPER:\n"
        + paper
    )


def mock_review(role: str, verification: dict[str, Any]) -> DomainReview:
    passed = bool(verification.get("passed"))
    return DomainReview(
        reviewer_role=role,
        summary="The artifact is a bounded synthetic PoC with explicit ledgers and a restricted claim boundary.",
        strengths=["Reproducible matched transition contract", "Machine-readable claim provenance"],
        weaknesses=["Synthetic coefficients are not empirically calibrated", "One development design does not establish real policy effects"],
        required_revisions=[] if passed else ["Repair failed numerical claim verification before interpretation"],
        causal_validity_score=4 if passed else 2,
        evidence_traceability_score=5 if passed else 1,
        service_continuity_score=4,
        fairness_rights_score=4,
        claim_boundary_score=5,
        recommendation="accept_poc" if passed else "reject_claims",
        confidence=4,
    )


def call_model(
    *,
    model: str,
    role: str,
    paper: str,
    verification: dict[str, Any],
    final_manifest: dict[str, Any],
    evidence: str,
) -> DomainReview:
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError("Install LiteLLM for domain review") from exc
    completion = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt(
                    role=role,
                    paper=paper,
                    verification=verification,
                    final_manifest=final_manifest,
                    evidence=evidence,
                ),
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        drop_params=True,
    )
    content = completion.choices[0].message.content
    return DomainReview.model_validate(_extract_json_object(content))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--claim-verification", type=Path, required=True)
    parser.add_argument("--final-paper-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-a", default=os.environ.get("POLICYLAB_SCIENTIFIC_REVIEW_MODEL", "gemini/gemini-2.5-pro"))
    parser.add_argument("--model-b", default=os.environ.get("POLICYLAB_PUBLIC_ADMIN_REVIEW_MODEL", "gemini/gemini-2.5-flash"))
    args = parser.parse_args()
    verification = json.loads(args.claim_verification.read_text(encoding="utf-8"))
    final_manifest = json.loads(args.final_paper_manifest.read_text(encoding="utf-8"))
    if final_manifest.get("status") != "complete":
        raise RuntimeError("Final-paper manifest is not complete")
    if final_manifest.get("final_paper_sha256") != hashlib.sha256(args.paper.read_bytes()).hexdigest():
        raise RuntimeError("Paper does not match final-paper manifest")
    if final_manifest.get("claim_verification_sha256") != hashlib.sha256(args.claim_verification.read_bytes()).hexdigest():
        raise RuntimeError("Claim verification does not match final-paper manifest")
    manifest_dir = args.final_paper_manifest.resolve().parent
    evidence_path = manifest_dir / "paper_evidence.md"
    evidence = _read_optional(evidence_path)
    if final_manifest.get("paper_evidence_markdown_sha256") and evidence_path.is_file():
        if final_manifest["paper_evidence_markdown_sha256"] != hashlib.sha256(evidence_path.read_bytes()).hexdigest():
            raise RuntimeError("Paper evidence does not match final-paper manifest")
    paper = read_paper(args.paper)
    use_mock = os.environ.get("POLICYLAB_MOCK_LLM") == "1"
    if use_mock and os.environ.get("POLICYLAB_ALLOW_TEST_MODE") != "1":
        raise RuntimeError("Mock review is test-only")
    reviews = [
        mock_review("computational_science", verification)
        if use_mock
        else call_model(
            model=args.model_a,
            role="computational_science",
            paper=paper,
            verification=verification,
            final_manifest=final_manifest,
            evidence=evidence,
        ),
        mock_review("public_administration", verification)
        if use_mock
        else call_model(
            model=args.model_b,
            role="public_administration",
            paper=paper,
            verification=verification,
            final_manifest=final_manifest,
            evidence=evidence,
        ),
    ]
    result = {
        "paper": str(args.paper.resolve()),
        "paper_sha256": hashlib.sha256(args.paper.read_bytes()).hexdigest(),
        "claim_verification": str(args.claim_verification.resolve()),
        "claim_verification_sha256": hashlib.sha256(
            args.claim_verification.read_bytes()
        ).hexdigest(),
        "claim_verification_passed": verification.get("passed"),
        "final_paper_manifest": str(args.final_paper_manifest.resolve()),
        "final_paper_manifest_sha256": hashlib.sha256(
            args.final_paper_manifest.read_bytes()
        ).hexdigest(),
        "paper_evidence": str(evidence_path.resolve()) if evidence_path.is_file() else None,
        "paper_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest() if evidence_path.is_file() else None,
        "models": {"computational_science": args.model_a, "public_administration": args.model_b},
        "calibration_claim": "None; this is an explicit domain-adaptation PoC, not a calibrated reviewer benchmark.",
        "reviews": [review.model_dump(mode="json") for review in reviews],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
