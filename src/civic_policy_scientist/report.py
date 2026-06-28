from __future__ import annotations

from pathlib import Path
from .evidence import EvidenceBundle


def write_markdown_report(bundle: EvidenceBundle, figure_dir: Path, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dev = bundle.development
    hold = bundle.holdout
    ver = bundle.verification
    lines = [
        "# Public Service Policy Lab replay report",
        "",
        "This report is generated deterministically from the public evidence bundle.",
        "It summarizes synthetic simulation results only and makes no real-world causal claim.",
        "",
        "## Development selection",
        f"Selected run: `{dev['selected_run']}` / `{dev['selected_label']}`.",
        f"Primary endpoint: `{dev['primary_endpoint']}`.",
        f"Selected delta vs reference: `{dev['selected_delta_vs_reference']:.12f}`.",
        "",
        "![Development candidate deltas](figures/development_candidate_primary_delta.png)",
        "",
        "## Multiseed holdout",
        f"Holdout cells: {hold['n_holdout_cells']}.",
        f"Mean primary delta: `{hold['aggregate']['primary_delta_mean']:.12f}`.",
        f"All holdout guardrails passed: `{hold['all_holdout_guardrails_passed']}`.",
        "",
        "![Holdout primary deltas](figures/holdout_primary_delta_by_seed.png)",
        "",
        "## Verification",
        f"Claim verification passed: `{ver['claim_verification_passed']}`.",
        f"Verified claim count: `{ver['verified_claim_count']}`.",
        f"Text gate passed: `{ver['text_gate_passed']}`.",
        f"LaTeX gate passed: `{ver['latex_gate_passed']}`.",
        f"Automated review recommendations: `{', '.join(ver['review_recommendations'])}`.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
