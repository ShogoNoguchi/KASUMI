from __future__ import annotations

import argparse
import json
from pathlib import Path
from .evidence import load_evidence
from .plotting import plot_development_candidates, plot_holdout_cells
from .report import write_markdown_report
from .verify import verify_public_bundle


def run(evidence_dir: str | Path, out_dir: str | Path) -> dict[str, object]:
    evidence_dir = Path(evidence_dir)
    out_dir = Path(out_dir)
    figures = out_dir / "figures"
    bundle = load_evidence(evidence_dir)
    verification = verify_public_bundle(evidence_dir)
    plot_development_candidates(bundle.development, figures)
    plot_holdout_cells(bundle.holdout, figures)
    report_path = write_markdown_report(bundle, figures, out_dir / "REPLAY_REPORT.md")
    result = {"report": str(report_path), **verification}
    (out_dir / "replay_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the public-service policy-science evidence bundle.")
    parser.add_argument("--evidence-dir", default="artifacts/evidence")
    parser.add_argument("--out-dir", default="outputs/replay")
    args = parser.parse_args()
    print(json.dumps(run(args.evidence_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
