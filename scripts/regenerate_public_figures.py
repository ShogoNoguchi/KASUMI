#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from civic_policy_scientist.evidence import load_evidence
from civic_policy_scientist.plotting import (
    plot_development_candidates,
    plot_holdout_cells,
    plot_primary_welfare_vs_service_loss,
)


def copy_to_docs(path: Path) -> None:
    target = Path("docs") / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def main() -> None:
    bundle = load_evidence("artifacts/evidence")
    figure_dir = Path("artifacts/figures")
    paths = [
        plot_primary_welfare_vs_service_loss(bundle.development, figure_dir),
        plot_development_candidates(bundle.development, figure_dir),
        plot_holdout_cells(bundle.holdout, figure_dir),
    ]
    for path in paths:
        copy_to_docs(path)
        print(f"wrote {path}")
        print(f"synced docs/{path}")


if __name__ == "__main__":
    main()
