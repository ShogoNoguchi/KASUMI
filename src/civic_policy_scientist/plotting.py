from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_development_candidates(development: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    rows = development["candidates"]
    labels = [r["label"].replace("_pathway", "").replace("_", "\n") for r in rows]
    values = [r["delta_vs_reference"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(labels, values)
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("Primary endpoint delta vs reference")
    ax.set_title("Development-stage candidate comparison")
    ax.tick_params(axis="x", labelsize=8)
    path = out_dir / "development_candidate_primary_delta.png"
    _save(fig, path)
    return path


def plot_holdout_cells(holdout: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    rows = holdout["cells"]
    labels = [str(r["seed"]) for r in rows]
    values = [r["primary_delta_selected_minus_reference"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(labels, values)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Holdout seed")
    ax.set_ylabel("Selected policy delta vs reference")
    ax.set_title("Multiseed holdout robustness")
    path = out_dir / "holdout_primary_delta_by_seed.png"
    _save(fig, path)
    return path
