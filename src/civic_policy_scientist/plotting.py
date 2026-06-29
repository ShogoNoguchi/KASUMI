from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


LABEL_MAP = {
    "synthetic_stressed_reference_v2": "Stressed\nreference",
    "responsive_management_pathway": "Responsive\nmanagement",
    "procedural_justice_pathway": "Procedural\njustice",
    "capital_deepening_pathway": "Capital\ndeepening",
    "direct_staffing_pathway": "Direct\nstaffing",
}


def _pretty_label(label: str) -> str:
    return LABEL_MAP.get(label, label.replace("_pathway", "").replace("_", "\n"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_development_candidates(development: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    rows = development["candidates"]
    labels = [_pretty_label(r["label"]) for r in rows]
    values = [r["delta_vs_reference"] for r in rows]
    colors = ["#0f766e" if r.get("selected") or r["run"] == development.get("selected_run") else "#2c6db2" if r["guardrails_passed"] else "#9aa8b8" for r in rows]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.bar(labels, values, color=colors, edgecolor="#10233d", linewidth=0.7)
    ax.axhline(0, color="#10233d", linewidth=1.0)
    ax.set_ylabel("Primary endpoint delta vs reference", fontsize=13)
    ax.set_title("Development-stage candidate comparison", fontsize=16, pad=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ymin = min(0.0, min(values))
    ymax = max(0.0, max(values))
    ypad = max((ymax - ymin) * 0.18, 0.0007)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    for i, v in enumerate(values):
        offset = 0.00018 if v >= 0 else -0.00032
        ax.text(i, v + offset, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    path = out_dir / "development_candidate_primary_delta.png"
    _save(fig, path)
    return path


def plot_holdout_cells(holdout: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    rows = holdout["cells"]
    labels = [str(r["seed"]) for r in rows]
    values = [r["primary_delta_selected_minus_reference"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.bar(labels, values, color="#0f766e", edgecolor="#10233d", linewidth=0.7)
    ax.axhline(0, color="#10233d", linewidth=1.0)
    ax.set_xlabel("Frozen holdout seed", fontsize=13)
    ax.set_ylabel("Selected policy delta vs reference", fontsize=13)
    ax.set_title("Multiseed holdout robustness", fontsize=16, pad=12)
    ax.tick_params(axis="both", labelsize=11)
    ymax = max(values)
    ax.set_ylim(0, ymax + max(ymax * 0.18, 0.001))
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    for i, v in enumerate(values):
        ax.text(i, v + 0.00018, f"{v:+.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    path = out_dir / "holdout_primary_delta_by_seed.png"
    _save(fig, path)
    return path


def plot_primary_welfare_vs_service_loss(development: dict[str, Any], out_dir: str | Path) -> Path:
    """Plot the public development-stage tradeoff from the sanitized evidence bundle.

    The x-axis is expressed as service-harm change against the stressed reference.
    A symmetric-log scale keeps the direct-staffing outlier visible without making
    the reference, procedural-justice, and capital-deepening points collapse.
    """
    out_dir = Path(out_dir)
    reference = development.get("reference") or {
        "run": development.get("reference_run", "run_0"),
        "label": development.get("reference_label", "synthetic_stressed_reference_v2"),
        "primary_endpoint": development["baseline_primary_endpoint"],
        "service_harm_points": development.get("baseline_service_harm_points"),
    }
    reference_primary = float(reference["primary_endpoint"])
    reference_harm = reference.get("service_harm_points")
    if reference_harm is None:
        raise ValueError("development summary must include reference.service_harm_points or baseline_service_harm_points")
    reference_harm = float(reference_harm)

    points = [{
        "run": reference.get("run", "run_0"),
        "label": reference.get("label", "synthetic_stressed_reference_v2"),
        "role": "reference",
        "x": 0.0,
        "y": 0.0,
        "guardrails_passed": True,
        "selected": False,
    }]
    for row in development["candidates"]:
        points.append({
            "run": row["run"],
            "label": row["label"],
            "role": "candidate",
            "x": float(row.get("service_harm_delta_vs_reference", float(row["service_harm_points"]) - reference_harm)),
            "y": float(row.get("delta_vs_reference", float(row["primary_endpoint"]) - reference_primary)),
            "guardrails_passed": bool(row["guardrails_passed"]),
            "selected": row["run"] == development.get("selected_run") or bool(row.get("selected", False)),
        })

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.axvline(0, color="#6b7280", linewidth=1.0, alpha=0.75)
    ax.axhline(0, color="#6b7280", linewidth=1.0, alpha=0.75)
    ax.grid(True, alpha=0.22, linestyle="--")

    style = {
        "reference": {"marker": "D", "color": "#d97706", "size": 135, "label": "Stressed reference"},
        "selected": {"marker": "*", "color": "#0f766e", "size": 310, "label": "Selected"},
        "pass": {"marker": "o", "color": "#2c6db2", "size": 145, "label": "Guardrail pass"},
        "fail": {"marker": "o", "color": "#9aa8b8", "size": 130, "label": "Guardrail fail"},
    }

    for p in points:
        if p["role"] == "reference":
            key = "reference"
        elif p["selected"]:
            key = "selected"
        elif p["guardrails_passed"]:
            key = "pass"
        else:
            key = "fail"
        st = style[key]
        ax.scatter(p["x"], p["y"], s=st["size"], marker=st["marker"], color=st["color"], edgecolor="#10233d", linewidth=0.9, zorder=5 if key == "selected" else 4)

    offsets = {
        "synthetic_stressed_reference_v2": (12, -24),
        "responsive_management_pathway": (12, 10),
        "procedural_justice_pathway": (12, -22),
        "capital_deepening_pathway": (16, -42),
        "direct_staffing_pathway": (-118, 12),
    }
    for p in points:
        label = _pretty_label(p["label"])
        if p["selected"]:
            label += "\n(selected)"
        dx, dy = offsets.get(p["label"], (10, 8))
        ax.annotate(
            label,
            (p["x"], p["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold" if p["selected"] else "normal",
            ha="right" if p["label"] == "direct_staffing_pathway" else "left",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.88},
        )

    ax.set_xscale("symlog", linthresh=75)
    ax.set_xlim(-70, 2600)
    ax.set_xticks([-50, -10, 0, 10, 100, 1000])
    ax.set_xticklabels(["-50", "-10", "0", "10", "100", "1,000"])
    ax.set_title("Development-stage tradeoff vs stressed reference", fontsize=17, pad=14, fontweight="bold")
    ax.set_xlabel("Service-harm change vs reference (lower is better; symmetric-log scale)", fontsize=13)
    ax.set_ylabel("Staff-welfare change vs reference (higher is better)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ymin = min(p["y"] for p in points)
    ymax = max(p["y"] for p in points)
    ypad = max((ymax - ymin) * 0.22, 0.001)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.text(
        0.02, 0.97,
        "Preferred region: upper-left",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#064e3b",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#ecfdf5", "ec": "#a7f3d0", "alpha": 0.96},
    )
    ax.annotate(
        "",
        xy=(0.12, 0.84),
        xytext=(0.27, 0.70),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#0f766e"},
    )

    handles = [
        plt.Line2D([0], [0], marker=style["selected"]["marker"], color="w", markerfacecolor=style["selected"]["color"], markeredgecolor="#10233d", markersize=14, label="Selected"),
        plt.Line2D([0], [0], marker=style["pass"]["marker"], color="w", markerfacecolor=style["pass"]["color"], markeredgecolor="#10233d", markersize=10, label="Guardrail pass"),
        plt.Line2D([0], [0], marker=style["fail"]["marker"], color="w", markerfacecolor=style["fail"]["color"], markeredgecolor="#10233d", markersize=10, label="Guardrail fail"),
        plt.Line2D([0], [0], marker=style["reference"]["marker"], color="w", markerfacecolor=style["reference"]["color"], markeredgecolor="#10233d", markersize=10, label="Reference"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=True, framealpha=0.96)

    fig.text(
        0.012, 0.012,
        "Synthetic simulation evidence only. Delta values are computed from the public evidence bundle.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#5d6f88",
    )

    path = out_dir / "primary_welfare_vs_service_loss.png"
    _save(fig, path)
    return path
