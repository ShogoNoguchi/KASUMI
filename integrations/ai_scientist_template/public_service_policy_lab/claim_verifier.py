"""Verify machine-readable numerical claims against exact development/holdout sources."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _safe_source(run_root: Path, source_relative_path: str) -> Path:
    source = (run_root / source_relative_path).resolve()
    try:
        source.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Claim source escapes run root: {source}") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _load_means_from_source(run_root: Path, source_relative_path: str) -> dict[str, Any]:
    source = _safe_source(run_root, source_relative_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload["policy_lab"]["means"]


def _compare(claimed: Any, observed: Any, tolerance: float) -> tuple[bool, float | None]:
    if claimed is None or observed is None:
        return claimed is None and observed is None, None
    if isinstance(claimed, (int, float)) and isinstance(observed, (int, float)):
        absolute_error = abs(float(claimed) - float(observed))
        return math.isfinite(absolute_error) and absolute_error <= tolerance, absolute_error
    return False, None


def _metric_mean_row(
    *,
    run_root: Path,
    cache: dict[str, dict[str, Any]],
    index: int,
    claim: dict[str, Any],
) -> dict[str, Any]:
    source_relative = str(
        claim.get("source_relative_path") or f"{claim['run']}/final_info.json"
    )
    metric = str(claim["metric"])
    claimed = claim.get("value")
    tolerance = float(claim.get("tolerance", 1e-9))
    means = cache.setdefault(source_relative, _load_means_from_source(run_root, source_relative))
    observed = means.get(metric)
    passed, absolute_error = _compare(claimed, observed, tolerance)
    return {
        "claim_index": index,
        "claim_type": "metric_mean",
        "scope": claim.get("scope", "development"),
        "run": str(claim.get("run", "")),
        "metric": metric,
        "claimed": claimed,
        "observed": observed,
        "tolerance": tolerance,
        "absolute_error": absolute_error,
        "passed": passed,
        "source_relative_path": source_relative,
        "source": str((run_root / source_relative).resolve()),
    }


def _metric_delta_row(
    *,
    run_root: Path,
    cache: dict[str, dict[str, Any]],
    index: int,
    claim: dict[str, Any],
) -> dict[str, Any]:
    baseline_relative = str(claim["baseline_source_relative_path"])
    selected_relative = str(claim["selected_source_relative_path"])
    metric = str(claim["metric"])
    claimed = claim.get("value")
    tolerance = float(claim.get("tolerance", 1e-9))
    baseline_means = cache.setdefault(
        baseline_relative, _load_means_from_source(run_root, baseline_relative)
    )
    selected_means = cache.setdefault(
        selected_relative, _load_means_from_source(run_root, selected_relative)
    )
    baseline_value = baseline_means.get(metric)
    selected_value = selected_means.get(metric)
    observed = None
    if baseline_value is not None and selected_value is not None:
        observed = float(selected_value) - float(baseline_value)
    passed, absolute_error = _compare(claimed, observed, tolerance)
    return {
        "claim_index": index,
        "claim_type": "metric_delta_selected_minus_run_0",
        "scope": claim.get("scope", "holdout_delta"),
        "run": str(claim.get("run", "")),
        "metric": metric,
        "claimed": claimed,
        "observed": observed,
        "baseline_value": baseline_value,
        "selected_value": selected_value,
        "tolerance": tolerance,
        "absolute_error": absolute_error,
        "passed": passed,
        "baseline_source_relative_path": baseline_relative,
        "selected_source_relative_path": selected_relative,
        "baseline_source": str((run_root / baseline_relative).resolve()),
        "selected_source": str((run_root / selected_relative).resolve()),
    }


def verify_claims(run_root: Path, claims: dict[str, Any]) -> dict[str, Any]:
    run_root = run_root.resolve()
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for index, claim in enumerate(claims.get("claims", []), start=1):
        claim_type = str(claim.get("claim_type", "metric_mean"))
        if claim_type == "metric_mean":
            row = _metric_mean_row(
                run_root=run_root, cache=cache, index=index, claim=claim
            )
        elif claim_type == "metric_delta_selected_minus_run_0":
            row = _metric_delta_row(
                run_root=run_root, cache=cache, index=index, claim=claim
            )
        else:
            row = {
                "claim_index": index,
                "claim_type": claim_type,
                "scope": claim.get("scope", "unknown"),
                "run": str(claim.get("run", "")),
                "metric": str(claim.get("metric", "")),
                "claimed": claim.get("value"),
                "observed": None,
                "tolerance": float(claim.get("tolerance", 1e-9)),
                "absolute_error": None,
                "passed": False,
                "error": f"Unknown claim_type: {claim_type}",
            }
        rows.append(row)
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "claim_count": len(rows),
        "passed_claims": sum(int(row["passed"]) for row in rows),
        "development_claims": sum(row["scope"] == "development" for row in rows),
        "holdout_claims": sum(row["scope"] == "holdout" for row in rows),
        "holdout_delta_claims": sum(row["scope"] == "holdout_delta" for row in rows),
        "claims": rows,
        "scope": (
            "Numerical equality and file provenance for the declared claims only. "
            "Holdout table deltas are recomputed from both arm artifacts. It does not "
            "validate causal interpretation or unlisted manuscript prose."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    result = verify_claims(args.run_root, claims)
    output = args.output or args.run_root / "claim_verification.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
