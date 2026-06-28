"""Generate machine-verified development and post-selection holdout tables.

The LaTeX tables produced here are generated from the same JSON artifacts that are
listed in ``verified_claims.json``.  Holdout deltas are recomputed from the two arm
``final_info.json`` files rather than trusted from ``holdout_summary.json`` so that
claim verification covers the numbers displayed in the final paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEVELOPMENT_METRICS = (
    "primary_staff_welfare_composite_post",
    "mechanical_welfare_anchor_post",
    "sealed_survey_welfare_composite_post",
    "mean_modeled_work_strain_post_person_month",
    "mean_modeled_work_strain_pressure_post_person_month",
    "p95_modeled_work_strain_pressure_post_person_month",
    "share_person_months_strain_pressure_ge_1_post",
    "strain_welfare_transform_post",
    "mean_turnover_intent_post_person_month",
    "share_relative_effort_gt_160_post",
    "cumulative_service_harm_points_post",
    "mean_critical_overdue_units_post",
    "terminal_liability_points",
    "initial_cohort_cumulative_exit_rate",
    "initial_high_family_involuntary_transfer_rate",
    "policy_implementation_cost_points",
    "management_outside_docket_deferred_count",
    "management_identity_invalidated_count",
)
HOLDOUT_METRICS = (
    "primary_staff_welfare_composite_post",
    "mechanical_welfare_anchor_post",
    "sealed_survey_welfare_composite_post",
    "mean_modeled_work_strain_pressure_post_person_month",
    "cumulative_service_harm_points_post",
    "mean_critical_overdue_units_post",
    "terminal_liability_points",
)
HOLDOUT_HEADERS = (
    "Welfare",
    "Mechanical",
    "Survey",
    "Strain pressure",
    "Service harm",
    "Critical overdue",
    "Terminal liability",
)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _means(path: Path) -> dict[str, Any]:
    return _read_json(path)["policy_lab"]["means"]


def _fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.4f}"


def _mean_claim(
    *,
    run_root: Path,
    source: Path,
    scope: str,
    run: str,
    metric: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "claim_type": "metric_mean",
        "scope": scope,
        "source_relative_path": str(source.relative_to(run_root)),
        "run": run,
        "metric": metric,
        "value": value,
        "tolerance": 1e-9,
    }


def _delta_claim(
    *,
    run_root: Path,
    baseline_source: Path,
    selected_source: Path,
    run: str,
    metric: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "claim_type": "metric_delta_selected_minus_run_0",
        "scope": "holdout_delta",
        "baseline_source_relative_path": str(baseline_source.relative_to(run_root)),
        "selected_source_relative_path": str(selected_source.relative_to(run_root)),
        "run": run,
        "metric": metric,
        "value": value,
        "tolerance": 1e-9,
    }


def _require_numeric_delta(
    selected_value: Any,
    baseline_value: Any,
    *,
    scenario: str,
    seed: str,
    metric: str,
) -> float:
    if selected_value is None or baseline_value is None:
        raise RuntimeError(
            f"Holdout metric {metric} is missing for {scenario}/seed_{seed}; "
            "delta table requires both arms"
        )
    return float(selected_value) - float(baseline_value)


def _check_summary_delta(row: dict[str, Any], metric: str, recomputed: float) -> None:
    metric_summary = row.get("metrics", {}).get(metric)
    if metric_summary is None:
        raise RuntimeError(f"holdout_summary.json lacks metric {metric}")
    recorded = metric_summary.get("delta_selected_minus_run_0")
    if recorded is None or abs(float(recorded) - recomputed) > 1e-9:
        raise RuntimeError(
            "holdout_summary.json drift for "
            f"{row.get('scenario')}/seed_{row.get('seed')}/{metric}: "
            f"summary={recorded!r}, recomputed={recomputed!r}"
        )


def generate(run_root: Path) -> tuple[Path, Path]:
    run_root = run_root.resolve()
    selection_path = run_root / "selection_result.json"
    if not selection_path.is_file():
        raise FileNotFoundError("selection_result.json is required before verified tables")
    selection = _read_json(selection_path)

    run_dirs = [run_root / f"run_{i}" for i in range(5)]
    for run in run_dirs:
        if not (run / "complete.marker").is_file():
            raise RuntimeError(f"Incomplete development artifact: {run.name}")

    claims: list[dict[str, Any]] = []
    labels: list[tuple[str, str]] = []
    values: dict[str, dict[str, Any]] = {}
    for run in run_dirs:
        manifest = _read_json(run / "run_manifest.json")
        means_path = run / "final_info.json"
        means = _means(means_path)
        labels.append((run.name, manifest["intervention_policy"]["label"]))
        values[run.name] = means
        for metric in DEVELOPMENT_METRICS:
            claims.append(
                _mean_claim(
                    run_root=run_root,
                    source=means_path,
                    scope="development",
                    run=run.name,
                    metric=metric,
                    value=means.get(metric),
                )
            )

    columns = "l" + "r" * len(run_dirs)
    dev_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Development results inserted directly from audited run artifacts. The simulation arm is the treatment unit; employee rows and person-months are not independent replications.}",
        r"\label{tab:verified-development-results}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Metric & " + " & ".join(latex_escape(name) for name, _ in labels) + r" \\",
        r"\midrule",
    ]
    for metric in DEVELOPMENT_METRICS:
        dev_lines.append(
            " & ".join(
                [latex_escape(metric)]
                + [_fmt(values[run.name].get(metric)) for run in run_dirs]
            )
            + r" \\")
    dev_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    holdout_lines: list[str] = []
    selected = selection.get("selected_policy")
    holdout_complete = run_root / "holdout_complete.json"
    if selected is None:
        holdout_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Post-selection robustness status.}",
            r"\label{tab:verified-holdout-results}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Status & No candidate strictly improved on the reference while passing all guardrails. \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    else:
        if not holdout_complete.is_file():
            raise FileNotFoundError(
                "A selected candidate requires holdout_complete.json before final evidence generation"
            )
        summary = _read_json(run_root / "holdout_summary.json")
        holdout_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Frozen post-selection robustness cells. Entries are selected-policy minus reference deltas recomputed from the two arm artifacts in each previously unused synthetic seed cell; no reselection is permitted.}",
            r"\label{tab:verified-holdout-results}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            "Scenario & Seed & " + " & ".join(HOLDOUT_HEADERS) + r" \\",
            r"\midrule",
        ]
        for row in summary["cells"]:
            scenario = str(row["scenario"])
            seed = str(row["seed"])
            cell_root = run_root / "holdout" / scenario / f"seed_{seed}"
            baseline_source = cell_root / "run_0" / "final_info.json"
            selected_source = cell_root / "selected_policy" / "final_info.json"
            baseline_means = _means(baseline_source)
            selected_means = _means(selected_source)
            deltas: list[float] = []
            for arm, source, arm_means in (
                ("run_0", baseline_source, baseline_means),
                ("selected_policy", selected_source, selected_means),
            ):
                for metric in HOLDOUT_METRICS:
                    claims.append(
                        _mean_claim(
                            run_root=run_root,
                            source=source,
                            scope="holdout",
                            run=f"{scenario}/seed_{seed}/{arm}",
                            metric=metric,
                            value=arm_means.get(metric),
                        )
                    )
            for metric in HOLDOUT_METRICS:
                delta = _require_numeric_delta(
                    selected_means.get(metric),
                    baseline_means.get(metric),
                    scenario=scenario,
                    seed=seed,
                    metric=metric,
                )
                _check_summary_delta(row, metric, delta)
                deltas.append(delta)
                claims.append(
                    _delta_claim(
                        run_root=run_root,
                        baseline_source=baseline_source,
                        selected_source=selected_source,
                        run=f"{scenario}/seed_{seed}",
                        metric=metric,
                        value=delta,
                    )
                )
            holdout_lines.append(
                " & ".join(
                    [latex_escape(scenario), seed] + [_fmt(value) for value in deltas]
                )
                + r" \\")
        holdout_lines.extend(
            [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        )

    combined = run_root / "verified_results.tex"
    combined.write_text(
        "\n".join(dev_lines + [""] + holdout_lines) + "\n", encoding="utf-8"
    )
    claims_path = run_root / "verified_claims.json"
    claims_path.write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "selection_status": selection.get("status"),
                "selected_run": selection.get("selected_run"),
                "holdout_included": bool(selected is not None),
                "claims": claims,
                "scope": (
                    "Machine-readable numerical provenance for development and frozen holdout artifacts. "
                    "Holdout deltas are recomputed from both arm artifacts. It does not validate causal interpretation."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return combined, claims_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Exact result root; defaults to the script directory",
    )
    args = parser.parse_args()
    tex, manifest = generate(args.run_root)
    print(tex)
    print(manifest)


if __name__ == "__main__":
    main()
