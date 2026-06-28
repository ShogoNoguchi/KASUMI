"""Generate current public release mechanism, task-quality, service, policy-cost, staffing, and cohort figures."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(os.environ.get("POLICYLAB_RUN_ROOT", ROOT)).resolve()
FIGURES = RUN_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def save_figure(filename: str) -> None:
    audit_path = FIGURES / filename
    plt.savefig(audit_path, dpi=180)
    root_path = RUN_ROOT / filename
    if root_path != audit_path:
        shutil.copy2(audit_path, root_path)


def completed_runs() -> list[Path]:
    return sorted(
        path
        for path in RUN_ROOT.glob("run_[0-4]")
        if (path / "complete.marker").exists()
        and (path / "final_info.json").exists()
    )


def label(run: Path) -> str:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    return str(manifest["intervention_policy"]["label"])


def load_means(run: Path) -> dict[str, float | None]:
    payload = json.loads((run / "final_info.json").read_text(encoding="utf-8"))
    return {
        key: (float(value) if isinstance(value, (int, float)) else None)
        for key, value in payload["policy_lab"]["means"].items()
    }


def metric_value(row: dict[str, float | None], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def load_department(run: Path) -> dict[int, dict[str, float]]:
    by_month: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with (run / "department_monthly.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            month = int(row["month"])
            for key in (
                "active_headcount_end",
                "backlog_units",
                "workload_ratio",
                "completion_ratio",
                "effective_demand_served_ratio",
                "deferred_work_units",
                "outsourced_work_units",
                "recipient_coordination_cost_units",
                "service_harm_points",
                "critical_overdue_units",
                "quality_error_units",
                "rework_generated_units",
                "terminal_liability_points",
            ):
                by_month[month][key].append(float(row[key]))
    result: dict[int, dict[str, float]] = {}
    for month, columns in by_month.items():
        result[month] = {}
        for key, values in columns.items():
            result[month][key] = (
                sum(values)
                if key == "active_headcount_end"
                else sum(values) / max(len(values), 1)
            )
    return result


def plot_trajectories(runs: list[Path]) -> None:
    specifications = (
        ("active_headcount_end", "Total active headcount at month end", "headcount_trajectory.png"),
        ("backlog_units", "Mean backlog units per department", "backlog_trajectory.png"),
        ("workload_ratio", "Required work / available capacity", "workload_trajectory.png"),
        ("completion_ratio", "Mean department completion ratio", "completion_trajectory.png"),
        (
            "effective_demand_served_ratio",
            "Mean effective demand served ratio",
            "effective_demand_served_trajectory.png",
        ),
        ("service_harm_points", "Mean monthly service-harm points per department", "service_loss_trajectory.png"),
        ("critical_overdue_units", "Mean critical overdue units per department", "critical_overdue_trajectory.png"),
        ("quality_error_units", "Mean monthly quality-error units per department", "quality_error_trajectory.png"),
        ("rework_generated_units", "Mean monthly rework generated per department", "rework_trajectory.png"),
    )
    for metric, ylabel, filename in specifications:
        plt.figure(figsize=(9, 5))
        for run in runs:
            rows = load_department(run)
            months = sorted(rows)
            plt.plot(months, [rows[month][metric] for month in months], label=label(run))
        plt.axvline(13, linestyle="--", linewidth=1)
        plt.xlabel("Month")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        save_figure(filename)
        plt.close()


def plot_headline(runs: list[Path]) -> None:
    metrics = (
        "initial_cohort_cumulative_exit_rate",
        "mean_modeled_work_strain_post_person_month",
        "mean_department_completion_ratio_post",
        "mean_department_backlog_units_post",
    )
    labels = [label(run) for run in runs]
    for metric in metrics:
        values = [metric_value(load_means(run), metric) for run in runs]
        plt.figure(figsize=(9, 5))
        plt.bar(range(len(runs)), values)
        plt.xticks(range(len(runs)), labels, rotation=25, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        save_figure(f"headline_{metric}.png")
        plt.close()


def plot_equity(runs: list[Path]) -> None:
    labels = [label(run) for run in runs]
    specifications = (
        (
            "equity_family_constraint_gap.png",
            "Turnover intent by family-constraint group",
            "mean_turnover_intent_high_family_constraint_post",
            "mean_turnover_intent_lower_family_constraint_post",
            "High family constraint",
            "Lower family constraint",
        ),
        (
            "equity_junior_gap.png",
            "Turnover intent by career-stage group",
            "mean_turnover_intent_junior_post",
            "mean_turnover_intent_nonjunior_post",
            "Junior",
            "Non-junior",
        ),
    )
    width = 0.36
    x = list(range(len(runs)))
    for filename, title, first_key, second_key, first_label, second_label in specifications:
        means = [load_means(run) for run in runs]
        first = [metric_value(row, first_key) for row in means]
        second = [metric_value(row, second_key) for row in means]
        plt.figure(figsize=(9, 5))
        plt.bar([index - width / 2 for index in x], first, width=width, label=first_label)
        plt.bar([index + width / 2 for index in x], second, width=width, label=second_label)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Mean normalized turnover intent")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        save_figure(filename)
        plt.close()


def plot_service_ledgers(runs: list[Path]) -> None:
    labels = [label(run) for run in runs]
    keys = (
        "cumulative_deferred_work_units_post",
        "cumulative_outsourced_work_units_post",
        "cumulative_recipient_coordination_cost_units_post",
    )
    width = 0.24
    x = list(range(len(runs)))
    means = [load_means(run) for run in runs]
    plt.figure(figsize=(10, 5))
    for offset, key in enumerate(keys):
        values = [metric_value(row, key) for row in means]
        positions = [index + (offset - 1) * width for index in x]
        plt.bar(positions, values, width=width, label=key)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Cumulative work units")
    plt.legend(fontsize=7)
    plt.tight_layout()
    save_figure("service_ledger_comparison.png")
    plt.close()


def plot_behavior(runs: list[Path]) -> None:
    labels = [label(run) for run in runs]
    keys = (
        "work_response_share__work_overtime",
        "work_response_share__request_support",
        "health_protecting_response_share_post",
        "career_action_share__request_transfer",
        "career_action_share__explore_external_exit",
    )
    width = 0.15
    x = list(range(len(runs)))
    means = [load_means(run) for run in runs]
    plt.figure(figsize=(11, 5))
    for offset, key in enumerate(keys):
        values = [metric_value(row, key) for row in means]
        positions = [index + (offset - 2) * width for index in x]
        plt.bar(positions, values, width=width, label=key)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Post-intervention share")
    plt.legend(fontsize=7)
    plt.tight_layout()
    save_figure("behavioral_action_comparison.png")
    plt.close()



def plot_policy_cost(runs: list[Path]) -> None:
    """Show scientist-visible synthetic implementation burden."""
    labels = [label(run) for run in runs]
    means = [load_means(run) for run in runs]
    costs = [metric_value(row, "policy_implementation_cost_points") for row in means]
    budgets = [metric_value(row, "policy_budget_max_points") for row in means]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(runs)), costs)
    if budgets:
        plt.axhline(max(budgets), linestyle="--", linewidth=1, label="Fixed policy budget")
        plt.legend()
    plt.xticks(range(len(runs)), labels, rotation=25, ha="right")
    plt.ylabel("Synthetic policy implementation cost points")
    plt.tight_layout()
    save_figure("policy_implementation_cost_points.png")
    plt.close()

    # Two transparent Pareto views. Lower cost/strain/backlog is preferable.
    for outcome_key, x_label, filename in (
        (
            "mean_modeled_work_strain_post_person_month",
            "Mean modeled work strain",
            "pareto_policy_cost_vs_work_strain.png",
        ),
        (
            "mean_department_backlog_units_post",
            "Mean department backlog units",
            "pareto_policy_cost_vs_backlog.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        x_values = [metric_value(row, outcome_key, math.nan) for row in means]
        plt.scatter(x_values, costs)
        for x_value, cost, run_label in zip(x_values, costs, labels, strict=True):
            if not math.isnan(x_value):
                plt.annotate(run_label, (x_value, cost), fontsize=7)
        plt.xlabel(x_label)
        plt.ylabel("Synthetic policy implementation cost points")
        plt.tight_layout()
        save_figure(filename)
        plt.close()


def plot_task_quality(runs: list[Path]) -> None:
    labels = [label(run) for run in runs]
    means = [load_means(run) for run in runs]
    x = list(range(len(runs)))

    work_unit_keys = (
        "mean_critical_overdue_units_post",
        "cumulative_quality_error_units_post",
        "cumulative_rework_generated_units_post",
    )
    width = 0.24
    plt.figure(figsize=(11, 5))
    for offset, key in enumerate(work_unit_keys):
        values = [metric_value(row, key) for row in means]
        positions = [index + (offset - 1) * width for index in x]
        plt.bar(positions, values, width=width, label=key)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Administrative work units")
    plt.legend(fontsize=7)
    plt.tight_layout()
    save_figure("task_work_unit_guardrails.png")
    plt.close()

    harm_point_keys = (
        "cumulative_service_harm_points_post",
        "terminal_liability_points",
    )
    width = 0.34
    plt.figure(figsize=(10, 5))
    for offset, key in enumerate(harm_point_keys):
        values = [metric_value(row, key) for row in means]
        positions = [index + (offset - 0.5) * width for index in x]
        plt.bar(positions, values, width=width, label=key)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Weighted public-harm points")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure("task_harm_point_guardrails.png")
    plt.close()


def plot_welfare_channels(runs: list[Path]) -> None:
    """Display the causal anchor, sealed survey, and predeclared composite."""
    labels = [label(run) for run in runs]
    means = [load_means(run) for run in runs]
    keys = (
        "mechanical_welfare_anchor_post",
        "sealed_survey_welfare_composite_post",
        "primary_staff_welfare_composite_post",
    )
    width = 0.24
    x = list(range(len(runs)))
    plt.figure(figsize=(11, 5))
    for offset, key in enumerate(keys):
        values = [metric_value(row, key) for row in means]
        positions = [index + (offset - 1) * width for index in x]
        plt.bar(positions, values, width=width, label=key)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Normalized welfare channel (higher is better)")
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=7)
    plt.tight_layout()
    save_figure("welfare_channel_decomposition.png")
    plt.close()


def plot_primary_vs_guardrail(runs: list[Path]) -> None:
    labels = [label(run) for run in runs]
    means = [load_means(run) for run in runs]
    x_values = [metric_value(row, "cumulative_service_harm_points_post", math.nan) for row in means]
    y_values = [metric_value(row, "primary_staff_welfare_composite_post", math.nan) for row in means]
    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values)
    for x_value, y_value, run_label in zip(x_values, y_values, labels, strict=True):
        if not math.isnan(x_value) and not math.isnan(y_value):
            plt.annotate(run_label, (x_value, y_value), fontsize=7)
    plt.xlabel("Cumulative service-harm points (lower is better)")
    plt.ylabel("Predeclared staff-welfare composite (higher is better)")
    plt.tight_layout()
    save_figure("primary_welfare_vs_service_loss.png")
    plt.close()

def write_summary(runs: list[Path]) -> None:
    rows = []
    all_keys: set[str] = set()
    for run in runs:
        means = load_means(run)
        all_keys.update(means)
        rows.append({"run": run.name, "label": label(run), **means})
    keys = ["run", "label", *sorted(all_keys)]
    with (RUN_ROOT / "summary_by_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    runs = completed_runs()
    if not runs:
        raise RuntimeError("No completed run_0 ... run_4 found")
    plot_trajectories(runs)
    plot_headline(runs)
    plot_equity(runs)
    plot_service_ledgers(runs)
    plot_behavior(runs)
    plot_policy_cost(runs)
    plot_task_quality(runs)
    plot_welfare_channels(runs)
    plot_primary_vs_guardrail(runs)
    write_summary(runs)
    # Final verified tables are generated only after immutable selection and,
    # when applicable, frozen holdout. Development plotting must not require a
    # selection_result.json that does not yet exist.
    if (RUN_ROOT / "selection_result.json").is_file():
        from verified_results import generate
        tex_path, claims_path = generate(RUN_ROOT)
        print(
            f"Wrote figures for {len(runs)} runs to {FIGURES}; "
            f"verified table={tex_path.name}; claims={claims_path.name}"
        )
    else:
        print(
            f"Wrote figures for {len(runs)} runs to {FIGURES}; "
            "verified tables deferred until selection/holdout"
        )


if __name__ == "__main__":
    main()
