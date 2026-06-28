"""Predeclared selection and immutable frozen holdout execution for current public release.

Development evidence is selected exactly once.  The resulting policy payload,
selection record, run_0-run_4 evidence, and scientific context are committed to
``selection_freeze.json``.  Later selection/holdout invocations are read-only:
any drift requires a new result root or campaign.  Every existing holdout arm is
routed through the normal experiment runner so its complete scientific contract
is verified before it can be reused without another provider call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from shachi.env.japan_policy_scientist import PolicyConfig, run_policy_experiment

from template_contract import BASELINE_POLICY

PRIMARY = "primary_staff_welfare_composite_post"
LOWER_IS_BETTER = (
    "cumulative_service_harm_points_post",
    "mean_critical_overdue_units_post",
    "terminal_liability_points",
)

FREEZE_SCHEMA_VERSION = 1
DEVELOPMENT_RUNS = tuple(f"run_{index}" for index in range(5))
SELECTION_RESULT_NAME = "selection_result.json"
SELECTION_FREEZE_NAME = "selection_freeze.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _development_artifact_hashes(run_root: Path) -> dict[str, dict[str, str]]:
    evidence: dict[str, dict[str, str]] = {}
    for run_name in DEVELOPMENT_RUNS:
        run_dir = run_root / run_name
        if not (run_dir / "complete.marker").is_file():
            raise FileNotFoundError(f"complete {run_name} is required before selection")
        evidence[run_name] = {
            filename: _sha256_file(run_dir / filename)
            for filename in ("run_manifest.json", "final_info.json", "complete.marker")
        }
    return evidence


def _scientific_context(run_root: Path) -> dict[str, Any]:
    template_dir = Path(__file__).resolve().parent
    config_path = template_dir / "japan_policy_scientist.yaml"
    prompt_path = template_dir / "prompt.json"
    contract_path = template_dir / "template_contract.py"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run0_manifest = json.loads(
        (run_root / "run_0" / "run_manifest.json").read_text(encoding="utf-8")
    )
    fingerprints = run0_manifest.get("fingerprints")
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    return {
        "package": config.get("package"),
        "configured_model": config.get("llm", {}).get("model"),
        "config_file_sha256": _sha256_file(config_path),
        "prompt_json_sha256": _sha256_file(prompt_path),
        "template_contract_sha256": _sha256_file(contract_path),
        "run_0_config_hash": run0_manifest.get("config_hash"),
        "run_0_model": run0_manifest.get("model"),
        "runtime_package_version": fingerprints.get("package_version"),
        "runtime_package_hash": fingerprints.get("package_hash"),
        "runtime_prompt_hash": fingerprints.get("prompt_hash"),
        "runtime_schema_hash": fingerprints.get("schema_hash"),
        "runtime_transition_hash": fingerprints.get("transition_hash"),
        "runtime_code_hash": fingerprints.get("code_hash"),
    }


def _selection_policy_hash(selection: dict[str, Any]) -> str | None:
    policy = selection.get("selected_policy")
    return None if policy is None else _canonical_hash(policy)


def verify_selection_freeze(run_root: Path) -> dict[str, Any]:
    """Verify every byte committed when the development choice was made."""

    run_root = run_root.resolve()
    selection_path = run_root / SELECTION_RESULT_NAME
    freeze_path = run_root / SELECTION_FREEZE_NAME
    if not freeze_path.is_file():
        raise FileNotFoundError(
            f"Missing immutable selection freeze: {freeze_path}. Run selection once before holdout."
        )
    if not selection_path.is_file():
        raise RuntimeError("selection_result.json was removed after selection freeze")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if int(freeze.get("freeze_schema_version", -1)) != FREEZE_SCHEMA_VERSION:
        raise RuntimeError("Unsupported or corrupted selection freeze schema")

    actual_selection_hash = _sha256_file(selection_path)
    if freeze.get("selection_result_sha256") != actual_selection_hash:
        raise RuntimeError(
            "selection_result.json changed after selection was frozen; use a new result root/campaign"
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for key in ("status", "selected_run", "selected_label"):
        if freeze.get(f"selection_{key}") != selection.get(key):
            raise RuntimeError(f"Frozen selection metadata mismatch for {key}")
    if freeze.get("selected_policy_sha256") != _selection_policy_hash(selection):
        raise RuntimeError("Selected policy payload changed after selection freeze")

    actual_development = _development_artifact_hashes(run_root)
    if freeze.get("development_artifacts") != actual_development:
        raise RuntimeError(
            "Development artifacts changed after selection freeze; holdout is forbidden. "
            "Start a new result root/campaign to make a new selection."
        )

    actual_context = _scientific_context(run_root)
    if freeze.get("scientific_context") != actual_context:
        raise RuntimeError(
            "Scientific config/model/prompt/schema/package context changed after selection freeze"
        )

    current_campaign = os.environ.get("POLICYLAB_CAMPAIGN_ID")
    frozen_campaign = freeze.get("campaign_id")
    if current_campaign and frozen_campaign != current_campaign:
        raise RuntimeError(
            f"Selection belongs to campaign {frozen_campaign!r}, not {current_campaign!r}"
        )
    return selection


def _load_means(path: Path) -> dict[str, float | None]:
    payload = json.loads((path / "final_info.json").read_text(encoding="utf-8"))
    return {
        key: float(value) if isinstance(value, (int, float)) else None
        for key, value in payload["policy_lab"]["means"].items()
    }


def _metric(means: dict[str, float | None], key: str) -> float | None:
    value = means.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _upper_guardrail(baseline: float, *, relative: float, absolute: float) -> float:
    return baseline * (1.0 + relative) + absolute


def evaluate_guardrails(
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for key, absolute in (
        ("cumulative_service_harm_points_post", 0.05),
        ("mean_critical_overdue_units_post", 0.01),
        ("terminal_liability_points", 0.10),
        ("mean_modeled_work_strain_pressure_post_person_month", 0.03),
    ):
        base = _metric(baseline, key)
        value = _metric(candidate, key)
        passed = base is not None and value is not None and value <= _upper_guardrail(
            base, relative=0.05, absolute=absolute
        )
        rows.append(
            {
                "metric": key,
                "direction": "lower_is_better",
                "baseline": base,
                "candidate": value,
                "threshold": (
                    None
                    if base is None
                    else _upper_guardrail(base, relative=0.05, absolute=absolute)
                ),
                "passed": passed,
            }
        )

    key = "initial_high_family_involuntary_transfer_rate"
    base = _metric(baseline, key)
    value = _metric(candidate, key)
    if base is None:
        passed = value is None or (value is not None and value <= 0.02)
        threshold = 0.02
    else:
        passed = value is not None and value <= base + 0.02
        threshold = base + 0.02
    rows.append(
        {
            "metric": key,
            "direction": "lower_is_better",
            "baseline": base,
            "candidate": value,
            "threshold": threshold,
            "passed": passed,
        }
    )
    return all(bool(row["passed"]) for row in rows), rows


def _compute_candidate_selection(run_root: Path) -> dict[str, Any]:
    run0 = run_root / "run_0"
    if not (run0 / "complete.marker").exists():
        raise FileNotFoundError("complete run_0 is required")
    baseline = _load_means(run0)
    baseline_primary = _metric(baseline, PRIMARY)
    if baseline_primary is None:
        raise ValueError(f"Baseline primary endpoint {PRIMARY!r} is undefined")

    candidates: list[dict[str, Any]] = []
    for index in range(1, 5):
        run = run_root / f"run_{index}"
        if not (run / "complete.marker").exists():
            raise FileNotFoundError(f"complete {run.name} is required before selection")
        means = _load_means(run)
        passed, guardrails = evaluate_guardrails(baseline, means)
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        primary = _metric(means, PRIMARY)
        improves_baseline = primary is not None and primary > baseline_primary + 1e-12
        candidates.append(
            {
                "run": run.name,
                "label": manifest["intervention_policy"]["label"],
                "primary_endpoint": primary,
                "baseline_primary_endpoint": baseline_primary,
                "primary_delta_vs_baseline": (
                    None if primary is None else primary - baseline_primary
                ),
                "strictly_improves_baseline": improves_baseline,
                "guardrails_passed": passed,
                "guardrails": guardrails,
                "service_harm_points": _metric(
                    means, "cumulative_service_harm_points_post"
                ),
                "policy_cost_points": _metric(means, "policy_implementation_cost_points"),
                "policy": manifest["intervention_policy"],
            }
        )

    eligible = [
        row
        for row in candidates
        if row["guardrails_passed"] and row["strictly_improves_baseline"]
    ]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                -float(row["primary_endpoint"]),
                float(
                    row["service_harm_points"]
                    if row["service_harm_points"] is not None
                    else float("inf")
                ),
                float(
                    row["policy_cost_points"]
                    if row["policy_cost_points"] is not None
                    else float("inf")
                ),
                row["run"],
            ),
        )[0]

    return {
        "selection_frozen": True,
        "development_scenario_only": True,
        "primary_endpoint": PRIMARY,
        "primary_endpoint_definition": (
            "0.75 * mechanical_welfare_anchor_post + 0.25 * "
            "sealed_survey_welfare_composite_post"
        ),
        "survey_firewall": (
            "Self-report values are contemporaneous evaluation outputs only; they are excluded "
            "from future prompts, memory, management priority, exits, staffing, and transitions."
        ),
        "guardrail_rule": (
            "No more than 5% plus a small absolute tolerance above run_0 for weighted "
            "service-harm points, critical overdue work, terminal liability, and modeled strain pressure; "
            "high-family involuntary-transfer rate no more than 2 percentage points "
            "above run_0. Formal event-ID validity is a run-validity invariant, not a policy guardrail."
        ),
        "baseline_primary_endpoint": baseline_primary,
        "benefit_rule": (
            "A candidate is eligible only if it passes every guardrail and its primary endpoint "
            "is strictly greater than run_0 by more than 1e-12. Equal or worse candidates are "
            "reported as a negative result and cannot enter holdout."
        ),
        "selected_run": None if selected is None else selected["run"],
        "selected_label": None if selected is None else selected["label"],
        "selected_policy": None if selected is None else selected["policy"],
        "status": (
            "selected"
            if selected is not None
            else "no_beneficial_guardrail_passing_candidate"
        ),
        "candidates": candidates,
        "exploratory_metrics_excluded_from_selection": [
            "mean_exploratory_ebpm_interest_post",
            "mean_exploratory_dx_improvement_interest_post",
        ],
    }


def select_candidate(run_root: Path) -> dict[str, Any]:
    """Create the development selection exactly once, then verify it forever."""

    run_root = run_root.resolve()
    freeze_path = run_root / SELECTION_FREEZE_NAME
    if freeze_path.exists():
        return verify_selection_freeze(run_root)

    result = _compute_candidate_selection(run_root)
    result_path = run_root / SELECTION_RESULT_NAME
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing != result:
            raise RuntimeError(
                "An unfrozen selection_result.json disagrees with current development artifacts; "
                "use a new result root rather than overwriting selection evidence"
            )
    else:
        _write_json(result_path, result)

    freeze = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": os.environ.get(
            "POLICYLAB_CAMPAIGN_ID", "not-set-at-selection"
        ),
        "selection_result_sha256": _sha256_file(result_path),
        "selection_status": result.get("status"),
        "selection_selected_run": result.get("selected_run"),
        "selection_selected_label": result.get("selected_label"),
        "selected_policy_sha256": _selection_policy_hash(result),
        "development_artifacts": _development_artifact_hashes(run_root),
        "scientific_context": _scientific_context(run_root),
        "reselection_forbidden": True,
        "new_selection_requires_new_result_root_or_campaign": True,
    }
    try:
        _write_json(freeze_path, freeze, exclusive=True)
    except FileExistsError:
        # A concurrent process won the exclusive create.  Never overwrite it;
        # verify that it commits exactly the same evidence instead.
        pass
    return verify_selection_freeze(run_root)


@contextmanager
def temporary_env(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_holdout_manifest(
    run_root: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    run_root = run_root.resolve()
    frozen_selection = verify_selection_freeze(run_root)
    if selection != frozen_selection:
        raise RuntimeError("Caller selection does not match the immutable selection freeze")

    config = yaml.safe_load(
        (Path(__file__).with_name("japan_policy_scientist.yaml")).read_text(
            encoding="utf-8"
        )
    )
    cells = [
        {"scenario": scenario, "seed": int(seed)}
        for scenario in config["experiment"]["holdout_scenarios"]
        for seed in config["experiment"]["holdout_seeds"]
    ]
    freeze_path = run_root / SELECTION_FREEZE_NAME
    manifest = {
        "selection_result": str((run_root / SELECTION_RESULT_NAME).resolve()),
        "selection_freeze": str(freeze_path.resolve()),
        "selection_result_sha256": _sha256_file(
            run_root / SELECTION_RESULT_NAME
        ),
        "selection_freeze_sha256": _sha256_file(freeze_path),
        "selection_frozen_before_holdout": True,
        "selected_run": selection["selected_run"],
        "selected_label": selection["selected_label"],
        "cells": cells,
        "arms_per_cell": ["run_0", "selected_policy"],
        "reselection_after_holdout_forbidden": True,
        "holdout_disabled_by_config": not bool(cells),
        "status": (
            "disabled_no_holdout_configured"
            if not cells
            else "ready" if selection["selected_run"] else "blocked_no_selected_candidate"
        ),
    }
    path = run_root / "holdout_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "Existing holdout_manifest.json disagrees with the frozen selection"
            )
    else:
        _write_json(path, manifest)
    return manifest


def _reuse_or_run(
    *,
    output_dir: Path,
    baseline_policy: PolicyConfig,
    intervention_policy: PolicyConfig,
    config_path: Path,
) -> None:
    """Route every holdout arm through the full runner contract.

    The runner validates config, model, prompt/schema/transition fingerprints,
    seed, scenario, policy payload, and contract hash before returning an
    existing complete run.  A valid complete arm is reused without a provider
    call; stale artifacts fail closed.
    """

    run_policy_experiment(
        output_dir=output_dir,
        baseline_policy=baseline_policy,
        intervention_policy=intervention_policy,
        config_path=config_path,
    )


def summarize_holdout(
    run_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    metrics = (
        PRIMARY,
        "mechanical_welfare_anchor_post",
        "sealed_survey_welfare_composite_post",
        "mean_modeled_work_strain_post_person_month",
        "mean_turnover_intent_post_person_month",
        "cumulative_service_harm_points_post",
        "mean_critical_overdue_units_post",
        "terminal_liability_points",
        "initial_high_family_involuntary_transfer_rate",
        "formal_event_id_reference_validity_rate",
        "mean_exploratory_ebpm_interest_post",
        "mean_exploratory_dx_improvement_interest_post",
    )
    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        cell_root = (
            run_root / "holdout" / str(cell["scenario"]) / f"seed_{cell['seed']}"
        )
        baseline = _load_means(cell_root / "run_0")
        selected = _load_means(cell_root / "selected_policy")
        values: dict[str, Any] = {}
        for metric in metrics:
            base = _metric(baseline, metric)
            value = _metric(selected, metric)
            values[metric] = {
                "run_0": base,
                "selected_policy": value,
                "delta_selected_minus_run_0": (
                    None if base is None or value is None else value - base
                ),
            }
        guardrails_passed, guardrails = evaluate_guardrails(baseline, selected)
        rows.append(
            {
                "scenario": cell["scenario"],
                "seed": cell["seed"],
                "guardrails_passed": guardrails_passed,
                "guardrails": guardrails,
                "metrics": values,
            }
        )

    aggregate: dict[str, Any] = {}
    for metric in metrics:
        deltas = [
            row["metrics"][metric]["delta_selected_minus_run_0"]
            for row in rows
            if row["metrics"][metric]["delta_selected_minus_run_0"] is not None
        ]
        aggregate[metric] = {
            "mean_delta_selected_minus_run_0": (
                None
                if not deltas
                else sum(float(value) for value in deltas) / len(deltas)
            ),
            "cells_with_defined_delta": len(deltas),
        }

    summary = {
        "selection_frozen": True,
        "reselection_performed": False,
        "selected_run": manifest["selected_run"],
        "selected_label": manifest["selected_label"],
        "all_holdout_guardrails_passed": all(
            row["guardrails_passed"] for row in rows
        ),
        "cells": rows,
        "aggregate": aggregate,
        "claim_boundary": (
            "Synthetic multi-scenario robustness only; these cells do not estimate "
            "effects in real Japanese ministries."
        ),
    }
    _write_json(run_root / "holdout_summary.json", summary)
    return summary


def run_holdout(
    run_root: Path,
    selection: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    run_root = run_root.resolve()
    frozen_selection = verify_selection_freeze(run_root)
    if selection != frozen_selection:
        raise RuntimeError("Holdout caller selection does not match the immutable freeze")
    if not selection.get("selected_policy"):
        raise RuntimeError("No selected policy; holdout execution is blocked")

    freeze_path = run_root / SELECTION_FREEZE_NAME
    frozen_selection_hash = _sha256_file(run_root / SELECTION_RESULT_NAME)
    frozen_freeze_hash = _sha256_file(freeze_path)
    if manifest.get("selection_result_sha256") != frozen_selection_hash:
        raise RuntimeError("Holdout manifest selection hash does not match the freeze")
    if manifest.get("selection_freeze_sha256") != frozen_freeze_hash:
        raise RuntimeError("Holdout manifest freeze hash does not match selection_freeze.json")

    selected_policy = PolicyConfig.model_validate(selection["selected_policy"])
    config_path = Path(__file__).with_name("japan_policy_scientist.yaml")
    holdout_root = run_root / "holdout"
    for cell in manifest["cells"]:
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        cell_root = holdout_root / scenario / f"seed_{seed}"
        env = {
            "POLICYLAB_SCENARIO": scenario,
            "POLICYLAB_FIXED_SEED": str(seed),
            "POLICYLAB_HOLDOUT_MODE": "1",
        }
        with temporary_env(env):
            _reuse_or_run(
                output_dir=cell_root / "run_0",
                baseline_policy=BASELINE_POLICY,
                intervention_policy=BASELINE_POLICY,
                config_path=config_path,
            )
            _reuse_or_run(
                output_dir=cell_root / "selected_policy",
                baseline_policy=BASELINE_POLICY,
                intervention_policy=selected_policy,
                config_path=config_path,
            )
        _write_json(
            cell_root / "holdout_cell_manifest.json",
            {
                "scenario": scenario,
                "seed": seed,
                "selected_label": selection["selected_label"],
                "selection_result_sha256": frozen_selection_hash,
                "selection_freeze_sha256": frozen_freeze_hash,
                "arms": ["run_0", "selected_policy"],
            },
        )

    # Re-read all frozen development evidence and scientific context after the
    # last paid cell.  Holdout outputs may be added, but development evidence may
    # never change in response to holdout results.
    verify_selection_freeze(run_root)
    if _sha256_file(run_root / SELECTION_RESULT_NAME) != frozen_selection_hash:
        raise RuntimeError("Selection result changed during holdout execution")
    if _sha256_file(freeze_path) != frozen_freeze_hash:
        raise RuntimeError("Selection freeze changed during holdout execution")

    summary = summarize_holdout(run_root, manifest)
    marker = {
        **manifest,
        "status": "complete",
        "selection_result_sha256": frozen_selection_hash,
        "selection_freeze_sha256": frozen_freeze_hash,
        "holdout_summary": str((run_root / "holdout_summary.json").resolve()),
    }
    _write_json(run_root / "holdout_complete.json", marker)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-holdout", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    selection = select_candidate(run_root)
    manifest = build_holdout_manifest(run_root, selection)
    holdout_status = manifest["status"]
    if args.run_holdout:
        run_holdout(run_root, selection, manifest)
        holdout_status = "complete"
    print(
        json.dumps(
            {"selection": selection["status"], "holdout": holdout_status}, indent=2
        )
    )


if __name__ == "__main__":
    main()
