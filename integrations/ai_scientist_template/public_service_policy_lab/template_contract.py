"""Human-fixed contract for Japan Policy Scientist current public release.

Only ``candidate_policy.json`` is scientist-editable. Executable experiment and
plot sources, model mechanics, prompts, schemas, transition equations, metrics,
cache identity, selection, and the four-run comparison contract remain
human-controlled.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from shachi.env.japan_policy_scientist import PolicyConfig
from shachi.env.japan_policy_scientist.experiment_contract import (
    validate_scientific_environment,
)
from shachi.env.japan_policy_scientist.policy_cost import (
    policy_cost_breakdown,
    validate_policy_budget,
)
from shachi.env.japan_policy_scientist.visibility import (
    is_operator_only_metric,
    operator_audit_dir,
    sanitize_scientist_payload,
)

BASELINE_POLICY = PolicyConfig.baseline()
REQUIRED_RUNS = ("run_0", "run_1", "run_2", "run_3", "run_4")
MAX_INTERVENTION_RUNS = 4
FIXED_CONTRACT: dict[str, Any] = {
    "seed": 20260619,
    "num_agents": 120,
    "months": 48,
    "warmup_months": 12,
    "intervention_start_month": 13,
    "hiring_interval_months": 6,
    "transfer_interval_months": 6,
    "baseline_label": "synthetic_stressed_reference_v2",
    "scenario": "reference_stressed",
    "manager_mode": "deterministic_priority",
    "default_intervention_runs": 4,
}
CANDIDATE_POLICY_FILENAME = "candidate_policy.json"
# Filled after the final current public release executable sources are frozen.
EXPECTED_EXPERIMENT_SHA256 = "298778a640690172ca13bb12585ed69a37bbf6105ce2e01603bc11231a374018"
EXPECTED_PLOT_SHA256 = "beacad4d36f20435e2f38cdd07283e4d085d026cd52b8ddba3dec3be2270ff6d"


def _template_dir_from(folder: str | Path) -> Path:
    candidate = Path(folder).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while not (candidate / "experiment.py").exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not (candidate / "experiment.py").exists():
        raise FileNotFoundError(f"Cannot locate experiment.py above {folder}")
    return candidate


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_source_hashes(template_dir: str | Path) -> dict[str, str]:
    template = _template_dir_from(template_dir)
    return {
        "experiment.py": file_sha256(template / "experiment.py"),
        "plot.py": file_sha256(template / "plot.py"),
    }


def validate_protected_sources(template_dir: str | Path) -> dict[str, str]:
    actual = protected_source_hashes(template_dir)
    expected = {
        "experiment.py": EXPECTED_EXPERIMENT_SHA256,
        "plot.py": EXPECTED_PLOT_SHA256,
    }
    mismatches = {
        name: {"actual": actual[name], "expected": expected[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise ValueError(
            "Protected executable source changed; only candidate_policy.json is editable: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_candidate_policy(path: str | Path, *, run_name: str) -> PolicyConfig:
    candidate_path = Path(path)
    if candidate_path.name != CANDIDATE_POLICY_FILENAME:
        raise ValueError(f"Candidate policy must be stored as {CANDIDATE_POLICY_FILENAME}")
    raw = candidate_path.read_text(encoding="utf-8")
    try:
        mapping = json.loads(raw, object_pairs_hook=_reject_duplicate_json_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Candidate policy is not strict JSON: {exc}") from exc
    if not isinstance(mapping, dict):
        raise TypeError("candidate_policy.json must contain exactly one JSON object")
    # Re-serialize and parse to ensure values are standard JSON data, not Python.
    json.loads(json.dumps(mapping, allow_nan=False))
    return policy_from_mapping(mapping, run_name=run_name)


def policy_from_mapping(mapping: dict[str, Any], *, run_name: str) -> PolicyConfig:
    unknown = set(mapping) - set(PolicyConfig.model_fields)
    if unknown:
        raise ValueError(f"Unknown PolicyConfig fields: {sorted(unknown)}")
    policy = PolicyConfig.model_validate(mapping)
    if run_name != "run_0" and policy.label == BASELINE_POLICY.label:
        raise ValueError("Intervention runs must use a non-baseline label")
    if len(policy.hypothesis.strip()) < 20:
        raise ValueError("Policy hypothesis must be explicit and at least 20 characters")
    return policy


def _policy_payload(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if key not in {"label", "hypothesis"}}


def changed_policy_fields(policy: PolicyConfig) -> list[str]:
    """Return latent policy knobs changed from the human baseline."""
    baseline = BASELINE_POLICY.model_dump()
    current = policy.model_dump()
    return sorted(
        key
        for key in _policy_payload(current)
        if abs(float(current[key]) - float(baseline[key])) > 1e-12
    )


def validate_mechanism_scope(policy: PolicyConfig, *, run_num: int) -> list[str]:
    """Protect interpretable contrasts with a mechanism-scope constraint.

    Runs 1-3 are focused mechanism tests. Run 4 may combine earlier evidence,
    but still cannot become an uninterpretable all-knob saturation package.
    """
    changed = changed_policy_fields(policy)
    if run_num == 0:
        if changed:
            raise ValueError(f"run_0 changed policy fields: {changed}")
        return changed
    max_changed = 8 if run_num in (1, 2, 3) else 12
    if not changed:
        raise ValueError("An intervention must change at least one policy mechanism")
    if len(changed) > max_changed:
        raise ValueError(
            f"run_{run_num} changes {len(changed)} knobs, exceeding the "
            f"mechanism-identifiability limit {max_changed}: {changed}"
        )
    return changed


IDEA_REQUIRED_FIELDS = {
    "Name",
    "Title",
    "ResearchQuestion",
    "Experiment",
    "Interventions",
    "Interestingness",
    "Feasibility",
    "Novelty",
}
IDEA_OPTIONAL_FIELDS = {"novel", "selection_rationale", "Score"}
INTERVENTION_REQUIRED_FIELDS = {
    "Run",
    "Mechanism",
    "ExpectedDirection",
    "AdverseEffectPrediction",
    "Policy",
}


def _policy_space_payload(template_dir: str | Path | None = None) -> dict[str, Any]:
    template = _template_dir_from(template_dir or Path(__file__).resolve().parent)
    path = template / "policy_space.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload.get("budget_max_points", -1.0)) != 35.0:
        raise ValueError("policy_space.json budget must remain 35.0 points")
    fields = payload.get("policy_fields")
    expected = set(_policy_payload(BASELINE_POLICY.model_dump()))
    if not isinstance(fields, dict) or set(fields) != expected:
        raise ValueError(
            "policy_space.json fields do not match PolicyConfig: "
            f"expected={sorted(expected)}, actual={sorted(fields) if isinstance(fields, dict) else fields}"
        )
    for name, contract in fields.items():
        if not isinstance(contract, dict):
            raise TypeError(f"policy_space field {name} must be an object")
        baseline = float(getattr(BASELINE_POLICY, name))
        if abs(float(contract.get("baseline", float("nan"))) - baseline) > 1e-12:
            raise ValueError(f"policy_space baseline drifted for {name}")
        for required in (
            "minimum",
            "maximum",
            "cost_rule",
            "meaning",
            "realization_path",
            "dormancy_condition",
        ):
            if required not in contract:
                raise ValueError(f"policy_space field {name} is missing {required}")
    return payload


def _compact_previous_ideas(previous_ideas: str) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for chunk in previous_ideas.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            idea = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        interventions = []
        for item in idea.get("Interventions", []):
            interventions.append(
                {
                    "Run": item.get("Run"),
                    "Mechanism": item.get("Mechanism"),
                    "ChangedKnobs": item.get("ChangedKnobs")
                    or sorted(
                        key
                        for key in (item.get("Policy") or {})
                        if key not in {"label", "hypothesis"}
                    ),
                }
            )
        compact.append(
            {
                "Name": idea.get("Name"),
                "Title": idea.get("Title"),
                "ResearchQuestion": idea.get("ResearchQuestion"),
                "Interventions": interventions,
            }
        )
    return compact[-16:]


def render_idea_prompt(
    *,
    task_description: str,
    previous_ideas: str,
    num_reflections: int,
) -> str:
    """Render a domain-specific idea prompt without code-editing jargon."""

    policy_space = _policy_space_payload()
    prior = _compact_previous_ideas(previous_ideas)
    schema = {
        "Name": "lowercase_identifier",
        "Title": "paper title",
        "ResearchQuestion": "one falsifiable question",
        "Experiment": "concise four-arm comparison plan",
        "Interventions": [
            {
                "Run": 1,
                "Mechanism": "causal mechanism inside the synthetic model",
                "ExpectedDirection": "directional prediction",
                "AdverseEffectPrediction": "specific failure or trade-off",
                "Policy": {
                    "label": "unique_label",
                    "hypothesis": "at least twenty characters",
                    "one_or_more_policy_fields": "exact numeric values",
                },
            }
        ],
        "Interestingness": "integer 1-10",
        "Feasibility": "integer 1-10",
        "Novelty": "integer 1-10",
    }
    return f"""{task_description}

The machine-readable policy space is:
```json
{json.dumps(policy_space, ensure_ascii=False, indent=2)}
```

Previously proposed programs, compacted to avoid repetition, are:
```json
{json.dumps(prior, ensure_ascii=False, indent=2)}
```

Propose one new research program that is materially different from the archive. It must contain exactly four intervention objects with Run values 1, 2, 3, and 4. Every Policy object must contain a unique label, an explicit hypothesis, and exact values for only the policy fields it changes; omitted fields remain at baseline. Runs 1-3 may change at most eight knobs, run 4 at most twelve. Every policy must pass the exact 35-point cost rule. Do not propose Python changes, new metrics, new datasets, a different population, or a different experimental design.

Return this schema exactly:
```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

Respond as:
THOUGHT:
<brief scientific rationale>

NEW IDEA JSON:
```json
<JSON>
```

You have {num_reflections} refinement rounds. Ratings must be cautious and realistic."""


def render_idea_reflection_prompt(*, current_round: int, num_reflections: int) -> str:
    return f"""Round {current_round}/{num_reflections}. Check the proposed program against policy_space.json: exactly four runs; valid fields and ranges; exact values; distinct mechanisms; explicit adverse effects; cost at most 35 points per run; at most eight changed knobs for runs 1-3 and twelve for run 4; no code or metric changes. Improve the scientific contrast without changing the fixed experiment. Return THOUGHT and NEW IDEA JSON in the same schema. If no change is needed, repeat the JSON exactly and write I am done before it."""


def render_novelty_context(*, task_description: str) -> str:
    policy_space = _policy_space_payload()
    return (
        task_description
        + "\n\nNovelty concerns the combination of mechanisms and the protected end-to-end research workflow, not merely new numerical values. "
        + "The executable model is fixed. Available policy fields are: "
        + ", ".join(sorted(policy_space["policy_fields"]))
        + "."
    )


def validate_idea(idea: Any) -> dict[str, Any]:
    """Validate and normalize an AI-generated four-arm policy program."""

    if not isinstance(idea, dict):
        raise TypeError("Idea must be one JSON object")
    missing = IDEA_REQUIRED_FIELDS - set(idea)
    unknown = set(idea) - IDEA_REQUIRED_FIELDS - IDEA_OPTIONAL_FIELDS
    if missing or unknown:
        raise ValueError(
            f"Idea schema mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    name = str(idea["Name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{2,63}", name):
        raise ValueError("Idea Name must be a 3-64 character lowercase identifier")
    for key in ("Title", "ResearchQuestion", "Experiment"):
        value = str(idea[key]).strip()
        if len(value) < 20:
            raise ValueError(f"{key} must be explicit and at least 20 characters")
    for key in ("Interestingness", "Feasibility", "Novelty"):
        value = idea[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric")
        if int(value) != float(value) or not 1 <= int(value) <= 10:
            raise ValueError(f"{key} must be an integer from 1 to 10")

    interventions = idea["Interventions"]
    if not isinstance(interventions, list) or len(interventions) != 4:
        raise ValueError("Interventions must contain exactly four objects")
    normalized: list[dict[str, Any]] = []
    labels: set[str] = set()
    payload_hashes: set[str] = set()
    for item in interventions:
        if not isinstance(item, dict):
            raise TypeError("Each intervention must be an object")
        if set(item) != INTERVENTION_REQUIRED_FIELDS:
            raise ValueError(
                "Intervention schema mismatch; expected "
                + repr(sorted(INTERVENTION_REQUIRED_FIELDS))
            )
        run_num = int(item["Run"])
        if run_num not in (1, 2, 3, 4):
            raise ValueError(f"Invalid intervention Run: {run_num}")
        for key in ("Mechanism", "ExpectedDirection", "AdverseEffectPrediction"):
            if len(str(item[key]).strip()) < 20:
                raise ValueError(f"Intervention {run_num} {key} is too short")
        mapping = item["Policy"]
        if not isinstance(mapping, dict):
            raise TypeError(f"Intervention {run_num} Policy must be an object")
        policy = policy_from_mapping(mapping, run_name=f"run_{run_num}")
        changed = validate_mechanism_scope(policy, run_num=run_num)
        cost = validate_policy_budget(
            policy, max_points=35.0, baseline=BASELINE_POLICY
        )
        label = policy.label
        if label in labels:
            raise ValueError(f"Duplicate intervention label: {label}")
        labels.add(label)
        payload_hash = hashlib.sha256(
            json.dumps(
                _policy_payload(policy.model_dump(mode="json")), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        if payload_hash in payload_hashes:
            raise ValueError("Two interventions have identical policy parameters")
        payload_hashes.add(payload_hash)
        normalized.append(
            {
                **item,
                "Run": run_num,
                "Policy": mapping,
                "ChangedKnobs": changed,
                "PolicyImplementationCostPoints": round(float(cost), 6),
            }
        )
    if sorted(item["Run"] for item in normalized) != [1, 2, 3, 4]:
        raise ValueError("Intervention Run values must be exactly 1, 2, 3, and 4")
    normalized.sort(key=lambda item: item["Run"])
    result = dict(idea)
    result["Interventions"] = normalized
    for key in ("Interestingness", "Feasibility", "Novelty"):
        result[key] = int(result[key])
    return result


def preflight_run(
    *,
    out_dir: str | Path,
    run_num: int,
    policy: PolicyConfig,
) -> dict[str, Any]:
    """Reject invalid or duplicate runs before any provider call."""
    out_dir = Path(out_dir)
    run_root = out_dir.parent.resolve()
    template_dir = _template_dir_from(run_root)
    validate_protected_sources(template_dir)
    if run_num < 0 or run_num > MAX_INTERVENTION_RUNS:
        raise ValueError("Only run_0 plus run_1..run_4 are permitted")
    if out_dir.name != f"run_{run_num}":
        raise ValueError(f"Run directory mismatch: {out_dir.name} != run_{run_num}")
    config = yaml.safe_load(
        (template_dir / "japan_policy_scientist.yaml").read_text(encoding="utf-8")
    )
    environment_contract = validate_scientific_environment(
        config, allow_holdout=False
    )
    configured_runs = int(config["experiment"]["default_intervention_runs"])
    if configured_runs != MAX_INTERVENTION_RUNS:
        raise ValueError(
            f"AI Scientist default intervention count drifted: {configured_runs} != 4"
        )
    changed_fields = validate_mechanism_scope(policy, run_num=run_num)
    policy_budget_max_points = float(config["policy_budget"]["max_points"])
    implementation_cost = validate_policy_budget(
        policy,
        max_points=policy_budget_max_points,
        baseline=BASELINE_POLICY,
    )
    implementation_breakdown = policy_cost_breakdown(policy, BASELINE_POLICY)
    if run_num == 0:
        if policy.model_dump() != BASELINE_POLICY.model_dump():
            raise ValueError("run_0 must use the human-authored baseline exactly")
    else:
        run0 = run_root / "run_0"
        if not (run0 / "complete.marker").exists():
            raise FileNotFoundError("A complete run_0 is required before intervention runs")
        current = policy.model_dump(mode="json")
        for prior_num in range(1, run_num):
            prior_manifest = run_root / f"run_{prior_num}" / "run_manifest.json"
            if not prior_manifest.exists():
                continue
            prior = json.loads(prior_manifest.read_text(encoding="utf-8"))[
                "intervention_policy"
            ]
            if prior.get("label") == current["label"]:
                raise ValueError(f"Policy label duplicates run_{prior_num}")
            if _policy_payload(prior) == _policy_payload(current):
                raise ValueError(f"Policy parameters duplicate run_{prior_num}")
    # Preserve failed attempts. If Aider changed the candidate after a
    # failure, archive the old run directory instead of deleting it. If the
    # policy is unchanged, leave the directory in place so the response cache
    # can resume the same contract.
    if out_dir.exists() and any(out_dir.iterdir()):
        existing_preflight_path = out_dir / "preflight.json"
        if (out_dir / "complete.marker").exists():
            if existing_preflight_path.exists():
                existing = json.loads(existing_preflight_path.read_text(encoding="utf-8"))
                if existing.get("policy") == policy.model_dump(mode="json"):
                    return existing
            raise FileExistsError(f"Completed run directory already exists: {out_dir}")
        if existing_preflight_path.exists():
            existing = json.loads(existing_preflight_path.read_text(encoding="utf-8"))
            if existing.get("policy") != policy.model_dump(mode="json"):
                archive_root = run_root / "failed_attempts" / out_dir.name
                archive_root.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                safe_label = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in str(existing.get("policy", {}).get("label", "unknown"))
                )[:80]
                destination = archive_root / f"attempt_{stamp}_{safe_label}"
                counter = 1
                while destination.exists():
                    counter += 1
                    destination = archive_root / f"attempt_{stamp}_{safe_label}_{counter}"
                audit = operator_audit_dir(out_dir)
                shutil.move(str(out_dir), str(destination))
                if audit.exists():
                    shutil.move(str(audit), str(destination / "operator_audit"))
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "status": "passed",
        "run_num": run_num,
        "policy": policy.model_dump(mode="json"),
        "changed_policy_fields": changed_fields,
        "resource_matched_scientific_discovery": True,
        "policy_implementation_cost_points": implementation_cost,
        "policy_budget_max_points": policy_budget_max_points,
        "policy_implementation_cost_breakdown": implementation_breakdown,
        "simulation_compute_accounting_hidden_from_scientist": True,
        "protected_source_sha256": protected_source_hashes(template_dir),
        "candidate_policy_sha256": file_sha256(template_dir / CANDIDATE_POLICY_FILENAME),
        "environment_contract_preflight": environment_contract,
        "checked_unix_time": time.time(),
        "provider_calls_before_preflight": 0,
    }
    (out_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return preflight


def scientist_visible_results(payload: Any) -> Any:
    """Sanitize baseline/run payloads before they are sent to The AI Scientist."""
    return sanitize_scientist_payload(payload)


def validate_run(folder: str | Path, run_num: int | None = None) -> dict[str, float | None]:
    folder = Path(folder)
    run_name = folder.name
    template_dir = _template_dir_from(folder.parent)
    validate_protected_sources(template_dir)
    if run_num is not None and run_name != f"run_{run_num}":
        raise ValueError(f"Run directory mismatch: {run_name} != run_{run_num}")
    required = [
        "preflight.json",
        "run_manifest.json",
        "final_info.json",
        "complete.marker",
        "department_monthly.csv",
        "population_profiles.jsonl",
        "monthly_agent_records.jsonl",
        "staffing_events.jsonl",
        "transfer_plan.jsonl",
        "management_outcomes.jsonl",
        "exposure_events.jsonl",
        "task_ledger.jsonl",
        "transfer_requests.jsonl",
        "policy_implementation_cost_breakdown.json",
        "initial_cohort_survival_curve.csv",
    ]
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {folder}: missing {missing}")
    audit = operator_audit_dir(folder)
    audit_required = [audit / "llm_usage.jsonl", audit / "operator_audit.json"]
    audit_missing = [str(path) for path in audit_required if not path.exists()]
    if audit_missing:
        raise FileNotFoundError(
            f"Incomplete operator audit for {folder}: missing {audit_missing}"
        )
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    expected = dict(FIXED_CONTRACT)
    if (
        os.environ.get("POLICYLAB_ALLOW_TEST_MODE") == "1"
        and os.environ.get("POLICYLAB_MOCK_LLM") == "1"
    ):
        expected["seed"] = int(os.environ.get("POLICYLAB_FIXED_SEED", expected["seed"]))
        expected["num_agents"] = int(os.environ.get("POLICYLAB_NUM_AGENTS", expected["num_agents"]))
        expected["months"] = int(os.environ.get("POLICYLAB_MONTHS", expected["months"]))
        expected["warmup_months"] = int(os.environ.get("POLICYLAB_WARMUP_MONTHS", expected["warmup_months"]))
        expected["intervention_start_month"] = int(
            os.environ.get(
                "POLICYLAB_INTERVENTION_START_MONTH",
                expected["intervention_start_month"],
            )
        )
    for key in (
        "seed",
        "num_agents",
        "months",
        "warmup_months",
        "intervention_start_month",
        "hiring_interval_months",
        "transfer_interval_months",
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"Fixed contract changed for {key}: {manifest.get(key)} != {expected[key]}"
            )
    if manifest.get("status") != "complete":
        raise ValueError("run manifest status is not complete")
    if not isinstance(manifest.get("fingerprints"), dict):
        raise ValueError("run manifest lacks current public release fingerprints")
    baseline_manifest = manifest.get("baseline_policy", {})
    intervention_manifest = manifest.get("intervention_policy", {})
    if baseline_manifest != BASELINE_POLICY.model_dump():
        raise ValueError("Human baseline was modified")
    if run_name == "run_0":
        if intervention_manifest != baseline_manifest:
            raise ValueError("run_0 must execute the human baseline")
    else:
        if intervention_manifest.get("label") == FIXED_CONTRACT["baseline_label"]:
            raise ValueError("Intervention run reused the baseline label")
        run0_manifest_path = folder.parent / "run_0" / "run_manifest.json"
        if not run0_manifest_path.exists():
            raise FileNotFoundError("run_0 manifest is required")
        run0_manifest = json.loads(run0_manifest_path.read_text(encoding="utf-8"))
        for key in (
            "seed",
            "num_agents",
            "months",
            "warmup_months",
            "intervention_start_month",
            "model",
            "temperature",
            "reasoning_effort",
            "feedback",
            "scenario",
            "manager_mode",
            "shock_tape_hash",
            "transition_parameters",
            "fingerprints",
        ):
            if manifest.get(key) != run0_manifest.get(key):
                raise ValueError(f"Matched-comparison field changed for {key}")
        current_num = int(run_name.split("_", 1)[1])
        current_payload = _policy_payload(intervention_manifest)
        for prior_num in range(1, current_num):
            prior_path = folder.parent / f"run_{prior_num}" / "run_manifest.json"
            if not prior_path.exists():
                continue
            prior = json.loads(prior_path.read_text(encoding="utf-8"))[
                "intervention_policy"
            ]
            if prior.get("label") == intervention_manifest.get("label"):
                raise ValueError(f"Policy label duplicates run_{prior_num}")
            if _policy_payload(prior) == current_payload:
                raise ValueError(f"Policy parameters duplicate run_{prior_num}")
    payload = json.loads((folder / "final_info.json").read_text(encoding="utf-8"))
    means = payload.get("policy_lab", {}).get("means")
    if not isinstance(means, dict) or not means:
        raise ValueError("final_info.json must contain policy_lab.means")
    for key, value in means.items():
        if is_operator_only_metric(key):
            raise ValueError(
                f"Operator-only metric leaked into scientist-visible final_info.json: {key}"
            )
        # Null is the required representation for empty subgroups and undefined
        # cost-effectiveness denominators.  Zero would create a false success.
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric {key} is not numeric or null: {value!r}")
    if means.get("active_headcount_end") != means.get("active_headcount_end_reconciled"):
        raise ValueError("End-of-month active headcount did not reconcile")
    preflight = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
    visible_cost = means.get("policy_implementation_cost_points")
    visible_budget = means.get("policy_budget_max_points")
    if not isinstance(visible_cost, (int, float)) or not isinstance(visible_budget, (int, float)):
        raise ValueError("Scientist-visible policy implementation cost and budget are required")
    if float(visible_cost) > float(visible_budget) + 1e-9:
        raise ValueError("Policy implementation cost exceeds the fixed scientist-visible budget")
    if abs(float(visible_cost) - float(preflight["policy_implementation_cost_points"])) > 1e-9:
        raise ValueError("Preflight and final policy implementation costs differ")
    return {
        key: (float(value) if isinstance(value, (int, float)) else None)
        for key, value in means.items()
    }
