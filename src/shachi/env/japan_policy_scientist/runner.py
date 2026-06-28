"""Resumable, fingerprinted runner for one fixed-seed policy condition."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import pydantic
import yaml

from shachi.agent.japan_policy_bureaucrat import (
    BudgetExceededError,
    JapanPolicyBureaucratAgent,
    JapanPolicyManagerAgent,
    PreparedAgentRequest,
    SharedLLMRuntime,
    UsageLedger,
)

from .batch_execution import resolve_employee_month_with_gemini_openai_batch
from .dynamics import TransitionParameters
from .experiment_contract import validate_scientific_environment
from .environment import JapanPolicyLabEnv
from .fingerprints import (
    canonical_json,
    runtime_fingerprints,
    scientific_run_config_payload,
)
from .memory import BureaucratMemory, ManagerFactMemory
from .population import normalize_scenario
from .metrics import add_baseline_deltas, aggregate_metrics, write_artifacts
from .policy_cost import (
    policy_cost_breakdown,
    policy_implementation_cost_points,
    validate_policy_budget,
)
from .schemas import BureaucracyObservation, BureaucratMonthlyAction, ManagerObservation, PolicyConfig
from .validation import validate_response_against_observation
from .visibility import (
    operator_audit_dir,
    operator_only_metrics,
    scientist_visible_metrics,
)


class CacheIdentityMismatchError(RuntimeError):
    """Existing cache belongs to a different prompt/code/schema contract."""


class ResponseCache:
    def __init__(
        self,
        *,
        root: Path,
        warmup_months: int,
        warmup_fallback_root: Path | None,
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.warmup_months = warmup_months
        self.warmup_fallback_root = warmup_fallback_root
        self.local_hits = 0
        self.warmup_hits = 0
        self.new_responses = 0
        self.identity_mismatches = 0

    @staticmethod
    def _filename(month: int, phase: str) -> str:
        return f"month_{month:02d}_{phase}.jsonl"

    @staticmethod
    def _read(
        path: Path,
        response_type: type[pydantic.BaseModel],
    ) -> dict[int, tuple[int, dict[str, Any], pydantic.BaseModel]]:
        rows: dict[int, tuple[int, dict[str, Any], pydantic.BaseModel]] = {}
        if not path.exists():
            return rows
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if payload["response_type"] != response_type.__name__:
                    raise ValueError("response type mismatch")
                slot_id = int(payload["slot_id"])
                epoch = int(payload["identity_epoch"])
                identity = payload.get("cache_identity")
                if not isinstance(identity, dict):
                    raise ValueError("legacy cache row has no cache_identity")
                if slot_id in rows:
                    raise ValueError(f"duplicate slot_id {slot_id} in one cache file")
                rows[slot_id] = (
                    epoch,
                    identity,
                    response_type.model_validate(payload["payload"]),
                )
            except Exception as exc:
                raise RuntimeError(f"Invalid cache {path}:{line_no}: {exc}") from exc
        return rows

    @staticmethod
    def _identity_diff(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
        keys = sorted(set(expected) | set(actual))
        return [key for key in keys if expected.get(key) != actual.get(key)]

    def load(
        self,
        requests: dict[int, PreparedAgentRequest],
    ) -> dict[int, pydantic.BaseModel]:
        if not requests:
            return {}
        sample = next(iter(requests.values()))
        response_type = sample.response_type
        filename = self._filename(sample.observation.month, sample.observation.phase)
        local = self._read(self.root / filename, response_type)
        fallback: dict[int, tuple[int, dict[str, Any], pydantic.BaseModel]] = {}
        if (
            sample.observation.month <= self.warmup_months
            and self.warmup_fallback_root is not None
        ):
            fallback = self._read(
                self.warmup_fallback_root / filename, response_type
            )

        resolved: dict[int, pydantic.BaseModel] = {}
        for slot_id, request in requests.items():
            observation = request.observation
            source_name = None
            row = None
            if slot_id in local:
                row = local[slot_id]
                source_name = "local"
            elif slot_id in fallback:
                row = fallback[slot_id]
                source_name = "warmup"
            if row is None:
                continue
            epoch, identity, response = row
            if epoch != observation.identity_epoch:
                continue
            differences = self._identity_diff(request.cache_identity, identity)
            if differences:
                self.identity_mismatches += 1
                raise CacheIdentityMismatchError(
                    f"Cache invalidated for month={observation.month}, phase={observation.phase}, "
                    f"slot={slot_id}, source={source_name}; differing identity fields={differences}. "
                    "Use a new run directory or deliberately archive the incompatible cache."
                )
            resolved[slot_id] = response
            if source_name == "local":
                self.local_hits += 1
            else:
                self.warmup_hits += 1

        if (
            sample.observation.month <= self.warmup_months
            and self.warmup_fallback_root is not None
        ):
            missing = set(requests) - set(resolved)
            if missing:
                raise RuntimeError(
                    "run_0 warm-up cache incomplete under the exact current public release identity for "
                    f"month={sample.observation.month}, phase={sample.observation.phase}; "
                    f"missing {len(missing)} slots"
                )
        return resolved

    def append(
        self,
        *,
        request: PreparedAgentRequest,
        response: pydantic.BaseModel,
    ) -> None:
        observation = request.observation
        path = self.root / self._filename(observation.month, observation.phase)
        row = {
            "slot_id": observation.agent_id,
            "identity_epoch": observation.identity_epoch,
            "response_type": type(response).__name__,
            "cache_identity": request.cache_identity,
            "payload": response.model_dump(mode="json"),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.new_responses += 1

    def snapshot(self) -> dict[str, float]:
        return {
            "response_cache_local_hits": float(self.local_hits),
            "response_cache_warmup_hits": float(self.warmup_hits),
            "response_cache_new_responses": float(self.new_responses),
            "response_cache_identity_mismatches": float(self.identity_mismatches),
        }




def _extract_prompt_field(text: str, key: str) -> str:
    marker = f"{key}="
    if marker not in text:
        return "unknown"
    return text.split(marker, 1)[1].split(";", 1)[0].strip()

def _numeric_prompt_field(text: str, key: str, default: float = 0.0) -> float:
    marker = f"{key}="
    if marker not in text:
        return default
    raw = text.split(marker, 1)[1].split(";", 1)[0].strip().rstrip("%")
    try:
        return float(raw)
    except ValueError:
        return default

def _band(value: float, *, low: float, high: float) -> str:
    return "high" if value >= high else "mid" if value >= low else "low"

def _cohort_signature(observation: BureaucracyObservation | ManagerObservation, granularity: str) -> tuple[str, ...]:
    if isinstance(observation, ManagerObservation):
        return ("manager", observation.department, str(observation.month))
    if observation.phase != "monthly_action":
        return ("non_monthly", observation.phase, str(observation.month), str(observation.agent_id))
    profile = observation.profile_summary
    dept = _extract_prompt_field(profile, "department")
    rank = _extract_prompt_field(profile, "rank")
    family = _extract_prompt_field(profile, "family_or_caregiving_constraints")
    field = _extract_prompt_field(profile, "professional_field")
    workload = _numeric_prompt_field(observation.department_summary, "current_workload_relative_to_effective_capacity", 1.0)
    backlog = _numeric_prompt_field(observation.department_summary, "backlog_units", 0.0)
    last_effort = _numeric_prompt_field(observation.personal_objective_summary, "last_relative_effort", 100.0)
    last_completion = _numeric_prompt_field(observation.personal_objective_summary, "last_personal_completion_ratio", 1.0)
    event_types = ",".join(sorted({getattr(event, "event_type", "event") for event in observation.recent_events}))
    if granularity == "very_coarse":
        return ("monthly", str(observation.month), dept, rank, family, _band(workload, low=0.95, high=1.15))
    if granularity == "medium":
        return (
            "monthly", str(observation.month), dept, rank, family, field,
            _band(workload, low=0.95, high=1.15),
            _band(backlog, low=2.0, high=8.0),
            _band(last_effort, low=95.0, high=115.0),
            _band(last_completion, low=0.75, high=0.95),
            event_types,
        )
    # coarse default: close to representative-agent ABM practice.  It preserves
    # department/rank/family/workload cells but does not spend one API call per
    # person-month.
    return (
        "monthly", str(observation.month), dept, rank, family,
        _band(workload, low=0.95, high=1.15),
        _band(last_effort, low=95.0, high=115.0),
    )

def _current_department(observation: BureaucracyObservation) -> str | None:
    marker = "department="
    if marker in observation.profile_summary:
        return observation.profile_summary.split(marker, 1)[1].split(";", 1)[0].strip()
    return None

def _adapt_representative_response(
    *,
    response: pydantic.BaseModel,
    observation: BureaucracyObservation | ManagerObservation,
) -> pydantic.BaseModel:
    # Representative-cohort sharing is allowed only for employee monthly actions.
    # The raw representative action is preserved in the response cache for the
    # representative slot; per-person event IDs are rewritten to each person's
    # own valid realized event ID so semantic validation remains identity-safe.
    if not isinstance(observation, BureaucracyObservation):
        return response
    if observation.phase != "monthly_action" or type(response).__name__ != "BureaucratMonthlyAction":
        return response
    payload = response.model_dump(mode="json")
    allowed = observation.allowed_event_ids()
    if allowed:
        payload["event_refs"] = allowed[: min(max(1, len(payload.get("event_refs", []))), min(3, len(allowed)))]
    current = _current_department(observation)
    pref = payload.get("transfer_preference")
    if pref and current:
        acceptable = [dept for dept in pref.get("acceptable_departments", []) if dept != current]
        if pref.get("preferred_department") == current or not acceptable:
            payload["career_action"] = "stay"
            payload["transfer_preference"] = None
            payload["reason"] = str(payload.get("reason", "")) + " Representative cohort transfer preference removed because it targeted this person's current department."
        else:
            if pref.get("preferred_department") == current:
                pref["preferred_department"] = acceptable[0]
            pref["acceptable_departments"] = acceptable
            payload["transfer_preference"] = pref
    return type(response).model_validate(payload)

async def _resolve_representative_cohort_batch(
    *,
    observations: dict[int, BureaucracyObservation | ManagerObservation],
    agents: dict[int, JapanPolicyBureaucratAgent | JapanPolicyManagerAgent],
    requests: dict[int, PreparedAgentRequest],
    missing: list[int],
    cache: ResponseCache,
    runtime: SharedLLMRuntime,
    backend: str,
    granularity: str,
    max_groups: int,
) -> dict[int, pydantic.BaseModel]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for slot_id in missing:
        obs = observations[slot_id]
        if not isinstance(obs, BureaucracyObservation) or obs.phase != "monthly_action":
            groups[("passthrough", str(slot_id))] = [slot_id]
            continue
        groups.setdefault(_cohort_signature(obs, granularity), []).append(slot_id)
    if max_groups > 0 and len(groups) > max_groups:
        # Merge the least-specific dimension by falling back to very_coarse.  This
        # is not a hidden outcome cap; it only controls API-call count while each
        # person still receives identity-safe event refs and Python-owned state
        # transitions.
        groups = {}
        for slot_id in missing:
            obs = observations[slot_id]
            if not isinstance(obs, BureaucracyObservation) or obs.phase != "monthly_action":
                groups[("passthrough", str(slot_id))] = [slot_id]
            else:
                groups.setdefault(_cohort_signature(obs, "very_coarse"), []).append(slot_id)
    reps = {members[0]: members for members in groups.values()}
    representative_ids = sorted(reps)
    rep_requests = {slot_id: requests[slot_id] for slot_id in representative_ids}
    rep_observations = {slot_id: observations[slot_id] for slot_id in representative_ids}
    if backend == "gemini_openai_batch" and os.environ.get("POLICYLAB_MOCK_LLM") != "1":
        representative_results = await resolve_employee_month_with_gemini_openai_batch(
            observations=rep_observations, requests=rep_requests, runtime=runtime, cache=cache
        )
    else:
        tasks = [agents[slot_id].step(observations[slot_id]) for slot_id in representative_ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        representative_results = {}
        failures = []
        for slot_id, result in zip(representative_ids, raw_results, strict=True):
            if isinstance(result, BudgetExceededError):
                raise result
            if isinstance(result, BaseException):
                failures.append(f"representative slot {slot_id}: {type(result).__name__}: {result}")
                continue
            representative_results[slot_id] = result
        if failures:
            raise RuntimeError("representative cohort calls failed; " + "; ".join(failures[:10]))
    audit_path = cache.root / "representative_cohort_audit.jsonl"
    resolved: dict[int, pydantic.BaseModel] = {}
    for key, members in groups.items():
        rep_id = members[0]
        rep_response = representative_results[rep_id]
        audit_row = {
            "event": "representative_cohort_assignment",
            "cohort_key": list(key),
            "representative_slot_id": rep_id,
            "member_slot_ids": members,
            "cohort_size": len(members),
            "backend": backend,
            "granularity": granularity,
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
        for slot_id in members:
            adapted = _adapt_representative_response(response=rep_response, observation=observations[slot_id])
            validate_response_against_observation(observations[slot_id], adapted)
            agents[slot_id].record_response(observations[slot_id], adapted)
            cache.append(request=requests[slot_id], response=adapted)
            resolved[slot_id] = adapted
    return resolved




def _monthly_decision_required(
    observation: BureaucracyObservation | ManagerObservation,
    *,
    interval_months: int,
    anchor_months: set[int],
    trigger_event_types: set[str],
) -> bool:
    if not isinstance(observation, BureaucracyObservation):
        return True
    if observation.phase != "monthly_action":
        return True
    month = int(observation.month)
    if interval_months <= 1:
        return True
    if month in anchor_months:
        return True
    if (month - 1) % interval_months == 0:
        return True
    if any(event.event_type in trigger_event_types for event in observation.recent_events):
        return True
    return False




async def _resolve_batch(
    *,
    observations: dict[int, BureaucracyObservation | ManagerObservation],
    agents: dict[int, JapanPolicyBureaucratAgent | JapanPolicyManagerAgent],
    cache: ResponseCache,
    runtime: SharedLLMRuntime,
    employee_monthly_backend: str,
    employee_decision_interval_months: int = 1,
    employee_decision_anchor_months: set[int] | None = None,
    employee_decision_event_triggers: set[str] | None = None,
) -> dict[int, pydantic.BaseModel]:
    anchor_months = employee_decision_anchor_months or set()
    trigger_events = employee_decision_event_triggers or set()
    projected: dict[int, pydantic.BaseModel] = {}
    provider_observations: dict[int, BureaucracyObservation | ManagerObservation] = {}
    for slot_id, observation in observations.items():
        agent = agents[slot_id]
        if (
            employee_monthly_backend.startswith("temporal_representative")
            and isinstance(agent, JapanPolicyBureaucratAgent)
            and not _monthly_decision_required(
                observation,
                interval_months=employee_decision_interval_months,
                anchor_months=anchor_months,
                trigger_event_types=trigger_events,
            )
        ):
            response = agent.project_monthly_action(observation)
            if response is not None:
                validate_response_against_observation(observation, response)
                agent.record_response(observation, response)
                projected[slot_id] = response
                continue
        provider_observations[slot_id] = observation

    if not provider_observations:
        return {slot_id: projected[slot_id] for slot_id in sorted(projected)}

    requests = {
        slot_id: agents[slot_id].prepare_request(observation)
        for slot_id, observation in provider_observations.items()
    }
    cached = cache.load(requests)
    for slot_id, response in cached.items():
        validate_response_against_observation(provider_observations[slot_id], response)
        agents[slot_id].record_response(provider_observations[slot_id], response)
    missing = sorted(set(provider_observations) - set(cached))
    if missing:
        if employee_monthly_backend.startswith("representative_cohort"):
            backend = os.environ.get("POLICYLAB_REPRESENTATIVE_COHORT_BACKEND", "live")
            if employee_monthly_backend.endswith("batch"):
                backend = "gemini_openai_batch"
            if employee_monthly_backend.endswith("local"):
                backend = "live"
            cohort_results = await _resolve_representative_cohort_batch(
                observations=observations,
                agents=agents,
                requests=requests,
                missing=missing,
                cache=cache,
                runtime=runtime,
                backend=backend,
                granularity=os.environ.get("POLICYLAB_REPRESENTATIVE_COHORT_GRANULARITY", "coarse"),
                max_groups=int(os.environ.get("POLICYLAB_REPRESENTATIVE_COHORT_MAX_GROUPS", "40")),
            )
            cached.update(cohort_results)
            missing_after_cohort = sorted(set(provider_observations) - set(cached))
            if missing_after_cohort:
                raise RuntimeError(f"Representative cohort backend left missing slots: {missing_after_cohort[:20]}")
            merged = {**projected, **cached}
            return {slot_id: merged[slot_id] for slot_id in sorted(observations)}

        if employee_monthly_backend == "gemini_openai_batch":
            missing_requests = {slot_id: requests[slot_id] for slot_id in missing}
            batch_results = await resolve_employee_month_with_gemini_openai_batch(
                observations={slot_id: provider_observations[slot_id] for slot_id in missing},
                requests=missing_requests,
                runtime=runtime,
                cache=cache,
            )
            for slot_id, result in batch_results.items():
                validate_response_against_observation(provider_observations[slot_id], result)
                agents[slot_id].record_response(provider_observations[slot_id], result)
                cache.append(request=requests[slot_id], response=result)
                cached[slot_id] = result
            missing_after_batch = sorted(set(provider_observations) - set(cached))
            if missing_after_batch:
                raise RuntimeError(f"Gemini batch backend left missing slots: {missing_after_batch[:20]}")
            merged = {**projected, **cached}
            return {slot_id: merged[slot_id] for slot_id in sorted(observations)}

        tasks = [agents[slot_id].step(provider_observations[slot_id]) for slot_id in missing]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures: list[str] = []
        for slot_id, result in zip(missing, results, strict=True):
            if isinstance(result, BudgetExceededError):
                # Do not wrap the operator-only circuit breaker as a generic
                # experiment failure. The AI Scientist wrapper recognizes this
                # exact type and pauses without redesigning the policy.
                raise result
            if isinstance(result, BaseException):
                failures.append(
                    f"slot {slot_id}: {type(result).__name__}: {result}"
                )
                continue
            if not isinstance(result, pydantic.BaseModel):
                failures.append(f"slot {slot_id}: unexpected result {type(result)}")
                continue
            validate_response_against_observation(provider_observations[slot_id], result)
            cache.append(request=requests[slot_id], response=result)
            cached[slot_id] = result
        if failures:
            raise RuntimeError(
                f"{len(failures)} agent calls failed; all valid responses and provider logs were preserved. "
                + "; ".join(failures[:10])
            )
    merged = {**projected, **cached}
    return {slot_id: merged[slot_id] for slot_id in sorted(observations)}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _warmup_fallback(output_dir: Path) -> Path | None:
    explicit = os.environ.get("POLICYLAB_WARMUP_CACHE_DIR")
    if explicit:
        return Path(explicit)
    if output_dir.name == "run_0":
        return None
    run0 = output_dir.parent / "run_0"
    if not (run0 / "complete.marker").exists():
        raise RuntimeError("Create a complete run_0 before intervention runs")
    cache = run0 / "response_cache"
    if not cache.exists():
        raise RuntimeError("run_0 response_cache is missing")
    return cache


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


async def _run_async(
    *,
    output_dir: Path,
    baseline_policy: PolicyConfig,
    intervention_policy: PolicyConfig,
    config: dict[str, Any],
) -> dict[str, Any]:
    fixed = config["experiment"]
    llm_cfg = config["llm"]
    environment_contract = validate_scientific_environment(
        config,
        allow_holdout=os.environ.get("POLICYLAB_HOLDOUT_MODE") == "1",
    )
    seed = int(os.environ.get("POLICYLAB_FIXED_SEED", fixed["fixed_seed"]))
    num_agents = int(os.environ.get("POLICYLAB_NUM_AGENTS", fixed["num_agents"]))
    months = int(os.environ.get("POLICYLAB_MONTHS", fixed["months"]))
    warmup = int(
        os.environ.get("POLICYLAB_WARMUP_MONTHS", fixed["warmup_months"])
    )
    intervention_start = int(
        os.environ.get(
            "POLICYLAB_INTERVENTION_START_MONTH",
            fixed["intervention_start_month"],
        )
    )
    model = os.environ.get("POLICYLAB_SHACHI_MODEL", llm_cfg["model"])
    feedback = os.environ.get("POLICYLAB_ENABLE_FEEDBACK", "1") == "1"
    scenario = normalize_scenario(
        os.environ.get(
            "POLICYLAB_SCENARIO",
            os.environ.get("POLICYLAB_WORLD", fixed.get("scenario", "reference_stressed")),
        )
    )
    manager_mode = os.environ.get(
        "POLICYLAB_MANAGER_MODE",
        config.get("management", {}).get("mode", "deterministic_priority"),
    )
    transition_parameters = TransitionParameters()
    policy_budget_max_points = float(config["policy_budget"]["max_points"])
    implementation_cost = validate_policy_budget(
        intervention_policy,
        max_points=policy_budget_max_points,
        baseline=baseline_policy,
    )
    implementation_breakdown = policy_cost_breakdown(
        intervention_policy, baseline_policy
    )

    env = JapanPolicyLabEnv(
        num_agents=num_agents,
        months=months,
        warmup_months=warmup,
        intervention_start_month=intervention_start,
        seed=seed,
        baseline_policy=baseline_policy,
        intervention_policy=intervention_policy,
        transition_parameters=transition_parameters,
        hiring_interval_months=int(fixed["hiring_interval_months"]),
        transfer_interval_months=int(fixed["transfer_interval_months"]),
        enable_feedback=feedback,
        scenario=scenario,
        manager_mode=manager_mode,
    )
    fingerprints = runtime_fingerprints()
    config_hash = _canonical_hash(scientific_run_config_payload(config))
    contract = {
        "schema_version": 6,
        "config_hash": config_hash,
        "seed": seed,
        "num_agents": num_agents,
        "months": months,
        "warmup_months": warmup,
        "intervention_start_month": intervention_start,
        "hiring_interval_months": int(fixed["hiring_interval_months"]),
        "transfer_interval_months": int(fixed["transfer_interval_months"]),
        "baseline_policy": baseline_policy.model_dump(),
        "intervention_policy": intervention_policy.model_dump(),
        "policy_implementation_cost_points": implementation_cost,
        "policy_budget_max_points": policy_budget_max_points,
        "model": model,
        "temperature": float(llm_cfg["temperature"]),
        "reasoning_effort": llm_cfg["reasoning_effort"],
        "feedback": feedback,
        "scenario": scenario,
        "manager_mode": manager_mode,
        "shock_tape_hash": env.shock_tape.manifest_hash(months),
        "transition_parameters": transition_parameters.as_dict(),
        "fingerprints": fingerprints,
        "environment_contract_preflight": environment_contract,
    }
    contract_hash = _canonical_hash(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_hash") != contract_hash:
            raise RuntimeError(
                f"Output directory {output_dir} contains a different run contract; use a new directory"
            )
        if (output_dir / "complete.marker").exists():
            payload = json.loads(
                (output_dir / "final_info.json").read_text(encoding="utf-8")
            )
            return {
                key: (float(value) if isinstance(value, (int, float)) else None)
                for key, value in payload["policy_lab"]["means"].items()
            }
    else:
        manifest_path.write_text(
            json.dumps(
                {
                    **contract,
                    "contract_hash": contract_hash,
                    "status": "running",
                    "created_unix_time": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Exact warm-up reuse additionally requires the run_0 manifest fingerprints.
    fallback_root = _warmup_fallback(output_dir)
    if fallback_root is not None:
        run0_manifest = json.loads(
            (fallback_root.parent / "run_manifest.json").read_text(encoding="utf-8")
        )
        if run0_manifest.get("fingerprints") != fingerprints:
            raise CacheIdentityMismatchError(
                "run_0 fingerprints differ from this intervention runtime; warm-up cache reuse is forbidden"
            )
        if run0_manifest.get("config_hash") != config_hash:
            raise CacheIdentityMismatchError(
                "run_0 experiment configuration differs from this intervention runtime; "
                "warm-up cache reuse is forbidden"
            )

    audit_dir = operator_audit_dir(output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    hard_stop_usd = float(
        os.environ.get(
            "POLICYLAB_OPERATOR_HARD_STOP_USD",
            os.environ.get("POLICYLAB_MAX_LLM_COST_USD", "300.0"),
        )
    )
    ledger = UsageLedger(
        audit_dir / "llm_usage.jsonl",
        max_cost_usd=hard_stop_usd,
        campaign_log_path=os.environ.get("POLICYLAB_CAMPAIGN_LEDGER"),
        campaign_id=os.environ.get("POLICYLAB_CAMPAIGN_ID"),
    )
    os.environ.setdefault("POLICYLAB_LIVE_CALL_SPACING_SECONDS", str(llm_cfg.get("live_call_spacing_seconds", 0.0)))
    os.environ.setdefault("POLICYLAB_REPRESENTATIVE_COHORT_GRANULARITY", str(llm_cfg.get("representative_cohort_granularity", "coarse")))
    # Backward/alias support: POLICYLAB_COHORT_MAX_REPRESENTATIVES_PER_MONTH is clearer
    # for operator runbooks; POLICYLAB_REPRESENTATIVE_COHORT_MAX_GROUPS remains the
    # internal name used by older current public release scripts.
    max_groups = str(
        llm_cfg.get(
            "representative_cohort_max_groups_per_month",
            llm_cfg.get("cohort_max_representatives_per_month", 40),
        )
    )
    os.environ.setdefault("POLICYLAB_COHORT_MAX_REPRESENTATIVES_PER_MONTH", max_groups)
    os.environ.setdefault("POLICYLAB_REPRESENTATIVE_COHORT_MAX_GROUPS", os.environ.get("POLICYLAB_COHORT_MAX_REPRESENTATIVES_PER_MONTH", max_groups))
    os.environ.setdefault("POLICYLAB_REPRESENTATIVE_COHORT_BACKEND", str(llm_cfg.get("representative_cohort_backend", "live")))
    runtime = SharedLLMRuntime(
        model=model,
        temperature=float(llm_cfg["temperature"]),
        max_concurrency=int(llm_cfg["max_concurrency"]),
        network_retries=int(llm_cfg["network_retries"]),
        validation_retries=int(llm_cfg["validation_retries"]),
        retry_sleep_seconds=float(llm_cfg["retry_sleep_seconds"]),
        reasoning_effort=str(llm_cfg["reasoning_effort"]),
        ledger=ledger,
        base_seed=seed,
    )
    agents: dict[int, JapanPolicyBureaucratAgent | JapanPolicyManagerAgent] = {
        slot_id: JapanPolicyBureaucratAgent(
            slot_id=slot_id,
            runtime=runtime,
            monthly_output_token_estimate=int(llm_cfg.get("monthly_output_token_estimate", 900)),
            quarterly_output_token_estimate=int(llm_cfg.get("quarterly_output_token_estimate", 900)),
            memory=BureaucratMemory(
                monthly_window=int(config["memory"]["monthly_window"]),
                quarterly_window=int(config["memory"]["quarterly_window"]),
                fact_window=int(config["memory"].get("fact_window", 18)),
            ),
        )
        for slot_id in range(num_agents)
    }
    if manager_mode == "llm":
        agents.update(
            {
                manager_id: JapanPolicyManagerAgent(
                    manager_id=manager_id,
                    runtime=runtime,
                    output_token_estimate=int(llm_cfg.get("manager_output_token_estimate", 900)),
                    memory=ManagerFactMemory(
                        window=int(config["memory"].get("manager_fact_window", 12))
                    ),
                )
                for manager_id in env.manager_departments
            }
        )
    cache = ResponseCache(
        root=output_dir / "response_cache",
        warmup_months=warmup,
        warmup_fallback_root=fallback_root,
    )

    start = time.perf_counter()
    observations = await env.reset()
    while not env.done():
        if not observations:
            observations = await env.step({})
            continue
        typed_observations: dict[int, BureaucracyObservation | ManagerObservation] = {}
        for slot_id, observation in observations.items():
            phase = getattr(observation, "phase", None)
            if phase == "manager_decision":
                typed_observations[slot_id] = ManagerObservation.model_validate(observation)
            else:
                typed_observations[slot_id] = BureaucracyObservation.model_validate(observation)
        responses = await _resolve_batch(
            observations=typed_observations,
            agents=agents,
            cache=cache,
            runtime=runtime,
            employee_monthly_backend=str(os.environ.get("POLICYLAB_EMPLOYEE_MONTHLY_BACKEND", llm_cfg.get("employee_monthly_backend", "live"))),
            employee_decision_interval_months=int(llm_cfg.get("employee_decision_interval_months", os.environ.get("POLICYLAB_EMPLOYEE_DECISION_INTERVAL_MONTHS", 1))),
            employee_decision_anchor_months={int(value) for value in llm_cfg.get("employee_decision_anchor_months", [])},
            employee_decision_event_triggers={str(value) for value in llm_cfg.get("employee_decision_event_triggers", [])},
        )
        observations = await env.step(responses)
    runtime_seconds = time.perf_counter() - start
    result = env.get_result()
    full_metrics = aggregate_metrics(
        result=result,
        intervention_start_month=intervention_start,
        initial_slots=num_agents,
        runtime_seconds=runtime_seconds,
        usage=ledger.snapshot(),
        cache=cache.snapshot(),
    )
    # Policy burden is scientific input and must be present before baseline
    # deltas/cost-effectiveness are calculated. Failed or non-improving
    # denominators remain null rather than appearing as zero-cost success.
    full_metrics["policy_implementation_cost_points"] = implementation_cost
    full_metrics["policy_budget_max_points"] = policy_budget_max_points
    baseline_info = (
        None
        if output_dir.name == "run_0"
        else output_dir.parent / "run_0" / "final_info.json"
    )
    full_metrics = add_baseline_deltas(full_metrics, baseline_info)
    science_metrics = scientist_visible_metrics(full_metrics)
    operator_metrics = operator_only_metrics(full_metrics)
    operator_metrics.update(
        {
            "operator_only": True,
            "visibility_contract": (
                "Simulation compute accounting is not passed to bureaucrat agents, idea generation, "
                "Aider, policy ranking, notes, plots, or final_info.json. Synthetic policy "
                "implementation cost remains scientist-visible."
            ),
            "operator_hard_stop_usd": hard_stop_usd,
            "operator_hard_stop_scope": (
                "Shachi employee/manager calls observed by this package only; upstream "
                "AI Scientist ideation, Aider, write-up, and review calls require separate "
                "provider-side accounting and human stage approval."
            ),
            "scientist_visible_policy_implementation_cost_points": implementation_cost,
            "policy_budget_max_points": policy_budget_max_points,
            "policy_budget_enforced_for_this_run": True,
            "llm_validation_failures": full_metrics.get("llm_validation_failures", 0.0),
            "llm_network_failures": full_metrics.get("llm_network_failures", 0.0),
        }
    )
    (output_dir / "policy_implementation_cost_breakdown.json").write_text(
        json.dumps(
            {
                "policy_implementation_cost_points": implementation_cost,
                "policy_budget_max_points": policy_budget_max_points,
                "breakdown": implementation_breakdown,
                "note": "Synthetic implementation burden; not API or infrastructure cost.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_artifacts(output_dir=output_dir, result=result, metrics=science_metrics)
    (audit_dir / "operator_audit.json").write_text(
        json.dumps(operator_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["completed_unix_time"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "complete.marker").write_text(contract_hash + "\n", encoding="utf-8")
    (output_dir / "failed.marker").unlink(missing_ok=True)
    (output_dir / "failure.json").unlink(missing_ok=True)
    return science_metrics


def run_policy_experiment(
    *,
    output_dir: str | Path,
    baseline_policy: PolicyConfig,
    intervention_policy: PolicyConfig,
    config_path: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    try:
        return asyncio.run(
            _run_async(
                output_dir=output_path,
                baseline_policy=baseline_policy,
                intervention_policy=intervention_policy,
                config=load_config(config_path),
            )
        )
    except Exception as exc:
        output_path.mkdir(parents=True, exist_ok=True)
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "failed_unix_time": time.time(),
            "artifacts_preserved": True,
        }
        (output_path / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_path / "failed.marker").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise
