"""Reproducibility fingerprints used by run manifests and response caches."""
from __future__ import annotations

import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .prompting import BUREAUCRAT_SYSTEM_PROMPT, MANAGER_SYSTEM_PROMPT
from .schemas import (
    BureaucracyObservation,
    BureaucratMonthlyAction,
    BureaucratQuarterlyReflection,
    ManagerDecision,
    ManagerObservation,
    WorkEvent,
)

CACHE_SCHEMA_VERSION = 3


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def runtime_fingerprints() -> dict[str, str | int]:
    module_dir = Path(__file__).resolve().parent
    shachi_dir = module_dir.parents[1]
    agent_path = shachi_dir / "agent" / "japan_policy_bureaucrat.py"
    package_meta_path = module_dir / "release_fingerprint.json"
    package_meta: dict[str, Any] = {}
    if package_meta_path.exists():
        package_meta = json.loads(package_meta_path.read_text(encoding="utf-8"))

    code_paths = list(module_dir.glob("*.py")) + [agent_path]
    transition_names = {
        "environment.py",
        "dynamics.py",
        "shock_tape.py",
        "transfer_planner.py",
        "policy_cost.py",
        "population.py",
        "task_queue.py",
    }
    transition_paths = [path for path in code_paths if path.name in transition_names]
    prompt_source = (
        BUREAUCRAT_SYSTEM_PROMPT
        + "\n"
        + MANAGER_SYSTEM_PROMPT
        + "\n"
        + inspect.getsource(BureaucracyObservation.format_as_prompt_text)
        + "\n"
        + inspect.getsource(ManagerObservation.format_as_prompt_text)
    )
    # The generic Shachi Observation contains Tool.callable fields, which are
    # intentionally not JSON-schema serializable. Hash the policy-lab
    # observation contract explicitly instead of asking Pydantic to serialize
    # unrelated callable tool definitions.
    observation_contract = {
        name: str(field.annotation)
        for name, field in BureaucracyObservation.model_fields.items()
        if name not in {"tools", "response_type"}
    }
    manager_observation_contract = {
        name: str(field.annotation)
        for name, field in ManagerObservation.model_fields.items()
        if name not in {"tools", "response_type"}
    }
    schema_payload = {
        "monthly": BureaucratMonthlyAction.model_json_schema(),
        "quarterly": BureaucratQuarterlyReflection.model_json_schema(),
        "manager": ManagerDecision.model_json_schema(),
        "work_event": WorkEvent.model_json_schema(),
        "observation_contract": observation_contract,
        "manager_observation_contract": manager_observation_contract,
        "observation_response_types": [
            BureaucratMonthlyAction.__name__,
            BureaucratQuarterlyReflection.__name__,
            ManagerDecision.__name__,
        ],
    }
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "package_version": str(package_meta.get("version", "development")),
        "package_hash": str(package_meta.get("package_hash", "development")),
        "upstream_shachi_ref": str(package_meta.get("shachi_ref", "unknown")),
        "upstream_ai_scientist_ref": str(package_meta.get("ai_scientist_ref", "unknown")),
        "code_hash": _hash_files(code_paths),
        "transition_hash": _hash_files(transition_paths),
        "prompt_hash": sha256_text(prompt_source),
        "schema_hash": sha256_text(canonical_json(schema_payload)),
    }



def _scientific_llm_payload(config: dict[str, Any]) -> dict[str, Any]:
    """LLM settings that can change model behavior, not operator cost estimates."""
    llm = dict(config.get("llm", {}))
    for key in list(llm):
        if key.endswith("_output_token_estimate"):
            llm.pop(key, None)
    return llm


def scientific_run_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return only configuration that can affect the scientific run contract.

    Operator hard stops, paid-stage budgets, and output-token cost estimates do
    not alter prompts, transitions, random tapes, or metrics.  The synthetic
    policy budget, model, temperature, reasoning effort, validation policy,
    memory, and experiment dimensions remain part of the scientific comparison
    contract.
    """
    payload = {
        key: config[key]
        for key in ("package", "experiment", "memory", "policy_budget")
        if key in config
    }
    if "llm" in config:
        payload["llm"] = _scientific_llm_payload(config)
    return payload


def behavioral_pilot_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the paired-pilot contract without its operator-only usage cap."""
    payload = scientific_run_config_payload(config)
    pilot = dict(config.get("behavioral_pilot", {}))
    pilot.pop("max_cost_usd", None)
    payload["behavioral_pilot"] = pilot
    return payload

def response_schema_hash(response_type: type) -> str:
    return sha256_text(canonical_json(response_type.model_json_schema()))


def observation_hash(observation: BureaucracyObservation | ManagerObservation) -> str:
    payload = {
        "agent_id": observation.agent_id,
        "month": observation.month,
        "phase": observation.phase,
        "identity_epoch": observation.identity_epoch,
        "prompt_text": observation.format_as_prompt_text(),
        "response_type": getattr(observation.response_type, "__name__", None),
    }
    return sha256_text(canonical_json(payload))


def cache_identity(
    *,
    observation: BureaucracyObservation | ManagerObservation,
    system_prompt: str,
    user_prompt: str,
    response_type: type,
    model: str,
    temperature: float,
    reasoning_effort: str,
) -> dict[str, Any]:
    fingerprints = runtime_fingerprints()
    return {
        **fingerprints,
        "observation_hash": observation_hash(observation),
        "system_prompt_hash": sha256_text(system_prompt),
        "user_prompt_hash": sha256_text(user_prompt),
        "response_schema_hash": response_schema_hash(response_type),
        "model": model,
        "temperature": float(temperature),
        "reasoning_effort": reasoning_effort,
        "output_cap_removed": True,
    }
