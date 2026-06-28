"""Fail-fast validation for scientific environment overrides.

The package intentionally accepts environment variables for test fixtures and the
predeclared holdout.  In paid development runs, however, stale shell variables
must not silently change the fixed seed, population, horizon, scenario, or
management mode.  This module rejects such drift before any provider call.
"""
from __future__ import annotations

import os
from typing import Any

from .population import normalize_scenario


_INT_OVERRIDES: dict[str, str] = {
    "POLICYLAB_FIXED_SEED": "fixed_seed",
    "POLICYLAB_NUM_AGENTS": "num_agents",
    "POLICYLAB_MONTHS": "months",
    "POLICYLAB_WARMUP_MONTHS": "warmup_months",
    "POLICYLAB_INTERVENTION_START_MONTH": "intervention_start_month",
}


def _manager_mode(config: dict[str, Any]) -> str:
    return str(config.get("management", {}).get("mode", "deterministic_priority"))


def validate_scientific_environment(
    config: dict[str, Any],
    *,
    allow_holdout: bool = False,
) -> dict[str, Any]:
    """Validate paid-run overrides against the config-owned experiment contract.

    A general bypass is available only when both ``POLICYLAB_ALLOW_TEST_MODE=1``
    and ``POLICYLAB_MOCK_LLM=1`` are set, so a stale test flag can never authorize
    a paid provider call.  A holdout caller may set ``allow_holdout=True``; then only the
    already-declared holdout seeds and scenarios may differ from development.
    All other scientific dimensions remain fixed.
    """

    fixed = config["experiment"]
    test_mode = os.environ.get("POLICYLAB_ALLOW_TEST_MODE") == "1"
    mock_mode = os.environ.get("POLICYLAB_MOCK_LLM") == "1"
    if test_mode != mock_mode:
        raise RuntimeError(
            "Test-contract bypass requires POLICYLAB_ALLOW_TEST_MODE=1 and "
            "POLICYLAB_MOCK_LLM=1 together; refusing a mixed paid/test state "
            "before any provider call"
        )

    raw_feedback = os.environ.get("POLICYLAB_ENABLE_FEEDBACK")
    if raw_feedback not in (None, "1") and not (test_mode and mock_mode):
        raise RuntimeError(
            "POLICYLAB_ENABLE_FEEDBACK must be 1 in paid execution; refusing to "
            "disable exits, transfers, hiring, or the finite management gate before "
            "any provider call. Feedback-off fixtures require both "
            "POLICYLAB_ALLOW_TEST_MODE=1 and POLICYLAB_MOCK_LLM=1."
        )

    if test_mode and mock_mode:
        return {
            "status": "test_mode_bypass",
            "provider_calls_before_validation": 0,
            "allow_holdout": allow_holdout,
            "feedback_override": raw_feedback,
        }

    observed: dict[str, Any] = {}
    if raw_feedback is not None:
        observed["POLICYLAB_ENABLE_FEEDBACK"] = raw_feedback
    mismatches: list[dict[str, Any]] = []

    holdout_seeds = {int(value) for value in fixed.get("holdout_seeds", [])}
    holdout_scenarios = {
        normalize_scenario(str(value)) for value in fixed.get("holdout_scenarios", [])
    }

    for env_name, config_key in _INT_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            actual = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{env_name} must be an integer, got {raw!r}") from exc
        expected = int(fixed[config_key])
        observed[env_name] = actual
        allowed = actual == expected
        if allow_holdout and config_key == "fixed_seed":
            allowed = actual in holdout_seeds
        if not allowed:
            mismatches.append(
                {
                    "environment_variable": env_name,
                    "actual": actual,
                    "expected": expected,
                    "holdout_allowed": sorted(holdout_seeds)
                    if allow_holdout and config_key == "fixed_seed"
                    else None,
                }
            )

    scenario_values = {
        name: os.environ[name]
        for name in ("POLICYLAB_SCENARIO", "POLICYLAB_WORLD")
        if name in os.environ
    }
    normalized_scenarios = {
        name: normalize_scenario(value) for name, value in scenario_values.items()
    }
    if len(set(normalized_scenarios.values())) > 1:
        mismatches.append(
            {
                "environment_variable": "POLICYLAB_SCENARIO/POLICYLAB_WORLD",
                "actual": normalized_scenarios,
                "expected": normalize_scenario(str(fixed.get("scenario", "reference_stressed"))),
                "holdout_allowed": sorted(holdout_scenarios) if allow_holdout else None,
            }
        )
    elif normalized_scenarios:
        actual_scenario = next(iter(normalized_scenarios.values()))
        expected_scenario = normalize_scenario(
            str(fixed.get("scenario", "reference_stressed"))
        )
        observed.update(normalized_scenarios)
        allowed = actual_scenario == expected_scenario
        if allow_holdout:
            allowed = actual_scenario in holdout_scenarios
        if not allowed:
            mismatches.append(
                {
                    "environment_variable": "/".join(normalized_scenarios),
                    "actual": actual_scenario,
                    "expected": expected_scenario,
                    "holdout_allowed": sorted(holdout_scenarios)
                    if allow_holdout
                    else None,
                }
            )

    raw_manager = os.environ.get("POLICYLAB_MANAGER_MODE")
    if raw_manager is not None:
        expected_manager = _manager_mode(config)
        observed["POLICYLAB_MANAGER_MODE"] = raw_manager
        if raw_manager != expected_manager:
            mismatches.append(
                {
                    "environment_variable": "POLICYLAB_MANAGER_MODE",
                    "actual": raw_manager,
                    "expected": expected_manager,
                    "holdout_allowed": None,
                }
            )

    if mismatches:
        raise RuntimeError(
            "Fixed scientific contract override rejected before provider call: "
            + repr(mismatches)
            + ". Unset stale POLICYLAB_* variables. Test overrides require "
            "POLICYLAB_ALLOW_TEST_MODE=1 together with POLICYLAB_MOCK_LLM=1; "
            "holdout seed/scenario overrides are "
            "created only by the frozen holdout runner."
        )

    return {
        "status": "passed",
        "provider_calls_before_validation": 0,
        "allow_holdout": allow_holdout,
        "observed_equal_overrides": observed,
    }
