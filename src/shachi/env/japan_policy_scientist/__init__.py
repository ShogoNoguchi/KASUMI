"""Public API for the Japan Policy Scientist Shachi environment.

The runner is imported lazily so ``shachi.agent.japan_policy_bureaucrat`` can be
imported independently without an agent -> env package -> runner -> agent cycle.
"""
from __future__ import annotations

from typing import Any

from .environment import JapanPolicyLabEnv
from .schemas import (
    BureaucracyObservation,
    BureaucratMonthlyAction,
    BureaucratQuarterlyReflection,
    PolicyConfig,
    PolicyLabResult,
)


def run_policy_experiment(*args: Any, **kwargs: Any) -> dict[str, float]:
    from .runner import run_policy_experiment as _run_policy_experiment

    return _run_policy_experiment(*args, **kwargs)


__all__ = [
    "JapanPolicyLabEnv",
    "run_policy_experiment",
    "BureaucracyObservation",
    "BureaucratMonthlyAction",
    "BureaucratQuarterlyReflection",
    "PolicyConfig",
    "PolicyLabResult",
]
