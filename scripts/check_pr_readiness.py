#!/usr/bin/env python3
"""Small PR-readiness check for the packaged task extension.

It checks that the public repository exposes a minimal task surface for a future
upstream template contribution without bundling private run artifacts.
"""
from __future__ import annotations

from pathlib import Path
import json

REQUIRED = [
    "integrations/ai_scientist_template/public_service_policy_lab/experiment.py",
    "integrations/ai_scientist_template/public_service_policy_lab/plot.py",
    "integrations/ai_scientist_template/public_service_policy_lab/selection_and_holdout.py",
    "integrations/ai_scientist_template/public_service_policy_lab/claim_verifier.py",
    "src/shachi/env/japan_policy_scientist/environment.py",
    "src/shachi/env/japan_policy_scientist/runner.py",
    "src/shachi/agent/japan_policy_bureaucrat.py",
]


def main() -> None:
    root = Path.cwd()
    result = {path: (root / path).is_file() for path in REQUIRED}
    missing = [path for path, ok in result.items() if not ok]
    print(json.dumps({"required_files": result, "missing": missing}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
