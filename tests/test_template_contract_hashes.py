from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_template_contract_hashes_match_current_protected_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    template = root / "integrations" / "ai_scientist_template" / "public_service_policy_lab"
    contract = (template / "template_contract.py").read_text(encoding="utf-8")

    expected_experiment = re.search(r'EXPECTED_EXPERIMENT_SHA256 = "([0-9a-f]{64})"', contract)
    expected_plot = re.search(r'EXPECTED_PLOT_SHA256 = "([0-9a-f]{64})"', contract)

    assert expected_experiment is not None
    assert expected_plot is not None
    assert expected_experiment.group(1) == _sha(template / "experiment.py")
    assert expected_plot.group(1) == _sha(template / "plot.py")
