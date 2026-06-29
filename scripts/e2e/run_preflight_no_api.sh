#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
if [[ -f "$WORKSPACE/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$WORKSPACE/.venv/bin/activate"
fi
python -m compileall -q \
  "$WORKSPACE/shachi/src/shachi/agent/japan_policy_bureaucrat.py" \
  "$WORKSPACE/shachi/src/shachi/env/japan_policy_scientist" \
  "$WORKSPACE/shachi/scripts/run_japan_policy_scientist.py" \
  "$WORKSPACE/shachi/scripts/run_japan_policy_behavioral_pilot.py" \
  "$WORKSPACE/shachi/scripts/run_japan_policy_identity_gate.py" \
  "$WORKSPACE/AI-Scientist/templates/japan_policy_scientist" \
  "$ROOT/scripts/e2e"
for script in "$ROOT"/scripts/e2e/*.sh; do bash -n "$script"; done
python - <<'PY'
import importlib.util
for name in ["pydantic", "yaml", "matplotlib", "networkx"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing import: {name}")
PY
for asset in fancyhdr.sty iclr2024_conference.bst iclr2024_conference.sty natbib.sty; do
  [[ -f "$WORKSPACE/AI-Scientist/templates/japan_policy_scientist/latex/$asset" ]] || {
    echo "Missing policy-template LaTeX asset after overlay: $asset" >&2
    exit 1
  }
done
python "$ROOT/scripts/audit_public_release.py" "$ROOT"
python "$ROOT/scripts/check_pr_readiness.py" >/dev/null
python -m pytest -q "$ROOT/tests"
echo "KASUMI no-API E2E preflight passed. No provider call was made."
