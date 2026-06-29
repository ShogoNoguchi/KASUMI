#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
"$ROOT/scripts/e2e/_guard_provider_calls.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
TEMPLATE="$WORKSPACE/AI-Scientist/templates/japan_policy_scientist"
cd "$TEMPLATE"
if [[ -e run_0/complete.marker ]]; then
  echo "run_0 already complete; validating and reusing it."
  python -c "from template_contract import validate_run; validate_run('run_0', run_num=0)"
  exit 0
fi
export POLICYLAB_SHACHI_MODEL="${POLICYLAB_SHACHI_MODEL:-gemini/gemini-2.5-flash-lite}"
mkdir -p run_0
rm -f run_0/README.md
python experiment.py --out_dir=run_0
python plot.py
python -c "from template_contract import validate_run; validate_run('run_0', run_num=0)"
echo "Baseline complete: $TEMPLATE/run_0/final_info.json"
