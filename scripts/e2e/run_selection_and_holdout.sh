#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
MANIFEST="${2:-${KASUMI_RESULT_MANIFEST:-$WORKSPACE/AI-Scientist/results/japan_policy_scientist/result_manifest.json}}"
"$ROOT/scripts/e2e/_guard_provider_calls.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
[[ -f "$MANIFEST" ]] || { echo "Missing AI Scientist result manifest: $MANIFEST" >&2; exit 1; }
RUN_ROOT="$(python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root=Path(data.get('run_root','')).resolve()
if data.get('status') != 'complete' or not data.get('success') or not root.is_dir():
    raise SystemExit('AI Scientist development result manifest is not complete')
print(root)
PY
)"
for run in 0 1 2 3 4; do
  [[ -f "$RUN_ROOT/run_${run}/complete.marker" ]] || { echo "Missing complete run_$run under $RUN_ROOT" >&2; exit 1; }
done
cd "$RUN_ROOT"
python selection_and_holdout.py --run-root "$RUN_ROOT" --run-holdout
python verified_results.py
python claim_verifier.py --run-root "$RUN_ROOT" --claims "$RUN_ROOT/verified_claims.json" --output "$RUN_ROOT/claim_verification.json"
echo "Selection freeze and frozen multiseed holdout complete: $RUN_ROOT"
