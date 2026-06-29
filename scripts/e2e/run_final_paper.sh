#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
MANIFEST="${2:-${KASUMI_RESULT_MANIFEST:-$WORKSPACE/AI-Scientist/results/japan_policy_scientist/result_manifest.json}}"
"$ROOT/scripts/e2e/_guard_provider_calls.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
[[ -f "$MANIFEST" ]] || { echo "Missing result manifest: $MANIFEST" >&2; exit 1; }
RUN_ROOT="$(python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root=Path(data.get('run_root','')).resolve()
if not root.is_dir():
    raise SystemExit('run_root is missing')
print(root)
PY
)"
[[ -f "$RUN_ROOT/selection_freeze.json" ]] || { echo "Run selection_and_holdout first." >&2; exit 1; }
[[ -f "$RUN_ROOT/holdout_complete.json" ]] || { echo "Run frozen holdout before final paper." >&2; exit 1; }
MODEL="${KASUMI_FINAL_PAPER_MODEL:-gemini-2.5-pro}"
ENGINE="${KASUMI_LITERATURE_ENGINE:-semanticscholar}"
python "$RUN_ROOT/finalize_policy_paper.py" --run-root "$RUN_ROOT" --result-manifest "$MANIFEST" --model "$MODEL" --engine "$ENGINE"
echo "Final paper complete: $RUN_ROOT/final_policy_paper.pdf"
