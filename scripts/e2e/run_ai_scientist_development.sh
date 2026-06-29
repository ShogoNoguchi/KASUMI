#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
"$ROOT/scripts/e2e/_guard_provider_calls.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
TEMPLATE="$WORKSPACE/AI-Scientist/templates/japan_policy_scientist"
RESULT_MANIFEST="$WORKSPACE/AI-Scientist/results/japan_policy_scientist/result_manifest.json"
[[ -f "$TEMPLATE/run_0/final_info.json" && -f "$TEMPLATE/run_0/complete.marker" ]] || { echo "Run baseline first." >&2; exit 1; }
[[ -f "$TEMPLATE/ideas.json" ]] || { echo "Run prepare_packaged_idea.sh first." >&2; exit 1; }

if [[ "${KASUMI_FORCE_AI_SCIENTIST:-0}" != "1" && -f "$RESULT_MANIFEST" ]]; then
  if python - "$RESULT_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root=Path(data.get('run_root',''))
ok=data.get('status') == 'complete' and data.get('success') is True and root.is_dir()
ok=ok and all((root/f'run_{i}'/'complete.marker').exists() for i in range(5))
raise SystemExit(0 if ok else 1)
PY
  then
    echo "Existing AI Scientist development result is complete; reusing it."
    echo "Result manifest: $RESULT_MANIFEST"
    exit 0
  fi
fi

export AI_SCIENTIST_MAX_RUNS=4
export AI_SCIENTIST_RUN_TIMEOUT_SECONDS="${AI_SCIENTIST_RUN_TIMEOUT_SECONDS:-172800}"
export POLICYLAB_AI_SCIENTIST_MODEL="${POLICYLAB_AI_SCIENTIST_MODEL:-gemini-2.5-pro}"
cd "$WORKSPACE/AI-Scientist"
python launch_scientist.py \
  --model "$POLICYLAB_AI_SCIENTIST_MODEL" \
  --experiment japan_policy_scientist \
  --skip-idea-generation \
  --skip-novelty-check \
  --parallel 0 \
  --writeup latex
[[ -f "$RESULT_MANIFEST" ]] || { echo "AI Scientist did not publish result_manifest.json" >&2; exit 1; }
echo "AI Scientist development complete: $RESULT_MANIFEST"
