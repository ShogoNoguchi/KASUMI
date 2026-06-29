#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
MANIFEST="${2:-${KASUMI_RESULT_MANIFEST:-$WORKSPACE/AI-Scientist/results/japan_policy_scientist/result_manifest.json}}"
"$ROOT/scripts/e2e/_guard_provider_calls.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
readarray -t ROUTED < <(python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root=Path(data.get('run_root','')).resolve()
final=root/'final_paper_manifest.json'
if not final.is_file():
    raise SystemExit('Run final paper first')
fm=json.loads(final.read_text(encoding='utf-8'))
paper=Path(fm.get('final_paper_path','')).resolve()
claims=Path(fm.get('claim_verification_path','')).resolve()
if fm.get('status') != 'complete' or not paper.is_file() or not claims.is_file():
    raise SystemExit('Final-paper manifest is incomplete')
print(root)
print(paper)
print(claims)
print(final)
PY
)
RUN_ROOT="${ROUTED[0]}"
PAPER="${ROUTED[1]}"
CLAIMS="${ROUTED[2]}"
FINAL_MANIFEST="${ROUTED[3]}"
MODEL_A="${KASUMI_SCIENTIFIC_REVIEW_MODEL:-gemini/gemini-2.5-pro}"
MODEL_B="${KASUMI_PUBLIC_ADMIN_REVIEW_MODEL:-gemini/gemini-2.5-flash}"
python "$RUN_ROOT/review_policy_paper.py" \
  --paper "$PAPER" \
  --claim-verification "$CLAIMS" \
  --final-paper-manifest "$FINAL_MANIFEST" \
  --output "$RUN_ROOT/domain_reviews.json" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B"
echo "Domain review complete: $RUN_ROOT/domain_reviews.json"
