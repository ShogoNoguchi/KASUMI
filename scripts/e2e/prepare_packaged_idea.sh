#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
TEMPLATE="$WORKSPACE/AI-Scientist/templates/japan_policy_scientist"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
python "$ROOT/scripts/e2e/prepare_packaged_idea.py" "$TEMPLATE"
