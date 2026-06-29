#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
cat <<MSG
KASUMI full provider-backed E2E pipeline:
  workspace: $WORKSPACE
  1. bootstrap pinned upstream AI Scientist + Shachi and apply KASUMI overlays
  2. no-API preflight
  3. Shachi baseline with Gemini Flash-Lite by default
  4. prepare packaged mechanism-portfolio idea without live idea generation
  5. AI Scientist development with Gemini Pro by default, producing run_1..run_4
  6. selection freeze + frozen multiseed holdout
  7. final paper generation
  8. automated domain review

This script runs paid/API stages only when KASUMI_E2E_ALLOW_PROVIDER_CALLS=1 and GEMINI_API_KEY is set.
MSG
"$ROOT/scripts/e2e/bootstrap_workspace.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/run_preflight_no_api.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/run_baseline.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/prepare_packaged_idea.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/run_ai_scientist_development.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/run_selection_and_holdout.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/run_final_paper.sh" "$WORKSPACE"
"$ROOT/scripts/e2e/run_domain_review.sh" "$WORKSPACE"
MSG2="$(cat <<'MSG'
Full E2E pipeline complete. Inspect:
  $WORKSPACE/AI-Scientist/results/japan_policy_scientist/result_manifest.json
MSG
)"
echo "$MSG2"
