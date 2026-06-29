#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
SHACHI_DIR="$WORKSPACE/shachi"
AI_DIR="$WORKSPACE/AI-Scientist"

[[ -f "$SHACHI_DIR/pyproject.toml" ]] || { echo "Missing Shachi clone at $SHACHI_DIR" >&2; exit 1; }
[[ -f "$AI_DIR/launch_scientist.py" ]] || { echo "Missing AI Scientist clone at $AI_DIR" >&2; exit 1; }

# KASUMI-specific Shachi task surface.
mkdir -p "$SHACHI_DIR/src/shachi/agent" "$SHACHI_DIR/src/shachi/env/japan_policy_scientist" "$SHACHI_DIR/scripts"
cp -a "$ROOT/src/shachi/agent/japan_policy_bureaucrat.py" "$SHACHI_DIR/src/shachi/agent/"
cp -a "$ROOT/src/shachi/env/japan_policy_scientist/." "$SHACHI_DIR/src/shachi/env/japan_policy_scientist/"
cp -a "$ROOT/scripts/e2e/shachi_run_japan_policy_scientist.py" "$SHACHI_DIR/scripts/run_japan_policy_scientist.py"
cp -a "$ROOT/scripts/e2e/shachi_run_japan_policy_behavioral_pilot.py" "$SHACHI_DIR/scripts/run_japan_policy_behavioral_pilot.py"
cp -a "$ROOT/scripts/e2e/shachi_run_japan_policy_identity_gate.py" "$SHACHI_DIR/scripts/run_japan_policy_identity_gate.py"

# KASUMI-specific AI Scientist task template.
mkdir -p "$AI_DIR/templates/japan_policy_scientist" "$AI_DIR/ai_scientist"
cp -a "$ROOT/integrations/ai_scientist_template/public_service_policy_lab/." "$AI_DIR/templates/japan_policy_scientist/"
cp -a "$ROOT/third_party/ai_scientist/policy_context.py" "$AI_DIR/ai_scientist/policy_context.py"

# Patch upstream AI Scientist v1 so protected public-administration templates can
# supply structured idea prompts, source cards, and validation hooks.
if ! grep -q "enrich_task_description" "$AI_DIR/ai_scientist/generate_ideas.py"; then
  (cd "$AI_DIR" && patch --forward --batch -p1 < "$ROOT/third_party/ai_scientist/ai_scientist_v1_policy.patch")
else
  echo "AI Scientist policy-template patch already present; skipping."
fi

# Full E2E reproduction uses the frozen multiseed holdout cells.  The published
# paper was generated after selection and holdout; public replay remains separate.
python - "$AI_DIR/templates/japan_policy_scientist/japan_policy_scientist.yaml" <<'PY'
from pathlib import Path
import sys
import yaml
path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8"))
exp = data.setdefault("experiment", {})
exp["holdout_scenarios"] = ["reference_stressed"]
exp["holdout_seeds"] = [20260631, 20260637, 20260641]
exp["fast_track_no_holdout"] = False
exp["scale_profile"] = "adaptive_120slot_48month_multiseed_holdout"
data.setdefault("urgent_scale_profile", {})["holdout_disabled"] = False
# Keep public wording neutral and reproducibility-oriented.
data.setdefault("package", {})["profile"] = "adaptive_api_local_multiseed_holdout"
path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

echo "Applied KASUMI E2E overlays to:"
echo "  $SHACHI_DIR"
echo "  $AI_DIR"
