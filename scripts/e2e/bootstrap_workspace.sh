#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${1:-${KASUMI_E2E_WORKSPACE:-$HOME/kasumi-e2e-workspace}}"
SHACHI_REF="${SHACHI_REF:-c6c8f2232948a0a16bd8c08a6d3654d892acd3dc}"
AI_SCIENTIST_REF="${AI_SCIENTIST_REF:-1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb}"

mkdir -p "$WORKSPACE"
for command in git curl patch; do
  command -v "$command" >/dev/null || { echo "Install required command: $command" >&2; exit 1; }
done
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

checkout_upstream() {
  local repo_url="$1" destination="$2" ref="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone "$repo_url" "$destination"
    git -C "$destination" checkout --detach "$ref"
    return
  fi
  local current_ref
  current_ref="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$current_ref" == "$ref" ]]; then
    return
  fi
  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "Refusing to switch dirty checkout $destination from $current_ref to $ref" >&2
    echo "Use a new KASUMI_E2E_WORKSPACE or preserve/clean the workspace manually." >&2
    exit 1
  fi
  git -C "$destination" fetch --tags origin
  git -C "$destination" checkout --detach "$ref"
}

checkout_upstream https://github.com/SakanaAI/shachi.git "$WORKSPACE/shachi" "$SHACHI_REF"
checkout_upstream https://github.com/SakanaAI/AI-Scientist.git "$WORKSPACE/AI-Scientist" "$AI_SCIENTIST_REF"

"$ROOT/scripts/e2e/apply_overlays.sh" "$WORKSPACE"

uv python install 3.11
uv venv --python 3.11 "$WORKSPACE/.venv"
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
uv pip install --python "$WORKSPACE/.venv/bin/python" --upgrade pip wheel setuptools
uv pip install --python "$WORKSPACE/.venv/bin/python" -r "$WORKSPACE/AI-Scientist/requirements.txt"
uv pip install --python "$WORKSPACE/.venv/bin/python" --no-deps -e "$WORKSPACE/shachi"
uv pip install --python "$WORKSPACE/.venv/bin/python" \
  'pydantic>=2,<3' 'PyYAML>=6' 'litellm>=1.61.3' 'matplotlib>=3.8' \
  'numpy>=1.26' 'networkx>=3.2,<4' 'google-genai>=2.10.0' 'openai==2.20.0'

cat <<MSG
KASUMI E2E workspace bootstrapped at: $WORKSPACE
Pinned upstream revisions:
  Shachi: $SHACHI_REF
  AI Scientist v1: $AI_SCIENTIST_REF

Next:
  source "$WORKSPACE/.venv/bin/activate"
  export GEMINI_API_KEY='...'
  cp "$ROOT/configs/operator_budget_plan.template.json" "$WORKSPACE/operator_budget_plan.json"
  edit "$WORKSPACE/operator_budget_plan.json"
  "$ROOT/scripts/e2e/run_preflight_no_api.sh" "$WORKSPACE"
MSG
