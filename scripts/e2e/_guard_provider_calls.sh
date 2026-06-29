#!/usr/bin/env bash
set -euo pipefail
if [[ "${KASUMI_E2E_ALLOW_PROVIDER_CALLS:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
This stage can make paid provider/API calls.
Set KASUMI_E2E_ALLOW_PROVIDER_CALLS=1 after confirming your budget and API key.
MSG
  exit 2
fi
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before provider-backed E2E stages}"
