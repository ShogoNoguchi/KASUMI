#!/usr/bin/env python3
"""Run the six-vignette bureaucratic institutional-identity gate.

KASUMI live mode asks for one vignette response per provider call and lets
Python assemble the final ``IdentityGateBatch``. This removes the old
all-or-nothing six-response JSON object and does not pass a provider-side hard
output-token cap. Schema/application validation, not token clipping, constrains
the content.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pydantic
import yaml

from shachi.agent.japan_policy_bureaucrat import UsageLedger
from shachi.env.japan_policy_scientist.bureaucratic_identity_gate import (
    VIGNETTES,
    IdentityGateBatch,
    IdentityVignette,
    IdentityVignetteResponse,
    canonical_mock_batch,
    evaluate_identity_gate,
    single_vignette_response_prompt,
    validate_single_response,
)
from shachi.env.japan_policy_scientist.fingerprints import runtime_fingerprints


def _response_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    parsed_obj = getattr(message, "parsed", None)
    if parsed_obj is not None:
        if isinstance(parsed_obj, pydantic.BaseModel):
            return parsed_obj.model_dump_json()
        return json.dumps(parsed_obj, ensure_ascii=False, default=str)
    raise ValueError(f"No structured content in response: {message!r}")


def _repair_prompt(*, original_prompt: str, error: Exception, response_text: str | None) -> str:
    diagnostic = "" if response_text is None else "\n\nPrevious invalid response was:\n" + response_text
    return (
        original_prompt
        + "\n\nCORRECTION REQUIRED: The previous provider response was not a complete valid JSON "
        + f"object for this one vignette: {type(error).__name__}: {error}. "
        + "Return exactly one complete JSON object only. Do not include markdown or prose outside JSON. "
        + "Preserve the same vignette_id and choose only allowed values."
        + diagnostic
    )


def _provider_single_vignette(
    *,
    litellm: Any,
    model: str,
    temperature: float,
    vignette: IdentityVignette,
    validation_retries: int,
    output_token_estimate: int,
    ledger: UsageLedger,
) -> IdentityVignetteResponse:
    base_prompt = single_vignette_response_prompt(vignette)
    prompt = base_prompt
    response_text: str | None = None
    last_error: Exception | None = None
    for validation_attempt in range(1, validation_retries + 2):
        reservation_id = asyncio.run(
            ledger.reserve(
                model=model,
                estimated_input_tokens=max(1, len(prompt) // 4),
                estimated_output_tokens=output_token_estimate,
            )
        )
        try:
            completion = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only the requested structured JSON object. Respect formal public-service "
                            "duties and do not infer hidden scoring."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format=IdentityVignetteResponse,
                drop_params=True,
            )
        except Exception as exc:
            asyncio.run(
                ledger.settle(
                    reservation_id=reservation_id,
                    model=model,
                    phase=f"bureaucratic_identity_gate:{vignette.vignette_id}",
                    slot_id=-1,
                    month=0,
                    network_attempt=1,
                    validation_attempt=validation_attempt,
                    input_tokens=0,
                    output_tokens=0,
                    provider_success=False,
                    parse_success=False,
                    validation_success=False,
                    failure_kind="provider",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

        provider_usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(provider_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(provider_usage, "completion_tokens", 0) or 0)
        parse_success = False
        validation_success = False
        try:
            response_text = _response_text(completion.choices[0].message)
            response = IdentityVignetteResponse.model_validate_json(response_text)
            parse_success = True
            validate_single_response(response, vignette)
            validation_success = True
            asyncio.run(
                ledger.settle(
                    reservation_id=reservation_id,
                    model=model,
                    phase=f"bureaucratic_identity_gate:{vignette.vignette_id}",
                    slot_id=-1,
                    month=0,
                    network_attempt=1,
                    validation_attempt=validation_attempt,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    provider_success=True,
                    parse_success=True,
                    validation_success=True,
                    failure_kind=None,
                    response_text=response_text,
                )
            )
            return response
        except Exception as exc:
            last_error = exc
            asyncio.run(
                ledger.settle(
                    reservation_id=reservation_id,
                    model=model,
                    phase=f"bureaucratic_identity_gate:{vignette.vignette_id}",
                    slot_id=-1,
                    month=0,
                    network_attempt=1,
                    validation_attempt=validation_attempt,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    provider_success=True,
                    parse_success=parse_success,
                    validation_success=validation_success,
                    failure_kind="validation",
                    error=f"{type(exc).__name__}: {exc}",
                    response_text=response_text,
                )
            )
            if validation_attempt <= validation_retries:
                prompt = _repair_prompt(
                    original_prompt=base_prompt,
                    error=exc,
                    response_text=response_text,
                )
                continue
            raise RuntimeError(
                f"Identity gate vignette {vignette.vignette_id} failed after "
                f"{validation_retries + 1} provider responses: {last_error}"
            ) from exc
    raise RuntimeError(f"identity gate loop ended unexpectedly for {vignette.vignette_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    minimum = float(config["bureaucratic_identity_gate"]["minimum_pass_rate"])
    model = os.environ.get("POLICYLAB_SHACHI_MODEL", config["llm"]["model"])
    evidence_mode = "mock" if os.environ.get("POLICYLAB_MOCK_LLM") == "1" else "live"

    if evidence_mode == "mock":
        if os.environ.get("POLICYLAB_ALLOW_TEST_MODE") != "1":
            raise RuntimeError("Mock identity gate is test-only")
        batch = canonical_mock_batch()
        usage = {"mock": True, "provider_attempts": 0, "provider_output_hard_cap_used": False}
    else:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("Install LiteLLM before running the provider identity gate") from exc
        audit_dir = args.output_dir / ".operator_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        ledger = UsageLedger(
            audit_dir / "llm_usage.jsonl",
            max_cost_usd=float(os.environ.get("POLICYLAB_OPERATOR_HARD_STOP_USD", "300.0")),
            campaign_log_path=os.environ.get("POLICYLAB_CAMPAIGN_LEDGER"),
            campaign_id=os.environ.get("POLICYLAB_CAMPAIGN_ID"),
        )
        validation_retries = int(config["llm"].get("validation_retries", 1))
        output_token_estimate = int(
            config["bureaucratic_identity_gate"].get("output_token_estimate_per_vignette", 900)
        )
        responses: list[IdentityVignetteResponse] = []
        partial_path = args.output_dir / "identity_gate_partial_responses.jsonl"
        if partial_path.exists():
            partial_path.unlink()
        for vignette in VIGNETTES:
            response = _provider_single_vignette(
                litellm=litellm,
                model=model,
                temperature=float(config["llm"]["temperature"]),
                vignette=vignette,
                validation_retries=validation_retries,
                output_token_estimate=output_token_estimate,
                ledger=ledger,
            )
            responses.append(response)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(response.model_dump_json() + "\n")
        batch = IdentityGateBatch(responses=responses)
        usage = {
            "mock": False,
            **ledger.snapshot(),
            "per_vignette_calls": True,
            "provider_output_hard_cap_used": False,
            "per_vignette_no_provider_output_cap": True,
        }

    report = evaluate_identity_gate(batch, minimum_pass_rate=minimum)
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    marker = {
        **report,
        "model": model,
        "config_sha256": config_hash,
        "fingerprints": runtime_fingerprints(),
        "evidence_mode": evidence_mode,
        "usage": usage,
        "provider_output_hard_cap_used": False,
        "per_vignette_no_provider_output_cap": True,
        "batch_generation_mode": "per_vignette_python_assembled",
    }
    (args.output_dir / "identity_gate_responses.json").write_text(
        batch.model_dump_json(indent=2), encoding="utf-8"
    )
    (args.output_dir / "bureaucratic_identity_gate_report.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["passed"]:
        (args.output_dir / "bureaucratic_identity_gate.pass.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        raise SystemExit("bureaucratic institutional-identity gate failed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
