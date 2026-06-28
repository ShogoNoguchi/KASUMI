"""Optional batch/offline execution helpers for employee-month LLM actions.

current public release adds this module so large Shachi employee-month batches are not forced
through the synchronous Gemini live API.  The scientific environment still
advances one month at a time: all observations for one month/phase are batched,
validated, cached, and then Python performs the transition.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pydantic

from shachi.agent.japan_policy_bureaucrat import (
    BudgetExceededError,
    PreparedAgentRequest,
    SharedLLMRuntime,
)

from .schemas import BureaucracyObservation, ManagerObservation
from .validation import validate_response_against_observation


def _normalize_gemini_model(model: str) -> str:
    return model.removeprefix("gemini/")


def _batch_root_from_cache(cache: Any) -> Path:
    cache_root = getattr(cache, "root", None)
    if cache_root is None:
        return Path(os.environ.get("POLICYLAB_BATCH_ARTIFACT_ROOT", "batch_jobs"))
    return Path(cache_root).parent / "batch_jobs"


def _first_sample(requests: dict[int, PreparedAgentRequest]) -> PreparedAgentRequest:
    if not requests:
        raise ValueError("empty batch")
    return next(iter(requests.values()))


def _openai_batch_request_line(*, custom_id: str, request: PreparedAgentRequest, model: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": _normalize_gemini_model(model),
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
    }


async def resolve_employee_month_with_gemini_openai_batch(
    *,
    observations: dict[int, BureaucracyObservation | ManagerObservation],
    requests: dict[int, PreparedAgentRequest],
    runtime: SharedLLMRuntime,
    cache: Any,
) -> dict[int, pydantic.BaseModel]:
    """Resolve one uncached employee-month phase through Gemini OpenAI Batch.

    This function is intentionally conservative:
    * only BureaucracyObservation monthly_action batches are accepted;
    * manager calls, quarterly reflections, and small gates remain live/local;
    * every response is Pydantic/application validated before entering cache;
    * partial provider results are preserved as artifacts for human resume.
    """

    if not requests:
        return {}
    max_requests = int(os.environ.get("POLICYLAB_GEMINI_BATCH_MAX_REQUESTS", "200"))
    if len(requests) > max_requests:
        combined: dict[int, pydantic.BaseModel] = {}
        ordered = sorted(requests)
        for chunk_index, start in enumerate(range(0, len(ordered), max_requests), start=1):
            chunk_slots = ordered[start : start + max_requests]
            chunk_result = await resolve_employee_month_with_gemini_openai_batch(
                observations={slot_id: observations[slot_id] for slot_id in chunk_slots},
                requests={slot_id: requests[slot_id] for slot_id in chunk_slots},
                runtime=runtime,
                cache=cache,
            )
            combined.update(chunk_result)
        return {slot_id: combined[slot_id] for slot_id in sorted(combined)}
    sample = _first_sample(requests)
    if sample.observation.phase != "monthly_action":
        raise ValueError("Gemini batch backend is only allowed for employee monthly_action calls")
    if not all(isinstance(req.observation, BureaucracyObservation) for req in requests.values()):
        raise ValueError("Gemini batch backend is not used for manager observations")

    try:
        from google import genai
        from google.genai import types
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Gemini OpenAI Batch backend requires google-genai and openai packages. "
            "Install them or set llm.employee_monthly_backend=live/local_openai."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini Batch backend")

    batch_root = _batch_root_from_cache(cache)
    month = int(sample.observation.month)
    phase = str(sample.observation.phase)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    slot_values = sorted(requests)
    slot_span = f"slots_{slot_values[0]:04d}_{slot_values[-1]:04d}"
    job_dir = batch_root / f"month_{month:02d}_{phase}_{slot_span}_{stamp}"
    counter = 1
    while job_dir.exists():
        counter += 1
        job_dir = batch_root / f"month_{month:02d}_{phase}_{slot_span}_{stamp}_{counter}"
    job_dir.mkdir(parents=True, exist_ok=False)

    id_to_slot: dict[str, int] = {}
    request_path = job_dir / "openai_batch_requests.jsonl"
    with request_path.open("w", encoding="utf-8") as handle:
        for slot_id in sorted(requests):
            req = requests[slot_id]
            custom_id = f"slot_{slot_id}_epoch_{req.observation.identity_epoch}_month_{month:02d}_{phase}"
            id_to_slot[custom_id] = slot_id
            handle.write(
                json.dumps(
                    _openai_batch_request_line(custom_id=custom_id, request=req, model=runtime.model),
                    ensure_ascii=False,
                )
                + "\n"
            )
    (job_dir / "custom_id_to_slot.json").write_text(json.dumps(id_to_slot, indent=2), encoding="utf-8")

    if os.environ.get("POLICYLAB_GEMINI_BATCH_DRY_RUN_ONLY") == "1":
        raise RuntimeError(f"Gemini batch dry run wrote request file and stopped before upload: {request_path}")

    genai_client = genai.Client(api_key=api_key)
    openai_client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    uploaded_file = genai_client.files.upload(
        file=str(request_path),
        config=types.UploadFileConfig(display_name=request_path.name, mime_type="jsonl"),
    )
    input_file_id = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "id", None)
    if not input_file_id:
        raise RuntimeError(f"Could not determine uploaded Gemini batch file id: {uploaded_file!r}")

    batch = openai_client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=os.environ.get("POLICYLAB_GEMINI_BATCH_COMPLETION_WINDOW", "24h"),
    )
    job_id = getattr(batch, "id", None) or getattr(batch, "name", None)
    estimated_input_tokens = sum(max(1, (len(req.system) + len(req.user)) // 4) for req in requests.values())
    estimated_output_tokens = sum(int(req.output_token_estimate) for req in requests.values())
    metadata = {
        "backend": "gemini_openai_batch",
        "model": runtime.model,
        "month": month,
        "phase": phase,
        "request_count": len(requests),
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "input_file_id": input_file_id,
        "batch_id": job_id,
        "created_unix_time": time.time(),
        "max_requests_per_batch_file": int(os.environ.get("POLICYLAB_GEMINI_BATCH_MAX_REQUESTS", "200")),
    }
    (job_dir / "batch_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    poll_seconds = float(os.environ.get("POLICYLAB_GEMINI_BATCH_POLL_SECONDS", "60"))
    timeout_seconds = float(os.environ.get("POLICYLAB_GEMINI_BATCH_TIMEOUT_SECONDS", str(26 * 3600)))
    started = time.time()
    final_status = None
    while True:
        batch = openai_client.batches.retrieve(batch.id)
        status = str(getattr(batch, "status", "")).lower()
        final_status = status
        (job_dir / "last_batch_status.json").write_text(json.dumps(batch.model_dump() if hasattr(batch, "model_dump") else dict(batch), default=str, indent=2), encoding="utf-8")
        if status in {"completed", "failed", "cancelled", "expired"}:
            break
        if time.time() - started > timeout_seconds:
            raise RuntimeError(f"Gemini batch job timed out after {timeout_seconds}s; job_dir={job_dir}; status={status}")
        await asyncio.sleep(poll_seconds)

    if final_status != "completed":
        raise RuntimeError(f"Gemini batch job did not complete successfully: status={final_status}; job_dir={job_dir}")

    output_file_id = getattr(batch, "output_file_id", None) or getattr(getattr(batch, "output_file", None), "id", None)
    if not output_file_id:
        # OpenAI SDK may expose output_file_id only in model_dump.
        dumped = batch.model_dump() if hasattr(batch, "model_dump") else {}
        output_file_id = dumped.get("output_file_id") or dumped.get("output_file", {}).get("id")
    if not output_file_id:
        raise RuntimeError(f"Completed batch lacks output_file_id; job_dir={job_dir}")

    raw_output = genai_client.files.download(file=output_file_id)
    if isinstance(raw_output, bytes):
        output_text = raw_output.decode("utf-8")
    else:
        output_text = str(raw_output)
    output_path = job_dir / "openai_batch_output.jsonl"
    output_path.write_text(output_text, encoding="utf-8")

    responses: dict[int, pydantic.BaseModel] = {}
    failures: list[str] = []
    for line_no, line in enumerate(output_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            custom_id = row["custom_id"]
            slot_id = id_to_slot[custom_id]
            req = requests[slot_id]
            response = row.get("response") or {}
            if int(response.get("status_code", 0)) != 200:
                failures.append(f"line {line_no} slot {slot_id}: status={response.get('status_code')} error={row.get('error')}")
                continue
            body = response.get("body") or {}
            message = body["choices"][0]["message"]
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            parsed = SharedLLMRuntime._parse_response_model(content, req.response_type)
            req_validator = getattr(req, "validator", None)
            validate_response_against_observation(observations[slot_id], parsed)
            responses[slot_id] = parsed
            usage = body.get("usage") or {}
            reservation_id = await runtime.ledger.reserve(
                model="gemini_openai_batch",
                estimated_input_tokens=int(usage.get("prompt_tokens") or max(1, (len(req.system) + len(req.user)) // 4)),
                estimated_output_tokens=int(usage.get("completion_tokens") or req.output_token_estimate),
            )
            await runtime.ledger.settle(
                reservation_id=reservation_id,
                model=f"gemini_openai_batch:{runtime.model}",
                phase=req.observation.phase,
                slot_id=slot_id,
                month=req.observation.month,
                network_attempt=1,
                validation_attempt=1,
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                provider_success=True,
                parse_success=True,
                validation_success=True,
                failure_kind=None,
                response_text=content,
            )
        except BudgetExceededError:
            raise
        except Exception as exc:
            failures.append(f"line {line_no}: {type(exc).__name__}: {exc}")

    missing = sorted(set(requests) - set(responses))
    if missing or failures:
        (job_dir / "batch_failures.json").write_text(
            json.dumps({"missing_slots": missing, "failures": failures}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Gemini batch returned incomplete/invalid responses; job_dir={job_dir}; "
            f"missing={missing[:10]}; failures={failures[:5]}"
        )

    return {slot_id: responses[slot_id] for slot_id in sorted(responses)}
