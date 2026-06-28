"""Shachi-native structured-output employee and manager agents for current public release.

current public release removes provider-side hard output-token caps from scientific LLM calls.
The schema and application validators constrain form and semantics; operator
budget reservations estimate possible cost but do not truncate model output.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import pydantic

from shachi import Agent, Observation
from shachi.env.japan_policy_scientist.fingerprints import cache_identity
from shachi.env.japan_policy_scientist.memory import BureaucratMemory, ManagerFactMemory
from shachi.env.japan_policy_scientist.prompting import (
    BUREAUCRAT_SYSTEM_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    build_user_prompt,
)
from shachi.env.japan_policy_scientist.schemas import (
    BureaucracyObservation,
    BureaucratMonthlyAction,
    BureaucratQuarterlyReflection,
    DEPARTMENTS,
    ManagerDecision,
    ManagerObservation,
    ManagerRequestDecision,
    MonthlySelfReport,
    TransferPreference,
    WorkMix,
)
from shachi.env.japan_policy_scientist.validation import (
    validate_response_against_observation,
)

T = TypeVar("T", bound=pydantic.BaseModel)

MODEL_PRICES_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "gemini/gemini-2.5-flash-lite": (0.10, 0.40),
    "local/openai-compatible": (0.0, 0.0),
}


class BudgetExceededError(RuntimeError):
    """Raised when actual plus reserved spend would exceed the audited cap."""


class StructuredValidationError(RuntimeError):
    """Provider returned data, but application/schema validation never succeeded."""


@dataclass(frozen=True)
class PreparedAgentRequest:
    observation: BureaucracyObservation | ManagerObservation
    system: str
    user: str
    response_type: type[pydantic.BaseModel]
    output_token_estimate: int
    cache_identity: dict[str, Any]


class UsageLedger:
    """Provider-attempt ledger with an optional campaign-wide hard stop.

    The local JSONL preserves per-stage audit details.  The campaign JSONL is a
    process-safe reservation/settlement journal shared by Shachi identity,
    behavioral, pilot, development, and holdout calls.  It does not meter the
    upstream AI Scientist's ideation, Aider, write-up, or review calls.  A
    reservation is written before each observed Shachi call, so independent
    Shachi processes cannot all pass the same budget check.
    """

    def __init__(
        self,
        log_path: str | Path | None,
        max_cost_usd: float,
        *,
        campaign_log_path: str | Path | None = None,
        campaign_id: str | None = None,
    ):
        self.log_path = Path(log_path) if log_path else None
        configured_campaign = campaign_log_path or os.environ.get("POLICYLAB_CAMPAIGN_LEDGER")
        self.campaign_log_path = Path(configured_campaign) if configured_campaign else None
        self.campaign_id = campaign_id or os.environ.get("POLICYLAB_CAMPAIGN_ID", "default")
        self.max_cost_usd = max_cost_usd
        self.provider_attempts = 0
        self.valid_responses = 0
        self.network_failures = 0
        self.provider_failures = 0
        self.validation_failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self._reserved_cost_usd = 0.0
        self._reservations: dict[str, float] = {}
        self._reservation_counter = 0
        self._lock = asyncio.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch(exist_ok=True)
            self._load_local_rows()
        if self.campaign_log_path:
            self.campaign_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.campaign_log_path.touch(exist_ok=True)

    def _load_local_rows(self) -> None:
        assert self.log_path is not None
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.provider_attempts += int(row.get("attempt_record", 1))
            self.valid_responses += int(bool(row.get("validation_success", False)))
            self.network_failures += int(row.get("failure_kind") == "network")
            self.provider_failures += int(row.get("failure_kind") == "provider")
            self.validation_failures += int(row.get("failure_kind") == "validation")
            self.input_tokens += int(row.get("input_tokens", 0) or 0)
            self.output_tokens += int(row.get("output_tokens", 0) or 0)
            self.estimated_cost_usd += float(
                row.get("estimated_incremental_cost_usd", 0.0) or 0.0
            )

    @staticmethod
    def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
        input_price, output_price = MODEL_PRICES_USD_PER_MILLION.get(model, (0.0, 0.0))
        return (
            max(0, input_tokens) / 1_000_000 * input_price
            + max(0, output_tokens) / 1_000_000 * output_price
        )

    @staticmethod
    def _campaign_state_from_text(text: str) -> tuple[float, dict[str, float]]:
        billed = 0.0
        active: dict[str, float] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = row.get("event_kind")
            reservation_id = str(row.get("reservation_id", ""))
            if kind == "campaign_reservation" and reservation_id:
                active[reservation_id] = float(row.get("reserved_cost_usd", 0.0) or 0.0)
            elif kind in {"campaign_settlement", "campaign_reservation_release"}:
                active.pop(reservation_id, None)
                if kind == "campaign_settlement":
                    billed += float(row.get("estimated_incremental_cost_usd", 0.0) or 0.0)
        return billed, active

    @staticmethod
    def _append_locked(handle: Any, row: dict[str, Any]) -> None:
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _campaign_snapshot(self) -> tuple[float, float]:
        if self.campaign_log_path is None:
            return self.estimated_cost_usd, self._reserved_cost_usd
        with self.campaign_log_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            billed, active = self._campaign_state_from_text(handle.read())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return billed, sum(active.values())

    def _append_local(self, row: dict[str, Any]) -> None:
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def reserve(
        self,
        *,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> str:
        estimate = self._cost(model, estimated_input_tokens, estimated_output_tokens)
        async with self._lock:
            self._reservation_counter += 1
            reservation_id = (
                f"{self.campaign_id}:p{os.getpid()}:r{self._reservation_counter:08d}"
            )
            if self.campaign_log_path is not None:
                with self.campaign_log_path.open("a+", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    handle.seek(0)
                    billed, active = self._campaign_state_from_text(handle.read())
                    projected = billed + sum(active.values()) + estimate
                    if projected > self.max_cost_usd + 1e-12:
                        rejection = {
                            "event_kind": "campaign_reservation_rejected",
                            "campaign_id": self.campaign_id,
                            "reservation_id": reservation_id,
                            "estimated_billed_cost_usd": billed,
                            "active_reserved_cost_usd": sum(active.values()),
                            "new_reservation_cost_usd": estimate,
                            "estimated_output_tokens_for_reservation": estimated_output_tokens,
                            "projected_cost_usd": projected,
                            "operator_hard_stop_usd": self.max_cost_usd,
                        }
                        self._append_locked(handle, rejection)
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                        self._append_local({"attempt_record": 0, **rejection})
                        raise BudgetExceededError(
                            "Campaign-wide operator guard reached. Preserve the exact policy and "
                            "cache; human intervention is required before resume."
                        )
                    self._append_locked(
                        handle,
                        {
                            "event_kind": "campaign_reservation",
                            "campaign_id": self.campaign_id,
                            "reservation_id": reservation_id,
                            "model": model,
                            "reserved_cost_usd": estimate,
                            "estimated_output_tokens": int(estimated_output_tokens),
                            "operator_hard_stop_usd": self.max_cost_usd,
                        },
                    )
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                projected = self.estimated_cost_usd + self._reserved_cost_usd + estimate
                if projected > self.max_cost_usd + 1e-12:
                    rejection = {
                        "attempt_record": 0,
                        "event_kind": "operator_guard_reservation_rejected",
                        "estimated_total_cost_usd": self.estimated_cost_usd,
                        "reserved_cost_usd": self._reserved_cost_usd,
                        "new_reservation_cost_usd": estimate,
                            "estimated_output_tokens_for_reservation": estimated_output_tokens,
                        "projected_cost_usd": projected,
                        "operator_hard_stop_usd": self.max_cost_usd,
                    }
                    self._append_local(rejection)
                    raise BudgetExceededError(
                        "Operator-only execution guard reached. Preserve the exact policy and cache; "
                        "human intervention is required before resume."
                    )
            self._reservations[reservation_id] = estimate
            self._reserved_cost_usd += estimate
            return reservation_id

    async def settle(
        self,
        *,
        reservation_id: str,
        model: str,
        phase: str,
        slot_id: int,
        month: int,
        network_attempt: int,
        validation_attempt: int,
        input_tokens: int,
        output_tokens: int,
        provider_success: bool,
        parse_success: bool,
        validation_success: bool,
        failure_kind: str | None,
        error: str | None = None,
        response_text: str | None = None,
    ) -> None:
        async with self._lock:
            reserved = self._reservations.pop(reservation_id, 0.0)
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - reserved)
            increment = self._cost(model, input_tokens, output_tokens)
            self.provider_attempts += 1
            self.valid_responses += int(validation_success)
            self.network_failures += int(failure_kind == "network")
            self.provider_failures += int(failure_kind == "provider")
            self.validation_failures += int(failure_kind == "validation")
            self.input_tokens += max(0, input_tokens)
            self.output_tokens += max(0, output_tokens)
            self.estimated_cost_usd += increment
            response_hash = (
                hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                if response_text is not None
                else None
            )
            row = {
                "attempt_record": 1,
                "campaign_id": self.campaign_id if self.campaign_log_path else None,
                "reservation_id": reservation_id,
                "model": model,
                "phase": phase,
                "slot_id": slot_id,
                "month": month,
                "network_attempt": network_attempt,
                "validation_attempt": validation_attempt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "provider_success": provider_success,
                "parse_success": parse_success,
                "validation_success": validation_success,
                "failure_kind": failure_kind,
                "error": error,
                "response_sha256": response_hash,
                "response_text": response_text,
                "validation_failed_response_text": (
                    response_text if failure_kind == "validation" else None
                ),
                "estimated_incremental_cost_usd": increment,
                "estimated_total_cost_usd": self.estimated_cost_usd,
                "reserved_cost_after_settlement_usd": self._reserved_cost_usd,
            }
            self._append_local(row)
            campaign_billed = self.estimated_cost_usd
            campaign_reserved = self._reserved_cost_usd
            if self.campaign_log_path is not None:
                with self.campaign_log_path.open("a+", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    handle.seek(0)
                    billed, active = self._campaign_state_from_text(handle.read())
                    active.pop(reservation_id, None)
                    campaign_billed = billed + increment
                    campaign_reserved = sum(active.values())
                    self._append_locked(
                        handle,
                        {
                            "event_kind": "campaign_settlement",
                            "campaign_id": self.campaign_id,
                            "reservation_id": reservation_id,
                            "model": model,
                            "phase": phase,
                            "estimated_incremental_cost_usd": increment,
                            "campaign_estimated_total_cost_usd": campaign_billed,
                            "campaign_active_reserved_cost_usd": campaign_reserved,
                            "operator_hard_stop_usd": self.max_cost_usd,
                        },
                    )
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if campaign_billed + campaign_reserved > self.max_cost_usd + 1e-12:
                raise BudgetExceededError(
                    "Campaign-wide operator guard reached after a billed provider response. "
                    "Preserve the exact policy and cache; human intervention is required before resume."
                )

    async def release_reservation(self, reservation_id: str) -> None:
        async with self._lock:
            reserved = self._reservations.pop(reservation_id, 0.0)
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - reserved)
            if self.campaign_log_path is not None:
                with self.campaign_log_path.open("a+", encoding="utf-8") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    self._append_locked(
                        handle,
                        {
                            "event_kind": "campaign_reservation_release",
                            "campaign_id": self.campaign_id,
                            "reservation_id": reservation_id,
                            "released_reserved_cost_usd": reserved,
                        },
                    )
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def snapshot(self) -> dict[str, float]:
        campaign_billed, campaign_reserved = self._campaign_snapshot()
        return {
            "llm_provider_attempts": float(self.provider_attempts),
            "llm_valid_responses": float(self.valid_responses),
            "llm_network_failures": float(self.network_failures),
            "llm_provider_failures": float(self.provider_failures),
            "llm_validation_failures": float(self.validation_failures),
            "llm_input_tokens": float(self.input_tokens),
            "llm_output_tokens": float(self.output_tokens),
            "estimated_llm_cost_usd": float(self.estimated_cost_usd),
            "llm_reserved_cost_usd_end": float(self._reserved_cost_usd),
            "campaign_estimated_llm_cost_usd": float(campaign_billed),
            "campaign_reserved_cost_usd": float(campaign_reserved),
            "campaign_operator_hard_stop_usd": float(self.max_cost_usd),
            "llm_calls": float(self.provider_attempts),
            "llm_failed_calls": float(
                self.network_failures + self.provider_failures + self.validation_failures
            ),
        }


class SharedLLMRuntime:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_concurrency: int,
        network_retries: int,
        validation_retries: int,
        retry_sleep_seconds: float,
        reasoning_effort: str,
        ledger: UsageLedger,
        base_seed: int = 0,
    ):
        if (
            model not in MODEL_PRICES_USD_PER_MILLION
            and os.environ.get("POLICYLAB_MOCK_LLM") != "1"
            and os.environ.get("POLICYLAB_ALLOW_UNPRICED_MODEL") != "1"
        ):
            raise ValueError(
                f"No audited price is configured for {model!r}; add it to MODEL_PRICES_USD_PER_MILLION "
                "or explicitly set POLICYLAB_ALLOW_UNPRICED_MODEL=1."
            )
        self.model = model
        self.temperature = temperature
        self.network_retries = max(0, network_retries)
        self.validation_retries = max(0, validation_retries)
        self.retry_sleep_seconds = retry_sleep_seconds
        self.reasoning_effort = reasoning_effort
        self.ledger = ledger
        self.base_seed = int(base_seed)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _seed(
        self,
        slot_id: int,
        month: int,
        phase: str,
        identity_epoch: int,
        validation_attempt: int,
    ) -> int:
        raw = (
            f"{self.base_seed}:{slot_id}:{month}:{phase}:{identity_epoch}:{validation_attempt}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")

    @staticmethod
    def _is_retryable_network_error(exc: BaseException) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        haystack = name + " " + message
        return any(
            token in haystack
            for token in (
                "ratelimit",
                "rate limit",
                "too many requests",
                "toomanyrequests",
                "429",
                "resource_exhausted",
                "quota",
                "retrydelay",
                "retryinfo",
                "retry in",
                "timeout",
                "connection",
                "serviceunavailable",
                "internalserver",
                "temporarilyunavailable",
            )
        )

    @staticmethod
    def _retry_delay_seconds(exc: BaseException, *, default: float, attempt: int) -> float:
        message = str(exc).lower()
        match = re.search(r"retry(?:delay| in)?[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*s", message)
        if match:
            return max(default, float(match.group(1)) + 5.0)
        return default * attempt

    @staticmethod
    def _local_endpoint_chain() -> list[dict[str, str]]:
        raw = os.environ.get("POLICYLAB_OPENAI_COMPAT_ENDPOINTS_JSON")
        if raw:
            value = json.loads(raw)
            if not isinstance(value, list):
                raise TypeError("POLICYLAB_OPENAI_COMPAT_ENDPOINTS_JSON must be a JSON list")
            return [dict(item) for item in value]
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url and model and api_key:
            return [{"name": "openai_compatible", "base_url": base_url, "model": model, "api_key": api_key}]
        return []

    @staticmethod
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

    @staticmethod
    def _extract_json_text(text: str) -> str:
        """Extract one JSON object from local/OpenAI-compatible model text.

        Local models often wrap otherwise valid JSON in Markdown fences or add
        small preambles.  This extractor is deliberately narrow: it only returns
        a balanced JSON object, never YAML, prose, or schema echoes.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove a single Markdown code fence if present.
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped).strip()
        if stripped.startswith("{"):
            try:
                json.loads(stripped)
                return stripped
            except json.JSONDecodeError:
                pass
        start = stripped.find("{")
        if start < 0:
            raise ValueError("response did not contain a JSON object")
        depth = 0
        in_string = False
        escape = False
        for index, character in enumerate(stripped[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif character == "\\":
                    escape = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : index + 1]
                    json.loads(candidate)
                    return candidate
        raise ValueError("response contained an unbalanced JSON object")

    @classmethod
    def _parse_response_model(cls, response_text: str, response_type: type[T]) -> T:
        return response_type.model_validate_json(cls._extract_json_text(response_text))

    @staticmethod
    def _local_json_correction_prompt(*, user: str, response_text: str, error: Exception) -> str:
        return (
            user
            + "\n\nCORRECTION REQUIRED FOR LOCAL OPENAI-COMPATIBLE MODEL:\n"
            + "Your previous answer was not accepted as the required JSON object. "
            + "Return exactly one raw JSON object for the same schema. Do not use Markdown fences. "
            + "Do not echo MONTHLY_ACTION_SCHEMA, explanations, YAML, comments, or any text before/after the JSON.\n"
            + f"Validation error: {type(error).__name__}: {error}\n"
            + "Previous answer to convert/correct:\n"
            + response_text[:4000]
        )

    async def _openai_compatible_call(
        self,
        *,
        endpoint: dict[str, str],
        system: str,
        user: str,
        response_type: type[T],
    ) -> tuple[T, int, int, str, str]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAI-compatible local endpoints") from exc
        client = AsyncOpenAI(
            base_url=endpoint["base_url"],
            api_key=endpoint.get("api_key") or os.environ.get(endpoint.get("api_key_env", ""), "EMPTY"),
            timeout=float(endpoint.get("timeout_seconds", os.environ.get("POLICYLAB_LOCAL_LLM_TIMEOUT_SECONDS", "300"))),
        )
        body: dict[str, Any] = {
            "model": endpoint["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(endpoint.get("temperature", self.temperature)),
            "response_format": {"type": "json_object"},
        }
        completion = await client.chat.completions.create(**body)
        message = completion.choices[0].message
        response_text = self._response_text(message)
        parsed = self._parse_response_model(response_text, response_type)
        usage = getattr(completion, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return parsed, input_tokens, output_tokens, response_text, endpoint.get("name", endpoint["base_url"])

    async def _adaptive_rate_limit_before_call(
        self,
        *,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> None:
        """Cross-process token-bucket guard for live Gemini calls.

        This is request pacing, not output-token clipping.  It prevents local
        bursts from crossing RPM/TPM limits while keeping throughput close to
        the configured quota.  RetryInfo/exponential backoff still handles
        provider-side 429s.
        """
        if os.environ.get("POLICYLAB_ENABLE_ADAPTIVE_GEMINI_RATE_LIMITER", "1") != "1":
            return
        if not model.startswith("gemini/") and not model.startswith("gemini-"):
            return
        normalized = model.removeprefix("gemini/")
        defaults = {
            "gemini-2.5-flash-lite": (4000, 4_000_000),
            "gemini-2.5-flash": (1000, 1_000_000),
            "gemini-2.0-flash": (2000, 4_000_000),
            "gemini-2.5-pro": (150, 2_000_000),
        }
        default_rpm, default_tpm = defaults.get(normalized, (0, 0))
        rpm = int(os.environ.get("POLICYLAB_GEMINI_RPM_LIMIT", str(default_rpm)) or 0)
        tpm = int(os.environ.get("POLICYLAB_GEMINI_INPUT_TPM_LIMIT", str(default_tpm)) or 0)
        if rpm <= 0 and tpm <= 0:
            return
        safety = float(os.environ.get("POLICYLAB_GEMINI_RATE_LIMIT_SAFETY", "0.80"))
        effective_rpm = max(1, int(rpm * safety)) if rpm > 0 else 0
        effective_tpm = max(1, int(tpm * safety)) if tpm > 0 else 0
        estimated_total_tokens = max(1, int(estimated_input_tokens) + int(estimated_output_tokens))
        state_path = Path(os.environ.get("POLICYLAB_GEMINI_RATE_STATE", "/tmp/policylab_gemini_rate_state.json"))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            now = time.time()
            wait_seconds = 0.0
            with state_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                try:
                    state = json.loads(handle.read() or "{}")
                except json.JSONDecodeError:
                    state = {}
                rows = [row for row in state.get("events", []) if now - float(row.get("time", 0.0)) < 60.0]
                req_count = len(rows)
                token_count = sum(int(row.get("tokens", 0) or 0) for row in rows)
                if effective_rpm and req_count + 1 > effective_rpm:
                    oldest = min(float(row.get("time", now)) for row in rows)
                    wait_seconds = max(wait_seconds, oldest + 60.0 - now + 0.25)
                if effective_tpm and token_count + estimated_total_tokens > effective_tpm:
                    cumulative = token_count + estimated_total_tokens
                    sorted_rows = sorted(rows, key=lambda row: float(row.get("time", 0.0)))
                    for row in sorted_rows:
                        if cumulative <= effective_tpm:
                            break
                        cumulative -= int(row.get("tokens", 0) or 0)
                        wait_seconds = max(wait_seconds, float(row.get("time", now)) + 60.0 - now + 0.25)
                if wait_seconds <= 0:
                    rows.append({"time": now, "model": normalized, "tokens": estimated_total_tokens})
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps({"events": rows}, ensure_ascii=False))
                    handle.flush()
                    os.fsync(handle.fileno())
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            await asyncio.sleep(min(max(wait_seconds, 0.5), 120.0))

    async def structured_call(
        self,
        *,
        slot_id: int,
        month: int,
        phase: str,
        identity_epoch: int,
        system: str,
        user: str,
        response_type: type[T],
        output_token_estimate: int,
        validator: Callable[[T], None],
    ) -> T:
        if os.environ.get("POLICYLAB_MOCK_LLM") == "1":
            if os.environ.get("POLICYLAB_ALLOW_TEST_MODE") != "1":
                raise RuntimeError(
                    "Mock mode is test-only; also set POLICYLAB_ALLOW_TEST_MODE=1"
                )
            response = self._mock_response(
                slot_id=slot_id,
                phase=phase,
                user=user,
                response_type=response_type,
            )
            validator(response)
            return response
        if self.model == "local/openai-compatible" or os.environ.get("POLICYLAB_FORCE_OPENAI_COMPATIBLE") == "1":
            endpoints = self._local_endpoint_chain()
            if not endpoints:
                raise RuntimeError("local/openai-compatible requested but no OPENAI_BASE_URL/OPENAI_MODEL/OPENAI_API_KEY or POLICYLAB_OPENAI_COMPAT_ENDPOINTS_JSON is configured")
            errors: list[str] = []
            corrected_user = user
            max_validation_attempts = self.validation_retries + 1 + int(os.environ.get("POLICYLAB_LOCAL_LLM_EXTRA_REPAIR_RETRIES", "2"))
            async with self._semaphore:
                for validation_attempt in range(1, max_validation_attempts + 1):
                    for endpoint in endpoints:
                        estimated_input = max(1, (len(system) + len(corrected_user)) // 4)
                        reservation_id = await self.ledger.reserve(
                            model="local/openai-compatible",
                            estimated_input_tokens=estimated_input,
                            estimated_output_tokens=output_token_estimate,
                        )
                        response_text: str | None = None
                        try:
                            response, input_tokens, output_tokens, response_text, endpoint_name = await self._openai_compatible_call(
                                endpoint=endpoint,
                                system=system,
                                user=corrected_user,
                                response_type=response_type,
                            )
                            validator(response)
                            await self.ledger.settle(
                                reservation_id=reservation_id,
                                model=f"local/openai-compatible:{endpoint_name}",
                                phase=phase,
                                slot_id=slot_id,
                                month=month,
                                network_attempt=1,
                                validation_attempt=validation_attempt,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                provider_success=True,
                                parse_success=True,
                                validation_success=True,
                                failure_kind=None,
                                response_text=response_text,
                            )
                            return response
                        except Exception as exc:
                            retryable_network = self._is_retryable_network_error(exc)
                            await self.ledger.settle(
                                reservation_id=reservation_id,
                                model=f"local/openai-compatible:{endpoint.get('name', endpoint.get('base_url', 'unknown'))}",
                                phase=phase,
                                slot_id=slot_id,
                                month=month,
                                network_attempt=1,
                                validation_attempt=validation_attempt,
                                input_tokens=0,
                                output_tokens=0,
                                provider_success=not retryable_network,
                                parse_success=False,
                                validation_success=False,
                                failure_kind="network" if retryable_network else "validation",
                                error=f"{type(exc).__name__}: {exc}",
                                response_text=response_text,
                            )
                            errors.append(f"{endpoint.get('name', endpoint.get('base_url'))}: attempt {validation_attempt}: {type(exc).__name__}: {exc}")
                            if response_text and not retryable_network:
                                corrected_user = self._local_json_correction_prompt(
                                    user=user,
                                    response_text=response_text,
                                    error=exc,
                                )
                            elif retryable_network:
                                await asyncio.sleep(self._retry_delay_seconds(exc, default=self.retry_sleep_seconds, attempt=validation_attempt))
                            continue
            raise RuntimeError("all OpenAI-compatible local endpoints failed: " + "; ".join(errors[-20:]))
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "Install LiteLLM through the Shachi environment before production runs"
            ) from exc

        validation_error: Exception | None = None
        corrected_user = user
        async with self._semaphore:
            for validation_attempt in range(1, self.validation_retries + 2):
                completion = None
                response_text: str | None = None
                for network_attempt in range(1, self.network_retries + 2):
                    estimated_input = max(1, (len(system) + len(corrected_user)) // 4)
                    reservation_id = await self.ledger.reserve(
                        model=self.model,
                        estimated_input_tokens=estimated_input,
                        estimated_output_tokens=output_token_estimate,
                    )
                    try:
                        await self._adaptive_rate_limit_before_call(
                            model=self.model,
                            estimated_input_tokens=estimated_input,
                            estimated_output_tokens=output_token_estimate,
                        )
                        call_spacing = float(os.environ.get("POLICYLAB_LIVE_CALL_SPACING_SECONDS", "0") or 0.0)
                        if call_spacing > 0:
                            await asyncio.sleep(call_spacing)
                        completion = await litellm.acompletion(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": corrected_user},
                            ],
                            temperature=self.temperature,
                            response_format=response_type,
                            reasoning_effort=self.reasoning_effort,
                            seed=self._seed(
                                slot_id,
                                month,
                                phase,
                                identity_epoch,
                                validation_attempt,
                            ),
                            drop_params=True,
                        )
                        break
                    except BudgetExceededError:
                        await self.ledger.release_reservation(reservation_id)
                        raise
                    except Exception as exc:
                        retryable = self._is_retryable_network_error(exc)
                        await self.ledger.settle(
                            reservation_id=reservation_id,
                            model=self.model,
                            phase=phase,
                            slot_id=slot_id,
                            month=month,
                            network_attempt=network_attempt,
                            validation_attempt=validation_attempt,
                            input_tokens=0,
                            output_tokens=0,
                            provider_success=False,
                            parse_success=False,
                            validation_success=False,
                            failure_kind="network" if retryable else "provider",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        if retryable and network_attempt <= self.network_retries:
                            await asyncio.sleep(
                                self._retry_delay_seconds(exc, default=self.retry_sleep_seconds, attempt=network_attempt)
                            )
                            continue
                        raise RuntimeError(
                            f"Provider call failed without a validation retry: {type(exc).__name__}: {exc}"
                        ) from exc

                if completion is None:
                    raise RuntimeError("provider retry loop ended without a completion")

                usage = getattr(completion, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                parse_success = False
                validation_success = False
                try:
                    message = completion.choices[0].message
                    response_text = self._response_text(message)
                    parsed = self._parse_response_model(response_text, response_type)
                    parse_success = True
                    validator(parsed)
                    validation_success = True
                    await self.ledger.settle(
                        reservation_id=reservation_id,
                        model=self.model,
                        phase=phase,
                        slot_id=slot_id,
                        month=month,
                        network_attempt=network_attempt,
                        validation_attempt=validation_attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        provider_success=True,
                        parse_success=True,
                        validation_success=True,
                        failure_kind=None,
                        response_text=response_text,
                    )
                    return parsed
                except BudgetExceededError:
                    raise
                except Exception as exc:
                    validation_error = exc
                    await self.ledger.settle(
                        reservation_id=reservation_id,
                        model=self.model,
                        phase=phase,
                        slot_id=slot_id,
                        month=month,
                        network_attempt=network_attempt,
                        validation_attempt=validation_attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        provider_success=True,
                        parse_success=parse_success,
                        validation_success=validation_success,
                        failure_kind="validation",
                        error=f"{type(exc).__name__}: {exc}",
                        response_text=response_text,
                    )
                    if validation_attempt <= self.validation_retries:
                        corrected_user = (
                            user
                            + "\n\nCORRECTION REQUIRED: The previous provider response failed application "
                            + f"validation: {type(exc).__name__}: {exc}. Return a corrected response that "
                            + "satisfies the same schema and all semantic constraints."
                        )
                        continue
                    break
        raise StructuredValidationError(
            "Structured response failed application validation after "
            f"{self.validation_retries + 1} provider responses: {validation_error}"
        )

    @staticmethod
    def _mock_response(
        *,
        slot_id: int,
        phase: str,
        user: str,
        response_type: type[T],
    ) -> T:
        event_ids: list[str] = []
        # Only the current EVENT LEDGER uses ``- <event_id>:`` rows.  Fact-memory
        # rows begin with ``- month=...`` and must never be echoed as event IDs.
        event_row = re.compile(r"^- (m\d{2}-[a-z0-9_-]+):", re.IGNORECASE)
        for line in user.splitlines():
            match = event_row.match(line.strip())
            if match:
                event_ids.append(match.group(1))
        if response_type is ManagerDecision or phase == "manager_decision":
            request_rows: list[tuple[str, str, float]] = []
            support_remaining = 0.0
            triage_remaining = 0.0
            reform_remaining = 0
            explanation_remaining = 0
            specialist_remaining = 0
            for line in user.splitlines():
                stripped = line.strip()
                if stripped.startswith("SUPPORT ENVELOPE UNITS:"):
                    try:
                        support_remaining = float(stripped.split(":", 1)[1])
                    except ValueError:
                        pass
                elif stripped.startswith("TRIAGE ENVELOPE UNITS:"):
                    try:
                        triage_remaining = float(stripped.split(":", 1)[1])
                    except ValueError:
                        pass
                elif stripped.startswith("REFORM SLOTS:"):
                    try:
                        reform_remaining = int(stripped.split(":", 1)[1])
                    except ValueError:
                        pass
                elif stripped.startswith("EXPLANATION SLOTS:"):
                    try:
                        explanation_remaining = int(stripped.split(":", 1)[1])
                    except ValueError:
                        pass
                elif stripped.startswith("SPECIALIST SLOTS:"):
                    try:
                        specialist_remaining = int(stripped.split(":", 1)[1])
                    except ValueError:
                        pass
                elif stripped.startswith("- mr-") and ": kind=" in stripped:
                    request_id = stripped[2:].split(":", 1)[0]
                    kind = stripped.split("kind=", 1)[1].split(";", 1)[0]
                    requested = 0.0
                    if "requested_units=" in stripped:
                        try:
                            requested = float(stripped.split("requested_units=", 1)[1].split(";", 1)[0])
                        except ValueError:
                            pass
                    request_rows.append((request_id, kind, requested))
            decisions: list[ManagerRequestDecision] = []
            for request_id, kind, requested in request_rows:
                committed = 0.0
                decision = "approve"
                if kind in {"operational_support", "staffing_relief"}:
                    committed = min(requested, support_remaining)
                    support_remaining -= committed
                    if committed <= 0:
                        decision = "defer"
                elif kind == "operational_risk":
                    committed = min(requested, triage_remaining)
                    triage_remaining -= committed
                    if committed <= 0:
                        decision = "defer"
                elif kind == "process_reform":
                    if reform_remaining > 0:
                        reform_remaining -= 1
                    else:
                        decision = "defer"
                elif kind == "explanation":
                    if explanation_remaining > 0:
                        explanation_remaining -= 1
                    else:
                        decision = "defer"
                elif kind == "specialist_track":
                    if specialist_remaining > 0:
                        specialist_remaining -= 1
                    else:
                        decision = "defer"
                decisions.append(
                    ManagerRequestDecision(
                        request_id=request_id,
                        decision=decision,
                        committed_units=committed,
                        public_message=(
                            "The request was reviewed against current deadlines, service continuity, and the finite monthly envelope."
                        ),
                    )
                )
            return response_type.model_validate(
                ManagerDecision(
                    department_message=(
                        "I reviewed the representative docket, prioritized statutory and operational continuity, and kept commitments within the finite envelope."
                    ),
                    decisions=decisions,
                    confidence=0.76,
                ).model_dump()
            )
        if not event_ids:
            event_ids = ["mock-event"]
        if response_type is BureaucratQuarterlyReflection or phase == "quarterly_reflection":
            return response_type.model_validate(
                BureaucratQuarterlyReflection(
                    quarter_summary=(
                        "Workload, staffing events, and available support shaped the quarter; I adjusted priorities accordingly."
                    ),
                    perceived_main_driver="workload",
                    desired_change="More predictable staffing and clearer prioritization.",
                    next_quarter_strategy="Protect core delivery while escalating risks earlier.",
                    event_refs=event_ids[:4],
                    confidence=0.75,
                ).model_dump()
            )
        workload = 1.0
        severity = 0
        current_department = DEPARTMENTS[slot_id % len(DEPARTMENTS)]
        for line in user.splitlines():
            if "realized_workload_ratio=" in line:
                try:
                    workload = float(
                        line.split("realized_workload_ratio=", 1)[1]
                        .split(";", 1)[0]
                        .split("]", 1)[0]
                    )
                except ValueError:
                    pass
            if "after_hours_severity=" in line:
                try:
                    severity = int(
                        line.split("after_hours_severity=", 1)[1]
                        .split(";", 1)[0]
                        .split("]", 1)[0]
                    )
                except ValueError:
                    pass
            if "PROFILE:" in line and "department=" in line:
                current_department = line.split("department=", 1)[1].split(";", 1)[0]
        fatigue = max(
            5,
            min(100, round(18 + 36 * max(0.0, workload - 0.90) + 6 * severity)),
        )
        turnover = max(
            2,
            min(100, round(4 + 42 * max(0.0, workload - 1.0) + 0.35 * fatigue)),
        )
        effort = max(
            45,
            min(115, round(106 - 32 * max(0.0, workload - 1.0) - 0.18 * fatigue)),
        )
        if severity >= 4 and fatigue >= 70:
            work_response = "protect_health_capacity"
            effort = min(effort, 80)
        elif workload > 1.15:
            work_response = "request_support"
        elif workload > 1.0:
            work_response = "prioritize_core_work"
        else:
            work_response = "deliver_normally"
        career_action = (
            "explore_external_exit"
            if turnover >= 90
            else "request_transfer"
            if turnover >= 72
            else "stay"
        )
        transfer_preference = None
        if career_action == "request_transfer":
            target = next(dept for dept in DEPARTMENTS if dept != current_department)
            transfer_preference = TransferPreference(
                preferred_department=target,
                acceptable_departments=[target],
                preferred_fields=["public_management"],
                priority="workload_recovery",
            )
        voice = "request_staffing_relief" if workload > 1.12 else "none"
        return response_type.model_validate(
            BureaucratMonthlyAction(
                relative_effort_pct=effort,
                work_mix=WorkMix(
                    core_delivery_pct=65,
                    coordination_pct=15,
                    learning_pct=10,
                    process_improvement_pct=10,
                ),
                work_response=work_response,
                voice_action=voice,
                career_action=career_action,
                transfer_preference=transfer_preference,
                self_report=MonthlySelfReport(
                    fatigue_pct=fatigue,
                    turnover_intent_pct=turnover,
                    organizational_trust_pct=max(
                        5, 70 - max(0, turnover - 30) // 2
                    ),
                    procedural_fairness_pct=max(
                        5, min(100, round(68 - max(0.0, workload - 1.0) * 40))
                    ),
                    ebpm_interest_pct=max(10, min(95, round(52 + 10 * (1.0 - workload)))),
                    dx_improvement_interest_pct=max(10, min(95, round(55 + 8 * (1.0 - workload)))),
                ),
                event_refs=event_ids[:4],
                reason=(
                    "I selected effort and institutional actions from the realized workload, staffing, and recent events."
                ),
                next_month_intent=(
                    "Continue core work while protecting sustainable capacity and monitoring staffing conditions."
                ),
                confidence=0.78,
            ).model_dump()
        )


class JapanPolicyBureaucratAgent(Agent):
    """LLM + configuration + bounded memory following Shachi's Agent API."""

    def __init__(
        self,
        *,
        slot_id: int,
        runtime: SharedLLMRuntime,
        monthly_output_token_estimate: int = 900,
        quarterly_output_token_estimate: int = 900,
        memory: BureaucratMemory | None = None,
    ):
        self.slot_id = slot_id
        self.runtime = runtime
        self.monthly_output_token_estimate = monthly_output_token_estimate
        self.quarterly_output_token_estimate = quarterly_output_token_estimate
        self.memory = memory or BureaucratMemory()
        self.identity_epoch: int | None = None
        self._last_monthly_action: BureaucratMonthlyAction | None = None

    def update_config(self, kwargs_as_dict: dict) -> None:
        if "slot_id" in kwargs_as_dict:
            self.slot_id = int(kwargs_as_dict["slot_id"])

    def _ensure_identity(self, observation: BureaucracyObservation) -> None:
        if self.identity_epoch != observation.identity_epoch:
            self.memory.clear()
            self.identity_epoch = observation.identity_epoch

    def prepare_request(self, observation: BureaucracyObservation) -> PreparedAgentRequest:
        self._ensure_identity(observation)
        if observation.response_type is None:
            raise ValueError("observation.response_type is required")
        output_token_estimate = (
            self.monthly_output_token_estimate
            if observation.phase == "monthly_action"
            else self.quarterly_output_token_estimate
        )
        user = build_user_prompt(observation, self.memory.retrieve())
        identity = cache_identity(
            observation=observation,
            system_prompt=BUREAUCRAT_SYSTEM_PROMPT,
            user_prompt=user,
            response_type=observation.response_type,
            model=self.runtime.model,
            temperature=self.runtime.temperature,
            reasoning_effort=self.runtime.reasoning_effort,
        )
        return PreparedAgentRequest(
            observation=observation,
            system=BUREAUCRAT_SYSTEM_PROMPT,
            user=user,
            response_type=observation.response_type,
            output_token_estimate=output_token_estimate,
            cache_identity=identity,
        )

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        typed = BureaucracyObservation.model_validate(observation)
        request = self.prepare_request(typed)
        response = await self.runtime.structured_call(
            slot_id=self.slot_id,
            month=typed.month,
            phase=typed.phase,
            identity_epoch=typed.identity_epoch,
            system=request.system,
            user=request.user,
            response_type=request.response_type,
            output_token_estimate=request.output_token_estimate,
            validator=lambda value: validate_response_against_observation(
                typed, value
            ),
        )
        self.record_response(typed, response)
        return response

    def record_response(
        self,
        observation: BureaucracyObservation,
        response: pydantic.BaseModel,
    ) -> None:
        self._ensure_identity(observation)
        self.memory.add_record(
            [
                {
                    "role": "fact",
                    "content": (
                        f"month={observation.month}; event_id={event.event_id}; "
                        f"event_type={event.event_type}; outcome={event.decision_status}; "
                        f"description={event.description}"
                    ),
                }
                for event in observation.recent_events
            ]
        )
        if isinstance(response, BureaucratMonthlyAction):
            # Survey firewall: no self-report value, narrative justification, or
            # next-intent text is fed back. Memory retains only realized events
            # and observable structured choices.
            preference = (
                response.transfer_preference.model_dump()
                if response.transfer_preference is not None
                else None
            )
            self.memory.add_record(
                [
                    {
                        "role": "monthly",
                        "content": (
                            f"month={observation.month}; effort={response.relative_effort_pct}; "
                            f"work_response={response.work_response}; voice_action={response.voice_action}; "
                            f"career_action={response.career_action}; transfer_preference={preference}; "
                            f"event_refs={response.event_refs}"
                        ),
                    }
                ]
            )
            self._last_monthly_action = response
        elif isinstance(response, BureaucratQuarterlyReflection):
            self.memory.add_record(
                [
                    {
                        "role": "quarterly",
                        "content": (
                            f"month={observation.month}; driver={response.perceived_main_driver}; "
                            f"summary={response.quarter_summary}; desired_change={response.desired_change}; "
                            f"strategy={response.next_quarter_strategy}"
                        ),
                    }
                ]
            )

    def project_monthly_action(
        self,
        observation: BureaucracyObservation,
        *,
        reason_suffix: str = "Temporal representative reuse between LLM decision months.",
    ) -> BureaucratMonthlyAction | None:
        """Project the latest monthly action onto a non-decision month.

        This is an ABM-scale execution primitive, not an LLM-output clip.  Many
        agent-based simulations query a behavioral policy at decision epochs and
        let the environment advance intermediate timesteps.  We preserve the
        48-month organization horizon while reducing employee-month provider
        calls.  The projected action keeps the last validated semantic choice but
        rewrites event references to the current observation so downstream
        identity/event validation remains exact.
        """

        self._ensure_identity(observation)
        if observation.phase != "monthly_action" or self._last_monthly_action is None:
            return None
        allowed = observation.allowed_event_ids()
        if not allowed:
            return None
        action = self._last_monthly_action.model_copy(deep=True)
        # Transfer preferences can become invalid after transfer or department
        # changes; requiring a fresh LLM decision for transfer events is handled
        # in the runner, but clearing here is a defensive validation guard.
        action.transfer_preference = None
        action.event_refs = [allowed[0]]
        action.reason = (action.reason + " " + reason_suffix)[:1000]
        action.next_month_intent = (
            action.next_month_intent
            or "Continue current work response until a new decision month."
        )
        validate_response_against_observation(observation, action)
        return action


class JapanPolicyManagerAgent(Agent):
    """One non-capacity manager per department with fact-only memory."""

    def __init__(
        self,
        *,
        manager_id: int,
        runtime: SharedLLMRuntime,
        output_token_estimate: int = 900,
        memory: ManagerFactMemory | None = None,
    ):
        self.manager_id = manager_id
        self.runtime = runtime
        self.output_token_estimate = output_token_estimate
        self.memory = memory or ManagerFactMemory()

    def update_config(self, kwargs_as_dict: dict) -> None:
        if "manager_id" in kwargs_as_dict:
            self.manager_id = int(kwargs_as_dict["manager_id"])

    def prepare_request(self, observation: ManagerObservation) -> PreparedAgentRequest:
        if observation.response_type is None:
            raise ValueError("observation.response_type is required")
        user = build_user_prompt(observation, self.memory.retrieve())
        identity = cache_identity(
            observation=observation,
            system_prompt=MANAGER_SYSTEM_PROMPT,
            user_prompt=user,
            response_type=observation.response_type,
            model=self.runtime.model,
            temperature=self.runtime.temperature,
            reasoning_effort=self.runtime.reasoning_effort,
        )
        return PreparedAgentRequest(
            observation=observation,
            system=MANAGER_SYSTEM_PROMPT,
            user=user,
            response_type=observation.response_type,
            output_token_estimate=self.output_token_estimate,
            cache_identity=identity,
        )

    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        typed = ManagerObservation.model_validate(observation)
        request = self.prepare_request(typed)
        response = await self.runtime.structured_call(
            slot_id=self.manager_id,
            month=typed.month,
            phase=typed.phase,
            identity_epoch=0,
            system=request.system,
            user=request.user,
            response_type=request.response_type,
            output_token_estimate=request.output_token_estimate,
            validator=lambda value: validate_response_against_observation(typed, value),
        )
        self.record_response(typed, response)
        return response

    def record_response(
        self, observation: ManagerObservation, response: pydantic.BaseModel
    ) -> None:
        if not isinstance(response, ManagerDecision):
            raise TypeError(f"manager expected ManagerDecision, got {type(response).__name__}")
        commitments = sum(item.committed_units for item in response.decisions)
        self.memory.add_record(
            [
                {
                    "role": "fact",
                    "content": (
                        f"month={observation.month}; department={observation.department}; "
                        f"docket_size={len(observation.request_docket)}; commitments={commitments:.2f}; "
                        f"message={response.department_message}"
                    ),
                }
            ]
        )
