"""Deterministic replay and single-request live review providers."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from codereviewops.contracts import (
    LIVE_STRUCTURED_OUTPUT_MODE,
    MODEL_PATTERN,
    PROMPT_VERSION,
    is_valid_model_identifier,
)
from codereviewops.models import ProviderResult, ReviewContext, ReviewReport, Usage
from codereviewops.prompt import (
    MAX_RESPONSE_BYTES,
    REVIEW_REPORT_JSON_SCHEMA,
    STRUCTURED_OUTPUT_SCHEMA_NAME,
    build_prompt_messages,
)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
USER_AGENT = "codereviewops/0.1.0"
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MODEL_UNAVAILABLE_CODES = {
    "unknown_model",
    "model_not_found",
    "decommissioned",
    "retired",
}
_REFUSAL_CODES = {"refusal", "content_filter", "safety"}
_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


class ProviderErrorKind(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_REQUEST = "invalid_request"
    SERVICE_UNAVAILABLE = "service_unavailable"
    REFUSAL = "refusal"
    MALFORMED_RESPONSE = "malformed_response"


class ProviderError(ValueError):
    """Safe provider failure metadata without provider-controlled prose."""

    def __init__(
        self,
        kind: ProviderErrorKind,
        *,
        provider: str,
        model: str | None,
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.kind = kind
        self.provider = provider
        self.model = model
        self.status = status
        self.code = code
        self.request_id = request_id
        self.retryable = retryable
        super().__init__(str(self))

    def __str__(self) -> str:
        fields = [
            f"provider={self.provider}",
            f"kind={self.kind.value}",
            f"model={self.model or '-'}",
            f"retryable={str(self.retryable).lower()}",
        ]
        if self.status is not None:
            fields.append(f"status={self.status}")
        if self.code is not None:
            fields.append(f"code={self.code}")
        if self.request_id is not None:
            fields.append(f"request_id={self.request_id}")
        return "provider error: " + " ".join(fields)

    def __repr__(self) -> str:
        return (
            "ProviderError("
            f"kind={self.kind.value!r}, provider={self.provider!r}, "
            f"model={self.model!r}, status={self.status!r}, code={self.code!r}, "
            f"request_id={self.request_id!r}, retryable={self.retryable!r})"
        )


class ReviewProvider(Protocol):
    def review(self, context: ReviewContext) -> ProviderResult:
        """Return a review plus safe provider metadata."""
        ...


def validate_model(model: str) -> str:
    if not is_valid_model_identifier(model):
        raise ValueError("model must match the supported identifier format")
    return model


def _safe_value(value: Any, secret: str) -> str | None:
    return (
        value
        if isinstance(value, str) and secret not in value and _SAFE_VALUE.fullmatch(value)
        else None
    )


def _error_metadata(
    body: bytes,
    response: httpx.Response,
    secret: str,
) -> tuple[str | None, str | None]:
    code: str | None = None
    request_id = _safe_value(response.headers.get("x-request-id"), secret)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return code, request_id
    if not isinstance(payload, dict):
        return code, request_id
    error = payload.get("error")
    source = error if isinstance(error, dict) else payload
    code = _safe_value(source.get("code"), secret)
    request_id = request_id or _safe_value(source.get("request_id"), secret)
    return code, request_id


def _contains_secret(value: Any, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(item, secret) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _read_limited(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise ProviderError(
                ProviderErrorKind.MALFORMED_RESPONSE,
                provider="response",
                model=None,
                code="response_too_large",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _classify_status(
    provider: str,
    model: str,
    status: int,
    code: str | None,
    request_id: str | None,
) -> ProviderError:
    normalized = code.casefold() if code else None
    if status == 401:
        kind, retryable = ProviderErrorKind.AUTHENTICATION, False
    elif status == 403:
        kind, retryable = ProviderErrorKind.PERMISSION, False
    elif status == 429:
        kind, retryable = ProviderErrorKind.RATE_LIMIT, True
    elif status == 404:
        kind, retryable = ProviderErrorKind.MODEL_UNAVAILABLE, False
    elif status == 498 or status >= 500:
        kind, retryable = ProviderErrorKind.SERVICE_UNAVAILABLE, True
    elif normalized in _MODEL_UNAVAILABLE_CODES:
        kind, retryable = ProviderErrorKind.MODEL_UNAVAILABLE, False
    elif normalized in _REFUSAL_CODES:
        kind, retryable = ProviderErrorKind.REFUSAL, False
    elif status in {400, 422}:
        kind, retryable = ProviderErrorKind.INVALID_REQUEST, False
    else:
        kind, retryable = ProviderErrorKind.INVALID_REQUEST, False
    return ProviderError(
        kind,
        provider=provider,
        model=model,
        status=status,
        code=code,
        request_id=request_id,
        retryable=retryable,
    )


def _parse_usage(raw: Any, provider: str, model: str) -> Usage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProviderError(
            ProviderErrorKind.MALFORMED_RESPONSE,
            provider=provider,
            model=model,
            code="invalid_usage",
        )
    try:
        retained = {
            key: raw[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in raw
        }
        return Usage.model_validate(retained)
    except ValidationError:
        raise ProviderError(
            ProviderErrorKind.MALFORMED_RESPONSE,
            provider=provider,
            model=model,
            code="invalid_usage",
        ) from None


class ReplayProvider:
    def __init__(self, replay_path: Path) -> None:
        try:
            raw = json.loads(replay_path.read_text(encoding="utf-8"))
            self._report = ReviewReport.model_validate(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
            raise ProviderError(
                ProviderErrorKind.MALFORMED_RESPONSE,
                provider="replay",
                model=None,
                code="invalid_replay",
            ) from None

    def review(self, context: ReviewContext) -> ProviderResult:
        del context
        return ProviderResult(
            report=self._report.model_copy(deep=True),
            requested_model=None,
            response_model=None,
            prompt_version=None,
            structured_output_mode="replay",
            latency_ms=0,
            usage=None,
        )


class LiveProvider:
    provider: str
    endpoint: str
    token_field: str

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            validated_model = validate_model(model)
        except ValueError:
            raise ProviderError(
                ProviderErrorKind.CONFIGURATION,
                provider=self.provider,
                model=None,
                code="invalid_model",
            ) from None
        if not api_key:
            raise ProviderError(
                ProviderErrorKind.CONFIGURATION,
                provider=self.provider,
                model=validated_model,
                code="missing_api_key",
            )
        if api_key in validated_model:
            raise ProviderError(
                ProviderErrorKind.CONFIGURATION,
                provider=self.provider,
                model=None,
                code="secret_in_model",
            )
        self.model = validated_model
        self._api_key = api_key
        self._transport = transport
        self._clock = clock

    def _payload(self, context: ReviewContext) -> dict[str, Any]:
        try:
            messages = build_prompt_messages(context)
        except ValueError:
            raise ProviderError(
                ProviderErrorKind.INVALID_REQUEST,
                provider=self.provider,
                model=self.model,
                code="prompt_too_large",
            ) from None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "n": 1,
            "stream": False,
            self.token_field: 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": STRUCTURED_OUTPUT_SCHEMA_NAME,
                    "strict": True,
                    "schema": REVIEW_REPORT_JSON_SCHEMA,
                },
            },
        }
        return payload

    def review(self, context: ReviewContext) -> ProviderResult:
        payload = self._payload(context)
        started = self._clock()
        try:
            with (
                httpx.Client(
                    transport=self._transport,
                    timeout=_TIMEOUT,
                    trust_env=False,
                    follow_redirects=False,
                ) as client,
                client.stream(
                    "POST",
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": USER_AGENT,
                    },
                    json=payload,
                ) as response,
            ):
                body = _read_limited(response)
        except ProviderError as exc:
            if exc.provider == "response":
                raise ProviderError(
                    exc.kind,
                    provider=self.provider,
                    model=self.model,
                    code=exc.code,
                ) from None
            raise
        except httpx.TimeoutException:
            raise ProviderError(
                ProviderErrorKind.TIMEOUT,
                provider=self.provider,
                model=self.model,
                retryable=True,
            ) from None
        except httpx.TransportError:
            raise ProviderError(
                ProviderErrorKind.TRANSPORT,
                provider=self.provider,
                model=self.model,
                retryable=True,
            ) from None
        latency_ms = max(0.0, (self._clock() - started) * 1000)
        if response.status_code < 200 or response.status_code >= 300:
            code, request_id = _error_metadata(body, response, self._api_key)
            raise _classify_status(
                self.provider,
                self.model,
                response.status_code,
                code,
                request_id,
            )
        try:
            envelope = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise self._malformed("invalid_envelope") from None
        if not isinstance(envelope, dict):
            raise self._malformed("invalid_envelope")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise self._malformed("invalid_choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise self._malformed("invalid_choice")
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._malformed("invalid_message")
        if message.get("refusal") not in {None, False, ""}:
            raise ProviderError(
                ProviderErrorKind.REFUSAL,
                provider=self.provider,
                model=self.model,
                code="refusal",
            )
        if finish_reason in {"content_filter", "safety"}:
            raise ProviderError(
                ProviderErrorKind.REFUSAL,
                provider=self.provider,
                model=self.model,
                code=str(finish_reason),
            )
        if finish_reason in {"length", "max_tokens"}:
            raise self._malformed("truncated")
        if finish_reason != "stop":
            raise self._malformed("invalid_finish_reason")
        content = message.get("content")
        raw_response_model = envelope.get("model")
        if not isinstance(content, str):
            raise self._malformed("invalid_content")
        if raw_response_model is None:
            response_model = None
        elif not isinstance(raw_response_model, str) or not MODEL_PATTERN.fullmatch(
            raw_response_model
        ):
            raise self._malformed("invalid_response_model")
        elif self._api_key in raw_response_model:
            raise self._malformed("secret_in_response")
        else:
            response_model = raw_response_model
        try:
            report = ReviewReport.model_validate_json(content)
        except ValidationError:
            raise self._malformed("invalid_review") from None
        if report.tests_run:
            raise self._malformed("nonempty_tests_run")
        usage = _parse_usage(envelope.get("usage"), self.provider, self.model)
        retained = {
            "report": report.model_dump(mode="python"),
            "response_model": response_model,
            "usage": usage.model_dump(mode="python") if usage is not None else None,
        }
        if _contains_secret(retained, self._api_key):
            raise self._malformed("secret_in_response")
        return ProviderResult(
            report=report,
            requested_model=self.model,
            response_model=response_model,
            prompt_version=PROMPT_VERSION,
            structured_output_mode=LIVE_STRUCTURED_OUTPUT_MODE,
            latency_ms=latency_ms,
            usage=usage,
        )

    def _malformed(self, code: str) -> ProviderError:
        return ProviderError(
            ProviderErrorKind.MALFORMED_RESPONSE,
            provider=self.provider,
            model=self.model,
            code=code,
        )


class GroqProvider(LiveProvider):
    provider = "groq"
    endpoint = GROQ_ENDPOINT
    token_field = "max_completion_tokens"


class MistralProvider(LiveProvider):
    provider = "mistral"
    endpoint = MISTRAL_ENDPOINT
    token_field = "max_tokens"
