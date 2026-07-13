from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from codereviewops.contracts import LIVE_STRUCTURED_OUTPUT_MODE
from codereviewops.models import EvaluationResult, ReviewContext, RunArtifact
from codereviewops.prompt import (
    MAX_PROMPT_BYTES,
    PROMPT_VERSION,
    REVIEW_REPORT_JSON_SCHEMA,
    STRUCTURED_OUTPUT_SCHEMA_NAME,
    SYSTEM_PROMPT,
)
from codereviewops.providers import (
    GROQ_ENDPOINT,
    MISTRAL_ENDPOINT,
    GroqProvider,
    MistralProvider,
    ProviderError,
    ProviderErrorKind,
    ReplayProvider,
)

MODEL = "test/model-1"
SECRET = "super-secret-provider-key"
PROVIDER_CASES = [
    (GroqProvider, GROQ_ENDPOINT, "max_completion_tokens"),
    (MistralProvider, MISTRAL_ENDPOINT, "max_tokens"),
]


def _context(diff_text: str = "+changed = True") -> ReviewContext:
    return ReviewContext(
        schema_version="1.0",
        task_id="task",
        title="Synthetic review",
        issue_description="Review only the supplied change.",
        diff_text=diff_text,
    )


def _report(**overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "summary": "No supported issues.",
        "overall_assessment": "pass",
        "findings": [],
        "tests_run": [],
        "limitations": [],
    }
    report.update(overrides)
    return report


def _envelope(
    *,
    report: dict[str, Any] | None = None,
    model: Any = MODEL,
    finish_reason: str = "stop",
    usage: Any = ...,
    refusal: Any = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"content": json.dumps(report if report is not None else _report())}
    if refusal is not None:
        message["refusal"] = refusal
    payload: dict[str, Any] = {"choices": [{"finish_reason": finish_reason, "message": message}]}
    if model is not ...:
        payload["model"] = model
    if usage is not ...:
        payload["usage"] = usage
    return payload


def _provider(
    provider_class,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: Callable[[], float] | None = None,
):
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "api_key": SECRET,
        "transport": httpx.MockTransport(handler),
    }
    if clock is not None:
        kwargs["clock"] = clock
    return provider_class(**kwargs)


def _assert_schema_objects_closed_and_required(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for value in node.values():
            _assert_schema_objects_closed_and_required(value)
    elif isinstance(node, list):
        for value in node:
            _assert_schema_objects_closed_and_required(value)


def test_review_schema_is_closed_and_all_properties_required() -> None:
    _assert_schema_objects_closed_and_required(REVIEW_REPORT_JSON_SCHEMA)


@pytest.mark.parametrize(("provider_class", "endpoint", "token_field"), PROVIDER_CASES)
def test_request_shape_endpoint_prompt_and_schema(
    provider_class,
    endpoint: str,
    token_field: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_envelope())

    result = _provider(provider_class, handler).review(_context())
    assert result.report.overall_assessment.value == "pass"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == endpoint
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert request.headers["user-agent"] == "codereviewops/0.1.0"
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["n"] == 1
    assert body["stream"] is False
    assert body[token_field] == 4096
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": STRUCTURED_OUTPUT_SCHEMA_NAME,
            "strict": True,
            "schema": REVIEW_REPORT_JSON_SCHEMA,
        },
    }
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user_data = json.loads(body["messages"][1]["content"])
    assert set(user_data) == {
        "schema_version",
        "task_id",
        "title",
        "issue_description",
        "diff_text",
    }
    serialized = json.dumps(body)
    assert "expected_findings" not in serialized
    assert "must_not_find" not in serialized
    assert "replay_response_path" not in serialized
    assert SECRET not in serialized


@pytest.mark.parametrize(("provider_class", "_endpoint", "_token_field"), PROVIDER_CASES)
def test_success_identity_usage_and_injected_clock(
    provider_class,
    _endpoint: str,
    _token_field: str,
) -> None:
    ticks = iter([10.0, 10.25])
    usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(model="response/model", usage=usage))

    result = _provider(provider_class, handler, clock=lambda: next(ticks)).review(_context())
    assert result.requested_model == MODEL
    assert result.response_model == "response/model"
    assert result.prompt_version == PROMPT_VERSION
    assert result.structured_output_mode == LIVE_STRUCTURED_OUTPUT_MODE
    assert result.latency_ms == 250
    assert result.usage is not None
    assert result.usage.model_dump() == usage


def test_absent_usage_is_allowed() -> None:
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope()),
    )
    assert provider.review(_context()).usage is None


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, ProviderErrorKind.AUTHENTICATION, False),
        (403, ProviderErrorKind.PERMISSION, False),
        (429, ProviderErrorKind.RATE_LIMIT, True),
        (404, ProviderErrorKind.MODEL_UNAVAILABLE, False),
        (400, ProviderErrorKind.INVALID_REQUEST, False),
        (422, ProviderErrorKind.INVALID_REQUEST, False),
        (498, ProviderErrorKind.SERVICE_UNAVAILABLE, True),
        (500, ProviderErrorKind.SERVICE_UNAVAILABLE, True),
        (503, ProviderErrorKind.SERVICE_UNAVAILABLE, True),
    ],
)
def test_http_error_classification(
    status: int,
    kind: ProviderErrorKind,
    retryable: bool,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            json={"error": {"code": "safe_code", "message": f"{SECRET} arbitrary"}},
            headers={"x-request-id": "request_123"},
        )

    with pytest.raises(ProviderError) as captured:
        _provider(GroqProvider, handler).review(_context())
    assert captured.value.kind == kind
    assert captured.value.retryable is retryable
    assert captured.value.request_id == "request_123"
    assert calls == 1
    assert SECRET not in str(captured.value)
    assert SECRET not in repr(captured.value)
    assert "arbitrary" not in str(captured.value)


@pytest.mark.parametrize("code", ["unknown_model", "model_not_found", "decommissioned", "retired"])
def test_safe_model_codes_classify_model_unavailable(code: str) -> None:
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(400, json={"error": {"code": code}}),
    )
    with pytest.raises(ProviderError) as captured:
        provider.review(_context())
    assert captured.value.kind == ProviderErrorKind.MODEL_UNAVAILABLE
    assert captured.value.code == code


@pytest.mark.parametrize("code", ["refusal", "content_filter", "safety"])
def test_safe_refusal_codes_classify_refusal(code: str) -> None:
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(400, json={"error": {"code": code}}),
    )
    with pytest.raises(ProviderError) as captured:
        provider.review(_context())
    assert captured.value.kind == ProviderErrorKind.REFUSAL


@pytest.mark.parametrize(
    ("exception_factory", "kind"),
    [
        (
            lambda request: httpx.ReadTimeout("timeout", request=request),
            ProviderErrorKind.TIMEOUT,
        ),
        (
            lambda request: httpx.ConnectError("transport", request=request),
            ProviderErrorKind.TRANSPORT,
        ),
    ],
)
def test_timeout_and_transport_are_safe_and_not_retried(exception_factory, kind) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exception_factory(request)

    with pytest.raises(ProviderError) as captured:
        _provider(GroqProvider, handler).review(_context())
    assert captured.value.kind == kind
    assert captured.value.retryable
    assert calls == 1


@pytest.mark.parametrize(
    ("envelope", "kind", "code"),
    [
        (_envelope(finish_reason="length"), ProviderErrorKind.MALFORMED_RESPONSE, "truncated"),
        (_envelope(finish_reason="max_tokens"), ProviderErrorKind.MALFORMED_RESPONSE, "truncated"),
        (_envelope(finish_reason="content_filter"), ProviderErrorKind.REFUSAL, "content_filter"),
        (_envelope(finish_reason="safety"), ProviderErrorKind.REFUSAL, "safety"),
        (_envelope(refusal="blocked"), ProviderErrorKind.REFUSAL, "refusal"),
        ({"model": MODEL, "choices": []}, ProviderErrorKind.MALFORMED_RESPONSE, "invalid_choices"),
        (
            {"model": MODEL, "choices": [{}, {}]},
            ProviderErrorKind.MALFORMED_RESPONSE,
            "invalid_choices",
        ),
        (
            {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {"content": 1}}]},
            ProviderErrorKind.MALFORMED_RESPONSE,
            "invalid_content",
        ),
        (
            _envelope(model="bad model"),
            ProviderErrorKind.MALFORMED_RESPONSE,
            "invalid_response_model",
        ),
        (_envelope(usage=-1), ProviderErrorKind.MALFORMED_RESPONSE, "invalid_usage"),
        (
            _envelope(usage={"prompt_tokens": -1}),
            ProviderErrorKind.MALFORMED_RESPONSE,
            "invalid_usage",
        ),
        (
            _envelope(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99}),
            ProviderErrorKind.MALFORMED_RESPONSE,
            "invalid_usage",
        ),
    ],
)
def test_refusal_truncation_and_malformed_envelopes(envelope, kind, code) -> None:
    provider = _provider(
        MistralProvider,
        lambda request: httpx.Response(200, json=envelope),
    )
    with pytest.raises(ProviderError) as captured:
        provider.review(_context())
    assert captured.value.kind == kind
    assert captured.value.code == code


def test_invalid_envelope_json_and_invalid_review_json() -> None:
    invalid_envelope = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, content=b"{"),
    )
    with pytest.raises(ProviderError) as first:
        invalid_envelope.review(_context())
    assert first.value.code == "invalid_envelope"

    envelope = _envelope()
    envelope["choices"][0]["message"]["content"] = "{"
    invalid_review = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=envelope),
    )
    with pytest.raises(ProviderError) as second:
        invalid_review.review(_context())
    assert second.value.code == "invalid_review"


def test_schema_invalid_review_and_nonempty_tests_are_rejected() -> None:
    invalid = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope(report={"schema_version": "1.0"})),
    )
    with pytest.raises(ProviderError) as first:
        invalid.review(_context())
    assert first.value.code == "invalid_review"

    report = _report(tests_run=[{"command": "pytest", "status": "passed", "summary": "claimed"}])
    nonempty_tests = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope(report=report)),
    )
    with pytest.raises(ProviderError) as second:
        nonempty_tests.review(_context())
    assert second.value.code == "nonempty_tests_run"


def test_prompt_and_response_byte_limits_make_zero_or_one_calls() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"x" * 1_048_577)

    provider = _provider(GroqProvider, handler)
    with pytest.raises(ProviderError) as prompt_error:
        provider.review(_context("x" * MAX_PROMPT_BYTES))
    assert prompt_error.value.code == "prompt_too_large"
    assert calls == 0

    with pytest.raises(ProviderError) as response_error:
        provider.review(_context())
    assert response_error.value.code == "response_too_large"
    assert calls == 1


def test_configuration_errors_are_safe_and_make_no_calls() -> None:
    transport = httpx.MockTransport(
        lambda request: pytest.fail("HTTP must not be called for configuration errors")
    )
    with pytest.raises(ProviderError) as missing:
        GroqProvider(model=MODEL, api_key="", transport=transport)
    assert missing.value.kind == ProviderErrorKind.CONFIGURATION
    with pytest.raises(ProviderError) as invalid:
        GroqProvider(model="bad model", api_key=SECRET, transport=transport)
    assert invalid.value.kind == ProviderErrorKind.CONFIGURATION
    assert SECRET not in str(missing.value)
    assert SECRET not in repr(invalid.value)


def test_redirect_is_not_followed_and_request_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(307, headers={"location": "https://example.test/redirect"})

    with pytest.raises(ProviderError) as captured:
        _provider(GroqProvider, handler).review(_context())
    assert captured.value.kind == ProviderErrorKind.INVALID_REQUEST
    assert calls == 1


def _formatted_exception(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


def test_groq_usage_extras_are_ignored_and_cannot_leak() -> None:
    usage = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "queue_time": SECRET,
        "prompt_time": 0.25,
        "completion_time": 0.5,
        "total_time": 0.75,
    }
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope(usage=usage)),
    )
    result = provider.review(_context())
    assert result.usage is not None
    assert result.usage.model_dump() == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    assert SECRET not in result.model_dump_json()
    artifact = RunArtifact(
        schema_version="1.1",
        task_id="task",
        provider="groq",
        review=result.report,
        evaluation=EvaluationResult(
            schema_version="1.0",
            matched=[],
            missed_expected_indices=[],
            hallucinated_actual_indices=[],
            prohibited_hits=[],
            precision=1,
            recall=1,
            hallucination_rate=0,
            task_success=True,
        ),
        requested_model=result.requested_model,
        response_model=result.response_model,
        prompt_version=result.prompt_version,
        structured_output_mode=result.structured_output_mode,
        latency_ms=result.latency_ms,
        usage=result.usage,
    )
    assert SECRET not in artifact.model_dump_json()


@pytest.mark.parametrize("response_model", [None, ...])
def test_absent_or_null_response_model_is_allowed(response_model: Any) -> None:
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope(model=response_model)),
    )
    assert provider.review(_context()).response_model is None


def test_api_key_substring_in_requested_model_is_rejected_safely() -> None:
    with pytest.raises(ProviderError) as captured:
        GroqProvider(model=f"prefix/{SECRET}", api_key=SECRET)
    assert captured.value.code == "secret_in_model"
    assert SECRET not in str(captured.value)
    assert SECRET not in repr(captured.value)
    assert SECRET not in _formatted_exception(captured.value)


def test_secret_is_dropped_from_error_code_and_request_ids() -> None:
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(
            400,
            json={"error": {"code": SECRET, "request_id": SECRET}},
            headers={"x-request-id": SECRET},
        ),
    )
    with pytest.raises(ProviderError) as captured:
        provider.review(_context())
    assert captured.value.code is None
    assert captured.value.request_id is None
    assert SECRET not in str(captured.value)
    assert SECRET not in repr(captured.value)


def test_secret_in_response_model_is_rejected_safely() -> None:
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(
            200,
            json=_envelope(model=f"prefix/{SECRET}"),
        ),
    )
    with pytest.raises(ProviderError) as captured:
        provider.review(_context())
    assert captured.value.code == "secret_in_response"
    assert SECRET not in str(captured.value)
    assert SECRET not in repr(captured.value)


@pytest.mark.parametrize(
    "field",
    ["summary", "limitations", "title", "file", "evidence", "reasoning", "recommendation"],
)
def test_secret_in_validated_review_strings_is_rejected(field: str) -> None:
    report = _report()
    if field == "summary":
        report["summary"] = SECRET
    elif field == "limitations":
        report["limitations"] = [SECRET]
    else:
        finding = {
            "title": "Supported issue",
            "severity": "medium",
            "category": "bug",
            "file": "src/example.py",
            "line_start": 1,
            "line_end": 1,
            "evidence": "Changed line",
            "reasoning": "The changed line violates the requirement.",
            "recommendation": "Correct the changed line.",
            "confidence": 0.9,
        }
        finding[field] = SECRET
        report["findings"] = [finding]
    provider = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope(report=report)),
    )
    with pytest.raises(ProviderError) as captured:
        provider.review(_context())
    assert captured.value.code == "secret_in_response"
    assert SECRET not in str(captured.value)
    assert SECRET not in repr(captured.value)


def test_untrusted_parse_and_validation_causes_are_suppressed(tmp_path: Path) -> None:
    invalid_envelope = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, content=b"{" + SECRET.encode()),
    )
    invalid_report = _report(overall_assessment=SECRET)
    invalid_schema = _provider(
        GroqProvider,
        lambda request: httpx.Response(200, json=_envelope(report=invalid_report)),
    )
    invalid_usage = _provider(
        GroqProvider,
        lambda request: httpx.Response(
            200,
            json=_envelope(usage={"prompt_tokens": SECRET}),
        ),
    )
    for provider in (invalid_envelope, invalid_schema, invalid_usage):
        with pytest.raises(ProviderError) as captured:
            provider.review(_context())
        assert captured.value.__cause__ is None
        assert captured.value.__suppress_context__ is True
        assert SECRET not in _formatted_exception(captured.value)

    replay_path = tmp_path / "invalid-replay.json"
    replay_path.write_text("{" + SECRET, encoding="utf-8")
    with pytest.raises(ProviderError) as replay:
        ReplayProvider(replay_path)
    assert replay.value.__cause__ is None
    assert replay.value.__suppress_context__ is True
    assert SECRET not in _formatted_exception(replay.value)


def test_transport_traceback_does_not_include_authorization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(ProviderError) as captured:
        _provider(GroqProvider, handler).review(_context())
    assert SECRET not in _formatted_exception(captured.value)
