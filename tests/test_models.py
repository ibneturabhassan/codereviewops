from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from codereviewops.contracts import LIVE_STRUCTURED_OUTPUT_MODE, PROMPT_VERSION
from codereviewops.models import (
    MAX_FINDINGS_PER_TASK,
    BenchmarkTask,
    Category,
    Difficulty,
    ExpectedFinding,
    Finding,
    OverallAssessment,
    RunArtifact,
    Severity,
)
from codereviewops.models import (
    TestStatus as ReviewTestStatus,
)


def test_enums_expose_exact_contract_values() -> None:
    assert {item.value for item in Severity} == {"low", "medium", "high", "critical"}
    assert {item.value for item in Category} == {
        "bug",
        "requirement_mismatch",
        "missing_test",
        "performance",
        "security",
        "maintainability",
    }
    assert {item.value for item in OverallAssessment} == {
        "pass",
        "needs_changes",
        "fail",
        "uncertain",
    }
    assert {item.value for item in ReviewTestStatus} == {"passed", "failed", "error"}
    assert {item.value for item in Difficulty} == {"low", "medium", "high"}


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_finding_rejects_confidence_outside_unit_interval(
    finding_factory, confidence: float
) -> None:
    with pytest.raises(ValidationError):
        finding_factory(confidence=confidence)


@pytest.mark.parametrize(
    ("line_start", "line_end"),
    [(0, 1), (1, 0), (5, 4)],
)
def test_line_ranges_are_positive_and_ordered(
    expected_factory, line_start: int, line_end: int
) -> None:
    with pytest.raises(ValidationError):
        expected_factory(line_start=line_start, line_end=line_end)


def test_models_forbid_extra_fields(finding_factory) -> None:
    with pytest.raises(ValidationError, match="extra"):
        finding_factory(unexpected=True)


@pytest.mark.parametrize(
    "path",
    [
        "/src/service.py",
        "../src/service.py",
        "src/../service.py",
        "src\\..\\service.py",
        "src\\service.py",
        "./src/service.py",
        "C:/src/service.py",
        "",
    ],
)
def test_file_paths_reject_absolute_traversal_and_non_posix_forms(
    finding_factory, path: str
) -> None:
    with pytest.raises(ValidationError):
        finding_factory(file=path)


def test_normalized_relative_posix_path_is_preserved(finding_factory) -> None:
    finding: Finding = finding_factory(file="src/api/service.py")
    assert finding.file == "src/api/service.py"


def test_schema_version_is_fixed(expected_factory) -> None:
    data = {
        "schema_version": "2.0",
        "task_id": "task",
        "title": "Title",
        "issue_description": "Issue",
        "diff_path": "fixtures/change.diff",
        "expected_findings": [expected_factory().model_dump(mode="json")],
        "must_not_find": [],
        "difficulty": "low",
        "tags": [],
        "replay_response_path": "replays/review.json",
    }
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(data)


def test_expected_finding_has_exact_fields(expected_factory) -> None:
    finding: ExpectedFinding = expected_factory()
    assert set(finding.model_dump()) == {
        "category",
        "file",
        "line_start",
        "line_end",
        "description",
        "severity",
    }

    assert finding.severity is None


def test_legacy_benchmark_findings_forbid_severity(expected_factory) -> None:
    finding = expected_factory().model_dump(mode="json")
    finding["severity"] = "high"
    data = {
        "schema_version": "1.1",
        "task_id": "legacy",
        "title": "Legacy",
        "issue_description": "Legacy tool task",
        "diff_path": "fixtures/change.diff",
        "expected_findings": [finding],
        "must_not_find": [],
        "difficulty": "low",
        "tags": [],
        "replay_response_path": "replays/review.json",
        "workspace_path": "workspaces/legacy",
        "tool_plan": {"read_files": ["module.py"]},
    }
    with pytest.raises(ValidationError, match="cannot declare severity"):
        BenchmarkTask.model_validate(data)


def test_schema_12_requires_severity_workspace_and_nonempty_plan(expected_factory) -> None:
    finding = expected_factory().model_dump(mode="json")
    finding["severity"] = "high"
    data = {
        "schema_version": "1.2",
        "task_id": "canonical",
        "title": "Canonical",
        "issue_description": "Canonical tool task",
        "diff_path": "fixtures/change.diff",
        "expected_findings": [finding],
        "must_not_find": [],
        "difficulty": "low",
        "tags": [],
        "replay_response_path": "replays/review.json",
        "workspace_path": "workspaces/canonical",
        "tool_plan": {},
    }
    with pytest.raises(ValidationError, match="non-empty tool plan"):
        BenchmarkTask.model_validate(data)


def test_benchmark_rejects_more_than_maximum_expected_findings(
    expected_factory,
) -> None:
    expected = expected_factory().model_dump(mode="json")
    data = {
        "schema_version": "1.0",
        "task_id": "task",
        "title": "Title",
        "issue_description": "Issue",
        "diff_path": "fixtures/change.diff",
        "expected_findings": [expected] * (MAX_FINDINGS_PER_TASK + 1),
        "must_not_find": [],
        "difficulty": "low",
        "tags": [],
        "replay_response_path": "replays/review.json",
    }
    with pytest.raises(ValidationError, match="too_long"):
        BenchmarkTask.model_validate(data)


def test_review_rejects_more_than_maximum_findings(
    finding_factory,
    report_factory,
) -> None:
    findings = [finding_factory()] * (MAX_FINDINGS_PER_TASK + 1)
    with pytest.raises(ValidationError, match="too_long"):
        report_factory(findings)


def test_legacy_run_artifact_fixture_remains_valid() -> None:
    fixture = (Path(__file__).parent / "fixtures" / "artifacts" / "run_v1.json").read_text(
        encoding="utf-8"
    )
    artifact = RunArtifact.model_validate_json(fixture)
    assert artifact.schema_version == "1.0"
    assert artifact.requested_model is None
    assert artifact.structured_output_mode is None


def _artifact_data() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "artifacts" / "run_v1.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def _live_artifact_data(**overrides: Any) -> dict[str, Any]:
    data = _artifact_data()
    data.update(
        schema_version="1.1",
        provider="groq",
        requested_model="request/model-1",
        response_model=None,
        prompt_version=PROMPT_VERSION,
        structured_output_mode=LIVE_STRUCTURED_OUTPUT_MODE,
        latency_ms=1.25,
        usage=None,
    )
    data.update(overrides)
    return data


def test_live_run_artifact_allows_missing_response_model() -> None:
    artifact = RunArtifact.model_validate(_live_artifact_data())
    assert artifact.provider == "groq"
    assert artifact.response_model is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "unsupported"},
        {"requested_model": "bad model"},
        {"requested_model": "model\ncontrol"},
        {"response_model": "bad model"},
        {"response_model": "model\ncontrol"},
        {"prompt_version": "review-v2"},
        {"prompt_version": None},
        {"structured_output_mode": "json_schema"},
        {"latency_ms": -0.01},
        {"latency_ms": float("nan")},
    ],
)
def test_live_run_artifact_rejects_invalid_contract_fields(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        RunArtifact.model_validate(_live_artifact_data(**overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_model", "model"),
        ("response_model", "model"),
        ("prompt_version", PROMPT_VERSION),
        ("structured_output_mode", LIVE_STRUCTURED_OUTPUT_MODE),
        ("latency_ms", 1),
        ("usage", {"prompt_tokens": 1}),
    ],
)
def test_replay_run_artifact_requires_exact_deterministic_metadata(
    field: str,
    value: Any,
) -> None:
    data = _artifact_data()
    data.update(
        schema_version="1.1",
        provider="replay",
        requested_model=None,
        response_model=None,
        prompt_version=None,
        structured_output_mode="replay",
        latency_ms=0,
        usage=None,
    )
    data[field] = value
    with pytest.raises(ValidationError):
        RunArtifact.model_validate(data)
