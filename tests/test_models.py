from __future__ import annotations

import pytest
from pydantic import ValidationError

from codereviewops.models import (
    MAX_FINDINGS_PER_TASK,
    BenchmarkTask,
    Category,
    Difficulty,
    ExpectedFinding,
    Finding,
    OverallAssessment,
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
    }


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
