"""Strict, versioned models for benchmark inputs and run artifacts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SchemaVersion = Literal["1.0"]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveLine = Annotated[int, Field(ge=1)]
MAX_FINDINGS_PER_TASK = 100


class StrictModel(BaseModel):
    """Base model that rejects undeclared input fields."""

    model_config = ConfigDict(extra="forbid")


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(StrEnum):
    BUG = "bug"
    REQUIREMENT_MISMATCH = "requirement_mismatch"
    MISSING_TEST = "missing_test"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"


class OverallAssessment(StrEnum):
    PASS = "pass"
    NEEDS_CHANGES = "needs_changes"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class TestStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class Difficulty(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def normalize_relative_posix_path(value: str) -> str:
    """Validate and canonicalize a repository-relative POSIX path."""

    if not value or "\\" in value:
        raise ValueError("path must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError("path must be normalized and may not traverse directories")
    normalized = path.as_posix()
    if normalized != value or normalized in {"", "."}:
        raise ValueError("path must be normalized and repository-relative")
    return normalized


class LineRangeModel(StrictModel):
    line_start: PositiveLine
    line_end: PositiveLine

    @model_validator(mode="after")
    def validate_line_range(self) -> LineRangeModel:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class ExpectedFinding(LineRangeModel):
    category: Category
    file: str
    description: str

    @field_validator("file")
    @classmethod
    def normalize_file(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class BenchmarkTask(StrictModel):
    schema_version: SchemaVersion
    task_id: str
    title: str
    issue_description: str
    diff_path: str
    expected_findings: Annotated[list[ExpectedFinding], Field(max_length=MAX_FINDINGS_PER_TASK)]
    must_not_find: list[str]
    difficulty: Difficulty
    tags: list[str]
    replay_response_path: str

    @field_validator("diff_path", "replay_response_path")
    @classmethod
    def normalize_reference(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class ReviewContext(StrictModel):
    schema_version: SchemaVersion
    task_id: str
    title: str
    issue_description: str
    diff_text: str


class Finding(LineRangeModel):
    title: str
    severity: Severity
    category: Category
    file: str
    evidence: str
    reasoning: str
    recommendation: str
    confidence: Confidence

    @field_validator("file")
    @classmethod
    def normalize_file(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class TestRun(StrictModel):
    command: str
    status: TestStatus
    summary: str


class ReviewReport(StrictModel):
    schema_version: SchemaVersion
    summary: str
    overall_assessment: OverallAssessment
    findings: Annotated[list[Finding], Field(max_length=MAX_FINDINGS_PER_TASK)]
    tests_run: list[TestRun]
    limitations: list[str]


class FindingMatch(StrictModel):
    expected_index: int
    actual_index: int


class ProhibitedHit(StrictModel):
    phrase: str
    actual_index: int


class EvaluationResult(StrictModel):
    schema_version: SchemaVersion
    matched: list[FindingMatch]
    missed_expected_indices: list[int]
    hallucinated_actual_indices: list[int]
    prohibited_hits: list[ProhibitedHit]
    precision: float
    recall: float
    hallucination_rate: float
    task_success: bool


class RunArtifact(StrictModel):
    schema_version: SchemaVersion
    task_id: str
    provider: str
    review: ReviewReport
    evaluation: EvaluationResult
