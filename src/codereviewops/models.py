"""Strict, versioned models for benchmark inputs and run artifacts."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codereviewops.contracts import (
    ARTIFACT_PROVIDERS,
    LIVE_PROVIDERS,
    LIVE_STRUCTURED_OUTPUT_MODE,
    PROMPT_VERSION,
    is_valid_model_identifier,
)

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


class Usage(StrictModel):
    prompt_tokens: Annotated[int, Field(ge=0, strict=True)] | None = None
    completion_tokens: Annotated[int, Field(ge=0, strict=True)] | None = None
    total_tokens: Annotated[int, Field(ge=0, strict=True)] | None = None

    @model_validator(mode="after")
    def validate_total(self) -> Usage:
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")
        return self


class ProviderResult(StrictModel):
    report: ReviewReport
    requested_model: str | None
    response_model: str | None
    prompt_version: str | None
    structured_output_mode: str
    latency_ms: float
    usage: Usage | None

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("latency_ms must be finite and nonnegative")
        return value


class RunArtifact(StrictModel):
    schema_version: Literal["1.0", "1.1"]
    task_id: str
    provider: str
    review: ReviewReport
    evaluation: EvaluationResult
    requested_model: str | None = None
    response_model: str | None = None
    prompt_version: str | None = None
    structured_output_mode: str | None = None
    latency_ms: float | None = None
    usage: Usage | None = None

    @model_validator(mode="after")
    def validate_provider_metadata(self) -> RunArtifact:
        if self.schema_version == "1.0":
            return self
        if self.provider not in ARTIFACT_PROVIDERS:
            raise ValueError("1.1 artifacts require a supported provider")
        if self.provider == "replay":
            if (
                self.requested_model is not None
                or self.response_model is not None
                or self.prompt_version is not None
                or self.structured_output_mode != "replay"
                or self.latency_ms != 0
                or self.usage is not None
            ):
                raise ValueError("replay 1.1 metadata must use deterministic null values")
            return self
        if (
            self.provider not in LIVE_PROVIDERS
            or self.requested_model is None
            or not is_valid_model_identifier(self.requested_model)
            or (
                self.response_model is not None
                and not is_valid_model_identifier(self.response_model)
            )
            or self.prompt_version != PROMPT_VERSION
            or self.structured_output_mode != LIVE_STRUCTURED_OUTPUT_MODE
            or self.latency_ms is None
            or not isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("live 1.1 artifacts require complete safe provider metadata")
        return self
