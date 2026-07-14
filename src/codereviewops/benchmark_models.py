"""Versioned DTOs for human-authored benchmark sources and suite manifests."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator

from codereviewops.evaluation import normalize_evaluator_text
from codereviewops.models import (
    Category,
    Difficulty,
    OverallAssessment,
    Severity,
    StrictModel,
    TestStatus,
    ToolPlan,
    normalize_relative_posix_path,
)

PrimaryCategory = Category | Literal["negative"]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


def _normalize_prohibited_phrase(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("prohibited phrases must be strings")
    normalized = value.strip()
    if not normalized or any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError("prohibited phrases must contain printable text")
    return normalized


ProhibitedPhrase = Annotated[
    str,
    BeforeValidator(_normalize_prohibited_phrase),
    Field(min_length=1, max_length=256),
]


class SourceFindingV1(StrictModel):
    category: Category
    severity: Severity
    file: str
    anchor: Annotated[str, Field(min_length=1, max_length=512)]
    description: Annotated[str, Field(min_length=1, max_length=1024)]
    title: Annotated[str, Field(min_length=1, max_length=256)]
    evidence: Annotated[str, Field(min_length=1, max_length=1024)]
    reasoning: Annotated[str, Field(min_length=1, max_length=2048)]
    recommendation: Annotated[str, Field(min_length=1, max_length=1024)]
    confidence: Confidence

    @field_validator("file")
    @classmethod
    def normalize_file(cls, value: str) -> str:
        return normalize_relative_posix_path(value)

    @field_validator("anchor")
    @classmethod
    def validate_anchor(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("anchor must be one exact source line")
        return value


class SourceCaseV1(StrictModel):
    schema_version: Literal["1.0"]
    task_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    title: Annotated[str, Field(min_length=1, max_length=256)]
    issue_description: Annotated[str, Field(min_length=1, max_length=2048)]
    difficulty: Difficulty
    tags: Annotated[list[str], Field(min_length=1, max_length=12)]
    primary_category: PrimaryCategory
    expected_findings: Annotated[list[SourceFindingV1], Field(max_length=4)]
    must_not_find: Annotated[list[ProhibitedPhrase], Field(min_length=1, max_length=4)]
    tool_plan: ToolPlan
    expected_test_status: TestStatus | None
    replay_summary: Annotated[str, Field(min_length=1, max_length=1024)]
    replay_assessment: OverallAssessment
    replay_limitations: Annotated[list[str], Field(max_length=8)]

    @model_validator(mode="after")
    def validate_case_shape(self) -> SourceCaseV1:
        negative = self.primary_category == "negative"
        if negative != (not self.expected_findings):
            raise ValueError(
                "negative cases must have no findings and positives must have findings"
            )
        if negative and self.replay_assessment != OverallAssessment.PASS:
            raise ValueError("negative cases require a pass replay assessment")
        if not negative and self.primary_category not in {
            finding.category for finding in self.expected_findings
        }:
            raise ValueError("positive primary category must match an expected finding")
        if any(finding.file not in self.tool_plan.read_files for finding in self.expected_findings):
            raise ValueError("every finding file must be present in planned reads")
        if not (
            self.tool_plan.read_files
            or self.tool_plan.searches
            or self.tool_plan.test_profile is not None
        ):
            raise ValueError("source cases require a non-empty tool plan")
        if self.tool_plan.test_profile is None and self.expected_test_status is not None:
            raise ValueError("test status requires a test profile")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("source tags must be unique")
        prohibited_keys = [normalize_evaluator_text(phrase) for phrase in self.must_not_find]
        if len(set(prohibited_keys)) != len(prohibited_keys):
            raise ValueError("prohibited phrases must be semantically unique")
        return self


class TaskEntryV1(StrictModel):
    task_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    task_path: str
    primary_category: PrimaryCategory
    difficulty: Difficulty
    negative: bool

    @field_validator("task_path")
    @classmethod
    def normalize_task_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)

    @model_validator(mode="after")
    def validate_negative_flag(self) -> TaskEntryV1:
        if self.negative != (self.primary_category == "negative"):
            raise ValueError("negative flag must match primary category")
        return self


class BenchmarkSuiteV1(StrictModel):
    schema_version: Literal["1.0"]
    suite_id: Annotated[str, Field(min_length=1, max_length=128)]
    suite_version: Annotated[str, Field(min_length=1, max_length=64)]
    expected_task_count: Literal[25]
    tasks: Annotated[list[TaskEntryV1], Field(min_length=25, max_length=25)]

    @model_validator(mode="after")
    def validate_tasks(self) -> BenchmarkSuiteV1:
        identifiers = [entry.task_id for entry in self.tasks]
        paths = [entry.task_path for entry in self.tasks]
        if len(set(identifiers)) != 25 or len(set(paths)) != 25:
            raise ValueError("suite tasks and paths must be unique")
        return self
