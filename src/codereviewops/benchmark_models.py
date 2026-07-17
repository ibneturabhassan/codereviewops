"""Versioned DTOs for human-authored benchmark sources and suite manifests."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field, field_validator, model_validator

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
REQUIRED_HASH_KEYS = frozenset(
    {
        "suite_content",
        "selection",
        "matrix",
        "configuration",
        "contracts",
        "tasks",
        "suite",
        "results",
    }
)


def _validate_hash_keys(value: dict[str, str]) -> dict[str, str]:
    if set(value) != REQUIRED_HASH_KEYS:
        raise ValueError("benchmark hashes must contain exactly the required keys")
    return value


HashValue = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
HashesV1 = Annotated[dict[str, HashValue], AfterValidator(_validate_hash_keys)]

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


ProviderName = Literal["replay", "groq", "mistral"]
ToolTransport = Literal["direct", "mcp-stdio"]
Rate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
FiniteNonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False, strict=True)]


def _rate(numerator: int, denominator: int, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _same_rate(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1e-12


class RunVariantV1(StrictModel):
    variant_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    provider: ProviderName
    model: Annotated[str, Field(min_length=1, max_length=128)] | None
    request_budget: Annotated[int, Field(ge=0, strict=True)]
    prompt_version: Annotated[str, Field(min_length=1, max_length=64)]
    agent_version: Annotated[str, Field(min_length=1, max_length=64)]
    planner_version: Annotated[str, Field(min_length=1, max_length=64)]
    tool_transport: ToolTransport

    @model_validator(mode="after")
    def validate_provider_budget(self) -> RunVariantV1:
        if self.provider == "replay":
            if self.model is not None or self.request_budget != 0:
                raise ValueError("replay variants require a null model and zero request budget")
        elif self.model is None or self.request_budget <= 0:
            raise ValueError("live variants require an explicit model and positive request budget")
        return self


class ComparisonV1(StrictModel):
    comparison_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    baseline_variant: Annotated[str, Field(min_length=1, max_length=128)]
    candidate_variant: Annotated[str, Field(min_length=1, max_length=128)]


class ThresholdProfileV1(StrictModel):
    completion_rate: Rate = 1.0
    task_success_rate: Rate = 1.0
    micro_precision: Rate = 1.0
    micro_recall: Rate = 1.0
    macro_precision: Rate = 1.0
    macro_recall: Rate = 1.0
    category_recall: Rate = 1.0
    severity_accuracy: Rate = 1.0
    tool_plan_accuracy: Rate = 1.0
    trace_equivalence: Rate = 1.0
    hallucination_rate: Rate = 0.0
    missed_rate: Rate = 0.0
    negative_false_positives: NonNegativeInt = 0
    prohibited_hits: NonNegativeInt = 0
    tool_failed: NonNegativeInt = 0
    tool_missing: NonNegativeInt = 0
    tool_unexpected: NonNegativeInt = 0
    tool_mismatched: NonNegativeInt = 0


class BenchmarkMatrixV1(StrictModel):
    schema_version: Literal["1.0"]
    matrix_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    suite_path: str
    variants: Annotated[list[RunVariantV1], Field(min_length=1)]
    comparisons: list[ComparisonV1] = Field(default_factory=list)
    threshold_profile: ThresholdProfileV1
    baseline_path: str | None = None
    max_concurrency: Literal[1]

    @field_validator("suite_path", "baseline_path")
    @classmethod
    def normalize_matrix_path(cls, value: str | None) -> str | None:
        return normalize_relative_posix_path(value) if value is not None else None

    @model_validator(mode="after")
    def validate_matrix(self) -> BenchmarkMatrixV1:
        identifiers = [variant.variant_id for variant in self.variants]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("variant identifiers must be unique")
        variants = {variant.variant_id: variant for variant in self.variants}
        comparison_ids = [comparison.comparison_id for comparison in self.comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise ValueError("comparison identifiers must be unique")
        for comparison in self.comparisons:
            left = variants.get(comparison.baseline_variant)
            right = variants.get(comparison.candidate_variant)
            if left is None or right is None or left is right:
                raise ValueError("comparisons must reference two distinct matrix variants")
            left_contract = left.model_dump(exclude={"variant_id", "tool_transport"})
            right_contract = right.model_dump(exclude={"variant_id", "tool_transport"})
            if left_contract != right_contract or left.tool_transport == right.tool_transport:
                raise ValueError("compared variants may differ only by tool transport")
        return self


class StableTaskMetricsV1(StrictModel):
    task_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    completed: bool
    task_success: bool
    expected_count: NonNegativeInt
    actual_count: NonNegativeInt
    true_positive_count: NonNegativeInt
    missed_count: NonNegativeInt
    hallucinated_count: NonNegativeInt
    prohibited_count: NonNegativeInt
    precision: Rate
    recall: Rate
    severity_correct: NonNegativeInt
    severity_total: NonNegativeInt
    tool_plan_correct: NonNegativeInt
    tool_plan_total: NonNegativeInt
    negative_false_positive: bool
    tool_planned: NonNegativeInt
    tool_observed: NonNegativeInt
    tool_succeeded: NonNegativeInt
    tool_failed: NonNegativeInt
    tool_missing: NonNegativeInt
    tool_unexpected: NonNegativeInt
    tool_mismatched: NonNegativeInt
    category_expected: dict[str, NonNegativeInt]
    category_matched: dict[str, NonNegativeInt]
    rejection_codes: list[str]
    test_statuses: list[str]
    semantic_trace_fingerprint: str | None

    @model_validator(mode="after")
    def validate_counts(self) -> StableTaskMetricsV1:
        if (
            self.true_positive_count > min(self.expected_count, self.actual_count)
            or self.missed_count != self.expected_count - self.true_positive_count
            or self.hallucinated_count != self.actual_count - self.true_positive_count
            or self.severity_correct > self.severity_total
            or self.severity_total > self.true_positive_count
            or self.tool_plan_correct > self.tool_plan_total
            or sum(self.category_expected.values()) != self.expected_count
            or sum(self.category_matched.values()) != self.true_positive_count
            or self.tool_plan_total != max(self.tool_planned, self.tool_observed)
            or self.tool_succeeded + self.tool_failed != self.tool_observed
            or self.tool_missing != max(self.tool_planned - self.tool_observed, 0)
            or self.tool_unexpected != max(self.tool_observed - self.tool_planned, 0)
            or self.tool_mismatched > min(self.tool_planned, self.tool_observed)
            or self.tool_plan_correct + self.tool_mismatched
            > min(self.tool_planned, self.tool_observed)
            or any(
                count > self.category_expected.get(category, 0)
                for category, count in self.category_matched.items()
            )
        ):
            raise ValueError("task metric counts are inconsistent")
        expected_precision = _rate(
            self.true_positive_count,
            self.actual_count,
            1.0 if self.expected_count == 0 else 0.0,
        )
        expected_recall = _rate(self.true_positive_count, self.expected_count, 1.0)
        expected_success = (
            self.completed
            and self.missed_count == 0
            and self.hallucinated_count == 0
            and self.prohibited_count == 0
        )
        expected_negative_fp = self.expected_count == 0 and self.actual_count > 0
        if (
            not _same_rate(self.precision, expected_precision)
            or not _same_rate(self.recall, expected_recall)
            or self.task_success != expected_success
            or self.negative_false_positive != expected_negative_fp
        ):
            raise ValueError("task metric rates or flags are inconsistent")
        return self


class TaskMetricsV1(StableTaskMetricsV1):
    latency_ms: FiniteNonNegative | None
    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None


class StableAggregateMetricsV1(StrictModel):
    task_count: Annotated[int, Field(ge=1, strict=True)]
    completed_count: NonNegativeInt
    successful_count: NonNegativeInt
    expected_count: NonNegativeInt
    actual_count: NonNegativeInt
    true_positive_count: NonNegativeInt
    missed_count: NonNegativeInt
    hallucinated_count: NonNegativeInt
    prohibited_count: NonNegativeInt
    negative_false_positives: NonNegativeInt
    completion_rate: Rate
    task_success_rate: Rate
    micro_precision: Rate
    micro_recall: Rate
    macro_precision: Rate
    macro_recall: Rate
    hallucination_rate: Rate
    missed_rate: Rate
    category_recall: dict[str, Rate]
    severity_accuracy: Rate
    difficulty_success: dict[str, Rate]
    tool_plan_accuracy: Rate
    tool_planned: NonNegativeInt
    tool_observed: NonNegativeInt
    tool_succeeded: NonNegativeInt
    tool_failed: NonNegativeInt
    tool_missing: NonNegativeInt
    tool_unexpected: NonNegativeInt
    tool_mismatched: NonNegativeInt
    rejection_codes: dict[str, NonNegativeInt]
    test_statuses: dict[str, NonNegativeInt]

    @model_validator(mode="after")
    def validate_counts(self) -> StableAggregateMetricsV1:
        if (
            self.completed_count > self.task_count
            or self.successful_count > self.completed_count
            or self.true_positive_count > min(self.expected_count, self.actual_count)
            or self.missed_count != self.expected_count - self.true_positive_count
            or self.hallucinated_count != self.actual_count - self.true_positive_count
            or self.negative_false_positives > self.task_count
            or self.tool_succeeded + self.tool_failed != self.tool_observed
        ):
            raise ValueError("aggregate metric counts are inconsistent")
        expected = {
            "completion_rate": self.completed_count / self.task_count,
            "task_success_rate": self.successful_count / self.task_count,
            "micro_precision": _rate(
                self.true_positive_count,
                self.actual_count,
                1.0 if self.expected_count == 0 else 0.0,
            ),
            "micro_recall": _rate(self.true_positive_count, self.expected_count, 1.0),
            "hallucination_rate": _rate(self.hallucinated_count, self.actual_count, 0.0),
            "missed_rate": _rate(self.missed_count, self.expected_count, 0.0),
        }
        if any(not _same_rate(getattr(self, name), value) for name, value in expected.items()):
            raise ValueError("aggregate metric rates are inconsistent")
        return self


class AggregateMetricsV1(StableAggregateMetricsV1):
    latency_ms_total: FiniteNonNegative
    input_tokens_total: NonNegativeInt
    output_tokens_total: NonNegativeInt


class StableVariantResultV1(StrictModel):
    variant: RunVariantV1
    tasks: Annotated[list[StableTaskMetricsV1], Field(min_length=1)]
    metrics: StableAggregateMetricsV1
    gate_passed: bool
    gate_failures: list[str]

    @model_validator(mode="after")
    def validate_tasks(self) -> StableVariantResultV1:
        category_expected: dict[str, int] = {}
        category_matched: dict[str, int] = {}
        for task in self.tasks:
            for category, count in task.category_expected.items():
                category_expected[category] = category_expected.get(category, 0) + count
            for category, count in task.category_matched.items():
                category_matched[category] = category_matched.get(category, 0) + count
        expected_category_recall = {
            category: _rate(category_matched.get(category, 0), count, 1.0)
            for category, count in category_expected.items()
        }
        severity_total = sum(task.severity_total for task in self.tasks)
        tool_total = sum(task.tool_plan_total for task in self.tasks)
        expected_rates = {
            "macro_precision": sum(task.precision for task in self.tasks) / len(self.tasks),
            "macro_recall": sum(task.recall for task in self.tasks) / len(self.tasks),
            "severity_accuracy": _rate(
                sum(task.severity_correct for task in self.tasks), severity_total, 1.0
            ),
            "tool_plan_accuracy": _rate(
                sum(task.tool_plan_correct for task in self.tasks), tool_total, 1.0
            ),
        }
        if (
            len(self.tasks) != self.metrics.task_count
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
            or self.metrics.completed_count != sum(task.completed for task in self.tasks)
            or self.metrics.successful_count != sum(task.task_success for task in self.tasks)
            or self.metrics.expected_count != sum(task.expected_count for task in self.tasks)
            or self.metrics.actual_count != sum(task.actual_count for task in self.tasks)
            or self.metrics.true_positive_count
            != sum(task.true_positive_count for task in self.tasks)
            or self.metrics.missed_count != sum(task.missed_count for task in self.tasks)
            or self.metrics.hallucinated_count
            != sum(task.hallucinated_count for task in self.tasks)
            or self.metrics.prohibited_count != sum(task.prohibited_count for task in self.tasks)
            or self.metrics.negative_false_positives
            != sum(task.negative_false_positive for task in self.tasks)
            or self.metrics.category_recall != expected_category_recall
            or any(
                not _same_rate(getattr(self.metrics, name), value)
                for name, value in expected_rates.items()
            )
            or self.metrics.tool_planned != sum(task.tool_planned for task in self.tasks)
            or self.metrics.tool_observed != sum(task.tool_observed for task in self.tasks)
            or self.metrics.tool_succeeded != sum(task.tool_succeeded for task in self.tasks)
            or self.metrics.tool_failed != sum(task.tool_failed for task in self.tasks)
            or self.metrics.tool_missing != sum(task.tool_missing for task in self.tasks)
            or self.metrics.tool_unexpected != sum(task.tool_unexpected for task in self.tasks)
            or self.metrics.tool_mismatched != sum(task.tool_mismatched for task in self.tasks)
            or self.metrics.rejection_codes
            != dict(Counter(code for task in self.tasks for code in task.rejection_codes))
            or self.metrics.test_statuses
            != dict(Counter(status for task in self.tasks for status in task.test_statuses))
            or self.gate_passed == bool(self.gate_failures)
        ):
            raise ValueError("variant metrics are inconsistent with task metrics")
        return self


class VariantResultV1(StrictModel):
    variant: RunVariantV1
    tasks: Annotated[list[TaskMetricsV1], Field(min_length=1)]
    metrics: AggregateMetricsV1
    gate_passed: bool
    gate_failures: list[str]

    @model_validator(mode="after")
    def validate_tasks(self) -> VariantResultV1:
        StableVariantResultV1.model_validate(
            {
                "variant": self.variant,
                "tasks": [
                    task.model_dump(exclude={"latency_ms", "input_tokens", "output_tokens"})
                    for task in self.tasks
                ],
                "metrics": self.metrics.model_dump(
                    exclude={
                        "latency_ms_total",
                        "input_tokens_total",
                        "output_tokens_total",
                    }
                ),
                "gate_passed": self.gate_passed,
                "gate_failures": self.gate_failures,
            }
        )
        if (
            not _same_rate(
                self.metrics.latency_ms_total,
                sum(task.latency_ms or 0.0 for task in self.tasks),
            )
            or self.metrics.input_tokens_total != sum(task.input_tokens or 0 for task in self.tasks)
            or self.metrics.output_tokens_total
            != sum(task.output_tokens or 0 for task in self.tasks)
        ):
            raise ValueError("informational totals are inconsistent")
        return self


class ComparisonResultV1(StrictModel):
    comparison_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    baseline_variant: str
    candidate_variant: str
    metric_deltas: dict[str, Annotated[float, Field(allow_inf_nan=False, strict=True)]]
    trace_equivalence: Rate
    task_trace_equivalence: dict[str, bool]
    passed: bool

    @model_validator(mode="after")
    def validate_trace_rate(self) -> ComparisonResultV1:
        if not self.task_trace_equivalence:
            raise ValueError("comparison requires task trace results")
        expected = sum(self.task_trace_equivalence.values()) / len(self.task_trace_equivalence)
        if not _same_rate(self.trace_equivalence, expected):
            raise ValueError("comparison trace rate is inconsistent")
        return self


class BenchmarkRunV1(StrictModel):
    schema_version: Literal["1.0"]
    matrix_id: str
    suite_id: str
    selected_task_ids: Annotated[list[str], Field(min_length=1)]
    variants: Annotated[list[VariantResultV1], Field(min_length=1)]
    comparisons: list[ComparisonResultV1]
    quality_gate_passed: bool
    baseline_passed: bool | None
    baseline_failures: list[str]
    hashes: HashesV1

    @model_validator(mode="after")
    def validate_result(self) -> BenchmarkRunV1:
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("selected task identifiers must be unique")
        variant_ids = [result.variant.variant_id for result in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("result variant identifiers must be unique")
        for result in self.variants:
            if [task.task_id for task in result.tasks] != self.selected_task_ids:
                raise ValueError("variant task order must match selected tasks")
        known = set(variant_ids)
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise ValueError("result comparison identifiers must be unique")
        if any(
            comparison.baseline_variant not in known
            or comparison.candidate_variant not in known
            or comparison.baseline_variant == comparison.candidate_variant
            for comparison in self.comparisons
        ):
            raise ValueError("result comparison references an unknown variant")
        expected_gate = all(result.gate_passed for result in self.variants) and all(
            comparison.passed for comparison in self.comparisons
        )
        if self.quality_gate_passed != expected_gate:
            raise ValueError("result quality gate is inconsistent")
        if (self.baseline_passed in {None, True} and self.baseline_failures) or (
            self.baseline_passed is False and not self.baseline_failures
        ):
            raise ValueError("baseline status is inconsistent")
        return self


class StableBenchmarkRunV1(StrictModel):
    schema_version: Literal["1.0"]
    matrix_id: str
    suite_id: str
    selected_task_ids: Annotated[list[str], Field(min_length=1)]
    variants: Annotated[list[StableVariantResultV1], Field(min_length=1)]
    comparisons: list[ComparisonResultV1]
    quality_gate_passed: bool
    hashes: HashesV1

    @model_validator(mode="after")
    def validate_result(self) -> StableBenchmarkRunV1:
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("selected task identifiers must be unique")
        variant_ids = [result.variant.variant_id for result in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("stable variant identifiers must be unique")
        if any(
            [task.task_id for task in result.tasks] != self.selected_task_ids
            for result in self.variants
        ):
            raise ValueError("stable variant tasks must match selected tasks")
        known = set(variant_ids)
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(set(comparison_ids)) != len(comparison_ids):
            raise ValueError("stable comparison identifiers must be unique")
        if any(
            comparison.baseline_variant not in known
            or comparison.candidate_variant not in known
            or comparison.baseline_variant == comparison.candidate_variant
            for comparison in self.comparisons
        ):
            raise ValueError("stable comparison references are invalid")
        expected_gate = all(result.gate_passed for result in self.variants) and all(
            comparison.passed for comparison in self.comparisons
        )
        if self.quality_gate_passed != expected_gate:
            raise ValueError("stable quality gate is inconsistent")
        return self


class BenchmarkBaselineV1(StrictModel):
    schema_version: Literal["1.0"]
    matrix_id: str
    suite_id: str
    selected_task_ids: Annotated[list[str], Field(min_length=1)]
    stable_result: StableBenchmarkRunV1
    hashes: HashesV1

    @model_validator(mode="after")
    def validate_projection(self) -> BenchmarkBaselineV1:
        if (
            self.matrix_id != self.stable_result.matrix_id
            or self.suite_id != self.stable_result.suite_id
            or self.selected_task_ids != self.stable_result.selected_task_ids
            or self.hashes != self.stable_result.hashes
        ):
            raise ValueError("baseline metadata disagrees with stable result")
        return self
