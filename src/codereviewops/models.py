"""Strict, versioned models for benchmark inputs and run artifacts."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from codereviewops.contracts import (
    ARTIFACT_PROVIDERS,
    LIVE_PROVIDERS,
    LIVE_STRUCTURED_OUTPUT_MODE,
    PROMPT_VERSION,
    TOOL_PROMPT_VERSION,
    is_valid_model_identifier,
)

SchemaVersion = Literal["1.0"]
BenchmarkSchemaVersion = Literal["1.0", "1.1"]
ReportSchemaVersion = Literal["1.0", "1.2"]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveLine = Annotated[int, Field(ge=1)]
MAX_FINDINGS_PER_TASK = 100

MAX_TOOL_READS = 20
MAX_TOOL_SEARCHES = 10
MAX_SAFE_TEXT = 64 * 1024
MAX_PROVENANCE = 100
MAX_TRACE_ENTRIES = MAX_TOOL_READS + MAX_TOOL_SEARCHES + 1
SafeText = Annotated[str, Field(max_length=MAX_SAFE_TEXT)]
ShortText = Annotated[str, Field(min_length=1, max_length=256)]
SafePath = Annotated[str, Field(min_length=1, max_length=512)]
FindingIndex = Annotated[int, Field(ge=0, lt=MAX_FINDINGS_PER_TASK, strict=True)]


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


class ToolName(StrEnum):
    READ_FILE = "read_file"
    SEARCH_CODE = "search_code"
    RUN_TESTS = "run_tests"


class ToolStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkflowState(StrEnum):
    INTAKE = "intake"
    CONTEXT = "context"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    REPORT = "report"
    EVALUATION = "evaluation"
    COMPLETE = "complete"
    FAILED = "failed"


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


class ChangedLocation(LineRangeModel):
    path: SafePath

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class ExpectedFinding(LineRangeModel):
    category: Category
    file: str
    description: str

    @field_validator("file")
    @classmethod
    def normalize_file(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class ToolPlan(StrictModel):
    read_files: Annotated[list[str], Field(max_length=MAX_TOOL_READS)] = []
    searches: Annotated[list[str], Field(max_length=MAX_TOOL_SEARCHES)] = []
    test_profile: Literal["python-unittest-v1"] | None = None

    @model_validator(mode="after")
    def validate_unique_entries(self) -> ToolPlan:
        if len(set(self.read_files)) != len(self.read_files) or len(set(self.searches)) != len(
            self.searches
        ):
            raise ValueError("tool plan entries must be unique")
        return self

    @field_validator("read_files")
    @classmethod
    def normalize_read_files(cls, values: list[str]) -> list[str]:
        return [normalize_relative_posix_path(value) for value in values]

    @field_validator("searches")
    @classmethod
    def validate_searches(cls, values: list[str]) -> list[str]:
        if any(
            not value or len(value) > 128 or any(ord(char) < 32 for char in value)
            for value in values
        ):
            raise ValueError("search queries must contain 1-128 printable characters")
        return values


class BenchmarkTask(StrictModel):
    schema_version: BenchmarkSchemaVersion
    task_id: str
    title: str
    issue_description: str
    diff_path: str
    expected_findings: Annotated[list[ExpectedFinding], Field(max_length=MAX_FINDINGS_PER_TASK)]
    must_not_find: list[str]
    difficulty: Difficulty
    tags: list[str]
    replay_response_path: str

    workspace_path: str | None = None
    tool_plan: ToolPlan | None = None

    @field_validator("diff_path", "replay_response_path", "workspace_path")
    @classmethod
    def normalize_reference(cls, value: str | None) -> str | None:
        return normalize_relative_posix_path(value) if value is not None else None

    @model_validator(mode="after")
    def validate_tool_configuration(self) -> BenchmarkTask:
        if self.schema_version == "1.0" and (
            self.workspace_path is not None or self.tool_plan is not None
        ):
            raise ValueError("schema 1.0 tasks cannot configure tools")
        if (self.workspace_path is None) != (self.tool_plan is None):
            raise ValueError("workspace_path and tool_plan must be configured together")
        if self.schema_version == "1.1" and self.workspace_path is None:
            raise ValueError("schema 1.1 tasks require workspace_path and tool_plan")
        return self


class ReadFileArguments(StrictModel):
    kind: Literal["read_file"]
    path: SafePath

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class SearchCodeArguments(StrictModel):
    kind: Literal["search_code"]
    query: Annotated[str, Field(min_length=1, max_length=128)]


class RunTestsArguments(StrictModel):
    kind: Literal["run_tests"]
    profile: Literal["python-unittest-v1"]


ToolArguments = Annotated[
    ReadFileArguments | SearchCodeArguments | RunTestsArguments,
    Field(discriminator="kind"),
]


class ReadFileResult(StrictModel):
    kind: Literal["read_file_success"]
    path: SafePath
    source_bytes: Annotated[int, Field(ge=0, le=256 * 1024, strict=True)]
    normalized_bytes: Annotated[int, Field(ge=0, le=256 * 1024, strict=True)]
    returned_bytes: Annotated[int, Field(ge=0, le=256 * 1024, strict=True)]
    content: SafeText
    truncated: bool

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)

    @model_validator(mode="after")
    def validate_accounting(self) -> ReadFileResult:
        if self.returned_bytes != len(self.content.encode("utf-8")):
            raise ValueError("returned byte count must match content")
        if self.returned_bytes > self.normalized_bytes:
            raise ValueError("returned bytes cannot exceed normalized bytes")
        if self.truncated != (self.returned_bytes < self.normalized_bytes):
            raise ValueError("read truncation flag is inconsistent")
        return self


class SearchMatch(StrictModel):
    path: SafePath
    line: PositiveLine
    column: PositiveLine
    excerpt: Annotated[str, Field(max_length=240)]

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class SearchCodeResult(StrictModel):
    kind: Literal["search_code_success"]
    query: Annotated[str, Field(min_length=1, max_length=128)]
    files_total: Annotated[int, Field(ge=0, le=2_000, strict=True)]
    files_scanned: Annotated[int, Field(ge=0, le=2_000, strict=True)]
    files_skipped: Annotated[int, Field(ge=0, le=2_000, strict=True)]
    matches: Annotated[list[SearchMatch], Field(max_length=100)]
    truncated: bool

    @model_validator(mode="after")
    def validate_accounting(self) -> SearchCodeResult:
        accounted = self.files_scanned + self.files_skipped
        if accounted > self.files_total:
            raise ValueError("search file accounting exceeds total")
        if not self.truncated and accounted != self.files_total:
            raise ValueError("complete search must account for every file")
        return self


class RunTestsResult(StrictModel):
    kind: Literal["run_tests_success"]
    command: Literal["python -B -m unittest discover -s tests -p test_*.py"]
    profile: Literal["python-unittest-v1"]
    status: TestStatus
    summary: SafeText
    output_truncated: bool


ToolErrorCode = Literal[
    "invalid_path",
    "changed_workspace",
    "read_failed",
    "invalid_utf8",
    "binary_file",
    "limit_exceeded",
    "docker_unavailable",
    "docker_infrastructure",
    "cleanup_failed",
    "unsupported_profile",
    "invalid_diff",
    "diff_no_added_lines",
    "unsupported_text",
]


class ToolFailureResult(StrictModel):
    kind: Literal["tool_failure"]
    code: ToolErrorCode
    message: ShortText
    retryable: bool = False


ToolResult = Annotated[
    ReadFileResult | SearchCodeResult | RunTestsResult | ToolFailureResult,
    Field(discriminator="kind"),
]


class TraceProvenance(StrictModel):
    path: SafePath
    line_start: PositiveLine | None = None
    line_end: PositiveLine | None = None

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)

    @model_validator(mode="after")
    def validate_lines(self) -> TraceProvenance:
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("provenance lines must be paired")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("provenance line range is invalid")
        return self


class TraceInfluence(StrictModel):
    finding_indices: Annotated[list[FindingIndex], Field(max_length=MAX_FINDINGS_PER_TASK)] = []
    report_sections: Annotated[
        list[Literal["findings", "tests_run", "limitations"]], Field(max_length=3)
    ] = []


class ToolTraceEntry(StrictModel):
    trace_id: Annotated[str, Field(pattern=r"^tool-[0-9]{3}$")]
    order: Annotated[int, Field(ge=1, strict=True)]
    tool: ToolName
    status: ToolStatus
    arguments: ToolArguments
    result: ToolResult
    latency_ms: Annotated[int, Field(ge=0, strict=True)]
    influence: TraceInfluence = TraceInfluence()
    provenance: Annotated[list[TraceProvenance], Field(max_length=MAX_PROVENANCE)] = []

    @model_validator(mode="after")
    def validate_discriminators(self) -> ToolTraceEntry:
        expected = self.tool.value
        if self.arguments.kind != expected:
            raise ValueError("tool and argument discriminator disagree")
        failed = isinstance(self.result, ToolFailureResult)
        if failed != (self.status == ToolStatus.FAILED):
            raise ValueError("tool status and result disagree")
        if not failed and not self.result.kind.startswith(expected):
            raise ValueError("tool and result discriminator disagree")
        if failed:
            if self.provenance or self.influence != TraceInfluence():
                raise ValueError("failed traces cannot claim provenance or influence")
            return self
        if isinstance(self.arguments, ReadFileArguments) and isinstance(
            self.result, ReadFileResult
        ):
            if self.result.path != self.arguments.path or self.provenance != [
                TraceProvenance(path=self.arguments.path)
            ]:
                raise ValueError("read trace arguments, result, and provenance disagree")
        elif isinstance(self.arguments, SearchCodeArguments) and isinstance(
            self.result, SearchCodeResult
        ):
            expected_provenance = [
                TraceProvenance(path=match.path, line_start=match.line, line_end=match.line)
                for match in self.result.matches
            ]
            if self.result.query != self.arguments.query or self.provenance != expected_provenance:
                raise ValueError("search trace arguments, result, and provenance disagree")
        elif isinstance(self.arguments, RunTestsArguments) and isinstance(
            self.result, RunTestsResult
        ):
            if self.result.profile != self.arguments.profile:
                raise ValueError("test trace profile disagrees with arguments")
        else:
            raise ValueError("tool trace argument and result types disagree")

        return self


class CandidateVerification(StrictModel):
    candidate_index: FindingIndex
    accepted: bool
    code: Literal["accepted", "unknown_file", "unchanged_line", "invalid_evidence"]
    evidence_trace_ids: Annotated[list[str], Field(max_length=MAX_TRACE_ENTRIES)] = []


class WorkflowTransition(StrictModel):
    order: Annotated[int, Field(ge=1, strict=True)]
    from_state: WorkflowState
    to_state: WorkflowState


class VerificationResult(StrictModel):
    profile: Literal["python-unittest-v1"]
    status: TestStatus
    summary: SafeText
    changed_locations: Annotated[list[ChangedLocation], Field(max_length=100)]


class ReviewContext(StrictModel):
    schema_version: Literal["1.0", "1.2"]
    task_id: str
    title: str
    issue_description: str
    diff_text: str
    tool_trace: list[ToolTraceEntry] = []

    @model_validator(mode="after")
    def validate_context_version(self) -> ReviewContext:
        if self.schema_version == "1.0" and self.tool_trace:
            raise ValueError("tool trace requires review context schema 1.2")
        return self

    @model_serializer(mode="wrap")
    def serialize_context(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        data = cast(dict[str, Any], handler(self))
        if self.schema_version == "1.0":
            data.pop("tool_trace", None)
        return data


class Finding(LineRangeModel):
    title: str
    severity: Severity
    category: Category
    file: str
    evidence: str
    reasoning: str
    recommendation: str
    confidence: Confidence

    evidence_trace_ids: Annotated[list[str], Field(max_length=MAX_TRACE_ENTRIES)] = []

    @field_validator("file")
    @classmethod
    def normalize_file(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class TestRun(StrictModel):
    command: str
    status: TestStatus
    profile: Literal["python-unittest-v1"] | None = None
    output_truncated: bool | None = None
    summary: str


class ReviewReport(StrictModel):
    schema_version: ReportSchemaVersion
    summary: str
    overall_assessment: OverallAssessment
    findings: Annotated[list[Finding], Field(max_length=MAX_FINDINGS_PER_TASK)]
    tests_run: list[TestRun]
    limitations: list[str]

    @model_serializer(mode="wrap")
    def serialize_report(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        data = cast(dict[str, Any], handler(self))
        if self.schema_version == "1.0":
            findings = data.get("findings")
            if isinstance(findings, list):
                for finding in findings:
                    if isinstance(finding, dict):
                        finding.pop("evidence_trace_ids", None)
            tests_run = data.get("tests_run")
            if isinstance(tests_run, list):
                for test_run in tests_run:
                    test_run.pop("profile", None)
                    test_run.pop("output_truncated", None)
        return data


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
    schema_version: Literal["1.0", "1.1", "1.2"]
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
    final_state: WorkflowState | None = None
    state_transitions: list[WorkflowTransition] = []
    tool_trace: list[ToolTraceEntry] = []
    verification: VerificationResult | None = None

    declared_tool_plan: ToolPlan | None = None
    provider_status: Literal["not_called", "succeeded"] | None = None
    candidate_review: ReviewReport | None = None
    candidate_verifications: Annotated[
        list[CandidateVerification], Field(max_length=MAX_FINDINGS_PER_TASK)
    ] = []
    planner_version: ShortText | None = None
    agent_version: ShortText | None = None
    failure_code: ShortText | None = None
    failure_message: ShortText | None = None

    @model_validator(mode="after")
    def validate_provider_metadata(self) -> RunArtifact:
        if self.schema_version == "1.0":
            return self
        if self.provider not in ARTIFACT_PROVIDERS:
            raise ValueError("1.1 artifacts require a supported provider")
        if self.schema_version == "1.2" and self.provider_status == "not_called":
            metadata = (
                self.requested_model,
                self.response_model,
                self.prompt_version,
                self.structured_output_mode,
                self.latency_ms,
                self.usage,
            )
            if any(value is not None for value in metadata):
                raise ValueError("provider metadata must be absent when provider was not called")
        if self.schema_version == "1.2" and self.final_state == WorkflowState.FAILED:
            return self._validate_tool_metadata()
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
            return self._validate_tool_metadata()
        if (
            self.provider not in LIVE_PROVIDERS
            or self.requested_model is None
            or not is_valid_model_identifier(self.requested_model)
            or (
                self.response_model is not None
                and not is_valid_model_identifier(self.response_model)
            )
            or self.prompt_version
            != (TOOL_PROMPT_VERSION if self.schema_version == "1.2" else PROMPT_VERSION)
            or self.structured_output_mode != LIVE_STRUCTURED_OUTPUT_MODE
            or self.latency_ms is None
            or not isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("live 1.1 artifacts require complete safe provider metadata")
        return self._validate_tool_metadata()

    def _validate_tool_metadata(self) -> RunArtifact:
        if self.schema_version != "1.2":
            if (
                self.final_state is not None
                or self.state_transitions
                or self.tool_trace
                or self.verification is not None
            ):
                raise ValueError("tool metadata requires artifact schema 1.2")
            return self
        if self.final_state not in {WorkflowState.COMPLETE, WorkflowState.FAILED}:
            raise ValueError("1.2 artifacts must reach a terminal state")
        if self.final_state == WorkflowState.FAILED:
            if (
                self.provider_status != "not_called"
                or self.candidate_review is not None
                or self.candidate_verifications
            ):
                raise ValueError("failed artifacts must stop before provider and candidates")
        elif (
            self.provider_status != "succeeded"
            or self.candidate_review is None
            or self.candidate_review.schema_version != "1.2"
            or self.candidate_review.tests_run
        ):
            raise ValueError("complete artifacts require the successful candidate review")

        success_states = [
            WorkflowState.INTAKE,
            WorkflowState.CONTEXT,
            WorkflowState.ANALYSIS,
            WorkflowState.VERIFICATION,
            WorkflowState.REPORT,
            WorkflowState.EVALUATION,
            WorkflowState.COMPLETE,
        ]
        observed = [self.state_transitions[0].from_state] if self.state_transitions else []
        observed.extend(transition.to_state for transition in self.state_transitions)
        orders = [transition.order for transition in self.state_transitions]
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError("state transition orders must be contiguous")
        if self.final_state == WorkflowState.COMPLETE:
            if observed != success_states:
                raise ValueError("successful artifacts require the canonical state path")
        elif (
            len(observed) < 2
            or observed[-1] != WorkflowState.FAILED
            or observed[:-1] != success_states[: len(observed) - 1]
        ):
            raise ValueError("failed artifacts require one legal transition to failed")

        trace_orders = [entry.order for entry in self.tool_trace]
        if trace_orders != list(range(1, len(trace_orders) + 1)):
            raise ValueError("tool trace orders must be contiguous")
        if [entry.trace_id for entry in self.tool_trace] != [
            f"tool-{index:03d}" for index in range(1, len(self.tool_trace) + 1)
        ]:
            raise ValueError("tool trace IDs must be deterministic")
        if self.declared_tool_plan is None:
            raise ValueError("1.2 artifacts require the declared tool plan")
        planned = [ToolName.READ_FILE] * len(self.declared_tool_plan.read_files)
        planned += [ToolName.SEARCH_CODE] * len(self.declared_tool_plan.searches)
        if self.declared_tool_plan.test_profile is not None:
            planned.append(ToolName.RUN_TESTS)
        actual = [entry.tool for entry in self.tool_trace]
        planned_arguments: list[ToolArguments] = [
            ReadFileArguments(kind="read_file", path=path)
            for path in self.declared_tool_plan.read_files
        ]
        planned_arguments.extend(
            SearchCodeArguments(kind="search_code", query=query)
            for query in self.declared_tool_plan.searches
        )
        if self.declared_tool_plan.test_profile is not None:
            planned_arguments.append(
                RunTestsArguments(kind="run_tests", profile=self.declared_tool_plan.test_profile)
            )
        actual_arguments = [entry.arguments for entry in self.tool_trace]
        if actual_arguments != planned_arguments[: len(actual_arguments)]:
            raise ValueError("tool trace arguments do not follow the declared plan")
        if actual != planned[: len(actual)]:
            raise ValueError("tool trace does not follow the declared plan")
        if self.final_state == WorkflowState.COMPLETE and actual != planned:
            raise ValueError("complete artifact tool trace is incomplete")
        if (
            self.final_state == WorkflowState.FAILED
            and actual
            and actual != planned
            and self.tool_trace[-1].status != ToolStatus.FAILED
        ):
            raise ValueError("partial failed artifact requires a terminal failed tool trace")

        completed_tests = any(isinstance(entry.result, RunTestsResult) for entry in self.tool_trace)
        if completed_tests != (self.verification is not None):
            raise ValueError("verification metadata must correspond to completed run_tests")
        if self.verification is not None:
            if not self.verification.changed_locations or len(self.review.tests_run) != 1:
                raise ValueError("test verification requires changed locations and one test run")
            test_trace = next(
                entry.result
                for entry in self.tool_trace
                if isinstance(entry.result, RunTestsResult)
            )
            test_run = self.review.tests_run[0]
            expected_locations = [
                ChangedLocation(
                    path=item.path,
                    line_start=cast(int, item.line_start),
                    line_end=cast(int, item.line_end),
                )
                for item in next(
                    entry.provenance
                    for entry in self.tool_trace
                    if isinstance(entry.result, RunTestsResult)
                )
            ]
            if (
                test_run.command != test_trace.command
                or test_run.profile != test_trace.profile
                or test_run.status != self.verification.status
                or test_run.status != test_trace.status
                or test_run.summary != self.verification.summary
                or test_run.summary != test_trace.summary
                or test_run.output_truncated != test_trace.output_truncated
                or self.verification.changed_locations != expected_locations
            ):
                raise ValueError("canonical test evidence must agree exactly")
        elif self.review.tests_run:
            raise ValueError("report tests require trusted verification")

        if self.final_state == WorkflowState.COMPLETE:
            assert self.candidate_review is not None
            candidates = self.candidate_review.findings
            if [record.candidate_index for record in self.candidate_verifications] != list(
                range(len(candidates))
            ):
                raise ValueError("candidate verification must cover candidates in order")
            accepted_candidates: list[Finding] = []
            expected_influence: dict[str, list[int]] = {}
            trace_by_id = {entry.trace_id: entry for entry in self.tool_trace}
            for candidate, record in zip(candidates, self.candidate_verifications, strict=True):
                if record.accepted != (record.code == "accepted"):
                    raise ValueError("candidate acceptance and code disagree")
                if record.evidence_trace_ids != candidate.evidence_trace_ids:
                    raise ValueError("candidate evidence and verification disagree")
                if not record.accepted:
                    continue
                final_index = len(accepted_candidates)
                accepted_candidates.append(candidate)
                for trace_id in candidate.evidence_trace_ids:
                    entry = trace_by_id.get(trace_id)
                    if entry is None or entry.status != ToolStatus.SUCCEEDED:
                        raise ValueError("accepted candidate cites invalid trace evidence")
                    compatible = any(
                        item.path == candidate.file
                        and (
                            item.line_start is None
                            or (
                                item.line_end is not None
                                and item.line_start <= candidate.line_end
                                and candidate.line_start <= item.line_end
                            )
                        )
                        for item in entry.provenance
                    )
                    if not compatible:
                        raise ValueError("accepted candidate cites incompatible provenance")
                    expected_influence.setdefault(trace_id, []).append(final_index)
            if self.review.findings != accepted_candidates:
                raise ValueError("final findings must equal accepted candidates")
            for entry in self.tool_trace:
                sections: list[Literal["findings", "tests_run", "limitations"]] = []
                if entry.trace_id in expected_influence:
                    sections.append("findings")
                if isinstance(entry.result, RunTestsResult):
                    sections.append("tests_run")
                expected = TraceInfluence(
                    finding_indices=expected_influence.get(entry.trace_id, []),
                    report_sections=sections,
                )
                if entry.influence != expected:
                    raise ValueError("trace influence must be derived exactly")
        if self.final_state == WorkflowState.FAILED:
            if not self.failure_code or not self.failure_message:
                raise ValueError("failed artifact requires safe failure metadata")
        elif self.failure_code or self.failure_message:
            raise ValueError("complete artifact cannot contain failure metadata")
        if not self.planner_version or not self.agent_version:
            raise ValueError("1.2 artifacts require planner and agent versions")
        return self

    @model_serializer(mode="wrap")
    def serialize_artifact(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        data = cast(dict[str, Any], handler(self))
        if self.schema_version != "1.2":
            for field in (
                "final_state",
                "state_transitions",
                "tool_trace",
                "verification",
                "provider_status",
                "candidate_review",
                "declared_tool_plan",
                "candidate_verifications",
                "planner_version",
                "agent_version",
                "failure_code",
                "failure_message",
            ):
                data.pop(field, None)
        return data
