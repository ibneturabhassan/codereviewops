"""Transition-enforced review workflow with bounded evidence tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from codereviewops.docker_runner import DockerTestRunner
from codereviewops.evaluation import evaluate_review
from codereviewops.io import InputError, LoadedTask, load_task
from codereviewops.models import (
    BenchmarkTask,
    CandidateVerification,
    EvaluationResult,
    Finding,
    OverallAssessment,
    ReviewContext,
    ReviewReport,
    RunArtifact,
    RunTestsResult,
    TestRun,
    ToolStatus,
    ToolTraceEntry,
    TraceInfluence,
    VerificationResult,
    WorkflowState,
    WorkflowTransition,
)
from codereviewops.providers import GroqProvider, MistralProvider, ReplayProvider, ReviewProvider
from codereviewops.tools import (
    TestExecution,
    TestRunner,
    ToolError,
    ToolExecutionError,
    ToolRun,
    Workspace,
    changed_locations,
    execute_tool_plan,
    parse_unified_diff,
)

PLANNER_VERSION = "manifest-v1"
AGENT_VERSION = "tool-agent-v1"


class WorkflowMachine:
    """Small state machine that rejects skipped or repeated phases."""

    _next: ClassVar[dict[WorkflowState, WorkflowState]] = {
        WorkflowState.INTAKE: WorkflowState.CONTEXT,
        WorkflowState.CONTEXT: WorkflowState.ANALYSIS,
        WorkflowState.ANALYSIS: WorkflowState.VERIFICATION,
        WorkflowState.VERIFICATION: WorkflowState.REPORT,
        WorkflowState.REPORT: WorkflowState.EVALUATION,
        WorkflowState.EVALUATION: WorkflowState.COMPLETE,
    }

    def __init__(self) -> None:
        self.state = WorkflowState.INTAKE
        self.transitions: list[WorkflowTransition] = []

    def advance(self, target: WorkflowState) -> None:
        if self.state in {WorkflowState.COMPLETE, WorkflowState.FAILED}:
            raise RuntimeError("workflow is already terminal")
        if target != self._next[self.state]:
            raise RuntimeError("illegal workflow transition")
        self._record(target)

    def fail(self) -> None:
        if self.state in {WorkflowState.COMPLETE, WorkflowState.FAILED}:
            raise RuntimeError("workflow is already terminal")
        self._record(WorkflowState.FAILED)

    def _record(self, target: WorkflowState) -> None:
        self.transitions.append(
            WorkflowTransition(
                order=len(self.transitions) + 1,
                from_state=self.state,
                to_state=target,
            )
        )
        self.state = target


class _NoTests:
    def run(self, workspace: Path, profile: str) -> TestExecution:
        del workspace, profile
        raise ToolError("docker_unavailable", "no isolated test runner was configured")


def _empty_evaluation() -> EvaluationResult:
    return EvaluationResult(
        schema_version="1.0",
        matched=[],
        missed_expected_indices=[],
        hallucinated_actual_indices=[],
        prohibited_hits=[],
        precision=0,
        recall=0,
        hallucination_rate=0,
        task_success=False,
    )


def _canonical_test_run(tool_run: ToolRun) -> list[TestRun]:
    if tool_run.verification is None:
        return []
    result = next(
        entry.result for entry in tool_run.trace if isinstance(entry.result, RunTestsResult)
    )
    return [
        TestRun(
            command=result.command,
            profile=result.profile,
            status=result.status,
            summary=result.summary,
            output_truncated=result.output_truncated,
        )
    ]


def _failure_artifact(
    loaded: LoadedTask,
    provider_name: str,
    machine: WorkflowMachine,
    trace: list[ToolTraceEntry],
    verification: VerificationResult | None,
    code: str,
    message: str,
) -> RunArtifact:
    machine.fail()
    tool_run = ToolRun(trace=trace, verification=verification)
    report = ReviewReport(
        schema_version="1.2",
        summary="The bounded review workflow did not complete.",
        overall_assessment=OverallAssessment.UNCERTAIN,
        findings=[],
        tests_run=_canonical_test_run(tool_run),
        limitations=["Review stopped after a safe workflow failure."],
    )
    return RunArtifact(
        schema_version="1.2",
        task_id=loaded.task.task_id,
        provider=provider_name,
        review=report,
        evaluation=_empty_evaluation(),
        structured_output_mode=None,
        latency_ms=None,
        final_state=WorkflowState.FAILED,
        state_transitions=machine.transitions,
        tool_trace=trace,
        verification=tool_run.verification,
        declared_tool_plan=loaded.task.tool_plan,
        provider_status="not_called",
        candidate_verifications=[],
        planner_version=PLANNER_VERSION,
        agent_version=AGENT_VERSION,
        failure_code=code,
        failure_message=message,
    )


def _diff_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {}
    for location in changed_locations(diff_text):
        ranges.setdefault(location.path, []).append((location.line_start, location.line_end))
    return ranges


def _compatible_trace(finding: Finding, entry: ToolTraceEntry) -> bool:
    if entry.status != ToolStatus.SUCCEEDED:
        return False
    for provenance in entry.provenance:
        if provenance.path != finding.file:
            continue
        if provenance.line_start is None:
            return True
        if (
            provenance.line_end is not None
            and provenance.line_start <= finding.line_end
            and finding.line_start <= provenance.line_end
        ):
            return True
    return False


def _verify_candidates(
    report: ReviewReport,
    trace: list[ToolTraceEntry],
    diff_text: str,
) -> tuple[list[Finding], list[CandidateVerification], list[ToolTraceEntry]]:
    ranges = _diff_ranges(diff_text)
    trace_by_id = {entry.trace_id: entry for entry in trace}
    accepted: list[Finding] = []
    records: list[CandidateVerification] = []
    influence: dict[str, list[int]] = {}
    for candidate_index, finding in enumerate(report.findings):
        code = "accepted"
        file_ranges = ranges.get(finding.file)
        if not file_ranges:
            code = "unknown_file"
        elif not any(
            start <= finding.line_end and finding.line_start <= end for start, end in file_ranges
        ):
            code = "unchanged_line"
        elif (
            not finding.evidence_trace_ids
            or len(set(finding.evidence_trace_ids)) != len(finding.evidence_trace_ids)
            or any(
                trace_id not in trace_by_id or not _compatible_trace(finding, trace_by_id[trace_id])
                for trace_id in finding.evidence_trace_ids
            )
        ):
            code = "invalid_evidence"
        accepted_flag = code == "accepted"
        records.append(
            CandidateVerification.model_validate(
                {
                    "candidate_index": candidate_index,
                    "accepted": accepted_flag,
                    "code": code,
                    "evidence_trace_ids": finding.evidence_trace_ids,
                }
            )
        )
        if accepted_flag:
            final_index = len(accepted)
            accepted.append(finding)
            for trace_id in finding.evidence_trace_ids:
                influence.setdefault(trace_id, []).append(final_index)

    updated: list[ToolTraceEntry] = []
    for entry in trace:
        sections: list[str] = []
        if entry.trace_id in influence:
            sections.append("findings")
        if isinstance(entry.result, RunTestsResult):
            sections.append("tests_run")
        updated.append(
            entry.model_copy(
                update={
                    "influence": TraceInfluence.model_validate(
                        {
                            "finding_indices": influence.get(entry.trace_id, []),
                            "report_sections": sections,
                        }
                    )
                }
            )
        )
    return accepted, records, updated


def run_loaded_task(
    loaded: LoadedTask,
    provider: ReviewProvider,
    provider_name: str = "replay",
    test_runner: TestRunner | None = None,
) -> tuple[BenchmarkTask, RunArtifact]:
    BenchmarkTask.model_validate(loaded.task.model_dump(mode="python"))
    machine = WorkflowMachine()
    trace: list[ToolTraceEntry] = []
    tool_run = ToolRun(trace=[], verification=None)
    try:
        diff_text = loaded.diff_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        if loaded.task.schema_version == "1.0":
            raise InputError("diff is missing or inaccessible") from None
        return loaded.task, _failure_artifact(
            loaded, provider_name, machine, trace, None, "read_failed", "diff could not be read"
        )

    if loaded.task.schema_version == "1.0":
        context = ReviewContext(
            schema_version="1.0",
            task_id=loaded.task.task_id,
            title=loaded.task.title,
            issue_description=loaded.task.issue_description,
            diff_text=diff_text,
        )
        provider_result = provider.review(context)
        review = provider_result.report
        evaluation = evaluate_review(
            loaded.task.expected_findings, loaded.task.must_not_find, review
        )
        return loaded.task, RunArtifact(
            schema_version="1.1",
            task_id=loaded.task.task_id,
            provider=provider_name,
            review=review,
            evaluation=evaluation,
            requested_model=provider_result.requested_model,
            response_model=provider_result.response_model,
            prompt_version=provider_result.prompt_version,
            structured_output_mode=provider_result.structured_output_mode,
            latency_ms=provider_result.latency_ms,
            usage=provider_result.usage,
        )

    machine.advance(WorkflowState.CONTEXT)
    try:
        added_lines = parse_unified_diff(diff_text)
    except ToolError as exc:
        return loaded.task, _failure_artifact(
            loaded, provider_name, machine, trace, None, exc.code, exc.message
        )
    if (
        loaded.task.tool_plan is not None
        and loaded.task.tool_plan.test_profile is not None
        and not any(added_lines.values())
    ):
        return loaded.task, _failure_artifact(
            loaded,
            provider_name,
            machine,
            trace,
            None,
            "diff_no_added_lines",
            "tool verification requires at least one added line",
        )
    if loaded.task.tool_plan is not None:
        if loaded.workspace_path is None or loaded.benchmark_root is None:
            return loaded.task, _failure_artifact(
                loaded,
                provider_name,
                machine,
                trace,
                None,
                "invalid_path",
                "tool workspace is unavailable",
            )
        runner = test_runner
        if runner is None:
            runner = DockerTestRunner() if loaded.task.tool_plan.test_profile else _NoTests()
        try:
            workspace = Workspace(loaded.workspace_path, loaded.benchmark_root)
            tool_run = execute_tool_plan(workspace, loaded.task.tool_plan, runner, diff_text)
            trace = tool_run.trace
        except ToolExecutionError as exc:
            return loaded.task, _failure_artifact(
                loaded, provider_name, machine, exc.trace, None, exc.code, exc.message
            )
        except ToolError as exc:
            return loaded.task, _failure_artifact(
                loaded, provider_name, machine, trace, None, exc.code, exc.message
            )

    machine.advance(WorkflowState.ANALYSIS)
    context = ReviewContext(
        schema_version="1.2",
        task_id=loaded.task.task_id,
        title=loaded.task.title,
        issue_description=loaded.task.issue_description,
        diff_text=diff_text,
        tool_trace=trace,
    )
    provider_result = provider.review(context)
    candidate_report = ReviewReport.model_validate(
        {
            **provider_result.report.model_dump(mode="python"),
            "schema_version": "1.2",
        }
    )
    if candidate_report.tests_run:
        raise ToolError("docker_infrastructure", "provider returned untrusted test claims")

    machine.advance(WorkflowState.VERIFICATION)
    candidates, records, trace = _verify_candidates(candidate_report, trace, diff_text)
    machine.advance(WorkflowState.REPORT)
    review = candidate_report.model_copy(
        update={
            "schema_version": "1.2",
            "findings": candidates,
            "tests_run": _canonical_test_run(tool_run),
        },
        deep=True,
    )
    machine.advance(WorkflowState.EVALUATION)
    evaluation = evaluate_review(loaded.task.expected_findings, loaded.task.must_not_find, review)
    machine.advance(WorkflowState.COMPLETE)
    return loaded.task, RunArtifact(
        schema_version="1.2",
        task_id=loaded.task.task_id,
        provider=provider_name,
        review=review,
        evaluation=evaluation,
        requested_model=provider_result.requested_model,
        response_model=provider_result.response_model,
        prompt_version=provider_result.prompt_version,
        structured_output_mode=provider_result.structured_output_mode,
        latency_ms=provider_result.latency_ms,
        usage=provider_result.usage,
        final_state=WorkflowState.COMPLETE,
        state_transitions=machine.transitions,
        tool_trace=trace,
        verification=tool_run.verification,
        provider_status="succeeded",
        candidate_review=candidate_report,
        declared_tool_plan=loaded.task.tool_plan,
        candidate_verifications=records,
        planner_version=PLANNER_VERSION,
        agent_version=AGENT_VERSION,
    )


def run_task(
    task_path: Path, provider_name: str, model: str | None = None
) -> tuple[BenchmarkTask, RunArtifact]:
    if provider_name == "replay":
        if model is not None:
            raise InputError("--model is forbidden for the replay provider")
        loaded = load_task(task_path)
        provider: ReviewProvider = ReplayProvider(loaded.replay_path)
    elif provider_name in {"groq", "mistral"}:
        if model is None:
            raise InputError(f"--model is required for the {provider_name} provider")
        loaded = load_task(task_path)
        provider = (
            GroqProvider(model=model, api_key=os.environ.get("GROQ_API_KEY", ""))
            if provider_name == "groq"
            else MistralProvider(model=model, api_key=os.environ.get("MISTRAL_API_KEY", ""))
        )
    else:
        raise InputError("unsupported provider; expected replay, groq, or mistral")
    return run_loaded_task(loaded, provider, provider_name)
