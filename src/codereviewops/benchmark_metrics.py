"""Deterministic benchmark metrics and strict quality gates."""

from __future__ import annotations

from collections import Counter, defaultdict

from codereviewops.benchmark_models import (
    AggregateMetricsV1,
    TaskMetricsV1,
    ThresholdProfileV1,
)
from codereviewops.models import BenchmarkTask, RunArtifact, ToolStatus, WorkflowState


def _planned_calls(task: BenchmarkTask) -> list[tuple[str, str]]:
    plan = task.tool_plan
    if plan is None:
        return []
    calls = [("read_file", value) for value in plan.read_files]
    calls.extend(("search_code", value) for value in plan.searches)
    if plan.test_profile is not None:
        calls.append(("run_tests", plan.test_profile))
    return calls


def _actual_calls(artifact: RunArtifact) -> list[tuple[str, str, ToolStatus]]:
    calls: list[tuple[str, str, ToolStatus]] = []
    for entry in artifact.tool_trace:
        arguments = entry.arguments
        value = (
            getattr(arguments, "path", None)
            or getattr(arguments, "query", None)
            or getattr(arguments, "profile", None)
            or ""
        )
        calls.append((entry.tool.value, value, entry.status))
    return calls


def task_metrics(task: BenchmarkTask, artifact: RunArtifact) -> TaskMetricsV1:
    evaluation = artifact.evaluation
    matched = evaluation.matched
    severity_correct = sum(
        task.expected_findings[item.expected_index].severity
        == artifact.review.findings[item.actual_index].severity
        for item in matched
        if task.expected_findings[item.expected_index].severity is not None
    )
    severity_total = sum(
        task.expected_findings[item.expected_index].severity is not None for item in matched
    )
    planned = _planned_calls(task)
    actual = _actual_calls(artifact)
    paired = list(zip(planned, actual, strict=False))
    positional = sum(
        expected == observed[:2] and observed[2] == ToolStatus.SUCCEEDED
        for expected, observed in paired
    )
    mismatched = sum(expected != observed[:2] for expected, observed in paired)
    succeeded = sum(call[2] == ToolStatus.SUCCEEDED for call in actual)
    failed = len(actual) - succeeded
    tool_total = max(len(planned), len(actual))
    completed = artifact.final_state in {None, WorkflowState.COMPLETE}
    usage = artifact.usage
    return TaskMetricsV1(
        task_id=task.task_id,
        completed=completed,
        task_success=completed and evaluation.task_success,
        expected_count=len(task.expected_findings),
        actual_count=len(artifact.review.findings),
        true_positive_count=len(matched),
        missed_count=len(evaluation.missed_expected_indices),
        hallucinated_count=len(evaluation.hallucinated_actual_indices),
        prohibited_count=len(evaluation.prohibited_hits),
        precision=evaluation.precision,
        recall=evaluation.recall,
        severity_correct=severity_correct,
        severity_total=severity_total,
        tool_plan_correct=positional,
        tool_plan_total=tool_total,
        tool_planned=len(planned),
        tool_observed=len(actual),
        tool_succeeded=succeeded,
        tool_failed=failed,
        tool_missing=max(len(planned) - len(actual), 0),
        tool_unexpected=max(len(actual) - len(planned), 0),
        tool_mismatched=mismatched,
        negative_false_positive=not task.expected_findings and bool(artifact.review.findings),
        category_expected={
            category.value: sum(finding.category == category for finding in task.expected_findings)
            for category in {finding.category for finding in task.expected_findings}
        },
        category_matched={
            category.value: sum(
                task.expected_findings[item.expected_index].category == category for item in matched
            )
            for category in {finding.category for finding in task.expected_findings}
        },
        rejection_codes=[
            record.code for record in artifact.candidate_verifications if not record.accepted
        ],
        test_statuses=[test.status.value for test in artifact.review.tests_run],
        semantic_trace_fingerprint=artifact.semantic_trace_fingerprint,
        latency_ms=artifact.latency_ms,
        input_tokens=usage.prompt_tokens if usage else None,
        output_tokens=usage.completion_tokens if usage else None,
    )


def aggregate_metrics(
    rows: list[tuple[BenchmarkTask, TaskMetricsV1]],
) -> AggregateMetricsV1:
    metrics = [metric for _, metric in rows]
    task_count = len(metrics)
    expected = sum(metric.expected_count for metric in metrics)
    actual = sum(metric.actual_count for metric in metrics)
    true_positive = sum(metric.true_positive_count for metric in metrics)
    missed = sum(metric.missed_count for metric in metrics)
    hallucinated = sum(metric.hallucinated_count for metric in metrics)
    category_expected: dict[str, int] = defaultdict(int)
    category_matched: dict[str, int] = defaultdict(int)
    difficulty_total: dict[str, int] = defaultdict(int)
    difficulty_success: dict[str, int] = defaultdict(int)
    for task, metric in rows:
        difficulty_total[task.difficulty.value] += 1
        difficulty_success[task.difficulty.value] += int(metric.task_success)
        for category, count in metric.category_expected.items():
            category_expected[category] += count
        for category, count in metric.category_matched.items():
            category_matched[category] += count
    severity_total = sum(metric.severity_total for metric in metrics)
    tool_total = sum(metric.tool_plan_total for metric in metrics)
    return AggregateMetricsV1(
        task_count=task_count,
        completed_count=sum(metric.completed for metric in metrics),
        successful_count=sum(metric.task_success for metric in metrics),
        expected_count=expected,
        actual_count=actual,
        true_positive_count=true_positive,
        missed_count=missed,
        hallucinated_count=hallucinated,
        prohibited_count=sum(metric.prohibited_count for metric in metrics),
        negative_false_positives=sum(metric.negative_false_positive for metric in metrics),
        completion_rate=sum(metric.completed for metric in metrics) / task_count,
        tool_planned=sum(metric.tool_planned for metric in metrics),
        tool_observed=sum(metric.tool_observed for metric in metrics),
        tool_succeeded=sum(metric.tool_succeeded for metric in metrics),
        tool_failed=sum(metric.tool_failed for metric in metrics),
        tool_missing=sum(metric.tool_missing for metric in metrics),
        tool_unexpected=sum(metric.tool_unexpected for metric in metrics),
        tool_mismatched=sum(metric.tool_mismatched for metric in metrics),
        rejection_codes=dict(
            Counter(code for metric in metrics for code in metric.rejection_codes)
        ),
        test_statuses=dict(
            Counter(status for metric in metrics for status in metric.test_statuses)
        ),
        task_success_rate=sum(metric.task_success for metric in metrics) / task_count,
        micro_precision=true_positive / actual if actual else (1.0 if expected == 0 else 0.0),
        micro_recall=true_positive / expected if expected else 1.0,
        macro_precision=sum(metric.precision for metric in metrics) / task_count,
        macro_recall=sum(metric.recall for metric in metrics) / task_count,
        hallucination_rate=hallucinated / actual if actual else 0.0,
        missed_rate=missed / expected if expected else 0.0,
        category_recall={
            category: category_matched[category] / count
            for category, count in sorted(category_expected.items())
        },
        severity_accuracy=(
            sum(metric.severity_correct for metric in metrics) / severity_total
            if severity_total
            else 1.0
        ),
        difficulty_success={
            difficulty: difficulty_success[difficulty] / count
            for difficulty, count in sorted(difficulty_total.items())
        },
        tool_plan_accuracy=(
            sum(metric.tool_plan_correct for metric in metrics) / tool_total if tool_total else 1.0
        ),
        latency_ms_total=sum(metric.latency_ms or 0.0 for metric in metrics),
        input_tokens_total=sum(metric.input_tokens or 0 for metric in metrics),
        output_tokens_total=sum(metric.output_tokens or 0 for metric in metrics),
    )


def gate_failures(metrics: AggregateMetricsV1, thresholds: ThresholdProfileV1) -> list[str]:
    failures: list[str] = []
    minimums = {
        "completion_rate": thresholds.completion_rate,
        "task_success_rate": thresholds.task_success_rate,
        "micro_precision": thresholds.micro_precision,
        "micro_recall": thresholds.micro_recall,
        "macro_precision": thresholds.macro_precision,
        "macro_recall": thresholds.macro_recall,
        "severity_accuracy": thresholds.severity_accuracy,
        "tool_plan_accuracy": thresholds.tool_plan_accuracy,
    }
    for name, threshold in minimums.items():
        if getattr(metrics, name) < threshold:
            failures.append(name)
    if any(value < thresholds.category_recall for value in metrics.category_recall.values()):
        failures.append("category_recall")
    maximums = {
        "hallucination_rate": thresholds.hallucination_rate,
        "missed_rate": thresholds.missed_rate,
        "negative_false_positives": thresholds.negative_false_positives,
        "prohibited_count": thresholds.prohibited_hits,
        "tool_failed": thresholds.tool_failed,
        "tool_missing": thresholds.tool_missing,
        "tool_unexpected": thresholds.tool_unexpected,
        "tool_mismatched": thresholds.tool_mismatched,
    }
    for name, threshold in maximums.items():
        if getattr(metrics, name) > threshold:
            failures.append(name)
    return failures
