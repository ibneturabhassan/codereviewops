from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codereviewops.io import LoadedTask, load_task
from codereviewops.models import (
    BenchmarkTask,
    ReviewContext,
    ReviewReport,
    RunArtifact,
    ToolName,
    ToolTraceEntry,
    WorkflowState,
)
from codereviewops.models import (
    TestStatus as ReviewTestStatus,
)
from codereviewops.providers import ReplayProvider
from codereviewops.tools import TestExecution as ToolTestExecution
from codereviewops.workflow import run_loaded_task

ROOT = Path(__file__).parents[1]


class FakeRunner:
    def run(self, workspace: Path, profile: str) -> ToolTestExecution:
        assert workspace.name == "python_tools_001"
        assert profile == "python-unittest-v1"
        return ToolTestExecution(ReviewTestStatus.FAILED, "one deterministic test failure")


def _run_tool_task():
    loaded = load_task(TOOL_TASK)
    return run_loaded_task(loaded, ReplayProvider(loaded.replay_path), test_runner=FakeRunner())


TOOL_TASK = ROOT / "benchmarks" / "tasks" / "python_tools_001.json"


def _task_data(**overrides):
    data = {
        "schema_version": "1.1",
        "task_id": "task",
        "title": "Tool task",
        "issue_description": "Review.",
        "diff_path": "fixtures/change.diff",
        "expected_findings": [],
        "must_not_find": [],
        "difficulty": "low",
        "tags": [],
        "replay_response_path": "replays/review.json",
        "workspace_path": "workspace",
        "tool_plan": {"read_files": [], "searches": [], "test_profile": None},
    }
    data.update(overrides)
    return data


def test_benchmark_11_tool_schema_is_strict_and_bounded() -> None:
    task = BenchmarkTask.model_validate(_task_data())
    assert task.schema_version == "1.1"
    assert task.tool_plan is not None
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(_task_data(workspace_path=None))
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(
            _task_data(tool_plan={"read_files": ["a.py", "a.py"], "searches": []})
        )
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(
            _task_data(tool_plan={"read_files": ["a.py"] * 21, "searches": []})
        )
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(
            _task_data(tool_plan={"read_files": [], "searches": ["x"] * 11})
        )
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(_task_data(tool_plan={"read_files": [], "searches": ["x\n"]}))
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(_task_data(tool_plan={"test_profile": "shell"}))


def test_schema_10_rejects_tool_configuration() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(_task_data(schema_version="1.0"))
    with pytest.raises(ValidationError):
        BenchmarkTask.model_validate(_task_data(workspace_path=None, tool_plan=None))


def test_synthetic_tool_replay_is_deterministic() -> None:
    _, first = _run_tool_task()
    _, second = _run_tool_task()
    first_data = first.model_dump(mode="python")
    second_data = second.model_dump(mode="python")
    for entry in first_data["tool_trace"] + second_data["tool_trace"]:
        assert entry["latency_ms"] >= 0
        entry["latency_ms"] = 0
    assert first_data == second_data
    assert first.schema_version == "1.2"
    assert first.review.schema_version == "1.2"
    assert first.final_state == WorkflowState.COMPLETE
    assert [entry.tool for entry in first.tool_trace] == [
        ToolName.READ_FILE,
        ToolName.READ_FILE,
        ToolName.SEARCH_CODE,
        ToolName.RUN_TESTS,
    ]
    assert first.verification is not None
    assert first.evaluation.task_success


def test_provider_is_called_once_after_tool_context() -> None:
    loaded = load_task(TOOL_TASK)
    replay = ReplayProvider(loaded.replay_path)
    contexts: list[ReviewContext] = []

    class CountingProvider:
        def review(self, context: ReviewContext):
            contexts.append(context)

            return replay.review(context)

    _, artifact = run_loaded_task(loaded, CountingProvider(), test_runner=FakeRunner())
    assert len(contexts) == 1
    assert contexts[0].schema_version == "1.2"
    assert len(contexts[0].tool_trace) == 4
    assert artifact.state_transitions[0].from_state == WorkflowState.INTAKE
    assert artifact.state_transitions[-1].to_state == WorkflowState.COMPLETE


def test_schema_10_provider_report_is_normalized_before_tool_verification() -> None:
    loaded = load_task(TOOL_TASK)
    replay = ReplayProvider(loaded.replay_path)

    class LegacyProvider:
        def review(self, context: ReviewContext):
            result = replay.review(context)
            legacy = ReviewReport.model_validate(
                {
                    **result.report.model_dump(mode="python"),
                    "schema_version": "1.0",
                }
            )
            return result.model_copy(update={"report": legacy})

    _, artifact = run_loaded_task(loaded, LegacyProvider(), test_runner=FakeRunner())
    assert artifact.candidate_review is not None
    assert artifact.candidate_review.schema_version == "1.2"
    assert artifact.review.findings == []
    assert artifact.candidate_verifications
    assert all(record.code == "invalid_evidence" for record in artifact.candidate_verifications)


def test_tool_artifact_rejects_noncanonical_state_path() -> None:
    _, artifact = _run_tool_task()
    data = artifact.model_dump(mode="python")
    data["state_transitions"] = list(reversed(data["state_transitions"]))
    with pytest.raises(ValidationError):
        type(artifact).model_validate(data)


def test_trace_deserialization_rejects_wrong_discriminator_and_bounds() -> None:
    _, artifact = _run_tool_task()
    data = artifact.tool_trace[0].model_dump(mode="python")
    data["arguments"] = {"kind": "search_code", "query": "x"}
    with pytest.raises(ValidationError):
        ToolTraceEntry.model_validate(data)
    data = artifact.tool_trace[0].model_dump(mode="python")
    data["result"]["content"] = "x" * (64 * 1024 + 1)
    with pytest.raises(ValidationError):
        ToolTraceEntry.model_validate(data)


def test_candidate_verification_rejects_file_line_and_provenance() -> None:
    loaded = load_task(TOOL_TASK)
    replay = ReplayProvider(loaded.replay_path)

    class CandidateProvider:
        def review(self, context: ReviewContext):
            result = replay.review(context)
            base = result.report.findings[0]
            candidates = [
                base.model_copy(update={"file": "unknown.py"}),
                base.model_copy(update={"line_start": 1, "line_end": 1}),
                base.model_copy(update={"evidence_trace_ids": ["tool-002"]}),
            ]
            return result.model_copy(
                update={"report": result.report.model_copy(update={"findings": candidates})}
            )

    _, artifact = run_loaded_task(loaded, CandidateProvider(), test_runner=FakeRunner())
    assert artifact.review.findings == []
    assert [record.code for record in artifact.candidate_verifications] == [
        "unknown_file",
        "unchanged_line",
        "invalid_evidence",
    ]


def test_influence_and_canonical_test_evidence_are_deterministic() -> None:
    _, artifact = _run_tool_task()
    assert artifact.review.tests_run[0].command == (
        "python -B -m unittest discover -s tests -p test_*.py"
    )
    assert artifact.review.tests_run[0].status == ReviewTestStatus.FAILED
    assert artifact.tool_trace[0].influence.finding_indices == [0]
    assert artifact.tool_trace[2].influence.finding_indices == [0]
    assert artifact.tool_trace[3].influence.report_sections == ["tests_run"]


def test_infrastructure_failure_stops_before_provider_and_records_failed_state() -> None:
    loaded = load_task(TOOL_TASK)
    calls = 0

    class NeverProvider:
        def review(self, context: ReviewContext):
            nonlocal calls
            calls += 1
            raise AssertionError

    class InfraRunner:
        def run(self, workspace: Path, profile: str) -> ToolTestExecution:
            return ToolTestExecution(
                ReviewTestStatus.ERROR,
                "safe infrastructure failure",
                infrastructure_error=True,
                error_code="docker_infrastructure",
            )

    _, artifact = run_loaded_task(loaded, NeverProvider(), test_runner=InfraRunner())
    assert calls == 0
    assert artifact.final_state == WorkflowState.FAILED
    assert artifact.state_transitions[-1].to_state == WorkflowState.FAILED
    assert artifact.tool_trace[-1].result.kind == "tool_failure"
    assert artifact.failure_code == "docker_infrastructure"


@pytest.mark.parametrize("provider_name", ["replay", "groq"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_model", "request/model-1"),
        ("response_model", "response/model-1"),
        ("prompt_version", "tool-review-v1"),
        ("structured_output_mode", "json_schema"),
        ("latency_ms", 1.0),
        ("usage", {"prompt_tokens": 1}),
    ],
)
def test_not_called_artifacts_reject_all_provider_execution_metadata(
    provider_name: str, field: str, value: object
) -> None:
    loaded = load_task(TOOL_TASK)

    class NeverProvider:
        def review(self, context: ReviewContext):
            raise AssertionError

    class InfraRunner:
        def run(self, workspace: Path, profile: str) -> ToolTestExecution:
            return ToolTestExecution(
                ReviewTestStatus.ERROR,
                "safe infrastructure failure",
                infrastructure_error=True,
                error_code="docker_infrastructure",
            )

    _, artifact = run_loaded_task(
        loaded,
        NeverProvider(),
        provider_name=provider_name,
        test_runner=InfraRunner(),
    )
    assert artifact.provider == provider_name
    assert artifact.provider_status == "not_called"
    assert all(
        metadata is None
        for metadata in (
            artifact.requested_model,
            artifact.response_model,
            artifact.prompt_version,
            artifact.structured_output_mode,
            artifact.latency_ms,
            artifact.usage,
        )
    )
    mutated = artifact.model_dump(mode="python")
    mutated[field] = value
    with pytest.raises(ValidationError, match="provider metadata must be absent"):
        RunArtifact.model_validate(mutated)


class StaticDiffPath:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def read_text(self, encoding: str) -> str:
        assert encoding == "utf-8"
        if self.text is None:
            raise OSError("private host detail")
        return self.text


def test_invalid_schema_11_stops_before_diff_and_provider() -> None:
    valid_task = BenchmarkTask.model_validate(_task_data())
    invalid_task = valid_task.model_copy(update={"workspace_path": None, "tool_plan": None})
    reads = 0
    calls = 0

    class UnreadDiff:
        def read_text(self, encoding: str) -> str:
            nonlocal reads
            reads += 1
            raise AssertionError

    class NeverProvider:
        def review(self, context: ReviewContext):
            nonlocal calls
            calls += 1
            raise AssertionError

    loaded = LoadedTask(
        task=invalid_task,
        diff_path=UnreadDiff(),
        replay_path=Path("unused.json"),
    )
    with pytest.raises(ValidationError, match=r"schema 1\.1 tasks require"):
        run_loaded_task(loaded, NeverProvider())
    assert reads == 0
    assert calls == 0


def test_deletion_only_tool_diff_stops_before_tools_and_provider() -> None:
    from dataclasses import replace

    loaded = replace(
        load_task(TOOL_TASK),
        diff_path=StaticDiffPath("--- a/calculator.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-deleted\n"),
    )
    calls = 0

    class NeverProvider:
        def review(self, context: ReviewContext):
            nonlocal calls
            calls += 1
            raise AssertionError

    _, artifact = run_loaded_task(loaded, NeverProvider(), test_runner=FakeRunner())
    assert calls == 0
    assert artifact.failure_code == "diff_no_added_lines"
    assert artifact.tool_trace == []
    assert artifact.provider_status == "not_called"


def test_legacy_unreadable_diff_raises_safe_input_error_before_provider() -> None:
    from dataclasses import replace

    from codereviewops.io import InputError

    loaded = replace(load_task(ROOT / "benchmarks" / "tasks" / "http_retry_001.json"))
    loaded = replace(loaded, diff_path=StaticDiffPath(None))
    calls = 0

    class NeverProvider:
        def review(self, context: ReviewContext):
            nonlocal calls
            calls += 1
            raise AssertionError

    with pytest.raises(InputError, match="diff is missing or inaccessible") as captured:
        run_loaded_task(loaded, NeverProvider())
    assert calls == 0
    assert "private host detail" not in str(captured.value)
