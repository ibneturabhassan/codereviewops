from __future__ import annotations

from pathlib import Path

import pytest

from codereviewops.docker_runner import DockerTestRunner
from codereviewops.io import load_task
from codereviewops.models import TestStatus as ReviewTestStatus
from codereviewops.models import ToolName, WorkflowState
from codereviewops.providers import ReplayProvider
from codereviewops.workflow import run_loaded_task

pytestmark = pytest.mark.docker


def test_full_tool_workflow_uses_pinned_runner() -> None:
    root = Path(__file__).parents[1]
    loaded = load_task(root / "benchmarks" / "tasks" / "python_tools_001.json")
    provider = ReplayProvider(loaded.replay_path)
    task, artifact = run_loaded_task(loaded, provider, test_runner=DockerTestRunner())
    assert task.task_id == "python_tools_001"
    assert artifact.final_state == WorkflowState.COMPLETE
    assert [entry.tool for entry in artifact.tool_trace] == [
        ToolName.READ_FILE,
        ToolName.READ_FILE,
        ToolName.SEARCH_CODE,
        ToolName.RUN_TESTS,
    ]
    assert artifact.verification is not None
    assert artifact.verification.status == ReviewTestStatus.FAILED
    assert len(artifact.review.tests_run) == 1
    assert artifact.review.tests_run[0].status == ReviewTestStatus.FAILED
    assert artifact.evaluation.task_success
