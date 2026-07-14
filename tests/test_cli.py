from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import codereviewops.cli as cli
import codereviewops.tools as tool_module
import codereviewops.workflow as workflow_module
from codereviewops.cli import app
from codereviewops.models import RunArtifact, WorkflowState
from codereviewops.models import TestStatus as ReviewTestStatus
from codereviewops.tools import ToolError

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "tasks" / "http_retry_001.json"
TOOL_BENCHMARK = ROOT / "benchmarks" / "tasks" / "python_tools_001.json"
REPLAY_VARIANTS = ROOT / "tests" / "fixtures" / "replays"
runner = CliRunner()


def _invoke(task: Path, output: Path, *extra: str):
    return runner.invoke(
        app,
        [
            "review",
            "--task",
            str(task),
            "--provider",
            "replay",
            "--output-dir",
            str(output),
            *extra,
        ],
    )


def _variant_task(tmp_path: Path, replay_name: str) -> Path:
    task_dir = tmp_path / "task"
    (task_dir / "fixtures").mkdir(parents=True)
    (task_dir / "replays").mkdir()
    manifest_data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    manifest_data["diff_path"] = "fixtures/change.diff"
    manifest_data["replay_response_path"] = "replays/review.json"
    canonical_diff = BENCHMARK.parent / "fixtures" / "http_retry_001.diff"
    shutil.copyfile(canonical_diff, task_dir / "fixtures" / "change.diff")
    shutil.copyfile(REPLAY_VARIANTS / replay_name, task_dir / "replays" / "review.json")
    manifest = task_dir / "task.json"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    return manifest


def test_cli_success_writes_valid_deterministic_json_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "output"
    result = _invoke(BENCHMARK, output)
    assert result.exit_code == 0, result.output
    run_text = (output / "run.json").read_text(encoding="utf-8")
    report_text = (output / "report.md").read_text(encoding="utf-8")
    artifact = RunArtifact.model_validate_json(run_text)
    assert artifact.evaluation.task_success
    assert artifact.task_id == "http_retry_001"
    assert artifact.schema_version == "1.1"
    assert artifact.provider == "replay"
    assert artifact.requested_model is None
    assert artifact.response_model is None
    assert artifact.prompt_version is None
    assert artifact.structured_output_mode == "replay"
    assert artifact.latency_ms == 0
    assert "$" not in report_text
    for heading in (
        "## Metrics",
        "## Findings",
        "## Missed expected findings",
        "## Hallucinated findings",
        "## Prohibited phrase hits",
        "## Tests",
        "## Limitations",
    ):
        assert heading in report_text

    repeated = _invoke(BENCHMARK, output, "--overwrite")
    assert repeated.exit_code == 0, repeated.output
    assert (output / "run.json").read_text(encoding="utf-8") == run_text
    assert (output / "report.md").read_text(encoding="utf-8") == report_text
    assert list(output.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    ("variant", "expected_markdown"),
    [
        ("miss.json", "[1] missing_test"),
        ("hallucination.json", "[1] Unrelated security concern"),
        ("prohibited.json", "SQL injection"),
    ],
)
def test_cli_valid_failed_evaluations_exit_one(
    tmp_path: Path, variant: str, expected_markdown: str
) -> None:
    task = _variant_task(tmp_path, variant)
    output = tmp_path / "output"
    result = _invoke(task, output)
    assert result.exit_code == 1, result.output
    artifact = RunArtifact.model_validate_json((output / "run.json").read_text(encoding="utf-8"))
    assert not artifact.evaluation.task_success
    assert expected_markdown in (output / "report.md").read_text(encoding="utf-8")


def test_cli_refuses_overwrite_without_altering_either_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    run_path = output / "run.json"
    report_path = output / "report.md"
    run_path.write_text("private run sentinel", encoding="utf-8")
    report_path.write_text("private report sentinel", encoding="utf-8")

    result = _invoke(BENCHMARK, output)
    assert result.exit_code == 2
    assert "--overwrite" in result.output
    assert run_path.read_text(encoding="utf-8") == "private run sentinel"
    assert report_path.read_text(encoding="utf-8") == "private report sentinel"
    assert list(output.glob(".*.tmp")) == []


def test_cli_overwrite_replaces_both_files(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run.json").write_text("old", encoding="utf-8")
    (output / "report.md").write_text("old", encoding="utf-8")
    result = _invoke(BENCHMARK, output, "--overwrite")
    assert result.exit_code == 0, result.output
    RunArtifact.model_validate_json((output / "run.json").read_text(encoding="utf-8"))
    assert (output / "report.md").read_text(encoding="utf-8").startswith("# Review Report:")


def test_cli_malformed_json_exits_two_without_traceback(tmp_path: Path) -> None:
    task = tmp_path / "bad.json"
    task.write_text("{", encoding="utf-8")
    result = _invoke(task, tmp_path / "output")
    assert result.exit_code == 2
    assert "error:" in result.output
    assert "Traceback" not in result.output


def test_cli_schema_invalid_replay_exits_two(tmp_path: Path) -> None:
    task = _variant_task(tmp_path, "miss.json")
    replay = task.parent / "replays" / "review.json"
    replay.write_text('{"schema_version": "1.0"}', encoding="utf-8")
    result = _invoke(task, tmp_path / "output")
    assert result.exit_code == 2
    assert "code=invalid_replay" in result.output


def test_cli_rejects_non_replay_provider_with_exit_two(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "review",
            "--task",
            str(BENCHMARK),
            "--provider",
            "live",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 2
    assert "unsupported provider" in result.output


def test_cli_forbids_model_for_replay(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "review",
            "--task",
            str(BENCHMARK),
            "--provider",
            "replay",
            "--model",
            "unused",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 2
    assert "--model is forbidden" in result.output


@pytest.mark.parametrize("provider", ["groq", "mistral"])
def test_cli_requires_model_for_live_provider(tmp_path: Path, provider: str) -> None:
    result = runner.invoke(
        app,
        [
            "review",
            "--task",
            str(BENCHMARK),
            "--provider",
            provider,
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 2
    assert "--model is required" in result.output


@pytest.mark.parametrize(
    ("provider", "environment_key"),
    [("groq", "GROQ_API_KEY"), ("mistral", "MISTRAL_API_KEY")],
)
def test_cli_missing_key_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    environment_key: str,
) -> None:
    monkeypatch.delenv(environment_key, raising=False)
    result = runner.invoke(
        app,
        [
            "review",
            "--task",
            str(BENCHMARK),
            "--provider",
            provider,
            "--model",
            "valid-model",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 2
    assert "code=missing_api_key" in result.output
    assert "Traceback" not in result.output


def test_cli_invalid_model_does_not_leak_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "cli-secret-value"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    result = runner.invoke(
        app,
        [
            "review",
            "--task",
            str(BENCHMARK),
            "--provider",
            "groq",
            "--model",
            "bad model",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 2
    assert "code=invalid_model" in result.output
    assert secret not in result.output


def test_tools_check_reports_ready_image(monkeypatch) -> None:
    class ReadyRunner:
        def check(self) -> str:
            return "sha256:" + ("a" * 64)

    monkeypatch.setattr(cli, "DockerTestRunner", ReadyRunner)
    result = runner.invoke(app, ["tools-check"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "tools ready: sha256:" + ("a" * 64)


def test_tools_check_reports_unavailable(monkeypatch) -> None:
    class UnavailableRunner:
        def check(self) -> str:
            raise ToolError("docker_unavailable", "pinned local runner image is unavailable")

    monkeypatch.setattr(cli, "DockerTestRunner", UnavailableRunner)
    result = runner.invoke(app, ["tools-check"])
    assert result.exit_code == 1
    assert "tools unavailable" in result.output
    assert "Traceback" not in result.output


def test_cli_tool_cleanup_failure_writes_safe_atomic_failed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_root = tmp_path / "system-snapshots"
    snapshot_root.mkdir()
    output = tmp_path / "output"
    private_detail = f"private cleanup failure at {tmp_path}"
    monkeypatch.setattr(tool_module.tempfile, "gettempdir", lambda: str(snapshot_root))
    real_rmtree = tool_module.shutil.rmtree

    def remove_then_fail(path: Path) -> None:
        real_rmtree(path)
        raise OSError(private_detail)

    class PassingRunner:
        requires_sealed_snapshot = True

        def run(self, workspace: Path, profile: str) -> tool_module.TestExecution:
            assert workspace.is_relative_to(snapshot_root)
            return tool_module.TestExecution(ReviewTestStatus.PASSED, "tests passed")

    monkeypatch.setattr(tool_module.shutil, "rmtree", remove_then_fail)
    monkeypatch.setattr(workflow_module, "DockerTestRunner", PassingRunner)

    first = _invoke(TOOL_BENCHMARK, output)
    assert first.exit_code == 2, first.output
    run_path = output / "run.json"
    report_path = output / "report.md"
    run_text = run_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    artifact = RunArtifact.model_validate_json(run_text)
    assert artifact.schema_version == "1.2"
    assert artifact.final_state == WorkflowState.FAILED
    assert artifact.state_transitions[-1].to_state == WorkflowState.FAILED
    assert artifact.failure_code == "cleanup_failed"
    assert artifact.failure_message == "isolated workspace could not be removed"
    assert artifact.tool_trace[-1].status.value == "failed"
    assert artifact.tool_trace[-1].result.kind == "tool_failure"
    assert artifact.tool_trace[-1].result.code == "cleanup_failed"
    assert private_detail not in first.output
    assert private_detail not in run_text
    assert private_detail not in report_text
    assert str(tmp_path) not in run_text
    assert str(tmp_path) not in report_text
    assert {path.name for path in output.iterdir()} == {"run.json", "report.md"}

    run_path.write_text("private run sentinel", encoding="utf-8")
    report_path.write_text("private report sentinel", encoding="utf-8")
    refused = _invoke(TOOL_BENCHMARK, output)
    assert refused.exit_code == 2
    assert "--overwrite" in refused.output
    assert run_path.read_text(encoding="utf-8") == "private run sentinel"
    assert report_path.read_text(encoding="utf-8") == "private report sentinel"

    overwritten = _invoke(TOOL_BENCHMARK, output, "--overwrite")
    assert overwritten.exit_code == 2, overwritten.output
    overwritten_artifact = RunArtifact.model_validate_json(run_path.read_text(encoding="utf-8"))
    assert overwritten_artifact.failure_code == "cleanup_failed"
    assert report_path.read_text(encoding="utf-8").startswith("# Review Report:")
    assert {path.name for path in output.iterdir()} == {"run.json", "report.md"}
