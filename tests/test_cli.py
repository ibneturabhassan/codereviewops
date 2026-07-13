from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codereviewops.cli import app
from codereviewops.models import RunArtifact

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "tasks" / "http_retry_001.json"
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
    assert "invalid replay response" in result.output


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
