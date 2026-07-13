from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codereviewops.io import InputError, load_task
from codereviewops.models import ReviewContext
from codereviewops.providers import ProviderError, ReplayProvider
from codereviewops.workflow import run_loaded_task


def _report_data() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "summary": "No findings",
        "overall_assessment": "pass",
        "findings": [],
        "tests_run": [],
        "limitations": [],
    }


def _task_data() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": "test_task",
        "title": "Test task",
        "issue_description": "Review the change.",
        "diff_path": "fixtures/change.diff",
        "expected_findings": [],
        "must_not_find": ["secret golden phrase"],
        "difficulty": "low",
        "tags": ["test"],
        "replay_response_path": "replays/review.json",
    }


def _write_tree(root: Path) -> Path:
    task_dir = root / "task"
    (task_dir / "fixtures").mkdir(parents=True)
    (task_dir / "replays").mkdir()
    (task_dir / "fixtures" / "change.diff").write_text("synthetic diff", encoding="utf-8")
    (task_dir / "replays" / "review.json").write_text(json.dumps(_report_data()), encoding="utf-8")
    manifest = task_dir / "task.json"
    manifest.write_text(json.dumps(_task_data()), encoding="utf-8")
    return manifest


def test_load_task_resolves_valid_references(tmp_path: Path) -> None:
    loaded = load_task(_write_tree(tmp_path))
    assert loaded.diff_path.read_text(encoding="utf-8") == "synthetic diff"
    assert loaded.replay_path.name == "review.json"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diff_path", "/outside.diff"),
        ("diff_path", "../outside.diff"),
        ("diff_path", "fixtures\\change.diff"),
        ("replay_response_path", "C:/outside.json"),
    ],
)
def test_load_task_rejects_unsafe_reference_forms(tmp_path: Path, field: str, value: str) -> None:
    manifest = _write_tree(tmp_path)
    data = _task_data()
    data[field] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError):
        load_task(manifest)


def test_load_task_rejects_symlink_escape(tmp_path: Path) -> None:
    manifest = _write_tree(tmp_path)
    external = tmp_path / "outside.diff"
    external.write_text("outside", encoding="utf-8")
    link = manifest.parent / "fixtures" / "escape.diff"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    data = _task_data()
    data["diff_path"] = "fixtures/escape.diff"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match="escapes"):
        load_task(manifest)


def test_load_task_rejects_missing_reference(tmp_path: Path) -> None:
    manifest = _write_tree(tmp_path)
    (manifest.parent / "fixtures" / "change.diff").unlink()
    with pytest.raises(InputError, match="missing"):
        load_task(manifest)


def test_load_task_rejects_malformed_manifest_json(tmp_path: Path) -> None:
    manifest = tmp_path / "task.json"
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(InputError, match="valid JSON"):
        load_task(manifest)


def test_replay_provider_rejects_malformed_and_schema_invalid_json(tmp_path: Path) -> None:
    replay = tmp_path / "review.json"
    replay.write_text("{", encoding="utf-8")
    with pytest.raises(ProviderError):
        ReplayProvider(replay)
    replay.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    with pytest.raises(ProviderError):
        ReplayProvider(replay)


def test_provider_receives_only_review_context(tmp_path: Path, report_factory) -> None:
    loaded = load_task(_write_tree(tmp_path))
    captured: list[ReviewContext] = []

    class SpyProvider:
        def review(self, context: ReviewContext):
            captured.append(context)
            return report_factory([])

    _, artifact = run_loaded_task(loaded, SpyProvider())
    assert artifact.evaluation.task_success
    assert len(captured) == 1
    assert set(captured[0].model_dump()) == {
        "schema_version",
        "task_id",
        "title",
        "issue_description",
        "diff_text",
    }
    serialized = captured[0].model_dump_json()
    assert "expected_findings" not in serialized
    assert "must_not_find" not in serialized
    assert "replay_response_path" not in serialized
    assert "secret golden phrase" not in serialized
