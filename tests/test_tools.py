from __future__ import annotations

import os
from pathlib import Path

import pytest

from codereviewops.models import TestStatus as ReviewTestStatus
from codereviewops.models import ToolName, ToolPlan
from codereviewops.tools import (
    MAX_READ_BYTES,
    MAX_TRACE_RESULT_BYTES,
    ToolError,
    ToolExecutionError,
    Workspace,
    changed_locations,
    execute_tool_plan,
    parse_unified_diff,
)
from codereviewops.tools import (
    TestExecution as ToolTestExecution,
)

VALID_TEST_DIFF = "--- a/alpha.py\n+++ b/alpha.py\n@@ -1 +1 @@\n-original\n+changed\n"


def _workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "beta.py").write_text("needle = 2\n", encoding="utf-8")
    (root / "alpha.py").write_text("needle = 1\nother = 3\n", encoding="utf-8")
    return Workspace(root, tmp_path)


def test_read_file_and_literal_search_are_deterministic(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    assert workspace.files == ("alpha.py", "src/beta.py")
    assert workspace.read_file("alpha.py") == "needle = 1\nother = 3\n"
    result = workspace.search_code("needle")
    assert [(match.path, match.line, match.column) for match in result.matches] == [
        ("alpha.py", 1, 1),
        ("src/beta.py", 1, 1),
    ]
    assert result.files_total == 2
    assert result.files_scanned == 2
    assert result.files_skipped == 0
    assert workspace.search_code("needle.*").matches == []


@pytest.mark.parametrize("reference", ["../outside.py", "/absolute.py", ".git/config"])
def test_read_file_rejects_unsafe_paths(tmp_path: Path, reference: str) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(ToolError):
        workspace.read_file(reference)


def test_read_file_enforces_limit_and_truthful_trace_truncation(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "boundary.txt").write_bytes(b"x" * MAX_READ_BYTES)
    (root / "large.txt").write_bytes(b"x" * (MAX_READ_BYTES + 1))
    (root / "trace.txt").write_bytes(b"y" * (MAX_TRACE_RESULT_BYTES + 1))
    (root / "binary.bin").write_bytes(b"a\x00b")
    (root / "invalid.txt").write_bytes(b"\xff")
    workspace = Workspace(root, tmp_path)

    boundary = workspace.read_result("boundary.txt")
    assert boundary.source_bytes == MAX_READ_BYTES
    assert boundary.normalized_bytes == MAX_READ_BYTES
    assert boundary.returned_bytes == MAX_TRACE_RESULT_BYTES
    assert boundary.truncated
    trace = workspace.read_result("trace.txt")
    assert trace.returned_bytes == MAX_TRACE_RESULT_BYTES
    assert len(trace.content.encode("utf-8")) == MAX_TRACE_RESULT_BYTES
    assert trace.truncated
    for reader in (workspace.read_file, workspace.read_result):
        with pytest.raises(ToolError) as captured:
            reader("large.txt")
        assert captured.value.code == "limit_exceeded"
    for reference in ("binary.bin", "invalid.txt"):
        with pytest.raises(ToolError):
            workspace.read_file(reference)


def test_workspace_rejects_git_and_symlink_entries(tmp_path: Path) -> None:
    git_root = tmp_path / "git-workspace"
    (git_root / ".git").mkdir(parents=True)
    with pytest.raises(ToolError, match="git"):
        Workspace(git_root, tmp_path)

    target = tmp_path / "target.txt"
    target.write_text("safe", encoding="utf-8")
    link_root = tmp_path / "link-workspace"
    link_root.mkdir()
    try:
        (link_root / "link.txt").symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ToolError, match="links"):
        Workspace(link_root, tmp_path)


def test_workspace_rejects_file_and_byte_caps(tmp_path: Path, monkeypatch) -> None:
    import codereviewops.tools as tools

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "one.txt").write_text("12", encoding="utf-8")
    (root / "two.txt").write_text("34", encoding="utf-8")
    monkeypatch.setattr(tools, "MAX_WORKSPACE_FILES", 1)
    with pytest.raises(ToolError, match="file limit"):
        Workspace(root, tmp_path)
    monkeypatch.setattr(tools, "MAX_WORKSPACE_FILES", 10)
    monkeypatch.setattr(tools, "MAX_WORKSPACE_BYTES", 3)
    with pytest.raises(ToolError, match="byte limit"):
        Workspace(root, tmp_path)


def test_workspace_rejects_depth_cap(tmp_path: Path, monkeypatch) -> None:
    import codereviewops.tools as tools

    root = tmp_path / "workspace"
    nested = root / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("safe", encoding="utf-8")
    monkeypatch.setattr(tools, "MAX_WORKSPACE_DEPTH", 1)
    with pytest.raises(ToolError, match="depth limit"):
        Workspace(root, tmp_path)


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent
    with pytest.raises(ToolError, match="escapes"):
        Workspace(outside, tmp_path)


def test_workspace_rejects_special_entry(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")

    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "entry"
    os.mkfifo(path)
    with pytest.raises(ToolError, match="special"):
        Workspace(root, tmp_path)


@pytest.mark.parametrize("query", ["", "x" * 129, "line\nfeed"])
def test_search_rejects_invalid_queries(tmp_path: Path, query: str) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(ToolError):
        workspace.search_code(query)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def run(self, workspace: Path, profile: str) -> ToolTestExecution:
        self.calls.append((workspace, profile))
        return ToolTestExecution(ReviewTestStatus.FAILED, "one deterministic failure")


def test_tool_plan_order_trace_and_verification(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner()
    plan = ToolPlan(
        read_files=["alpha.py"],
        searches=["needle"],
        test_profile="python-unittest-v1",
    )
    result = execute_tool_plan(
        workspace,
        plan,
        runner,
        "diff --git a/alpha.py b/alpha.py\n"
        "--- a/alpha.py\n"
        "+++ b/alpha.py\n"
        "@@ -1 +1 @@\n"
        "-old\n+new\n",
    )
    assert [entry.tool for entry in result.trace] == [
        ToolName.READ_FILE,
        ToolName.SEARCH_CODE,
        ToolName.RUN_TESTS,
    ]
    assert [entry.trace_id for entry in result.trace] == ["tool-001", "tool-002", "tool-003"]
    assert len(runner.calls) == 1
    assert runner.calls[0][1] == "python-unittest-v1"
    assert result.verification is not None
    assert result.verification.status == ReviewTestStatus.FAILED
    assert [item.model_dump() for item in result.verification.changed_locations] == [
        {"path": "alpha.py", "line_start": 1, "line_end": 1}
    ]


def test_changed_locations_are_bounded_and_safe() -> None:
    with pytest.raises(ToolError) as captured:
        changed_locations("--- a/safe.py\n+++ b/../escape.py\n@@ -1 +1 @@\n-old\n+new\n")
    assert captured.value.code == "invalid_diff"


def test_unified_diff_parser_tracks_only_exact_added_lines() -> None:
    diff = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+first\n"
        "++++ literal header text\n"
        "+third\n"
        "\\ No newline at end of file\n"
    )
    assert parse_unified_diff(diff) == {"new.py": frozenset({1, 2, 3})}
    assert [item.model_dump() for item in changed_locations(diff)] == [
        {"path": "new.py", "line_start": 1, "line_end": 3}
    ]


def test_unified_diff_parser_accepts_markers_after_removed_and_added_lines() -> None:
    diff = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )
    assert parse_unified_diff(diff) == {"a.py": frozenset({1})}


def test_unified_diff_parser_tracks_two_files_without_git_metadata() -> None:
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old-a\n+new-a\n"
        "--- a/b.py\n+++ b/b.py\n@@ -2 +2 @@\n-old-b\n+new-b\n"
    )
    assert parse_unified_diff(diff) == {
        "a.py": frozenset({1}),
        "b.py": frozenset({2}),
    }


@pytest.mark.parametrize(
    "diff",
    [
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+only-new\n",
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n+extra\n",
        "--- a/a.py\n+++ b/a.py\n\\ No newline at end of file\n",
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n\\ No newline at end of file\n-old\n+new\n",
        (
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n"
            "\\ No newline at end of file\n\\ No newline at end of file\n+new\n"
        ),
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n\\ No newline at EOF\n+new\n",
        (
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
            "diff --git a/b.py b/b.py\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    ],
)
def test_unified_diff_parser_rejects_malformed_counts_and_markers(diff: str) -> None:
    with pytest.raises(ToolError) as captured:
        parse_unified_diff(diff)
    assert captured.value.code == "invalid_diff"


def test_replaced_file_produces_one_failed_trace(tmp_path: Path) -> None:

    workspace = _workspace(tmp_path)
    (workspace.root / "alpha.py").write_text("replacement", encoding="utf-8")
    with pytest.raises(ToolExecutionError) as captured:
        execute_tool_plan(workspace, ToolPlan(read_files=["alpha.py"]), FakeRunner(), "")
    assert len(captured.value.trace) == 1
    assert captured.value.trace[0].status.value == "failed"
    assert captured.value.trace[0].result.kind == "tool_failure"
    assert captured.value.trace[0].result.code == "changed_workspace"


def _snapshot_workspace(tmp_path: Path) -> tuple[Workspace, Path, Path]:
    benchmark = tmp_path / "benchmark"
    root = benchmark / "workspace"
    snapshot_root = tmp_path / "snapshots"
    (root / "nested").mkdir(parents=True)
    snapshot_root.mkdir()
    (root / "alpha.py").write_text("original\n", encoding="utf-8")
    (root / "nested" / "beta.py").write_text("nested\n", encoding="utf-8")
    return (
        Workspace(root, benchmark, snapshot_root=snapshot_root),
        root,
        snapshot_root,
    )


def test_sealed_snapshot_copies_immutable_contents_and_is_removed(tmp_path: Path) -> None:
    workspace, source_root, snapshot_root = _snapshot_workspace(tmp_path)

    class SnapshotRunner:
        requires_sealed_snapshot = True

        def __init__(self) -> None:
            self.snapshot: Path | None = None

        def run(self, snapshot: Path, profile: str) -> ToolTestExecution:
            self.snapshot = snapshot
            assert snapshot.is_relative_to(snapshot_root)
            assert not snapshot.is_relative_to(workspace._benchmark_root)
            assert (snapshot / "nested" / "beta.py").read_text(encoding="utf-8") == "nested\n"
            source_root.joinpath("alpha.py").write_text("changed\n", encoding="utf-8")
            assert (snapshot / "alpha.py").read_text(encoding="utf-8") == "original\n"
            return ToolTestExecution(ReviewTestStatus.PASSED, "tests passed")

    runner = SnapshotRunner()
    result = execute_tool_plan(
        workspace,
        ToolPlan(test_profile="python-unittest-v1"),
        runner,
        VALID_TEST_DIFF,
    )
    assert result.verification is not None
    assert runner.snapshot is not None
    assert not runner.snapshot.exists()
    assert list(snapshot_root.iterdir()) == []


def test_sealed_snapshot_cleanup_failure_is_safe_failed_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codereviewops.tools as tools

    workspace, _, _ = _snapshot_workspace(tmp_path)
    real_rmtree = tools.shutil.rmtree

    def remove_then_fail(path: Path) -> None:
        real_rmtree(path)
        raise OSError(f"private cleanup path: {path}")

    class SnapshotRunner:
        requires_sealed_snapshot = True

        def run(self, workspace: Path, profile: str) -> ToolTestExecution:
            return ToolTestExecution(ReviewTestStatus.PASSED, "tests passed")

    monkeypatch.setattr(tools.shutil, "rmtree", remove_then_fail)
    with pytest.raises(tools.ToolExecutionError) as captured:
        execute_tool_plan(
            workspace,
            ToolPlan(test_profile="python-unittest-v1"),
            SnapshotRunner(),
            VALID_TEST_DIFF,
        )
    assert captured.value.code == "cleanup_failed"
    assert "private cleanup path" not in captured.value.message
    assert len(captured.value.trace) == 1
    assert captured.value.trace[0].result.code == "cleanup_failed"


def test_sealed_snapshot_creation_failure_is_safe_failed_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codereviewops.tools as tools

    workspace, _, _ = _snapshot_workspace(tmp_path)

    def fail_creation(root: Path) -> Path:
        raise OSError("private creation path")

    monkeypatch.setattr(tools, "_create_disposable_directory", fail_creation)
    with pytest.raises(tools.ToolExecutionError) as captured:
        execute_tool_plan(
            workspace,
            ToolPlan(test_profile="python-unittest-v1"),
            type("Runner", (), {"requires_sealed_snapshot": True})(),
            VALID_TEST_DIFF,
        )
    assert captured.value.code == "cleanup_failed"
    assert "private creation path" not in captured.value.message
    assert captured.value.trace[0].result.code == "cleanup_failed"
