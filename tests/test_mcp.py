from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import anyio
import pytest
from mcp import types
from typer.testing import CliRunner

import codereviewops.workflow as workflow
from codereviewops.cli import app
from codereviewops.io import InputError, load_task
from codereviewops.mcp_backend import McpStdioToolBackend
from codereviewops.mcp_contracts import create_repo_server, create_test_server
from codereviewops.mcp_manifest import (
    CAPABILITY_FINGERPRINT,
    expected_schema_fingerprint,
)
from codereviewops.models import ToolPlan
from codereviewops.providers import ReplayProvider
from codereviewops.tools import (
    DirectToolBackend,
    ToolError,
    Workspace,
    execute_tool_plan_with_backend,
)

ROOT = Path(__file__).parents[1]
BENCHMARK_ROOT = ROOT / "tests" / "fixtures" / "legacy_benchmarks"
WORKSPACE_ROOT = BENCHMARK_ROOT / "workspaces" / "python_tools_001"
DIFF = (BENCHMARK_ROOT / "fixtures" / "python_tools_001.diff").read_text(encoding="utf-8")


class NoTests:
    def run(self, workspace: Path, profile: str):
        raise AssertionError


def _semantic_trace(run):
    result = []
    entries = run.tool_trace if hasattr(run, "tool_trace") else run.trace
    for entry in entries:
        data = entry.model_dump(mode="python")
        data["latency_ms"] = 0
        result.append(data)
    return result


def test_exact_stable_mcp_sdk_is_pinned() -> None:
    assert importlib.metadata.version("mcp") == "1.27.2"


@pytest.mark.mcp
def test_real_stdio_repo_backend_matches_direct_and_closes() -> None:
    plan = ToolPlan(read_files=["calculator.py"], searches=["subtract"])
    workspace = Workspace(WORKSPACE_ROOT, BENCHMARK_ROOT)
    direct = execute_tool_plan_with_backend(DirectToolBackend(workspace, NoTests()), plan, DIFF)

    backend = McpStdioToolBackend(WORKSPACE_ROOT, BENCHMARK_ROOT, plan)
    backend.open()
    child_pids = backend.child_pids
    try:
        mcp = execute_tool_plan_with_backend(backend, plan, DIFF)
    finally:
        backend.close()

    assert _semantic_trace(mcp) == _semantic_trace(direct)
    assert mcp.verification == direct.verification
    assert backend.protocol_version == "2025-11-25"
    assert child_pids and all(pid > 0 for pid in child_pids)
    assert backend.all_children_reaped
    assert backend.records[0]["lifecycle"] == [
        "planned",
        "spawned",
        "initialized",
        "tools_validated",
        "close_requested",
        "closed",
    ]


def test_child_authority_environment_is_minimal() -> None:
    backend = object.__new__(McpStdioToolBackend)
    backend.workspace = WORKSPACE_ROOT.resolve()
    backend.benchmark_root = BENCHMARK_ROOT.resolve()
    environment = backend._environment("repo")
    expected = {
        "CODEREVIEWOPS_MCP_WORKSPACE",
        "CODEREVIEWOPS_MCP_BENCHMARK_ROOT",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
    }
    if os.name == "nt":
        expected.add("SYSTEMROOT")
    assert set(environment) == expected
    assert not any(
        key in environment
        for key in (
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        )
    )


class FakeRunner:
    def run(self, workspace: Path, profile: str):
        from codereviewops.models import TestStatus
        from codereviewops.tools import TestExecution

        return TestExecution(TestStatus.FAILED, "one deterministic test failure")


def test_workflow_mcp_transport_emits_13_and_matches_direct(monkeypatch) -> None:
    task_path = BENCHMARK_ROOT / "python_tools_001.json"
    loaded = load_task(task_path)

    class FakeMcpBackend:
        def __init__(self, workspace: Path, benchmark_root: Path, plan: ToolPlan) -> None:
            del plan
            authority = Workspace(workspace, benchmark_root)
            self.direct = DirectToolBackend(authority, FakeRunner())
            self.records = [
                {
                    "server_name": "codereviewops-repo-mcp",
                    "server_version": "0.1.0",
                    "protocol_version": "2025-11-25",
                    "capability_fingerprint": CAPABILITY_FINGERPRINT,
                    "schema_fingerprint": expected_schema_fingerprint("repo"),
                    "lifecycle": [
                        "planned",
                        "spawned",
                        "initialized",
                        "tools_validated",
                        "close_requested",
                        "closed",
                    ],
                    "failure_stage": None,
                },
                {
                    "server_name": "codereviewops-test-mcp",
                    "server_version": "0.1.0",
                    "protocol_version": "2025-11-25",
                    "capability_fingerprint": CAPABILITY_FINGERPRINT,
                    "schema_fingerprint": expected_schema_fingerprint("test"),
                    "lifecycle": [
                        "planned",
                        "spawned",
                        "initialized",
                        "tools_validated",
                        "close_requested",
                        "closed",
                    ],
                    "failure_stage": None,
                },
            ]
            self.protocol_version = "2025-11-25"

        def open(self) -> None:
            return None

        def snapshot(self):
            return self.records

        def abort_before_open(self) -> None:
            return None

        def read_file(self, path: str):
            return self.direct.read_file(path)

        def search_code(self, query: str):
            return self.direct.search_code(query)

        def run_tests(self, profile: str):
            return self.direct.run_tests(profile)

        def close(self) -> None:
            return None

    monkeypatch.setattr(workflow, "McpStdioToolBackend", FakeMcpBackend)
    _, direct = workflow.run_loaded_task(
        loaded, ReplayProvider(loaded.replay_path), test_runner=FakeRunner()
    )
    _, mcp = workflow.run_loaded_task(
        loaded,
        ReplayProvider(loaded.replay_path),
        test_runner=FakeRunner(),
        tool_transport="mcp-stdio",
    )

    assert mcp.schema_version == "1.3"
    assert mcp.tool_transport == "mcp_stdio"
    assert mcp.mcp_protocol_version == "2025-11-25"
    assert len(mcp.mcp_servers) == 2
    assert _semantic_trace(mcp) == _semantic_trace(direct)
    assert mcp.review == direct.review
    assert mcp.evaluation == direct.evaluation
    assert mcp.candidate_verifications == direct.candidate_verifications
    assert mcp.semantic_trace_fingerprint is not None
    payload = mcp.model_dump(mode="python")
    payload["semantic_trace_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="semantic tool trace fingerprint is invalid"):
        type(mcp).model_validate(payload)


def test_schema_10_rejects_mcp_before_provider() -> None:
    loaded = load_task(BENCHMARK_ROOT / "http_retry_001.json")

    class NeverProvider:
        def review(self, context):
            raise AssertionError

    with pytest.raises(InputError, match=r"schema 1\.0 tasks do not support MCP"):
        workflow.run_loaded_task(loaded, NeverProvider(), tool_transport="mcp-stdio")


def _call_handler(server, name: str, arguments: dict):
    handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name=name, arguments=arguments)
    )
    return anyio.run(handler, request).root


def test_servers_reject_extra_arguments_and_enforce_session_limits() -> None:
    workspace = Workspace(WORKSPACE_ROOT, BENCHMARK_ROOT)
    repo = create_repo_server(workspace)
    malformed = _call_handler(repo, "read_file", {"path": "calculator.py", "root": "elsewhere"})
    assert malformed.isError is True
    assert malformed.structuredContent["error"]["code"] == "mcp_schema"
    assert json.loads(malformed.content[0].text) == malformed.structuredContent

    for _ in range(20):
        assert _call_handler(repo, "read_file", {"path": "calculator.py"}).isError is False
    limited = _call_handler(repo, "read_file", {"path": "calculator.py"})
    assert limited.isError is True
    assert limited.structuredContent["error"]["code"] == "limit_exceeded"

    tests = create_test_server(workspace, DirectToolBackend(workspace, FakeRunner()))
    assert _call_handler(tests, "run_tests", {"profile": "python-unittest-v1"}).isError is False
    second = _call_handler(tests, "run_tests", {"profile": "python-unittest-v1"})
    assert second.isError is True
    assert second.structuredContent["error"]["code"] == "limit_exceeded"


def test_mcp_call_deadline_returns_safe_timeout() -> None:
    class SlowSession:
        async def call_tool(self, *args, **kwargs):
            await anyio.sleep_forever()

    backend = object.__new__(McpStdioToolBackend)
    backend._sessions = {"repo": SlowSession()}
    with pytest.raises(ToolError) as caught:
        anyio.run(backend._call, "repo", "read_file", {"path": "calculator.py"}, 0.001)
    assert getattr(caught.value, "code", None) == "mcp_timeout"
    assert str(caught.value) == "MCP tool call timed out"


def test_cli_exposes_transport_and_rejects_schema_10_mcp(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["review", "--help"])
    assert help_result.exit_code == 0
    assert "--tool-transport" in help_result.stdout
    result = runner.invoke(
        app,
        [
            "review",
            "--task",
            str(BENCHMARK_ROOT / "http_retry_001.json"),
            "--provider",
            "replay",
            "--tool-transport",
            "mcp-stdio",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "schema 1.0 tasks do not support MCP tool transport" in result.stderr
    assert not list(tmp_path.iterdir())
