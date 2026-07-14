from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from mcp import types
from pydantic import ValidationError

import codereviewops.workflow as workflow
from codereviewops.io import InputError, load_task
from codereviewops.mcp_manifest import (
    CAPABILITY_FINGERPRINT,
    MCP_PROTOCOL_VERSION,
    expected_schema_fingerprint,
)
from codereviewops.mcp_owned_backend import McpStdioToolBackend
from codereviewops.mcp_servers import create_repo_server
from codereviewops.mcp_typed import parse_envelope
from codereviewops.models import McpServerRecord, ToolPlan
from codereviewops.tool_contracts import ToolError
from codereviewops.tools import Workspace

ROOT = Path(__file__).parents[1]
BENCHMARK_ROOT = ROOT / "tests" / "fixtures" / "legacy_benchmarks"
WORKSPACE_ROOT = BENCHMARK_ROOT / "workspaces" / "python_tools_001"


def _call_read() -> types.CallToolResult:
    server = create_repo_server(Workspace(WORKSPACE_ROOT, BENCHMARK_ROOT))
    handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name="read_file", arguments={"path": "calculator.py"})
    )
    return anyio.run(handler, request).root


def test_owned_transport_has_no_sdk_stdio_process_dependency() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "codereviewops").glob("mcp_*.py")
    )
    assert "mcp.client.stdio" not in production
    assert "stdio_client" not in production
    assert "_create_platform_compatible_process" not in production


def test_typed_envelopes_reject_cross_tool_and_extra_fields() -> None:
    response = _call_read()
    assert parse_envelope("read_file", response.structuredContent).ok is True
    with pytest.raises(ValidationError):
        parse_envelope("search_code", response.structuredContent)
    malformed = dict(response.structuredContent or {})
    malformed["extra"] = True
    with pytest.raises(ValidationError):
        parse_envelope("read_file", malformed)


def test_server_record_rejects_arbitrary_and_swapped_fingerprints() -> None:
    base = {
        "server_name": "codereviewops-repo-mcp",
        "server_version": "0.1.0",
        "protocol_version": MCP_PROTOCOL_VERSION,
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
    }
    McpServerRecord.model_validate(base)
    with pytest.raises(ValidationError):
        McpServerRecord.model_validate({**base, "capability_fingerprint": "sha256:" + "0" * 64})
    with pytest.raises(ValidationError):
        McpServerRecord.model_validate(
            {**base, "schema_fingerprint": expected_schema_fingerprint("test")}
        )


def test_protocol_drift_is_schema_13_failure_before_provider(monkeypatch) -> None:
    loaded = load_task(BENCHMARK_ROOT / "python_tools_001.json")

    class NeverProvider:
        def review(self, context):
            raise AssertionError

    monkeypatch.setattr(types, "LATEST_PROTOCOL_VERSION", "2024-11-05")
    _, artifact = workflow.run_loaded_task(loaded, NeverProvider(), tool_transport="mcp-stdio")
    assert artifact.schema_version == "1.3"
    assert artifact.provider_status == "not_called"
    assert artifact.failure_code == "mcp_protocol"
    assert artifact.mcp_protocol_version == MCP_PROTOCOL_VERSION
    assert artifact.semantic_trace_fingerprint is not None
    assert [record.failure_stage for record in artifact.mcp_servers] == [
        "protocol_preflight",
        "skipped_after_failure",
    ]


def test_empty_mcp_plan_rejects_before_diff_spawn_and_provider(tmp_path: Path) -> None:
    loaded = load_task(BENCHMARK_ROOT / "python_tools_001.json")
    empty = replace(
        loaded,
        task=loaded.task.model_copy(update={"tool_plan": ToolPlan()}),
        diff_path=tmp_path / "missing.diff",
    )

    class NeverProvider:
        def review(self, context):
            raise AssertionError

    with pytest.raises(InputError, match="at least one declared tool"):
        workflow.run_loaded_task(empty, NeverProvider(), tool_transport="mcp-stdio")


def test_relative_docker_locator_is_rejected_before_spawn(monkeypatch) -> None:
    monkeypatch.setattr("codereviewops.mcp_owned_backend.shutil.which", lambda _: "docker")
    backend = McpStdioToolBackend(
        WORKSPACE_ROOT,
        BENCHMARK_ROOT,
        ToolPlan(test_profile="python-unittest-v1"),
    )

    with pytest.raises(ToolError, match="Docker CLI authority is invalid"):
        backend.open()
    backend.close()

    assert backend.child_pids == ()
    [record] = backend.records
    assert record["lifecycle"] == ["planned", "failed"]
    assert record["failure_stage"] == "configuration"
