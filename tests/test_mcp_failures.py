from __future__ import annotations

from pathlib import Path

from codereviewops.mcp_manifest import (
    CAPABILITY_FINGERPRINT,
    MCP_PROTOCOL_VERSION,
    expected_schema_fingerprint,
)
from codereviewops.models import ToolPlan
from codereviewops.tool_contracts import ToolError
from codereviewops.tools import DirectToolBackend, Workspace
from codereviewops.workflow import _execute_mcp_tools

ROOT = Path(__file__).parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks" / "tasks"
WORKSPACE_ROOT = BENCHMARK_ROOT / "workspaces" / "python_tools_001"
DIFF = (BENCHMARK_ROOT / "fixtures" / "python_tools_001.diff").read_text(encoding="utf-8")


class NoTests:
    def run(self, workspace: Path, profile: str):
        raise AssertionError


def test_shutdown_failure_overrides_success_and_snapshots_terminal_record() -> None:
    plan = ToolPlan(read_files=["calculator.py"])
    direct = DirectToolBackend(Workspace(WORKSPACE_ROOT, BENCHMARK_ROOT), NoTests())

    class FailingCloseBackend:
        def open(self) -> None:
            return None

        def read_file(self, path: str):
            return direct.read_file(path)

        def search_code(self, query: str):
            return direct.search_code(query)

        def run_tests(self, profile: str):
            return direct.run_tests(profile)

        def close(self) -> None:
            raise ToolError("mcp_lifecycle", "MCP process cleanup failed")

        def snapshot(self):
            return [
                {
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
                        "failed",
                    ],
                    "failure_stage": "shutdown",
                }
            ]

    run, records, failure, trace = _execute_mcp_tools(
        FailingCloseBackend(),
        plan,
        DIFF,  # type: ignore[arg-type]
    )
    assert run.trace == trace
    assert failure is not None and failure.code == "mcp_lifecycle"
    assert records[0].failure_stage == "shutdown"
    assert records[0].lifecycle[-1] == "failed"
