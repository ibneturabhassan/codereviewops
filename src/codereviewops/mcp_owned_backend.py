"""Strict MCP stdio ToolBackend using a repository-owned child transport."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from concurrent.futures import Future
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession, types
from pydantic import ValidationError

from codereviewops.mcp_manifest import (
    BENCHMARK_ROOT_ENV,
    CAPABILITIES_JSON,
    DOCKER_ENV,
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_VERSION,
    READ_INPUT_SCHEMA,
    SAFE_ANNOTATIONS_JSON,
    SEARCH_INPUT_SCHEMA,
    TEST_INPUT_SCHEMA,
    WORKSPACE_ENV,
    canonical_fingerprint,
    expected_server_name,
)
from codereviewops.mcp_transport import OwnedStdioTransport
from codereviewops.mcp_typed import output_schema_for, parse_envelope
from codereviewops.models import ReadFileResult, RunTestsResult, SearchCodeResult, ToolPlan
from codereviewops.tool_contracts import ToolError

_INIT_TIMEOUT = 5.0
_CALL_TIMEOUT = 5.0
_TEST_TIMEOUT = 40.0
_SHUTDOWN_TIMEOUT = 5.0
Kind = Literal["repo", "test"]


class BackendState(StrEnum):
    NEW = "new"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


def _planned_kinds(plan: ToolPlan) -> list[Kind]:
    result: list[Kind] = []
    if plan.read_files or plan.searches:
        result.append("repo")
    if plan.test_profile is not None:
        result.append("test")
    return result


def _expected_tools(kind: Kind) -> list[dict[str, Any]]:
    definitions = (
        [("read_file", READ_INPUT_SCHEMA), ("search_code", SEARCH_INPUT_SCHEMA)]
        if kind == "repo"
        else [("run_tests", TEST_INPUT_SCHEMA)]
    )
    return [
        {
            "name": name,
            "inputSchema": schema,
            "outputSchema": output_schema_for(name),
            "annotations": SAFE_ANNOTATIONS_JSON,
        }
        for name, schema in definitions
    ]


class McpStdioToolBackend:
    def __init__(self, workspace: Path, benchmark_root: Path, plan: ToolPlan) -> None:
        self.workspace = workspace.resolve(strict=True)
        self.benchmark_root = benchmark_root.resolve(strict=True)
        if not self.workspace.is_relative_to(self.benchmark_root):
            raise ToolError("mcp_lifecycle", "MCP workspace authority is invalid")
        self.plan = plan
        self._kinds = _planned_kinds(plan)
        self._records: dict[Kind, dict[str, Any]] = {
            kind: {
                "server_name": expected_server_name(kind),
                "lifecycle": ["planned"],
                "failure_stage": None,
            }
            for kind in self._kinds
        }
        self._portal_context: Any = None
        self._portal: BlockingPortal | None = None
        self._sessions: dict[Kind, ClientSession] = {}
        self._stops: dict[Kind, anyio.Event] = {}
        self._errors: dict[Kind, ToolError] = {}
        self._futures: dict[Kind, Future[None]] = {}
        self._transports: dict[Kind, OwnedStdioTransport] = {}
        self.state = BackendState.NEW
        self._close_error: ToolError | None = None
        self._close_completed = False

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(self._records[kind]) for kind in self._kinds]

    @property
    def protocol_version(self) -> str:
        return MCP_PROTOCOL_VERSION

    @property
    def child_pids(self) -> tuple[int, ...]:
        return tuple(
            transport.pid for transport in self._transports.values() if transport.pid is not None
        )

    @property
    def all_children_reaped(self) -> bool:
        return all(
            transport.process is not None and transport.process.poll() is not None
            for transport in self._transports.values()
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return self.records

    def abort_before_open(self) -> None:
        if self.state is not BackendState.NEW or not self._kinds:
            return
        first, *later = self._kinds
        self._fail(
            first, "configuration", ToolError("mcp_lifecycle", "MCP run stopped before spawn")
        )
        for kind in later:
            self._records[kind]["failure_stage"] = "skipped_after_failure"
            self._record(kind, "skipped")

    def _record(self, kind: Kind, event: str) -> None:
        lifecycle = self._records[kind]["lifecycle"]
        if not lifecycle or lifecycle[-1] != event:
            lifecycle.append(event)

    def _fail(self, kind: Kind, stage: str, error: ToolError) -> None:
        self._records[kind]["failure_stage"] = stage
        self._errors[kind] = error
        self._record(kind, "failed")

    def _system_root(self) -> str | None:
        if os.name != "nt":
            return None
        value = os.environ.get("SYSTEMROOT")
        if not value:
            raise ToolError("mcp_unavailable", "Windows system authority is unavailable")
        root = Path(value)
        if not root.is_absolute() or not root.is_dir():
            raise ToolError("mcp_unavailable", "Windows system authority is invalid")
        return str(root.resolve())

    def _docker(self) -> str:
        value = shutil.which("docker")
        if value is None:
            raise ToolError("docker_unavailable", "Docker CLI is unavailable")
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ToolError("docker_unavailable", "Docker CLI authority is invalid")
        return str(path.resolve())

    def _environment(self, kind: Kind) -> dict[str, str]:
        environment = {
            WORKSPACE_ENV: str(self.workspace),
            BENCHMARK_ROOT_ENV: str(self.benchmark_root),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        system_root = self._system_root()
        if system_root is not None:
            environment["SYSTEMROOT"] = system_root
        if kind == "test":
            environment[DOCKER_ENV] = self._docker()
        return environment

    def open(self) -> None:
        if self.state is BackendState.OPEN:
            return
        if self.state is not BackendState.NEW:
            raise ToolError("mcp_lifecycle", "MCP backend cannot be opened")
        self.state = BackendState.OPENING
        try:
            if types.LATEST_PROTOCOL_VERSION != MCP_PROTOCOL_VERSION:
                first, *later = self._kinds
                self._fail(
                    first,
                    "protocol_preflight",
                    ToolError("mcp_protocol", "MCP SDK protocol drifted"),
                )
                for kind in later:
                    self._records[kind]["failure_stage"] = "skipped_after_failure"
                    self._record(kind, "skipped")
                raise ToolError("mcp_protocol", "MCP SDK protocol drifted")
            context = start_blocking_portal(name="codereviewops-mcp")
            portal = context.__enter__()
            self._portal_context = context
            self._portal = portal
            for index, kind in enumerate(self._kinds):
                try:
                    self._start(kind, self._environment(kind))
                except ToolError as exc:
                    if self._records[kind]["lifecycle"] == ["planned"]:
                        self._fail(kind, "configuration", exc)
                    for skipped in self._kinds[index + 1 :]:
                        self._records[skipped]["failure_stage"] = "skipped_after_failure"
                        self._record(skipped, "skipped")
                    raise
            self.state = BackendState.OPEN
        except ToolError:
            self.state = BackendState.FAILED
            self.close()
            raise
        except Exception:
            first, *later = self._kinds
            self._fail(
                first,
                "client_runtime",
                ToolError("mcp_lifecycle", "MCP client runtime failed"),
            )
            for kind in later:
                self._records[kind]["failure_stage"] = "skipped_after_failure"
                self._record(kind, "skipped")
            self.state = BackendState.FAILED
            self.close()
            raise ToolError("mcp_lifecycle", "MCP client runtime failed") from None

    def _start(self, kind: Kind, environment: dict[str, str]) -> None:
        assert self._portal is not None
        ready = threading.Event()
        future = self._portal.start_task_soon(self._session_loop, kind, environment, ready)
        self._futures[kind] = future
        if not ready.wait(_INIT_TIMEOUT):
            error = ToolError("mcp_timeout", "MCP server initialization timed out")
            self._fail(kind, "initialize", error)
            raise error
        session_error = self._errors.get(kind)
        if session_error is not None:
            raise session_error
        if kind not in self._sessions:
            raise ToolError("mcp_lifecycle", "MCP server did not initialize")

    async def _session_loop(
        self, kind: Kind, environment: dict[str, str], ready: threading.Event
    ) -> None:
        module = (
            "codereviewops.mcp_repo_server" if kind == "repo" else "codereviewops.mcp_test_server"
        )
        stop = anyio.Event()
        self._stops[kind] = stop
        transport = OwnedStdioTransport(
            Path(sys.executable).resolve(strict=True), module, environment
        )
        self._transports[kind] = transport
        stage = "spawn"
        try:
            async with transport as streams:
                self._record(kind, "spawned")
                stage = "initialize"
                async with ClientSession(*streams) as session:
                    with anyio.fail_after(_INIT_TIMEOUT):
                        initialized = await session.initialize()
                    self._record(kind, "initialized")
                    stage = "list_tools"
                    with anyio.fail_after(_INIT_TIMEOUT):
                        listed = await session.list_tools()
                    self._validate_server(kind, initialized, listed)
                    self._record(kind, "tools_validated")
                    self._sessions[kind] = session
                    ready.set()
                    stage = "shutdown"
                    await stop.wait()
                    self._sessions.pop(kind, None)
        except TimeoutError:
            self._fail(kind, stage, ToolError("mcp_timeout", "MCP server timed out"))
        except ToolError as exc:
            self._fail(kind, stage, exc)
        except Exception:
            self._fail(kind, stage, ToolError("mcp_lifecycle", "MCP server lifecycle failed"))
        finally:
            if "close_requested" not in self._records[kind]["lifecycle"]:
                self._record(kind, "close_requested")
            if transport.process is not None and transport.process.poll() is not None:
                self._record(kind, "closed")
            ready.set()

    def _validate_server(
        self, kind: Kind, initialized: types.InitializeResult, listed: types.ListToolsResult
    ) -> None:
        if initialized.protocolVersion != MCP_PROTOCOL_VERSION:
            raise ToolError("mcp_protocol", "MCP protocol negotiation is invalid")
        expected_name = expected_server_name(kind)
        if (
            initialized.serverInfo.name != expected_name
            or initialized.serverInfo.version != MCP_SERVER_VERSION
        ):
            raise ToolError("mcp_protocol", "MCP server identity is invalid")
        capabilities = initialized.capabilities.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        if capabilities != CAPABILITIES_JSON:
            raise ToolError("mcp_protocol", "MCP server capabilities are invalid")
        actual_tools = [
            {
                "name": tool.name,
                "inputSchema": tool.inputSchema,
                "outputSchema": tool.outputSchema,
                "annotations": tool.annotations.model_dump(mode="json", exclude_none=True)
                if tool.annotations is not None
                else None,
            }
            for tool in listed.tools
        ]
        if actual_tools != _expected_tools(kind):
            raise ToolError("mcp_schema", "MCP tool schemas are invalid")
        self._records[kind].update(
            {
                "server_version": initialized.serverInfo.version,
                "protocol_version": initialized.protocolVersion,
                "capability_fingerprint": canonical_fingerprint(capabilities),
                "schema_fingerprint": canonical_fingerprint(actual_tools),
            }
        )

    async def _call(
        self, kind: Kind, name: str, arguments: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        session = self._sessions.get(kind)
        if session is None:
            raise ToolError("mcp_lifecycle", "MCP session is unavailable")
        try:
            with anyio.fail_after(timeout):
                response = await session.call_tool(
                    name, arguments, read_timeout_seconds=timedelta(seconds=timeout)
                )
        except TimeoutError:
            raise ToolError("mcp_timeout", "MCP tool call timed out") from None
        if len(response.content) != 1 or not isinstance(response.content[0], types.TextContent):
            raise ToolError("mcp_schema", "MCP content blocks are invalid")
        if response.structuredContent is None or len(response.content[0].text) > 1024 * 1024:
            raise ToolError("mcp_schema", "MCP structured content is invalid")
        try:
            parsed_text = json.loads(response.content[0].text)
            if parsed_text != response.structuredContent:
                raise ValueError
            envelope = parse_envelope(name, response.structuredContent)
        except (ValueError, ValidationError, json.JSONDecodeError):
            raise ToolError("mcp_schema", "MCP response envelope is invalid") from None
        if response.isError != (not envelope.ok):
            raise ToolError("mcp_schema", "MCP error status is inconsistent")
        if not envelope.ok:
            assert envelope.error is not None
            raise ToolError(
                cast(Any, envelope.error.code),
                envelope.error.message,
                retryable=envelope.error.retryable,
            )
        assert envelope.result is not None
        return envelope.result.model_dump(mode="python")

    def _portal_call(
        self, kind: Kind, name: str, arguments: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        if self.state is BackendState.NEW:
            self.open()
        if self.state is not BackendState.OPEN or self._portal is None:
            raise ToolError("mcp_lifecycle", "MCP backend is not open")
        try:
            return cast(
                dict[str, Any], self._portal.call(self._call, kind, name, arguments, timeout)
            )
        except ToolError:
            self._records[kind]["failure_stage"] = "call_tool"
            raise

    def read_file(self, path: str) -> ReadFileResult:
        return ReadFileResult.model_validate(
            self._portal_call("repo", "read_file", {"path": path}, _CALL_TIMEOUT)
        )

    def search_code(self, query: str) -> SearchCodeResult:
        return SearchCodeResult.model_validate(
            self._portal_call("repo", "search_code", {"query": query}, _CALL_TIMEOUT)
        )

    def run_tests(self, profile: str) -> RunTestsResult:
        return RunTestsResult.model_validate(
            self._portal_call("test", "run_tests", {"profile": profile}, _TEST_TIMEOUT)
        )

    def close(self) -> ToolError | None:
        if self._close_completed or self.state is BackendState.CLOSING:
            return self._close_error
        failed_before_close = self.state is BackendState.FAILED
        self.state = BackendState.CLOSING
        cleanup_error = False
        if self._portal is not None:
            for kind, stop in list(self._stops.items()):
                self._record(kind, "close_requested")
                try:
                    self._portal.call(stop.set)
                except Exception:
                    cleanup_error = True
                    self._records[kind]["failure_stage"] = "shutdown"
                    self._record(kind, "failed")
            for kind, future in self._futures.items():
                try:
                    future.result(timeout=_SHUTDOWN_TIMEOUT)
                except Exception:
                    cleanup_error = True
                    self._records[kind]["failure_stage"] = "shutdown"
                    self._record(kind, "failed")
            try:
                assert self._portal_context is not None
                self._portal_context.__exit__(None, None, None)
            except Exception:
                cleanup_error = True
                for kind in self._stops:
                    self._records[kind]["failure_stage"] = "client_runtime"
                    self._record(kind, "failed")
        for kind, transport in self._transports.items():
            if transport.process is not None and transport.process.poll() is not None:
                self._record(kind, "closed")
            else:
                cleanup_error = True
                self._records[kind]["failure_stage"] = "shutdown"
                self._record(kind, "failed")
        if cleanup_error:
            self._close_error = ToolError("mcp_lifecycle", "MCP process cleanup failed")
        self.state = (
            BackendState.FAILED
            if failed_before_close or self._close_error is not None
            else BackendState.CLOSED
        )
        self._close_completed = True
        return self._close_error
