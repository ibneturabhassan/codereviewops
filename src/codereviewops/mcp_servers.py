"""Fixed local stdio MCP server definitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anyio
from mcp import types
from mcp.server import InitializationOptions, Server
from mcp.server.stdio import stdio_server

from codereviewops.docker_runner import DockerTestRunner
from codereviewops.mcp_manifest import (
    BENCHMARK_ROOT_ENV,
    DOCKER_ENV,
    MCP_SERVER_VERSION,
    READ_INPUT_SCHEMA,
    REPO_SERVER_NAME,
    SEARCH_INPUT_SCHEMA,
    TEST_INPUT_SCHEMA,
    TEST_SERVER_NAME,
    WORKSPACE_ENV,
)
from codereviewops.mcp_typed import (
    ErrorEnvelope,
    McpToolError,
    ReadSuccess,
    SearchSuccess,
    TestSuccess,
    output_schema_for,
)
from codereviewops.tools import DirectToolBackend, ToolError, Workspace

SAFE_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)


def _result(envelope: Any) -> types.CallToolResult:
    structured = envelope.model_dump(mode="json")
    text = json.dumps(structured, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=structured,
        isError=not envelope.ok,
    )


def _error(error: ToolError) -> types.CallToolResult:
    return _result(
        ErrorEnvelope(
            error=McpToolError(
                code=error.code,
                message=error.message,
                retryable=error.retryable,
            )
        )
    )


def _workspace() -> Workspace:
    workspace = os.environ.get(WORKSPACE_ENV)
    root = os.environ.get(BENCHMARK_ROOT_ENV)
    if (
        not workspace
        or not root
        or not Path(workspace).is_absolute()
        or not Path(root).is_absolute()
    ):
        raise RuntimeError("MCP workspace authority is invalid")
    return Workspace(Path(workspace), Path(root))


def _docker() -> str:
    value = os.environ.get(DOCKER_ENV)
    if not value:
        raise RuntimeError("MCP Docker authority is unavailable")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError("MCP Docker authority is invalid")
    return str(path.resolve())


def _argument(arguments: dict[str, Any], key: str, maximum: int) -> str:
    value = arguments.get(key)
    if set(arguments) != {key} or type(value) is not str or not value or len(value) > maximum:
        raise ToolError("mcp_schema", "MCP tool arguments are invalid")
    return value


def create_repo_server(workspace: Workspace | None = None) -> Server[Any]:
    authority = workspace or _workspace()
    server: Server[Any] = Server(REPO_SERVER_NAME, version=MCP_SERVER_VERSION)
    counts = {"read_file": 0, "search_code": 0}
    tools = [
        types.Tool(
            name="read_file",
            description="Read one validated repository-relative UTF-8 text file.",
            inputSchema=READ_INPUT_SCHEMA,
            outputSchema=output_schema_for("read_file"),
            annotations=SAFE_ANNOTATIONS,
        ),
        types.Tool(
            name="search_code",
            description="Search validated workspace text files for a literal query.",
            inputSchema=SEARCH_INPUT_SCHEMA,
            outputSchema=output_schema_for("search_code"),
            annotations=SAFE_ANNOTATIONS,
        ),
    ]

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        try:
            if name == "read_file":
                path = _argument(arguments, "path", 512)
                if counts[name] >= 20:
                    raise ToolError("limit_exceeded", "read_file session limit exceeded")
                counts[name] += 1
                return _result(ReadSuccess(result=authority.read_result(path)))
            if name == "search_code":
                query = _argument(arguments, "query", 128)
                if counts[name] >= 10:
                    raise ToolError("limit_exceeded", "search_code session limit exceeded")
                counts[name] += 1
                return _result(SearchSuccess(result=authority.search_code(query)))
            raise ToolError("mcp_schema", "unknown MCP tool")
        except ToolError as exc:
            return _error(exc)
        except Exception:
            return _error(ToolError("mcp_lifecycle", "MCP tool execution failed"))

    return server


def create_test_server(
    workspace: Workspace | None = None, test_backend: DirectToolBackend | None = None
) -> Server[Any]:
    authority = workspace or _workspace()
    backend = test_backend or DirectToolBackend(authority, DockerTestRunner(docker=_docker()))
    server: Server[Any] = Server(TEST_SERVER_NAME, version=MCP_SERVER_VERSION)
    called = False
    tools = [
        types.Tool(
            name="run_tests",
            description="Run the fixed isolated Python unittest profile once.",
            inputSchema=TEST_INPUT_SCHEMA,
            outputSchema=output_schema_for("run_tests"),
            annotations=SAFE_ANNOTATIONS,
        )
    ]

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        nonlocal called
        try:
            if name != "run_tests":
                raise ToolError("mcp_schema", "unknown MCP tool")
            profile = _argument(arguments, "profile", 18)
            if profile != "python-unittest-v1":
                raise ToolError("mcp_schema", "MCP tool arguments are invalid")
            if called:
                raise ToolError("limit_exceeded", "run_tests session limit exceeded")
            called = True
            return _result(TestSuccess(result=backend.run_tests(profile)))
        except ToolError as exc:
            return _error(exc)
        except Exception:
            return _error(ToolError("mcp_lifecycle", "MCP tool execution failed"))

    return server


async def run_stdio_server(server: Server[Any], name: str) -> None:
    options = InitializationOptions(
        server_name=name,
        server_version=MCP_SERVER_VERSION,
        capabilities=types.ServerCapabilities(tools=types.ToolsCapability(listChanged=False)),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


def run_repo_server() -> None:
    anyio.run(run_stdio_server, create_repo_server(), REPO_SERVER_NAME)


def run_test_server() -> None:
    anyio.run(run_stdio_server, create_test_server(), TEST_SERVER_NAME)
