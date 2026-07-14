"""Repository-owned, process-observable MCP stdio transport."""

from __future__ import annotations

import json
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import types

from codereviewops.process_control import ProcessLease, ProcessTreeController
from codereviewops.tool_contracts import ToolError

MAX_JSON_LINE = 1024 * 1024
MAX_STDERR = 64 * 1024
GRACEFUL_EXIT_SECONDS = 1.0


@dataclass
class OwnedSessionMessage:
    """Structural equivalent accepted by the public ClientSession streams."""

    message: types.JSONRPCMessage
    metadata: object | None = None


class OwnedStdioTransport:
    """Own the child Popen, protocol pipes, drain workers, and process-tree cleanup."""

    def __init__(
        self,
        executable: Path,
        module: str,
        environment: dict[str, str],
        *,
        controller: ProcessTreeController | None = None,
    ) -> None:
        if not executable.is_absolute() or not executable.is_file():
            raise ToolError("mcp_unavailable", "MCP Python executable is invalid")
        kinds = {
            "codereviewops.mcp_repo_server": "repo",
            "codereviewops.mcp_test_server": "test",
        }
        if module not in kinds:
            raise ToolError("mcp_unavailable", "MCP server module is invalid")
        self.executable = executable
        self.module = module
        self.kind: Literal["repo", "test"] = (
            "repo" if module == "codereviewops.mcp_repo_server" else "test"
        )
        self.environment = dict(environment)
        self.controller = controller or ProcessTreeController()
        try:
            self.lease: ProcessLease = self.controller.prepare()
        except (OSError, ValueError) as exc:
            raise ToolError("mcp_unavailable", "MCP process ownership is unavailable") from exc
        self.process: subprocess.Popen[bytes] | None = None
        self.pid: int | None = None
        self.stderr = bytearray()
        self.stderr_truncated = False
        self._closed = False
        self._cleanup_ok = True
        self._task_group: Any = None
        self._client_receive: MemoryObjectReceiveStream[Any] | None = None
        self._client_send: MemoryObjectSendStream[Any] | None = None
        self._reader_send: MemoryObjectSendStream[Any] | None = None
        self._writer_receive: MemoryObjectReceiveStream[Any] | None = None

    async def __aenter__(
        self,
    ) -> tuple[MemoryObjectReceiveStream[Any], MemoryObjectSendStream[Any]]:
        try:
            process: subprocess.Popen[bytes] = subprocess.Popen(
                self.lease.command(self.executable, self.module, self.kind),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.environment,
                shell=False,
                bufsize=0,
                **self.lease.popen_kwargs(),
            )
        except OSError as exc:
            self.lease.close()
            raise ToolError("mcp_unavailable", "MCP server could not start") from exc
        self.process = process
        if not self.lease.attach(process):
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with suppress(OSError):
                        stream.close()
            self.lease.close()
            raise ToolError("mcp_lifecycle", "MCP process ownership failed")
        self.pid = process.pid
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            await self.aclose()
            raise ToolError("mcp_lifecycle", "MCP server pipes are unavailable")
        self._reader_send, self._client_receive = anyio.create_memory_object_stream(0)
        self._client_send, self._writer_receive = anyio.create_memory_object_stream(0)
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._read_stdout)
        self._task_group.start_soon(self._write_stdin)
        self._task_group.start_soon(self._drain_stderr)
        return self._client_receive, self._client_send

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._reader_send is not None
        try:
            async with self._reader_send:
                while True:
                    line = await anyio.to_thread.run_sync(
                        self.process.stdout.readline,
                        MAX_JSON_LINE + 1,
                        abandon_on_cancel=True,
                    )
                    if not line:
                        await self._reader_send.send(EOFError("MCP server closed stdout"))
                        return
                    if len(line) > MAX_JSON_LINE or not line.endswith(b"\n"):
                        await self._reader_send.send(ValueError("MCP JSON-RPC line is invalid"))
                        return
                    try:
                        payload = json.loads(line.decode("utf-8"))
                        message = types.JSONRPCMessage.model_validate(payload)
                    except (UnicodeError, ValueError):
                        await self._reader_send.send(ValueError("MCP JSON-RPC message is invalid"))
                        return
                    await self._reader_send.send(OwnedSessionMessage(message))
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    async def _write_stdin(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        assert self._writer_receive is not None
        try:
            async with self._writer_receive:
                async for session_message in self._writer_receive:
                    message = getattr(session_message, "message", None)
                    if not isinstance(message, types.JSONRPCMessage):
                        raise ValueError("MCP client message is invalid")
                    encoded = (
                        message.model_dump_json(by_alias=True, exclude_none=True) + "\n"
                    ).encode("utf-8")
                    if len(encoded) > MAX_JSON_LINE:
                        raise ValueError("MCP client message exceeds the line limit")
                    await anyio.to_thread.run_sync(
                        self._write_bytes, encoded, abandon_on_cancel=True
                    )
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    def _write_bytes(self, value: bytes) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(value)
        self.process.stdin.flush()

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            chunk = await anyio.to_thread.run_sync(
                self.process.stderr.read, 8192, abandon_on_cancel=True
            )
            if not chunk:
                return
            remaining = max(0, MAX_STDERR - len(self.stderr))
            self.stderr.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.stderr_truncated = True

    async def aclose(self) -> None:
        if self._closed:
            if not self._cleanup_ok:
                raise ToolError("mcp_lifecycle", "MCP process cleanup failed")
            return
        self._closed = True
        process = self.process
        if self._client_send is not None:
            try:
                await self._client_send.aclose()
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                self._cleanup_ok = False
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                self._cleanup_ok = False
        if process is not None:
            try:
                await anyio.to_thread.run_sync(
                    process.wait,
                    GRACEFUL_EXIT_SECONDS,
                    abandon_on_cancel=True,
                )
            except subprocess.TimeoutExpired:
                pass
            except OSError:
                self._cleanup_ok = False
            ownership_clean = await anyio.to_thread.run_sync(
                self.lease.terminate_and_verify,
                abandon_on_cancel=True,
            )
            self._cleanup_ok = ownership_clean and self._cleanup_ok
            self.lease.close()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        self._cleanup_ok = False
        if self._task_group is not None:
            self._task_group.cancel_scope.cancel()
            await self._task_group.__aexit__(None, None, None)
        if process is not None and process.poll() is None:
            self._cleanup_ok = False
        if not self._cleanup_ok:
            raise ToolError("mcp_lifecycle", "MCP process cleanup failed")

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()
