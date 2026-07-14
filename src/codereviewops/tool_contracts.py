"""Leaf error contracts shared by direct and MCP tool transports."""

from __future__ import annotations

from typing import Literal

type ToolErrorCode = Literal[
    "invalid_path",
    "changed_workspace",
    "read_failed",
    "invalid_utf8",
    "binary_file",
    "limit_exceeded",
    "docker_unavailable",
    "docker_infrastructure",
    "cleanup_failed",
    "unsupported_profile",
    "invalid_diff",
    "diff_no_added_lines",
    "mcp_unavailable",
    "mcp_protocol",
    "mcp_schema",
    "mcp_timeout",
    "mcp_lifecycle",
    "unsupported_text",
]


class ToolError(ValueError):
    """Safe categorized failure that cannot carry arbitrary child output."""

    def __init__(self, code: ToolErrorCode, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(message)
