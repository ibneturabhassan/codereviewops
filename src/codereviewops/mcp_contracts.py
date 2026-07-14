"""Compatibility exports for fixed MCP server and typed-envelope contracts."""

from codereviewops.mcp_manifest import (
    BENCHMARK_ROOT_ENV,
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_VERSION,
    READ_INPUT_SCHEMA,
    REPO_SERVER_NAME,
    SEARCH_INPUT_SCHEMA,
    TEST_INPUT_SCHEMA,
    TEST_SERVER_NAME,
    WORKSPACE_ENV,
)
from codereviewops.mcp_servers import create_repo_server, create_test_server
from codereviewops.mcp_typed import ErrorEnvelope, McpToolError, output_schema_for, parse_envelope

__all__ = [
    "BENCHMARK_ROOT_ENV",
    "MCP_PROTOCOL_VERSION",
    "MCP_SERVER_VERSION",
    "READ_INPUT_SCHEMA",
    "REPO_SERVER_NAME",
    "SEARCH_INPUT_SCHEMA",
    "TEST_INPUT_SCHEMA",
    "TEST_SERVER_NAME",
    "WORKSPACE_ENV",
    "ErrorEnvelope",
    "McpToolError",
    "create_repo_server",
    "create_test_server",
    "output_schema_for",
    "parse_envelope",
]
