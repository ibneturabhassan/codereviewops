"""Canonical, immutable MCP protocol and server manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Literal

type McpProtocolVersion = Literal["2025-11-25"]
MCP_PROTOCOL_VERSION: Final[McpProtocolVersion] = "2025-11-25"
MCP_SERVER_VERSION: Final[Literal["0.1.0"]] = "0.1.0"
REPO_SERVER_NAME = "codereviewops-repo-mcp"
TEST_SERVER_NAME = "codereviewops-test-mcp"
type ServerKind = Literal["repo", "test"]

WORKSPACE_ENV = "CODEREVIEWOPS_MCP_WORKSPACE"
BENCHMARK_ROOT_ENV = "CODEREVIEWOPS_MCP_BENCHMARK_ROOT"
DOCKER_ENV = "CODEREVIEWOPS_MCP_DOCKER"

SAFE_ANNOTATIONS_JSON: dict[str, bool] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
CAPABILITIES_JSON: dict[str, Any] = {"tools": {"listChanged": False}}

READ_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 512}},
    "required": ["path"],
    "additionalProperties": False,
}
SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 128}},
    "required": ["query"],
    "additionalProperties": False,
}
TEST_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"profile": {"const": "python-unittest-v1", "type": "string"}},
    "required": ["profile"],
    "additionalProperties": False,
}


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


CAPABILITY_FINGERPRINT = canonical_fingerprint(CAPABILITIES_JSON)


def expected_server_name(kind: ServerKind) -> str:
    return REPO_SERVER_NAME if kind == "repo" else TEST_SERVER_NAME


def expected_tools(kind: ServerKind) -> list[dict[str, Any]]:
    from codereviewops.mcp_typed import output_schema_for

    definitions = (
        [("read_file", READ_INPUT_SCHEMA), ("search_code", SEARCH_INPUT_SCHEMA)]
        if kind == "repo"
        else [("run_tests", TEST_INPUT_SCHEMA)]
    )
    return [
        {
            "name": name,
            "inputSchema": input_schema,
            "outputSchema": output_schema_for(name),
            "annotations": SAFE_ANNOTATIONS_JSON,
        }
        for name, input_schema in definitions
    ]


def expected_schema_fingerprint(kind: ServerKind) -> str:
    return canonical_fingerprint(expected_tools(kind))
