"""Typed MCP tool envelopes and exact per-tool output schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from codereviewops.models import ReadFileResult, RunTestsResult, SearchCodeResult
from codereviewops.tool_contracts import ToolErrorCode


class McpToolError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: ToolErrorCode
    message: Annotated[
        str, Field(min_length=1, max_length=256, pattern=r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*$")
    ]
    retryable: bool = False


class ReadSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    ok: Literal[True] = True
    result: ReadFileResult
    error: None = None


class SearchSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    ok: Literal[True] = True
    result: SearchCodeResult
    error: None = None


class TestSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    ok: Literal[True] = True
    result: RunTestsResult
    error: None = None


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    ok: Literal[False] = False
    result: None = None
    error: McpToolError


type ReadEnvelope = ReadSuccess | ErrorEnvelope
type SearchEnvelope = SearchSuccess | ErrorEnvelope
type TestEnvelope = TestSuccess | ErrorEnvelope

ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "read_file": TypeAdapter(ReadEnvelope),
    "search_code": TypeAdapter(SearchEnvelope),
    "run_tests": TypeAdapter(TestEnvelope),
}


def output_schema_for(name: str) -> dict[str, Any]:
    return ADAPTERS[name].json_schema()


def parse_envelope(name: str, value: object) -> ReadEnvelope | SearchEnvelope | TestEnvelope:
    return ADAPTERS[name].validate_python(value)  # type: ignore[no-any-return]
