"""Shared provider and artifact contract constants."""

from __future__ import annotations

import re

TOOL_PROMPT_VERSION = "review-tools-v1"
PROMPT_VERSION = "review-v1"
LIVE_STRUCTURED_OUTPUT_MODE = "json_schema_strict"
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
LIVE_PROVIDERS = frozenset({"groq", "mistral"})
ARTIFACT_PROVIDERS = LIVE_PROVIDERS | {"replay"}


def is_valid_model_identifier(value: str) -> bool:
    """Return whether a model identifier is safe to retain in metadata."""

    return MODEL_PATTERN.fullmatch(value) is not None
