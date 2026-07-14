"""Safe benchmark input loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from codereviewops.models import BenchmarkTask


class InputError(ValueError):
    """Raised when benchmark input cannot be safely loaded."""


@dataclass(frozen=True)
class LoadedTask:
    """Validated task and its safely resolved references."""

    task: BenchmarkTask
    diff_path: Path
    replay_path: Path
    benchmark_root: Path | None = None
    workspace_path: Path | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"could not read valid JSON from {path}: {exc}") from exc


def _resolve_reference(base: Path, reference: str, label: str) -> Path:
    try:
        base_resolved = base.resolve(strict=True)
        candidate = (base_resolved / reference).resolve(strict=True)
    except OSError as exc:
        raise InputError(f"{label} is missing or inaccessible: {reference}") from exc
    if not candidate.is_relative_to(base_resolved):
        raise InputError(f"{label} escapes the benchmark task directory: {reference}")
    if not candidate.is_file():
        raise InputError(f"{label} is not a file: {reference}")
    return candidate


def _resolve_workspace(base: Path, reference: str) -> Path:
    try:
        base_resolved = base.resolve(strict=True)
        lexical = base_resolved / reference
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise InputError(f"workspace is missing or inaccessible: {reference}") from exc
    if not resolved.is_relative_to(base_resolved):
        raise InputError(f"workspace escapes the benchmark task directory: {reference}")
    if not lexical.is_dir():
        raise InputError(f"workspace is not a directory: {reference}")
    return lexical.absolute()


def load_task(manifest_path: Path) -> LoadedTask:
    """Load a task and confine both referenced files beneath its directory."""

    try:
        manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise InputError(f"task manifest is missing or inaccessible: {manifest_path}") from exc
    if not manifest.is_file():
        raise InputError(f"task manifest is not a file: {manifest_path}")
    try:
        task = BenchmarkTask.model_validate(_read_json(manifest))
    except ValidationError as exc:
        raise InputError(f"task manifest failed schema validation: {exc}") from exc
    base = manifest.parent
    return LoadedTask(
        task=task,
        diff_path=_resolve_reference(base, task.diff_path, "diff"),
        replay_path=_resolve_reference(base, task.replay_response_path, "replay response"),
        workspace_path=(
            _resolve_workspace(base, task.workspace_path)
            if task.workspace_path is not None
            else None
        ),
        benchmark_root=base.resolve(strict=True),
    )
