"""Stable benchmark matrix loading and task selection."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from codereviewops.benchmark_models import BenchmarkMatrixV1, BenchmarkSuiteV1, TaskEntryV1
from codereviewops.io import InputError
from codereviewops.models import Category, Difficulty

DEFAULT_SUITE = Path("benchmarks/tasks/suites/m4_25.json")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"could not read valid benchmark JSON: {path}") from exc


def load_suite(path: Path) -> BenchmarkSuiteV1:
    try:
        return BenchmarkSuiteV1.model_validate(_load_json(path))
    except ValidationError as exc:
        raise InputError("benchmark suite failed schema validation") from exc


def load_matrix(path: Path) -> BenchmarkMatrixV1:
    try:
        return BenchmarkMatrixV1.model_validate(_load_json(path))
    except ValidationError as exc:
        raise InputError("benchmark matrix failed schema validation") from exc


def resolve_confined(base: Path, reference: str, label: str) -> Path:
    try:
        root = base.resolve(strict=True)
        candidate = (root / reference).resolve(strict=True)
    except OSError as exc:
        raise InputError(f"{label} is missing or inaccessible") from exc
    if not candidate.is_relative_to(root):
        raise InputError(f"{label} escapes its owning directory")
    return candidate


def select_tasks(
    suite: BenchmarkSuiteV1,
    *,
    task_ids: set[str] | None = None,
    categories: set[Category] | None = None,
    difficulties: set[Difficulty] | None = None,
    polarity: str | None = None,
) -> list[TaskEntryV1]:
    if polarity not in {None, "positive", "negative"}:
        raise InputError("polarity must be positive or negative")
    selected = [
        entry
        for entry in suite.tasks
        if (not task_ids or entry.task_id in task_ids)
        and (not categories or entry.primary_category in categories)
        and (not difficulties or entry.difficulty in difficulties)
        and (polarity is None or entry.negative == (polarity == "negative"))
    ]
    if task_ids:
        unknown = task_ids - {entry.task_id for entry in suite.tasks}
        if unknown:
            raise InputError("unknown task filter")
    if not selected:
        raise InputError("benchmark filters selected no tasks")
    return selected
