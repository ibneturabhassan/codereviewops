"""Milestone 1 review workflow."""

from __future__ import annotations

from pathlib import Path

from codereviewops.evaluation import evaluate_review
from codereviewops.io import InputError, LoadedTask, load_task
from codereviewops.models import BenchmarkTask, ReviewContext, RunArtifact
from codereviewops.providers import ReplayProvider, ReviewProvider


def run_loaded_task(
    loaded: LoadedTask, provider: ReviewProvider, provider_name: str = "replay"
) -> tuple[BenchmarkTask, RunArtifact]:
    """Run an already-loaded task through a provider and evaluator."""

    try:
        diff_text = loaded.diff_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InputError(f"could not read diff {loaded.diff_path}: {exc}") from exc
    context = ReviewContext(
        schema_version="1.0",
        task_id=loaded.task.task_id,
        title=loaded.task.title,
        issue_description=loaded.task.issue_description,
        diff_text=diff_text,
    )
    review = provider.review(context)
    evaluation = evaluate_review(loaded.task.expected_findings, loaded.task.must_not_find, review)
    return loaded.task, RunArtifact(
        schema_version="1.0",
        task_id=loaded.task.task_id,
        provider=provider_name,
        review=review,
        evaluation=evaluation,
    )


def run_task(task_path: Path, provider_name: str) -> tuple[BenchmarkTask, RunArtifact]:
    """Load and run a task with the selected Milestone 1 provider."""

    if provider_name != "replay":
        raise InputError(f"unsupported provider: {provider_name}; expected replay")
    loaded = load_task(task_path)
    return run_loaded_task(loaded, ReplayProvider(loaded.replay_path), provider_name)
