"""Local benchmark workflow with replay or one-shot live inference."""

from __future__ import annotations

import os
from pathlib import Path

from codereviewops.evaluation import evaluate_review
from codereviewops.io import InputError, LoadedTask, load_task
from codereviewops.models import BenchmarkTask, ReviewContext, RunArtifact
from codereviewops.providers import (
    GroqProvider,
    MistralProvider,
    ReplayProvider,
    ReviewProvider,
)


def run_loaded_task(
    loaded: LoadedTask, provider: ReviewProvider, provider_name: str = "replay"
) -> tuple[BenchmarkTask, RunArtifact]:
    """Run an already-loaded task through one provider and the evaluator."""

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
    provider_result = provider.review(context)
    review = provider_result.report
    evaluation = evaluate_review(loaded.task.expected_findings, loaded.task.must_not_find, review)
    return loaded.task, RunArtifact(
        schema_version="1.1",
        task_id=loaded.task.task_id,
        provider=provider_name,
        review=review,
        evaluation=evaluation,
        requested_model=provider_result.requested_model,
        response_model=provider_result.response_model,
        prompt_version=provider_result.prompt_version,
        structured_output_mode=provider_result.structured_output_mode,
        latency_ms=provider_result.latency_ms,
        usage=provider_result.usage,
    )


def run_task(
    task_path: Path,
    provider_name: str,
    model: str | None = None,
) -> tuple[BenchmarkTask, RunArtifact]:
    """Load and run one task without retries, fallbacks, or model substitution."""

    if provider_name == "replay":
        if model is not None:
            raise InputError("--model is forbidden for the replay provider")
        loaded = load_task(task_path)
        provider: ReviewProvider = ReplayProvider(loaded.replay_path)
    elif provider_name in {"groq", "mistral"}:
        if model is None:
            raise InputError(f"--model is required for the {provider_name} provider")
        loaded = load_task(task_path)
        if provider_name == "groq":
            provider = GroqProvider(model=model, api_key=os.environ.get("GROQ_API_KEY", ""))
        else:
            provider = MistralProvider(
                model=model,
                api_key=os.environ.get("MISTRAL_API_KEY", ""),
            )
    else:
        raise InputError("unsupported provider; expected replay, groq, or mistral")
    return run_loaded_task(loaded, provider, provider_name)
