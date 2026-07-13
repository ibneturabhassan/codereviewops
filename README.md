# CodeReviewOps

CodeReviewOps is a local, deterministic harness for measuring whether a code-review
agent finds expected problems in a pull request without inventing unsupported ones.
Milestone 1 uses synthetic fixtures and a replay provider so evaluation is reproducible,
auditable, and does not require API keys or network access.

## What Milestone 1 does

- Loads an issue, pull-request diff, expected findings, and replay response from a
  versioned benchmark task.
- Gives the provider only review context, never golden labels or prohibited claims.
- Validates structured reviews with strict, versioned Pydantic schemas.
- Matches findings one-to-one by category, normalized file, and overlapping lines.
- Reports misses, hallucinations, and prohibited phrases as JSON and Markdown.
- Confines task references to files beneath the benchmark task directory.
- Limits expected and reported findings to 100 per task to bound matching work.
- Publishes the JSON/Markdown pair with cooperative locking and rollback.

## Setup

Python 3.12 and uv are required.

    uv sync --frozen

Run the synthetic HTTP retry benchmark:

    uv run codereviewops review --task benchmarks/tasks/http_retry_001.json --provider replay --output-dir artifacts/http_retry_001

The command writes run.json and report.md. It refuses to replace either file unless
--overwrite is supplied. Exit status 0 means evaluation passed, 1 means a valid review
failed evaluation, and 2 means the input, provider, or output was invalid.

Artifact publication holds a cooperative exclusive lock while it performs preflight,
temporary writes, backup, and commit. If either destination cannot be installed, prior
files are restored and partial new files are removed. Because the result uses two
filenames, both files cannot become visible at the exact same filesystem instant;
concurrent readers should honor the same lock.

## Development

    uv run ruff check .
    uv run mypy src
    uv run pytest

## Current scope and limitations

This milestone intentionally has no live model provider, GitHub integration, repository
tools, database, API, or dashboard. Replay output proves the evaluation boundary and
workflow deterministically; it does not measure a production model yet.

Future milestones will add tool-using agents, MCP integrations, a larger benchmark
suite, model comparisons, and a dashboard.
