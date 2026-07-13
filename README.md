# CodeReviewOps

CodeReviewOps is a local, deterministic harness for measuring whether a code-review
agent finds expected problems in a pull request without inventing unsupported ones.
Milestone 1 uses synthetic fixtures and a replay provider so evaluation is reproducible,
auditable, and does not require API keys or network access. Milestone 2 adds direct,
single-request Groq and Mistral inference for small opt-in evaluations; free-tier limits
and model availability are controlled by those providers.

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

### Deterministic replay

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

### Live inference

Live runs require an explicit model and the provider's API key. The CLI reads only
`GROQ_API_KEY` or `MISTRAL_API_KEY`; do not commit keys or place them in benchmark
files. For example, in Command Prompt:

    set GROQ_API_KEY=your-key
    uv run codereviewops review --task benchmarks/tasks/http_retry_001.json --provider groq --model your-model --output-dir artifacts/groq

Or use Mistral:

    set MISTRAL_API_KEY=your-key
    uv run codereviewops review --task benchmarks/tasks/http_retry_001.json --provider mistral --model your-model --output-dir artifacts/mistral

Each live run makes exactly one HTTPS request to the selected provider. It does not
retry, follow redirects, fall back to another provider, or substitute another model.
Provider failures are reported with safe error categories without including keys or
arbitrary response text.

The optional live smoke test is skipped unless explicitly enabled and configured:

    set CODEREVIEWOPS_RUN_LIVE=1
    set CODEREVIEWOPS_LIVE_PROVIDER=groq
    set CODEREVIEWOPS_LIVE_MODEL=your-model
    uv run pytest -m live

## Development

    uv run ruff check .
    uv run mypy src
    uv run pytest -m "not live"

## Current scope and limitations

The project intentionally has no GitHub integration, repository tools, database, API,
or dashboard. Replay output proves the evaluation boundary deterministically. Live
inference measures a selected hosted model but remains a bounded, one-request workflow.

Future milestones will add tool-using agents, MCP integrations, a larger benchmark
suite, model comparisons, and a dashboard.
