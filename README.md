# CodeReviewOps

CodeReviewOps is a local, deterministic harness for measuring whether a code-review
agent finds expected problems in a pull request without inventing unsupported ones.
Milestone 1 uses synthetic fixtures and a replay provider so evaluation is reproducible,
auditable, and does not require API keys or network access. Milestone 2 adds direct,
single-request Groq and Mistral inference for small opt-in evaluations; free-tier limits
and model availability are controlled by those providers. Milestone 3 adds a bounded,
manifest-driven tool workflow for deterministic workspace reads, literal code searches,
and an opt-in isolated Python unittest profile. Milestone 4 phases 1-2 add two
local, stdio-only MCP servers and a transport adapter while preserving the same bounded tool
semantics.

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

### Bounded tools

Tool-enabled benchmark schema 1.1 declares every allowed file read, literal search,
and test profile up front. Workspace paths must remain beneath the benchmark directory;
absolute paths, traversal, links, reparse points, special files, `.git`, binary content,
and oversized workspaces are rejected. Tool traces and workflow transitions are recorded
in schema 1.2 artifacts without exposing host paths.

Build the pinned unittest runner and verify that it is available:

    docker build --pull=false -f runner/Dockerfile -t codereviewops/python-unittest:0.1.0 .
    uv run codereviewops tools-check

Then run the deterministic tool benchmark:

    uv run codereviewops review --task benchmarks/tasks/python_tools_001.json --provider replay --output-dir artifacts/python_tools_001

### Local MCP transport

Tool-enabled schema 1.1 tasks can use the same repository and test tools through the
local MCP stdio transport:

    uv run codereviewops review --task benchmarks/tasks/python_tools_001.json --provider replay --tool-transport mcp-stdio --output-dir artifacts/python_tools_001_mcp

The default remains `--tool-transport direct`. MCP mode launches only the fixed
`codereviewops-repo-mcp` and `codereviewops-test-mcp` Python modules with workspace
authority supplied by the parent process. Benchmark inputs cannot choose commands,
working directories, environment variables, roots, or images. The servers expose only
`read_file`, `search_code`, and the fixed `run_tests` profile over stdio; they advertise
no network, resource, prompt, or subscription capabilities.

MCP runs emit schema 1.3 artifacts with the negotiated protocol, exact server identity
and schema fingerprints, completed lifecycle records, and a latency-neutral semantic
trace fingerprint. Direct runs continue to emit schema 1.2 artifacts. Schema 1.0 tasks
reject MCP mode.
The runner accepts only the fixed `python-unittest-v1` profile. It starts Docker with no
network, no added capabilities, no new privileges, a read-only root filesystem and
workspace mount, resource limits, a non-root user, and a 30-second deadline. CodeReviewOps
never sends benchmark-controlled shell commands to the host or container, and it does not
fall back to host execution when Docker is unavailable.

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
    uv run pytest tests/test_mcp.py
    uv run pytest -m "not live and not docker"

## Current scope and limitations

The project intentionally has no GitHub integration, arbitrary shell tool, database,
API, or dashboard. Replay output proves the evaluation boundary deterministically. Live
inference measures a selected hosted model but remains a bounded, one-request workflow;
tool context is gathered before that single request.

Future milestones will expand the benchmark suite, model comparisons, and dashboard work.
The current MCP scope is deliberately local and stdio-only.
