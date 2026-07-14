"""Command-line interface for CodeReviewOps."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from codereviewops.artifacts import write_artifacts
from codereviewops.benchmarking import BenchmarkError, generate, validate
from codereviewops.docker_runner import DockerTestRunner
from codereviewops.io import InputError
from codereviewops.models import WorkflowState
from codereviewops.tools import ToolError
from codereviewops.workflow import run_task

app = typer.Typer(
    help="Evaluate deterministic code-review benchmark tasks.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run CodeReviewOps commands."""


benchmark_app = typer.Typer(
    help="Generate and validate deterministic benchmark suites.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("generate")
def benchmark_generate(
    source: Annotated[
        Path,
        typer.Option("--source", help="Directory containing human-authored source cases."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Managed benchmark output root."),
    ],
    check: Annotated[
        bool,
        typer.Option("--check", help="Fail if tracked generated outputs are stale."),
    ] = False,
) -> None:
    """Generate artifacts or verify byte-for-byte reproducibility."""

    try:
        generate(source, output_root, check=check)
    except BenchmarkError:
        typer.echo("error: benchmark generation failed", err=True)
        raise typer.Exit(code=2) from None
    typer.echo("benchmark outputs are current" if check else "generated benchmark outputs")


@benchmark_app.command("validate")
def benchmark_validate(
    suite: Annotated[
        Path,
        typer.Option("--suite", help="Path to a generated benchmark suite manifest."),
    ],
) -> None:
    """Validate inventory, distributions, provenance, and direct replays."""

    try:
        validate(suite)
    except (BenchmarkError, InputError):
        typer.echo("error: benchmark validation failed", err=True)
        raise typer.Exit(code=2) from None
    typer.echo("benchmark suite is valid")


@app.command("tools-check")
def tools_check() -> None:
    """Check whether the immutable Docker tool runner is ready."""

    try:
        image_id = DockerTestRunner().check()
    except ToolError as exc:
        typer.echo(f"tools unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"tools ready: {image_id}")


@app.command()
def review(
    task: Annotated[
        Path,
        typer.Option("--task", help="Path to a benchmark task manifest."),
    ],
    provider: Annotated[
        str,
        typer.Option("--provider", help="Review provider: replay, groq, or mistral."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for run.json and report.md."),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Required model identifier for live providers."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing output artifacts."),
    ] = False,
    tool_transport: Annotated[
        Literal["direct", "mcp-stdio"],
        typer.Option("--tool-transport", help="Tool transport: direct or mcp-stdio."),
    ] = "direct",
) -> None:
    """Run one benchmark review and write its evaluation artifacts."""

    try:
        benchmark, artifact = run_task(task, provider, model, tool_transport=tool_transport)
        run_path, report_path = write_artifacts(
            output_dir, benchmark, artifact, overwrite=overwrite
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None

    typer.echo(f"wrote {run_path}")
    typer.echo(f"wrote {report_path}")
    if artifact.final_state == WorkflowState.FAILED:
        typer.echo(f"workflow failed: {artifact.failure_code}", err=True)
        raise typer.Exit(code=2)

    if not artifact.evaluation.task_success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
