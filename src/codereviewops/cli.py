"""Command-line interface for CodeReviewOps."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codereviewops.artifacts import write_artifacts
from codereviewops.docker_runner import DockerTestRunner
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
) -> None:
    """Run one benchmark review and write its evaluation artifacts."""

    try:
        benchmark, artifact = run_task(task, provider, model)
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
