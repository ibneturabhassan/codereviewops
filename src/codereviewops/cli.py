"""Command-line interface for CodeReviewOps."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codereviewops.artifacts import write_artifacts
from codereviewops.workflow import run_task

app = typer.Typer(
    help="Evaluate deterministic code-review benchmark tasks.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run CodeReviewOps commands."""


@app.command()
def review(
    task: Annotated[
        Path,
        typer.Option("--task", help="Path to a benchmark task manifest."),
    ],
    provider: Annotated[
        str,
        typer.Option("--provider", help="Review provider (Milestone 1: replay)."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for run.json and report.md."),
    ],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing output artifacts."),
    ] = False,
) -> None:
    """Run one benchmark review and write its evaluation artifacts."""

    try:
        benchmark, artifact = run_task(task, provider)
        run_path, report_path = write_artifacts(
            output_dir, benchmark, artifact, overwrite=overwrite
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None

    typer.echo(f"wrote {run_path}")
    typer.echo(f"wrote {report_path}")
    if not artifact.evaluation.task_success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
