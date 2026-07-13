"""Artifact rendering and atomic output."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from uuid import uuid4

from codereviewops.models import BenchmarkTask, RunArtifact


class OutputError(ValueError):
    """Raised when run artifacts cannot be written safely."""


def render_markdown(task: BenchmarkTask, artifact: RunArtifact) -> str:
    """Render a human-readable evaluation report."""

    review = artifact.review
    evaluation = artifact.evaluation
    lines = [
        f"# Review Report: {task.title}",
        "",
        f"- Task: {task.task_id}",
        f"- Provider: {artifact.provider}",
        f"- Assessment: **{review.overall_assessment.value}**",
        f"- Task success: **{str(evaluation.task_success).lower()}**",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision | {evaluation.precision:.3f} |",
        f"| Recall | {evaluation.recall:.3f} |",
        f"| Hallucination rate | {evaluation.hallucination_rate:.3f} |",
        "",
        "## Summary",
        "",
        review.summary,
        "",
        "## Findings",
        "",
    ]
    if review.findings:
        for index, finding in enumerate(review.findings):
            lines.extend(
                [
                    f"### {index}. {finding.title}",
                    "",
                    (
                        f"- {finding.severity.value} / {finding.category.value} / "
                        f"{finding.file}:{finding.line_start}-{finding.line_end}"
                    ),
                    f"- Confidence: {finding.confidence:.2f}",
                    f"- Evidence: {finding.evidence}",
                    f"- Reasoning: {finding.reasoning}",
                    f"- Recommendation: {finding.recommendation}",
                    "",
                ]
            )
    else:
        lines.extend(["No findings.", ""])

    lines.extend(["## Missed expected findings", ""])
    if evaluation.missed_expected_indices:
        for index in evaluation.missed_expected_indices:
            expected = task.expected_findings[index]
            lines.append(
                f"- [{index}] {expected.category.value} in "
                f"{expected.file}:{expected.line_start}-{expected.line_end}: "
                f"{expected.description}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Hallucinated findings", ""])
    if evaluation.hallucinated_actual_indices:
        for index in evaluation.hallucinated_actual_indices:
            lines.append(f"- [{index}] {review.findings[index].title}")
    else:
        lines.append("- None")

    lines.extend(["", "## Prohibited phrase hits", ""])
    if evaluation.prohibited_hits:
        for hit in evaluation.prohibited_hits:
            lines.append(f"- Finding [{hit.actual_index}]: {hit.phrase}")
    else:
        lines.append("- None")

    lines.extend(["", "## Tests", ""])
    if review.tests_run:
        for test in review.tests_run:
            lines.append(f"- {test.command} - **{test.status.value}**: {test.summary}")
    else:
        lines.append("- None reported")

    lines.extend(["", "## Limitations", ""])
    if review.limitations:
        lines.extend(f"- {limitation}" for limitation in review.limitations)
    else:
        lines.append("- None reported")
    return "\n".join(lines) + "\n"


ARTIFACT_LOCK_NAME = ".codereviewops.lock"


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_directory_entry(path: Path, mode: int) -> bool:
    if stat.S_ISDIR(mode) or path.is_junction():
        return True
    if not stat.S_ISLNK(mode):
        return False
    try:
        target_mode = os.stat(path).st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(target_mode)


def _validate_destination_entries(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not _lexists(path):
            continue
        try:
            mode = os.lstat(path).st_mode
            is_directory = _is_directory_entry(path, mode)
        except OSError as exc:
            raise OutputError(f"could not inspect destination {path.name}: {exc}") from exc
        if is_directory:
            raise OutputError(f"destination is a directory entry: {path.name}")
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise OutputError(f"destination is not a regular file or file symlink: {path.name}")


def _cleanup_paths(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not _lexists(path):
            continue
        try:
            mode = os.lstat(path).st_mode
            is_directory = _is_directory_entry(path, mode)
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if is_directory:
            errors.append(f"{path.name}: refused to unlink a directory entry")
            continue
        try:
            path.unlink()
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
    return errors


def _rollback_commit(
    installed: list[Path],
    backups: list[tuple[Path, Path]],
) -> list[str]:
    errors: list[str] = []
    backed_destinations = {destination for destination, _ in backups}
    for destination in reversed(installed):
        if destination not in backed_destinations:
            errors.extend(_cleanup_paths([destination]))
    for destination, backup in reversed(backups):
        try:
            os.replace(backup, destination)
        except OSError as exc:
            errors.append(f"restore {destination.name}: {exc}")
    return errors


def _write_artifacts_locked(
    output_dir: Path,
    task: BenchmarkTask,
    artifact: RunArtifact,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    run_path = output_dir / "run.json"
    report_path = output_dir / "report.md"
    destinations = (run_path, report_path)
    _validate_destination_entries(destinations)
    if not overwrite and any(_lexists(path) for path in destinations):
        raise OutputError("output exists; use --overwrite to replace both artifacts")

    json_text = json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    markdown_text = render_markdown(task, artifact)
    token = uuid4().hex
    run_temp = output_dir / f".run.json.{token}.tmp"
    report_temp = output_dir / f".report.md.{token}.tmp"
    run_backup = output_dir / f".run.json.{token}.bak"
    report_backup = output_dir / f".report.md.{token}.bak"
    temporary_paths = [run_temp, report_temp]
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []

    try:
        run_temp.write_text(json_text, encoding="utf-8", newline="\n")
        report_temp.write_text(markdown_text, encoding="utf-8", newline="\n")
        if overwrite:
            for destination, backup in (
                (run_path, run_backup),
                (report_path, report_backup),
            ):
                if _lexists(destination):
                    os.replace(destination, backup)
                    backups.append((destination, backup))
        os.replace(run_temp, run_path)
        installed.append(run_path)
        os.replace(report_temp, report_path)
        installed.append(report_path)
    except OSError as exc:
        rollback_errors = _rollback_commit(installed, backups)
        cleanup_errors = _cleanup_paths(temporary_paths)
        details = rollback_errors + cleanup_errors
        context = f"; rollback cleanup failed: {'; '.join(details)}" if details else ""
        raise OutputError(f"could not commit output artifacts: {exc}{context}") from exc

    cleanup_errors = _cleanup_paths(temporary_paths + [backup for _, backup in backups])
    if cleanup_errors:
        raise OutputError("artifacts written but cleanup failed: " + "; ".join(cleanup_errors))
    return run_path, report_path


def write_artifacts(
    output_dir: Path,
    task: BenchmarkTask,
    artifact: RunArtifact,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Publish the artifact pair under a cooperative exclusive lock."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise OutputError(f"output path is not a directory: {output_dir}")
    except OutputError:
        raise
    except OSError as exc:
        raise OutputError(f"could not prepare output directory: {exc}") from exc

    lock_path = output_dir / ARTIFACT_LOCK_NAME
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise OutputError("output is locked by another writer") from exc
    except OSError as exc:
        raise OutputError(f"could not acquire output lock: {exc}") from exc

    result: tuple[Path, Path] | None = None
    operation_error: Exception | None = None
    lock_close_error: OSError | None = None
    lock_cleanup_error: OSError | None = None
    try:
        try:
            result = _write_artifacts_locked(
                output_dir,
                task,
                artifact,
                overwrite=overwrite,
            )
        except Exception as exc:
            operation_error = exc
    finally:
        try:
            os.close(lock_descriptor)
        except OSError as exc:
            lock_close_error = exc
        try:
            lock_path.unlink(missing_ok=True)
        except OSError as exc:
            lock_cleanup_error = exc

    if operation_error is not None:
        lock_errors = [
            str(error) for error in (lock_close_error, lock_cleanup_error) if error is not None
        ]
        if lock_errors:
            raise OutputError(
                f"{operation_error}; additionally failed to release output lock: "
                + "; ".join(lock_errors)
            ) from operation_error
        raise operation_error
    if lock_close_error is not None:
        raise OutputError(f"failed to close output lock: {lock_close_error}")
    if lock_cleanup_error is not None:
        raise OutputError(f"failed to remove output lock: {lock_cleanup_error}")
    if result is None:
        raise OutputError("artifact publication ended without a result")
    return result
