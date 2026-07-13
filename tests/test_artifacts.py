from __future__ import annotations

import os
from pathlib import Path

import pytest

import codereviewops.artifacts as artifact_module
from codereviewops.artifacts import ARTIFACT_LOCK_NAME, OutputError, write_artifacts
from codereviewops.workflow import run_task

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "tasks" / "http_retry_001.json"


def _benchmark_artifact():
    return run_task(BENCHMARK, "replay")


def _create_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    assert link.is_symlink()


def test_existing_cooperative_lock_does_not_clobber_destinations(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    run_path = output / "run.json"
    report_path = output / "report.md"
    lock_path = output / ARTIFACT_LOCK_NAME
    run_path.write_bytes(b"existing run")
    report_path.write_bytes(b"existing report")
    lock_path.write_bytes(b"other writer")

    task, artifact = _benchmark_artifact()
    with pytest.raises(OutputError, match="locked"):
        write_artifacts(output, task, artifact, overwrite=True)

    assert run_path.read_bytes() == b"existing run"
    assert report_path.read_bytes() == b"existing report"
    assert lock_path.read_bytes() == b"other writer"
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


def test_second_install_failure_restores_pair_and_cleans_transaction_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    task, artifact = _benchmark_artifact()
    run_path, report_path = write_artifacts(
        output,
        task,
        artifact,
        overwrite=False,
    )
    previous_run = run_path.read_bytes()
    previous_report = report_path.read_bytes()
    real_replace = os.replace

    def fail_report_install(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == report_path
            and source_path.suffix == ".tmp"
            and source_path.name.startswith(".report.md.")
        ):
            raise OSError("injected second destination failure")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", fail_report_install)

    with pytest.raises(OutputError, match="could not commit output artifacts"):
        write_artifacts(output, task, artifact, overwrite=True)

    assert run_path.read_bytes() == previous_run
    assert report_path.read_bytes() == previous_report
    assert not (output / ARTIFACT_LOCK_NAME).exists()
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


def test_no_overwrite_refuses_and_preserves_dangling_symlink(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    run_path = output / "run.json"
    report_path = output / "report.md"
    missing_target = output / "missing-target.json"
    _create_symlink_or_skip(run_path, missing_target)
    original_link_target = os.readlink(run_path)
    report_path.write_bytes(b"existing report")
    task, artifact = _benchmark_artifact()

    with pytest.raises(OutputError, match="output exists"):
        write_artifacts(output, task, artifact, overwrite=False)

    assert os.path.lexists(run_path)
    assert run_path.is_symlink()
    assert os.readlink(run_path) == original_link_target
    assert report_path.read_bytes() == b"existing report"
    assert not (output / ARTIFACT_LOCK_NAME).exists()
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


def test_failed_install_restores_dangling_symlink_and_other_prior_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    run_path = output / "run.json"
    report_path = output / "report.md"
    missing_target = output / "missing-target.json"
    _create_symlink_or_skip(run_path, missing_target)
    original_link_target = os.readlink(run_path)
    previous_report = b"existing report bytes"
    report_path.write_bytes(previous_report)
    real_replace = os.replace

    def fail_report_install(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == report_path
            and source_path.suffix == ".tmp"
            and source_path.name.startswith(".report.md.")
        ):
            raise OSError("injected second destination failure")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", fail_report_install)
    task, artifact = _benchmark_artifact()

    with pytest.raises(OutputError, match="could not commit output artifacts"):
        write_artifacts(output, task, artifact, overwrite=True)

    assert os.path.lexists(run_path)
    assert run_path.is_symlink()
    assert os.readlink(run_path) == original_link_target
    assert report_path.read_bytes() == previous_report
    assert not (output / ARTIFACT_LOCK_NAME).exists()
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []


def test_directory_destination_rejected_before_any_mutation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    run_path = output / "run.json"
    report_path = output / "report.md"
    run_path.mkdir()
    report_path.write_bytes(b"existing report")
    task, artifact = _benchmark_artifact()

    with pytest.raises(OutputError, match="directory entry"):
        write_artifacts(output, task, artifact, overwrite=True)

    assert run_path.is_dir()
    assert list(run_path.iterdir()) == []
    assert report_path.read_bytes() == b"existing report"
    assert not (output / ARTIFACT_LOCK_NAME).exists()
    assert list(output.glob(".*.tmp")) == []
    assert list(output.glob(".*.bak")) == []
