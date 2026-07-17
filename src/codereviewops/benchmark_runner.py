"""Sequential, transactional benchmark comparison runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from codereviewops.artifacts import render_markdown
from codereviewops.benchmark_baseline import (
    canonical_hash,
    compare_baseline,
    load_baseline,
    stable_result,
    stable_result_hash,
    validate_baseline_inputs,
)
from codereviewops.benchmark_metrics import aggregate_metrics, gate_failures, task_metrics
from codereviewops.benchmark_models import (
    BenchmarkBaselineV1,
    BenchmarkMatrixV1,
    BenchmarkRunV1,
    BenchmarkSuiteV1,
    ComparisonResultV1,
    RunVariantV1,
    TaskEntryV1,
    ThresholdProfileV1,
    VariantResultV1,
)
from codereviewops.benchmark_selection import load_matrix, select_tasks
from codereviewops.contracts import TOOL_PROMPT_VERSION
from codereviewops.io import InputError, LoadedTask, load_task
from codereviewops.models import (
    BenchmarkTask,
    Category,
    Difficulty,
    ProviderResult,
    ReviewContext,
    RunArtifact,
    WorkflowState,
)
from codereviewops.providers import GroqProvider, MistralProvider, ReplayProvider, ReviewProvider
from codereviewops.workflow import AGENT_VERSION, PLANNER_VERSION, run_loaded_task


class BenchmarkRunError(ValueError):
    """Raised when a benchmark cannot complete or publish safely."""


ProviderFactory = Callable[[RunVariantV1, Path], ReviewProvider]


def default_matrix(suite_path: Path) -> BenchmarkMatrixV1:
    return BenchmarkMatrixV1(
        schema_version="1.0",
        matrix_id="default-replay-direct",
        suite_path=suite_path.name,
        variants=[
            RunVariantV1(
                variant_id="replay-direct",
                provider="replay",
                model=None,
                request_budget=0,
                prompt_version=TOOL_PROMPT_VERSION,
                agent_version=AGENT_VERSION,
                planner_version=PLANNER_VERSION,
                tool_transport="direct",
            )
        ],
        comparisons=[],
        threshold_profile=ThresholdProfileV1(),
        max_concurrency=1,
    )


def _validate_versions(matrix: BenchmarkMatrixV1) -> None:
    for variant in matrix.variants:
        if (
            variant.prompt_version != TOOL_PROMPT_VERSION
            or variant.agent_version != AGENT_VERSION
            or variant.planner_version != PLANNER_VERSION
        ):
            raise InputError("matrix references an unregistered contract version")


def _provider(variant: RunVariantV1, replay_path: Path) -> ReviewProvider:
    if variant.provider == "replay":
        return ReplayProvider(replay_path)
    assert variant.model is not None
    key_name = "GROQ_API_KEY" if variant.provider == "groq" else "MISTRAL_API_KEY"
    key = os.environ.get(key_name, "")
    if not key:
        raise InputError(f"{key_name} is required in the process environment")
    return (
        GroqProvider(model=variant.model, api_key=key)
        if variant.provider == "groq"
        else MistralProvider(model=variant.model, api_key=key)
    )


def _semantic_trace(artifact: RunArtifact) -> str:
    trace = []
    for entry in artifact.tool_trace:
        value = entry.model_dump(mode="json")
        value["latency_ms"] = 0
        trace.append(value)
    return canonical_hash(trace)


def _metric_deltas(base: VariantResultV1, candidate: VariantResultV1) -> dict[str, float]:
    names = (
        "completion_rate",
        "task_success_rate",
        "micro_precision",
        "micro_recall",
        "macro_precision",
        "macro_recall",
        "hallucination_rate",
        "missed_rate",
        "severity_accuracy",
        "tool_plan_accuracy",
    )
    return {
        name: float(getattr(candidate.metrics, name) - getattr(base.metrics, name))
        for name in names
    }


def _comparisons(
    matrix: BenchmarkMatrixV1,
    variants: list[VariantResultV1],
    traces: dict[str, dict[str, str]],
) -> list[ComparisonResultV1]:
    indexed = {result.variant.variant_id: result for result in variants}
    results: list[ComparisonResultV1] = []
    for comparison in matrix.comparisons:
        base = indexed[comparison.baseline_variant]
        candidate = indexed[comparison.candidate_variant]
        task_ids = list(traces[comparison.baseline_variant])
        equivalence = {
            task_id: traces[comparison.baseline_variant][task_id]
            == traces[comparison.candidate_variant][task_id]
            for task_id in task_ids
        }
        rate = sum(equivalence.values()) / len(equivalence)
        results.append(
            ComparisonResultV1(
                comparison_id=comparison.comparison_id,
                baseline_variant=comparison.baseline_variant,
                candidate_variant=comparison.candidate_variant,
                metric_deltas=_metric_deltas(base, candidate),
                trace_equivalence=rate,
                task_trace_equivalence=equivalence,
                passed=rate >= matrix.threshold_profile.trace_equivalence,
            )
        )
    return results


def _bytes_hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _validate_confined_component(
    path: Path,
    root: Path,
    *,
    directory: bool,
) -> Path:
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise InputError("benchmark task path escapes task authority") from exc
    current = lexical_root
    components = [current]
    for part in relative.parts:
        current /= part
        components.append(current)
    for component in components:
        try:
            metadata = os.lstat(component)
        except OSError as exc:
            raise InputError("benchmark task path is missing or inaccessible") from exc
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
            raise InputError("benchmark task paths cannot contain links or reparse points")
    final_mode = os.lstat(lexical_path).st_mode
    if directory != stat.S_ISDIR(final_mode):
        expected = "directory" if directory else "regular file"
        raise InputError(f"benchmark task path is not a {expected}")
    if not directory and not stat.S_ISREG(final_mode):
        raise InputError("benchmark task path is not a regular file")
    return lexical_path


def _secure_read_bytes(path: Path, root: Path) -> bytes:
    lexical = _validate_confined_component(path, root, directory=False)
    try:
        before = os.lstat(lexical)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags | no_follow)
    except OSError as exc:
        raise InputError("benchmark input could not be securely opened") from exc
    try:
        held = os.fstat(descriptor)
        current = os.lstat(lexical)
        attributes = getattr(current, "st_file_attributes", 0)
        identity = (current.st_dev, current.st_ino)
        if (
            stat.S_ISLNK(current.st_mode)
            or attributes & 0x400
            or not stat.S_ISREG(current.st_mode)
            or identity != (before.st_dev, before.st_ino)
            or identity != (held.st_dev, held.st_ino)
        ):
            raise InputError("benchmark input authority changed before read")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.lstat(lexical)
        after_attributes = getattr(after, "st_file_attributes", 0)
        if (
            stat.S_ISLNK(after.st_mode)
            or after_attributes & 0x400
            or (after.st_dev, after.st_ino) != identity
            or (held.st_size, held.st_mtime_ns)
            != (os.fstat(descriptor).st_size, os.fstat(descriptor).st_mtime_ns)
        ):
            raise InputError("benchmark input authority changed during read")
        return b"".join(chunks)
    except OSError as exc:
        raise InputError("benchmark input could not be securely read") from exc
    finally:
        os.close(descriptor)


def _parse_captured_suite(content: bytes) -> BenchmarkSuiteV1:
    try:
        return BenchmarkSuiteV1.model_validate_json(content)
    except ValidationError as exc:
        raise InputError("benchmark suite failed schema validation") from exc


def _secure_workspace_bytes(workspace: Path, task_root: Path) -> dict[Path, bytes]:
    captured: dict[Path, bytes] = {}
    pending = [workspace]
    while pending:
        directory = _validate_confined_component(pending.pop(), task_root, directory=True)
        before = os.lstat(directory)
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise InputError("benchmark workspace could not be inspected") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                raise InputError("benchmark workspace entry could not be inspected") from exc
            attributes = getattr(metadata, "st_file_attributes", 0)
            if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
                raise InputError("benchmark task paths cannot contain links or reparse points")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                captured[path] = _secure_read_bytes(path, task_root)
            else:
                raise InputError("benchmark workspaces require regular files and directories")
        after = os.lstat(directory)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise InputError("benchmark workspace authority changed during traversal")
    return captured


def _write_snapshot_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise InputError("benchmark snapshot paths collide")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class SelectedSnapshot:
    entry: TaskEntryV1
    loaded: LoadedTask
    manifest_relative: Path
    content_hash: str
    diff_text: str
    replay_provider: ReplayProvider


def _snapshot_selected_tasks(
    task_root: Path,
    selected: list[TaskEntryV1],
    snapshot_root: Path,
) -> list[SelectedSnapshot]:
    root = _validate_confined_component(task_root, task_root, directory=True)
    snapshot_task_root = snapshot_root / "tasks"
    snapshots: list[SelectedSnapshot] = []
    for entry in selected:
        manifest = root / entry.task_path
        manifest_bytes = _secure_read_bytes(manifest, root)
        try:
            task = BenchmarkTask.model_validate_json(manifest_bytes)
        except ValidationError as exc:
            raise InputError("task manifest failed schema validation") from exc
        if task.task_id != entry.task_id:
            raise InputError("suite task identity does not match task manifest")
        manifest_relative = manifest.relative_to(root)
        source_base = manifest.parent
        captured: dict[Path, bytes] = {manifest: manifest_bytes}
        diff = source_base / task.diff_path
        replay = source_base / task.replay_response_path
        captured[diff] = _secure_read_bytes(diff, root)
        captured[replay] = _secure_read_bytes(replay, root)
        workspace: Path | None = None
        if task.workspace_path is not None:
            workspace = _validate_confined_component(
                source_base / task.workspace_path,
                root,
                directory=True,
            )
            captured.update(_secure_workspace_bytes(workspace, root))
            (snapshot_task_root / workspace.relative_to(root)).mkdir(parents=True, exist_ok=True)
        for source, content in captured.items():
            destination = snapshot_task_root / source.relative_to(root)
            _write_snapshot_file(destination, content)
        snapshot_manifest = snapshot_task_root / manifest_relative
        loaded = load_task(snapshot_manifest)
        try:
            diff_text = captured[diff].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputError("benchmark diff is not valid UTF-8") from exc
        replay_provider = ReplayProvider(loaded.replay_path)
        if loaded.task.task_id != entry.task_id:
            raise InputError("snapshot task identity does not match suite entry")
        content_hash = canonical_hash(
            {
                source.relative_to(root).as_posix(): _bytes_hash(content)
                for source, content in sorted(captured.items())
            }
        )
        snapshots.append(
            SelectedSnapshot(
                entry=entry,
                loaded=loaded,
                manifest_relative=manifest_relative,
                content_hash=content_hash,
                diff_text=diff_text,
                replay_provider=replay_provider,
            )
        )
    return snapshots


def _snapshot_entries(snapshot: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = []
    pending = [snapshot]
    while pending:
        directory = pending.pop()
        directories.append(directory)
        for entry in os.scandir(directory):
            path = Path(entry.path)
            metadata = os.lstat(path)
            attributes = getattr(metadata, "st_file_attributes", 0)
            if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
                raise OSError("benchmark snapshot authority changed")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(path)
            else:
                raise OSError("benchmark snapshot contains a special file")
    return files, directories


def _seal_snapshot(snapshot: Path) -> None:
    files, directories = _snapshot_entries(snapshot)
    for path in files:
        path.chmod(0o400)
    for path in reversed(directories):
        path.chmod(0o500)


def _cleanup_snapshot(snapshot: Path | None) -> None:
    if snapshot is None or not os.path.lexists(snapshot):
        return
    metadata = os.lstat(snapshot)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400 or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("benchmark snapshot authority changed")
    files, directories = _snapshot_entries(snapshot)
    for path in directories:
        path.chmod(0o700)
    for path in files:
        path.chmod(0o600)
    shutil.rmtree(snapshot)


def _hashes(
    matrix: BenchmarkMatrixV1,
    suite: BenchmarkSuiteV1,
    suite_content: bytes,
    selected_tasks: list[SelectedSnapshot],
) -> dict[str, str]:
    task_hashes = {
        snapshot.manifest_relative.as_posix(): snapshot.content_hash for snapshot in selected_tasks
    }
    contracts = {
        "prompt": TOOL_PROMPT_VERSION,
        "agent": AGENT_VERSION,
        "planner": PLANNER_VERSION,
        "evaluator": "evaluation-v1",
        "harness": "benchmark-harness-v1",
    }
    return {
        "suite_content": _bytes_hash(suite_content),
        "selection": canonical_hash(
            [snapshot.manifest_relative.as_posix() for snapshot in selected_tasks]
        ),
        "matrix": canonical_hash(matrix.model_dump(mode="json", exclude={"baseline_path"})),
        "configuration": canonical_hash(matrix.threshold_profile.model_dump(mode="json")),
        "contracts": canonical_hash(contracts),
        "tasks": canonical_hash(task_hashes),
        "suite": canonical_hash(suite.model_dump(mode="json")),
    }


def _validate_output_ancestors(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if not os.path.lexists(ancestor):
            continue
        try:
            metadata = os.lstat(ancestor)
        except OSError as exc:
            raise BenchmarkRunError("output ancestor inspection failed") from exc
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
            raise BenchmarkRunError("output ancestor is a link or reparse point")
        if not stat.S_ISDIR(metadata.st_mode):
            raise BenchmarkRunError("output ancestor is not a directory")


def _ensure_safe_output(output: Path) -> tuple[Path, Path, int]:
    output = Path(os.path.abspath(output))
    parent = output.parent
    _validate_output_ancestors(parent)
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BenchmarkRunError("benchmark output parent could not be created") from exc
    _validate_output_ancestors(parent)
    if os.path.lexists(output):
        raise BenchmarkRunError("benchmark output already exists")
    lock = parent / f".{output.name}.lock"
    if os.path.lexists(lock):
        raise BenchmarkRunError("benchmark output is locked")
    residue_prefix = f".{output.name}."
    try:
        entries = list(parent.iterdir())
    except OSError as exc:
        raise BenchmarkRunError("benchmark output preflight failed") from exc
    if any(entry != lock and entry.name.startswith(residue_prefix) for entry in entries):
        raise BenchmarkRunError("benchmark publication residue exists")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BenchmarkRunError("benchmark output is locked") from exc
    except OSError as exc:
        raise BenchmarkRunError("benchmark output lock could not be acquired") from exc
    return output, lock, descriptor


def _validate_commit_authority(output: Path, lock: Path, descriptor: int) -> None:
    _validate_output_ancestors(output.parent)
    if os.path.lexists(output):
        raise BenchmarkRunError("benchmark output appeared during execution")
    try:
        path_metadata = os.lstat(lock)
        held_metadata = os.fstat(descriptor)
    except OSError as exc:
        raise BenchmarkRunError("benchmark output lock authority changed") from exc
    attributes = getattr(path_metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or attributes & 0x400
        or not stat.S_ISREG(path_metadata.st_mode)
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (held_metadata.st_dev, held_metadata.st_ino)
    ):
        raise BenchmarkRunError("benchmark output lock authority changed")


def _cleanup_staging(staging: Path) -> None:
    if not os.path.lexists(staging):
        return
    metadata = os.lstat(staging)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400 or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("staging cleanup authority changed")
    shutil.rmtree(staging)


def _write_tree(
    staging: Path,
    run: BenchmarkRunV1,
    artifacts: dict[str, list[tuple[str, Any, RunArtifact]]],
) -> None:
    for variant in run.variants:
        records = artifacts[variant.variant.variant_id]
        for suite_task_id, task, artifact in records:
            directory = staging / "runs" / variant.variant.variant_id / suite_task_id
            directory.mkdir(parents=True)
            (directory / "run.json").write_text(
                json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (directory / "report.md").write_text(
                render_markdown(task, artifact), encoding="utf-8", newline="\n"
            )
    comparisons = staging / "comparisons"
    comparisons.mkdir(parents=True)
    for comparison in run.comparisons:
        (comparisons / f"{comparison.comparison_id}.json").write_text(
            json.dumps(comparison.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    benchmark_path = staging / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        f"# Benchmark Report: {run.matrix_id}",
        "",
        f"- Suite: {run.suite_id}",
        f"- Selected tasks: {len(run.selected_task_ids)}",
        f"- Quality gate: **{'pass' if run.quality_gate_passed else 'fail'}**",
        "",
        "## Variants",
        "",
    ]
    for result in run.variants:
        lines.append(
            f"- {result.variant.variant_id}: **{'pass' if result.gate_passed else 'fail'}** "
            f"({result.metrics.successful_count}/{result.metrics.task_count})"
        )
    (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    BenchmarkRunV1.model_validate_json(benchmark_path.read_text(encoding="utf-8"))
    for path in staging.glob("runs/*/*/run.json"):
        RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    for path in staging.glob("comparisons/*.json"):
        ComparisonResultV1.model_validate_json(path.read_text(encoding="utf-8"))
    for path in staging.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in staging.rglob("*.md"):
        if not path.read_text(encoding="utf-8").strip():
            raise BenchmarkRunError("benchmark report is empty")
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())


class _LazyProvider:
    def __init__(
        self,
        factory: ProviderFactory,
        variant: RunVariantV1,
        replay_path: Path,
    ) -> None:
        self._factory = factory
        self._variant = variant
        self._replay_path = replay_path

    def review(self, context: ReviewContext) -> ProviderResult:
        return self._factory(self._variant, self._replay_path).review(context)


def _execute_matrix(
    *,
    matrix: BenchmarkMatrixV1,
    suite: BenchmarkSuiteV1,
    selected_tasks: list[SelectedSnapshot],
    selected_task_ids: list[str],
    factory: ProviderFactory,
    factory_is_default: bool,
    hashes: dict[str, str],
    baseline: BenchmarkBaselineV1 | None,
) -> tuple[BenchmarkRunV1, dict[str, list[tuple[str, Any, RunArtifact]]], bool]:
    variant_results: list[VariantResultV1] = []
    artifacts: dict[str, list[tuple[str, Any, RunArtifact]]] = {}
    traces: dict[str, dict[str, str]] = {}
    for variant in matrix.variants:
        rows = []
        variant_artifacts = []
        variant_traces: dict[str, str] = {}
        for snapshot in selected_tasks:
            _entry = snapshot.entry
            loaded = snapshot.loaded
            provider: ReviewProvider = (
                snapshot.replay_provider
                if factory_is_default and variant.provider == "replay"
                else _LazyProvider(factory, variant, loaded.replay_path)
            )
            task, artifact = run_loaded_task(
                loaded,
                provider,
                variant.provider,
                tool_transport=variant.tool_transport,
                trusted_diff_text=snapshot.diff_text,
            )
            if artifact.final_state == WorkflowState.FAILED:
                raise BenchmarkRunError("benchmark workflow failed")
            metric = task_metrics(task, artifact)
            rows.append((task, metric))
            variant_artifacts.append((_entry.task_id, task, artifact))
            variant_traces[task.task_id] = _semantic_trace(artifact)
        aggregate = aggregate_metrics(rows)
        failures = gate_failures(aggregate, matrix.threshold_profile)
        variant_results.append(
            VariantResultV1(
                variant=variant,
                tasks=[metric for _, metric in rows],
                metrics=aggregate,
                gate_passed=not failures,
                gate_failures=failures,
            )
        )
        artifacts[variant.variant_id] = variant_artifacts
        traces[variant.variant_id] = variant_traces

    comparisons = _comparisons(matrix, variant_results, traces)
    quality_passed = all(result.gate_passed for result in variant_results) and all(
        comparison.passed for comparison in comparisons
    )
    run_data: dict[str, Any] = {
        "schema_version": "1.0",
        "matrix_id": matrix.matrix_id,
        "suite_id": suite.suite_id,
        "selected_task_ids": selected_task_ids,
        "variants": variant_results,
        "comparisons": comparisons,
        "quality_gate_passed": quality_passed,
        "baseline_passed": None,
        "baseline_failures": [],
        "hashes": {**hashes, "results": "sha256:" + "0" * 64},
    }
    provisional = BenchmarkRunV1.model_validate(run_data)
    run_data["hashes"] = {
        **hashes,
        "results": stable_result_hash(stable_result(provisional)),
    }
    run = BenchmarkRunV1.model_validate(run_data)
    if baseline is not None:
        failures = compare_baseline(run, baseline)
        run = run.model_copy(
            update={"baseline_passed": not failures, "baseline_failures": failures}
        )
    return run, artifacts, quality_passed


def _publish_tree(staging: Path, output: Path) -> None:
    before = os.lstat(staging)
    identity = (before.st_dev, before.st_ino)
    try:
        os.replace(staging, output)
    except BaseException:
        if not os.path.lexists(staging) and os.path.lexists(output):
            current = os.lstat(output)
            attributes = getattr(current, "st_file_attributes", 0)
            if (
                stat.S_ISDIR(current.st_mode)
                and not stat.S_ISLNK(current.st_mode)
                and not attributes & 0x400
                and (current.st_dev, current.st_ino) == identity
            ):
                warnings.warn(
                    "benchmark output was published despite an interrupted publish return",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
        raise


def run_benchmark(
    *,
    suite_path: Path,
    output_dir: Path,
    matrix_path: Path | None = None,
    task_ids: set[str] | None = None,
    categories: set[Category] | None = None,
    difficulties: set[Difficulty] | None = None,
    polarity: str | None = None,
    allow_live: bool = False,
    max_live_requests: int | None = None,
    use_baseline: bool = True,
    provider_factory: ProviderFactory | None = None,
) -> tuple[BenchmarkRunV1, int]:
    if matrix_path is None:
        matrix = default_matrix(suite_path)
        resolved_suite = Path(os.path.abspath(suite_path))
        matrix_root = Path.cwd().resolve()
    else:
        matrix_file = matrix_path.resolve(strict=True)
        matrix = load_matrix(matrix_file)
        matrix_root = matrix_file.parent.parent
        resolved_suite = Path(os.path.abspath(matrix_root / matrix.suite_path))
        if not resolved_suite.is_relative_to(matrix_root):
            raise InputError("matrix suite path escapes benchmark roots")
    _validate_versions(matrix)
    task_root = resolved_suite.parent.parent
    suite_content = _secure_read_bytes(resolved_suite, task_root)
    suite = _parse_captured_suite(suite_content)
    selected = select_tasks(
        suite,
        task_ids=task_ids,
        categories=categories,
        difficulties=difficulties,
        polarity=polarity,
    )
    live_variants = [variant for variant in matrix.variants if variant.provider != "replay"]
    exact_live_requests = len(selected) * len(live_variants)
    if live_variants:
        if not allow_live:
            raise InputError("live benchmark variants require --allow-live")
        if max_live_requests != exact_live_requests:
            raise InputError("--max-live-requests must exactly match the planned request count")
        if any(variant.request_budget != len(selected) for variant in live_variants):
            raise InputError("live variant request budget must exactly match selected tasks")
        for variant in live_variants:
            key = "GROQ_API_KEY" if variant.provider == "groq" else "MISTRAL_API_KEY"
            if not os.environ.get(key):
                raise InputError(f"{key} is required in the process environment")
    elif max_live_requests not in {None, 0}:
        raise InputError("--max-live-requests must be zero for replay-only matrices")

    selected_ids = [entry.task_id for entry in selected]
    baseline: BenchmarkBaselineV1 | None = None
    if use_baseline and matrix.baseline_path:
        baseline_path = (matrix_root / matrix.baseline_path).resolve(strict=True)
        if not baseline_path.is_relative_to(matrix_root):
            raise InputError("matrix baseline path escapes benchmark roots")
        baseline = load_baseline(baseline_path)

    output, lock, lock_descriptor = _ensure_safe_output(output_dir)
    staging = output.parent / f".{output.name}.{os.getpid()}.{uuid4().hex}.staging"
    snapshot_root: Path | None = None
    operation_error: BaseException | None = None
    operation_traceback = None
    cleanup_errors: list[str] = []
    final_published = False
    result: tuple[BenchmarkRunV1, int] | None = None
    try:
        snapshot_root = Path(mkdtemp(prefix=f".{output.name}.input.", dir=output.parent))
        selected_tasks = _snapshot_selected_tasks(task_root, selected, snapshot_root)
        _seal_snapshot(snapshot_root)
        hashes = _hashes(matrix, suite, suite_content, selected_tasks)
        if baseline is not None:
            validate_baseline_inputs(
                matrix_id=matrix.matrix_id,
                suite_id=suite.suite_id,
                selected_task_ids=selected_ids,
                hashes=hashes,
                baseline=baseline,
            )
        factory = provider_factory or _provider
        run, artifacts, quality_passed = _execute_matrix(
            matrix=matrix,
            suite=suite,
            selected_tasks=selected_tasks,
            selected_task_ids=selected_ids,
            factory=factory,
            factory_is_default=provider_factory is None,
            hashes=hashes,
            baseline=baseline,
        )
        staging.mkdir()
        _write_tree(staging, run, artifacts)
        _validate_commit_authority(output, lock, lock_descriptor)
        _publish_tree(staging, output)
        final_published = True
        passed = quality_passed and run.baseline_passed is not False
        result = (run, 0 if passed else 1)
    except BaseException as exc:
        operation_error = exc
        operation_traceback = exc.__traceback__
    finally:
        try:
            _cleanup_staging(staging)
        except BaseException:
            cleanup_errors.append("staging cleanup failed")
        try:
            _cleanup_snapshot(snapshot_root)
        except BaseException:
            cleanup_errors.append("input snapshot cleanup failed")
        try:
            os.close(lock_descriptor)
        except BaseException:
            cleanup_errors.append("lock close failed")
        try:
            lock.unlink(missing_ok=True)
        except BaseException:
            cleanup_errors.append("lock removal failed")

    if operation_error is not None:
        raise operation_error.with_traceback(operation_traceback)
    if cleanup_errors:
        diagnostic = "; ".join(cleanup_errors)
        if final_published:
            warnings.warn(
                f"benchmark output published with cleanup diagnostic: {diagnostic}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise BenchmarkRunError(f"benchmark cleanup failed: {diagnostic}")
    if result is None:
        raise BenchmarkRunError("benchmark publication ended without a result")
    return result
