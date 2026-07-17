"""Stable benchmark baseline creation and regression comparison."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import warnings
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from codereviewops.benchmark_models import (
    BenchmarkBaselineV1,
    BenchmarkRunV1,
    StableBenchmarkRunV1,
)
from codereviewops.io import InputError


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def stable_result(run: BenchmarkRunV1) -> StableBenchmarkRunV1:
    data = run.model_dump(mode="json")
    data.pop("baseline_passed")
    data.pop("baseline_failures")
    for variant in data["variants"]:
        metrics = variant["metrics"]
        for name in ("latency_ms_total", "input_tokens_total", "output_tokens_total"):
            metrics.pop(name)
        for task in variant["tasks"]:
            for name in ("latency_ms", "input_tokens", "output_tokens"):
                task.pop(name)
    return StableBenchmarkRunV1.model_validate(data)


def stable_result_payload(result: StableBenchmarkRunV1) -> dict[str, object]:
    payload = result.model_dump(mode="json")
    payload["hashes"] = {key: value for key, value in result.hashes.items() if key != "results"}
    return payload


def stable_result_hash(result: StableBenchmarkRunV1) -> str:
    return canonical_hash(stable_result_payload(result))


def _validate_result_hash(result: StableBenchmarkRunV1) -> None:
    if result.hashes["results"] != stable_result_hash(result):
        raise InputError("benchmark result hash does not match stable result")


def _validate_ancestors(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if not os.path.lexists(ancestor):
            continue
        try:
            metadata = os.lstat(ancestor)
        except OSError as exc:
            raise InputError("baseline ancestor inspection failed") from exc
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
            raise InputError("baseline ancestors cannot contain links or reparse points")
        if not stat.S_ISDIR(metadata.st_mode):
            raise InputError("baseline ancestor is not a directory")


def _acquire_reservation(output: Path) -> tuple[Path, int]:
    parent = output.parent
    _validate_ancestors(parent)
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError("baseline parent could not be created") from exc
    _validate_ancestors(parent)
    if os.path.lexists(output):
        raise InputError("baseline output already exists")
    lock = parent / f".{output.name}.lock"
    if os.path.lexists(lock):
        raise InputError("baseline output is locked")
    prefix = f".{output.name}."
    try:
        entries = list(parent.iterdir())
    except OSError as exc:
        raise InputError("baseline output preflight failed") from exc
    if any(entry != lock and entry.name.startswith(prefix) for entry in entries):
        raise InputError("baseline publication residue exists")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise InputError("baseline output is locked") from exc
    except OSError as exc:
        raise InputError("baseline reservation could not be acquired") from exc
    return lock, descriptor


def _cleanup_temporary(path: Path) -> None:
    if not os.path.lexists(path):
        return
    metadata = os.lstat(path)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400 or not stat.S_ISREG(metadata.st_mode):
        raise OSError("baseline temporary authority changed")
    path.unlink()


def _validate_commit(
    output: Path,
    temporary: Path,
    temporary_identity: tuple[int, int],
    lock: Path,
    lock_descriptor: int,
) -> None:
    _validate_ancestors(output.parent)
    if os.path.lexists(output):
        raise InputError("baseline output appeared during publication")
    try:
        lock_path = os.lstat(lock)
        lock_held = os.fstat(lock_descriptor)
        temp_path = os.lstat(temporary)
    except OSError as exc:
        raise InputError("baseline publication authority changed") from exc
    attributes = getattr(lock_path, "st_file_attributes", 0)
    temp_attributes = getattr(temp_path, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(lock_path.st_mode)
        or attributes & 0x400
        or not stat.S_ISREG(lock_path.st_mode)
        or (lock_path.st_dev, lock_path.st_ino) != (lock_held.st_dev, lock_held.st_ino)
        or stat.S_ISLNK(temp_path.st_mode)
        or temp_attributes & 0x400
        or not stat.S_ISREG(temp_path.st_mode)
        or (temp_path.st_dev, temp_path.st_ino) != temporary_identity
    ):
        raise InputError("baseline publication authority changed")


def _publish_baseline(
    temporary: Path,
    output: Path,
    temporary_identity: tuple[int, int],
) -> None:
    try:
        os.link(temporary, output, follow_symlinks=False)
    except BaseException:
        if os.path.lexists(output):
            current = os.lstat(output)
            attributes = getattr(current, "st_file_attributes", 0)
            if (
                stat.S_ISREG(current.st_mode)
                and not stat.S_ISLNK(current.st_mode)
                and not attributes & 0x400
                and (current.st_dev, current.st_ino) == temporary_identity
            ):
                warnings.warn(
                    "baseline was published despite an interrupted publish return",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
        raise


def create_baseline(result_path: Path, output_path: Path) -> BenchmarkBaselineV1:
    try:
        run = BenchmarkRunV1.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InputError("benchmark result is missing or invalid") from exc
    if not run.quality_gate_passed:
        raise InputError("a baseline can only be created from a passing benchmark")
    stable = stable_result(run)
    _validate_result_hash(stable)
    baseline = BenchmarkBaselineV1(
        schema_version="1.0",
        matrix_id=run.matrix_id,
        suite_id=run.suite_id,
        selected_task_ids=run.selected_task_ids,
        stable_result=stable,
        hashes=run.hashes,
    )

    output = Path(os.path.abspath(output_path))
    lock, lock_descriptor = _acquire_reservation(output)
    temporary = output.parent / f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp"
    primary: BaseException | None = None
    primary_traceback = None
    cleanup_errors: list[str] = []
    published = False
    temporary_descriptor: int | None = None
    try:
        temporary_descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        stream = os.fdopen(temporary_descriptor, "w", encoding="utf-8", newline="\n")
        temporary_descriptor = None
        with stream:
            json.dump(baseline.model_dump(mode="json"), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            metadata = os.fstat(stream.fileno())
            temporary_identity = (metadata.st_dev, metadata.st_ino)
        BenchmarkBaselineV1.model_validate_json(temporary.read_text(encoding="utf-8"))
        _validate_commit(output, temporary, temporary_identity, lock, lock_descriptor)
        try:
            _publish_baseline(temporary, output, temporary_identity)
        except FileExistsError as exc:
            raise InputError("baseline output appeared during publication") from exc
        published = True
    except BaseException as exc:
        primary = exc
        primary_traceback = exc.__traceback__
    finally:
        if temporary_descriptor is not None:
            try:
                os.close(temporary_descriptor)
            except BaseException:
                cleanup_errors.append("temporary close failed")
        try:
            _cleanup_temporary(temporary)
        except BaseException:
            cleanup_errors.append("temporary cleanup failed")
        try:
            os.close(lock_descriptor)
        except BaseException:
            cleanup_errors.append("reservation close failed")
        try:
            lock.unlink(missing_ok=True)
        except BaseException:
            cleanup_errors.append("reservation removal failed")

    if primary is not None:
        raise primary.with_traceback(primary_traceback)
    if cleanup_errors:
        diagnostic = "; ".join(cleanup_errors)
        if published:
            warnings.warn(
                f"baseline published with cleanup diagnostic: {diagnostic}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise InputError(f"baseline cleanup failed: {diagnostic}")
    if not published:
        raise InputError("baseline publication ended without a result")
    return baseline


def load_baseline(path: Path) -> BenchmarkBaselineV1:
    try:
        baseline = BenchmarkBaselineV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InputError("benchmark baseline is missing or invalid") from exc
    _validate_result_hash(baseline.stable_result)
    return baseline


def validate_baseline_inputs(
    *,
    matrix_id: str,
    suite_id: str,
    selected_task_ids: list[str],
    hashes: dict[str, str],
    baseline: BenchmarkBaselineV1,
) -> None:
    if (
        matrix_id != baseline.matrix_id
        or suite_id != baseline.suite_id
        or selected_task_ids != baseline.selected_task_ids
    ):
        raise InputError("baseline selected set does not match benchmark run")
    current_contracts = {key: value for key, value in hashes.items() if key != "results"}
    baseline_contracts = {key: value for key, value in baseline.hashes.items() if key != "results"}
    if current_contracts != baseline_contracts:
        raise InputError("baseline contract or input hashes do not match benchmark run")


def compare_baseline(run: BenchmarkRunV1, baseline: BenchmarkBaselineV1) -> list[str]:
    validate_baseline_inputs(
        matrix_id=run.matrix_id,
        suite_id=run.suite_id,
        selected_task_ids=run.selected_task_ids,
        hashes=run.hashes,
        baseline=baseline,
    )
    current = stable_result(run)
    return [] if current == baseline.stable_result else ["stable result differs from baseline"]
