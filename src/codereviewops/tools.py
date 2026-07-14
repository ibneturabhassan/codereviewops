"""Deterministic, bounded tools over an identity-checked workspace."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from codereviewops.models import (
    ChangedLocation,
    ReadFileArguments,
    ReadFileResult,
    RunTestsArguments,
    RunTestsResult,
    SearchCodeArguments,
    SearchCodeResult,
    SearchMatch,
    TestStatus,
    ToolArguments,
    ToolErrorCode,
    ToolFailureResult,
    ToolName,
    ToolPlan,
    ToolResult,
    ToolStatus,
    ToolTraceEntry,
    TraceInfluence,
    TraceProvenance,
    VerificationResult,
    normalize_relative_posix_path,
)

MAX_WORKSPACE_DEPTH = 12
MAX_WORKSPACE_FILES = 2_000
MAX_WORKSPACE_BYTES = 10 * 1024 * 1024
MAX_READ_BYTES = 256 * 1024
MAX_SEARCH_RESULTS = 100
MAX_TRACE_RESULT_CHARS = 64 * 1024
MAX_TRACE_RESULT_BYTES = 64 * 1024
TEST_COMMAND: Literal["python -B -m unittest discover -s tests -p test_*.py"] = (
    "python -B -m unittest discover -s tests -p test_*.py"
)
_DIFF_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
MAX_DIFF_LINES = 100_000


class ToolError(ValueError):
    """Safe categorized tool failure without host or provider-controlled text."""

    def __init__(self, code: ToolErrorCode, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(message)


class ToolExecutionError(ToolError):
    def __init__(self, error: ToolError, trace: list[ToolTraceEntry]) -> None:
        self.trace = trace
        super().__init__(error.code, error.message, retryable=error.retryable)


@dataclass(frozen=True)
class TestExecution:
    status: TestStatus
    summary: str
    output_truncated: bool = False
    infrastructure_error: bool = False
    error_code: str | None = None


class TestRunner(Protocol):
    def run(self, workspace: Path, profile: str) -> TestExecution: ...


@dataclass(frozen=True)
class ToolRun:
    trace: list[ToolTraceEntry]
    verification: VerificationResult | None


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    size: int
    modified_ns: int


def _identity(metadata: os.stat_result) -> _Identity:
    return _Identity(metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _is_reparse(path: Path, metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag) or path.is_junction()


def _sanitize_text(value: str, limit: int = MAX_TRACE_RESULT_CHARS) -> str:
    normalized = "".join(
        char if char in {"\n", "\t"} or ord(char) >= 32 else "\ufffd" for char in value
    )
    return normalized if len(normalized) <= limit else normalized[:limit]


def _create_disposable_directory(root: Path) -> Path:
    for _ in range(10):
        candidate = root / f"codereviewops-snapshot-{secrets.token_hex(8)}"
        try:
            candidate.mkdir(mode=0o700 if os.name != "nt" else 0o777)
        except FileExistsError:
            continue
        return candidate
    raise OSError("could not allocate a unique snapshot directory")


class Workspace:
    """A scanned workspace whose files are read through no-follow identity checks."""

    def __init__(
        self,
        root: Path,
        benchmark_root: Path,
        *,
        snapshot_root: Path | None = None,
    ) -> None:
        self.root = root.absolute()
        try:
            self._benchmark_root = benchmark_root.resolve(strict=True)
            resolved_root = self.root.resolve(strict=True)
        except OSError as exc:
            raise ToolError("invalid_path", "workspace is missing or inaccessible") from exc
        if not resolved_root.is_relative_to(self._benchmark_root):
            raise ToolError("invalid_path", "workspace escapes the benchmark root")
        ancestors = (self._benchmark_root, *self._benchmark_root.parents)
        self._repository_root = next(
            (parent for parent in ancestors if (parent / ".git").exists()), None
        )
        self._snapshot_root = snapshot_root
        self._validate_components(self.root)
        self._identities = self._scan()

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(sorted(self._identities))

    def _validate_components(self, path: Path) -> None:
        try:
            relative = path.relative_to(self._benchmark_root)
        except ValueError as exc:
            raise ToolError("invalid_path", "workspace escapes the benchmark root") from exc
        current = self._benchmark_root
        for part in relative.parts:
            if part.casefold() == ".git":
                raise ToolError("invalid_path", ".git paths are forbidden")
            current /= part
            try:
                metadata = os.lstat(current)
            except OSError as exc:
                raise ToolError("invalid_path", "workspace entry is inaccessible") from exc
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(current, metadata):
                raise ToolError("invalid_path", "workspace links and reparse points are forbidden")

    def _scan(self) -> dict[str, _Identity]:
        identities: dict[str, _Identity] = {}
        total_bytes = 0
        pending = [(self.root, 0)]
        while pending:
            directory, depth = pending.pop()
            if depth > MAX_WORKSPACE_DEPTH:
                raise ToolError("limit_exceeded", "workspace exceeds the depth limit")
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name)
            except OSError as exc:
                raise ToolError("read_failed", "workspace directory is inaccessible") from exc
            children: list[Path] = []
            for entry in entries:
                path = Path(entry.path)
                if entry.name.casefold() == ".git":
                    raise ToolError("invalid_path", ".git paths are forbidden")
                try:
                    metadata = os.lstat(path)
                except OSError as exc:
                    raise ToolError("read_failed", "workspace entry is inaccessible") from exc
                if entry.is_symlink() or _is_reparse(path, metadata):
                    raise ToolError(
                        "invalid_path", "workspace links and reparse points are forbidden"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    children.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    total_bytes += metadata.st_size
                    reference = path.relative_to(self.root).as_posix()
                    identities[reference] = _identity(metadata)
                    if len(identities) > MAX_WORKSPACE_FILES:
                        raise ToolError("limit_exceeded", "workspace exceeds the file limit")
                    if total_bytes > MAX_WORKSPACE_BYTES:
                        raise ToolError("limit_exceeded", "workspace exceeds the byte limit")
                else:
                    raise ToolError("invalid_path", "workspace special files are forbidden")
            pending.extend((child, depth + 1) for child in reversed(children))
        return identities

    def _path(self, reference: str) -> Path:
        try:
            normalized = normalize_relative_posix_path(reference)
        except ValueError as exc:
            raise ToolError("invalid_path", "file path is unsafe") from exc
        if normalized not in self._identities:
            raise ToolError("invalid_path", "requested file is not in the scanned workspace")
        return self.root.joinpath(*normalized.split("/"))

    def read_bytes(self, reference: str) -> bytes:
        path = self._path(reference)
        expected = self._identities[reference]
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            before = os.lstat(path)
            if (
                stat.S_ISLNK(before.st_mode)
                or _is_reparse(path, before)
                or _identity(before) != expected
            ):
                raise ToolError("changed_workspace", "workspace file changed after validation")
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != expected:
                raise ToolError("changed_workspace", "workspace file identity changed")
            content = os.read(descriptor, MAX_READ_BYTES + 1)
            if len(content) > MAX_READ_BYTES:
                raise ToolError("limit_exceeded", "requested file exceeds the read limit")
            after = os.fstat(descriptor)
            current = os.lstat(path)
            if _identity(after) != expected or _identity(current) != expected:
                raise ToolError("changed_workspace", "workspace file changed while reading")
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError("read_failed", "requested file could not be read") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return content

    def read_file(self, reference: str) -> str:
        content = self.read_bytes(reference)
        if b"\x00" in content:
            raise ToolError("binary_file", "binary files are not supported")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("invalid_utf8", "requested file is not valid UTF-8") from exc
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if any(
            ord(char) == 127 or (ord(char) < 32 and char not in {"\n", "\t"}) for char in normalized
        ):
            raise ToolError("unsupported_text", "requested file contains unsupported controls")
        return normalized

    def read_result(self, reference: str) -> ReadFileResult:
        source = self.read_bytes(reference)
        if b"\x00" in source:
            raise ToolError("binary_file", "binary files are not supported")
        try:
            decoded = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("invalid_utf8", "requested file is not valid UTF-8") from exc
        normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
        if any(
            ord(char) == 127 or (ord(char) < 32 and char not in {"\n", "\t"}) for char in normalized
        ):
            raise ToolError("unsupported_text", "requested file contains unsupported controls")
        encoded = normalized.encode("utf-8")
        returned = encoded[:MAX_TRACE_RESULT_BYTES]
        while True:
            try:
                content = returned.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                returned = returned[: exc.start]
        return ReadFileResult(
            kind="read_file_success",
            path=reference,
            source_bytes=len(source),
            normalized_bytes=len(encoded),
            returned_bytes=len(returned),
            content=content,
            truncated=len(returned) < len(encoded),
        )

    def search_code(self, query: str) -> SearchCodeResult:
        if (
            not query
            or len(query) > 128
            or any(ord(char) == 127 or ord(char) < 32 for char in query)
        ):
            raise ToolError("limit_exceeded", "search query is invalid")
        matches: list[SearchMatch] = []
        files_scanned = 0
        files_skipped = 0
        truncated = False
        for reference in self.files:
            try:
                text = self.read_file(reference)
            except ToolError as exc:
                if exc.code in {"binary_file", "invalid_utf8", "unsupported_text"}:
                    files_skipped += 1
                    continue
                raise
            files_scanned += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                start = 0
                while True:
                    column = line.find(query, start)
                    if column < 0:
                        break
                    if len(matches) == MAX_SEARCH_RESULTS:
                        truncated = True
                        return SearchCodeResult(
                            kind="search_code_success",
                            query=query,
                            files_total=len(self.files),
                            files_scanned=files_scanned,
                            files_skipped=files_skipped,
                            matches=matches,
                            truncated=truncated,
                        )
                    matches.append(
                        SearchMatch(
                            path=reference,
                            line=line_number,
                            column=column + 1,
                            excerpt=_sanitize_text(line.strip(), 240),
                        )
                    )
                    start = column + max(1, len(query))
        return SearchCodeResult(
            kind="search_code_success",
            query=query,
            files_total=len(self.files),
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            matches=matches,
            truncated=truncated,
        )

    @contextmanager
    def sealed_snapshot(self) -> Iterator[Path]:
        protected_roots = [self._benchmark_root]
        if self._repository_root is not None:
            protected_roots.append(self._repository_root)
        try:
            base = (self._snapshot_root or Path(tempfile.gettempdir())).resolve(strict=True)
        except OSError:
            raise ToolError("cleanup_failed", "isolated workspace could not be prepared") from None
        if any(base.is_relative_to(protected) for protected in protected_roots):
            raise ToolError("cleanup_failed", "isolated workspace location is unsafe")
        try:
            root = _create_disposable_directory(base)
        except OSError:
            raise ToolError("cleanup_failed", "isolated workspace could not be prepared") from None
        try:
            for reference in self.files:
                destination = root.joinpath(*reference.split("/"))
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(self.read_bytes(reference))
                except ToolError:
                    raise
                except OSError:
                    raise ToolError(
                        "cleanup_failed", "isolated workspace could not be prepared"
                    ) from None
            yield root
        finally:
            try:
                shutil.rmtree(root)
            except OSError:
                raise ToolError(
                    "cleanup_failed", "isolated workspace could not be removed"
                ) from None


def _diff_path(raw: str, prefix: str) -> str | None:
    value = raw.split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if not value.startswith(prefix):
        raise ToolError("invalid_diff", "diff path header is malformed")
    try:
        return normalize_relative_posix_path(value[len(prefix) :])
    except ValueError:
        raise ToolError("invalid_diff", "diff path header is unsafe") from None


def parse_unified_diff(diff_text: str) -> dict[str, frozenset[int]]:
    lines = diff_text.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        raise ToolError("invalid_diff", "diff exceeds the line limit")
    added: dict[str, set[int]] = {}
    current_file: str | None = None
    header_ready = False
    saw_file = False
    saw_hunk = False
    index = 0
    metadata_prefixes = (
        "diff --git ",
        "index ",
        "new file mode ",
        "deleted file mode ",
        "old mode ",
        "new mode ",
        "similarity index ",
        "dissimilarity index ",
        "rename from ",
        "rename to ",
    )
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            current_file = None
            header_ready = False
            index += 1
            continue
        if line.startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise ToolError("invalid_diff", "diff file headers are incomplete")
            _diff_path(line[4:], "a/")
            current_file = _diff_path(lines[index + 1][4:], "b/")
            saw_file = True
            header_ready = True
            if current_file is not None:
                added.setdefault(current_file, set())
                if len(added) > 100:
                    raise ToolError("invalid_diff", "diff exceeds the file limit")
            index += 2
            continue
        match = _DIFF_HUNK.fullmatch(line)
        if match is not None:
            if not header_ready:
                raise ToolError("invalid_diff", "diff hunk has no file header")
            old_start, old_raw, new_start, new_raw = match.groups()
            old_count = 1 if old_raw is None else int(old_raw)
            new_count = 1 if new_raw is None else int(new_raw)
            old_line = int(old_start)
            new_line = int(new_start)
            if (old_count and old_line < 1) or (new_count and new_line < 1):
                raise ToolError("invalid_diff", "diff hunk range is invalid")
            old_seen = new_seen = 0
            index += 1
            while old_seen < old_count or new_seen < new_count:
                if index >= len(lines):
                    raise ToolError("invalid_diff", "diff hunk body is incomplete")
                body = lines[index]
                if body.startswith("\\ "):
                    raise ToolError("invalid_diff", "diff newline marker is misplaced")
                if not body:
                    raise ToolError("invalid_diff", "diff hunk body is malformed")
                prefix = body[0]
                if prefix == " ":
                    old_seen += 1
                    new_seen += 1
                    old_line += 1
                    new_line += 1
                elif prefix == "-":
                    old_seen += 1
                    old_line += 1
                elif prefix == "+":
                    new_seen += 1
                    if current_file is None:
                        raise ToolError("invalid_diff", "deleted file contains added lines")
                    added[current_file].add(new_line)
                    new_line += 1
                else:
                    raise ToolError("invalid_diff", "diff hunk body is malformed")
                if old_seen > old_count or new_seen > new_count:
                    raise ToolError("invalid_diff", "diff hunk counts do not match its body")
                index += 1
                if index < len(lines) and lines[index] == "\\ No newline at end of file":
                    index += 1
            saw_hunk = True
            continue
        if not line or line.startswith(metadata_prefixes):
            index += 1
            continue
        raise ToolError("invalid_diff", "diff contains unsupported or malformed content")
    if not saw_file or not saw_hunk:
        raise ToolError("invalid_diff", "diff contains no file hunks")
    return {path: frozenset(line_numbers) for path, line_numbers in added.items()}


def changed_locations(diff_text: str) -> list[ChangedLocation]:
    locations: list[ChangedLocation] = []
    for path, added_lines in sorted(parse_unified_diff(diff_text).items()):
        ordered = sorted(added_lines)
        if not ordered:
            continue
        start = previous = ordered[0]
        for line in ordered[1:]:
            if line == previous + 1:
                previous = line
                continue
            locations.append(ChangedLocation(path=path, line_start=start, line_end=previous))
            start = previous = line
        locations.append(ChangedLocation(path=path, line_start=start, line_end=previous))
        if len(locations) > 100:
            raise ToolError("invalid_diff", "diff exceeds the changed-range limit")
    return locations


def _failure(error: ToolError) -> ToolFailureResult:
    return ToolFailureResult(
        kind="tool_failure", code=error.code, message=error.message, retryable=error.retryable
    )


def execute_tool_plan(
    workspace: Workspace, plan: ToolPlan, test_runner: TestRunner, diff_text: str
) -> ToolRun:
    trace: list[ToolTraceEntry] = []

    def append(
        tool: ToolName,
        arguments: ToolArguments,
        result: ToolResult,
        started: float,
        provenance: list[TraceProvenance] | None = None,
    ) -> None:
        failed = isinstance(result, ToolFailureResult)
        trace.append(
            ToolTraceEntry(
                trace_id=f"tool-{len(trace) + 1:03d}",
                order=len(trace) + 1,
                tool=tool,
                status=ToolStatus.FAILED if failed else ToolStatus.SUCCEEDED,
                arguments=arguments,
                result=result,
                latency_ms=max(0, int((time.monotonic() - started) * 1000)),
                influence=TraceInfluence(),
                provenance=provenance or [],
            )
        )

    for reference in plan.read_files:
        started = time.monotonic()
        read_arguments = ReadFileArguments(kind="read_file", path=reference)
        try:
            read_result = workspace.read_result(reference)
            append(
                ToolName.READ_FILE,
                read_arguments,
                read_result,
                started,
                [TraceProvenance(path=reference)],
            )
        except ToolError as exc:
            append(ToolName.READ_FILE, read_arguments, _failure(exc), started)
            raise ToolExecutionError(exc, trace) from None

    for query in plan.searches:
        started = time.monotonic()
        search_arguments = SearchCodeArguments(kind="search_code", query=query)
        try:
            search_result = workspace.search_code(query)
            provenance = [
                TraceProvenance(path=match.path, line_start=match.line, line_end=match.line)
                for match in search_result.matches
            ]
            append(ToolName.SEARCH_CODE, search_arguments, search_result, started, provenance)
        except ToolError as exc:
            append(ToolName.SEARCH_CODE, search_arguments, _failure(exc), started)
            raise ToolExecutionError(exc, trace) from None

    verification: VerificationResult | None = None
    if plan.test_profile is not None:
        started = time.monotonic()
        test_arguments = RunTestsArguments(kind="run_tests", profile=plan.test_profile)
        try:
            if getattr(test_runner, "requires_sealed_snapshot", False):
                with workspace.sealed_snapshot() as snapshot:
                    execution = test_runner.run(snapshot, plan.test_profile)
            else:
                execution = test_runner.run(workspace.root, plan.test_profile)
            if execution.infrastructure_error:
                error = ToolError(
                    "cleanup_failed"
                    if execution.error_code == "cleanup_failed"
                    else "docker_infrastructure",
                    "isolated test infrastructure failed",
                )
                append(ToolName.RUN_TESTS, test_arguments, _failure(error), started)
                raise ToolExecutionError(error, trace)
            locations = changed_locations(diff_text)
            test_result = RunTestsResult(
                kind="run_tests_success",
                command=TEST_COMMAND,
                status=execution.status,
                profile=plan.test_profile,
                summary=_sanitize_text(execution.summary),
                output_truncated=execution.output_truncated,
            )
            provenance = []
            for location in locations:
                provenance.append(
                    TraceProvenance(
                        path=location.path,
                        line_start=location.line_start,
                        line_end=location.line_end,
                    )
                )
            append(ToolName.RUN_TESTS, test_arguments, test_result, started, provenance)
            verification = VerificationResult(
                profile=plan.test_profile,
                status=execution.status,
                summary=_sanitize_text(execution.summary),
                changed_locations=locations,
            )
        except ToolExecutionError:
            raise
        except ToolError as exc:
            append(ToolName.RUN_TESTS, test_arguments, _failure(exc), started)
            raise ToolExecutionError(exc, trace) from None
    return ToolRun(trace=trace, verification=verification)
