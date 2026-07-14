"""Deterministic generation and validation for versioned synthetic benchmarks."""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from codereviewops.benchmark_models import BenchmarkSuiteV1, SourceCaseV1, TaskEntryV1
from codereviewops.io import load_task
from codereviewops.models import BenchmarkTask, WorkflowState
from codereviewops.providers import ReplayProvider
from codereviewops.workflow import run_loaded_task

ORDER = [
    "http_retry_001",
    "python_tools_001",
    "pagination_boundary_001",
    "nullable_header_001",
    "timezone_rollover_001",
    "status_contract_001",
    "stable_order_001",
    "error_shape_001",
    "boundary_test_001",
    "exception_test_001",
    "retry_budget_test_001",
    "redirect_allowlist_001",
    "path_traversal_001",
    "token_redaction_001",
    "repeated_regex_001",
    "quadratic_membership_001",
    "swallowed_exception_001",
    "duplicated_policy_001",
    "misleading_default_001",
    "zero_timeout_001",
    "clean_parser_001",
    "clean_retry_001",
    "clean_validation_001",
    "clean_cache_001",
    "clean_security_001",
]
EXPECTED_PRIMARY = {
    "bug": 5,
    "requirement_mismatch": 4,
    "missing_test": 3,
    "security": 3,
    "performance": 2,
    "maintainability": 3,
    "negative": 5,
}
EXPECTED_DIFFICULTY = {"low": 9, "medium": 9, "high": 7}
EXPECTED_SEVERITY = {"critical": 2, "high": 9, "medium": 9, "low": 4}
MANAGED = ("fixtures", "workspaces", "replays", "suites")
MAX_SOURCE_BYTES = 64 * 1024
FORBIDDEN_NAMES = {
    ".git",
    "__pycache__",
    "agents.md",
    "01_codereviewops_spec.md",
    ".codex",
}
NONDETERMINISTIC = {"random", "secrets", "uuid", "time.time", "datetime.now"}
PUBLICATION_SIDECAR = re.compile(
    r"^\.(?P<target>[^/\\\r\n]+)\.codereviewops-(?:new|old)-(?P<pid>[1-9][0-9]{0,19})$"
)


class BenchmarkError(ValueError):
    """Safe benchmark generation or validation failure."""


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _validate_component(path: Path, root: Path | None = None) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BenchmarkError("benchmark filesystem validation failed") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag):
        raise BenchmarkError("benchmark paths cannot contain links or reparse points")
    if root is not None:
        lexical_root = _lexical(root)
        lexical_path = _lexical(path)
        try:
            lexical_path.relative_to(lexical_root)
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise BenchmarkError("benchmark path escapes its authority root") from exc


def _validate_existing_ancestor_chain(path: Path) -> None:
    path = _lexical(path)
    for component in reversed((path, *path.parents)):
        if not os.path.lexists(component):
            break
        _validate_component(component)


def _ensure_directory(path: Path) -> None:
    path = _lexical(path)
    chain = list(reversed((path, *path.parents)))
    for component in chain:
        if os.path.lexists(component):
            _validate_component(component)
            if not component.is_dir():
                raise BenchmarkError("benchmark directory path is invalid")
        else:
            try:
                component.mkdir()
            except OSError as exc:
                raise BenchmarkError("benchmark directory creation failed") from exc
            _validate_component(component)


def _ensure_managed_parent(directory: Path, root: Path) -> None:
    root = _lexical(root)
    directory = _lexical(directory)
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise BenchmarkError("benchmark output path escapes its authority root") from exc
    _validate_component(root)
    current = root
    for part in relative.parts:
        current /= part
        if os.path.lexists(current):
            _validate_component(current, root)
            if not current.is_dir():
                raise BenchmarkError("benchmark managed parent is invalid")
        else:
            try:
                current.mkdir()
            except OSError as exc:
                raise BenchmarkError("benchmark managed parent creation failed") from exc
            _validate_component(current, root)


def _preflight_output_sidecars(output_root: Path) -> None:
    if not os.path.lexists(output_root):
        return
    _validate_component(output_root)
    if not output_root.is_dir():
        raise BenchmarkError("benchmark output root is invalid")
    pending = [output_root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise BenchmarkError("benchmark output preflight failed") from exc
        for entry in entries:
            path = Path(entry.path)
            _validate_component(path, output_root)
            if PUBLICATION_SIDECAR.fullmatch(entry.name):
                raise BenchmarkError("benchmark publication sidecar exists")
            if entry.is_dir(follow_symlinks=False) and (
                directory != output_root or entry.name in MANAGED
            ):
                pending.append(path)


def _scan_tree(root: Path) -> list[Path]:
    _validate_component(root)
    found: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise BenchmarkError("benchmark directory scan failed") from exc
        for entry in entries:
            path = Path(entry.path)
            _validate_component(path, root)
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                found.append(path)
            else:
                raise BenchmarkError("benchmark trees require regular files")
    return sorted(found)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _read_case(case_dir: Path) -> SourceCaseV1:
    try:
        raw = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        return SourceCaseV1.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise BenchmarkError(f"invalid source case: {case_dir.name}") from exc


def _tree(root: Path) -> dict[str, bytes]:
    _validate_component(root)
    if not root.is_dir():
        raise BenchmarkError(f"missing source tree: {root.name}")
    result: dict[str, bytes] = {}
    for path in _scan_tree(root):
        relative_path = path.relative_to(root)
        if any(
            part.casefold() in FORBIDDEN_NAMES or part.casefold().startswith(".env")
            for part in relative_path.parts
        ):
            raise BenchmarkError("source tree contains a private or generated path")
        data = path.read_bytes()
        if not data or len(data) > MAX_SOURCE_BYTES or b"\0" in data:
            raise BenchmarkError("source file is empty, binary, or oversized")
        try:
            text = data.decode("utf-8")
        except UnicodeError as exc:
            raise BenchmarkError("source file is not UTF-8") from exc
        if "\r" in text or not text.endswith("\n"):
            raise BenchmarkError("source files require LF and a final newline")
        relative = relative_path.as_posix()
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                raise BenchmarkError("source Python is invalid") from exc
            if any(marker in text for marker in NONDETERMINISTIC):
                raise BenchmarkError("source Python contains nondeterminism")
        result[relative] = data
    if not result:
        raise BenchmarkError("source tree cannot be empty")
    return result


def _local_module_inventory(tree: dict[str, bytes]) -> set[str]:
    modules: set[str] = set()
    for name in tree:
        if not name.endswith(".py"):
            continue
        parts = name.removesuffix(".py").split("/")
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            modules.add(".".join(parts))
    return modules


def _relative_import_targets(file: str, node: ast.ImportFrom, local_modules: set[str]) -> set[str]:
    parts = file.removesuffix(".py").split("/")
    package = parts[:-1]
    package_name = ".".join(package)
    ascents = node.level - 1
    if not package or package_name not in local_modules or ascents >= len(package):
        raise BenchmarkError("relative source import escapes or lacks a local package")
    base = package[: len(package) - ascents]
    if node.module is not None:
        targets = {".".join([*base, *node.module.split(".")])}
    else:
        targets = {".".join([*base, alias.name]) for alias in node.names}
    if "*" in targets or not targets <= local_modules:
        raise BenchmarkError("relative source import target is absent from the after tree")
    return targets


def _validate_imports(tree: dict[str, bytes], local_modules: set[str]) -> None:
    local_roots = {module.split(".", 1)[0] for module in local_modules}
    allowed = set(sys.stdlib_module_names) | local_roots | {"__future__"}
    for name, data in tree.items():
        if not name.endswith(".py"):
            continue
        module = ast.parse(data.decode("utf-8"))
        imported: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    _relative_import_targets(name, node, local_modules)
                elif node.module is not None:
                    imported.add(node.module.split(".", 1)[0])
        if imported - allowed:
            raise BenchmarkError("source Python imports must be standard-library or local")


def _diff(before: dict[str, bytes], after: dict[str, bytes]) -> bytes:
    chunks: list[str] = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name, b"").decode().splitlines(keepends=True)
        new = after.get(name, b"").decode().splitlines(keepends=True)
        if old == new:
            continue
        old_name = f"a/{name}" if name in before else "/dev/null"
        new_name = f"b/{name}" if name in after else "/dev/null"
        chunks.append(f"diff --git a/{name} b/{name}\n")
        chunks.extend(difflib.unified_diff(old, new, fromfile=old_name, tofile=new_name, n=3))
    if not chunks:
        raise BenchmarkError("before and after trees must differ")
    return "".join(chunks).encode()


def _line_for_anchor(after: dict[str, bytes], file: str, anchor: str, diff: bytes) -> int:
    if file not in after:
        raise BenchmarkError("finding file is absent from after tree")
    lines = after[file].decode().splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if line == anchor]
    if len(matches) != 1 or diff.decode().splitlines().count("+" + anchor) != 1:
        raise BenchmarkError("finding anchor must be one unique added line")
    return matches[0]


def _render_case(case_dir: Path) -> tuple[SourceCaseV1, dict[str, bytes]]:
    case = _read_case(case_dir)
    if case.task_id != case_dir.name:
        raise BenchmarkError("source directory must match task_id")
    after = _tree(case_dir / "after")
    local_modules = _local_module_inventory(after)
    _validate_imports(after, local_modules)
    before = _tree(case_dir / "before")
    _validate_imports(before, local_modules)
    diff = _diff(before, after)
    for path in case.tool_plan.read_files:
        if path not in after:
            raise BenchmarkError("planned read file is absent")
    text = "\n".join(data.decode() for data in after.values())
    if any(query not in text for query in case.tool_plan.searches):
        raise BenchmarkError("planned search literal is absent")
    findings: list[dict[str, Any]] = []
    replay_findings: list[dict[str, Any]] = []
    for source in case.expected_findings:
        line = _line_for_anchor(after, source.file, source.anchor, diff)
        findings.append(
            {
                "category": source.category.value,
                "severity": source.severity.value,
                "file": source.file,
                "line_start": line,
                "line_end": line,
                "description": source.description,
            }
        )
        read_index = case.tool_plan.read_files.index(source.file) + 1
        replay_findings.append(
            {
                "title": source.title,
                "severity": source.severity.value,
                "category": source.category.value,
                "file": source.file,
                "line_start": line,
                "line_end": line,
                "evidence": source.evidence,
                "reasoning": source.reasoning,
                "recommendation": source.recommendation,
                "confidence": source.confidence,
                "evidence_trace_ids": [f"tool-{read_index:03d}"],
            }
        )
    task = {
        "schema_version": "1.2",
        "task_id": case.task_id,
        "title": case.title,
        "issue_description": case.issue_description,
        "diff_path": f"fixtures/{case.task_id}.diff",
        "expected_findings": findings,
        "must_not_find": case.must_not_find,
        "difficulty": case.difficulty.value,
        "tags": case.tags,
        "replay_response_path": f"replays/{case.task_id}.json",
        "workspace_path": f"workspaces/{case.task_id}",
        "tool_plan": case.tool_plan.model_dump(mode="json"),
    }
    BenchmarkTask.model_validate(task)
    replay = {
        "schema_version": "1.2",
        "summary": case.replay_summary,
        "overall_assessment": case.replay_assessment.value,
        "findings": replay_findings,
        "tests_run": [],
        "limitations": case.replay_limitations,
    }
    files = {
        f"{case.task_id}.json": _json_bytes(task),
        f"fixtures/{case.task_id}.diff": diff,
        f"replays/{case.task_id}.json": _json_bytes(replay),
    }
    files.update({f"workspaces/{case.task_id}/{name}": data for name, data in after.items()})
    return case, files


def _render(source: Path) -> dict[str, bytes]:
    source = _lexical(source)
    _validate_existing_ancestor_chain(source)
    _validate_component(source)
    if not source.is_dir():
        raise BenchmarkError("benchmark source root is invalid")
    _scan_tree(source)
    try:
        root_entries = sorted(os.scandir(source), key=lambda entry: entry.name)
    except OSError as exc:
        raise BenchmarkError("benchmark source scan failed") from exc
    case_dirs: list[Path] = []
    for entry in root_entries:
        path = Path(entry.path)
        _validate_component(path, source)
        if entry.is_dir(follow_symlinks=False):
            case_dirs.append(path)
    cases: dict[str, SourceCaseV1] = {}
    rendered: dict[str, bytes] = {}
    for case_dir in case_dirs:
        case, files = _render_case(case_dir)
        if case.task_id in cases:
            raise BenchmarkError("duplicate source task")
        cases[case.task_id] = case
        rendered.update(files)
    if list(cases) != sorted(ORDER) or set(cases) != set(ORDER):
        raise BenchmarkError("source inventory does not match the canonical 25 tasks")
    entries = [
        TaskEntryV1(
            task_id=task_id,
            task_path=f"{task_id}.json",
            primary_category=cases[task_id].primary_category,
            difficulty=cases[task_id].difficulty,
            negative=cases[task_id].primary_category == "negative",
        )
        for task_id in ORDER
    ]
    suite = BenchmarkSuiteV1(
        schema_version="1.0",
        suite_id="m4_25",
        suite_version="1.0.0",
        expected_task_count=25,
        tasks=entries,
    )
    rendered["suites/m4_25.json"] = _json_bytes(suite.model_dump(mode="json"))
    return rendered


def render(source: Path) -> dict[str, bytes]:
    try:
        return _render(source)
    except BenchmarkError:
        raise
    except OSError as exc:
        raise BenchmarkError("benchmark source processing failed") from exc


def _actual_files(output_root: Path) -> set[str]:
    _validate_component(output_root)
    actual: set[str] = set()
    try:
        entries = sorted(os.scandir(output_root), key=lambda entry: entry.name)
    except OSError as exc:
        raise BenchmarkError("benchmark output scan failed") from exc
    for entry in entries:
        path = Path(entry.path)
        _validate_component(path, output_root)
        if entry.is_file(follow_symlinks=False) and path.suffix == ".json":
            actual.add(path.name)
        elif entry.is_dir(follow_symlinks=False) and entry.name in MANAGED:
            actual.update(child.relative_to(output_root).as_posix() for child in _scan_tree(path))
    return actual


def _generate(source: Path, output_root: Path, *, check: bool = False) -> None:
    output_root = _lexical(output_root)
    _validate_existing_ancestor_chain(output_root)
    _preflight_output_sidecars(output_root)
    expected = render(source)
    if check:
        actual = _actual_files(output_root)
        if actual != set(expected):
            raise BenchmarkError("generated benchmark inventory is stale")
        for relative, data in expected.items():
            target = output_root / relative
            _validate_component(target, output_root)
            if target.read_bytes() != data:
                raise BenchmarkError(f"generated benchmark is stale: {relative}")
        return
    _ensure_directory(output_root)
    _validate_component(output_root)
    stale = _actual_files(output_root) - set(expected)
    targets = {relative: output_root / relative for relative in expected}
    stale_targets = [output_root / relative for relative in sorted(stale)]
    nonce = str(os.getpid())
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    mutated: list[Path] = []

    def sidecar(target: Path, purpose: str) -> Path:
        return target.with_name(f".{target.name}.codereviewops-{purpose}-{nonce}")

    original_error: BaseException | None = None
    restore_failures: set[Path] = set()
    cleanup_interrupt: BaseException | None = None
    cleanup_failures: list[OSError] = []
    try:
        for relative in sorted(expected):
            target = targets[relative]
            _ensure_managed_parent(target.parent, output_root)
            temporary = sidecar(target, "new")
            if os.path.lexists(temporary):
                raise BenchmarkError("benchmark publication sidecar already exists")
            staged[target] = temporary
            with temporary.open("xb") as handle:
                handle.write(expected[relative])
                handle.flush()
                os.fsync(handle.fileno())
            _validate_component(temporary, output_root)

        for target in [*targets.values(), *stale_targets]:
            if not os.path.lexists(target):
                continue
            _validate_component(target, output_root)
            backup = sidecar(target, "old")
            if os.path.lexists(backup):
                raise BenchmarkError("benchmark publication sidecar already exists")
            backups[target] = backup
            os.link(target, backup)
            _validate_component(backup, output_root)

        for target, temporary in staged.items():
            _validate_component(target.parent, output_root)
            _validate_component(temporary, output_root)
            mutated.append(target)
            os.replace(temporary, target)
        for target in stale_targets:
            _validate_component(target, output_root)
            mutated.append(target)
            target.unlink()
    except BaseException as caught:
        original_error = caught
        for target in reversed(mutated):
            prior = backups.get(target)
            try:
                _validate_component(target.parent, output_root)
                if prior is not None and os.path.lexists(prior):
                    _validate_component(prior, output_root)
                    os.replace(prior, target)
                else:
                    target.unlink(missing_ok=True)
            except BaseException:
                restore_failures.add(target)
                continue
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError as caught:
                cleanup_failures.append(caught)
                continue
            except BaseException as caught:
                if cleanup_interrupt is None:
                    cleanup_interrupt = caught
        for target, backup in backups.items():
            if target in restore_failures:
                continue
            try:
                backup.unlink(missing_ok=True)
            except OSError as caught:
                cleanup_failures.append(caught)
                continue
            except BaseException as caught:
                if cleanup_interrupt is None:
                    cleanup_interrupt = caught

    if original_error is not None:
        if isinstance(original_error, Exception):
            raise BenchmarkError("benchmark publication failed") from original_error
        raise original_error.with_traceback(original_error.__traceback__)
    if cleanup_interrupt is not None:
        raise cleanup_interrupt.with_traceback(cleanup_interrupt.__traceback__)
    if cleanup_failures:
        raise BenchmarkError("benchmark publication cleanup failed") from cleanup_failures[0]


def generate(source: Path, output_root: Path, *, check: bool = False) -> None:
    try:
        _generate(source, output_root, check=check)
    except BenchmarkError:
        raise
    except OSError as exc:
        raise BenchmarkError("benchmark generation failed") from exc


def _validate(suite_path: Path, *, execute_replays: bool = True) -> None:
    suite_path = _lexical(suite_path)
    _validate_existing_ancestor_chain(suite_path)
    root = suite_path.parent.parent
    _validate_existing_ancestor_chain(root)
    _validate_component(root)
    _validate_component(suite_path.parent, root)
    _validate_component(suite_path, root)
    try:
        suite = BenchmarkSuiteV1.model_validate_json(suite_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise BenchmarkError("suite manifest is invalid") from exc
    if [entry.task_id for entry in suite.tasks] != ORDER:
        raise BenchmarkError("suite task order is invalid")
    primary = Counter(str(entry.primary_category) for entry in suite.tasks)
    generate(root.parent / "source", root, check=True)
    difficulty = Counter(entry.difficulty.value for entry in suite.tasks)
    severity: Counter[str] = Counter()
    labels = positives = negatives = 0
    for entry in suite.tasks:
        loaded = load_task(root / entry.task_path)
        task = loaded.task
        if task.task_id != entry.task_id or task.difficulty != entry.difficulty:
            raise BenchmarkError("suite metadata does not match task")
        if entry.negative != (not task.expected_findings):
            raise BenchmarkError("suite negative metadata does not match task")
        labels += len(task.expected_findings)
        positives += bool(task.expected_findings)
        negatives += not task.expected_findings
        severity.update(
            finding.severity.value for finding in task.expected_findings if finding.severity
        )
        if execute_replays:
            _, artifact = run_loaded_task(loaded, ReplayProvider(loaded.replay_path))
            if (
                not artifact.evaluation.task_success
                or artifact.final_state != WorkflowState.COMPLETE
            ):
                raise BenchmarkError(f"replay validation failed: {task.task_id}")
    if (
        dict(primary) != EXPECTED_PRIMARY
        or dict(difficulty) != EXPECTED_DIFFICULTY
        or dict(severity) != EXPECTED_SEVERITY
        or labels != 24
        or positives != 20
        or negatives != 5
    ):
        raise BenchmarkError("suite distributions are invalid")


def validate(suite_path: Path, *, execute_replays: bool = True) -> None:
    try:
        _validate(suite_path, execute_replays=execute_replays)
    except BenchmarkError:
        raise
    except OSError as exc:
        raise BenchmarkError("benchmark validation failed") from exc
