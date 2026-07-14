from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

import codereviewops.benchmarking as benchmarking_module
from codereviewops.benchmark_models import BenchmarkSuiteV1, SourceCaseV1
from codereviewops.benchmarking import (
    EXPECTED_DIFFICULTY,
    EXPECTED_PRIMARY,
    EXPECTED_SEVERITY,
    BenchmarkError,
    generate,
    render,
    validate,
)
from codereviewops.cli import app
from codereviewops.evaluation import evaluate_review
from codereviewops.models import BenchmarkTask, ReviewReport

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "benchmarks" / "source"
OUTPUT = ROOT / "benchmarks" / "tasks"
SUITE = OUTPUT / "suites" / "m4_25.json"


def _copy_source(tmp_path: Path) -> Path:
    return Path(shutil.copytree(SOURCE, tmp_path / "source"))


def _case_json(source: Path, task_id: str = "http_retry_001") -> tuple[Path, dict[str, object]]:
    path = source / task_id / "case.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_frozen_legacy_task_schemas_remain_valid() -> None:
    fixtures = ROOT / "tests" / "fixtures" / "benchmarks"
    legacy_10 = BenchmarkTask.model_validate_json(
        (fixtures / "legacy_task_10.json").read_text(encoding="utf-8")
    )
    legacy_11 = BenchmarkTask.model_validate_json(
        (fixtures / "legacy_task_11.json").read_text(encoding="utf-8")
    )
    assert legacy_10.schema_version == "1.0" and legacy_10.tool_plan is None
    assert legacy_11.schema_version == "1.1" and legacy_11.tool_plan is not None
    assert all(finding.severity is None for finding in legacy_10.expected_findings)
    assert all(finding.severity is None for finding in legacy_11.expected_findings)


def test_source_cases_are_exact_explicit_and_line_number_free() -> None:
    directories = sorted(path for path in SOURCE.iterdir() if path.is_dir())
    assert len(directories) == 25
    for directory in directories:
        raw = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        case = SourceCaseV1.model_validate(raw)
        assert case.task_id == directory.name
        assert "line_start" not in json.dumps(raw)
        assert "line_end" not in json.dumps(raw)
        assert case.tool_plan.test_profile is None
        assert case.expected_test_status is None


def test_canonical_suite_has_exact_inventory_and_distributions() -> None:
    suite = BenchmarkSuiteV1.model_validate_json(SUITE.read_text(encoding="utf-8"))
    tasks = [
        BenchmarkTask.model_validate_json((OUTPUT / entry.task_path).read_text(encoding="utf-8"))
        for entry in suite.tasks
    ]
    primary = Counter(str(entry.primary_category) for entry in suite.tasks)
    difficulty = Counter(entry.difficulty.value for entry in suite.tasks)
    severity = Counter(
        finding.severity.value
        for task in tasks
        for finding in task.expected_findings
        if finding.severity is not None
    )
    assert dict(primary) == EXPECTED_PRIMARY
    assert dict(difficulty) == EXPECTED_DIFFICULTY
    assert dict(severity) == EXPECTED_SEVERITY
    assert sum(len(task.expected_findings) for task in tasks) == 24
    assert sum(bool(task.expected_findings) for task in tasks) == 20


def test_generator_is_reproducible_and_rejects_stale_extras(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    generate(SOURCE, output, check=True)
    extra = output / "fixtures" / "stale.diff"
    extra.write_text("stale\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="inventory is stale"):
        generate(SOURCE, output, check=True)


def test_generator_rolls_back_when_atomic_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    stale = output / "fixtures" / "stale.diff"
    stale.write_text("preserve me\n", encoding="utf-8")
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    real_replace = benchmarking_module.os.replace
    calls = 0

    def fail_second_install(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_replace(source, target)

    monkeypatch.setattr(benchmarking_module.os, "replace", fail_second_install)
    with pytest.raises(BenchmarkError, match="benchmark publication failed"):
        generate(SOURCE, output)

    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(output.rglob("*.codereviewops-*-*"))


def test_keyboard_interrupt_during_install_restores_exact_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    stale = output / "fixtures" / "stale.diff"
    stale.write_bytes(b"preserve me\n")
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    real_replace = benchmarking_module.os.replace
    calls = 0
    interrupt = KeyboardInterrupt("stop publication")

    def interrupt_second_install(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt
        real_replace(source, target)

    monkeypatch.setattr(benchmarking_module.os, "replace", interrupt_second_install)
    with pytest.raises(KeyboardInterrupt) as captured:
        generate(SOURCE, output)
    assert captured.value is interrupt
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(output.rglob("*.codereviewops-*-*"))


def test_fsync_failure_cleans_partially_written_registered_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError("private fsync failure")

    monkeypatch.setattr(benchmarking_module.os, "fsync", fail_fsync)
    with pytest.raises(BenchmarkError, match="benchmark publication failed"):
        generate(SOURCE, output)
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(output.rglob("*.codereviewops-*-*"))


def test_successful_install_reports_staged_cleanup_failure_and_leaves_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    real_replace = benchmarking_module.os.replace
    real_unlink = Path.unlink
    orphan: list[Path] = []

    def replace_and_recreate_first_sidecar(source: Path, target: Path) -> None:
        real_replace(source, target)
        if not orphan:
            source.write_bytes(b"identifiable staged sidecar\n")
            orphan.append(source)

    def fail_orphan_cleanup(path: Path, missing_ok: bool = False) -> None:
        if orphan and path == orphan[0]:
            raise OSError("private staged cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(benchmarking_module.os, "replace", replace_and_recreate_first_sidecar)
    monkeypatch.setattr(Path, "unlink", fail_orphan_cleanup)
    with pytest.raises(BenchmarkError, match="benchmark publication cleanup failed"):
        generate(SOURCE, output)
    assert len(orphan) == 1
    assert orphan[0].read_bytes() == b"identifiable staged sidecar\n"

    monkeypatch.undo()
    with pytest.raises(BenchmarkError, match="benchmark publication sidecar exists"):
        generate(SOURCE, output)
    assert orphan[0].read_bytes() == b"identifiable staged sidecar\n"
    orphan[0].unlink()


def test_successful_install_reports_backup_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    real_link = benchmarking_module.os.link
    real_unlink = Path.unlink
    retained: list[Path] = []

    def capture_first_backup(source: Path, target: Path) -> None:
        real_link(source, target)
        if not retained:
            retained.append(target)

    def fail_backup_cleanup(path: Path, missing_ok: bool = False) -> None:
        if retained and path == retained[0]:
            raise OSError("private backup cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(benchmarking_module.os, "link", capture_first_backup)
    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    with pytest.raises(BenchmarkError, match="benchmark publication cleanup failed"):
        generate(SOURCE, output)
    assert len(retained) == 1
    assert retained[0].is_file()

    monkeypatch.undo()
    with pytest.raises(BenchmarkError, match="benchmark publication sidecar exists"):
        generate(SOURCE, output)
    assert retained[0].is_file()
    retained[0].unlink()


def test_keyboard_interrupt_remains_primary_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    real_unlink = Path.unlink
    interrupt = KeyboardInterrupt("stop staging")

    def interrupt_fsync(file_descriptor: int) -> None:
        raise interrupt

    def fail_new_sidecar_cleanup(path: Path, missing_ok: bool = False) -> None:
        if ".codereviewops-new-" in path.name:
            raise OSError("private cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(benchmarking_module.os, "fsync", interrupt_fsync)
    monkeypatch.setattr(Path, "unlink", fail_new_sidecar_cleanup)
    with pytest.raises(KeyboardInterrupt) as captured:
        generate(SOURCE, output)
    assert captured.value is interrupt
    assert list(output.rglob("*.codereviewops-new-*"))


def test_ordinary_failure_remains_primary_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    real_unlink = Path.unlink

    def fail_fsync(file_descriptor: int) -> None:
        raise OSError("private original failure")

    def fail_new_sidecar_cleanup(path: Path, missing_ok: bool = False) -> None:
        if ".codereviewops-new-" in path.name:
            raise OSError("private cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(benchmarking_module.os, "fsync", fail_fsync)
    monkeypatch.setattr(Path, "unlink", fail_new_sidecar_cleanup)
    with pytest.raises(BenchmarkError, match="benchmark publication failed") as captured:
        generate(SOURCE, output)
    assert "cleanup" not in str(captured.value)
    assert list(output.rglob("*.codereviewops-new-*"))


def test_rollback_continues_after_restore_failure_and_retains_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    ordered = sorted(render(SOURCE))
    first = output / ordered[0]
    second = output / ordered[1]
    first.write_bytes(b"original-first\n")
    second.write_bytes(b"original-second\n")
    real_replace = benchmarking_module.os.replace
    calls = 0

    def fail_publication_and_one_restore(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls in {3, 5}:
            raise PermissionError("private rollback path")
        real_replace(source, target)

    monkeypatch.setattr(benchmarking_module.os, "replace", fail_publication_and_one_restore)
    with pytest.raises(BenchmarkError, match="benchmark publication failed"):
        generate(SOURCE, output)

    assert first.read_bytes() == b"original-first\n"
    assert second.read_bytes() != b"original-second\n"
    retained = list(second.parent.glob(f".{second.name}.codereviewops-old-*"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == b"original-second\n"
    assert not list(first.parent.glob(f".{first.name}.codereviewops-*-*"))


def test_validator_executes_all_direct_replays() -> None:
    validate(SUITE)


def test_benchmark_cli_generate_check_and_validate() -> None:
    runner = CliRunner()
    checked = runner.invoke(
        app,
        [
            "benchmark",
            "generate",
            "--source",
            str(SOURCE),
            "--output-root",
            str(OUTPUT),
            "--check",
        ],
    )
    assert checked.exit_code == 0
    validated = runner.invoke(app, ["benchmark", "validate", "--suite", str(SUITE)])
    assert validated.exit_code == 0


@pytest.mark.parametrize(
    "statement",
    [
        "import acme_unknown\n",
        "from acme_unknown import feature\n",
        "import json, acme_unknown\n",
    ],
)
def test_source_import_policy_rejects_unknown_roots(tmp_path: Path, statement: str) -> None:
    source = _copy_source(tmp_path)
    for tree in ("before", "after"):
        path = source / "http_retry_001" / tree / "solution.py"
        path.write_bytes((statement + path.read_text(encoding="utf-8")).encode("utf-8"))
    with pytest.raises(BenchmarkError, match="standard-library or local"):
        render(source)


def test_source_import_policy_allows_after_tree_local_modules(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    for tree in ("before", "after"):
        root = source / "http_retry_001" / tree
        solution = root / "solution.py"
        solution.write_bytes(
            ("import helper\n" + solution.read_text(encoding="utf-8")).encode("utf-8")
        )
        (root / "helper.py").write_bytes(b"VALUE = 1\n")
    render(source)


def test_relative_imports_resolve_within_after_tree_package(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    for tree in ("before", "after"):
        package = source / "http_retry_001" / tree / "localpkg"
        package.mkdir()
        (package / "__init__.py").write_bytes(b"# local package\n")
        (package / "helper.py").write_bytes(b"VALUE = 1\n")
        (package / "consumer.py").write_bytes(b"from . import helper\nfrom .helper import VALUE\n")
    render(source)


@pytest.mark.parametrize(
    "statement",
    [
        "from . import missing\n",
        "from .missing import VALUE\n",
        "from ..helper import VALUE\n",
    ],
)
def test_relative_imports_reject_missing_or_escaping_targets(
    tmp_path: Path, statement: str
) -> None:
    source = _copy_source(tmp_path)
    for tree in ("before", "after"):
        package = source / "http_retry_001" / tree / "localpkg"
        package.mkdir()
        (package / "__init__.py").write_bytes(b"# local package\n")
        (package / "helper.py").write_bytes(b"VALUE = 1\n")
        (package / "consumer.py").write_bytes(statement.encode("utf-8"))
    with pytest.raises(BenchmarkError, match="relative source import"):
        render(source)


@pytest.mark.parametrize(
    "private_path",
    [
        ".ENV",
        ".env.local",
        ".Git/config",
        "__PyCache__/cache.py",
        "AGENTS.md",
        "01_CodeReviewOps_SPEC.md",
        ".CoDeX/settings.json",
    ],
)
def test_source_tree_rejects_private_names_case_insensitively(
    tmp_path: Path, private_path: str
) -> None:
    source = _copy_source(tmp_path)
    path = source / "http_retry_001" / "after" / private_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"private\n")
    with pytest.raises(BenchmarkError, match="private or generated"):
        render(source)


@pytest.mark.parametrize("phrase", ["   ", "\u200b", "safe\nunsafe", "x" * 257])
def test_source_case_rejects_ineffective_prohibited_phrases(phrase: str) -> None:
    _, raw = _case_json(SOURCE)
    raw["must_not_find"] = [phrase]
    with pytest.raises(ValueError, match=r"prohibited phrases|too_long"):
        SourceCaseV1.model_validate(raw)


@pytest.mark.parametrize(
    "phrases",
    [
        ["SQL Injection", "  sql   injection  "],
        ["K", "\N{KELVIN SIGN}"],
        [
            "SQL",
            (
                "\N{FULLWIDTH LATIN CAPITAL LETTER S}"
                "\N{FULLWIDTH LATIN CAPITAL LETTER Q}"
                "\N{FULLWIDTH LATIN CAPITAL LETTER L}"
            ),
        ],
    ],
)
def test_prohibited_phrases_reject_evaluator_semantic_duplicates(
    phrases: list[str],
) -> None:
    _, raw = _case_json(SOURCE)
    raw["must_not_find"] = phrases
    with pytest.raises(ValueError, match="semantically unique"):
        SourceCaseV1.model_validate(raw)


def test_normalized_prohibited_phrase_remains_evaluator_effective() -> None:
    _, raw = _case_json(SOURCE)
    raw["must_not_find"] = ["  retries client errors  "]
    case = SourceCaseV1.model_validate(raw)
    report = ReviewReport.model_validate_json(
        (OUTPUT / "replays" / "http_retry_001.json").read_text(encoding="utf-8")
    )
    result = evaluate_review([], case.must_not_find, report)
    assert case.must_not_find == ["retries client errors"]
    assert result.prohibited_hits[0].phrase == "retries client errors"


@pytest.mark.parametrize(
    ("source_phrase", "finding_title"),
    [
        (
            "SQL Injection",
            (
                "\N{FULLWIDTH LATIN CAPITAL LETTER S}"
                "\N{FULLWIDTH LATIN CAPITAL LETTER Q}"
                "\N{FULLWIDTH LATIN CAPITAL LETTER L} injection vulnerability"
            ),
        ),
        (
            (
                "\N{FULLWIDTH LATIN CAPITAL LETTER S}"
                "\N{FULLWIDTH LATIN CAPITAL LETTER Q}"
                "\N{FULLWIDTH LATIN CAPITAL LETTER L}   Injection"
            ),
            "SQL injection vulnerability",
        ),
    ],
)
def test_nfkc_prohibited_phrase_matches_finding_text_once(
    source_phrase: str, finding_title: str
) -> None:
    _, raw = _case_json(SOURCE)
    raw["must_not_find"] = [source_phrase]
    case = SourceCaseV1.model_validate(raw)

    report_raw = json.loads(
        (OUTPUT / "replays" / "http_retry_001.json").read_text(encoding="utf-8")
    )
    report_raw["findings"] = [report_raw["findings"][0]]
    report_raw["findings"][0]["title"] = finding_title
    report = ReviewReport.model_validate(report_raw)

    result = evaluate_review([], case.must_not_find, report)
    assert len(result.prohibited_hits) == 1
    assert result.prohibited_hits[0].phrase == source_phrase.strip()


def test_positive_primary_category_must_match_a_finding() -> None:
    _, raw = _case_json(SOURCE)
    raw["primary_category"] = "performance"
    with pytest.raises(ValueError, match="primary category"):
        SourceCaseV1.model_validate(raw)


def test_negative_case_requires_empty_findings_and_pass_replay() -> None:
    _, raw = _case_json(SOURCE, "clean_parser_001")
    raw["replay_assessment"] = "needs_changes"
    with pytest.raises(ValueError, match="negative cases require a pass"):
        SourceCaseV1.model_validate(raw)

    _, raw = _case_json(SOURCE, "clean_parser_001")
    _, positive = _case_json(SOURCE)
    raw["expected_findings"] = positive["expected_findings"]
    raw["tool_plan"] = positive["tool_plan"]
    with pytest.raises(ValueError, match="negative cases must have no findings"):
        SourceCaseV1.model_validate(raw)


def test_unplanned_finding_file_is_safe_cli_failure(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    case_path, raw = _case_json(source)
    raw["tool_plan"]["read_files"].remove(raw["expected_findings"][0]["file"])
    _write_json(case_path, raw)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "generate",
            "--source",
            str(source),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 2
    assert result.output == "error: benchmark generation failed\n"
    assert "Traceback" not in result.output


def test_missing_search_literal_is_rejected(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    case_path, raw = _case_json(source)
    raw["tool_plan"]["searches"] = ["literal absent from the after tree"]
    _write_json(case_path, raw)
    with pytest.raises(BenchmarkError, match="search literal is absent"):
        render(source)


def test_missing_finding_anchor_is_rejected(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    case_path, raw = _case_json(source)
    raw["expected_findings"][0]["anchor"] = "anchor absent from the after tree"
    _write_json(case_path, raw)
    with pytest.raises(BenchmarkError, match="anchor must be one unique added line"):
        render(source)


@pytest.mark.parametrize(
    "payload",
    [b"binary\x00data", b"x" * (64 * 1024 + 1)],
    ids=["binary", "oversized"],
)
def test_binary_and_oversized_sources_are_rejected(tmp_path: Path, payload: bytes) -> None:
    source = _copy_source(tmp_path)
    (source / "http_retry_001" / "after" / "unsafe.dat").write_bytes(payload)
    with pytest.raises(BenchmarkError, match="empty, binary, or oversized"):
        render(source)


def test_source_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    link = source / "http_retry_001" / "after" / "linked.py"
    try:
        os.symlink(source / "http_retry_001" / "after" / "solution.py", link)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        render(source)


def _directory_symlink(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")


@pytest.mark.parametrize("scope", ["root", "case", "before"])
def test_source_authority_rejects_directory_links_without_external_reads(
    tmp_path: Path, scope: str
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_bytes(b"unchanged\n")
    if scope == "root":
        source = tmp_path / "source-link"
        _directory_symlink(external, source)
    else:
        source = _copy_source(tmp_path)
        target = source / "http_retry_001"
        if scope == "before":
            target = target / "before"
        shutil.rmtree(target)
        _directory_symlink(external, target)
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        render(source)
    assert marker.read_bytes() == b"unchanged\n"


@pytest.mark.parametrize("managed", ["fixtures", "workspaces"])
def test_output_authority_rejects_managed_links_without_external_writes(
    tmp_path: Path, managed: str
) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_bytes(b"unchanged\n")
    shutil.rmtree(output / managed)
    _directory_symlink(external, output / managed)
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        generate(SOURCE, output, check=True)
    assert list(external.iterdir()) == [marker]
    assert marker.read_bytes() == b"unchanged\n"


def test_output_root_link_is_rejected_without_external_writes(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "output-link"
    _directory_symlink(external, output)
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        generate(SOURCE, output)
    assert list(external.iterdir()) == []


def test_windows_source_reparse_point_is_rejected_when_available(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows reparse attributes are platform-specific")
    external = tmp_path / "external"
    external.mkdir()
    link = tmp_path / "source-link"
    _directory_symlink(external, link)
    attributes = getattr(link.lstat(), "st_file_attributes", 0)
    reparse = getattr(benchmarking_module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    assert attributes & reparse
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        render(link)


@pytest.mark.parametrize(
    ("attribute", "arguments", "expected"),
    [
        (
            "_generate",
            ["benchmark", "generate", "--source", "source", "--output-root", "output"],
            "error: benchmark generation failed\n",
        ),
        (
            "_generate",
            [
                "benchmark",
                "generate",
                "--source",
                "source",
                "--output-root",
                "output",
                "--check",
            ],
            "error: benchmark generation failed\n",
        ),
        (
            "_validate",
            ["benchmark", "validate", "--suite", "private-suite.json"],
            "error: benchmark validation failed\n",
        ),
    ],
)
def test_benchmark_cli_redacts_path_bearing_failures(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    arguments: list[str],
    expected: str,
) -> None:
    private_path = "C:/private/customer/secret.json"

    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError(private_path)

    monkeypatch.setattr(benchmarking_module, attribute, fail)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert result.output == expected
    assert private_path not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("boundary", "message"), [("generate", "generation"), ("validate", "validation")]
)
def test_benchmark_boundaries_convert_path_bearing_oserrors(
    monkeypatch: pytest.MonkeyPatch, boundary: str, message: str
) -> None:
    private_path = "C:/private/customer/secret.json"

    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError(private_path)

    monkeypatch.setattr(benchmarking_module, f"_{boundary}", fail)
    function = getattr(benchmarking_module, boundary)
    with pytest.raises(BenchmarkError, match=f"benchmark {message} failed") as captured:
        function(Path("unused"), Path("unused")) if boundary == "generate" else function(
            Path("unused")
        )
    assert private_path not in str(captured.value)


def test_render_rejects_symlink_ancestor_before_external_read(tmp_path: Path) -> None:
    external = tmp_path / "external"
    (external / "source").mkdir(parents=True)
    marker = external / "marker.txt"
    marker.write_bytes(b"unchanged\n")
    ancestor = tmp_path / "authority-link"
    _directory_symlink(external, ancestor)
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        render(ancestor / "source")
    assert marker.read_bytes() == b"unchanged\n"


def test_generate_check_rejects_symlink_ancestor_without_external_write(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    (external / "output").mkdir(parents=True)
    marker = external / "marker.txt"
    marker.write_bytes(b"unchanged\n")
    ancestor = tmp_path / "authority-link"
    _directory_symlink(external, ancestor)
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        generate(SOURCE, ancestor / "output", check=True)
    assert marker.read_bytes() == b"unchanged\n"
    assert sorted(path.name for path in external.iterdir()) == ["marker.txt", "output"]


def test_validate_rejects_symlink_ancestor_before_external_read(tmp_path: Path) -> None:
    external = tmp_path / "external"
    suite = external / "tasks" / "suites" / "suite.json"
    suite.parent.mkdir(parents=True)
    suite.write_bytes(b"private marker\n")
    marker = external / "marker.txt"
    marker.write_bytes(b"unchanged\n")
    ancestor = tmp_path / "authority-link"
    _directory_symlink(external, ancestor)
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        validate(ancestor / "tasks" / "suites" / "suite.json")
    assert marker.read_bytes() == b"unchanged\n"
    assert suite.read_bytes() == b"private marker\n"


def test_windows_reparse_ancestor_is_rejected_when_available(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows reparse attributes are platform-specific")
    external = tmp_path / "external"
    (external / "source").mkdir(parents=True)
    ancestor = tmp_path / "authority-link"
    _directory_symlink(external, ancestor)
    attributes = getattr(ancestor.lstat(), "st_file_attributes", 0)
    reparse = getattr(benchmarking_module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    assert attributes & reparse
    with pytest.raises(BenchmarkError, match="links or reparse points"):
        render(ancestor / "source")


@pytest.mark.parametrize("purpose", ["new", "old"])
def test_cross_process_root_sidecar_preflight_runs_before_render_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, purpose: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    pid = os.getpid() + 100_000
    sidecar = output / f".task.json.codereviewops-{purpose}-{pid}"
    sidecar.write_bytes(b"foreign transaction\n")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def render_must_not_run(source: Path) -> dict[str, bytes]:
        raise AssertionError("render ran before publication preflight")

    monkeypatch.setattr(benchmarking_module, "render", render_must_not_run)
    with pytest.raises(BenchmarkError, match="benchmark publication sidecar exists"):
        generate(Path("unused-source"), output, check=purpose == "old")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_cross_process_managed_sidecar_preflight_rejects_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    managed = output / "fixtures" / "nested"
    managed.mkdir(parents=True)
    pid = os.getpid() + 100_000
    sidecar = managed / f".task.diff.codereviewops-new-{pid}"
    sidecar.write_bytes(b"foreign managed transaction\n")
    before = sidecar.read_bytes()

    def render_must_not_run(source: Path) -> dict[str, bytes]:
        raise AssertionError("render ran before publication preflight")

    monkeypatch.setattr(benchmarking_module, "render", render_must_not_run)
    with pytest.raises(BenchmarkError, match="benchmark publication sidecar exists"):
        generate(Path("unused-source"), output)
    assert sidecar.read_bytes() == before


def test_similar_malformed_sidecars_follow_normal_inventory_rules(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    generate(SOURCE, output)
    root_similar = output / ".task.json.codereviewops-new-not-a-pid"
    root_similar.write_bytes(b"unmanaged root file\n")
    generate(SOURCE, output, check=True)

    managed_similar = output / "fixtures" / ".task.diff.codereviewops-old-not-a-pid"
    managed_similar.write_bytes(b"managed stale file\n")
    with pytest.raises(BenchmarkError, match="generated benchmark inventory is stale") as captured:
        generate(SOURCE, output, check=True)
    assert "publication sidecar" not in str(captured.value)
