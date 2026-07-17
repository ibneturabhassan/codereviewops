from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from codereviewops import benchmark_baseline, benchmark_runner
from codereviewops.benchmark_baseline import (
    compare_baseline,
    create_baseline,
    load_baseline,
    stable_result_hash,
)
from codereviewops.benchmark_metrics import aggregate_metrics, gate_failures, task_metrics
from codereviewops.benchmark_models import (
    BenchmarkBaselineV1,
    BenchmarkMatrixV1,
    BenchmarkRunV1,
    RunVariantV1,
    StableAggregateMetricsV1,
    StableBenchmarkRunV1,
    StableTaskMetricsV1,
    ThresholdProfileV1,
)
from codereviewops.benchmark_runner import BenchmarkRunError, run_benchmark
from codereviewops.benchmark_selection import load_suite, select_tasks
from codereviewops.cli import app
from codereviewops.io import InputError
from codereviewops.models import (
    ReviewContext,
    RunArtifact,
    ToolFailureResult,
    ToolStatus,
)
from codereviewops.providers import ReplayProvider

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "benchmarks/tasks/suites/m4_25.json"
MATRIX = ROOT / "benchmarks/matrices/m4_replay_transport_v1.json"


def _walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value, *[item for child in value.values() for item in _walk(child)]]
    if isinstance(value, list):
        return [value, *[item for child in value for item in _walk(child)]]
    return [value]


def test_variant_contract_rejects_replay_model_and_live_zero_budget() -> None:
    common = {
        "variant_id": "variant",
        "prompt_version": "review-tools-v1",
        "agent_version": "tool-agent-v1",
        "planner_version": "manifest-v1",
        "tool_transport": "direct",
    }
    with pytest.raises(ValidationError):
        RunVariantV1(provider="replay", model="x", request_budget=0, **common)
    with pytest.raises(ValidationError):
        RunVariantV1(provider="groq", model="x", request_budget=0, **common)


def test_filters_intersect_and_preserve_suite_order() -> None:
    suite = load_suite(SUITE)
    selected = select_tasks(
        suite,
        categories={suite.tasks[0].primary_category},
        difficulties={suite.tasks[0].difficulty},
        polarity="positive",
    )
    assert [entry.task_id for entry in selected] == ["http_retry_001", "status_contract_001"]
    with pytest.raises(InputError):
        select_tasks(suite, task_ids={"missing"})


def test_replay_provider_receives_no_golden_fields_or_values(tmp_path: Path) -> None:
    seen: list[ReviewContext] = []

    class Spy:
        def __init__(self, replay: Path) -> None:
            self.delegate = ReplayProvider(replay)

        def review(self, context: ReviewContext):
            seen.append(context)
            payload = context.model_dump(mode="json")
            flattened = _walk(payload)
            assert "expected_findings" not in payload
            assert "must_not_find" not in payload
            assert not any(item == "clean parser must remain finding-free" for item in flattened)
            return self.delegate.review(context)

    output = tmp_path / "result"
    run, code = run_benchmark(
        suite_path=SUITE,
        output_dir=output,
        task_ids={"clean_parser_001"},
        provider_factory=lambda _variant, replay: Spy(replay),
    )
    assert code == 0
    assert run.selected_task_ids == ["clean_parser_001"]
    assert len(seen) == 1
    assert (output / "runs/replay-direct/clean_parser_001/run.json").is_file()


def test_live_preflight_fails_before_output_or_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = BenchmarkMatrixV1.model_validate(
        {
            "schema_version": "1.0",
            "matrix_id": "live",
            "suite_path": "tasks/suites/m4_25.json",
            "variants": [
                {
                    "variant_id": "groq-direct",
                    "provider": "groq",
                    "model": "model-v1",
                    "request_budget": 1,
                    "prompt_version": "review-tools-v1",
                    "agent_version": "tool-agent-v1",
                    "planner_version": "manifest-v1",
                    "tool_transport": "direct",
                }
            ],
            "comparisons": [],
            "threshold_profile": {},
            "max_concurrency": 1,
        }
    )
    matrix_path = tmp_path / "benchmarks/matrices/live.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(json.dumps(matrix.model_dump(mode="json")), encoding="utf-8")
    suite_target = tmp_path / "benchmarks/tasks/suites"
    suite_target.mkdir(parents=True)
    suite_target.joinpath("m4_25.json").write_bytes(SUITE.read_bytes())
    (suite_target.parent / "clean_parser_001.json").write_bytes(
        (ROOT / "benchmarks/tasks/clean_parser_001.json").read_bytes()
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    called = False

    def factory(_variant, _replay):
        nonlocal called
        called = True
        raise AssertionError

    output = tmp_path / "out"
    with pytest.raises(InputError):
        run_benchmark(
            suite_path=SUITE,
            matrix_path=matrix_path,
            output_dir=output,
            task_ids={"clean_parser_001"},
            allow_live=True,
            max_live_requests=1,
            provider_factory=factory,
        )
    assert not called
    assert not output.exists()


def test_provider_failure_leaves_no_final_output(tmp_path: Path) -> None:
    class Failure:
        def review(self, context: ReviewContext):
            del context
            raise RuntimeError("provider failed")

    output = tmp_path / "out"
    with pytest.raises(RuntimeError):
        run_benchmark(
            suite_path=SUITE,
            output_dir=output,
            task_ids={"clean_parser_001"},
            provider_factory=lambda _variant, _replay: Failure(),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".out.*.staging"))


def test_baseline_create_is_non_overwriting(tmp_path: Path) -> None:
    result = tmp_path / "result"
    run_benchmark(
        suite_path=SUITE,
        output_dir=result,
        task_ids={"clean_parser_001"},
    )
    baseline = tmp_path / "baseline.json"
    create_baseline(result / "benchmark.json", baseline)
    with pytest.raises(InputError):
        create_baseline(result / "benchmark.json", baseline)


def test_cli_empty_intersection_is_configuration_exit_two(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--suite",
            str(SUITE),
            "--output-dir",
            str(tmp_path / "out"),
            "--task",
            "clean_parser_001",
            "--positive",
        ],
    )
    assert result.exit_code == 2
    assert not (tmp_path / "out").exists()


def test_tracked_matrix_is_strict_direct_mcp_comparison() -> None:
    matrix = BenchmarkMatrixV1.model_validate_json(MATRIX.read_text(encoding="utf-8"))
    assert [variant.tool_transport for variant in matrix.variants] == ["direct", "mcp-stdio"]
    assert matrix.threshold_profile.trace_equivalence == 1.0
    assert matrix.baseline_path == "baselines/m4_replay_v1.json"


def test_existing_output_preflight_prevents_provider_construction(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    calls = 0

    def factory(_variant, replay):
        nonlocal calls
        calls += 1
        return ReplayProvider(replay)

    with pytest.raises(BenchmarkRunError, match="already exists"):
        run_benchmark(
            suite_path=SUITE,
            output_dir=output,
            task_ids={"clean_parser_001"},
            provider_factory=factory,
        )
    assert calls == 0


@pytest.mark.parametrize("sidecar", [".out.lock", ".out.999.foreign.staging"])
def test_foreign_lock_or_residue_prevents_provider_execution(tmp_path: Path, sidecar: str) -> None:
    path = tmp_path / sidecar
    if sidecar.endswith(".staging"):
        path.mkdir()
    else:
        path.write_text("foreign", encoding="utf-8")
    calls = 0

    def factory(_variant, replay):
        nonlocal calls
        calls += 1
        return ReplayProvider(replay)

    with pytest.raises(BenchmarkRunError):
        run_benchmark(
            suite_path=SUITE,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
            provider_factory=factory,
        )
    assert calls == 0
    assert not (tmp_path / "out").exists()


def test_write_and_cleanup_failures_preserve_primary_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = RuntimeError("write failed")
    real_rmtree = benchmark_runner.shutil.rmtree

    def fail_write(*_args, **_kwargs):
        raise primary

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(benchmark_runner, "_write_tree", fail_write)
    monkeypatch.setattr(benchmark_runner.shutil, "rmtree", fail_cleanup)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError) as caught:
        run_benchmark(
            suite_path=SUITE,
            output_dir=output,
            task_ids={"clean_parser_001"},
        )
    assert caught.value is primary
    assert not output.exists()
    assert not (tmp_path / ".out.lock").exists()
    residues = list(tmp_path.glob(".out.*.staging"))
    snapshots = list(tmp_path.glob(".out.input.*"))
    assert len(residues) == 1
    assert len(snapshots) == 1
    monkeypatch.setattr(benchmark_runner.shutil, "rmtree", real_rmtree)
    real_rmtree(residues[0])
    benchmark_runner._cleanup_snapshot(snapshots[0])


def test_keyboard_interrupt_identity_survives_lock_cleanup(tmp_path: Path) -> None:
    interrupt = KeyboardInterrupt()

    class InterruptingProvider:
        def review(self, context: ReviewContext):
            del context
            raise interrupt

    output = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt) as caught:
        run_benchmark(
            suite_path=SUITE,
            output_dir=output,
            task_ids={"clean_parser_001"},
            provider_factory=lambda _variant, _replay: InterruptingProvider(),
        )
    assert caught.value is interrupt
    assert not output.exists()
    assert not (tmp_path / ".out.lock").exists()


def test_lock_cleanup_failure_after_rename_is_published_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_unlink = Path.unlink
    lock = tmp_path / ".out.lock"

    def fail_lock_unlink(path: Path, *args, **kwargs):
        if path == lock:
            raise OSError("injected lock cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_lock_unlink)
    with pytest.warns(RuntimeWarning, match="output published"):
        run, code = run_benchmark(
            suite_path=SUITE,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
        )
    assert code == 0
    assert run.quality_gate_passed
    assert (tmp_path / "out/benchmark.json").is_file()
    assert lock.exists()
    real_unlink(lock)


def test_result_hash_regression_publishes_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passing = tmp_path / "passing"
    run_benchmark(
        suite_path=SUITE,
        output_dir=passing,
        task_ids={"http_retry_001"},
    )
    baseline_path = tmp_path / "baseline.json"
    create_baseline(passing / "benchmark.json", baseline_path)
    baseline = load_baseline(baseline_path)
    original_default = benchmark_runner.default_matrix

    def matrix_with_baseline(path: Path):
        return original_default(path).model_copy(update={"baseline_path": "README.md"})

    monkeypatch.setattr(benchmark_runner, "default_matrix", matrix_with_baseline)
    monkeypatch.setattr(benchmark_runner, "load_baseline", lambda _path: baseline)
    clean_replay = ROOT / "benchmarks/tasks/replays/clean_parser_001.json"
    output = tmp_path / "regression"
    run, code = run_benchmark(
        suite_path=SUITE,
        output_dir=output,
        task_ids={"http_retry_001"},
        provider_factory=lambda _variant, _replay: ReplayProvider(clean_replay),
    )
    assert code == 1
    assert not run.quality_gate_passed
    assert run.baseline_passed is False
    assert run.hashes["results"] != baseline.hashes["results"]
    assert (output / "benchmark.json").is_file()


def test_contract_hash_drift_is_exit_two_before_provider_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passing = tmp_path / "passing"
    run_benchmark(
        suite_path=SUITE,
        output_dir=passing,
        task_ids={"clean_parser_001"},
    )
    baseline_path = tmp_path / "baseline.json"
    create_baseline(passing / "benchmark.json", baseline_path)
    baseline = load_baseline(baseline_path)
    baseline.hashes["tasks"] = "sha256:" + "0" * 64
    original_default = benchmark_runner.default_matrix

    def matrix_with_baseline(path: Path):
        return original_default(path).model_copy(update={"baseline_path": "README.md"})

    monkeypatch.setattr(benchmark_runner, "default_matrix", matrix_with_baseline)
    monkeypatch.setattr(benchmark_runner, "load_baseline", lambda _path: baseline)
    calls = 0

    def factory(_variant, replay):
        nonlocal calls
        calls += 1
        return ReplayProvider(replay)

    output = tmp_path / "drift"
    with pytest.raises(InputError, match="hashes"):
        run_benchmark(
            suite_path=SUITE,
            output_dir=output,
            task_ids={"clean_parser_001"},
            provider_factory=factory,
        )
    assert calls == 0
    assert not output.exists()
    assert not (tmp_path / ".drift.lock").exists()


def test_lock_is_held_while_provider_executes(tmp_path: Path) -> None:
    lock = tmp_path / ".out.lock"
    observed = False

    class LockCheckingProvider:
        def __init__(self, replay: Path) -> None:
            self.delegate = ReplayProvider(replay)

        def review(self, context: ReviewContext):
            nonlocal observed
            observed = lock.is_file()
            with pytest.raises(FileExistsError):
                descriptor = benchmark_runner.os.open(
                    lock,
                    benchmark_runner.os.O_CREAT
                    | benchmark_runner.os.O_EXCL
                    | benchmark_runner.os.O_WRONLY,
                    0o600,
                )
                benchmark_runner.os.close(descriptor)
            return self.delegate.review(context)

    run_benchmark(
        suite_path=SUITE,
        output_dir=tmp_path / "out",
        task_ids={"clean_parser_001"},
        provider_factory=lambda _variant, replay: LockCheckingProvider(replay),
    )
    assert observed
    assert not lock.exists()


def test_linked_output_ancestor_preflight_prevents_provider(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    calls = 0

    def factory(_variant, replay):
        nonlocal calls
        calls += 1
        return ReplayProvider(replay)

    with pytest.raises(BenchmarkRunError, match="link or reparse"):
        run_benchmark(
            suite_path=SUITE,
            output_dir=linked / "out",
            task_ids={"clean_parser_001"},
            provider_factory=factory,
        )
    assert calls == 0
    assert not (target / "out").exists()


def _copy_case_tree(tmp_path: Path, task_id: str) -> tuple[Path, Path]:
    source_root = ROOT / "benchmarks/tasks"
    task_root = tmp_path / "benchmarks/tasks"
    suite = task_root / "suites/m4_25.json"
    suite.parent.mkdir(parents=True)
    shutil.copy2(SUITE, suite)
    source_manifest = source_root / f"{task_id}.json"
    manifest_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    shutil.copy2(source_manifest, task_root / source_manifest.name)
    for key in ("diff_path", "replay_response_path"):
        source = source_root / manifest_data[key]
        destination = task_root / manifest_data[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    workspace = manifest_data.get("workspace_path")
    if workspace is not None:
        shutil.copytree(source_root / workspace, task_root / workspace)
    return suite, task_root


def test_mismatched_manifest_identity_fails_before_provider_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_load = benchmark_runner.load_task

    def mismatched(path: Path):
        loaded = real_load(path)
        return replace(
            loaded,
            task=loaded.task.model_copy(update={"task_id": "../../outside"}),
        )

    monkeypatch.setattr(benchmark_runner, "load_task", mismatched)
    calls = 0

    def factory(_variant, replay):
        nonlocal calls
        calls += 1
        return ReplayProvider(replay)

    output = tmp_path / "out"
    with pytest.raises(InputError, match="identity"):
        run_benchmark(
            suite_path=SUITE,
            output_dir=output,
            task_ids={"clean_parser_001"},
            provider_factory=factory,
        )
    assert calls == 0
    assert not output.exists()
    assert not (tmp_path / "outside").exists()


def test_suite_task_symlink_is_rejected_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, task_root = _copy_case_tree(tmp_path, "clean_parser_001")
    manifest = task_root / "clean_parser_001.json"
    manifest.unlink()
    private = tmp_path / "private.json"
    private.write_text('{"secret": true}', encoding="utf-8")
    try:
        manifest.symlink_to(private)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    real_read_text = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path == private:
            raise AssertionError("private symlink target was read")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(InputError, match="links or reparse"):
        run_benchmark(
            suite_path=suite,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("linked_directory", [False, True])
def test_workspace_link_is_rejected_before_provider(tmp_path: Path, linked_directory: bool) -> None:
    suite, task_root = _copy_case_tree(tmp_path, "clean_parser_001")
    workspace = task_root / "workspaces/clean_parser_001"
    external = tmp_path / ("external-dir" if linked_directory else "external.py")
    if linked_directory:
        external.mkdir()
        link = workspace / "linked"
    else:
        external.write_text("PRIVATE = True", encoding="utf-8")
        link = workspace / "solution.py"
        link.unlink()
    try:
        link.symlink_to(external, target_is_directory=linked_directory)
    except OSError:
        pytest.skip("workspace symlink creation is unavailable")
    calls = 0

    def factory(_variant, replay):
        nonlocal calls
        calls += 1
        return ReplayProvider(replay)

    with pytest.raises(InputError, match="links or reparse"):
        run_benchmark(
            suite_path=suite,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
            provider_factory=factory,
        )
    assert calls == 0
    assert not (tmp_path / "out").exists()


def test_external_manifest_reference_is_rejected_before_private_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, task_root = _copy_case_tree(tmp_path, "clean_parser_001")
    manifest = task_root / "clean_parser_001.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["diff_path"] = "../../private.diff"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    private = tmp_path / "private.diff"
    private.write_text("PRIVATE", encoding="utf-8")
    real_read_text = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path == private:
            raise AssertionError("private reference was read")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(InputError):
        run_benchmark(
            suite_path=suite,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_threshold_rates_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        ThresholdProfileV1(completion_rate=value)


def test_threshold_counts_reject_bool() -> None:
    with pytest.raises(ValidationError):
        ThresholdProfileV1(negative_false_positives=False)


def test_metric_models_reject_inconsistent_counts_and_rates() -> None:
    baseline = load_baseline(ROOT / "benchmarks/baselines/m4_replay_v1.json")
    task = baseline.stable_result.variants[0].tasks[0].model_dump(mode="json")
    task["missed_count"] += 1
    with pytest.raises(ValidationError):
        StableTaskMetricsV1.model_validate(task)
    aggregate = baseline.stable_result.variants[0].metrics.model_dump(mode="json")
    aggregate["completion_rate"] = 0.5
    with pytest.raises(ValidationError):
        StableAggregateMetricsV1.model_validate(aggregate)


def test_malformed_stable_baseline_is_safe_input_error(tmp_path: Path) -> None:
    data = json.loads((ROOT / "benchmarks/baselines/m4_replay_v1.json").read_text(encoding="utf-8"))
    del data["stable_result"]["variants"][0]["tasks"][0]["precision"]
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match="invalid"):
        load_baseline(path)


def test_comparison_rejects_same_transport() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    data["variants"][1]["tool_transport"] = "direct"
    with pytest.raises(ValidationError):
        BenchmarkMatrixV1.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_version", "other-prompt"),
        ("agent_version", "other-agent"),
        ("planner_version", "other-planner"),
        ("request_budget", 2),
    ],
)
def test_comparison_rejects_contract_or_budget_mismatch(field: str, value: object) -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    if field == "request_budget":
        for variant in data["variants"]:
            variant.update(provider="groq", model="model-v1", request_budget=1)
    data["variants"][1][field] = value
    with pytest.raises(ValidationError):
        BenchmarkMatrixV1.model_validate(data)


def test_comparison_rejects_cross_provider() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    data["variants"][1].update(provider="groq", model="model-v1", request_budget=1)
    with pytest.raises(ValidationError):
        BenchmarkMatrixV1.model_validate(data)


def _passing_result(tmp_path: Path) -> Path:
    output = tmp_path / "passing"
    run_benchmark(
        suite_path=SUITE,
        output_dir=output,
        task_ids={"clean_parser_001"},
    )
    return output / "benchmark.json"


def test_baseline_race_never_overwrites_created_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _passing_result(tmp_path)
    output = tmp_path / "baseline.json"
    real_link = benchmark_baseline.os.link

    def racing_link(source, destination, **kwargs):
        Path(destination).write_text("FOREIGN", encoding="utf-8")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(benchmark_baseline.os, "link", racing_link)
    with pytest.raises(InputError, match="appeared"):
        create_baseline(result, output)
    assert output.read_text(encoding="utf-8") == "FOREIGN"
    assert not (tmp_path / ".baseline.json.lock").exists()


def test_baseline_interrupt_identity_and_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _passing_result(tmp_path)
    interrupt = KeyboardInterrupt()

    def interrupted_dump(*_args, **_kwargs):
        raise interrupt

    monkeypatch.setattr(benchmark_baseline.json, "dump", interrupted_dump)
    output = tmp_path / "baseline.json"
    with pytest.raises(KeyboardInterrupt) as caught:
        create_baseline(result, output)
    assert caught.value is interrupt
    assert not output.exists()
    assert not (tmp_path / ".baseline.json.lock").exists()
    assert not list(tmp_path.glob(".baseline.json.*.tmp"))


def test_baseline_primary_failure_survives_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _passing_result(tmp_path)
    primary = RuntimeError("baseline write failed")
    real_cleanup = benchmark_baseline._cleanup_temporary

    def failed_dump(*_args, **_kwargs):
        raise primary

    def failed_cleanup(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(benchmark_baseline.json, "dump", failed_dump)
    monkeypatch.setattr(benchmark_baseline, "_cleanup_temporary", failed_cleanup)
    with pytest.raises(RuntimeError) as caught:
        create_baseline(result, tmp_path / "baseline.json")
    assert caught.value is primary
    assert not (tmp_path / "baseline.json").exists()
    assert not (tmp_path / ".baseline.json.lock").exists()
    residues = list(tmp_path.glob(".baseline.json.*.tmp"))
    assert len(residues) == 1
    real_cleanup(residues[0])


def test_baseline_linked_ancestor_is_rejected(
    tmp_path: Path,
) -> None:
    result = _passing_result(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(InputError, match="links or reparse"):
        create_baseline(result, linked / "baseline.json")


def test_baseline_residue_is_rejected(tmp_path: Path) -> None:
    result = _passing_result(tmp_path)
    residue = tmp_path / ".baseline.json.123.tmp"
    residue.write_text("residue", encoding="utf-8")
    with pytest.raises(InputError, match="residue"):
        create_baseline(result, tmp_path / "baseline.json")


def test_baseline_reservation_is_held_through_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _passing_result(tmp_path)
    output = tmp_path / "baseline.json"
    lock = tmp_path / ".baseline.json.lock"
    real_link = benchmark_baseline.os.link
    observed = False

    def checking_link(source, destination, **kwargs):
        nonlocal observed
        observed = lock.is_file()
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(benchmark_baseline.os, "link", checking_link)
    create_baseline(result, output)
    assert observed
    assert output.is_file()
    assert not lock.exists()


def test_secure_read_swap_rejects_before_external_content_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "input.json"
    source.write_text("TRUSTED", encoding="utf-8")
    backup = root / "backup.json"
    private = tmp_path / "private.json"
    private.write_text("PRIVATE", encoding="utf-8")
    real_open = benchmark_runner.os.open
    real_read = benchmark_runner.os.read
    swapped = False
    private_descriptor: int | None = None
    private_read = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped, private_descriptor
        if Path(path) == source and not swapped:
            swapped = True
            benchmark_runner.os.replace(source, backup)
            benchmark_runner.os.link(private, source)
            private_descriptor = real_open(path, flags, *args, **kwargs)
            return private_descriptor
        return real_open(path, flags, *args, **kwargs)

    def guarded_read(descriptor, size):
        nonlocal private_read
        if descriptor == private_descriptor:
            private_read = True
        return real_read(descriptor, size)

    monkeypatch.setattr(benchmark_runner.os, "open", racing_open)
    monkeypatch.setattr(benchmark_runner.os, "read", guarded_read)
    with pytest.raises(InputError, match="authority changed"):
        benchmark_runner._secure_read_bytes(source, root)
    assert not private_read


def test_execution_uses_snapshot_after_source_inputs_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, task_root = _copy_case_tree(tmp_path, "clean_parser_001")
    real_snapshot = benchmark_runner._snapshot_selected_tasks

    def snapshot_then_mutate(root, selected, snapshot_root):
        snapshots = real_snapshot(root, selected, snapshot_root)
        manifest = json.loads((task_root / "clean_parser_001.json").read_text(encoding="utf-8"))
        (task_root / manifest["diff_path"]).write_text("PRIVATE DIFF", encoding="utf-8")
        (task_root / manifest["replay_response_path"]).write_text(
            "PRIVATE REPLAY", encoding="utf-8"
        )
        workspace = task_root / manifest["workspace_path"]
        (workspace / "solution.py").write_text("PRIVATE = True", encoding="utf-8")
        return snapshots

    monkeypatch.setattr(benchmark_runner, "_snapshot_selected_tasks", snapshot_then_mutate)
    run, code = run_benchmark(
        suite_path=suite,
        output_dir=tmp_path / "out",
        task_ids={"clean_parser_001"},
    )
    assert code == 0
    assert run.quality_gate_passed


def test_suite_capture_rejects_external_swap_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, _task_root = _copy_case_tree(tmp_path, "clean_parser_001")
    backup = suite.with_name("suite-backup.json")
    external = tmp_path / "external-suite.json"
    external.write_text('{"private": true}', encoding="utf-8")
    real_open = benchmark_runner.os.open
    real_read = benchmark_runner.os.read
    external_descriptor: int | None = None
    external_read = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal external_descriptor
        if Path(path) == suite and external_descriptor is None:
            benchmark_runner.os.replace(suite, backup)
            benchmark_runner.os.link(external, suite)
            external_descriptor = real_open(path, flags, *args, **kwargs)
            return external_descriptor
        return real_open(path, flags, *args, **kwargs)

    def guard_external_read(descriptor, size):
        nonlocal external_read
        if descriptor == external_descriptor:
            external_read = True
        return real_read(descriptor, size)

    monkeypatch.setattr(benchmark_runner.os, "open", swap_before_open)
    monkeypatch.setattr(benchmark_runner.os, "read", guard_external_read)
    with pytest.raises(InputError, match="authority changed"):
        run_benchmark(
            suite_path=suite,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
        )
    assert not external_read
    assert not (tmp_path / "out").exists()


def test_nested_duplicate_manifest_basenames_do_not_collapse_task_hash() -> None:
    suite = load_suite(SUITE)
    matrix = benchmark_runner.default_matrix(SUITE)
    loaded = benchmark_runner.load_task(ROOT / "benchmarks/tasks/clean_parser_001.json")
    entry = suite.tasks[-5]
    first = benchmark_runner.SelectedSnapshot(
        entry=entry,
        loaded=loaded,
        manifest_relative=Path("one/task.json"),
        content_hash="sha256:" + "1" * 64,
        diff_text="",
        replay_provider=ReplayProvider(loaded.replay_path),
    )
    second = benchmark_runner.SelectedSnapshot(
        entry=entry,
        loaded=loaded,
        manifest_relative=Path("two/task.json"),
        content_hash="sha256:" + "2" * 64,
        diff_text="",
        replay_provider=ReplayProvider(loaded.replay_path),
    )
    changed = replace(second, content_hash="sha256:" + "3" * 64)
    suite_content = SUITE.read_bytes()
    left = benchmark_runner._hashes(matrix, suite, suite_content, [first, second])
    right = benchmark_runner._hashes(matrix, suite, suite_content, [first, changed])
    assert left["tasks"] != right["tasks"]


@pytest.mark.parametrize("mutation", ["duplicate_variant", "missing_ref", "duplicate_comparison"])
def test_stable_baseline_rejects_ambiguous_variant_or_comparison_graph(
    tmp_path: Path, mutation: str
) -> None:
    data = json.loads((ROOT / "benchmarks/baselines/m4_replay_v1.json").read_text(encoding="utf-8"))
    stable = data["stable_result"]
    if mutation == "duplicate_variant":
        stable["variants"].append(stable["variants"][0])
    elif mutation == "missing_ref":
        stable["comparisons"][0]["candidate_variant"] = "missing"
    else:
        stable["comparisons"].append(stable["comparisons"][0])
    path = tmp_path / "bad-baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match="invalid"):
        load_baseline(path)


def test_runner_reconciles_interrupt_after_successful_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = benchmark_runner.os.replace

    def replace_then_interrupt(source, destination):
        real_replace(source, destination)
        raise KeyboardInterrupt()

    monkeypatch.setattr(benchmark_runner.os, "replace", replace_then_interrupt)
    with pytest.warns(RuntimeWarning, match="published despite"):
        run, code = run_benchmark(
            suite_path=SUITE,
            output_dir=tmp_path / "out",
            task_ids={"clean_parser_001"},
        )
    assert code == 0
    assert run.quality_gate_passed
    assert (tmp_path / "out/benchmark.json").is_file()


def test_baseline_reconciles_interrupt_after_successful_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _passing_result(tmp_path)
    real_link = benchmark_baseline.os.link

    def link_then_interrupt(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(benchmark_baseline.os, "link", link_then_interrupt)
    output = tmp_path / "baseline.json"
    with pytest.warns(RuntimeWarning, match="published despite"):
        baseline = create_baseline(result, output)
    assert baseline.matrix_id
    assert output.is_file()


def _tool_metric_fixture(tmp_path: Path):
    output = tmp_path / "metric-run"
    run_benchmark(
        suite_path=SUITE,
        output_dir=output,
        task_ids={"clean_parser_001"},
    )
    task = benchmark_runner.load_task(ROOT / "benchmarks/tasks/clean_parser_001.json").task
    artifact = RunArtifact.model_validate_json(
        (output / "runs/replay-direct/clean_parser_001/run.json").read_text(encoding="utf-8")
    )
    return task, artifact


def test_tool_metrics_cover_zero_calls(tmp_path: Path) -> None:
    task, artifact = _tool_metric_fixture(tmp_path)
    task = task.model_copy(update={"tool_plan": None})
    artifact = artifact.model_copy(update={"tool_trace": []})
    metric = task_metrics(task, artifact)
    assert (
        metric.tool_planned,
        metric.tool_observed,
        metric.tool_succeeded,
        metric.tool_failed,
        metric.tool_missing,
        metric.tool_unexpected,
        metric.tool_mismatched,
        metric.tool_plan_correct,
        metric.tool_plan_total,
    ) == (0, 0, 0, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("duplicate", (2, 2, 2, 0, 0, 0, 1, 1)),
        ("order", (2, 2, 2, 0, 0, 0, 2, 0)),
        ("failure", (2, 2, 1, 1, 0, 0, 0, 1)),
        ("unexpected", (2, 3, 3, 0, 0, 1, 0, 2)),
        ("mismatch", (2, 2, 2, 0, 0, 0, 1, 1)),
        ("missing", (2, 1, 1, 0, 1, 0, 0, 1)),
    ],
)
def test_tool_metrics_distinguish_plan_position_and_outcome(
    tmp_path: Path,
    mutation: str,
    expected: tuple[int, int, int, int, int, int, int, int],
) -> None:
    task, artifact = _tool_metric_fixture(tmp_path)
    trace = list(artifact.tool_trace)
    if mutation == "duplicate":
        trace[1] = trace[0].model_copy(update={"trace_id": "tool-002", "order": 2})
    elif mutation == "order":
        trace = [
            trace[1].model_copy(update={"trace_id": "tool-001", "order": 1}),
            trace[0].model_copy(update={"trace_id": "tool-002", "order": 2}),
        ]
    elif mutation == "failure":
        trace[0] = trace[0].model_copy(
            update={
                "status": ToolStatus.FAILED,
                "result": ToolFailureResult(
                    kind="tool_failure",
                    code="read_failed",
                    message="read failed",
                ),
                "provenance": [],
                "influence": trace[0].influence.model_copy(
                    update={"finding_indices": [], "report_sections": []}
                ),
            }
        )
    elif mutation == "unexpected":
        trace.append(trace[0].model_copy(update={"trace_id": "tool-003", "order": 3}))
    elif mutation == "mismatch":
        trace[0] = trace[0].model_copy(
            update={"arguments": trace[0].arguments.model_copy(update={"path": "other.py"})}
        )
    else:
        trace = trace[:1]
    metric = task_metrics(task, artifact.model_copy(update={"tool_trace": trace}))
    assert (
        metric.tool_planned,
        metric.tool_observed,
        metric.tool_succeeded,
        metric.tool_failed,
        metric.tool_missing,
        metric.tool_unexpected,
        metric.tool_mismatched,
        metric.tool_plan_correct,
    ) == expected
    aggregate = aggregate_metrics([(task, metric)])
    assert aggregate.tool_failed == metric.tool_failed
    assert aggregate.tool_missing == metric.tool_missing
    assert aggregate.tool_unexpected == metric.tool_unexpected
    assert aggregate.tool_mismatched == metric.tool_mismatched
    if mutation in {"failure", "unexpected", "mismatch", "missing"}:
        assert gate_failures(aggregate, ThresholdProfileV1())


def test_aggregate_includes_rejection_and_test_status_counts(tmp_path: Path) -> None:
    task, artifact = _tool_metric_fixture(tmp_path)
    metric = task_metrics(task, artifact)
    metric = metric.model_copy(
        update={"rejection_codes": ["invalid_path"], "test_statuses": ["passed"]}
    )
    aggregate = aggregate_metrics([(task, metric)])
    assert aggregate.rejection_codes == {"invalid_path": 1}
    assert aggregate.test_statuses == {"passed": 1}


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_hash_contract_rejects_missing_or_extra_keys(mutation: str) -> None:
    data = json.loads((ROOT / "benchmarks/baselines/m4_replay_v1.json").read_text(encoding="utf-8"))
    if mutation == "missing":
        data["hashes"].pop("tasks")
    else:
        data["hashes"]["other"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="required keys"):
        BenchmarkBaselineV1.model_validate(data)


@pytest.mark.parametrize("mutation", ["metric", "task", "comparison", "result_hash"])
def test_baseline_load_rejects_stable_content_or_result_hash_tampering(
    tmp_path: Path, mutation: str
) -> None:
    data = json.loads((ROOT / "benchmarks/baselines/m4_replay_v1.json").read_text(encoding="utf-8"))
    stable = data["stable_result"]
    if mutation == "metric":
        stable["variants"][0]["metrics"]["difficulty_success"]["low"] = 0.5
    elif mutation == "task":
        stable["variants"][0]["tasks"][0]["rejection_codes"].append("tampered")
    elif mutation == "comparison":
        stable["comparisons"][0]["metric_deltas"]["micro_precision"] = 0.25
    else:
        replacement = "sha256:" + "0" * 64
        stable["hashes"]["results"] = replacement
        data["hashes"]["results"] = replacement
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match=r"hash|invalid"):
        load_baseline(path)


def test_compare_baseline_reports_any_valid_stable_content_change(tmp_path: Path) -> None:
    result_path = _passing_result(tmp_path)
    run = BenchmarkRunV1.model_validate_json(result_path.read_text(encoding="utf-8"))
    baseline_path = tmp_path / "baseline.json"
    create_baseline(result_path, baseline_path)
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    data["stable_result"]["variants"][0]["tasks"][0]["semantic_trace_fingerprint"] = "changed"
    stable = StableBenchmarkRunV1.model_validate(data["stable_result"])
    replacement = stable_result_hash(stable)
    data["stable_result"]["hashes"]["results"] = replacement
    data["hashes"]["results"] = replacement
    changed = BenchmarkBaselineV1.model_validate(data)
    assert compare_baseline(run, changed) == ["stable result differs from baseline"]


def test_exact_byte_hash_distinguishes_line_endings() -> None:
    assert benchmark_runner._bytes_hash(b"line\n") != benchmark_runner._bytes_hash(b"line\r\n")
