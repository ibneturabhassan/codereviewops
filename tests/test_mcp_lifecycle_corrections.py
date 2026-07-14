from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import codereviewops.mcp_owned_backend as owned_backend
import codereviewops.workflow as workflow
from codereviewops.io import load_task
from codereviewops.mcp_owned_backend import BackendState, McpStdioToolBackend
from codereviewops.models import ToolPlan
from codereviewops.process_control import ProcessTreeController
from codereviewops.tool_contracts import ToolError

ROOT = Path(__file__).parents[1]
BENCHMARK_ROOT = ROOT / "tests" / "fixtures" / "legacy_benchmarks"
WORKSPACE_ROOT = BENCHMARK_ROOT / "workspaces" / "python_tools_001"


class FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.waits = 0
        self.terminated = False
        self._handle = 99

    def wait(self, timeout: int) -> int:
        del timeout
        self.waits += 1
        return 0

    def terminate(self) -> None:
        self.terminated = True


class FakeJobApi:
    def __init__(self, *, assign: bool = True, active: int | None = 0) -> None:
        self.assign_result = assign
        self.active = active
        self.calls: list[object] = []

    def create_job(self) -> int:
        self.calls.append("create_job")
        return 10

    def create_barrier(self) -> int:
        self.calls.append("create_barrier")
        return 20

    def startup_info(self, barrier: int) -> object:
        self.calls.append(("startup_info", barrier))
        return SimpleNamespace(lpAttributeList={"handle_list": [barrier]})

    def assign(self, job: int, process: FakeProcess) -> bool:
        self.calls.append(("assign", job, process.pid))
        return self.assign_result

    def release(self, event: int) -> bool:
        self.calls.append(("release", event))
        return True

    def terminate_job(self, job: int) -> bool:
        self.calls.append(("terminate_job", job))
        return True

    def active_processes(self, job: int) -> int | None:
        self.calls.append(("active_processes", job))
        return self.active

    def close_handle(self, handle: int) -> None:
        self.calls.append(("close_handle", handle))


def test_windows_lease_assigns_before_release_retains_job_and_is_idempotent() -> None:
    api = FakeJobApi()
    lease = ProcessTreeController(platform="windows", windows_api=api).prepare()
    kwargs = lease.popen_kwargs()
    assert kwargs["close_fds"] is True
    assert kwargs["startupinfo"].lpAttributeList == {"handle_list": [20]}
    assert lease.command(Path("python.exe"), "ignored", "repo")[2:] == [
        "codereviewops.mcp_bootstrap",
        "repo",
        "20",
    ]
    process = FakeProcess()

    assert lease.attach(process)  # type: ignore[arg-type]
    assert api.calls.index(("assign", 10, 4321)) < api.calls.index(("release", 20))
    assert ("close_handle", 10) not in api.calls
    assert lease.terminate_and_verify()
    assert lease.terminate_and_verify()
    assert api.calls.count(("terminate_job", 10)) == 1
    assert api.calls.count(("active_processes", 10)) == 1
    assert ("close_handle", 10) not in api.calls
    lease.close()
    lease.close()
    assert api.calls.count(("close_handle", 10)) == 1


def test_windows_assignment_failure_never_releases_blocked_bootstrap() -> None:
    api = FakeJobApi(assign=False)
    lease = ProcessTreeController(platform="windows", windows_api=api).prepare()
    process = FakeProcess()

    assert not lease.attach(process)  # type: ignore[arg-type]
    assert process.terminated and process.waits == 1
    assert not any(isinstance(call, tuple) and call[0] == "release" for call in api.calls)
    lease.close()
    assert ("close_handle", 20) in api.calls
    assert ("close_handle", 10) in api.calls


def test_windows_active_process_query_failure_fails_cleanup() -> None:
    api = FakeJobApi(active=None)
    lease = ProcessTreeController(platform="windows", windows_api=api).prepare()
    assert lease.attach(FakeProcess())  # type: ignore[arg-type]
    assert not lease.terminate_and_verify()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group integration")
def test_posix_lease_cleans_descendant_after_graceful_leader_exit() -> None:
    lease = ProcessTreeController(platform="posix").prepare()
    script = (
        f"import subprocess; p=subprocess.Popen([{sys.executable!r}, "
        "'-c', 'import time; time.sleep(30)']); "
        "print(p.pid, flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert lease.attach(process)
    assert process.stdout is not None
    child_pid = int(process.stdout.readline())
    process.wait(timeout=3)
    assert lease.terminate_and_verify()
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    lease.close()


def test_portal_constructor_failure_is_safe_terminal_and_idempotent(monkeypatch) -> None:
    def fail_portal(**kwargs):
        del kwargs
        raise RuntimeError("unsafe detail")

    monkeypatch.setattr(owned_backend, "start_blocking_portal", fail_portal)
    backend = McpStdioToolBackend(
        WORKSPACE_ROOT, BENCHMARK_ROOT, ToolPlan(read_files=["calculator.py"])
    )
    with pytest.raises(ToolError, match="MCP client runtime failed"):
        backend.open()

    assert backend.state is BackendState.FAILED
    assert backend.records[0]["lifecycle"] == ["planned", "failed"]
    assert backend.records[0]["failure_stage"] == "client_runtime"
    assert backend.close() is None
    assert backend.close() is None


def test_portal_enter_failure_never_calls_exit(monkeypatch) -> None:
    class Context:
        exited = False

        def __enter__(self):
            raise RuntimeError("unsafe detail")

        def __exit__(self, *args):
            self.exited = True

    context = Context()
    monkeypatch.setattr(owned_backend, "start_blocking_portal", lambda **kwargs: context)
    backend = McpStdioToolBackend(
        WORKSPACE_ROOT, BENCHMARK_ROOT, ToolPlan(read_files=["calculator.py"])
    )
    with pytest.raises(ToolError, match="MCP client runtime failed"):
        backend.open()
    assert not context.exited
    assert backend.state is BackendState.FAILED


def test_portal_exit_failure_returns_stable_cleanup_error(monkeypatch) -> None:
    class Context:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise RuntimeError("unsafe detail")

    monkeypatch.setattr(owned_backend, "start_blocking_portal", lambda **kwargs: Context())
    backend = McpStdioToolBackend(WORKSPACE_ROOT, BENCHMARK_ROOT, ToolPlan())
    backend.open()
    assert backend.state is BackendState.OPEN

    failure = backend.close()
    assert failure is not None and failure.code == "mcp_lifecycle"
    assert backend.state is BackendState.FAILED
    assert backend.close() is failure


def test_portal_failure_workflow_is_schema_13_and_never_calls_provider(monkeypatch) -> None:
    loaded = load_task(BENCHMARK_ROOT / "python_tools_001.json")

    class NeverProvider:
        def review(self, context):
            del context
            raise AssertionError("provider must not be called")

    def fail_portal(**kwargs):
        del kwargs
        raise RuntimeError("unsafe detail")

    monkeypatch.setattr(owned_backend, "start_blocking_portal", fail_portal)
    _, artifact = workflow.run_loaded_task(loaded, NeverProvider(), tool_transport="mcp-stdio")
    assert artifact.schema_version == "1.3"
    assert artifact.provider_status == "not_called"
    assert artifact.failure_code == "mcp_lifecycle"
    assert artifact.mcp_servers[0].failure_stage == "client_runtime"
