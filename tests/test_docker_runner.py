from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

import codereviewops.docker_runner as docker_runner
from codereviewops.docker_runner import (
    OUTPUT_LIMIT,
    RUNNER_TAG,
    TEST_ARGV,
    DockerTestRunner,
    ProcessTreeController,
)
from codereviewops.models import TestStatus as ReviewTestStatus
from codereviewops.tools import ToolError

IMAGE_ID = "sha256:" + ("a" * 64)


def _completed(returncode: int = 0, stdout: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["docker"],
        returncode=returncode,
        stdout=stdout if stdout is not None else (IMAGE_ID + "\n").encode(),
        stderr=b"",
    )


def test_command_has_exact_hardening_and_immutable_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = DockerTestRunner(docker="docker")
    command = runner._command(workspace, IMAGE_ID, "codereviewops-" + ("b" * 32))
    assert command[:2] == ["docker", "run"]
    for pair in [
        ["--network", "none"],
        ["--pull", "never"],
        ["--cap-drop", "ALL"],
        ["--security-opt", "no-new-privileges"],
        ["--pids-limit", "64"],
        ["--memory", "256m"],
        ["--cpus", "1"],
        ["--user", "65534:65534"],
        ["--workdir", "/workspace"],
    ]:
        index = command.index(pair[0])
        assert command[index : index + 2] == pair
    assert "--rm" in command
    assert "--init" in command
    assert "--read-only" in command
    mount = command[command.index("--mount") + 1]
    assert mount.endswith(",target=/workspace,readonly")
    assert command[-(len(TEST_ARGV) + 1)] == IMAGE_ID
    assert tuple(command[-len(TEST_ARGV) :]) == TEST_ARGV
    assert RUNNER_TAG not in command
    assert "/var/run/docker.sock" not in " ".join(command)


def test_mount_and_id_validation_rejects_unsafe_values(tmp_path: Path) -> None:
    runner = DockerTestRunner(docker="docker")
    comma_workspace = tmp_path / "unsafe,workspace"
    comma_workspace.mkdir()
    with pytest.raises(ToolError, match="mount"):
        runner._command(comma_workspace, IMAGE_ID, "codereviewops-" + ("b" * 32))
    safe = tmp_path / "safe"
    safe.mkdir()
    with pytest.raises(ToolError, match="image"):
        runner._command(safe, RUNNER_TAG, "codereviewops-" + ("b" * 32))
    with pytest.raises(ToolError, match="container"):
        runner._command(safe, IMAGE_ID, "unsafe")


def test_image_inspection_requires_local_immutable_id(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "info":
            return _completed(stdout=b"linux\n")
        if command[1] == "run":
            return _completed(stdout=b"probe-ok\n")
        return _completed()

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
    runner = DockerTestRunner(docker="docker")
    assert runner.check() == IMAGE_ID
    assert calls[0] == ["docker", "info", "--format", "{{.OSType}}"]
    assert calls[1] == ["docker", "image", "inspect", "--format", "{{.Id}}", RUNNER_TAG]
    assert calls[2][0:2] == ["docker", "run"]

    monkeypatch.setattr(
        docker_runner.subprocess,
        "run",
        lambda command, **kwargs: (
            _completed(stdout=b"linux\n")
            if command[1] == "info"
            else _completed(stdout=b"codereviewops/python-unittest:latest\n")
        ),
    )
    with pytest.raises(ToolError, match="unavailable"):
        runner.check()


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, return_code: int = 0) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.return_code = return_code

    def wait(self, timeout: int) -> int:
        return self.return_code


def test_run_drains_bounded_output_and_never_uses_host_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(docker_runner.subprocess, "run", lambda command, **kwargs: _completed())

    def fake_popen(command, **kwargs):
        popen_calls.append(command)
        return FakeProcess(b"x" * (OUTPUT_LIMIT + 100), b"stderr")

    monkeypatch.setattr(docker_runner.subprocess, "Popen", fake_popen)
    execution = DockerTestRunner(docker="docker").run(workspace, "python-unittest-v1")
    assert execution.status == ReviewTestStatus.PASSED
    assert "[output truncated]" in execution.summary
    assert len(popen_calls) == 1
    assert popen_calls[0][0:2] == ["docker", "run"]
    with pytest.raises(ToolError, match="unsupported"):
        DockerTestRunner(docker="docker").run(workspace, "arbitrary-command")
    assert len(popen_calls) == 1


class TimeoutProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(b"", b"")
        self.waits = 0

    def wait(self, timeout: int) -> int:
        self.waits += 1
        if self.waits == 1:
            raise subprocess.TimeoutExpired("docker", timeout)
        return 137


def test_timeout_cleans_only_generated_container(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        run_calls.append(command)
        return _completed()

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        docker_runner.subprocess, "Popen", lambda command, **kwargs: TimeoutProcess()
    )
    execution = DockerTestRunner(docker="docker").run(workspace, "python-unittest-v1")
    assert execution.status == ReviewTestStatus.ERROR
    cleanup = [command for command in run_calls if command[1:3] == ["rm", "-f"]]
    assert len(cleanup) == 1
    assert cleanup[0][3].startswith("codereviewops-")


class OrphanProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(b"", b"")
        self.terminated = False
        self.killed = False

    def wait(self, timeout: int) -> int:
        raise subprocess.TimeoutExpired("docker", timeout)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class FailingTreeController:
    def popen_kwargs(self) -> dict[str, object]:
        return {}

    def terminate(self, process: OrphanProcess) -> bool:
        process.terminate()
        process.kill()
        return False


def test_second_wait_timeout_kills_client_and_reports_cleanup_failure(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    process = OrphanProcess()
    monkeypatch.setattr(docker_runner.subprocess, "run", lambda command, **kwargs: _completed())
    monkeypatch.setattr(docker_runner.subprocess, "Popen", lambda command, **kwargs: process)
    execution = DockerTestRunner(docker="docker", process_controller=FailingTreeController()).run(
        workspace, "python-unittest-v1"
    )
    assert process.terminated and process.killed
    assert execution.infrastructure_error
    assert execution.error_code == "cleanup_failed"


def test_cleanup_failure_is_infrastructure_error(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run(command, **kwargs):
        if command[1:3] == ["rm", "-f"]:
            return _completed(returncode=1)
        return _completed()

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        docker_runner.subprocess, "Popen", lambda command, **kwargs: TimeoutProcess()
    )
    execution = DockerTestRunner(docker="docker").run(workspace, "python-unittest-v1")
    assert execution.infrastructure_error
    assert execution.error_code == "cleanup_failed"


@pytest.mark.parametrize(
    ("return_code", "status", "infrastructure"),
    [
        (1, ReviewTestStatus.FAILED, False),
        (125, ReviewTestStatus.ERROR, True),
    ],
)
def test_docker_exit_classification(
    tmp_path: Path,
    monkeypatch,
    return_code: int,
    status: ReviewTestStatus,
    infrastructure: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(docker_runner.subprocess, "run", lambda command, **kwargs: _completed())
    monkeypatch.setattr(
        docker_runner.subprocess,
        "Popen",
        lambda command, **kwargs: FakeProcess(b"", b"", return_code),
    )
    execution = DockerTestRunner(docker="docker").run(workspace, "python-unittest-v1")
    assert execution.status == status
    assert execution.infrastructure_error is infrastructure


def test_process_controller_uses_platform_group_creation_flags() -> None:
    assert ProcessTreeController(platform="posix").popen_kwargs() == {"start_new_session": True}
    assert ProcessTreeController(platform="windows").popen_kwargs() == {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
    }


def test_posix_process_controller_terminates_and_reaps_group(monkeypatch) -> None:
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 123
        waits = 0

        def wait(self, timeout: int) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("docker", timeout)
            return 137

    def fake_killpg(pid: int, sent_signal: int) -> None:
        signals.append((pid, sent_signal))
        if sent_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(docker_runner.os, "killpg", fake_killpg, raising=False)
    assert ProcessTreeController(platform="posix").terminate(Process())
    assert signals == [
        (123, docker_runner.signal.SIGTERM),
        (123, getattr(docker_runner.signal, "SIGKILL", 9)),
        (123, 0),
    ]


def test_posix_parent_exit_does_not_hide_surviving_child(monkeypatch) -> None:
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 321

        def __init__(self) -> None:
            self.waits = 0

        def wait(self, timeout: int) -> int:
            del timeout
            self.waits += 1
            return 0

    def fake_killpg(pid: int, sent_signal: int) -> None:
        signals.append((pid, sent_signal))
        if sent_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(docker_runner.os, "killpg", fake_killpg, raising=False)
    process = Process()
    assert ProcessTreeController(platform="posix").terminate(process)
    assert process.waits == 2
    assert signals == [
        (321, docker_runner.signal.SIGTERM),
        (321, getattr(docker_runner.signal, "SIGKILL", 9)),
        (321, 0),
    ]


def test_windows_process_controller_uses_validated_absolute_taskkill(
    tmp_path: Path, monkeypatch
) -> None:
    system_root = tmp_path / "Windows"
    taskkill = system_root / "System32" / "taskkill.exe"
    taskkill.parent.mkdir(parents=True)
    taskkill.write_bytes(b"stub")
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Process:
        pid = 456
        waits = 0

        def __init__(self) -> None:
            self.pid = 456
            self.waits = 0
            self.signals: list[int] = []

        def send_signal(self, sent_signal: int) -> None:
            self.signals.append(sent_signal)

        def wait(self, timeout: int) -> int:
            self.waits += 1
            del timeout
            return 0

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(stdout=b"")

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
    process = Process()
    controller = ProcessTreeController(platform="windows", system_root=system_root)
    assert controller.terminate(process)
    assert process.signals == []
    assert calls[0][0] == [
        str(taskkill.resolve()),
        "/PID",
        "456",
        "/T",
        "/F",
    ]
    assert calls[0][1]["shell"] is False


@pytest.mark.parametrize(("taskkill_code", "expected"), [(0, True), (1, False)])
def test_windows_parent_exit_still_requires_verified_tree_cleanup(
    tmp_path: Path, monkeypatch, taskkill_code: int, expected: bool
) -> None:
    system_root = tmp_path / "Windows"
    taskkill = system_root / "System32" / "taskkill.exe"
    taskkill.parent.mkdir(parents=True)
    taskkill.write_bytes(b"stub")
    calls: list[list[str]] = []

    class Process:
        pid = 789

        def __init__(self) -> None:
            self.waits = 0
            self.signals: list[int] = []

        def send_signal(self, sent_signal: int) -> None:
            self.signals.append(sent_signal)

        def wait(self, timeout: int) -> int:
            self.waits += 1
            return 0

    def fake_run(command, **kwargs):
        assert kwargs["shell"] is False
        calls.append(command)
        return _completed(returncode=taskkill_code, stdout=b"")

    monkeypatch.setattr(docker_runner.subprocess, "run", fake_run)
    process = Process()
    controller = ProcessTreeController(platform="windows", system_root=system_root)
    assert controller.terminate(process) is expected
    assert process.signals == []
    assert process.waits == 1
    assert calls == [[str(taskkill.resolve()), "/PID", "789", "/T", "/F"]]
