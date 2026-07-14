"""Hardened Docker-only runner for the fixed Python unittest profile."""

from __future__ import annotations

import os
import re
import shutil
import signal
import stat
import subprocess
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from codereviewops.models import TestStatus
from codereviewops.tools import TestExecution, ToolError

RUNNER_TAG = "codereviewops/python-unittest:0.1.0"
RUNNER_IMAGE = (
    "python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf"
)
TEST_PROFILE = "python-unittest-v1"
TEST_ARGV = ("python", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
OUTPUT_LIMIT = 1024 * 1024
RUN_TIMEOUT_SECONDS = 30
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_NAME = re.compile(r"^codereviewops-[0-9a-f]{32}$")


def _minimal_host_env() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("PATH", "SystemRoot", "SYSTEMROOT", "ComSpec", "COMSPEC"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _docker_executable() -> str:
    executable = shutil.which("docker")
    if executable is None:
        raise ToolError("docker_unavailable", "Docker CLI is unavailable")
    return executable


def _validated_mount(workspace: Path) -> str:
    try:
        resolved = workspace.resolve(strict=True)
        metadata = os.lstat(workspace)
    except OSError as exc:
        raise ToolError("invalid_path", "workspace mount is unavailable") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    source = str(resolved)
    if (
        not resolved.is_dir()
        or workspace.is_symlink()
        or workspace.is_junction()
        or bool(attributes & reparse_flag)
        or "," in source
        or any(ord(char) < 32 for char in source)
    ):
        raise ToolError("invalid_path", "workspace mount path is unsafe")
    return source


class _OutputCollector:
    def __init__(self) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()
        self._total = 0
        self._lock = threading.Lock()
        self.truncated = False

    def drain(self, stream: object, destination: bytearray) -> None:
        reader = getattr(stream, "read", None)
        if reader is None:
            return
        while True:
            chunk = reader(8192)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                continue
            with self._lock:
                remaining = max(0, OUTPUT_LIMIT - self._total)
                accepted = chunk[:remaining]
                destination.extend(accepted)
                self._total += len(accepted)
                if len(accepted) != len(chunk):
                    self.truncated = True


class ProcessTreeController:
    """Start and terminate the Docker client as an isolated process tree."""

    def __init__(
        self,
        platform: str | None = None,
        system_root: Path | None = None,
    ) -> None:
        self.platform = platform or ("windows" if os.name == "nt" else "posix")
        self.system_root = system_root

    def popen_kwargs(self) -> dict[str, Any]:
        if self.platform == "posix":
            return {"start_new_session": True}
        if self.platform == "windows":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        raise ToolError("docker_infrastructure", "process platform is unsupported")

    def _taskkill(self, pid: int) -> bool:
        root_value = self.system_root or Path(os.environ.get("SYSTEMROOT") or "")
        if not root_value.is_absolute():
            return False
        try:
            root = root_value.resolve(strict=True)
            executable = (root / "System32" / "taskkill.exe").resolve(strict=True)
        except OSError:
            return False
        if (
            executable.name.casefold() != "taskkill.exe"
            or not executable.is_file()
            or not executable.is_relative_to(root)
        ):
            return False
        try:
            completed = subprocess.run(
                [str(executable), "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=5,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def terminate(self, process: subprocess.Popen[bytes]) -> bool:
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            return False

        def reap_parent() -> bool:
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                return False
            return True

        if self.platform == "windows":
            tree_cleaned = self._taskkill(pid)
            parent_reaped = reap_parent()
            return tree_cleaned and parent_reaped
        if self.platform != "posix":
            return False

        kill_group: Any = getattr(os, "killpg")  # noqa: B009
        try:
            kill_group(pid, signal.SIGTERM)
        except ProcessLookupError:
            return reap_parent()
        except OSError:
            reap_parent()
            return False

        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=2)
        try:
            kill_group(pid, getattr(signal, "SIGKILL", 9))
        except ProcessLookupError:
            return reap_parent()
        except OSError:
            reap_parent()
            return False

        if not reap_parent():
            return False
        try:
            kill_group(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False


class DockerTestRunner:
    """Resolve the pinned local tag and execute only the approved profile."""

    requires_sealed_snapshot = True

    def __init__(
        self,
        docker: str | None = None,
        process_controller: ProcessTreeController | None = None,
    ) -> None:
        self._docker = docker or shutil.which("docker") or "docker"
        self._host_env = _minimal_host_env()
        self._process_controller = process_controller or ProcessTreeController()

    def image_id(self) -> str:
        try:
            completed = subprocess.run(
                [
                    self._docker,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    RUNNER_TAG,
                ],
                check=False,
                capture_output=True,
                timeout=10,
                env=self._host_env,
            )
        except FileNotFoundError as exc:
            raise ToolError("docker_unavailable", "Docker CLI is unavailable") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError("docker_infrastructure", "Docker image inspection failed") from exc
        output = completed.stdout.decode("ascii", errors="ignore").strip()
        if completed.returncode != 0 or _IMAGE_ID.fullmatch(output) is None:
            raise ToolError("docker_unavailable", "pinned local runner image is unavailable")
        return output

    def check(self) -> str:
        """Verify CLI, Linux daemon, immutable image, and hardened probe."""

        try:
            info = subprocess.run(
                [self._docker, "info", "--format", "{{.OSType}}"],
                check=False,
                capture_output=True,
                timeout=10,
                env=self._host_env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError("docker_unavailable", "Docker daemon is unavailable") from exc
        if info.returncode != 0 or info.stdout.decode("ascii", errors="ignore").strip() != "linux":
            raise ToolError("docker_unavailable", "a Linux Docker engine is required")
        image_id = self.image_id()
        probe = [
            self._docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--pull",
            "never",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "--workdir",
            "/workspace",
            "--tmpfs",
            "/workspace:rw,noexec,nosuid,nodev,size=1m",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=1m",
            image_id,
            "python",
            "-I",
            "-c",
            "print('probe-ok')",
        ]
        try:
            completed = subprocess.run(
                probe, check=False, capture_output=True, timeout=10, env=self._host_env
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError("docker_infrastructure", "hardened Docker probe failed") from exc
        if completed.returncode != 0 or completed.stdout.strip() != b"probe-ok":
            raise ToolError("docker_infrastructure", "hardened Docker probe failed")
        return image_id

    def _command(self, workspace: Path, image_id: str, container_name: str) -> list[str]:
        if _IMAGE_ID.fullmatch(image_id) is None:
            raise ToolError("docker_infrastructure", "Docker image identity is invalid")
        if _CONTAINER_NAME.fullmatch(container_name) is None:
            raise ToolError("docker_infrastructure", "Docker container identity is invalid")
        source = _validated_mount(workspace)
        return [
            self._docker,
            "run",
            "--rm",
            "--init",
            "--network",
            "none",
            "--read-only",
            "--pull",
            "never",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "--workdir",
            "/workspace",
            "--name",
            container_name,
            "--mount",
            f"type=bind,source={source},target=/workspace,readonly",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m",
            "--tmpfs",
            "/output:rw,noexec,nosuid,nodev,size=16m",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            image_id,
            *TEST_ARGV,
        ]

    def _cleanup(self, container_name: str) -> bool:
        if _CONTAINER_NAME.fullmatch(container_name) is None:
            return False
        try:
            completed = subprocess.run(
                [self._docker, "rm", "-f", container_name],
                check=False,
                capture_output=True,
                timeout=5,
                env=self._host_env,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def run(self, workspace: Path, profile: str) -> TestExecution:
        if profile != TEST_PROFILE:
            raise ToolError("unsupported_profile", "unsupported test profile")
        image_id = self.image_id()
        container_name = f"codereviewops-{uuid4().hex}"
        command = self._command(workspace, image_id, container_name)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._host_env,
                **self._process_controller.popen_kwargs(),
            )
        except OSError as exc:
            raise ToolError("docker_infrastructure", "Docker test runner could not start") from exc
        if process.stdout is None or process.stderr is None:
            cleaned = self._cleanup(container_name)
            tree_exited = self._process_controller.terminate(process)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            if not cleaned or not tree_exited:
                raise ToolError("cleanup_failed", "isolated test cleanup failed")
            raise ToolError("docker_infrastructure", "Docker output pipes are unavailable")

        collector = _OutputCollector()
        threads = [
            threading.Thread(
                target=collector.drain,
                args=(process.stdout, collector.stdout),
                daemon=True,
            ),
            threading.Thread(
                target=collector.drain,
                args=(process.stderr, collector.stderr),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            return_code = process.wait(timeout=RUN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            cleaned = self._cleanup(container_name)
            try:
                return_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tree_exited = self._process_controller.terminate(process)
                return_code = 137 if tree_exited else -1
        for thread in threads:
            thread.join(timeout=5)
        process.stdout.close()
        process.stderr.close()

        if timed_out:
            if not cleaned or return_code == -1 or any(thread.is_alive() for thread in threads):
                return TestExecution(
                    TestStatus.ERROR,
                    "isolated test cleanup failed",
                    infrastructure_error=True,
                    error_code="cleanup_failed",
                )
            return TestExecution(TestStatus.ERROR, "python-unittest-v1 timed out after 30s.")
        stdout = collector.stdout.decode("utf-8", errors="replace").strip()
        stderr = collector.stderr.decode("utf-8", errors="replace").strip()
        output = "\n".join(part for part in (stdout, stderr) if part)
        if collector.truncated:
            output = (output + "\n[output truncated]").strip()
        if return_code in {125, 126, 127}:
            return TestExecution(
                TestStatus.ERROR,
                "isolated test infrastructure failed",
                output_truncated=collector.truncated,
                infrastructure_error=True,
                error_code="docker_infrastructure",
            )
        status = TestStatus.PASSED if return_code == 0 else TestStatus.FAILED
        label = "passed" if status == TestStatus.PASSED else f"failed with exit {return_code}"
        summary = f"python-unittest-v1 {label}."
        if output:
            summary += "\n" + output
        return TestExecution(status, summary, output_truncated=collector.truncated)
