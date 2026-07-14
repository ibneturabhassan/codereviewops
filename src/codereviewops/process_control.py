"""Cross-platform subprocess tree ownership and bounded termination."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal


class ProcessTreeController:
    """Start a subprocess in its own tree and terminate/reap that tree."""

    def __init__(
        self,
        platform: str | None = None,
        system_root: Path | None = None,
        *,
        windows_api: Any | None = None,
    ) -> None:
        self.platform = platform or ("windows" if os.name == "nt" else "posix")
        self.system_root = system_root
        self._windows_api = windows_api

    def prepare(self) -> ProcessLease:
        if self.platform == "posix":
            return _PosixProcessLease()
        if self.platform == "windows":
            return _WindowsProcessLease(self._windows_api or _WindowsJobApi())
        raise ValueError("process platform is unsupported")

    def popen_kwargs(self) -> dict[str, Any]:
        if self.platform == "posix":
            return {"start_new_session": True}
        if self.platform == "windows":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        raise ValueError("process platform is unsupported")

    def _taskkill(self, pid: int) -> bool:
        root_value = self.system_root or Path(os.environ.get("SYSTEMROOT") or "")
        if not root_value.is_absolute():
            return False
        try:
            root = root_value.resolve(strict=True)
            executable = (root / "System32" / "taskkill.exe").resolve(strict=True)
        except OSError:
            return False
        if not executable.is_file() or not executable.is_relative_to(root):
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

        def reap() -> bool:
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                return False
            return True

        if self.platform == "windows":
            tree_cleaned = self._taskkill(pid)
            parent_reaped = reap()
            return tree_cleaned and parent_reaped
        if self.platform != "posix":
            return False
        kill_group: Any = getattr(os, "killpg")  # noqa: B009
        try:
            kill_group(pid, signal.SIGTERM)
        except ProcessLookupError:
            return reap()
        except OSError:
            reap()
            return False
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=2)
        try:
            kill_group(pid, getattr(signal, "SIGKILL", 9))
        except ProcessLookupError:
            return reap()
        except OSError:
            reap()
            return False
        if not reap():
            return False
        try:
            kill_group(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False


class ProcessLease:
    """Prepared ownership boundary for exactly one subprocess tree."""

    def popen_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError

    def command(self, executable: Path, module: str, kind: Literal["repo", "test"]) -> list[str]:
        del kind
        return [str(executable), "-m", module]

    def attach(self, process: subprocess.Popen[bytes]) -> bool:
        raise NotImplementedError

    def terminate_and_verify(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _PosixProcessLease(ProcessLease):
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.pgid: int | None = None
        self._result: bool | None = None
        self._closed = False

    def popen_kwargs(self) -> dict[str, Any]:
        return {"start_new_session": True}

    def attach(self, process: subprocess.Popen[bytes]) -> bool:
        if self.process is not None or self._closed or process.pid <= 0:
            return False
        self.process = process
        self.pgid = process.pid
        return True

    def _kill_group(self, sent_signal: int) -> None:
        kill_group: Any = getattr(os, "killpg")  # noqa: B009
        kill_group(self.pgid, sent_signal)

    def _group_exists(self) -> bool:
        assert self.pgid is not None
        try:
            self._kill_group(0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True

    def _wait_group_gone(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while self._group_exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        return not self._group_exists()

    def _reap_leader(self) -> bool:
        assert self.process is not None
        try:
            self.process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return True

    def terminate_and_verify(self) -> bool:
        if self._result is not None:
            return self._result
        if self.process is None or self.pgid is None or self._closed:
            self._result = False
            return False
        try:
            self._kill_group(signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            self._reap_leader()
            self._result = False
            return False
        if not self._wait_group_gone(1.0):
            try:
                self._kill_group(getattr(signal, "SIGKILL", 9))
            except ProcessLookupError:
                pass
            except OSError:
                self._reap_leader()
                self._result = False
                return False
        self._result = self._reap_leader() and self._wait_group_gone(2.0)
        return self._result

    def close(self) -> None:
        self._closed = True


class _WindowsJobApi:
    KILL_ON_JOB_CLOSE = 0x00002000
    HANDLE_FLAG_INHERIT = 0x00000001

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class BasicAccounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self.ExtendedLimit = ExtendedLimit
        self.BasicAccounting = BasicAccounting
        k32 = self.kernel32
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.CreateEventW.argtypes = [
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        k32.CreateEventW.restype = wintypes.HANDLE
        k32.SetHandleInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        k32.SetHandleInformation.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.SetEvent.argtypes = [wintypes.HANDLE]
        k32.SetEvent.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL

    def create_job(self) -> int:
        job = self.kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")
        info = self.ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
        if not self.kernel32.SetInformationJobObject(
            job, 9, self.ctypes.byref(info), self.ctypes.sizeof(info)
        ):
            self.kernel32.CloseHandle(job)
            raise OSError("SetInformationJobObject failed")
        return int(job)

    def create_barrier(self) -> int:
        event = self.kernel32.CreateEventW(None, True, False, None)
        if not event:
            raise OSError("CreateEventW failed")
        if not self.kernel32.SetHandleInformation(
            event, self.HANDLE_FLAG_INHERIT, self.HANDLE_FLAG_INHERIT
        ):
            self.kernel32.CloseHandle(event)
            raise OSError("SetHandleInformation failed")
        return int(event)

    def startup_info(self, barrier: int) -> Any:
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": [barrier]}
        return startup

    def assign(self, job: int, process: subprocess.Popen[bytes]) -> bool:
        handle = getattr(process, "_handle", None)
        return bool(handle and self.kernel32.AssignProcessToJobObject(job, handle))

    def release(self, event: int) -> bool:
        return bool(self.kernel32.SetEvent(event))

    def terminate_job(self, job: int) -> bool:
        return bool(self.kernel32.TerminateJobObject(job, 1))

    def active_processes(self, job: int) -> int | None:
        info = self.BasicAccounting()
        if not self.kernel32.QueryInformationJobObject(
            job, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None
        ):
            return None
        return int(info.ActiveProcesses)

    def close_handle(self, handle: int) -> None:
        self.kernel32.CloseHandle(handle)


class _WindowsProcessLease(ProcessLease):
    def __init__(self, api: Any) -> None:
        self.api = api
        self.job = int(api.create_job())
        try:
            self.barrier = int(api.create_barrier())
        except Exception:
            api.close_handle(self.job)
            raise
        self.process: subprocess.Popen[bytes] | None = None
        self._attached = False
        self._result: bool | None = None
        self._closed = False

    def popen_kwargs(self) -> dict[str, Any]:
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            "close_fds": True,
            "startupinfo": self.api.startup_info(self.barrier),
        }

    def command(self, executable: Path, module: str, kind: Literal["repo", "test"]) -> list[str]:
        del module
        return [
            str(executable),
            "-m",
            "codereviewops.mcp_bootstrap",
            kind,
            str(self.barrier),
        ]

    def _reap(self) -> bool:
        assert self.process is not None
        try:
            self.process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return True

    def attach(self, process: subprocess.Popen[bytes]) -> bool:
        if self.process is not None or self._closed:
            return False
        self.process = process
        if not self.api.assign(self.job, process):
            with suppress(OSError):
                process.terminate()
            self._reap()
            return False
        self._attached = True
        if not self.api.release(self.barrier):
            self.api.terminate_job(self.job)
            self._reap()
            return False
        self.api.close_handle(self.barrier)
        self.barrier = 0
        return True

    def terminate_and_verify(self) -> bool:
        if self._result is not None:
            return self._result
        if not self._attached or self.process is None or self._closed:
            self._result = False
            return False
        terminated = bool(self.api.terminate_job(self.job))
        leader_reaped = self._reap()
        active = self.api.active_processes(self.job)
        self._result = terminated and leader_reaped and active == 0
        return self._result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.barrier:
            self.api.close_handle(self.barrier)
            self.barrier = 0
        if self.job:
            self.api.close_handle(self.job)
            self.job = 0
