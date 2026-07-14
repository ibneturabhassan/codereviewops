"""Windows bootstrap that blocks until parent process ownership is established."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

_WAIT_OBJECT_0 = 0
_BOOTSTRAP_TIMEOUT_MS = 10_000


def _server(kind: str) -> Callable[[], None]:
    from codereviewops.mcp_servers import run_repo_server, run_test_server

    servers = {"repo": run_repo_server, "test": run_test_server}
    try:
        return servers[kind]
    except KeyError:
        raise ValueError("invalid MCP bootstrap kind") from None


def main() -> int:
    if os.name != "nt" or len(sys.argv) != 3:
        return 2
    kind = sys.argv[1]
    if kind not in {"repo", "test"}:
        return 2
    try:
        handle = int(sys.argv[2], 10)
    except (ValueError, OverflowError):
        return 2
    if handle <= 0:
        return 2

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    try:
        if kernel32.WaitForSingleObject(handle, _BOOTSTRAP_TIMEOUT_MS) != _WAIT_OBJECT_0:
            return 3
    finally:
        kernel32.CloseHandle(handle)
    _server(kind)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
