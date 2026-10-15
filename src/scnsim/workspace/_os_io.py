"""Private platform authority for Workspace locks and filesystem publication.

Callers own descriptor acquisition/order and retain handles through their actual
operation lifetime. Unix flock semantics stay native. Windows uses synchronous
LockFileEx on the same byte [0, 1) for every shared/exclusive consumer; no handle
inheritance or Linux open-file-description guarantee is inferred from fd dup.
"""
from __future__ import annotations

import errno
import os
import sys
from functools import lru_cache
from pathlib import Path

from ..errors import UnsupportedRuntimePlatformError

if sys.platform in {"linux", "darwin"}:
    import fcntl


def require_workspace_platform(*, stage: str) -> None:
    if sys.platform not in {"linux", "darwin", "win32"}:
        raise UnsupportedRuntimePlatformError(
            "Workspace I/O is supported on Linux, macOS and Windows.",
            stage=stage, evidence={"platform": sys.platform},
        )


@lru_cache(maxsize=1)
def _windows_api():
    """Bind only immutable Win32 ABI metadata; never retain operation handles."""
    import ctypes
    from ctypes import wintypes

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(OVERLAPPED),
    ]
    kernel.LockFileEx.restype = wintypes.BOOL
    kernel.UnlockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, ctypes.POINTER(OVERLAPPED),
    ]
    kernel.UnlockFileEx.restype = wintypes.BOOL
    kernel.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel.FlushFileBuffers.restype = wintypes.BOOL
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    kernel.MoveFileExW.restype = wintypes.BOOL
    return kernel, OVERLAPPED


def lock_fd(descriptor: int, *, exclusive: bool, nonblocking: bool = False) -> None:
    """Acquire one native lock; only actual nonblocking contention means busy."""
    require_workspace_platform(stage="workspace_lock")
    if sys.platform != "win32":
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, mode | (fcntl.LOCK_NB if nonblocking else 0))
        return
    import ctypes
    import msvcrt

    kernel, overlapped_type = _windows_api()
    overlapped = overlapped_type()  # zero offset, reserved fields and event
    flags = (0x2 if exclusive else 0) | (0x1 if nonblocking else 0)
    if not kernel.LockFileEx(msvcrt.get_osfhandle(descriptor), flags, 0, 1, 0,
                             ctypes.byref(overlapped)):
        code = ctypes.get_last_error()
        error = ctypes.WinError(code)
        if nonblocking and code == 33:  # ERROR_LOCK_VIOLATION
            raise BlockingIOError(errno.EAGAIN, error.strerror) from error
        raise error


def unlock_fd(descriptor: int) -> None:
    """Release exactly the acquired [0, 1) Windows range or native Unix flock."""
    if sys.platform != "win32":
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    import ctypes
    import msvcrt

    kernel, overlapped_type = _windows_api()
    overlapped = overlapped_type()
    if not kernel.UnlockFileEx(msvcrt.get_osfhandle(descriptor), 0, 1, 0,
                               ctypes.byref(overlapped)):
        raise ctypes.WinError(ctypes.get_last_error())


def sync_fd(descriptor: int) -> None:
    """Flush a regular file using the platform's native write-through barrier."""
    if sys.platform != "win32":
        os.fsync(descriptor)
        return
    import ctypes
    import msvcrt

    kernel, _ = _windows_api()
    if not kernel.FlushFileBuffers(msvcrt.get_osfhandle(descriptor)):
        raise ctypes.WinError(ctypes.get_last_error())


def sync_directory(path: Path) -> None:
    """Issue the native directory barrier; unsupported/error is never success.

    Windows requests a real writable directory handle and FlushFileBuffers.
    This does not infer POSIX-equivalent hardware power-loss guarantees or
    swallow a filesystem's access/invalid-handle errors.
    """
    if sys.platform != "win32":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    import ctypes

    kernel, _ = _windows_api()
    handle = kernel.CreateFileW(
        str(Path(path).absolute()), 0x40000000, 0x7, None, 3, 0x02000000, None,
    )  # GENERIC_WRITE, SHARE_READ|WRITE|DELETE, OPEN_EXISTING, BACKUP_SEMANTICS
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        primary = sys.exc_info()[1]
        if not kernel.CloseHandle(handle):
            error = ctypes.WinError(ctypes.get_last_error())
            if primary is None:
                raise error
            primary.add_note(f"Directory handle close also failed: {error!r}")


def replace_path(source: Path, target: Path) -> None:
    """Publish on the same volume; never enable Windows copy/delete fallback.

    Only a native failure returned by the issued combined Windows move/flush
    receives an uncertainty marker. Logical-pointer callers retain their own
    indeterminate-commit authority; setup/path/argument failures do not get it.
    """
    if sys.platform != "win32":
        os.replace(source, target)
        return
    import ctypes

    kernel, _ = _windows_api()
    old_path = str(Path(source).absolute())
    new_path = str(Path(target).absolute())
    if not kernel.MoveFileExW(old_path, new_path, 0x1 | 0x8):
        error = ctypes.WinError(ctypes.get_last_error())
        error._scnsim_move_issued = True
        raise error
