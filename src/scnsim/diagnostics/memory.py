"""Low-cost, process-local memory snapshots for operation diagnostics.

Each reported metric carries its own source and unit. In particular, Linux
RSS/PSS and macOS process peak RSS are not physical-footprint measurements.
Unavailable platform counters stay explicitly unavailable.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter_ns


def _metric(value: int | None, *, source: str, status: str = "measured") -> dict[str, object]:
    return {
        "value": value,
        "unit": "bytes",
        "source": source,
        "status": status,
    }


def _unavailable(source: str) -> dict[str, object]:
    return _metric(None, source=source, status="unavailable")


def _linux_current_memory() -> dict[str, dict[str, object]]:
    metrics = {
        "rss": _unavailable("/proc/self/statm"),
        "pss": _unavailable("/proc/self/smaps_rollup:Pss"),
        "private_clean": _unavailable("/proc/self/smaps_rollup:Private_Clean"),
        "private_dirty": _unavailable("/proc/self/smaps_rollup:Private_Dirty"),
        "swap": _unavailable("/proc/self/smaps_rollup:Swap"),
    }
    try:
        with open("/proc/self/statm", "rt", encoding="ascii") as source:
            fields = source.read().split()
        if len(fields) >= 2:
            metrics["rss"] = _metric(int(fields[1]) * os.sysconf("SC_PAGE_SIZE"),
                                     source="/proc/self/statm resident pages")
    except (OSError, ValueError, IndexError):
        pass

    # smaps_rollup provides process-level aggregates without retaining or
    # walking individual mappings. Missing kernel counters remain unknown.
    try:
        with open("/proc/self/smaps_rollup", "rt", encoding="ascii") as source:
            values = {}
            for line in source:
                key, separator, rest = line.partition(":")
                if not separator:
                    continue
                fields = rest.split()
                if len(fields) >= 2 and fields[1] == "kB":
                    try:
                        values[key] = int(fields[0]) * 1024
                    except ValueError:
                        continue
        for name, key in (
            ("pss", "Pss"),
            ("private_clean", "Private_Clean"),
            ("private_dirty", "Private_Dirty"),
            ("swap", "Swap"),
        ):
            if key in values:
                metrics[name] = _metric(values[key], source=f"/proc/self/smaps_rollup:{key}")
    except OSError:
        pass
    return metrics


def _peak_rss() -> dict[str, object]:
    source = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError, ValueError):
        return _unavailable(source)
    if sys.platform.startswith("linux"):
        return _metric(raw * 1024, source=f"{source} (KiB on Linux)")
    if sys.platform == "darwin":
        return _metric(raw, source=f"{source} (bytes on macOS)")
    return _unavailable(f"{source} (unit not established on {sys.platform})")


def _darwin_physical_footprint() -> dict[str, object]:
    """Read this process's macOS physical footprint through libproc."""

    source = "libproc.proc_pid_rusage(RUSAGE_INFO_V0).ri_phys_footprint"
    try:
        import ctypes

        class _RusageInfoV0(ctypes.Structure):
            _fields_ = [
                ("ri_uuid", ctypes.c_uint8 * 16),
                ("ri_user_time", ctypes.c_uint64),
                ("ri_system_time", ctypes.c_uint64),
                ("ri_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_interrupt_wkups", ctypes.c_uint64),
                ("ri_pageins", ctypes.c_uint64),
                ("ri_wired_size", ctypes.c_uint64),
                ("ri_resident_size", ctypes.c_uint64),
                ("ri_phys_footprint", ctypes.c_uint64),
                ("ri_proc_start_abstime", ctypes.c_uint64),
                ("ri_proc_exit_abstime", ctypes.c_uint64),
            ]

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pid_rusage = libproc.proc_pid_rusage
        proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        proc_pid_rusage.restype = ctypes.c_int
        usage = _RusageInfoV0()
        buffer_address = ctypes.cast(ctypes.byref(usage), ctypes.c_void_p)
        status = proc_pid_rusage(os.getpid(), 0, buffer_address)
        if status != 0:
            error_number = ctypes.get_errno()
            detail = f"errno={error_number}" if error_number else f"status={status}"
            return _unavailable(f"{source} ({detail})")
        return _metric(int(usage.ri_phys_footprint), source=source)
    except (AttributeError, OSError, ValueError):
        return _unavailable(f"{source} (provider unavailable)")


def process_memory_snapshot(*, boundary: str) -> dict[str, object]:
    """Return one best-effort current-process snapshot with explicit domains."""

    if boundary not in {"operation_start", "operation_end"}:
        raise ValueError("memory snapshot boundary must name an operation boundary")
    metrics = {
        "rss": _unavailable("current RSS source unavailable on this platform"),
        "pss": _unavailable("current PSS source unavailable on this platform"),
        "private_clean": _unavailable("current private-clean source unavailable on this platform"),
        "private_dirty": _unavailable("current private-dirty source unavailable on this platform"),
        "swap": _unavailable("current process swap source unavailable on this platform"),
    }
    if sys.platform.startswith("linux"):
        metrics.update(_linux_current_memory())
    metrics["peak_rss"] = _peak_rss()
    if sys.platform == "darwin":
        metrics["physical_footprint"] = _darwin_physical_footprint()
    else:
        # Never substitute RSS for this distinct counter.
        metrics["physical_footprint"] = _unavailable(
            "platform physical-footprint provider not configured"
        )
    return {
        "scope": "current_process",
        "platform": sys.platform,
        "boundary": boundary,
        "sampled_monotonic_ns": perf_counter_ns(),
        "metrics": metrics,
    }
