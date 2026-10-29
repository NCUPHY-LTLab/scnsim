"""Constant-space, process-local timing for operations with no Workspace."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock, local
from time import perf_counter_ns
from types import MappingProxyType
from typing import Literal
from uuid import UUID

from .memory import process_memory_snapshot


OperationStatus = Literal["running", "success", "failure", "interrupted"]


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Detach the small scalar trees used by timing and OS memory metadata."""

    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(nested) for key, nested in item.items()})
        if isinstance(item, (tuple, list)):
            return tuple(freeze(nested) for nested in item)
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError("in-memory operation metadata must contain only detached scalar values")

    return freeze(value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class InMemoryTiming:
    """One bounded aggregate for a named operation phase."""

    phase: str
    count: int
    elapsed_ns: int
    minimum_ns: int
    maximum_ns: int
    status_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("timing phase must be nonempty")
        object.__setattr__(self, "status_counts", _freeze_mapping(self.status_counts))


@dataclass(frozen=True, slots=True)
class InMemoryBenchmarkResult:
    """Read-only timings for one in-memory operation, without saved-run identity.

    This value has no Workspace path, attempt/result digest, or persistence
    authority. Its UUID names only the live in-memory execution that produced
    the attached result.
    """

    execution_id: str
    operation: str | None
    request_sha256: str | None
    backend: str | None
    precision: str | None
    status: OperationStatus
    elapsed_ns: int
    timings: tuple[InMemoryTiming, ...]
    memory: Mapping[str, object]
    failure: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        UUID(self.execution_id)
        if self.request_sha256 is not None and (
            len(self.request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.request_sha256)
        ):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        if self.status not in {"running", "success", "failure", "interrupted"}:
            raise ValueError("unsupported in-memory operation status")
        if self.elapsed_ns < 0:
            raise ValueError("elapsed_ns cannot be negative")
        object.__setattr__(self, "timings", tuple(self.timings))
        object.__setattr__(self, "memory", _freeze_mapping(self.memory))
        if self.failure is not None:
            object.__setattr__(self, "failure", _freeze_mapping(self.failure))

    def summary(self) -> Mapping[str, object]:
        """Return a compact detached summary; phase detail remains in timings."""

        return MappingProxyType({
            "execution_id": self.execution_id,
            "operation": self.operation,
            "request_sha256": self.request_sha256,
            "backend": self.backend,
            "precision": self.precision,
            "status": self.status,
            "elapsed_ns": self.elapsed_ns,
            "phase_count": len(self.timings),
            "timed_call_count": sum(item.count for item in self.timings),
            "storage": "none",
            "scope": "one in-memory operation",
        })

    def document(self) -> Mapping[str, object]:
        """Expose the supported in-memory report fields without saved-report APIs."""

        values: dict[str, object] = dict(self.summary())
        values["timings"] = self.timings
        values["memory"] = self.memory
        if self.failure is not None:
            values["failure"] = self.failure
        return MappingProxyType(values)


class _TimingAggregate:
    __slots__ = ("count", "elapsed_ns", "minimum_ns", "maximum_ns", "status_counts")

    def __init__(self) -> None:
        self.count = 0
        self.elapsed_ns = 0
        self.minimum_ns: int | None = None
        self.maximum_ns: int | None = None
        self.status_counts: dict[str, int] = {}

    def add(self, elapsed_ns: int, status: str) -> None:
        self.count += 1
        self.elapsed_ns += elapsed_ns
        self.minimum_ns = elapsed_ns if self.minimum_ns is None else min(self.minimum_ns, elapsed_ns)
        self.maximum_ns = elapsed_ns if self.maximum_ns is None else max(self.maximum_ns, elapsed_ns)
        self.status_counts[status] = self.status_counts.get(status, 0) + 1

    def freeze(self, phase: str) -> InMemoryTiming:
        return InMemoryTiming(
            phase=phase,
            count=self.count,
            elapsed_ns=self.elapsed_ns,
            minimum_ns=0 if self.minimum_ns is None else self.minimum_ns,
            maximum_ns=0 if self.maximum_ns is None else self.maximum_ns,
            status_counts=self.status_counts,
        )


class MemoryOperationRecorder:
    """Record phase aggregates and process-boundary memory without persistence."""

    def __init__(
        self,
        *,
        execution_id: str,
        kind: str,
        backend: str | None,
        precision: str | None,
        start_tick_ns: int | None = None,
    ) -> None:
        UUID(execution_id)
        if not isinstance(kind, str) or not kind:
            raise ValueError("operation kind must be nonempty")
        self.execution_id = execution_id
        self.method = kind
        self.backend = backend
        self.precision = precision
        self._start_ns = perf_counter_ns() if start_tick_ns is None else start_tick_ns
        self._lock = RLock()
        self._local = local()
        self._aggregates: dict[str, _TimingAggregate] = {}
        self._memory: dict[str, object] = {
            "start": process_memory_snapshot(boundary="operation_start")
        }
        self._operation: str | None = kind
        self._request_sha256: str | None = None
        self._end_ns: int | None = None
        self._status: OperationStatus = "running"
        self._failure: Mapping[str, object] | None = None

    def __enter__(self) -> MemoryOperationRecorder:
        with self._lock:
            if self._end_ns is not None:
                raise RuntimeError("finished in-memory operation cannot be entered")
        return self

    def __exit__(self, error_type, error: BaseException | None, traceback) -> bool:
        del error_type, traceback
        self.finish(error=error)
        return False

    @property
    def current_scope(self) -> str:
        stack = getattr(self._local, "stack", None)
        return "operation" if not stack else stack[-1]

    @property
    def timing_mode(self) -> Literal["aggregate"]:
        return "aggregate"

    @property
    def root_start_tick_ns(self) -> int:
        return self._start_ns

    def bind(
        self,
        *,
        operation: str,
        request_sha256: str,
        task_id: str | None = None,
        environment_sha256: str | None = None,
        attempt_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Bind live request metadata; task/attempt details are not retained."""

        del task_id, environment_sha256, attempt_id, details
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be nonempty")
        if (
            not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256)
        ):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        with self._lock:
            self._ensure_open()
            self._operation = operation
            self._request_sha256 = request_sha256

    @contextmanager
    def span(
        self,
        kind: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        del details
        if not isinstance(kind, str) or not kind:
            raise ValueError("span kind must be nonempty")
        with self._lock:
            self._ensure_open()
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        stack.append(kind)
        start_ns = perf_counter_ns()
        error: BaseException | None = None
        try:
            yield kind
        except BaseException as caught:
            error = caught
            raise
        finally:
            end_ns = perf_counter_ns()
            if stack and stack[-1] == kind:
                stack.pop()
            self._record(kind, start_ns, end_ns, self._error_status(error))

    def measure(
        self,
        kind: str,
        *,
        start_tick_ns: int,
        end_tick_ns: int,
        parent_span_id: str | None = None,
        details: Mapping[str, object] | None = None,
        status: str = "success",
    ) -> None:
        del parent_span_id, details
        self._record(kind, start_tick_ns, end_tick_ns, status)

    def sparse_runner(
        self,
        host_function,
        operands: tuple[object, ...],
        options: Mapping[str, object],
    ) -> object:
        """Pass a per-call phase observer to the existing checked host seam."""

        def observe(phase: str, *, equation: str):
            return self.span(phase, details={"equation": equation, "transport": "python_host"})

        return host_function(*operands, **options, observer=observe)

    def _record(self, kind: str, start_ns: int, end_ns: int, status: str) -> None:
        if not isinstance(kind, str) or not kind:
            raise ValueError("timing kind must be nonempty")
        if end_ns < start_ns:
            raise ValueError("timing end precedes start")
        elapsed = end_ns - start_ns
        with self._lock:
            self._ensure_open()
            aggregate = self._aggregates.get(kind)
            if aggregate is None:
                aggregate = self._aggregates[kind] = _TimingAggregate()
            aggregate.add(elapsed, status)

    def _ensure_open(self) -> None:
        if self._end_ns is not None:
            raise RuntimeError("finished in-memory operation cannot be changed")

    @staticmethod
    def _error_status(error: BaseException | None) -> str:
        if error is None:
            return "success"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return "interrupted"
        return "failure"

    @staticmethod
    def _failure_record(error: BaseException | None) -> Mapping[str, object] | None:
        if error is None:
            return None
        record: dict[str, object] = {
            "type": type(error).__name__,
            "module": type(error).__module__,
        }
        # Diagnostics must not replace the exception that the caller is
        # already handling if a custom exception formatter itself fails.
        try:
            record["message"] = str(error)
        except BaseException as formatting_error:
            record["message"] = None
            record["message_error"] = type(formatting_error).__name__
        for name in ("kind", "category", "stage"):
            try:
                value = getattr(error, name, None)
            except BaseException as attribute_error:
                record[f"{name}_error"] = type(attribute_error).__name__
                continue
            if value is not None:
                try:
                    record[name] = str(value)
                except BaseException as formatting_error:
                    record[f"{name}_error"] = type(formatting_error).__name__
        return MappingProxyType(record)

    def finish(
        self,
        *,
        status: OperationStatus | None = None,
        error: BaseException | None = None,
    ) -> InMemoryBenchmarkResult:
        """Capture the end boundary once and return the immutable report."""

        with self._lock:
            if self._end_ns is None:
                inferred_status = self._error_status(error)
                if status is not None and status != inferred_status:
                    raise ValueError("status disagrees with the primary operation outcome")
                self._end_ns = perf_counter_ns()
                self._status = inferred_status  # type: ignore[assignment]
                self._failure = self._failure_record(error)
                self._memory["end"] = process_memory_snapshot(boundary="operation_end")
            elif error is not None and self._failure is None:
                raise RuntimeError("finished operation cannot be rebound to a later error")
            return self._snapshot_locked()

    def snapshot(self) -> InMemoryBenchmarkResult:
        """Return a compact read-only view; running snapshots have no end sample."""

        with self._lock:
            return self._snapshot_locked()

    def stats(self) -> tuple[InMemoryTiming, ...]:
        """Return current constant-space phase aggregates."""

        with self._lock:
            return tuple(
                self._aggregates[name].freeze(name)
                for name in sorted(self._aggregates)
            )

    def _snapshot_locked(self) -> InMemoryBenchmarkResult:
        end_ns = self._end_ns if self._end_ns is not None else perf_counter_ns()
        if end_ns < self._start_ns:
            raise ValueError("operation end precedes its monotonic start")
        elapsed = end_ns - self._start_ns
        return InMemoryBenchmarkResult(
            execution_id=self.execution_id,
            operation=self._operation,
            request_sha256=self._request_sha256,
            backend=self.backend,
            precision=self.precision,
            status=self._status,
            elapsed_ns=elapsed,
            timings=tuple(
                self._aggregates[name].freeze(name)
                for name in sorted(self._aggregates)
            ),
            memory=self._memory,
            failure=self._failure,
        )


__all__ = ["InMemoryBenchmarkResult", "InMemoryTiming", "MemoryOperationRecorder"]
