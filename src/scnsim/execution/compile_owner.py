"""Run-local bounded static compilation ownership, never a numerical cache.

Keys include actual structural/dtype/device identity supplied by compilation.
Entries may contain immutable programs or compiled handles, never candidate
operands, tapes, factors, Results, Run, or Workspace. Consumers borrow through
synchronization and any required pullback use. Eviction only drops this owner's
references; it never purges process-global JAX state or promises RSS release.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from threading import RLock

from .policy import ExecutionPolicy


@dataclass(frozen=True, slots=True)
class CompileOwnerStats:
    program_entries: int
    variant_entries: int
    known_bytes: int
    unknown_byte_entries: int
    pinned_entries: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    operation_owned: int


@dataclass(slots=True)
class _Entry:
    value: object
    known_bytes: int | None
    touched: int
    pins: int = 0
    retire: bool = False


class CompileOwner:
    """Separate program/variant LRUs with one shared known-byte target.

    ``admit`` returns False when active pins or budget prevent admission; the
    caller's item remains operation-owned. ``acquire`` builds on a miss and
    yields the actual item either way. ``borrow`` pins an existing item. Raw
    ``get`` is a lookup only and does not extend lifetime through computation.
    Unknown sizes are counted explicitly, never treated as measured zero bytes.
    """

    def __init__(self, *, policy: ExecutionPolicy):
        self.policy = policy
        self._programs = OrderedDict()
        self._variants = OrderedDict()
        self._lock = RLock()
        self._clock = 0
        self._hits = self._misses = self._admissions = self._evictions = self._operation_owned = 0

    @contextmanager
    def hold_exact_core(self, core):
        """Pin THIS core when resident; otherwise retain it only for this call.

        Eviction/readmission may put an equivalent but different static object at
        the same key. That resident object is not the callable's captured core and
        must not be reported/held as its pin. Detached callers remain valid.
        """
        if core is None:
            yield
            return
        with ExitStack() as stack:
            resident = stack.enter_context(self.acquire(core.owner_key, lambda: core,
                                                        kind='program', known_bytes=None))
            if resident is not core:
                stack.close()
            # The argument is a strong call-local reference even when unadmitted.
            yield


    def _table(self, kind):
        if kind == 'program':
            return self._programs, self.policy.compile_entries
        if kind == 'variant':
            return self._variants, self.policy.compile_variants
        raise ValueError("kind must be 'program' or 'variant'")

    def _touch(self, table, key, entry):
        self._clock += 1
        entry.touched = self._clock
        table.move_to_end(key)

    def _lookup(self, key, kind):
        table, _ = self._table(kind)
        entry = table.get(key)
        if entry is None or entry.retire:
            self._misses += 1
            return None
        self._hits += 1
        self._touch(table, key, entry)
        return entry

    def get(self, key, default=None, *, kind='program'):
        with self._lock:
            entry = self._lookup(key, kind)
            return default if entry is None else entry.value

    def _known_bytes(self):
        return sum(e.known_bytes for table in (self._programs, self._variants)
                   for e in table.values() if e.known_bytes is not None)

    def _evict_one(self, tables):
        choices = [(entry.touched, table, key) for table in tables
                   for key, entry in table.items() if not entry.pins]
        if not choices:
            return False
        _, table, key = min(choices, key=lambda item: item[0])
        del table[key]
        self._evictions += 1
        return True

    def _admit(self, key, value, kind, known_bytes):
        table, limit = self._table(kind)
        existing = table.get(key)
        if existing is not None:
            # Static identity is supplied by the compiler. Never replace a
            # borrowed object under an identical key.
            if existing.retire:
                self._operation_owned += 1
                return None
            self._touch(table, key, existing)
            return existing
        if known_bytes is not None and known_bytes > self.policy.compile_known_bytes:
            self._operation_owned += 1
            return None
        while len(table) >= limit:
            if not self._evict_one((table,)):
                self._operation_owned += 1
                return None
        added = 0 if known_bytes is None else known_bytes
        while self._known_bytes() + added > self.policy.compile_known_bytes:
            if not self._evict_one((self._programs, self._variants)):
                self._operation_owned += 1
                return None
        self._clock += 1
        entry = _Entry(value, known_bytes, self._clock)
        table[key] = entry
        self._admissions += 1
        return entry

    def admit(self, key, value, *, kind='program', known_bytes=None):
        """Try retaining a static item; False means caller-owned overflow."""
        if known_bytes is not None and (type(known_bytes) is not int or known_bytes < 0):
            raise ValueError('known_bytes must be None or a nonnegative integer')
        with self._lock:
            return self._admit(key, value, kind, known_bytes) is not None

    def remove(self, key, *, kind='program'):
        """Retire an item; active borrowers keep it until their final release."""
        with self._lock:
            table, _ = self._table(kind)
            entry = table.get(key)
            if entry is None:
                return False
            if entry.pins:
                entry.retire = True
            else:
                del table[key]
            return True

    def _release(self, key, kind, entry):
        with self._lock:
            entry.pins -= 1
            table, _ = self._table(kind)
            if not entry.pins and entry.retire and table.get(key) is entry:
                del table[key]

    @contextmanager
    def borrow(self, key, *, kind='program'):
        with self._lock:
            entry = self._lookup(key, kind)
            if entry is None:
                raise KeyError(key)
            entry.pins += 1
        try:
            yield entry.value
        finally:
            self._release(key, kind, entry)

    @contextmanager
    def acquire(self, key, factory, *, kind='program', known_bytes=None):
        """Borrow a hit or create and optionally admit a static miss.

        known_bytes may measure the created value via a callable.
        Factory executes outside the owner lock. A concurrent admission under
        the same structural key wins; the unused local value is released.
        """
        with self._lock:
            entry = self._lookup(key, kind)
            if entry is not None:
                entry.pins += 1
        if entry is None:
            value = factory()
            size = known_bytes(value) if callable(known_bytes) else known_bytes
            if size is not None and (type(size) is not int or size < 0):
                raise ValueError('known_bytes must be None or a nonnegative integer')
            with self._lock:
                entry = self._admit(key, value, kind, size)
                if entry is not None:
                    entry.pins += 1
            if entry is None:
                yield value
                return
            del value
        try:
            yield entry.value
        finally:
            self._release(key, kind, entry)

    def clear(self):
        """Release inactive entries and retire active ones; no global purge."""
        with self._lock:
            for table in (self._programs, self._variants):
                for key, entry in tuple(table.items()):
                    if entry.pins:
                        entry.retire = True
                    else:
                        del table[key]

    def stats(self):
        """Return detached counts; unknown byte sizes remain explicit."""
        with self._lock:
            entries = tuple(self._programs.values()) + tuple(self._variants.values())
            return CompileOwnerStats(len(self._programs), len(self._variants),
                self._known_bytes(), sum(e.known_bytes is None for e in entries),
                sum(bool(e.pins) for e in entries), self._hits, self._misses,
                self._admissions, self._evictions, self._operation_owned)
