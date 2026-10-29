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
from threading import RLock, Event, get_ident

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
    actual_builds: int = 0
    singleflight_waits: int = 0
    pending_builds: int = 0


@dataclass(slots=True)
class _Entry:
    value: object
    known_bytes: int | None
    touched: int
    pins: int = 0
    retire: bool = False


@dataclass(slots=True)
class _Flight:
    event: object
    builder_thread: int
    epoch: int
    users: int = 0
    ready: bool = False
    value: object = None
    error: BaseException | None = None
    entry: object = None


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
        self._epoch = 0
        self._flights = {}
        self._actual_builds = self._singleflight_waits = 0
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

    def _active_pins(self, entry):
        # Flight participants are virtual pins until their contexts finish. This
        # avoids per-waiter reservation mutations during terminal publication.
        return entry.pins + sum(flight.users for flight in self._flights.values()
                                if flight.entry is entry)

    def _evict_one(self, tables):
        choices = [(entry.touched, table, key) for table in tables
                   for key, entry in table.items() if not self._active_pins(entry)]
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
            if self._active_pins(entry):
                entry.retire = True
            else:
                del table[key]
            return True

    def _release(self, key, kind, entry):
        with self._lock:
            entry.pins -= 1
            table, _ = self._table(kind)
            if not self._active_pins(entry) and entry.retire and table.get(key) is entry:
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
    def acquire(self, key, factory, *, kind='program', known_bytes=None, acquisition=None):
        """Share one actual build per owner/key, without locking over compile.

        Registered followers pin the published item before notification. A
        follower interruption detaches only that follower. Build failure wakes
        every participant; no failed flight becomes a persistent cache entry.
        """
        flight_key = (kind, key)
        waited = False
        with self._lock:
            entry = self._lookup(key, kind)
            if entry is not None:
                entry.pins += 1
                flight = None
                leader = False
            else:
                flight = self._flights.get(flight_key)
                leader = flight is None
                if leader:
                    flight = _Flight(Event(), get_ident(), self._epoch)
                    self._flights[flight_key] = flight
                    self._actual_builds += 1
                elif not flight.ready:
                    if flight.builder_thread == get_ident():
                        raise RuntimeError('recursive compilation acquisition of the same owner/key')
                    self._singleflight_waits += 1
                    waited = True
                flight.users += 1
        if flight is None:
            try:
                if acquisition is not None:
                    acquisition.update(state="reused", admitted=True)
                yield entry.value
            finally:
                self._release(key, kind, entry)
            return
        try:
            if leader:
                try:
                    value = factory()
                    size = known_bytes(value) if callable(known_bytes) else known_bytes
                    if size is not None and (type(size) is not int or size < 0):
                        raise ValueError('known_bytes must be None or a nonnegative integer')
                    with self._lock:
                        entry = (self._admit(key, value, kind, size)
                                 if flight.epoch == self._epoch else None)
                        if entry is None:
                            if flight.epoch != self._epoch:
                                self._operation_owned += 1
                            flight.value = value
                        else:
                            flight.entry = entry
                            flight.value = entry.value
                        del value
                except BaseException as error:
                    with self._lock:
                        flight.error = error
                finally:
                    with self._lock:
                        flight.ready = True
                        flight.event.set()
            else:
                flight.event.wait()
            with self._lock:
                error, value = flight.error, flight.value
            if error is not None:
                raise error
            if acquisition is not None:
                with self._lock:
                    table, _ = self._table(kind)
                    admitted = flight.entry is not None and table.get(key) is flight.entry and not flight.entry.retire
                acquisition.update(state="compiled" if leader else "waited" if waited else "reused", admitted=admitted)
            yield value
        finally:
            with self._lock:
                flight.users -= 1
                if not flight.users and self._flights.get(flight_key) is flight:
                    del self._flights[flight_key]
                    table, _ = self._table(kind)
                    if flight.entry is not None and flight.entry.retire and not self._active_pins(flight.entry) and table.get(key) is flight.entry:
                        del table[key]
                    # Do not retain completed values or traceback-bearing errors.
                    flight.value = flight.error = flight.entry = None


    def clear(self):
        """Release inactive entries and retire active ones; no global purge."""
        with self._lock:
            self._epoch += 1
            for table in (self._programs, self._variants):
                for key, entry in tuple(table.items()):
                    if self._active_pins(entry):
                        entry.retire = True
                    else:
                        del table[key]

    def stats(self):
        """Return detached counts; unknown byte sizes remain explicit."""
        with self._lock:
            entries = tuple(self._programs.values()) + tuple(self._variants.values())
            return CompileOwnerStats(len(self._programs), len(self._variants),
                self._known_bytes(), sum(e.known_bytes is None for e in entries),
                sum(bool(self._active_pins(e)) for e in entries), self._hits, self._misses,
                self._admissions, self._evictions, self._operation_owned,
                self._actual_builds, self._singleflight_waits,
                sum(not flight.ready for flight in self._flights.values()))
