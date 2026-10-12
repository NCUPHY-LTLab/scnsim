"""Immutable execution signatures shared by compile-only warmup and consumers.

Numerical producers supply the actual callable/key; this module selects and
owns only static descriptors. No Run, Workspace, candidate arrays, native
factor or live linearization residual belongs to a plan. Lazy frame consumers
must name their prerequisite rather than invent a compatible static tree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
    role: str
    key: tuple
    function: Callable | None
    abstract_operands: tuple
    prerequisites: tuple[tuple, ...] = ()
    lifetime: str = 'run_static'
    lazy_reason: str | None = None


@dataclass(frozen=True, slots=True)
class VariantPreparation:
    role: str
    key: tuple
    state: str
    elapsed_ns: int
    compile_ns: int | None
    admitted: bool
    generated_code_bytes: int | None
    lazy_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionVariantPlan:
    variants: tuple[ExecutionVariant, ...]

    def compile(self, owner, compiler):
        """Compile requested descriptors; injected compiler owns device scope."""
        from time import perf_counter_ns
        rows = []
        for variant in self.variants:
            if variant.lazy_reason is not None:
                rows.append(VariantPreparation(variant.role, variant.key, 'lazy',
                    0, None, False, None, variant.lazy_reason))
                continue
            elapsed_start = perf_counter_ns()
            compile_ns = None
            def build():
                nonlocal compile_ns
                started = perf_counter_ns()
                value = compiler(variant)
                compile_ns = perf_counter_ns() - started
                return value
            acquisition = {}
            with owner.acquire(variant.key, build, kind='variant', known_bytes=None, acquisition=acquisition) as value:
                admitted = acquisition['admitted']
                state = acquisition['state']
                size = getattr(value, 'size_of_generated_code_in_bytes', None)
                rows.append(VariantPreparation(variant.role, variant.key, state,
                    perf_counter_ns() - elapsed_start, compile_ns, admitted,
                    None if size is None else size()))
        return tuple(rows)
