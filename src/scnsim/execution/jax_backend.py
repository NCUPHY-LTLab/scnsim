"""Operation-local configuration and execution of the shared physical program.

Compilation/native physical programs own stable transformable functions and explicit
warm signatures. This adapter owns configured device/precision and per-call
synchronization, never candidate coefficients, tapes, factors or kernel caches.
Execution owns ready-wave scheduling and passes prepared sparse packets/runner
explicitly. Numerical callbacks and physical equations have one shared authority.
"""
from __future__ import annotations

import importlib.metadata
import os
from contextlib import nullcontext
from time import perf_counter_ns
import numpy as np


def _span(trace, kind, **details):
    return nullcontext(None) if trace is None else trace.span(kind, details=details)


def get_jax_backend(*, precision, resources, trace, operation_resources=None, diagnostics=True):
    return JaxBackend(precision=precision, resources=resources, trace=trace,
                      operation_resources=operation_resources, diagnostics=diagnostics)


class JaxBackend:
    def __init__(self, *, precision="float64", resources=None, trace=None, device="cpu", cpu_threads=None,
                 operation_resources=None, diagnostics=True):
        if device != "cpu":
            raise ValueError("JAX execution supports only the CPU device")
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        from .config import ensure_jax_configuration, get_runtime_configuration, mark_jax_initialized
        configured = get_runtime_configuration()
        resources = configured if resources is None else resources
        if resources != configured:
            raise RuntimeError("JAX resources differ from configure_runtime; restart the Kernel to change initialized resources")
        if cpu_threads is not None and cpu_threads != resources.cpu_threads:
            raise RuntimeError("use configure_runtime(cpu_threads=...) before JAX initialization")
        self.trace = trace
        self.operation_resources = operation_resources
        self.diagnostics = diagnostics
        started = perf_counter_ns()
        with _span(trace, "numerical_backend_initialization", precision=precision):
            from .._native import require_compensated_kernel
            require_compensated_kernel()
            ensure_jax_configuration()
            import jax
            self.jax = jax
            self.device = jax.devices("cpu")[0]
            mark_jax_initialized(resources)
        self.precision = precision
        self.real_dtype = np.float32 if precision == "float32" else np.float64
        self.complex_dtype = np.complex64 if precision == "float32" else np.complex128
        self.cpu_threads = resources.cpu_threads
        self._closed = False
        self.initialization_ns = perf_counter_ns() - started


    def identity(self) -> dict[str, object]:
        return {
            "backend": "jax", "algorithm_id": "scnsim.jax.native_physical.stage_bundle.v1",
            "jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
            "scipy": importlib.metadata.version("scipy"),
            "threadpoolctl": importlib.metadata.version("threadpoolctl"),
            "sparse_solver": "scipy.sparse.linalg.splu", "factorization_backend": "SuperLU",
            "superlu_version": None, "superlu_version_status": "not exposed by scipy public API",
            "assembly": "stable_parameter_field_view_sparse_value_program",
            "executable_ownership": "explicit_Run_warm_or_operation_static_program",
            "pure_inductive_stamp": "exact-realized-zero-R-real-dtype-scaled-analytic-width1-2-L-inverse-LU-larger",
            "compensated_arithmetic": {
                "scope": "transfer_zero.Z.non_port_realizable",
                "representation": "two_components_in_requested_base_dtype",
                "coverage": "indexed_assembly_pair_residual_selected_formation",
                "pair_product_kernel": "callback_local_cpp17_ordered_actual_dtype_csr.v1",
                "correction": "one_per_actual_primal_or_derivative_equation_same_high_factor",
                "projection": "F_Fp_then_Y_Yp_declared_complex_dtype",
            },
            "dtype": f"{self.precision}/{np.dtype(self.complex_dtype).name}", "arithmetic_precision": self.precision, "device": str(self.device),
            "platform": self.device.platform, "device_kind": self.device.device_kind,
            "requested_cpu_threads": self.cpu_threads,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
            "initialization_ns": self.initialization_ns,
        }






    def execute_native(self, function, *operands, precision, compile_key,
                       owner=None, phase='native_physical_execution', details=None):
        """Execute one pinned static native variant and drain before releasing it.

        The caller supplies the same canonical variant key used by warmup. The
        operation resource/BLAS scope remains outside this call and must outlive
        synchronization; no resource, trace or owner enters a numerical closure.
        """
        if self._closed:
            raise RuntimeError('numerical backend is closed')
        if precision != self.precision:
            raise ValueError('native precision differs from its execution backend')
        trace = self.trace if self.diagnostics else None
        # Original physical theta is binary64 even for selected32 numerical
        # arithmetic. Native functions explicitly own their selected casts.
        with self.jax.enable_x64(True), self.jax.default_device(self.device):
            def compile_variant():
                with _span(trace, 'native_compilation', phase=phase):
                    return self.jax.jit(function).lower(*operands).compile()
            acquired = (owner.acquire(compile_key, compile_variant, kind='variant', known_bytes=None)
                        if owner is not None else nullcontext(compile_variant()))
            with acquired as executable:
                with _span(trace, phase, **({} if details is None else details)):
                    outcome = executable(*operands)
                    self.jax.block_until_ready(outcome)
                    return outcome

    def native_program_identity(self, operation):
        """Actual single native route and its mathematical algorithm identity."""
        if operation not in ('direct', 'evaluate', 'optimize', 'sensitivity'):
            raise ValueError('unknown native physical operation identity')
        identity = self.identity()
        identity.update(algorithm_id=f'scnsim.jax.native_physical.{operation}.stage_bundle.v1',
                        assembly='array_only_candidate_once_sparse_values_frequency',
                        executable_ownership='explicit_bounded_compile_owner_pinned_until_synchronized')
        return identity


    def close(self) -> None:
        self._closed = True
        self.trace = None
        self.operation_resources = None

