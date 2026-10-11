"""Operation-local configuration and execution of the shared physical program.

Compilation/PhysicalProgram owns stable transformable functions and explicit
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
            "backend": "jax", "algorithm_id": "scnsim.jax.physical_program.v1",
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
                "pair_product_kernel": "callback_local_ordered_actual_dtype_csr.v1",
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


    def program_identity(self, operation):
        """Execution identity of the shared checked physical-program route."""
        if operation not in ('direct', 'evaluate', 'optimize', 'sensitivity'):
            raise ValueError('unknown physical-program operation identity')
        identity = self.identity()
        identity.update(algorithm_id=f'scnsim.jax.physical_program.{operation}.v1',
                        assembly='stable_parameter_field_view_sparse_value_program',
                        executable_ownership='explicit_Run_warm_or_operation_static_program',
                        physical_program=True)
        identity['compensated_arithmetic'] = {
            'scope': 'transfer_zero.Z.non_port_realizable',
            'representation': 'two_components_in_requested_base_dtype',
            'coverage': 'indexed_assembly_pair_residual_selected_formation',
            'pair_product_kernel': 'callback_local_ordered_actual_dtype_csr.v1',
            'correction': 'one_per_actual_primal_or_derivative_equation_same_high_factor',
            'projection': 'F_Fp_then_Y_Yp_declared_complex_dtype',
        }
        return identity


    def execute(self, program, method, inputs, *operands, sparse_runner=None, **options):
        """Run a checked program without retaining its point or equation frame.

        Execution owns resource scheduling and recording. This adapter scopes
        only configured device/dtype and an optional call-local sparse runner;
        it never serializes the whole program or captures resources in JIT.
        """
        if self._closed:
            raise RuntimeError('numerical backend is closed')
        if program.precision != self.precision:
            raise ValueError('program precision differs from its execution backend')
        with self.jax.enable_x64(self.precision == 'float64'), self.jax.default_device(self.device):
            with _span(self.trace if self.diagnostics else None, 'physical_program_evaluation', method=method):
                if method == 'evaluate_branch':
                    outcome = program.evaluate_branch(inputs, *operands, sparse_runner=sparse_runner, **options)
                else:
                    outcome = getattr(program, method)(inputs, *operands, sparse_runner=sparse_runner, **options)
                # Only synchronize actual device arrays. Detached typed root
                # results/certificates retain their own host representations.
                for leaf in self.jax.tree.leaves(outcome):
                    if hasattr(leaf, 'block_until_ready'):
                        leaf.block_until_ready()
                return outcome


    def close(self) -> None:
        self._closed = True
        self.trace = None
        self.operation_resources = None

