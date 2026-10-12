"""Acquire a detached native function from exact Quantity/View declarations.

Static compilation owns model and layout selection. This adapter packs the
complete effective physical point, freezes non-wrt constants and scatters the
ordered requested coordinates. It cannot solve, create an attempt, persist a
Result, or capture the Run/Workspace/compile owner in the returned callable.
"""
from collections.abc import Mapping
from hashlib import sha256
import numpy as np

from ..canonical import canonical_json_bytes
from ..execution.prepared_function import PreparedFunction
from .compiler import parameter_key
from .quantity_program import compile_quantity_program


def compile_prepared_function(*, plan_document, quantity_declarations,
                              parameter_record, wrt, mesh=None,
                              precision='float64', program_owner=None):
    """Prepare full physical operands without evaluating assembly or quantities."""
    from ..execution.config import ensure_jax_configuration, mark_jax_initialized
    configuration = ensure_jax_configuration()
    if isinstance(quantity_declarations, Mapping):
        declarations = dict(quantity_declarations)
    else:
        declarations = {row['id']: row['quantity'] for row in quantity_declarations}
    authorized = frozenset(parameter_key(ref) for ref in
                           parameter_record.get('allow_extrapolation', ()))
    import jax
    with jax.enable_x64(True):
        program = compile_quantity_program(
            plan_document, declarations, parameter_record, mesh=mesh,
            precision=precision, authorized=authorized, program_owner=program_owner,
            admit_program=True,
        )
        prepared = prepared_from_quantity_program(program, parameter_record, wrt=wrt,
                                                  program_owner=program_owner)
    mark_jax_initialized(configuration)
    return prepared


def prepared_from_quantity_program(program, parameter_record, *, wrt=None, program_owner=None):
    """Detach the same acquired program, including objective-only warm preparation."""
    precision = program.program.precision
    layout = program.program.model_program.fields.layout
    keys = tuple(dict.fromkeys(axis.key for axis in layout.axes)) if wrt is None else tuple(wrt)
    indices = layout.input_indices(keys)
    # Physical input classification retains the original binary64 SI point.
    # The detached native core owns the subsequent selected-dtype boundary.
    dtype = np.float64
    constants = layout.pack(parameter_record, dtype=dtype)
    constants[list(indices)] = 0
    # Immutable host constants, not a retained runtime point/device array.
    constants = np.frombuffer(constants.tobytes(), dtype=dtype)
    import jax
    with jax.enable_x64(True):
        native = program.native_program(program.inputs(parameter_record),
                                        program_owner=program_owner, admit_core=True)
    function = native.numeric_function
    core_input_layout = dict(native.input_layout)
    selected_axes = tuple(dict(key=axis.key, name=axis.name, si_unit=axis.si_unit,
                               matrix_field=axis.matrix_field,
                               matrix_entry=axis.matrix_entry,
                               full_index=index)
                          for index in indices for axis in (layout.axes[index],))
    input_layout = {**core_input_layout,
                    'theta': {'shape': (len(indices),), 'dtype': np.dtype(dtype).name,
                              'coordinates': selected_axes,
                              'numeric_dtype': precision},
                    'full_theta_shape': (len(layout.axes),),
                    'active_keys': keys}
    fixed_mesh = {'planning': program.identity['mesh'],
                  'sections': program.identity['sections']}
    branch_controls = {'seeds': core_input_layout['seeds']}
    identity = {'schema': 'scnsim.prepared_function', 'schema_version': 1,
                'program_sha256': program.program_sha256,
                'native_program_sha256': native.structural_identity,
                'active_indices': indices,
                'constant_bytes': constants.tobytes().hex(),
                'precision': precision}

    def numeric_function(theta_SI, omega_rad_per_s, seeds):
        import jax
        import jax.numpy as jnp
        with jax.enable_x64(True):
            full_theta = jnp.asarray(constants).at[jnp.asarray(indices, dtype=jnp.int32)].set(theta_SI)
            return function(full_theta, omega_rad_per_s, seeds)

    return PreparedFunction(
        numeric_function=numeric_function, input_layout=input_layout,
        output_layout=native.output_layout, aux_layout=native.aux_layout,
        structural_identity=sha256(canonical_json_bytes(identity)).hexdigest(),
        fixed_mesh=fixed_mesh, branch_controls=branch_controls,
        precision=precision, _compile_core=native.core,
    )
