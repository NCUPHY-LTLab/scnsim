"""Call-local compiled numerical kernels; no factor, device or result ownership.

The installed platform extension is required by compensated host equations.
Build/install and public runtime preparation own availability and its errors.
"""


def require_compensated_kernel():
    """Prepare the required platform kernel before native equation callbacks.

    Authoring imports do not load this extension. Missing/incompatible wheel
    availability is a runtime preparation error retaining its native cause.
    """
    try:
        from ._compensated_cpu import csr_pair_product
    except ImportError as error:
        from ..errors import RuntimePreparationError
        raise RuntimePreparationError(
            "The installed SCNSim platform wheel lacks its compensated CPU kernel",
            stage="runtime_prepare",
        ) from error
    return csr_pair_product
