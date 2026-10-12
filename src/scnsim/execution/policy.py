"""Owner-local scheduling hints; never scientific eligibility or request identity."""
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Execution resolves unknown scratch estimates conservatively, without remeshing."""
    frequency_chunk_size: int = 8
    max_in_flight: int | None = None
    scratch_budget_bytes: int = 256 * 1024 * 1024
    compile_entries: int = 16
    compile_variants: int = 16
    compile_known_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        for name in ("frequency_chunk_size", "scratch_budget_bytes", "compile_entries", "compile_variants", "compile_known_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_in_flight is not None and (type(self.max_in_flight) is not int or self.max_in_flight <= 0):
            raise ValueError("max_in_flight must be None or a positive integer")
