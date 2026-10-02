"""Core primitives: gate library and statevector circuit simulator."""

from __future__ import annotations

from qcpapers.core.circuit import Circuit, StateVectorResult
from qcpapers.core.gates import (
    ALL_GATES,
    gate_matrix,
    is_clifford,
)

__all__ = [
    "Circuit",
    "StateVectorResult",
    "gate_matrix",
    "ALL_GATES",
    "is_clifford",
]
