"""
qcpapers - reproducible implementations of landmark quantum computing papers.

Every module in this package maps to one or more published papers and carries
the citation in its docstring so results can be traced back to the source.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "iamsandeshdhital"
__email__ = "mesandeshdhital@gmail.com"

from qcpapers.core.circuit import Circuit
from qcpapers.core.gates import apply_gate, controlled_gate, gate_matrix

__all__ = ["Circuit", "apply_gate", "controlled_gate", "gate_matrix", "__version__"]
