"""
Quantum algorithm implementations.

Every module in this package reproduces the *specific* algorithm of a
published paper and cites it. Where a paper reports experimental parameters,
the paper's values are the defaults.
"""

from __future__ import annotations

from qcpapers.algorithms.grover import (
    GroverSearch,
    amplitude_amplification,
    grover_search,
    optimal_iteration_count,
)
from qcpapers.algorithms.classical_shadows import (
    ClassicalShadow,
    ShadowResult,
    Snapshot,
    take_snapshot,
    take_snapshots,
)
from qcpapers.algorithms.hhl import (
    HHLResult,
    conditioning_number,
    hhl_phase_estimates,
    hhl_solve,
    lcu_decomposition,
)
from qcpapers.algorithms.qaoa import (
    QAOA,
    QAOAResult,
    exact_maxcut,
    grover_walk_search,
    maxcut_cost_operator,
    maxcut_qaoa,
    qaoa_parameter_scan,
)
from qcpapers.algorithms.qpe import (
    QuantumPhaseEstimation,
    inverse_qft,
    iterative_qpe,
    qft,
    qpe_circuit,
)
from qcpapers.algorithms.vqe import (
    EfficientSU2Ansatz,
    ParticleNumberAnsatz,
    VQE,
    VQEResult,
    optimise_energy,
    pairwise_ising_hamiltonian,
)

__all__ = [
    # Grover and amplitude amplification
    "GroverSearch",
    "grover_search",
    "optimal_iteration_count",
    "amplitude_amplification",
    # Phase estimation
    "QuantumPhaseEstimation",
    "qpe_circuit",
    "qft",
    "inverse_qft",
    "iterative_qpe",
    # Variational eigensolver
    "VQE",
    "VQEResult",
    "EfficientSU2Ansatz",
    "ParticleNumberAnsatz",
    "optimise_energy",
    "pairwise_ising_hamiltonian",
    # QAOA
    "QAOA",
    "QAOAResult",
    "maxcut_cost_operator",
    "maxcut_qaoa",
    "exact_maxcut",
    "qaoa_parameter_scan",
    "grover_walk_search",
    # Classical shadows
    "ClassicalShadow",
    "ShadowResult",
    "Snapshot",
    "take_snapshot",
    "take_snapshots",
    # HHL linear systems
    "HHLResult",
    "hhl_solve",
    "lcu_decomposition",
    "conditioning_number",
    "hhl_phase_estimates",
]
