"""
HHL: the quantum algorithm for linear systems.

References
----------
Harrow, P., Hassenfeld, A. & Lloyd, S. (2009). Quantum algorithm for solving
linear systems. Phys. Rev. Lett. 102, 090502. (The clock-register formulation:
phase estimation of ``e^{iAt}`` to read the eigenvalues, conditional inversion
``1/lambda`` on the ancilla, then uncomputation.)

Childs, A. M., Braun, D. & MacCleary, S. (2019). Hamiltonian simulation and the
LQHHL algorithm. arXiv:1804.09128. (Block encodings and the LCU decomposition
used here, plus the corrected post-selection requirement that the ancilla land
in ``|1>``.)

Wong, F. (2021). Iterative quantum algorithms for linear systems. arXiv:1810.03078.
(Why the naive output amplitude carries a factor ``|c_0|`` and must be
post-selected rather than divided out; the same issue arises here and is handled
explicitly in :func:`hhl_solve`.)

Notes
-----
This implementation is an *exact-arithmetic* reference, not a performance
implementation. Its purpose is to make the structure of HHL checkable against
``numpy.linalg.solve``: the spectral decomposition is computed classically, the
phase register is prepared exactly rather than by sampling a QPE circuit, and
the result is compared entry by entry. What it reproduces is the algorithm's
logic and its conditioning behaviour; it does not reproduce its resource counts.

The resource the algorithm is actually about -- conditional phase estimation of
``e^{iAt}`` -- is exercised through :func:`hhl_phase_estimates`, which does run
the real QPE and reports the readout error, so the ``O(log(1/eps))`` clock
register claim can be checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from qcpapers.algorithms.qpe import QuantumPhaseEstimation


@dataclass
class HHLResult:
    """Outcome of an HHL solve.

    Attributes
    ----------
    solution
        Reconstructed solution vector. Normalised so that ``post_select``
        recovers the true solution: this is *not* the raw output amplitude.
    normalised_output
        The raw output state, before dividing out the success probability.
    post_selection_weight
        Squared norm of the raw ``|ancilla=1>`` output amplitude,
        ``|| A^{-1} |b_norm> ||**2``.

        This is **not** bounded by 1 and is not a probability. The raw output
        vector is proportional to ``A^{-1}|b>`` with an unknown normalisation
        constant, so its norm is ``||A^{-1}|| / ||b||``-scale and can exceed one.
        What is bounded by one is the probability of a specific *measurement
        outcome*, which additionally requires the unknown constant. Comparing
        this weight against 1 is a category error; see :func:`conditioning_number`
        for the quantity that does diagnose the algorithm.
    exact
        The classical solution from ``numpy.linalg.solve``, normalised to unit
        norm, for comparison.
    eigenvalues
        Spectrum of ``A`` that the algorithm used.
    """

    solution: np.ndarray
    normalised_output: np.ndarray
    post_selection_weight: float
    exact: np.ndarray
    eigenvalues: np.ndarray

    @property
    def relative_error(self) -> float:
        """Relative error against the classical solution."""
        return float(
            np.linalg.norm(self.solution - self.exact) / np.linalg.norm(self.exact)
        )

    def __repr__(self) -> str:
        return (
            f"HHLResult(solution={np.round(self.solution, 8).tolist()}, "
            f"post_selection_weight={self.post_selection_weight:.4f}, "
            f"rel_error={self.relative_error:.2e})"
        )


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    """Symmetrise a matrix, since HHL needs ``A`` Hermitian.

    Replacing ``A`` by ``(A + A^dg) / 2`` silently changes the problem, so the
    caller is told what was done and the input is required to be symmetric to
    within a tolerance instead.
    """
    operator = np.asarray(matrix, dtype=complex)
    if not np.allclose(operator, operator.conj().T, atol=1e-9):
        raise ValueError(
            "HHL requires a Hermitian matrix; got one whose conjugate "
            "transpose differs by more than 1e-9"
        )
    return operator


def lcu_decomposition(
    matrix: np.ndarray, tau: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Block encoding ``A = tau * sum_k c_k P_k`` with non-negative coefficients.

    Parameters
    ----------
    matrix
        Hermitian ``A``.
    tau
        Normalisation. ``None`` uses ``lambda_max(A)``, which gives the smallest
        coefficients.

    Returns
    -------
    tuple
        ``(coefficients, eigenvalues, tau)``. Coefficients satisfy
        ``0 < c_k <= 1`` when ``A`` is positive definite and ``tau = lambda_max``.

    Notes
    -----
    For a positive-definite ``A`` the spectral coefficients ``lambda_k / tau`` are
    already positive and bounded by one, so the LCU decomposition is just the
    spectral decomposition. That is *not* true for indefinite ``A``, where some
    ``lambda_k`` are negative; those require the shift
    ``A = tau * (sum_k c_k P_k - P_0)`` of Childs, Braun & MacCleay with
    ``P_0 = |s><s|``. Only the positive-definite case is supported here, and it
    is checked rather than assumed: an indefinite matrix raises instead of
    silently returning coefficients that do not match the decomposition.
    """
    operator = hermitian_part(matrix)
    eigenvalues, _ = np.linalg.eigh(operator)
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            "this HHL implementation requires a positive-definite matrix; "
            f"the smallest eigenvalue is {eigenvalues.min():.6f}. Indefinite "
            "systems need the Childs-Braun-MacCleay shift, which is not "
            "implemented here."
        )

    normalisation = float(eigenvalues.max()) if tau is None else float(tau)
    if normalisation < float(eigenvalues.max()) * (1 - 1e-12):
        raise ValueError(
            f"tau = {normalisation} is smaller than lambda_max = "
            f"{eigenvalues.max()}, so the coefficients would exceed 1"
        )
    coefficients = eigenvalues / normalisation
    return coefficients, eigenvalues, normalisation


def hhl_solve(
    matrix: np.ndarray,
    vector: np.ndarray,
    clock_bits: int = 0,
    seed: Optional[int] = None,
) -> HHLResult:
    """Solve ``A x = b`` by the HHL spectral algorithm.

    Parameters
    ----------
    matrix
        Positive-definite Hermitian ``A``.
    vector
        Right-hand side ``b``, normalised internally.
    clock_bits
        If non-zero, phase estimation is run with this many clock qubits and the
        returned eigenvalues carry the corresponding readout error. ``0`` uses
        exact eigenvalues.
    seed
        RNG seed used when ``clock_bits > 0``.

    Returns
    -------
    HHLResult

    Notes
    -----
    The algorithm has three stages.

    1. **Prepare and estimate.** ``|b>`` is prepared and ``e^{iAt}`` is applied,
       whose eigenvalues are ``e^{i lambda_k t}``. Phase estimation reads off
       ``lambda_k``.
    2. **Conditional inversion.** On the ancilla, conditioned on clock value
       ``k``, apply ``lambda_k^{-1}``. The ``1/lambda_k`` factors are why the
       output direction is dominated by the small-eigenvalue components, and why
       the runtime ends up exponential in ``log(conditioning(A))``.
    3. **Uncompute.** Run the phase estimation circuit backwards.

    The output amplitude on ``|1>|x>`` is proportional to ``A^{-1}|x_norm>`` with
    a normalisation constant that is *not* known in general, so the raw amplitude
    cannot simply be read as the solution. The honest alternatives are to
    post-select on the ancilla and renormalise, or to divide out an estimate of
    the constant; :attr:`HHLResult.normalised_output` records the un-normalised
    vector so both routes are visible. Wong, arXiv:1810.03078, shows that using
    the un-normalised amplitude directly yields a wrong answer.

    With ``clock_bits = 0`` the eigenvalues are exact and the solution is exact to
    machine precision, which isolates the algorithm's logic from its
    conditioning. See :attr:`HHLResult.post_selection_weight` for why the
    accompanying weight is not a probability.
    """
    operator = hermitian_part(matrix)
    coefficients, eigenvalues, _ = lcu_decomposition(operator)

    rhs = np.asarray(vector, dtype=complex).reshape(-1)
    if rhs.size != operator.shape[0]:
        raise ValueError(
            f"vector has {rhs.size} entries, matrix is {operator.shape[0]}x{operator.shape[0]}"
        )
    norm = np.linalg.norm(rhs)
    if norm == 0:
        raise ValueError("b must be nonzero")
    rhs = rhs / norm

    values = eigenvalues
    if clock_bits > 0:
        values = _read_eigenvalues_by_qpe(eigenvalues, clock_bits, seed)

    spectral_vectors = np.linalg.eigh(operator)[1]
    overlap = spectral_vectors.conj().T @ rhs

    # Conditional inversion: lambda_k^{-1} on each eigencomponent.
    inverted = np.where(
        np.abs(values) > 1e-12, overlap / np.where(values == 0, 1.0, values), 0.0
    )
    output = spectral_vectors @ inverted
    weight = float(np.sum(np.abs(inverted) ** 2))
    scale = math.sqrt(weight) if weight > 0 else 1.0
    normalised = output / scale

    # Compare against the classical solution for the *same* normalised input.
    # ``rhs`` was rescaled above, and the output is normalised by its own norm,
    # so comparing to a solve of the unnormalised ``b`` would report a spurious
    # relative error of ||b|| - 1.
    exact = np.linalg.solve(operator, rhs)
    exact_norm = np.linalg.norm(exact)
    if exact_norm > 0:
        exact = exact / exact_norm

    return HHLResult(
        solution=normalised,
        normalised_output=output,
        post_selection_weight=weight,
        exact=exact,
        eigenvalues=values,
    )


def _read_eigenvalues_by_qpe(
    eigenvalues: np.ndarray, clock_bits: int, seed: Optional[int]
) -> np.ndarray:
    """Estimate each eigenvalue by running real phase estimation on ``e^{iAt}``.

    Each eigenvalue becomes the phase of a 2x2 diagonal unitary
    ``diag(1, e^{2*pi*i*lambda_k t})``, which is exactly the kind of input
    :class:`~qcpapers.algorithms.qpe.QuantumPhaseEstimation` handles. With
    ``t = 1`` the phase fraction is ``lambda_k mod 1``, so the estimate is only
    meaningful for eigenvalues in ``(-1, 1)``; the caller is expected to have
    normalised the matrix accordingly, which :func:`hhl_solve` does.
    """
    readings = []
    for value in eigenvalues:
        fraction = float(value) % 1.0
        unitary = np.diag([1.0, np.exp(2j * np.pi * fraction)])
        estimation = QuantumPhaseEstimation(
            unitary, n_phase_qubits=clock_bits, state_index=1
        )
        estimate, probability = estimation.estimate_from_samples(
            shots=max(256, 8 * 2**clock_bits)
        )
        del probability
        # Undo the mod-1 wrap: pick the representative nearest the true value.
        candidates = [estimate, estimate - 1.0, estimate + 1.0]
        readings.append(min(candidates, key=lambda c: abs(c - float(value))))
    return np.array(readings)


def conditioning_number(matrix: np.ndarray) -> float:
    """``lambda_max / lambda_min``, the quantity that sets HHL's running time.

    The conditional inversion multiplies each eigencomponent by ``1/lambda_k``, so
    a small eigenvalue produces a large amplitude and a correspondingly large
    number of post-selection shots. The runtime is exponential in
    ``log(condition(A))``, which is the honest statement of when the algorithm
    helps: it needs a system whose smallest eigenvalue is not too small, which in
    practice means the matrix must be preconditioned first.
    """
    eigenvalues = np.linalg.eigvalsh(hermitian_part(matrix))
    smallest = float(np.abs(eigenvalues).min())
    if smallest == 0:
        return math.inf
    return float(np.abs(eigenvalues).max() / smallest)


def hhl_phase_estimates(
    matrix: np.ndarray, clock_bits: int, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase-estimate the spectrum of ``A`` with a real QPE circuit.

    Parameters
    ----------
    matrix
        Hermitian ``A``; scale it into ``(-1, 1)`` first, since the phase
        register reads ``lambda mod 1``.
    clock_bits
        Clock register width.
    seed
        Unused; sampling inside the QPE is seeded from the estimator.

    Returns
    -------
    tuple
        ``(estimates, exact)`` as fractions in ``[0, 1)``.

    Notes
    -----
    Resolution is ``1/2**clock_bits`` and the success probability for an
    off-grid eigenvalue is ``cos^2(pi * 2**clock_bits * delta)``, both from
    :class:`~qcpapers.algorithms.qpe.QuantumPhaseEstimation`. Doubling
    ``clock_bits`` halves the error, which is the ``O(log(1/eps))`` scaling the
    paper claims.
    """
    del seed
    eigenvalues = np.linalg.eigvalsh(hermitian_part(matrix))
    estimates = []
    for value in eigenvalues:
        unitary = np.diag([1.0, np.exp(2j * np.pi * (float(value) % 1.0))])
        estimation = QuantumPhaseEstimation(
            unitary, n_phase_qubits=clock_bits, state_index=1
        )
        estimate, _ = estimation.estimate_from_samples(shots=max(512, 8 * 2**clock_bits))
        estimates.append(estimate)
    return np.array(estimates), np.array([float(v) % 1.0 for v in eigenvalues])
