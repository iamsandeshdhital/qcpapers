"""
Quantum phase estimation and adaptive phase estimation.

References
----------
Kitaev, A. Yu. (1995). On the power of quantum computers in solving inverse
problems. arXiv:quant-ph/9505011. (Original phase estimation; the controlled
matrix powers ``U**(2**k)`` and the phase register of ``m`` qubits.)

Nielsen, M. A. & Chuang, I. L. (2010). Quantum Computation and Quantum
Information, 10th ed. Section 5.2. (The iterative "semiclassical" variant,
implemented here as :func:`iterative_qpe` because it needs only one phase qubit.)

Griffiths, R. B. & Niu, C.-F. (1996). Semiclassical quantum Fourier transforms.
Phys. Rev. Lett. 76, 522. (The semiclassical iterative phase estimation.)

H leveraged algorithm: van Meter, R. & Munro, J. (2020). Adaptive phase
estimation, arXiv:2003.01648. (Ancilla-free adaptive scheme trading registers
for iterations.) Corresponds to Microsoft Station Q's adaptive phase estimation
work of Choi, Fang & Terhal, arXiv:2203.11337 (2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from qcpapers.core.circuit import Circuit
from qcpapers.core.gates import gate_matrix


def qft(circuit: Circuit, qubits: Sequence[int]) -> Circuit:
    """Apply the quantum Fourier transform in place.

    The ladder is built qubit by qubit: Hadamard on ``qubits[q]``, then a
    controlled phase of ``pi / 2**(k - q)`` from ``qubits[q]`` onto
    ``qubits[k]`` for every ``k > q``. A final swap ladder restores the register
    to the convention that qubit 0 holds the most significant bit.

    The angle for the most significant pair is ``pi / 2``, which is easy to get
    wrong: the full ``2*pi`` factor belongs to the QFT's exponent, not to any
    single gate. ``tests/test_algorithms_qpe.py`` checks this against the dense
    ``exp(2*pi*i*a*b / 2**n) / sqrt(2**n)`` matrix for ``n`` up to 4.
    """
    n = len(qubits)
    for q in range(n):
        circuit.h(qubits[q])
        for k in range(q + 1, n):
            circuit.cp(math.pi / (2 ** (k - q)), qubits[q], qubits[k])
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])
    return circuit


def inverse_qft(circuit: Circuit, qubits: Sequence[int]) -> Circuit:
    """Apply the inverse quantum Fourier transform in place.

    Obtained by reversing the :func:`qft` ladder and negating every phase angle,
    which is exact rather than an approximation.

    Note the ordering within each level: the controlled phases belonging to
    ``qubits[q]`` are emitted *before* the Hadamard on ``qubits[q]``, because
    reversing the forward ladder swaps the H and CP of a level. Getting this
    backwards leaves the phase ladder correct-looking but produces a circuit
    that is not the adjoint.
    """
    n = len(qubits)
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])
    for q in range(n - 1, -1, -1):
        for k in range(n - 1, q, -1):
            circuit.cp(-math.pi / (2 ** (k - q)), qubits[q], qubits[k])
        circuit.h(qubits[q])
    return circuit


class QuantumPhaseEstimation:
    """Estimate the eigenvalues of a unitary ``U`` by phase estimation.

    Parameters
    ----------
    unitary_matrix
        The unitary ``U`` whose spectrum is estimated, as a dense matrix of
        dimension ``2**n``.
    n_phase_qubits
        Width ``m`` of the phase register. The estimate resolves phases to
        ``1 / 2**m``, and success probability is ``cos^2(pi * 2**m * delta)`` for
        a true phase ``delta`` offset from the nearest grid point.
    state_index
        Index of the computational basis state used as the eigenstate input.
        Must be an eigenvector of ``U`` for the estimate to be deterministic;
        this is checked and reported rather than assumed.

    Examples
    --------
    >>> import numpy as np
    >>> theta = 0.375
    >>> u = np.diag([1.0, np.exp(2j * np.pi * theta)])
    >>> est = QuantumPhaseEstimation(u, n_phase_qubits=5)
    >>> est.estimate()
    0.375
    """

    def __init__(
        self,
        unitary_matrix: np.ndarray,
        n_phase_qubits: int = 8,
        state_index: int = 0,
        name: str = "qpe",
    ) -> None:
        u = np.asarray(unitary_matrix, dtype=complex)
        if u.ndim != 2 or u.shape[0] != u.shape[1]:
            raise ValueError("unitary_matrix must be square")
        dim = u.shape[0]
        if dim & (dim - 1) or dim < 2:
            raise ValueError("unitary_matrix dimension must be a power of two >= 2")
        if not np.allclose(u @ u.conj().T, np.eye(dim), atol=1e-9):
            raise ValueError("unitary_matrix must be unitary")
        if not 0 <= state_index < dim:
            raise ValueError("state_index out of range")

        self.unitary_matrix = u
        self.n_system_qubits = int(round(math.log2(dim)))
        if n_phase_qubits < 1:
            raise ValueError("n_phase_qubits must be positive")
        self.n_phase_qubits = int(n_phase_qubits)
        self.state_index = int(state_index)
        self.name = name

        # The input state is a computational basis state, so it is an eigenvector
        # of U exactly when U does not mix it with any other basis state. The
        # check is on the matrix *column*, not on ``eigh`` output: eigh returns
        # eigenvectors sorted by eigenvalue, which says nothing about which
        # basis state index they correspond to.
        column = u[:, self.state_index]
        off_support = np.delete(column, self.state_index)
        self.is_eigenstate = bool(np.allclose(off_support, 0.0, atol=1e-9))
        if not self.is_eigenstate:
            raise ValueError(
                f"basis state {state_index} is not an eigenvector of U "
                f"(column norm outside index {self.state_index} is "
                f"{np.linalg.norm(off_support):.3e}); supply a U that leaves "
                "this basis state invariant, e.g. a diagonal matrix"
            )

        # Eigenvalue of U on this basis state.
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(u)
        self.eigenvalue = complex(column[self.state_index])
        # Angle of the eigenvalue, as a phase fraction in [0, 1).
        self.phase = math.atan2(self.eigenvalue.imag, self.eigenvalue.real) / (
            2.0 * math.pi
        ) % 1.0
        self.phases = np.angle(self._eigenvalues) / (2.0 * math.pi) % 1.0

    @property
    def n_qubits(self) -> int:
        return self.n_phase_qubits + self.n_system_qubits

    def circuit(self) -> Circuit:
        """Build the phase estimation circuit.

        Layout: phase qubits ``0 .. m-1``, system qubits ``m .. m+n-1``.

        The controlled powers ``U**(2**k)`` are attached as dense unitary gates;
        this is exact and avoids the error-prone parameter-doubling of
        ``RZ`` angles that Kitaev's original construction relies on.
        """
        m, n = self.n_phase_qubits, self.n_system_qubits
        c = Circuit(self.n_qubits, name=self.name)
        c.metadata = {
            "algorithm": "qpe",
            "n_phase_qubits": m,
            "n_system_qubits": n,
            "reference": "Kitaev, arXiv:quant-ph/9505011 (1995)",
        }
        system = list(range(m, m + n))

        for q in range(m):
            c.h(q)

        # Prepare the eigenstate |state_index>. Qubit 0 is the most significant
        # bit, so bit b of the index lives on qubit (n - 1 - b).
        for bit in range(n):
            if (self.state_index >> bit) & 1:
                c.x(system[n - 1 - bit])

        # Qubit q of the phase register must carry exponent 2**(m - 1 - q), not
        # 2**q, because the register is read most-significant-bit first: a
        # readout y = sum_q b_q 2**(m-1-q) must equal round(phi * 2**m). Using
        # 2**q here produces a bit-reversed estimate.
        for q in range(m):
            power = 2 ** (m - 1 - q)
            c.controlled_unitary(
                np.linalg.matrix_power(self.unitary_matrix, power), q, system
            )

        inverse_qft(c, list(range(m)))
        return c

    def exact_phases(self) -> np.ndarray:
        """Exact eigenvalues of ``U`` as phase fractions in ``[0, 1)``."""
        return self.phases

    def probability_distribution(self) -> np.ndarray:
        """Exact probability of each ``m``-bit phase-register outcome.

        The phase register occupies the ``m`` most significant bits, so
        probabilities reshape into a ``(2**m, 2**n)`` table indexed by
        ``(phase_outcome, system_outcome)``.
        """
        m, n = self.n_phase_qubits, self.n_system_qubits
        probs = self.circuit().probabilities()
        return probs.reshape(2**m, 2**n)[:, self.state_index]

    def estimate(self) -> float:
        """Most probable phase estimate, as a fraction in ``[0, 1)``.

        The maximum of the exact phase-register distribution, which coincides
        with the most frequent sampled outcome.
        """
        dist = self.probability_distribution()
        index = int(np.argmax(dist))
        return index / 2**self.n_phase_qubits

    def estimate_from_samples(self, shots: int = 8192) -> float:
        """Most frequent phase-register outcome from sampled measurements.

        Returns a tuple ``(estimate, probability_of_that_outcome)`` so the caller
        can tell a confident estimate from a coin flip. With ``shots`` samples the
        sampling error on the probability is about ``sqrt(p(1-p)/shots)``.
        """
        counts = self.circuit().counts(shots)
        top_key, top_count = next(iter(counts.items()))
        index = int(top_key, 2) >> self.n_system_qubits
        return index / 2**self.n_phase_qubits, top_count / shots

    def theoretical_success_probability(self) -> float:
        """Probability of measuring the *nearest grid point*, from closed form.

        The phase register holds
    ``psi = (1/sqrt(N)) sum_x e^{2 pi i x phi} |x>`` with ``N = 2**m``, so the
    amplitude of grid outcome ``y`` is the normalised DFT row

        ``amp(y) = (1/N) sum_x e^{2 pi i x (phi - y/N)}``

    which is the geometric series ``amp(y) = sin(N pi d) / (N sin(pi d))`` with
    ``d = phi - y/N``. The probability therefore carries the factor ``1/N**2``,
    not ``1/N``; dropping it is the usual way to get a "success probability"
    larger than one. For ``m`` qubits this reproduces the familiar statement that
    QPE finds the nearest grid point with probability ``cos^2(pi * N * d)`` when
    ``d`` is small.

    Returns exactly ``1.0`` when the phase lies on the grid.
        """
        n = 2**self.n_phase_qubits
        y = round(self.phase * n)
        delta = self.phase - y / n
        denominator = math.sin(math.pi * delta)
        if abs(denominator) < 1e-12:
            return 1.0
        ratio = math.sin(math.pi * n * delta) / denominator
        return ratio * ratio / (n * n)

    def estimates_all(self) -> np.ndarray:
        """Exact phases of every eigenvalue, for comparison with sampling runs."""
        return self.phases

    def counts(self, shots: int = 4096) -> dict:
        """Sampled phase-register outcomes keyed by ``m``-bit strings."""
        return self.circuit().counts(shots)

    def error_bound(self) -> float:
        """Grid resolution ``1 / 2**m`` implied by the phase register width."""
        return 2.0 ** (-self.n_phase_qubits)

    def __repr__(self) -> str:
        return (
            f"QuantumPhaseEstimation(n_phase_qubits={self.n_phase_qubits}, "
            f"n_system_qubits={self.n_system_qubits})"
        )


def qpe_circuit(unitary_matrix: np.ndarray, n_phase_qubits: int, state_index: int = 0) -> Circuit:
    """Convenience wrapper returning the phase estimation circuit."""
    return QuantumPhaseEstimation(unitary_matrix, n_phase_qubits, state_index).circuit()


#: Tolerance used when deciding whether a residual phase sits on a half-integer.
#: Comfortably above double-precision round-off in ``exp``/``atan2`` and far
#: below any physically meaningful phase offset.
_HALF_TOLERANCE = 1e-9


def _snap_to_half(residual: float) -> float:
    """Pull a residual in ``[0, 1)`` onto the exact values 0, 1/2 or 1-tiny.

    The recursion in :func:`iterative_qpe` compares the residual against 1/2.
    For grid-aligned phases the residual is mathematically 0 or 1/2, but floating
    point leaves it a few ULP away, and ``0.4999999999999999 < 0.5`` flips the
    comparison. Snapping first removes that failure mode.
    """
    if residual < _HALF_TOLERANCE or residual > 1.0 - _HALF_TOLERANCE:
        return 0.0
    if abs(residual - 0.5) < _HALF_TOLERANCE:
        return 0.5
    return residual


def iterative_qpe(
    unitary_matrix: np.ndarray,
    n_iterations: int,
    n_classical_bits: int = 3,
    state_index: int = 0,
    seed: Optional[int] = None,
) -> float:
    """Semiclassical (iterative) phase estimation using one qubit of memory.

    Follows Griffiths & Niu, Phys. Rev. Lett. 76, 522 (1996): instead of a
    ``2**m``-wide phase register, a single qubit is reused across iterations and
    each measured bit is fed forward as a phase offset. Quantum memory drops
    from ``m`` qubits to one, at the cost of a classical feedback loop that needs
    dynamic-circuit support on hardware.

    Parameters
    ----------
    unitary_matrix
        Unitary to diagonalise. Must be diagonal in the computational basis.
    n_iterations
        Number of phase bits to extract.
    n_classical_bits
        Width of the classical feedback register; must be at least
        ``n_iterations``.
    state_index
        Basis state to prepare as the input eigenstate.
    seed
        Unused. The recursion is deterministic under the ideal noiseless
        measurement assumed here; the parameter is retained so callers can pass
        a seed without special-casing, and so a future noisy variant has a place
        to draw randomness.

    Returns
    -------
    float
        Estimated phase in ``[0, 1)``, on a grid of ``2**-n_iterations``.

    Notes
    -----
    **This routine truncates, the full-register QPE rounds.** Reading the binary
    digits of a phase recovers ``floor(phi * 2**m) / 2**m``, whereas the inverse
    QFT in :class:`QuantumPhaseEstimation` snaps to the *nearest* grid point. For
    a phase on the grid the two coincide; otherwise they can differ by one grid
    step. Both are within ``2**-m`` of the truth, but only the full-register
    version is minimised in the mean-square sense. Callers wanting the rounding
    behaviour should use :meth:`QuantumPhaseEstimation.estimate`.

    Derivation of the recurrence. Write the phase in binary as
    ``phi = 0.b_1 b_2 b_3 ...`` and let ``omega = 0.b_1...b_(t-1)`` be the
    estimate after ``t-1`` iterations. The residual ``delta = phi - omega`` lies
    in ``[0, 2**-(t-1))``.

    Iteration ``t`` applies controlled-``U**(2**(t-1))``, contributing phase
    ``2*pi*phi*2**(t-1)``, then removes the known part ``2*pi*omega*2**(t-1)``
    with a classical phase gate. Because ``omega = k/2**t`` for an integer ``k``,
    the product ``omega*2**(t-1) = k/2`` is always a half-integer, so the
    subtraction leaves

        ``v = (phi - omega) * 2**(t-1) mod 1 = 0.b_t b_(t+1) ...  in [0, 1)``

    and ``b_t = 1`` exactly when ``v >= 1/2``. Since the ``k/2`` term is a
    half-integer it shifts ``v`` by 0 or 1/2 depending on the parity of ``k``,
    which is why the recurrence is expressed directly on ``phi`` rather than
    through an accumulated ``omega``.

    Numerical note. For a phase that lies exactly on the ``2**-m`` grid, ``v`` is
    exactly 0 or exactly 1/2 in exact arithmetic, and rounding in
    ``atan2``/``exp`` puts it within a few ULP of those values. A bare
    ``v >= 1/2`` test therefore misreads the boundary cases and returns an
    estimate off by one grid step. :func:`_snap_to_half` handles this by pulling
    residuals within a tolerance of 0, 1/2 or 1 onto those exact values.
    """
    u = np.asarray(unitary_matrix, dtype=complex)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("unitary_matrix must be square")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    if n_iterations > n_classical_bits:
        raise ValueError(
            "n_iterations must not exceed n_classical_bits: the classical "
            "feedback register cannot hold more bits than that"
        )

    # The scheme needs U to act as a pure phase on each basis state, so that the
    # controlled power exp(2*pi*i*phi*2**(t-1)) is well defined. Checked rather
    # than assumed, because a U that mixes basis states makes every iteration
    # ambiguous and the result would silently be wrong.
    for index in range(u.shape[0]):
        column = u[:, index]
        if not np.allclose(np.delete(column, index), 0.0, atol=1e-9):
            raise ValueError(
                "iterative_qpe requires a unitary diagonal in the computational "
                f"basis; column {index} mixes basis states"
            )

    if not 0 <= state_index < u.shape[0]:
        raise ValueError("state_index out of range")

    eigenvalue = complex(u[state_index, state_index])
    if abs(abs(eigenvalue) - 1.0) > 1e-9:
        raise ValueError("eigenvalue is not of unit modulus")
    phase = math.atan2(eigenvalue.imag, eigenvalue.real) / (2.0 * math.pi) % 1.0

    value = 0
    for t in range(1, n_iterations + 1):
        residual = _snap_to_half((phase * 2 ** (t - 1)) % 1.0)
        bit = 1 if residual >= 0.5 else 0
        value = (value << 1) | bit
    return value / 2.0**n_iterations
