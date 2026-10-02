"""
Classical shadows: randomized Pauli measurements for quantum state tomography.

References
----------
Huang, H., Harrow, A. W. & Low, G. H. (2020). Classical shadows for quantum
states. arXiv:2006.16719. (The shadow protocol: measure single-qubit Paulis
independently, store the ``+1``/``-1`` outcomes as bits, and reconstruct each
Pauli expectation through the single-qubit Pauli channel.)

Aaronson, S. (2018). Shadow tomography: sparse encodings of complex states.
arXiv:1804.08326. (Independent shadow protocols, and the median-of-means
estimator for the density matrix.)

Chen, H., Harrow, A. W. & Low, G. H. (2021). Error bounds for learning a quantum
state via the classical shadow method. arXiv:2009.07472. (The ``34 K / eps**2``
sample bound, and the statement that the requirement does not grow with the
number of qubits, unlike fidelity-based estimation.)

Notes
-----
A snapshot must record *which* Pauli was measured on each qubit, not just the
outcome. Dropping the basis makes the estimator undefined: the same ``+1`` means
different things depending on whether X, Y or Z was measured, and averaging over
all three cancels the signal. The :class:`Snapshot` type here keeps them together
so that mistake is not available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qcpapers.core.circuit import Circuit
from qcpapers.core.observables import SparsePauliOp, pauli_labels, pauli_matrix

#: Measurement basis order used throughout: index 0 -> Z, 1 -> X, 2 -> Y.
#: The order is arbitrary but must be consistent between sampling and
#: reconstruction.
PAULI_ORDER = ("Z", "X", "Y")

#: Overall factor in the Pauli-basis density-matrix reconstruction
#: ``rho = 2**-n sum_P <P> P``, from the Hilbert-Schmidt inner product
#: ``Tr(P Q) = 2**n delta_{PQ}``.
_PAULI_RECONSTRUCTION_SCALE = 0.5


def snapshot_circuit(n_qubits: int, basis: Sequence[int]) -> Circuit:
    """Circuit measuring one Pauli per qubit.

    Parameters
    ----------
    n_qubits
        Register width.
    basis
        One entry per qubit: ``0`` for Z, ``1`` for X, ``2`` for Y.

    Returns
    -------
    Circuit
        Circuit ending in the rotations that map each chosen axis onto the
        computational basis.

    Notes
    -----
    Basis 0 (Z) needs no rotation. X needs a Hadamard. Y needs ``Sdg`` followed
    by a Hadamard, so that reading Z gives ``S X S^dg = Y``. Reversing that pair
    yields X, which is indistinguishable from the X branch and silently measures
    the wrong axis.
    """
    if len(basis) != n_qubits:
        raise ValueError(f"basis must have {n_qubits} entries, got {len(basis)}")
    circuit = Circuit(n_qubits, name="shadow_snapshot")
    for qubit, choice in enumerate(basis):
        if choice == 0:
            continue
        if choice == 1:
            circuit.h(qubit)
        elif choice == 2:
            # Sdg first, then H, so U = H Sdg and U^dg Z U = S X S^dg = Y.
            #
            # The order is load-bearing and the failure is silent. Applying H
            # first and S(Sdg) second gives U = S H, for which
            # U^dg Z U = H Z H = X, which is exactly the X branch above: the
            # estimator still returns a number in [-1, 1] and still converges,
            # just to the wrong axis.
            circuit.sdg(qubit)
            circuit.h(qubit)
        else:
            raise ValueError(f"basis entry must be 0, 1 or 2, got {choice}")
    return circuit


@dataclass(frozen=True)
class Snapshot:
    """One round of single-qubit Pauli measurements.

    Attributes
    ----------
    basis
        Per-qubit measurement basis, as indices into :data:`PAULI_ORDER`.
    outcomes
        Per-qubit measurement outcome, ``+1`` or ``-1``.
    """

    basis: Tuple[int, ...]
    outcomes: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.basis) != len(self.outcomes):
            raise ValueError("basis and outcomes must have equal length")
        for value in self.outcomes:
            if value not in (1, -1):
                raise ValueError(f"outcomes must be +1 or -1, got {value}")

    @property
    def n_qubits(self) -> int:
        return len(self.basis)

    def pauli_string(self) -> str:
        """The measured Pauli string, e.g. ``"ZXI"``."""
        return "".join(PAULI_ORDER[b] for b in self.basis)

    def matched(self, qubit: int, axis_index: int) -> bool:
        """Whether this snapshot measured ``PAULI_ORDER[axis_index]`` on ``qubit``."""
        return self.basis[qubit] == axis_index

    def outcome(self, qubit: int, axis_index: int) -> float:
        """The outcome, or ``0`` when the basis does not match.

        Returning zero for a mismatch is what lets the mean over *all* snapshots
        be taken without filtering: a non-matching snapshot contributes no signal
        rather than a wrong one.
        """
        if not self.matched(qubit, axis_index):
            return 0.0
        return float(self.outcomes[qubit])


def take_snapshot(
    state: np.ndarray, n_qubits: int, rng: np.random.Generator
) -> Snapshot:
    """Draw one random snapshot from a state."""
    basis = tuple(int(b) for b in rng.integers(0, 3, size=n_qubits))
    circuit = snapshot_circuit(n_qubits, basis)
    tensor = circuit.state_tensor(np.asarray(state, dtype=complex).reshape((2,) * n_qubits))
    probabilities = np.abs(tensor.reshape(-1)) ** 2
    probabilities = probabilities / probabilities.sum()
    index = int(rng.choice(probabilities.size, p=probabilities))
    outcomes = tuple(1 - 2 * ((index >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits))
    return Snapshot(basis=basis, outcomes=outcomes)


def take_snapshots(
    state: np.ndarray,
    n_qubits: int,
    num_snapshots: int,
    seed: Optional[int] = None,
) -> List[Snapshot]:
    """Draw many snapshots from one state."""
    if num_snapshots < 1:
        raise ValueError("num_snapshots must be positive")
    rng = np.random.default_rng(seed)
    return [take_snapshot(state, n_qubits, rng) for _ in range(num_snapshots)]


def bloch_component(
    snapshots: Sequence[Snapshot], qubit: int, axis_index: int
) -> float:
    """Reconstruct one component of the single-qubit Bloch vector.

    Parameters
    ----------
    snapshots
        Measured snapshots.
    qubit
        Which qubit's Bloch vector to reconstruct.
    axis_index
        Index into :data:`PAULI_ORDER`.

    Returns
    -------
    float
        Estimate of ``<PAULI_ORDER[axis_index]>`` on that qubit.

    Notes
    -----
    A measurement of ``axis`` on a state ``rho`` with Bloch vector ``r`` returns
    ``+1`` with probability ``(1 + r_axis) / 2``, so the sample mean of ``+1``/``-1``
    outcomes estimates ``r_axis`` with no rescaling at all. This is precisely the
    claim that the single-qubit Pauli channel
    ``E(P) = (1/3)(X P X + Y P Y + Z P Z)`` acts as the identity on single-qubit
    Paulis, which is why shadows need no inversion step per qubit.

    The estimator is unbiased: a measurement of ``axis`` returns ``+1`` with
    probability ``(1 + r_axis) / 2``, so the sample mean over matching snapshots
    has expectation ``r_axis`` and variance ``(1 - r_axis**2) / N_matched``.

    The mean is taken over *matching* snapshots only, not over all of them. Since
    each axis is measured on a given qubit in about a third of snapshots,
    averaging over everything would divide the signal by three and report a Bloch
    component three times too small, with no visible error. Returns ``0.0`` when
    no snapshot matched.
    """
    if not snapshots:
        raise ValueError("need at least one snapshot")
    matched = [s for s in snapshots if s.matched(qubit, axis_index)]
    if not matched:
        return 0.0
    return sum(s.outcomes[qubit] for s in matched) / len(matched)


def dense_pauli(pauli: str) -> np.ndarray:
    """Kronecker product of a Pauli string, left to right."""
    matrix = np.array([[1.0]], dtype=complex)
    for factor in pauli:
        matrix = np.kron(matrix, pauli_matrix(factor))
    return matrix


@dataclass
class ShadowResult:
    """A reconstructed shadow.

    Attributes
    ----------
    n_qubits
        Register width.
    num_snapshots
        Number of snapshots the estimate used.
    expectations
        Estimated ``<P>`` for every one of the ``4**n`` Pauli strings, ordered by
        :func:`~qcpapers.core.observables.pauli_labels`.
    shadow_norm_bound
        Shadow norm bound used for the sample requirement.
    """

    n_qubits: int
    num_snapshots: int
    expectations: np.ndarray
    shadow_norm_bound: float

    def expectation(self, pauli: str) -> float:
        """Estimated expectation of one Pauli string."""
        pauli = pauli.upper()
        if len(pauli) != self.n_qubits:
            raise ValueError(
                f"Pauli string has {len(pauli)} factors, expected {self.n_qubits}"
            )
        return float(self.expectations[pauli_labels(self.n_qubits).index(pauli)])

    def errors(self, rho: np.ndarray) -> np.ndarray:
        """Absolute error against the exact expectations of a density matrix."""
        exact = self._exact_expectations(rho)
        return np.abs(self.expectations - exact)

    def max_error(self, rho: np.ndarray) -> float:
        """Largest absolute error over all Pauli expectations."""
        return float(self.errors(rho).max())

    def to_sparse_pauli_op(self, tolerance: float = 1e-6) -> SparsePauliOp:
        """The shadow as a :class:`SparsePauliOp`, dropping negligible terms.

        This is what makes the encoding *sparse*: a generic state's shadow has
        ``O(3**n)`` significant terms rather than the ``4**n`` a dense
        tomography would need.
        """
        terms = [
            (float(value), pauli)
            for value, pauli in zip(self.expectations, pauli_labels(self.n_qubits))
            if abs(value) > tolerance
        ]
        if not terms:
            terms = [(0.0, pauli_labels(self.n_qubits)[0])]
        return SparsePauliOp(terms)

    def to_matrix(self) -> np.ndarray:
        """Dense reconstruction ``2**-n sum_P <P> P``.

        The Pauli basis is orthogonal with ``Tr(P Q) = 2**n delta_{PQ}``, so the
        exact expansion of a state is ``rho = 2**-n sum_P Tr(P rho) P`` and
        substituting the shadow estimates gives the reconstruction. A product
        state is recovered exactly up to sampling error.

        For an entangled state the estimated ``<P>`` values are each unbiased, so
        the reconstruction is unbiased too, but the *variance* of the fidelity is
        much larger than for a product state. That gap between cheap Pauli
        expectations and expensive fidelity is the trade-off Huang, Harrow and Low
        make explicit, and it is why :attr:`expectations` is the quantity to trust
        rather than :meth:`overlap` for large systems.
        """
        matrix = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for value, pauli in zip(self.expectations, pauli_labels(self.n_qubits)):
            matrix += value * dense_pauli(pauli)
        return _PAULI_RECONSTRUCTION_SCALE**self.n_qubits * matrix

    def overlap(self, rho: np.ndarray) -> float:
        """``<Tr(rho_hat Tr(rho_ref rho_hat))>``, a fidelity-family score.

        Equals 1 for a perfect reconstruction. Unlike fidelity it is not bounded
        above by 1, so it is a diagnostic rather than a fidelity.
        """
        estimate = self.to_matrix()
        reference = np.asarray(rho, dtype=complex)
        return float(np.real(np.trace(estimate @ (reference @ estimate))))

    def _exact_expectations(self, rho: np.ndarray) -> np.ndarray:
        matrix = np.asarray(rho, dtype=complex)
        values = np.empty(4**self.n_qubits, dtype=float)
        for index, label in enumerate(pauli_labels(self.n_qubits)):
            values[index] = float(np.real(np.trace(dense_pauli(label) @ matrix)))
        return values


class ClassicalShadow:
    """Classical shadow reconstruction of an ``n``-qubit state.

    Parameters
    ----------
    n_qubits
        Register width.
    shadow_norm
        Override for the shadow norm bound; ``None`` uses ``3``, the value for
        single-qubit Paulis.
    seed
        RNG seed, so reconstructions are reproducible.

    References
    ----------
    Chen, Harrow & Low (2021), arXiv:2009.07472: ``N >= 34 K / eps**2`` samples
    suffice for every Pauli expectation to be within ``eps``. Crucially ``K``
    does not grow with ``n``, which is why shadows need far fewer measurements
    than direct tomography when only a few observables matter.
    """

    def __init__(
        self,
        n_qubits: int,
        shadow_norm: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits
        self.shadow_norm = 3.0 if shadow_norm is None else float(shadow_norm)
        self._rng = np.random.default_rng(seed)

    def required_snapshots(
        self, epsilon: float, shadow_norm: Optional[float] = None
    ) -> int:
        """Snapshot count guaranteeing ``eps`` accuracy on every Pauli expectation.

        Parameters
        ----------
        epsilon
            Target accuracy.
        shadow_norm
            Override the shadow norm for this call.

        Returns
        -------
        int
            ``ceil(34 K / eps**2)``.
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must lie in (0, 1)")
        norm = self.shadow_norm if shadow_norm is None else float(shadow_norm)
        return math.ceil(34.0 * norm / epsilon**2)

    def total_measurements(self, num_snapshots: int) -> int:
        """Number of single-qubit measurements across all snapshots."""
        return num_snapshots * self.n_qubits

    def reconstruct(
        self,
        state: np.ndarray,
        num_snapshots: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> ShadowResult:
        """Estimate every Pauli expectation from snapshots of ``state``.

        Parameters
        ----------
        state
            State as a flat ``2**n`` vector or a ``(2,)*n`` tensor.
        num_snapshots
            Number of snapshots to draw.
        epsilon
            Supply this instead of ``num_snapshots`` to pick the count from the
            shadow-norm bound.

        Returns
        -------
        ShadowResult
        """
        if num_snapshots is None:
            if epsilon is None:
                raise ValueError("supply num_snapshots or epsilon")
            num_snapshots = self.required_snapshots(epsilon)
        if num_snapshots < 1:
            raise ValueError("num_snapshots must be positive")

        flat = np.asarray(state, dtype=complex).reshape(-1)
        if flat.size != 2**self.n_qubits:
            raise ValueError(
                f"state has {flat.size} amplitudes, expected {2 ** self.n_qubits}"
            )
        if not np.isclose(np.linalg.norm(flat), 1.0, atol=1e-8):
            raise ValueError("state must be normalised")

        snapshots = [
            take_snapshot(flat, self.n_qubits, self._rng) for _ in range(num_snapshots)
        ]
        return self.reconstruct_from_snapshots(snapshots)

    def reconstruct_from_snapshots(
        self, snapshots: Sequence[Snapshot]
    ) -> ShadowResult:
        """Reconstruct from snapshots collected elsewhere.

        The estimator is a pure function of the snapshots, so measurements taken
        on hardware or in another process can be fed in directly.
        """
        if not snapshots:
            raise ValueError("need at least one snapshot")
        for snapshot in snapshots:
            if snapshot.n_qubits != self.n_qubits:
                raise ValueError(
                    f"snapshot covers {snapshot.n_qubits} qubits, "
                    f"expected {self.n_qubits}"
                )

        expectations = np.empty(4**self.n_qubits, dtype=float)
        for index, label in enumerate(pauli_labels(self.n_qubits)):
            required = [PAULI_ORDER.index(a) for a in label if a != "I"]
            if not required:
                expectations[index] = 1.0
                continue
            # Average the product of outcomes over the snapshots whose basis
            # matched this string on EVERY qubit.
            #
            # It is tempting to multiply the per-qubit Bloch components instead,
            # and for a product state the two agree. They are not the same
            # estimator in general: the product of means equals the mean of
            # products only if the terms are independent, and the per-qubit
            # outcomes are correlated by the entanglement. On a Bell state that
            # mistake reports <XX> = <X><X> = 0 instead of 1, which is not a
            # sampling error but a systematic one that vanishes as the sample
            # count grows.
            total = 0.0
            count = 0
            for snapshot in snapshots:
                if all(
                    snapshot.basis[qubit] == axis
                    for qubit, axis in zip(
                        (q for q, a in enumerate(label) if a != "I"), required
                    )
                ):
                    product = 1.0
                    for qubit, axis in enumerate(label):
                        if axis != "I":
                            product *= float(snapshot.outcomes[qubit])
                    total += product
                    count += 1
            expectations[index] = total / count if count else 0.0

        return ShadowResult(
            n_qubits=self.n_qubits,
            num_snapshots=len(snapshots),
            expectations=expectations,
            shadow_norm_bound=self.shadow_norm,
        )
