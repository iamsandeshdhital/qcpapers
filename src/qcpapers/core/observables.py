"""
Observables: Pauli strings, sparse Pauli operators, and expectation machinery.

References
----------
Nielsen, M. A. & Chuang, I. L. (2010). Quantum Computation and Quantum
Information, 10th ed. Section 2.2 (Pauli algebra) and Appendix C.

Seeley, J., Keating, G. & Ryan, M. (2012). Monte Carlo method for Hamiltonian
simulation. Phys. Rev. A 85, 032405. Grouping the terms of a sparse Pauli
Hamiltonian into commuting sets so that each group needs only a single
measurement basis. This is what :meth:`SparsePauliOp.group_commuting`
implements and underpins the VQE and QAOA cost functions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from qcpapers.core.gates import I2, X, Y, Z, from_tensor, gate_matrix, apply_matrix

PAULI_MATRICES = {"I": I2, "X": X, "Y": Y, "Z": Z}

#: All 16 single-qubit Pauli basis strings, ordered by the binary label ``iiii``.
PAULI_LABELS: Tuple[str, ...] = tuple("".join(p) for p in itertools.product("IXYZ", repeat=4))


def pauli_labels(n_qubits: int) -> Tuple[str, ...]:
    """All ``4**n`` Pauli strings on ``n_qubits``, in canonical lexicographic order."""
    return tuple("".join(p) for p in itertools.product("IXYZ", repeat=n_qubits))


def pauli_matrix(pauli: str) -> np.ndarray:
    """Single-qubit Pauli matrix for ``I``, ``X``, ``Y`` or ``Z``."""
    try:
        return PAULI_MATRICES[pauli.upper()]
    except KeyError as exc:
        raise ValueError(f"{pauli!r} is not a Pauli operator") from exc


def pauli_string_matrix(pauli_string: str) -> np.ndarray:
    """Dense matrix of a Pauli string such as ``"XIZ"`` (Kronecker product).

    ``numpy.kron`` accepts exactly two arguments, so the product is folded
    pairwise rather than spread over a single call. Order matters: the leftmost
    letter is the most significant factor, matching the big-endian basis used
    throughout this package.
    """
    factors = [pauli_matrix(p) for p in pauli_string.upper()]
    if not factors:
        raise ValueError("a Pauli string must have at least one factor")
    matrix = factors[0]
    for factor in factors[1:]:
        matrix = np.kron(matrix, factor)
    return matrix


def pauli_matrix_product(state: np.ndarray, pauli_string: str) -> complex:
    """Expectation ``<psi| P |psi>`` for a statevector tensor and a Pauli string.

    Applies each Pauli factor in turn and takes the overlap, which costs one
    sparse gate contraction per non-identity factor rather than forming the full
    ``2**n x 2**n`` observable.
    """
    tensor = np.asarray(state, dtype=complex)
    expectation = tensor
    for q, pauli in enumerate(pauli_string.upper()):
        if pauli != "I":
            expectation = apply_matrix(expectation, pauli_matrix(pauli), [q])
    return complex(np.vdot(from_tensor(tensor), from_tensor(expectation)))


def pauli_weight(pauli_string: str) -> int:
    """Number of non-identity factors in a Pauli string."""
    return sum(p != "I" for p in pauli_string.upper())


def commute(p: str, q: str) -> bool:
    """Whether two Pauli strings commute."""
    return pauli_weight(_anticommutation_mask(p, q)) % 2 == 0


def _anticommutation_mask(p: str, q: str) -> str:
    if len(p) != len(q):
        raise ValueError("Pauli strings must have equal length")
    mask = []
    for a, b in zip(p.upper(), q.upper()):
        if a == "I" or b == "I":
            mask.append("I")
        elif a == b:
            mask.append("I")
        else:
            mask.append("X")  # any anticommuting pair, tracked only in parity
    return "".join(mask)


def multiply_pauli(p: str, q: str) -> str:
    """Product of two Pauli strings, tracking the phase separately via sign."""
    if len(p) != len(q):
        raise ValueError("Pauli strings must have equal length")
    out = []
    for a, b in zip(p.upper(), q.upper()):
        if a == "I":
            out.append(b)
        elif b == "I":
            out.append(a)
        elif a == b:
            out.append("I")
        else:
            order = "XYZ".index(a) - "XYZ".index(b)
            out.append("XYZ"[order % 3])
    return "".join(out)


@dataclass(frozen=True)
class SparsePauliOp:
    """A weighted sum of Pauli strings, ``sum_i coeff_i * P_i``.

    This is the representation every variational-algorithm paper works with:
    the VQE energy of Peruzzo et al. (2014), the QAOA cost of Farhi, Goldstone
    & Gutmann (2014), and the quantum-volume depth-2 Hamiltonian of
    Cross et al., PRL 127, 180501 (2021) are all of this form.
    """

    terms: Tuple[Tuple[complex, str], ...]

    def __init__(self, terms: Iterable[Tuple[complex, str]] = ()):
        normalised = tuple((complex(coeff), pauli.upper()) for coeff, pauli in terms)
        if not normalised:
            raise ValueError("a SparsePauliOp needs at least one term")
        lengths = {len(p) for _, p in normalised}
        if len(lengths) != 1:
            raise ValueError("all Pauli strings in an operator must have equal length")
        object.__setattr__(self, "terms", normalised)

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "SparsePauliOp":
        """Decompose a dense Hermitian matrix into a Pauli expansion.

        Uses the Hilbert-Schmidt inner product
        ``coeff_P = 2**-n tr(P H)`` with the traces precomputed, which is exact
        rather than iterative (the same construction as Qiskit's
        ``SparsePauliOp.from_operator``).
        """
        matrix = np.asarray(matrix, dtype=complex)
        n = int(round(np.log2(matrix.shape[0])))
        if 2**n != matrix.shape[0]:
            raise ValueError("matrix dimension must be a power of two")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
            raise ValueError("SparsePauliOp.from_matrix requires a Hermitian matrix")

        dim = matrix.shape[0]
        terms: List[Tuple[complex, str]] = []
        for letters in itertools.product("IXYZ", repeat=n):
            label = "".join(letters)
            basis = pauli_string_matrix(label)
            coeff = complex(np.trace(basis @ matrix) / dim)
            if abs(coeff) > 1e-10:
                terms.append((coeff, label))
        return cls(terms)

    @classmethod
    def identity(cls, n_qubits: int) -> "SparsePauliOp":
        return cls([(1.0, "I" * n_qubits)])

    # -- accessors ---------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return len(self.terms[0][1])

    @property
    def size(self) -> int:
        return len(self.terms)

    def to_matrix(self) -> np.ndarray:
        """Dense matrix of the operator."""
        dim = 2**self.n_qubits
        out = np.zeros((dim, dim), dtype=complex)
        for coeff, pauli in self.terms:
            out += coeff * pauli_string_matrix(pauli)
        return out

    def expectation(self, state: np.ndarray, bra: np.ndarray | None = None) -> complex:
        """``<bra| H |state>`` for statevector tensors, term by term.

        Parameters
        ----------
        state
            Ket state as a ``(2,)*n`` tensor.
        bra
            Bra state as a ``(2,)*n`` tensor. Defaults to ``state``, giving the
            usual expectation value.

        Notes
        -----
        Passing a distinct bra is what makes this usable for gradient
        calculations: ``dE/dtheta = 2 Re[<d|psi>/dtheta| H |psi>]`` for a real
        Hermitian ``H`` needs the ket and the derivative ket on opposite sides.

        Returns a complex number; the imaginary part vanishes for a Hermitian
        operator with ``bra == state`` and a normalised state.
        """
        ket = state
        if bra is None:
            bra = state
        total = 0j
        for coeff, pauli in self.terms:
            acted = ket
            for q, factor in enumerate(pauli):
                if factor != "I":
                    acted = apply_matrix(acted, pauli_matrix(factor), [q])
            total += coeff * complex(np.vdot(from_tensor(bra), from_tensor(acted)))
        return total

    def expectation_from_counts(self, counts: Mapping[str, int], shots: int | None = None) -> float:
        """Estimate the expectation from measurement counts.

        ``counts`` must be keyed by full-register bitstrings, which is the format
        :meth:`~qcpapers.core.circuit.Circuit.counts` produces.

        For a Pauli string measured in its own eigenbasis, every non-identity
        factor contributes ``+1`` for outcome 0 and ``-1`` for outcome 1. The
        eigenvalue of a full-register bitstring is therefore
        ``(-1)**(number of active positions whose measured bit is 1)``, where
        the active positions are those on which a given Pauli string acts
        non-trivially.

        This assumes counts were collected in the diagonal (Z) basis. For
        counts obtained by rotating into an X or Y eigenbasis, rotate the counts
        first; :meth:`measure_group` does that automatically for grouped
        measurement. No grouping is applied here.
        """
        if not counts:
            return 0.0
        n = self.n_qubits
        weighted = 0.0
        total_shots = 0
        for bitstring, count in counts.items():
            if len(bitstring) != n:
                raise ValueError(
                    f"count key {bitstring!r} has {len(bitstring)} bits, expected {n}"
                )
            count = int(count)
            total_shots += count
            measured = np.array([int(b) for b in bitstring])
            energy = 0.0
            for coeff, pauli in self.terms:
                active = np.array([p != "I" for p in pauli])
                if not active.any():
                    energy += float(np.real(coeff))
                    continue
                negatives = int(measured[active].sum())
                energy += float(np.real(coeff)) * ((-1.0) ** negatives)
            weighted += energy * count
        return weighted / (shots if shots is not None else total_shots)

    # -- analysis ----------------------------------------------------------

    def group_commuting(self, return_groups: bool = False):
        """Partition the terms into mutually commuting sets.

        Uses a deterministic greedy colouring: repeatedly take the first unused
        term and absorb every later term that commutes with the whole set. For
        ``k`` terms this yields at most ``k`` groups and in practice far fewer,
        which is the minimum number of measurement bases needed.

        Parameters
        ----------
        return_groups
            When True, also return the list of Pauli-string groups.

        References
            Seeley, Keating & Ryan, Phys. Rev. A 85, 032405 (2012), for the
            commuting-group measurement strategy.
            Hugenholtz, O'Gorman & Browne, Phys. Rev. A 97, 032329 (2018), for
            the same colouring in the Pauli-LDPC setting.
        """
        colours: Dict[int, List[str]] = {}
        colour_of: Dict[str, int] = {}

        for _, pauli in self.terms:
            if pauli in colour_of:
                continue
            placed = False
            for cid, group in colours.items():
                if all(commute(pauli, existing) for existing in group):
                    group.append(pauli)
                    colour_of[pauli] = cid
                    placed = True
                    break
            if not placed:
                cid = len(colours)
                colours[cid] = [pauli]
                colour_of[pauli] = cid

        grouped = SparsePauliOp(
            [
                (sum(c for c, p in self.terms if p == pauli), pauli)
                for group in colours.values()
                for pauli in group
            ]
        )
        if return_groups:
            return grouped, list(colours.values())
        return grouped

    def pauli_fidelities(self, state: np.ndarray) -> np.ndarray:
        """All ``2**n`` Pauli expectation values ``<psi|P|psi>``.

        Returned in the ``PAULI_LABELS`` order, which lets any state be
        reconstructed by classical shadows (Huang, Harrow & Low, arXiv:2006.16719)
        without re-running the circuit.
        """
        n = self.n_qubits
        values = np.zeros(4**n, dtype=float)
        for i, letters in enumerate(itertools.product("IXYZ", repeat=n)):
            label = "".join(letters)
            values[i] = float(np.real(pauli_matrix_product(state, label)))
        return values

    def ground_energy_exact(self) -> float:
        """Exact lowest eigenvalue, for use as an oracle in tests.

        Only tractable for ``n_qubits <= 12`` or so, which is precisely the regime
        in which the reference implementations are validated.
        """
        return float(np.linalg.eigvalsh(self.to_matrix())[0].real)

    def __add__(self, other: "SparsePauliOp") -> "SparsePauliOp":
        return SparsePauliOp(self.terms + other.terms)

    def __mul__(self, scalar: complex) -> "SparsePauliOp":
        return SparsePauliOp([(coeff * scalar, pauli) for coeff, pauli in self.terms])

    __rmul__ = __mul__

    def __repr__(self) -> str:
        head = ", ".join(f"{c:.3f}*{p}" for c, p in self.terms[:4])
        more = "" if len(self.terms) <= 4 else f", ... (+{len(self.terms) - 4} terms)"
        return f"SparsePauliOp({head}{more})"
