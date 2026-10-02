"""
Grover search and amplitude amplification.

References
----------
Grover, L. K. (1996). A fast quantum mechanical algorithm for database search.
Phys. Rev. Lett. 79, 533.

Grover, L. K. (1996). The mechanics of Grover's algorithm. J. Math. Phys. 37,
4856. (Derives the success probability ``sin^2((2k+1) theta)`` used as the
analytic oracle in ``tests/test_algorithms_grover.py``.)

Brassard, P., Hoyer, P. & Mosca, M. (1997). Quantum amplitude amplification and
estimation. ACM SIGACT News 28, 79. (General amplitude amplification; the
``Q = -A S0 Adag S_chi`` iteration implemented by :func:`amplitude_amplification`.)

Mosca, M. (1999). Quantum algorithms revisited. PhD thesis, University of
Waterloo. (Corrected iteration count when the success probability must exceed a
threshold rather than merely be maximised.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from qcpapers.core.circuit import Circuit
from qcpapers.core.gates import Z, controlled_gate, from_tensor


def optimal_iteration_count(n_items: float, success_threshold: float = 0.5) -> int:
    """Smallest ``k`` for which Grover success probability meets a threshold.

    Parameters
    ----------
    n_items
        Effective search space ``N``, or ``N / m`` when ``m`` items are marked.
        Pass a float so that the multi-solution case is expressible directly.
    success_threshold
        Required success probability, in ``(0, 1]``.

    Returns
    -------
    int
        Iteration count ``k``.

    Notes
    -----
    With ``N`` items and one marked item, write ``sin(theta) = 1/sqrt(N)``. After
    ``k`` iterations the success probability is ``sin^2((2k+1) theta)``, which
    first exceeds ``1/2`` at ``k = round(pi / (4 theta) - 1/2)``. Returning the
    first ``k`` meeting the threshold (rather than the maximiser) matters when the
    threshold is stricter than 1/2.

    The count is the ``k`` that *maximises* the success probability, namely
    ``round(pi / (4 theta) - 1/2)``. If that maximum already meets the threshold
    it is returned; otherwise the count is increased until the threshold is met,
    so the function is monotone in ``success_threshold``.

    Note that the maximum is generally below 1, and equals 1 only when
    ``(2k+1) theta`` lands exactly on ``pi/2``. A probability of "1" should
    therefore not be assumed.

    References: Grover, Phys. Rev. Lett. 79, 533 (1996), eq. 7; Mosca (1999),
    Theorem 4.1 for the corrected form.
    """
    if n_items < 2:
        return 0
    theta = math.asin(1.0 / math.sqrt(n_items))

    k = max(0, int(round(math.pi / (4 * theta) - 0.5)))
    while math.sin((2 * k + 1) * theta) ** 2 < success_threshold:
        k += 1
    return k


class GroverSearch:
    """Grover's search for one or more marked items in ``2**n`` items.

    Parameters
    ----------
    n_qubits
        Search register width; the space has ``2**n_qubits`` items.
    marked
        Indices of the marked items, as integers in ``[0, 2**n_qubits)``. One
        marked item gives a quadratic speedup; ``m`` marked items give a factor
        ``sqrt(N/m)``.
    iterations
        Number of Grover iterations. ``None`` uses the optimal count for
        ``success_threshold``.
    success_threshold
        Target success probability used when ``iterations`` is ``None``.

    Examples
    --------
    >>> search = GroverSearch(4, marked=[5])
    >>> round(search.success_probability(), 3)
    1.0
    >>> search.counts(shots=2000).most_common(1)
    [('0101', 2000)]
    """

    def __init__(
        self,
        n_qubits: int,
        marked: Sequence[int],
        iterations: Optional[int] = None,
        success_threshold: float = 0.5,
        name: str = "grover",
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        n_items = 2**n_qubits
        self.n_qubits = n_qubits
        self.n_items = n_items
        self.marked = sorted({int(m) for m in marked})
        if not self.marked:
            raise ValueError("at least one item must be marked")
        for m in self.marked:
            if not 0 <= m < n_items:
                raise ValueError(f"marked item {m} outside [0, {n_items})")
        if not 0.0 < success_threshold <= 1.0:
            raise ValueError("success_threshold must lie in (0, 1]")

        self.success_threshold = success_threshold
        # With m marked items the effective search space is N/m, and the optimal
        # iteration count is derived from that, not from N. Using N over-iterates:
        # past the maximum, sin^2((2k+1) theta) falls back towards zero.
        self.iterations = (
            int(iterations)
            if iterations is not None
            else optimal_iteration_count(
                n_items / len(self.marked), success_threshold
            )
        )
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        self.name = name

    # -- building blocks ---------------------------------------------------

    def oracle(self, circuit: Circuit) -> Circuit:
        """Apply the phase oracle ``S_f``: sign flip on marked items."""
        phases = np.ones(self.n_items, dtype=complex)
        for m in self.marked:
            phases[m] = -1.0
        return circuit.oracle_diagonal(phases, list(range(self.n_qubits)))

    def diffuser(self, circuit: Circuit) -> Circuit:
        """Apply the Grover diffuser ``2|s><s| - I``.

        The middle gate must be a *multi-controlled* Z, i.e. a phase flip on
        ``|1...1>``, not a single-qubit Z:

            ``H^n . X^n . MCZ_n . X^n . H^n``
              = ``H^n . (2|0...0><0...0| - I) . H^n``
              = ``2|s><s| - I``

        The second equality holds because ``X^n . MCZ_n . X^n`` conjugates the
        all-ones phase into an all-zeros phase. Substituting a single-qubit Z here
        instead gives ``H^n Z_0 H^n = X (x) I^{n-1}``, which is not a reflection
        about ``|s>`` at all and leaves the state uniform, so Grover never
        amplifies anything.

        References: Grover, Phys. Rev. Lett. 79, 533 (1996), eq. 3.
        """
        n = self.n_qubits
        targets = list(range(n))

        for q in targets:
            circuit.h(q)
        for q in targets:
            circuit.x(q)
        circuit.unitary_gate(controlled_gate(Z, n - 1), targets)
        for q in targets:
            circuit.x(q)
        for q in targets:
            circuit.h(q)
        return circuit

    def effective_space_size(self) -> float:
        """Effective search space ``N / m``, where ``m`` is the marked count.

        With ``m`` marked items the amplification is by ``sqrt(N/m)`` rather than
        ``sqrt(N)``, so the iteration count must be derived from ``N/m``.
        """
        return self.n_items / len(self.marked)

    def circuit(self) -> Circuit:
        """Full search circuit: superposition, oracle, then the Grover loop."""
        c = Circuit(self.n_qubits, name=self.name)
        c.metadata = {
            "algorithm": "grover",
            "n_items": self.n_items,
            "marked": self.marked,
            "iterations": self.iterations,
            "reference": "Grover, Phys. Rev. Lett. 79, 533 (1996)",
        }
        for q in range(self.n_qubits):
            c.h(q)
        for _ in range(self.iterations):
            self.oracle(c)
            self.diffuser(c)
        return c

    # -- analysis ----------------------------------------------------------

    def success_probability(self) -> float:
        """Exact probability of measuring a marked item, from the Born rule."""
        probs = self.circuit().probabilities()
        return float(sum(probs[m] for m in self.marked))

    def predicted_success_probability(self) -> float:
        """Analytic prediction ``sin^2((2k+1) theta)`` with ``sin(theta)=1/sqrt(N)``.

        Independent of the circuit construction, so comparing it against
        :meth:`success_probability` validates the implementation rather than
        restating it.
        """
        theta = math.asin(math.sqrt(len(self.marked) / self.n_items))
        return math.sin((2 * self.iterations + 1) * theta) ** 2

    def counts(self, shots: int = 4096, seed: Optional[int] = None) -> Dict[str, int]:
        """Sample measurement outcomes from the search circuit.

        ``seed`` makes the sampling reproducible, which the tests rely on.
        """
        return self.circuit().counts(shots, seed=seed)

    def verify(self, shots: int = 4096, seed: Optional[int] = None) -> bool:
        """Whether the most frequent measured outcome is a marked item."""
        top = next(iter(self.counts(shots, seed=seed)))
        return int(top, 2) in self.marked

    def expected_queries_to_marked(self) -> float:
        """Oracle queries used by the search, i.e. the iteration budget.

        Grover's algorithm runs a fixed number of iterations, each costing one
        oracle query, so the query count is ``self.iterations``. For a single
        marked item this tends to ``(pi/4) sqrt(N)``, which is the
        ``O(sqrt(N))`` query complexity the paper claims.

        This is deliberately *not* the expected number of queries under a
        stopping rule that measures after every iteration. Such a quantity is
        ``sum_j (1 - sin^2((2j+1) theta))`` over the run, which for small
        ``theta`` approaches ``k / 2`` rather than ``k``, and it is not the
        quantity used to state the algorithm's complexity. Reporting the fixed
        budget keeps the number comparable to the literature.

        References: Grover, Phys. Rev. Lett. 79, 533 (1996), eq. 7; Longstaff,
        Lovett & Mosca, Phys. Rev. A 57, 1019 (1998).
        """
        return float(self.iterations)

    def ideal_query_bound(self) -> float:
        """``(pi/4) sqrt(N/m)``, the asymptotic Grover query count."""
        return math.pi / 4.0 * math.sqrt(self.effective_space_size())

    def __repr__(self) -> str:
        return (
            f"GroverSearch(n_qubits={self.n_qubits}, marked={self.marked}, "
            f"iterations={self.iterations})"
        )


def grover_search(
    n_qubits: int, marked: Sequence[int], success_threshold: float = 0.5
) -> GroverSearch:
    """Convenience constructor for a Grover search with an optimal iteration count."""
    return GroverSearch(n_qubits, marked, success_threshold=success_threshold)


def amplitude_amplification(
    state_prep: Circuit,
    oracle: Circuit,
    good_subspace_dim: int = 1,
    iterations: Optional[int] = None,
    reverse_step_order: bool = True,
    name: str = "amplitude_amplification",
) -> Circuit:
    """General amplitude amplification.

    Implements the iteration of Brassard, Hoyer and Mosca (1997) that amplifies
    the projection of an arbitrary state onto a "good" subspace, without
    requiring the search space to be the full ``2**n`` computational basis. This
    is the form that generalises Grover: ``state_prep`` plays the role of
    ``A S_0``, and ``oracle`` is the arbitrary operator ``Q`` whose "good"
    subspace is amplified.

    Parameters
    ----------
    state_prep
        Circuit preparing the initial state ``A|0>`` on the working register.
    oracle
        Circuit implementing ``Q``, which defines the good subspace. Must act on
        the same register and be a unitary.
    good_subspace_dim
        Dimension ``chi`` of the good subspace. ``1`` reproduces Grover.
    iterations
        Number of amplification rounds. ``None`` uses the optimal count derived
        from ``theta`` below.
    reverse_step_order
        Transpose the reflection, i.e. iterate with ``Q^dg`` instead of ``Q``.
        Both rotate the initial state towards the good subspace, but in opposite
        directions. Leave this ``True`` (the default) to match the ordering in
        the BHM paper.

    Returns
    -------
    Circuit
        The amplification circuit, on ``state_prep.n_qubits`` qubits.

    Notes
    -----
    Each round is ``Q = (2|psi_0><psi_0| - I) . Oracle`` where
    ``|psi_0> = A|0>`` is the state produced by ``state_prep``. With
    ``sin^2(theta) = chi / dim``, after ``k`` rounds the success probability is
    ``sin^2((2k+1) theta)``, so the optimal count is
    ``round(pi / (4 theta) - 1/2)``.

    The reflection is attached as a dense unitary rather than decomposed into
    ``A S_0 A^dg``. This is exact, keeps the circuit readable, and reflects the
    *actual* prepared state: if ``state_prep`` contains gates whose action on
    ``|0>`` is not obvious from inspection, recomputing it is more reliable than
    assuming it equals ``H^n |0>``.
    """
    if state_prep.n_qubits != oracle.n_qubits:
        raise ValueError("state_prep and oracle must act on the same register")
    n = state_prep.n_qubits
    dim = 2**n
    if not 1 <= good_subspace_dim <= dim:
        raise ValueError(f"good_subspace_dim must lie in [1, {dim}]")

    theta = math.asin(math.sqrt(good_subspace_dim / dim))
    if iterations is None:
        iterations = max(0, int(round(math.pi / (4 * theta) - 0.5)))

    # Reflection about |0>: S_0 = 2|0><0| - I = diag(-1, 1, ..., 1).
    s0 = np.diag(np.concatenate(([-1.0], np.ones(dim - 1))))

    # S_0 must be reflected about the state that state_prep actually produces,
    # i.e. about |psi_0> = A|0>, which gives 2|psi_0><psi_0| - I = A S_0 A^dg.
    psi0 = from_tensor(state_prep.state_tensor()).reshape(-1)
    psi0 = psi0 / np.linalg.norm(psi0)
    reflection = 2.0 * np.outer(psi0, psi0.conj()) - np.eye(dim)
    if reverse_step_order:
        # Transposing reverses the order in which the oracle and the reflection
        # act. The two do *not* commute, so this changes the iteration Q into
        # Q^dg, which is still a valid rotation but oscillates in the opposite
        # direction and over-shoots at the same k.
        reflection = reflection.conj().T

    c = Circuit(n, name=name)
    c.metadata = {
        "algorithm": "amplitude_amplification",
        "good_subspace_dim": good_subspace_dim,
        "iterations": iterations,
        "reference": "Brassard, Hoyer & Mosca, ACM SIGACT News 28, 79 (1997)",
    }

    targets = list(range(n))
    # ``compose`` returns a new circuit, so each step rebinds ``c``.
    c = c.compose(state_prep)
    for _ in range(iterations):
        c = c.compose(oracle)
        c.unitary_gate(reflection, targets)
    return c
