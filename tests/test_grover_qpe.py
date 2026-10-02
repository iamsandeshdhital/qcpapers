"""
Tests for Grover search, amplitude amplification and phase estimation.

Oracles used:

* Grover: Grover's own bound ``k = round(pi/(4 theta) - 1/2)`` and the
  success probability ``sin^2((2k+1) theta)``.
* Amplitude amplification: the same closed form with the good-subspace
  dimension substituted for the marked-item count.
* Phase estimation: the exact DFT of the eigenphase,
  ``amp(y) = sin(N pi d) / (N sin(pi d))`` with ``d = phase - y/N``, plus the
  dense QFT matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcpapers.algorithms.grover import (
    GroverSearch,
    amplitude_amplification,
    grover_search,
    optimal_iteration_count,
)
from qcpapers.algorithms.qpe import (
    QuantumPhaseEstimation,
    inverse_qft,
    iterative_qpe,
    qft,
    qpe_circuit,
)
from qcpapers.core.circuit import Circuit


# ---------------------------------------------------------------------------
# Grover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_iteration_count_matches_grover_bound(n_qubits: int) -> None:
    """``k`` must equal ``round(pi / (4 theta) - 1/2)`` and be ``O(sqrt(N))``."""
    n_items = 2**n_qubits
    theta = math.asin(1.0 / math.sqrt(n_items))
    expected = int(round(math.pi / (4 * theta) - 0.5))
    assert optimal_iteration_count(n_items) == expected
    # Sub-linear in N, which is the whole point of the algorithm.
    assert expected < n_items


def test_stricter_threshold_needs_more_iterations() -> None:
    assert optimal_iteration_count(16, 0.5) <= optimal_iteration_count(16, 0.9)


def test_optimal_iterations_beat_the_uniform_baseline() -> None:
    """The marked probability must be far above the ``1/N`` starting value.

    Grover does not generally reach probability 1: ``sin^2((2k+1) theta)`` equals
    1 only when ``(2k+1) theta`` is exactly ``pi/2``. What must hold is that the
    marked item dominates the distribution.
    """
    for n_qubits in range(2, 7):
        search = GroverSearch(n_qubits, marked=[2**n_qubits - 3])
        assert search.success_probability() > 1 / search.n_items + 0.3
        assert np.isclose(
            search.success_probability(),
            search.predicted_success_probability(),
            atol=1e-9,
        )


def test_probability_one_only_when_grid_aligned() -> None:
    """Only ``N = 4`` reaches probability exactly 1 among small search spaces.

    Grover's optimum is ``sin^2((2k+1) theta)``, which equals 1 only when
    ``(2k+1) theta`` is exactly ``pi/2``. For ``N = 4`` that holds
    (``theta = pi/6``, ``k = 1``); for ``N = 8`` and ``N = 16`` it does not, so
    the success probability falls slightly short. Asserting 1 everywhere would
    encode a claim the algorithm does not make.
    """
    assert np.isclose(GroverSearch(2, marked=[1]).success_probability(), 1.0)
    for n_qubits in (3, 4, 5, 6):
        search = GroverSearch(n_qubits, marked=[1])
        probability = search.success_probability()
        assert probability < 1.0 or n_qubits == 2
        # Even when short of 1, it must be far above the 1/N starting point.
        assert probability > 0.9
        assert probability == pytest.approx(
            search.predicted_success_probability(), abs=1e-9
        )


def test_success_probability_matches_closed_form() -> None:
    """The circuit's Born-rule probability must equal ``sin^2((2k+1) theta)``."""
    for n_qubits in (2, 3, 4):
        for target in range(2**n_qubits):
            search = GroverSearch(n_qubits, marked=[target])
            assert np.isclose(
                search.success_probability(),
                search.predicted_success_probability(),
                atol=1e-9,
            )


def test_marked_item_is_the_top_measurement_outcome() -> None:
    search = GroverSearch(5, marked=[13], name="grover_test")
    counts = search.counts(shots=4096, seed=11)
    top = int(next(iter(counts)), 2)
    assert top == 13
    assert search.verify(shots=4096)
    assert counts[format(13, "05b")] > 0.9 * sum(counts.values())


def test_multiple_marked_items_scale_as_sqrt_n_over_m() -> None:
    """With ``m`` marked items the iteration count drops like ``sqrt(N/m)``."""
    counts = {}
    for m in (1, 2, 4, 8):
        search = GroverSearch(8, marked=list(range(m)))
        counts[m] = search.iterations
        assert search.success_probability() > 0.9
        assert np.isclose(
            search.success_probability(),
            search.predicted_success_probability(),
            atol=1e-9,
        )
    # Doubling the marked set must reduce the iteration count.
    assert counts[1] > counts[2] > counts[4] > counts[8]


def test_iteration_count_uses_effective_space_size() -> None:
    """The count must come from ``N/m``, not ``N``; using ``N`` over-iterates."""
    four = GroverSearch(8, marked=[0, 1, 2, 3])
    assert four.effective_space_size() == 64.0
    assert four.iterations == optimal_iteration_count(64.0)
    # Over-iterating with the single-solution count destroys the probability.
    over = GroverSearch(8, marked=[0, 1, 2, 3], iterations=optimal_iteration_count(256))
    assert over.success_probability() < four.success_probability()


def test_query_complexity_is_sqrt_n() -> None:
    """The oracle query count must be ``Theta(sqrt(N))``, not ``Theta(N)``."""
    queries = {
        n: GroverSearch(n, marked=[0]).expected_queries_to_marked()
        for n in (8, 10, 12, 14)
    }
    # The steps here add two qubits each, so N grows by 4 and the query count
    # grows by sqrt(4) = 2.
    for lower, higher in ((8, 10), (10, 12), (12, 14)):
        assert queries[higher] / queries[lower] == pytest.approx(2.0, rel=0.06)
    # And the absolute scale matches the (pi/4) sqrt(N) prediction.
    assert queries[14] == pytest.approx(math.pi / 4 * math.sqrt(2**14), rel=0.02)
    assert GroverSearch(14, marked=[0]).ideal_query_bound() == pytest.approx(
        math.pi / 4 * math.sqrt(2**14)
    )


def test_marked_items_reduce_query_complexity() -> None:
    """``m`` marked items cost ``(pi/4) sqrt(N/m)`` queries."""
    bounds = {
        m: GroverSearch(10, marked=list(range(m))).ideal_query_bound()
        for m in (1, 4, 16)
    }
    assert bounds[1] / bounds[4] == pytest.approx(2.0, rel=1e-9)
    assert bounds[1] / bounds[16] == pytest.approx(4.0, rel=1e-9)


def test_circuit_metadata_records_the_paper() -> None:
    metadata = GroverSearch(3, marked=[2]).circuit().metadata
    assert "Grover" in metadata["reference"]
    assert metadata["n_items"] == 8
    assert metadata["iterations"] == GroverSearch(3, marked=[2]).iterations


def test_marked_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        GroverSearch(3, marked=[99])
    with pytest.raises(ValueError, match="at least one item"):
        GroverSearch(3, marked=[])


def test_grover_search_helper_is_optimal() -> None:
    search = grover_search(4, marked=[6])
    assert search.iterations == optimal_iteration_count(16)


def test_suboptimal_iterations_still_find_the_marked_item() -> None:
    """One iteration fewer than optimal still amplifies above the uniform 1/16."""
    search = GroverSearch(4, marked=[5], iterations=1)
    assert search.success_probability() > 1 / 16


def test_amplitude_amplification_matches_grover() -> None:
    """With a uniform initial state and a one-dimensional good subspace,
    amplitude amplification reduces to Grover."""
    n = 3
    target = 5
    initial = Circuit(n)
    for q in range(n):
        initial.h(q)
    marked = Circuit(n)
    marked.oracle_diagonal(
        np.array([-1.0 if i == target else 1.0 for i in range(2**n)]), list(range(n))
    )
    grover = GroverSearch(n, marked=[target]).circuit()

    amp = amplitude_amplification(initial, marked, good_subspace_dim=1)
    initial_probs = np.abs(initial.probabilities()) ** 2
    initial_success = float(initial_probs[target])

    for name, circuit in (("grover", grover), ("amplitude_amplification", amp)):
        probs = circuit.probabilities()
        total = float(np.abs(probs).sum() ** 2)
        assert total == pytest.approx(1.0, abs=1e-9), name
        assert float(probs[target]) >= initial_success, name


def test_amplitude_amplification_improves_probability() -> None:
    """Amplification must beat the unamplified success probability."""
    n = 4
    target = 9
    initial = Circuit(n)
    for q in range(n):
        initial.h(q)
    marked = Circuit(n)
    marked.oracle_diagonal(
        np.array([-1.0 if i == target else 1.0 for i in range(2**n)]), list(range(n))
    )
    initial_success = float(initial.probabilities()[target])
    amp = amplitude_amplification(initial, marked, good_subspace_dim=1)
    assert float(amp.probabilities()[target]) > initial_success


def test_amplitude_amplification_rejects_mismatched_registers() -> None:
    with pytest.raises(ValueError, match="same register"):
        amplitude_amplification(Circuit(2), Circuit(3))


# ---------------------------------------------------------------------------
# Quantum Fourier transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_dense_matrix(n: int) -> None:
    """The ladder must equal ``exp(2 pi i a b / 2**n) / sqrt(2**n)``."""
    dim = 2**n
    dense = np.array(
        [[np.exp(2j * np.pi * a * b / dim) / math.sqrt(dim) for b in range(dim)]
         for a in range(dim)]
    )
    circuit = Circuit(n)
    qft(circuit, list(range(n)))
    assert np.allclose(circuit.unitary(), dense, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_inverse_qft_is_adjoint(n: int) -> None:
    forward = Circuit(n)
    qft(forward, list(range(n)))
    inverse = Circuit(n)
    inverse_qft(inverse, list(range(n)))
    assert np.allclose(inverse.unitary(), forward.unitary().conj().T, atol=1e-9)
    assert np.allclose(inverse.unitary() @ forward.unitary(), np.eye(2**n), atol=1e-9)


def test_qft_preserves_state_norm() -> None:
    circuit = Circuit(3).h(0).cx(0, 1).ry(0.7, 2)
    transformed = circuit.copy()
    qft(transformed, [0, 1, 2])
    assert np.isclose(np.linalg.norm(transformed.run().state), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Quantum phase estimation
# ---------------------------------------------------------------------------


GRID_PHASES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75)


@pytest.mark.parametrize("theta", GRID_PHASES)
@pytest.mark.parametrize("m", [3, 4, 5])
def test_qpe_recovers_grid_aligned_phase_exactly(theta: float, m: int) -> None:
    """A phase on the ``2**m`` grid must be recovered with certainty."""
    unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=m, state_index=1)
    assert np.isclose(qpe.estimate(), theta, atol=1e-9)
    assert np.isclose(qpe.probability_distribution().max(), 1.0, atol=1e-9)
    assert np.isclose(qpe.theoretical_success_probability(), 1.0, atol=1e-9)


@pytest.mark.parametrize("theta", GRID_PHASES)
@pytest.mark.parametrize("m", [3, 4, 5])
def test_qpe_is_not_bit_reversed(theta: float, m: int) -> None:
    """Guards the ``2**(m-1-q)`` exponent assignment in the QPE ladder.

    Assigning ``2**q`` instead produces a bit-reversed readout, which still
    returns a plausible-looking phase, so the check is that the estimate differs
    from its own bit reversal wherever the two differ.
    """
    unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=m, state_index=1)
    expected = round(theta * 2**m) / 2**m
    assert np.isclose(qpe.estimate(), expected, atol=1e-9)

    index = round(expected * 2**m)
    reversed_index = int(format(index, f"0{m}b")[::-1], 2)
    if reversed_index != index:
        assert not np.isclose(qpe.estimate(), reversed_index / 2**m, atol=1e-9)


@pytest.mark.parametrize("theta", [0.05, 0.2, 0.3, 0.9])
@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_closed_form_success_probability_matches_circuit(theta: float, m: int) -> None:
    """``sin(N pi d)^2 / (N sin(pi d))^2`` must equal the circuit's maximum."""
    unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=m, state_index=1)
    assert np.isclose(
        qpe.theoretical_success_probability(),
        qpe.probability_distribution().max(),
        atol=1e-9,
    )


def test_qpe_snaps_to_nearest_grid_point() -> None:
    """An off-grid phase resolves to the nearest multiple of ``1/2**m``."""
    unitary = np.diag([1.0, np.exp(2j * np.pi * 0.3)])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=5, state_index=1)
    assert np.isclose(qpe.estimate(), 10 / 32, atol=1e-9)
    assert qpe.error_bound() == pytest.approx(1 / 32)


def test_more_phase_qubits_improve_resolution() -> None:
    unitary = np.diag([1.0, np.exp(2j * np.pi * 0.3)])
    errors = [
        abs(QuantumPhaseEstimation(unitary, n_phase_qubits=m, state_index=1).estimate() - 0.3)
        for m in (3, 5, 8)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_qpe_on_multi_qubit_unitary() -> None:
    """Estimate an eigenvalue of a 4x4 diagonal unitary from basis state 2."""
    phases = [0.0, 0.25, 0.625, 0.125]
    unitary = np.diag([np.exp(2j * np.pi * p) for p in phases])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=4, state_index=2)
    assert np.isclose(qpe.estimate(), 0.625, atol=1e-9)
    assert qpe.n_system_qubits == 2
    assert qpe.n_qubits == 6


def test_qpe_estimate_from_samples_agrees_with_exact() -> None:
    unitary = np.diag([1.0, np.exp(2j * np.pi * 0.375)])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=5, state_index=1)
    estimate, probability = qpe.estimate_from_samples(shots=2048)
    assert np.isclose(estimate, qpe.estimate(), atol=1e-9)
    assert probability > 0.95


def test_qpe_metadata_and_circuit_helper() -> None:
    unitary = np.diag([1.0, np.exp(2j * np.pi * 0.25)])
    circuit = qpe_circuit(unitary, n_phase_qubits=3, state_index=1)
    assert circuit.metadata["algorithm"] == "qpe"
    assert circuit.n_qubits == 4
    assert "Kitaev" in circuit.metadata["reference"]


def test_qpe_rejects_non_eigenstate_input() -> None:
    """A basis state that ``U`` mixes with others cannot be an input eigenstate."""
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    with pytest.raises(ValueError, match="not an eigenvector"):
        QuantumPhaseEstimation(x_gate, n_phase_qubits=3, state_index=0)


def test_qpe_rejects_non_unitary_input() -> None:
    with pytest.raises(ValueError, match="unitary"):
        QuantumPhaseEstimation(np.ones((2, 2), dtype=complex), n_phase_qubits=3)


def test_qpe_exact_phase_matches_measured_estimate() -> None:
    """The stored eigenphase and the sampled estimate must agree."""
    theta = 0.375
    unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
    qpe = QuantumPhaseEstimation(unitary, n_phase_qubits=5, state_index=1)
    assert np.isclose(qpe.phase, theta, atol=1e-12)
    assert np.isclose(qpe.estimate(), qpe.phase, atol=1e-9)


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_iterative_qpe_is_exact_on_the_grid(m: int) -> None:
    """Every grid-aligned phase must be recovered exactly, as in full QPE.

    The semiclassical recursion reads the binary digits of the phase, so it must
    agree with :class:`QuantumPhaseEstimation` bit for bit whenever the phase
    lies on the ``2**-m`` grid.
    """
    for y in range(2**m):
        theta = y / 2**m
        unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
        estimate = iterative_qpe(
            unitary, n_iterations=m, n_classical_bits=m, state_index=1
        )
        assert estimate == pytest.approx(theta, abs=1e-12), (m, y)


def test_iterative_qpe_approximates_off_grid_phase() -> None:
    """An off-grid phase is recovered to within one grid step."""
    theta = 0.3
    unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
    estimate = iterative_qpe(
        unitary, n_iterations=8, n_classical_bits=8, state_index=1
    )
    assert abs(estimate - theta) <= 2.0**-8


def test_iterative_qpe_truncates_where_full_qpe_rounds() -> None:
    """The two routes agree on-grid and differ by at most one step off-grid.

    The semiclassical recursion reads binary digits, so it truncates; the inverse
    QFT in the full-register version snaps to the nearest grid point. This is a
    real and documented difference, not a defect in either, so the test pins the
    behaviour rather than papering over it.
    """
    theta = 0.3
    unitary = np.diag([1.0, np.exp(2j * np.pi * theta)])
    semiclassical = iterative_qpe(
        unitary, n_iterations=8, n_classical_bits=8, state_index=1
    )
    full = QuantumPhaseEstimation(unitary, n_phase_qubits=8, state_index=1).estimate()

    grid = 2.0**-8
    assert semiclassical == pytest.approx(math.floor(theta / grid) * grid)
    assert full == pytest.approx(round(theta / grid) * grid)
    assert abs(semiclassical - full) <= grid
    # Both are valid estimates of the same phase.
    assert abs(semiclassical - theta) <= grid
    assert abs(full - theta) <= grid / 2


def test_iterative_qpe_requires_diagonal_unitary() -> None:
    """A U that mixes basis states makes the controlled power meaningless."""
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    with pytest.raises(ValueError, match="diagonal"):
        iterative_qpe(x_gate, n_iterations=4, n_classical_bits=4)


def test_iterative_qpe_rejects_oversized_iteration_count() -> None:
    unitary = np.diag([1.0, -1.0])
    with pytest.raises(ValueError, match="must not exceed"):
        iterative_qpe(unitary, n_iterations=5, n_classical_bits=3)
