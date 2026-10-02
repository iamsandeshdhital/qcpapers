"""
Tests for classical shadows.

Every expectation is checked against ``Tr(P rho)`` computed from the dense state,
which is independent of the shadow estimator. Two systematic-error modes are
pinned explicitly, since both produce results that look plausible and converge
to the wrong answer as the sample count grows:

* averaging over *all* snapshots rather than the matching ones, which shrinks
  every Bloch component by a factor of three;
* multiplying per-qubit Bloch components instead of averaging the product of
  outcomes, which is only valid for product states.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcpapers.algorithms.classical_shadows import (
    PAULI_ORDER,
    ClassicalShadow,
    ShadowResult,
    Snapshot,
    bloch_component,
    dense_pauli,
    snapshot_circuit,
    take_snapshot,
    take_snapshots,
)
from qcpapers.core.circuit import Circuit
from qcpapers.core.observables import pauli_labels


def exact_expectation(state: np.ndarray, pauli: str) -> float:
    """``<P>`` for a pure state, from the dense Pauli matrix."""
    vector = np.asarray(state, dtype=complex).reshape(-1)
    return float(np.real(vector.conj() @ dense_pauli(pauli) @ vector))


# ---------------------------------------------------------------------------
# Snapshot protocol
# ---------------------------------------------------------------------------


def test_snapshot_outcomes_are_plus_or_minus_one() -> None:
    state = Circuit(2).h(0).state_tensor().reshape(-1)
    snapshot = take_snapshot(state, 2, np.random.default_rng(0))
    assert snapshot.n_qubits == 2
    assert all(value in (1, -1) for value in snapshot.outcomes)
    assert all(0 <= b <= 2 for b in snapshot.basis)


def test_snapshot_pauli_string_matches_basis() -> None:
    snapshot = Snapshot(basis=(0, 1, 2), outcomes=(1, -1, 1))
    assert snapshot.pauli_string() == "ZXY"
    assert snapshot.matched(1, 1)
    assert not snapshot.matched(1, 0)


def test_snapshot_outcome_returns_zero_on_basis_mismatch() -> None:
    """A non-matching snapshot must contribute nothing, not a wrong number."""
    snapshot = Snapshot(basis=(0, 1), outcomes=(1, -1))
    assert snapshot.outcome(0, 0) == 1.0
    assert snapshot.outcome(1, 0) == 0.0
    assert snapshot.outcome(1, 1) == -1.0


def test_snapshot_rejects_malformed_data() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Snapshot(basis=(0, 1), outcomes=(1,))
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        Snapshot(basis=(0,), outcomes=(0,))


def test_snapshot_circuit_measures_the_requested_pauli() -> None:
    """Each basis branch must read its own axis.

    Pins the order bug in the Y branch: applying the Hadamard before the phase
    gate gives ``U^dg Z U = X``, identical to the X branch, so a state with
    ``<Y> = 0`` and ``<X> = 1`` would be reported as Y-polarised.

    Note that only Z has a unit eigenvalue on ``|0>``; X and Y both give ``0``
    there, which is why the test compares whole matrices rather than diagonal
    entries.
    """
    from qcpapers.core.gates import X, Y, Z

    targets = {"X": X, "Y": Y, "Z": Z}
    for basis_index, axis in enumerate(PAULI_ORDER):
        unitary = snapshot_circuit(1, [basis_index]).unitary()
        effective = unitary.conj().T @ Z @ unitary
        assert np.allclose(effective, targets[axis], atol=1e-9), axis


def test_y_basis_branch_is_distinct_from_x() -> None:
    """The Y branch must not coincide with the X branch."""
    from qcpapers.core.gates import Z

    x_unitary = snapshot_circuit(1, [1]).unitary()
    y_unitary = snapshot_circuit(1, [2]).unitary()
    assert not np.allclose(
        x_unitary.conj().T @ Z @ x_unitary, y_unitary.conj().T @ Z @ y_unitary, atol=1e-6
    )


def test_snapshot_circuit_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="basis must have"):
        snapshot_circuit(3, [0, 1])
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        snapshot_circuit(1, [5])


def test_snapshot_sampling_is_reproducible() -> None:
    state = Circuit(2).h(0).cx(0, 1).state_tensor().reshape(-1)
    first = take_snapshots(state, 2, 50, seed=7)
    second = take_snapshots(state, 2, 50, seed=7)
    assert first == second


def test_snapshot_circuit_preserves_norm() -> None:
    """The rotations are unitary, so probabilities must renormalise exactly."""
    state = Circuit(2).h(0).cx(0, 1).state_tensor()
    for basis in ([0, 0], [1, 2], [2, 1]):
        result = snapshot_circuit(2, basis).run(state)
        assert np.isclose(np.linalg.norm(result.state), 1.0, atol=1e-12)
        assert np.isclose(result.probabilities.sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Single-qubit reconstruction
# ---------------------------------------------------------------------------


def test_bloch_component_averages_only_matching_snapshots() -> None:
    """Pins the factor-of-three bug.

    Each axis is measured in roughly a third of snapshots. Averaging over all of
    them divides the signal by three, reporting a Bloch component three times too
    small, and the error does not vanish with more samples.
    """
    snapshots = [Snapshot(basis=(1,), outcomes=(1,)) for _ in range(10)]
    # Ten Z-basis snapshots with a mix of outcomes dilute the single X snapshot.
    snapshots += [Snapshot(basis=(0,), outcomes=(1,)) for _ in range(20)]
    # Matching-only: 1.  All-snapshots: 10/30 = 0.333.
    assert bloch_component(snapshots, 0, PAULI_ORDER.index("X")) == pytest.approx(1.0)


def test_bloch_component_is_unbiased_for_plus_state() -> None:
    state = Circuit(1).h(0).state_tensor().reshape(-1)
    snapshots = take_snapshots(state, 1, 3000, seed=3)
    assert bloch_component(snapshots, 0, PAULI_ORDER.index("X")) == pytest.approx(
        1.0, abs=0.05
    )
    assert bloch_component(snapshots, 0, PAULI_ORDER.index("Z")) == pytest.approx(
        0.0, abs=0.05
    )


def test_bloch_component_returns_zero_without_matches() -> None:
    snapshots = [Snapshot(basis=(0,), outcomes=(1,))]
    assert bloch_component(snapshots, 0, PAULI_ORDER.index("X")) == 0.0


# ---------------------------------------------------------------------------
# Many-qubit expectations
# ---------------------------------------------------------------------------


def test_identity_expectation_is_exactly_one() -> None:
    state = Circuit(3).h(0).cx(0, 1).state_tensor().reshape(-1)
    result = ClassicalShadow(3, seed=0).reconstruct(state, num_snapshots=200)
    assert result.expectation("III") == 1.0


def test_product_state_expectations_match_exactly() -> None:
    """A product state's Pauli expectations factorise, so recovery is easy."""
    circuit = Circuit(3).ry(0.7, 0).rx(1.2, 1).rz(-0.4, 2)
    state = circuit.state_tensor().reshape(-1)
    result = ClassicalShadow(3, seed=1).reconstruct(state, num_snapshots=8000)

    for pauli in ("III", "XII", "YII", "IZI", "IIZ", "XZI", "IYI"):
        assert result.expectation(pauli) == pytest.approx(
            exact_expectation(state, pauli), abs=0.1
        ), pauli


def test_bell_state_correlations_are_recovered() -> None:
    """Pins the factorisation bug: ``<XX>`` on a Bell state is 1, not ``<X><X>``.

    Multiplying per-qubit Bloch components would give ``0 * 0 = 0`` here. That
    error is systematic, so it survives any number of snapshots.
    """
    state = Circuit(2).h(0).cx(0, 1).state_tensor().reshape(-1)
    result = ClassicalShadow(2, seed=2).reconstruct(state, num_snapshots=6000)

    assert result.expectation("XX") == pytest.approx(1.0, abs=0.1)
    assert result.expectation("ZZ") == pytest.approx(1.0, abs=0.1)
    assert result.expectation("YY") == pytest.approx(-1.0, abs=0.1)
    # The product of the single-qubit values is zero for every one of these.
    assert abs(result.expectation("XX")) > 0.9


def test_ghz_state_correlations_are_recovered() -> None:
    state = Circuit(3).h(0).cx(0, 1).cx(1, 2).state_tensor().reshape(-1)
    result = ClassicalShadow(3, seed=3).reconstruct(state, num_snapshots=20000)
    assert result.expectation("XXX") == pytest.approx(1.0, abs=0.1)
    assert result.expectation("ZZI") == pytest.approx(1.0, abs=0.1)


def test_max_error_is_reported_against_the_true_state() -> None:
    state = Circuit(2).ry(0.9, 0).ry(2.1, 1).state_tensor().reshape(-1)
    density = np.outer(state, state.conj())
    result = ClassicalShadow(2, seed=4).reconstruct(state, num_snapshots=4000)
    assert 0.0 <= result.max_error(density) < 0.35


def test_error_decreases_like_one_over_sqrt_n() -> None:
    """The convergence rate is the shadow method's central practical claim."""
    circuit = Circuit(2).ry(0.9, 0).ry(2.1, 1)
    state = circuit.state_tensor().reshape(-1)
    target = exact_expectation(state, "XX")
    assert target == pytest.approx(0.6762, abs=1e-3)

    scaled = []
    for num_snapshots in (1000, 4000, 16000):
        errors = []
        for seed in range(6):
            result = ClassicalShadow(2, seed=seed).reconstruct(
                state, num_snapshots=num_snapshots
            )
            errors.append(abs(result.expectation("XX") - target))
        scaled.append(float(np.mean(errors)) * math.sqrt(num_snapshots))

    # If the error is proportional to 1/sqrt(N), each scaled value is constant.
    for value in scaled[1:]:
        assert value == pytest.approx(scaled[0], rel=0.5)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_dense_reconstruction_recovers_a_product_state() -> None:
    """``rho = 2**-n sum_P <P> P`` is exact for a product state up to noise."""
    circuit = Circuit(2).ry(0.7, 0).rx(1.2, 1)
    state = circuit.state_tensor().reshape(-1)
    density = np.outer(state, state.conj())

    result = ClassicalShadow(2, seed=5).reconstruct(state, num_snapshots=6000)
    reconstructed = result.to_matrix()

    assert np.isclose(np.trace(reconstructed).real, 1.0, atol=0.1)
    assert np.max(np.abs(reconstructed - density)) < 0.15


def test_dense_pauli_is_orthonormal_up_to_dimension() -> None:
    """``Tr(P Q) = 2**n delta_{PQ}`` underpins the reconstruction factor."""
    for label in ("ZI", "IZ", "XX", "YY"):
        for other in ("ZI", "IZ", "XX", "YY"):
            expected = 4.0 if label == other else 0.0
            trace = np.trace(dense_pauli(label) @ dense_pauli(other))
            assert float(trace.real) == pytest.approx(expected, abs=1e-9)


def test_sparse_pauli_op_drops_negligible_terms() -> None:
    state = Circuit(2).h(0).cx(0, 1).state_tensor().reshape(-1)
    result = ClassicalShadow(2, seed=6).reconstruct(state, num_snapshots=3000)
    dense_terms = result.to_sparse_pauli_op(tolerance=0.0).size
    sparse_terms = result.to_sparse_pauli_op(tolerance=0.1).size
    assert sparse_terms < dense_terms


def test_overlap_is_one_for_a_good_reconstruction() -> None:
    state = Circuit(2).h(0).state_tensor().reshape(-1)
    density = np.outer(state, state.conj())
    result = ClassicalShadow(2, seed=7).reconstruct(state, num_snapshots=5000)
    assert result.overlap(density) == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Sample complexity
# ---------------------------------------------------------------------------


def test_required_snapshots_follows_the_paper_bound() -> None:
    """``N = ceil(34 K / eps**2)`` with ``K = 3`` for single-qubit Paulis."""
    shadow = ClassicalShadow(2)
    assert shadow.required_snapshots(0.5) == math.ceil(34 * 3 / 0.25)
    assert shadow.required_snapshots(0.1) == math.ceil(34 * 3 / 0.01)
    # Doubling the shadow norm doubles the requirement.
    assert shadow.required_snapshots(0.5, shadow_norm=6.0) == 2 * shadow.required_snapshots(
        0.5
    )


def test_required_snapshots_is_independent_of_qubit_count() -> None:
    """The bound's key property: it does not grow with ``n``."""
    small = ClassicalShadow(2).required_snapshots(0.3)
    large = ClassicalShadow(10).required_snapshots(0.3)
    assert small == large


def test_required_snapshots_rejects_bad_epsilon() -> None:
    shadow = ClassicalShadow(2)
    with pytest.raises(ValueError, match="epsilon"):
        shadow.required_snapshots(0.0)
    with pytest.raises(ValueError, match="epsilon"):
        shadow.required_snapshots(1.5)


def test_epsilon_selects_the_snapshot_count() -> None:
    state = Circuit(1).ry(0.8, 0).state_tensor().reshape(-1)
    shadow = ClassicalShadow(1, seed=8)
    result = shadow.reconstruct(state, epsilon=0.5)
    assert result.num_snapshots == shadow.required_snapshots(0.5)


def test_total_measurements_counts_single_qubit_shots() -> None:
    assert ClassicalShadow(4).total_measurements(100) == 400


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_reconstruct_requires_a_sample_count() -> None:
    shadow = ClassicalShadow(2)
    state = Circuit(2).h(0).state_tensor().reshape(-1)
    with pytest.raises(ValueError, match="num_snapshots or epsilon"):
        shadow.reconstruct(state)


def test_reconstruct_rejects_wrong_state_size() -> None:
    shadow = ClassicalShadow(3)
    state = Circuit(2).h(0).state_tensor().reshape(-1)
    with pytest.raises(ValueError, match="amplitudes"):
        shadow.reconstruct(state, num_snapshots=10)


def test_reconstruct_rejects_unnormalised_state() -> None:
    shadow = ClassicalShadow(1)
    with pytest.raises(ValueError, match="normalised"):
        shadow.reconstruct(np.array([1.0, 1.0]), num_snapshots=10)


def test_reconstruct_rejects_nonpositive_snapshot_count() -> None:
    shadow = ClassicalShadow(1)
    state = Circuit(1).h(0).state_tensor().reshape(-1)
    with pytest.raises(ValueError, match="positive"):
        shadow.reconstruct(state, num_snapshots=0)


def test_reconstruct_from_snapshots_is_a_pure_function() -> None:
    """Feeding snapshots in directly must match sampling them internally."""
    state = Circuit(2).ry(1.1, 0).rx(0.6, 1).state_tensor().reshape(-1)
    snapshots = take_snapshots(state, 2, 2000, seed=9)

    shadow = ClassicalShadow(2, seed=0)
    from_snapshots = shadow.reconstruct_from_snapshots(snapshots)

    shadow_direct = ClassicalShadow(2, seed=10).reconstruct(state, num_snapshots=2000)
    assert from_snapshots.num_snapshots == shadow_direct.num_snapshots
    assert np.allclose(
        from_snapshots.expectations, shadow_direct.expectations, atol=0.25
    )


def test_reconstruct_from_snapshots_validates_shape() -> None:
    shadow = ClassicalShadow(2)
    with pytest.raises(ValueError, match="at least one snapshot"):
        shadow.reconstruct_from_snapshots([])
    with pytest.raises(ValueError, match="qubits"):
        shadow.reconstruct_from_snapshots([Snapshot(basis=(0,), outcomes=(1,))])


def test_expectation_rejects_wrong_length_pauli() -> None:
    state = Circuit(2).h(0).state_tensor().reshape(-1)
    result = ClassicalShadow(2, seed=0).reconstruct(state, num_snapshots=50)
    with pytest.raises(ValueError, match="factors"):
        result.expectation("X")


def test_classical_shadow_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ClassicalShadow(0)


def test_take_snapshots_rejects_nonpositive_count() -> None:
    state = Circuit(1).h(0).state_tensor().reshape(-1)
    with pytest.raises(ValueError, match="positive"):
        take_snapshots(state, 1, 0)
