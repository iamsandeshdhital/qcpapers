"""
Tests for the core gate library and circuit simulator.

Every test uses an oracle that is independent of the implementation: dense
matrix unitarity, the closed-form QFT matrix, or ``numpy.linalg`` on the same
object. Nothing here asserts against a previously recorded output.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcpapers.core.circuit import Circuit, Operation, StateVectorResult
from qcpapers.core.gates import (
    ALL_GATES,
    CCX,
    CNOT,
    CSWAP,
    I2,
    X,
    Y,
    Z,
    apply_gate,
    basis_state,
    controlled_gate,
    expected_param_count,
    from_tensor,
    gate_arity,
    gate_inverse_sequence,
    gate_matrix,
    is_clifford,
    to_tensor,
)
from qcpapers.core.observables import (
    SparsePauliOp,
    commute,
    pauli_labels,
    pauli_matrix_product,
    pauli_string_matrix,
    pauli_weight,
)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_GATES)
def test_every_gate_is_unitary(name: str) -> None:
    n = gate_arity(name)
    params = [0.4 + 0.1 * k for k in range(expected_param_count(name))]
    matrix = gate_matrix(name, *params)
    assert matrix.shape == (2**n, 2**n)
    identity = np.eye(2**n)
    assert np.allclose(matrix @ matrix.conj().T, identity, atol=1e-9)


@pytest.mark.parametrize("name", ALL_GATES)
def test_every_gate_inverse_is_correct(name: str) -> None:
    """The decomposition in ``gate_inverse_sequence`` must equal the adjoint."""
    n = gate_arity(name)
    params = [0.3 + 0.1 * k for k in range(expected_param_count(name))]
    matrix = gate_matrix(name, *params)

    inverse = Circuit(n, name="inverse")
    for gate, qubits, gate_params in gate_inverse_sequence(
        name, params, list(range(n))
    ):
        inverse.add(gate, list(qubits), *gate_params)

    assert np.allclose(inverse.unitary(), matrix.conj().T, atol=1e-9)


def test_iswap_inverse_uses_z_corrections() -> None:
    """``ISWAP**2 = Z (x) Z``, so its inverse needs two extra Z gates."""
    sequence = gate_inverse_sequence("ISWAP", (), [0, 1])
    assert [entry[0] for entry in sequence] == ["ISWAP", "Z", "Z"]


def test_u3_inverse_swaps_euler_angles() -> None:
    """``U3(t, p, l)^dg = U3(-t, -l, -p)``; the two Z angles exchange."""
    sequence = gate_inverse_sequence("U", (0.1, 0.2, 0.3), [0])
    assert sequence == [("U", (0,), (-0.1, -0.3, -0.2))]


def test_dcx_is_not_self_inverse() -> None:
    """``DCX**2 = X (x) X``, so it must not be treated as self-inverse."""
    matrix = gate_matrix("DCX")
    assert not np.allclose(matrix @ matrix, np.eye(4))
    assert np.allclose(matrix @ matrix, np.kron(X, X))


def test_t_and_tdg_are_clifford() -> None:
    """``T`` is a Clifford gate; omitting it breaks stabilizer simulation."""
    assert is_clifford("T")
    assert is_clifford("TDG")
    assert is_clifford("H")
    assert not is_clifford("RX")


def test_controlled_rotations_match_projector_definition() -> None:
    """``CR_P = |0><0| (x) I + |1><1| (x) R_P``."""
    p0 = np.diag([1.0, 0.0]).astype(complex)
    p1 = np.diag([0.0, 1.0]).astype(complex)
    for name, rotation in (
        ("CRX", gate_matrix("RX", 0.7)),
        ("CRY", gate_matrix("RY", 0.7)),
        ("CRZ", gate_matrix("RZ", 0.7)),
    ):
        expected = np.kron(p0, I2) + np.kron(p1, rotation)
        assert np.allclose(gate_matrix(name, 0.7), expected, atol=1e-9)


def test_cswap_is_a_permutation_of_basis_states() -> None:
    """``CSWAP`` maps ``|110> -> |101>``: control is qubit 0, targets are 1 and 2."""
    circuit = Circuit(3).x(0).x(1).cswap(0, 1, 2)
    assert circuit.run().top_states(1)[0][0] == "101"


def test_aliases_resolve_to_canonical_gates() -> None:
    """Vendor spellings map onto the same matrix as the canonical name.

    Each pair is checked with the canonical gate's own parameter count, since a
    fixed gate such as ``CX`` legitimately takes none.
    """
    for alias, canonical in (
        ("CX", "CNOT"),
        ("TOFFOLI", "CCX"),
        ("FSWAP", "CSWAP"),
        ("PHASE", "P"),
        ("CXX", "CRX"),
        ("CZZ", "CRZ"),
        ("ZZ", "RZZ"),
        ("XX", "RXX"),
    ):
        params = [0.3] * expected_param_count(canonical)
        assert np.allclose(
            gate_matrix(alias, *params), gate_matrix(canonical, *params), atol=1e-9
        ), alias


def test_wrong_parameter_count_is_reported_clearly() -> None:
    """A fixed gate called with a parameter must not say "unknown gate"."""
    with pytest.raises(ValueError, match="expects 0 parameter"):
        gate_matrix("CNOT", 0.3)
    with pytest.raises(ValueError, match="expects 1 parameter"):
        gate_matrix("RZ")


def test_unknown_gate_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError, match="unknown gate"):
        gate_matrix("NOTAGATE")


def test_controlled_gate_rejects_non_unitary() -> None:
    with pytest.raises(ValueError, match="unitary"):
        controlled_gate(np.ones((2, 2)), 1)


def test_controlled_gate_construction() -> None:
    """One control on a 1-qubit gate is a CNOT; two controls give a Toffoli."""
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(controlled_gate(x_matrix, 1), CNOT, atol=1e-9)
    assert np.allclose(controlled_gate(x_matrix, 2), CCX, atol=1e-9)
    # A controlled CNOT is the Toffoli as well.
    assert np.allclose(controlled_gate(CNOT, 1), CCX, atol=1e-9)


def test_three_control_gates_are_unitary() -> None:
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    four_fold_x = controlled_gate(x_matrix, 3)
    assert four_fold_x.shape == (16, 16)
    assert np.allclose(four_fold_x @ four_fold_x.conj().T, np.eye(16), atol=1e-9)
    # X**4 = I, so four controls give a self-inverse gate.
    assert np.allclose(four_fold_x @ four_fold_x, np.eye(16), atol=1e-9)


def test_three_control_x_fires_only_on_all_ones() -> None:
    """The three-control X must be inert unless every control qubit is set."""
    controlled = controlled_gate(np.array([[0, 1], [1, 0]], dtype=complex), 3)
    assert np.allclose(controlled[15, 14], 1.0)  # |1110> -> |1111>
    assert np.allclose(controlled[7, 7], 1.0)  # |0111> unchanged


# ---------------------------------------------------------------------------
# Circuit construction and simulation
# ---------------------------------------------------------------------------


def test_bell_state_probabilities() -> None:
    result = Circuit(2).h(0).cx(0, 1).run()
    assert isinstance(result, StateVectorResult)
    assert np.isclose(result.probabilities[0], 0.5)
    assert np.isclose(result.probabilities[3], 0.5)
    assert np.isclose(result.probabilities.sum(), 1.0)


def test_ghz_state_entangles_all_qubits() -> None:
    circuit = Circuit(3).h(0)
    for q in (1, 2):
        circuit.cx(0, q)
    probs = circuit.probabilities()
    assert np.isclose(probs[0], 0.5)
    assert np.isclose(probs[7], 0.5)
    assert np.isclose(probs[1:7].sum(), 0.0, atol=1e-12)


def test_cnot_flips_target_conditionally() -> None:
    result = Circuit(2).x(0).cx(0, 1).run()
    assert result.top_states(1)[0][0] == "11"


def test_most_significant_bit_is_qubit_zero() -> None:
    """``x(0)`` on a 3-qubit register must give ``|100>``, not ``|001>``."""
    assert Circuit(3).x(0).run().top_states(1)[0][0] == "100"
    assert Circuit(3).x(2).run().top_states(1)[0][0] == "001"


def test_state_is_normalised_after_random_circuit() -> None:
    rng = np.random.default_rng(12345)
    circuit = Circuit(4)
    for _ in range(60):
        q, r = rng.integers(0, 4, 2)
        circuit.ry(float(rng.random() * 6.28), int(q))
        a, b = rng.choice(4, 2, replace=False)
        circuit.cz(int(a), int(b))
    result = circuit.run()
    assert np.isclose(np.linalg.norm(result.state), 1.0, atol=1e-10)
    assert np.isclose(result.probabilities.sum(), 1.0, atol=1e-12)


def test_circuit_inverse_is_adjoint() -> None:
    circuit = (
        Circuit(3)
        .h(0)
        .cx(0, 1)
        .iswap(1, 2)
        .rz(0.3, 2)
        .t(0)
        .ccx(0, 1, 2)
        .dcx(1, 2)
        .u(0.1, 0.2, 0.3, 1)
    )
    adjoint = circuit.inverse()
    assert np.allclose(adjoint.unitary(), circuit.unitary().conj().T, atol=1e-9)
    assert np.allclose(adjoint.inverse().unitary(), circuit.unitary(), atol=1e-9)
    assert np.allclose(adjoint.unitary() @ circuit.unitary(), np.eye(8), atol=1e-9)


def test_compose_matches_matrix_product() -> None:
    first = Circuit(2).h(0)
    second = Circuit(2).cx(0, 1).x(0)
    assert np.allclose(first.compose(second).unitary(), second.unitary() @ first.unitary())


def test_unitary_gate_applies_dense_matrix() -> None:
    """A CNOT attached as a dense unitary behaves like the named gate."""
    circuit = Circuit(2).x(0)
    circuit.unitary_gate(CNOT, [0, 1])
    assert circuit.run().top_states(1)[0][0] == "11"


def test_controlled_unitary_gates_only_when_control_set() -> None:
    off = Circuit(2).controlled_unitary(X, 0, [1])
    on = Circuit(2).x(0).controlled_unitary(X, 0, [1])
    assert off.run().top_states(1)[0][0] == "00"
    assert on.run().top_states(1)[0][0] == "11"


def test_oracle_diagonal_marks_one_basis_state() -> None:
    phases = np.array([1.0, -1.0, 1.0, 1.0], dtype=complex)
    circuit = Circuit(2).h(0).h(1).oracle_diagonal(phases, [0, 1])
    result = circuit.run()
    assert result.top_states(1)[0][0] == "11"


# ---------------------------------------------------------------------------
# Symbolic parameters
# ---------------------------------------------------------------------------


def test_symbolic_parameters_are_bound_by_name() -> None:
    circuit = Circuit(2).ry("theta", 0).cx(0, 1).rz("phi", 1)
    assert circuit.parameter_names() == ["phi", "theta"]
    bound = circuit.assign({"theta": np.pi / 2, "phi": 0.0})
    assert bound.num_parameters() == 0
    # RY(pi/2)|0> = |+>, so a superposition of |00> and |10> (qubit 0 is the most
    # significant bit). The CNOT then correlates qubit 1 with qubit 0, turning
    # |10> into |11> and leaving the support on |00> and |11>.
    probs = bound.probabilities()
    assert np.isclose(probs[0], 0.5)
    assert np.isclose(probs[3], 0.5)
    assert np.isclose(probs[1], 0.0)
    assert np.isclose(probs[2], 0.0)


def test_unbound_parameters_raise_on_simulation() -> None:
    with pytest.raises(ValueError, match="unbound parameter"):
        Circuit(1).ry("theta", 0).run()


def test_assign_array_uses_sorted_order() -> None:
    circuit = Circuit(1).ry("b", 0).rz("a", 0)
    bound = circuit.assign_array([0.0, np.pi / 2])
    assert np.isclose(abs(bound.run().state[1]), np.sin(np.pi / 4))


def test_missing_parameter_is_reported() -> None:
    with pytest.raises(KeyError, match="no value supplied"):
        Circuit(1).ry("theta", 0).assign({"other": 1.0})


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def test_pauli_string_matrix_matches_kron() -> None:
    assert np.allclose(pauli_string_matrix("XZ"), np.kron(X, Z))
    assert np.allclose(pauli_string_matrix("IZ"), np.kron(I2, Z))


def test_bell_state_pauli_expectations() -> None:
    circuit = Circuit(2).h(0).cx(0, 1)
    assert np.isclose(circuit.expectation("ZZ"), 1.0)
    assert np.isclose(circuit.expectation("XX"), 1.0)
    assert np.isclose(circuit.expectation("YY"), -1.0)
    assert np.isclose(circuit.expectation("ZI"), 0.0, atol=1e-12)


def test_pauli_algebra_anticommutation() -> None:
    assert commute("XX", "YY")
    assert commute("XI", "XI")
    assert not commute("XI", "ZZ")
    assert pauli_weight("IXZI") == 2


def test_sparse_pauli_op_roundtrip_from_matrix() -> None:
    hamiltonian = SparsePauliOp([(0.5, "XI"), (-0.5, "IX"), (-0.5, "ZZ")])
    matrix = hamiltonian.to_matrix()
    recovered = SparsePauliOp.from_matrix(matrix)
    assert np.allclose(recovered.to_matrix(), matrix, atol=1e-9)


def test_sparse_pauli_op_ground_energy_matches_diagonalisation() -> None:
    hamiltonian = SparsePauliOp([(0.5, "XI"), (-0.5, "IX"), (-0.5, "ZZ")])
    exact = np.linalg.eigvalsh(hamiltonian.to_matrix())[0]
    assert np.isclose(hamiltonian.ground_energy_exact(), exact)


def test_hamiltonian_expectation_on_product_state() -> None:
    hamiltonian = SparsePauliOp([(0.5, "XI"), (-0.5, "IX"), (-0.5, "ZZ")])
    # |00> is not an eigenstate; check against the dense matrix product.
    state = from_tensor(Circuit(2).state_tensor())
    dense = state.conj() @ hamiltonian.to_matrix() @ state
    assert np.isclose(Circuit(2).hamiltonian_expectation(hamiltonian), dense.real)


def test_commuting_grouping_partitions_terms() -> None:
    terms = [
        (1.0, "XII"),
        (2.0, "IIX"),
        (3.0, "ZII"),
        (4.0, "IIZ"),
        (5.0, "YIY"),
        (6.0, "IZI"),
    ]
    hamiltonian = SparsePauliOp(terms)
    grouped, groups = hamiltonian.group_commuting(return_groups=True)
    # Every original term appears in exactly one group, and group members commute.
    assert sorted(sum(groups, [])) == sorted(p for _, p in terms)
    for group in groups:
        for a in group:
            for b in group:
                assert commute(a, b)
    assert grouped.size == hamiltonian.size


def test_pauli_fidelities_of_bell_state() -> None:
    circuit = Circuit(2).h(0).cx(0, 1)
    values = circuit.state_tensor()
    labels = pauli_labels(2)
    index = {label: i for i, label in enumerate(labels)}
    fidelities = SparsePauliOp.identity(2).pauli_fidelities(values)
    assert np.isclose(fidelities[index["II"]], 1.0)
    assert np.isclose(fidelities[index["ZZ"]], 1.0)
    assert np.isclose(fidelities[index["XX"]], 1.0)
    assert np.isclose(fidelities[index["YY"]], -1.0)
    assert np.isclose(fidelities[index["ZI"]], 0.0, atol=1e-12)


def test_expectation_from_counts_matches_exact() -> None:
    """Counts in the Z basis must reproduce the exact Pauli expectation."""
    circuit = Circuit(2).h(0).cx(0, 1)
    hamiltonian = SparsePauliOp([(0.5, "ZI"), (-0.5, "IZ"), (-0.5, "ZZ")])
    exact = circuit.hamiltonian_expectation(hamiltonian)
    counts = circuit.counts(shots=200_000, seed=7)
    assert np.isclose(hamiltonian.expectation_from_counts(counts), exact, atol=0.01)


def test_sparse_pauli_op_rejects_mixed_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        SparsePauliOp([(1.0, "XX"), (1.0, "Z")])


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def test_depth_of_linear_chain() -> None:
    circuit = Circuit(3).h(0).cx(0, 1).cx(1, 2)
    assert circuit.depth() == 3


def test_depth_of_parallel_layers() -> None:
    circuit = Circuit(4).h(0).h(1).h(2).h(3).cx(0, 1).cx(2, 3)
    assert circuit.depth() == 2


def test_count_ops_tallies_gates() -> None:
    circuit = Circuit(2).h(0).h(1).cx(0, 1)
    assert circuit.count_ops() == {"H": 2, "CNOT": 1}


def test_out_of_range_qubit_is_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        Circuit(2).x(5)


def test_wrong_arity_is_rejected() -> None:
    with pytest.raises(ValueError, match="acts on 2 qubit"):
        Circuit(2).add("CNOT", [0, 1, 0])
