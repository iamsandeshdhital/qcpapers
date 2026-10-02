"""
Tests for the variational quantum eigensolver and its ansätze.

Every gradient is validated against a central finite difference of the energy,
which is independent of both the tangent-state and parameter-shift
implementations. Energies are validated against exact diagonalisation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcpapers.algorithms.vqe import (
    PARAMETER_SHIFT,
    EfficientSU2Ansatz,
    ParticleNumberAnsatz,
    VQE,
    optimise_energy,
    pairwise_ising_hamiltonian,
    parameter_shift_gradient,
)
from qcpapers.core.circuit import Circuit
from qcpapers.core.observables import SparsePauliOp


def finite_difference_gradient(vqe: VQE, step: float = 1e-6) -> np.ndarray:
    """Central difference of the energy, independent of the analytic gradient."""
    gradient = []
    for index in range(len(vqe.parameter_names)):
        plus = vqe.initial_parameters.copy()
        minus = vqe.initial_parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient.append((vqe.energy(plus) - vqe.energy(minus)) / (2 * step))
    return np.array(gradient)


# ---------------------------------------------------------------------------
# Gate-level derivatives
# ---------------------------------------------------------------------------


PARAMETRIC_GATES = ("RX", "RY", "RZ", "P", "U", "RXX", "RYY", "RZZ", "RZX", "CRX", "CRY", "CRZ", "CP")


@pytest.mark.parametrize("name", PARAMETRIC_GATES)
def test_gate_parameter_derivative_matches_finite_difference(name: str) -> None:
    """``dU/dp`` must be the exact derivative of the gate matrix."""
    from qcpapers.core.gates import (
        expected_param_count,
        gate_matrix,
        gate_parameter_derivative,
    )

    count = expected_param_count(name)
    for index in range(count):
        for value in (0.3, 0.9, 1.7):
            params = [0.4 + 0.1 * k for k in range(count)]
            params[index] = value
            analytic = gate_parameter_derivative(name, params, index)

            step = 1e-6
            plus = list(params)
            minus = list(params)
            plus[index] += step
            minus[index] -= step
            numeric = (
                gate_matrix(name, *plus) - gate_matrix(name, *minus)
            ) / (2 * step)

            assert np.allclose(analytic, numeric, atol=1e-7), (name, index, value)


def test_tangent_state_derivatives_match_finite_difference() -> None:
    """Tangent-state derivatives are correct even for a shared parameter.

    The parameter ``shared`` drives ``RY`` on two qubits and an ``RXX``. The
    parameter-shift rule cannot handle this; the tangent method can.
    """
    circuit = Circuit(2)
    circuit.ry("shared", 0)
    circuit.ry("shared", 1)
    circuit.u("t", "p", "l", 0)
    circuit.cx(0, 1)
    circuit.rz("l", 1)
    circuit.rxx("shared", 0, 1)

    names = circuit.parameter_names()
    values = {"l": -0.4, "p": 0.7, "shared": 0.3, "t": 1.1}
    ordered = [values[name] for name in names]

    state, tangents = circuit.state_and_derivatives(ordered)
    assert np.allclose(state.reshape(-1), circuit.assign_array(ordered).run().state)

    for index, name in enumerate(names):
        step = 1e-6
        plus = list(ordered)
        minus = list(ordered)
        plus[index] += step
        minus[index] -= step
        numeric = (
            circuit.assign_array(plus).run().state
            - circuit.assign_array(minus).run().state
        ) / (2 * step)
        assert np.allclose(tangents[name].reshape(-1), numeric, atol=1e-7), name


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_tangent_gradient_matches_finite_difference() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    vqe = VQE(hamiltonian, EfficientSU2Ansatz(3, reps=2))
    assert np.allclose(
        vqe.gradient(vqe.initial_parameters),
        finite_difference_gradient(vqe),
        atol=1e-6,
    )


def test_parameter_shift_agrees_when_every_parameter_is_independent() -> None:
    """The two gradient routes must agree for a circuit the shift rule supports."""
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    vqe = VQE(hamiltonian, EfficientSU2Ansatz(3, reps=2))
    assert np.allclose(
        vqe.gradient(vqe.initial_parameters),
        vqe.gradient_by_parameter_shift(vqe.initial_parameters),
        atol=1e-9,
    )


def test_parameter_shift_fails_for_tied_parameters() -> None:
    """Pins the failure mode that motivates the tangent-state gradient.

    ``ParticleNumberAnsatz`` drives ``RXX`` and ``RYY`` from one angle, so the
    effective generator is ``(XX + YY)/2`` with eigenvalues ``+-1`` instead of
    ``+-1/2``. The parameter-shift amount must then be ``pi``, not ``pi/2``, and
    using ``pi/2`` returns a wrong gradient that still looks reasonable.
    """
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    vqe = VQE(hamiltonian, ParticleNumberAnsatz(3, 2, layers=2))

    numeric = finite_difference_gradient(vqe)
    tangent = vqe.gradient(vqe.initial_parameters)
    shifted = vqe.gradient_by_parameter_shift(vqe.initial_parameters)

    assert np.allclose(tangent, numeric, atol=1e-6)
    assert not np.allclose(shifted, numeric, atol=1e-3)


def test_parameter_shift_rule_uses_a_full_half_pi_shift() -> None:
    """A bare single-qubit rotation must use ``pi/2``, not ``pi/4``.

    The objective is ``P(1) = cos^2(theta/2)`` for ``RY(theta)|1>``, whose
    derivative is ``-sin(theta) / 2``.
    """
    circuit = Circuit(1).x(0).ry("theta", 0)

    def energy(parameters: list[float]) -> float:
        probabilities = circuit.assign_array(parameters).probabilities()
        return float(probabilities[1])

    gradient = parameter_shift_gradient(energy, [0.3])
    numeric = (energy([0.3 + 1e-6]) - energy([0.3 - 1e-6])) / 2e-6

    assert gradient[0] == pytest.approx(numeric, abs=1e-6)
    assert gradient[0] == pytest.approx(-math.sin(0.3) / 2, abs=1e-6)
    assert PARAMETER_SHIFT == pytest.approx(math.pi / 2)


def test_parameter_shift_divisor_is_two_not_twice_the_shift() -> None:
    """Pins the coefficient: the rule divides by 2, not by ``2 * shift``.

    Dividing by ``2 * shift`` is a natural-looking mistake and gives a gradient
    smaller by a factor ``pi/2``, which still descends but converges slowly and
    looks like a badly conditioned problem.
    """
    circuit = Circuit(1).x(0).ry("theta", 0)

    def energy(parameters: list[float]) -> float:
        return float(circuit.assign_array(parameters).probabilities()[1])

    correct = parameter_shift_gradient(energy, [0.3], shift=math.pi / 2)[0]
    wrong = (energy([0.3 + math.pi / 2]) - energy([0.3 - math.pi / 2])) / math.pi
    assert correct == pytest.approx(wrong * (math.pi / 2), rel=1e-9)


# ---------------------------------------------------------------------------
# Ansätze
# ---------------------------------------------------------------------------


def test_ising_hamiltonian_matches_explicit_pauli_terms() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    assert hamiltonian.size == 2
    assert set(p for _, p in hamiltonian.terms) == {"ZZI", "IZZ"}


def test_ising_hamiltonian_combines_duplicate_edges() -> None:
    """A repeated edge doubles its coefficient rather than appearing twice."""
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 0)], n_qubits=2)
    assert hamiltonian.size == 1
    assert hamiltonian.terms[0][1] == "ZZ"
    assert hamiltonian.terms[0][0].real == pytest.approx(2.0)


def test_ising_hamiltonian_adds_field_terms() -> None:
    hamiltonian = pairwise_ising_hamiltonian([], n_qubits=3, field_strength=0.4)
    assert hamiltonian.size == 3
    assert hamiltonian.ground_energy_exact() == pytest.approx(-1.2)


def test_ising_hamiltonian_rejects_self_loops() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        pairwise_ising_hamiltonian([(1, 1)], n_qubits=2)


def test_efficient_su2_parameter_count() -> None:
    """``reps`` blocks of ``n_qubits`` rotations on two axes."""
    ansatz = EfficientSU2Ansatz(4, reps=3, rotation_axes=("Y", "Z"))
    assert len(ansatz.parameter_names()) == 4 * 2 * 3


def test_efficient_su2_skip_final_rotation_reduces_parameters() -> None:
    full = EfficientSU2Ansatz(3, reps=2)
    trimmed = EfficientSU2Ansatz(3, reps=2, skip_final_rotation=True)
    assert len(trimmed.parameter_names()) < len(full.parameter_names())


def test_efficient_su2_entangle_topologies() -> None:
    assert EfficientSU2Ansatz(4, entangler="linear").pairs() == [(0, 1), (1, 2), (2, 3)]
    assert EfficientSU2Ansatz(4, entangler="ring").pairs()[-1] == (3, 0)
    assert len(EfficientSU2Ansatz(4, entangler="full").pairs()) == 6
    # A two-qubit ring degenerates to the linear case.
    assert EfficientSU2Ansatz(2, entangler="ring").pairs() == [(0, 1)]


def test_efficient_su2_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="entangler"):
        EfficientSU2Ansatz(2, entangler="triangle")
    with pytest.raises(ValueError, match="Pauli axis"):
        EfficientSU2Ansatz(2, rotation_axes=("Q",))
    with pytest.raises(ValueError, match="initial_state"):
        EfficientSU2Ansatz(2, initial_state="random")


def test_efficient_su2_uniform_initial_state_is_the_maximally_mixed_state() -> None:
    """``H**n |0>`` puts equal weight on every basis state.

    The rotations must still be bound, otherwise this measures the initial state
    and not the prepared one.
    """
    ansatz = EfficientSU2Ansatz(3, reps=1, initial_state="uniform")
    circuit = ansatz.circuit()
    assert circuit.num_parameters() > 0
    bound = circuit.assign_array(np.zeros(circuit.num_parameters()))
    probabilities = bound.probabilities()
    # Rotations at angle zero are the identity, so the state stays uniform.
    assert np.allclose(probabilities, 1 / 8, atol=1e-9)


def test_particle_number_ansatz_preserves_particle_number() -> None:
    """The circuit must stay inside the fixed-particle-number subspace.

    This is the defining property of the ansatz: leaking weight into other
    particle-number sectors means the variational minimum is no longer a
    physical fermionic state, and the energy it reports is meaningless.
    """
    ansatz = ParticleNumberAnsatz(4, n_particles=2, layers=2)
    rng = np.random.default_rng(0)
    values = list(rng.random(len(ansatz.parameter_names())) * 2)
    probabilities = ansatz.circuit().assign_array(values).probabilities()

    weights: dict[int, float] = {}
    for index, probability in enumerate(probabilities):
        if probability > 1e-12:
            weight = bin(index).count("1")
            weights[weight] = weights.get(weight, 0.0) + float(probability)

    assert set(weights) == {2}
    assert weights[2] == pytest.approx(1.0, abs=1e-9)


def test_particle_number_ansatz_shares_one_angle_between_rxx_and_ryy() -> None:
    """``RXX`` and ``RYY`` must be driven by one parameter, not two.

    With independent angles the product no longer annihilates ``|00>`` and
    ``|11>``, and probability leaks out of the target subspace.
    """
    circuit = ParticleNumberAnsatz(3, n_particles=1, layers=1).circuit()
    names = circuit.parameter_names()
    hopping = [n for n in names if n.startswith("h0_")]
    assert hopping, "expected paired hopping parameters"
    # Each pair contributes one parameter: 3 choose 2 = 3 pairs.
    assert len(hopping) == 3


def test_particle_number_ansatz_rejects_impossible_particle_count() -> None:
    with pytest.raises(ValueError, match="n_particles"):
        ParticleNumberAnsatz(3, n_particles=5)


# ---------------------------------------------------------------------------
# VQE correctness
# ---------------------------------------------------------------------------


def test_vqe_finds_the_exact_ground_energy_on_a_chain() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    result = VQE(hamiltonian, EfficientSU2Ansatz(3, reps=2)).run(max_iterations=150)
    assert result.energy == pytest.approx(hamiltonian.ground_energy_exact(), abs=1e-6)
    # The variational principle forbids beating the true ground energy.
    assert result.energy_error >= -1e-9
    assert result.ground_state_overlap > 0.99


def test_vqe_finds_the_exact_ground_energy_with_a_field() -> None:
    hamiltonian = pairwise_ising_hamiltonian(
        [(0, 1), (1, 2), (2, 3), (3, 0)], n_qubits=4, field_strength=0.3
    )
    result = VQE(hamiltonian, EfficientSU2Ansatz(4, reps=3)).run(max_iterations=200)
    assert result.energy == pytest.approx(hamiltonian.ground_energy_exact(), abs=1e-6)
    assert result.ground_state_overlap > 0.99


def test_vqe_never_beats_the_exact_ground_energy() -> None:
    """The variational principle is a hard constraint, not a soft target.

    A shallow ansatz cannot represent the ground state, so the energy must come
    out *above* the exact value. Getting below it would mean the circuit or the
    Hamiltonian is wrong.
    """
    hamiltonian = pairwise_ising_hamiltonian(
        [(0, 1), (1, 2), (2, 3), (3, 0)], n_qubits=4, field_strength=0.3
    )
    shallow = VQE(hamiltonian, EfficientSU2Ansatz(4, reps=1)).run(max_iterations=150)
    assert shallow.energy >= shallow.exact_energy - 1e-9
    assert shallow.energy_error > 1e-6


def test_vqe_energy_decreases_monotonically() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    result = VQE(hamiltonian, EfficientSU2Ansatz(3, reps=2)).run(max_iterations=100)
    history = np.array(result.history)
    assert np.all(np.diff(history) <= 1e-12)


def test_vqe_reaches_the_ground_state_with_the_fermionic_ansatz() -> None:
    """The number-preserving ansatz must also find the exact minimum.

    On three qubits with two particles the accessible subspace is spanned by
    ``|011>``, ``|101>`` and ``|110>``, whose restricted Hamiltonian has ground
    energy ``-2``.
    """
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    result = VQE(
        hamiltonian, ParticleNumberAnsatz(3, n_particles=2, layers=2)
    ).run(max_iterations=300)
    assert result.energy == pytest.approx(-2.0, abs=1e-6)


def test_vqe_rejects_mismatched_hamiltonian_and_ansatz() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1)], n_qubits=2)
    with pytest.raises(ValueError, match="qubits"):
        VQE(hamiltonian, EfficientSU2Ansatz(3))


def test_vqe_rejects_wrong_parameter_vector_length() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1)], n_qubits=2)
    vqe = VQE(hamiltonian, EfficientSU2Ansatz(2, reps=1))
    with pytest.raises(ValueError, match="expected"):
        vqe.energy([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="expected"):
        VQE(hamiltonian, EfficientSU2Ansatz(2, reps=1), initial_parameters=[0.1])


def test_vqe_without_exact_reference_reports_nan() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1)], n_qubits=2)
    result = VQE(
        hamiltonian, EfficientSU2Ansatz(2, reps=1), exact_reference=False
    ).run(max_iterations=20)
    assert math.isnan(result.exact_energy)
    assert math.isnan(result.ground_state_overlap)


def test_overlap_handles_degenerate_ground_states() -> None:
    """A degenerate ground state must still give overlap 1.

    Comparing against a single eigenvector reports ~0 for a state that is a
    superposition inside the degenerate subspace, which looks like a broken
    circuit when nothing is wrong.
    """
    # A 4-cycle with no field has a twofold-degenerate ground state.
    hamiltonian = pairwise_ising_hamiltonian(
        [(0, 1), (1, 2), (2, 3), (3, 0)], n_qubits=4
    )
    eigenvalues = np.linalg.eigvalsh(hamiltonian.to_matrix())
    assert np.count_nonzero(eigenvalues <= eigenvalues[0] + 1e-9) >= 2

    result = VQE(hamiltonian, EfficientSU2Ansatz(4, reps=3)).run(max_iterations=300)
    assert result.ground_state_overlap > 0.99


def test_energy_is_evaluated_without_shot_noise() -> None:
    """Repeated evaluations at one point must return identical values."""
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    vqe = VQE(hamiltonian, EfficientSU2Ansatz(3, reps=2))
    values = {vqe.energy(vqe.initial_parameters) for _ in range(5)}
    assert len(values) == 1


def test_optimise_energy_helper_matches_the_class() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    ansatz = EfficientSU2Ansatz(3, reps=2)
    direct = VQE(hamiltonian, ansatz).run(max_iterations=100)
    helper = optimise_energy(hamiltonian, ansatz, max_iterations=100)
    assert helper.energy == pytest.approx(direct.energy, abs=1e-9)


def test_vqe_handles_a_hamiltonian_with_a_single_term() -> None:
    """``ZZ`` alone is diagonal, so its ground state is computable by hand."""
    hamiltonian = SparsePauliOp([(1.0, "ZZ")])
    result = VQE(hamiltonian, EfficientSU2Ansatz(2, reps=2)).run(max_iterations=150)
    assert result.energy == pytest.approx(-1.0, abs=1e-6)


def test_vqe_result_repr_contains_the_error() -> None:
    hamiltonian = pairwise_ising_hamiltonian([(0, 1)], n_qubits=2)
    result = VQE(hamiltonian, EfficientSU2Ansatz(2, reps=1)).run(max_iterations=20)
    assert "error=" in repr(result)
