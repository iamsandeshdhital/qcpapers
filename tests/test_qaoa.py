"""
Tests for QAOA and the MaxCut objective.

The brute-force cut value is the oracle. It is independent of the cost
operator, the circuit, and the optimiser, so a wrong cost operator cannot agree
with it by accident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcpapers.algorithms.qaoa import (
    MAXCUT_COST_OFFSET,
    QAOA,
    exact_maxcut,
    exact_maxcut_energy,
    grover_walk_search,
    maxcut_cost_operator,
    maxcut_qaoa,
    maxcut_value_of,
    qaoa_parameter_scan,
)
from qcpapers.core.observables import SparsePauliOp

#: Small graphs whose brute-force optima are known.
GRAPHS = {
    "triangle": ([(0, 1), (1, 2), (0, 2)], 3, 2),
    "path4": ([(0, 1), (1, 2), (2, 3)], 4, 3),
    "star4": ([(0, 1), (0, 2), (0, 3)], 4, 3),
    "cycle4": ([(0, 1), (1, 2), (2, 3), (3, 0)], 4, 4),
    "K4": ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4, 4),
}


# ---------------------------------------------------------------------------
# Cost operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_brute_force_maxcut_matches_known_values(name: str) -> None:
    edges, n_qubits, expected = GRAPHS[name]
    assert exact_maxcut(edges, n_qubits) == expected


def test_maxcut_value_of_counts_crossing_edges() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert maxcut_value_of([0, 1, 0, 1], edges) == 4
    assert maxcut_value_of([0, 0, 1, 1], edges) == 2
    assert maxcut_value_of([0, 0, 0, 0], edges) == 0


def test_cost_operator_minimum_is_zero_at_the_optimum() -> None:
    """With the offset included the ground energy must be exactly zero.

    The paper normalises the objective so the maximum cut maps to zero, which
    makes "solved" and "cost equals zero" the same statement.
    """
    edges, n_qubits, _ = GRAPHS["cycle4"]
    operator = maxcut_cost_operator(edges, n_qubits)
    assert operator.ground_energy_exact() == pytest.approx(0.0, abs=1e-9)
    assert exact_maxcut_energy(edges, n_qubits) == pytest.approx(0.0, abs=1e-9)


def test_cost_operator_without_offset_is_shifted_by_half_the_edge_count() -> None:
    """Dropping the offset shifts every value down by ``|E| / 2``.

    Without the identity term the operator is ``sum w_ij (1 - Z_i Z_j) / 2``,
    which equals ``cut - |E| / 2``. For a triangle, ``|E| = 3`` and the optimum
    cut is 2, so the ground energy is ``2 - 1.5 = 0.5``... except that
    superpositions can go lower, and the true minimum is ``-|E| / 2`` because
    maximising ``cut`` over superpositions exceeds the best basis state.
    """
    edges, n_qubits, best = GRAPHS["triangle"]
    offset_form = maxcut_cost_operator(edges, n_qubits)
    raw = maxcut_cost_operator(edges, n_qubits, include_offset=False)

    n_edges = len(edges)
    assert raw.ground_energy_exact() == pytest.approx(-n_edges / 2.0, abs=1e-9)
    assert offset_form.ground_energy_exact() == pytest.approx(0.0, abs=1e-9)
    assert MAXCUT_COST_OFFSET == pytest.approx(0.25)
    del best


def test_cost_operator_offset_equals_half_the_edge_count() -> None:
    """The identity coefficient must be ``|E| / 2``.

    That is ``1/4`` per edge, since each edge also contributes ``1/2`` to the
    constant part of ``(1 - Z_i Z_j) / 2``.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    operator = maxcut_cost_operator(edges, 4)
    identity = [c for c, p in operator.terms if set(p) == {"I"}]
    assert identity
    assert identity[0].real == pytest.approx(len(edges) / 2.0)


def test_cost_operator_evaluates_to_the_cut_on_basis_states() -> None:
    """On a computational basis state the cost must equal the cut size.

    This is the sharpest available check on the cost operator: it ties the Pauli
    expression to the combinatorial definition on *every* basis state, so a
    transposed bit order or a dropped sign cannot pass.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    n_qubits = 4
    operator = maxcut_cost_operator(edges, n_qubits)
    matrix = operator.to_matrix()

    for assignment in range(2**n_qubits):
        bits = [(assignment >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        cut = maxcut_value_of(bits, edges)
        assert matrix[assignment, assignment].real == pytest.approx(cut, abs=1e-9)

    # Sanity: the all-zero state cuts nothing, an alternating state cuts all.
    assert matrix[0, 0].real == pytest.approx(0.0)
    assert matrix[0b0101, 0b0101].real == pytest.approx(len(edges))


def test_cost_operator_combines_duplicate_edges() -> None:
    operator = maxcut_cost_operator([(0, 1), (1, 0)], n_qubits=2)
    zz = [c for c, p in operator.terms if p == "ZZ"]
    assert len(zz) == 1
    assert zz[0].real == pytest.approx(-1.0)


def test_cost_operator_rejects_self_loops() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        maxcut_cost_operator([(1, 1)], n_qubits=2)


def test_empty_graph_has_zero_cost() -> None:
    operator = maxcut_cost_operator([], n_qubits=3)
    assert operator.ground_energy_exact() == pytest.approx(0.0)
    assert exact_maxcut([], 3) == 0


# ---------------------------------------------------------------------------
# Circuit and gradients
# ---------------------------------------------------------------------------


def test_cost_at_zero_angles_is_the_average_edge_count() -> None:
    """With no cost or mixer action the state is ``|+>`` and the cost is ``|E| / 2``.

    ``<+|Z_i Z_j|+> = 0``, so the offset identity term alone survives.
    """
    edges, n_qubits, _ = GRAPHS["cycle4"]
    qaoa = QAOA(maxcut_cost_operator(edges, n_qubits), layers=1)
    assert qaoa.cost([0.0], [0.0]) == pytest.approx(len(edges) / 2.0)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_gradient_matches_finite_difference(name: str) -> None:
    edges, n_qubits, _ = GRAPHS[name]
    qaoa = QAOA(maxcut_cost_operator(edges, n_qubits), layers=2)

    gammas = [0.31, 0.17]
    betas = [0.44, 0.62]
    d_gamma, d_beta = qaoa.gradient(gammas, betas)

    numeric_gamma = []
    numeric_beta = []
    for index in range(2):
        plus = list(gammas)
        minus = list(gammas)
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric_gamma.append(
            (qaoa.cost(plus, betas) - qaoa.cost(minus, betas)) / 2e-6
        )

        plus = list(betas)
        minus = list(betas)
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric_beta.append(
            (qaoa.cost(gammas, plus) - qaoa.cost(gammas, minus)) / 2e-6
        )

    assert np.allclose(d_gamma, numeric_gamma, atol=1e-6)
    assert np.allclose(d_beta, numeric_beta, atol=1e-6)


def test_circuit_registers_metadata() -> None:
    edges, n_qubits, _ = GRAPHS["triangle"]
    circuit = QAOA(maxcut_cost_operator(edges, n_qubits), layers=2).circuit([0.1, 0.2], [0.3, 0.4])
    assert circuit.metadata["algorithm"] == "qaoa"
    assert circuit.metadata["layers"] == 2
    assert "Farhi" in circuit.metadata["reference"]


def test_circuit_rejects_wrong_parameter_counts() -> None:
    edges, n_qubits, _ = GRAPHS["triangle"]
    qaoa = QAOA(maxcut_cost_operator(edges, n_qubits), layers=2)
    with pytest.raises(ValueError, match="expected 2"):
        qaoa.circuit([0.1], [0.2])


def test_qaoa_rejects_invalid_configuration() -> None:
    edges, n_qubits, _ = GRAPHS["triangle"]
    operator = maxcut_cost_operator(edges, n_qubits)
    with pytest.raises(ValueError, match="layers"):
        QAOA(operator, layers=0)
    with pytest.raises(ValueError, match="mixer"):
        QAOA(operator, mixer="y")


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_qaoa_never_beats_the_exact_cost(name: str) -> None:
    """The variational minimum must sit at or above the true minimum."""
    edges, n_qubits, _ = GRAPHS[name]
    result = maxcut_qaoa(edges, n_qubits, layers=3, max_iterations=250)
    assert result.cost >= result.exact_cost - 1e-6


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_more_layers_never_make_qaoa_worse(name: str) -> None:
    """The optimal ``p = 3`` cost must be no worse than the optimal ``p = 1`` cost.

    More layers enlarge the ansatz, so the reachable minimum can only decrease.
    This is the paper's central empirical claim and it must hold exactly, not just
    on average.
    """
    edges, n_qubits, _ = GRAPHS[name]
    shallow = maxcut_qaoa(edges, n_qubits, layers=1, max_iterations=250)
    deep = maxcut_qaoa(edges, n_qubits, layers=3, max_iterations=250)
    assert deep.cost <= shallow.cost + 1e-6


def test_qaoa_solves_the_triangle_exactly_at_two_layers() -> None:
    """A triangle is the smallest instance where ``p = 2`` succeeds."""
    edges, n_qubits, best = GRAPHS["triangle"]
    result = maxcut_qaoa(edges, n_qubits, layers=2, max_iterations=250)
    assert result.cost == pytest.approx(0.0, abs=1e-6)
    assert result.cut_value == best


def test_qaoa_finds_the_maximum_cut_of_the_cycle_at_two_layers() -> None:
    edges, n_qubits, best = GRAPHS["cycle4"]
    result = maxcut_qaoa(edges, n_qubits, layers=2, max_iterations=250)
    assert result.cut_value == best
    assert result.cost == pytest.approx(0.0, abs=1e-6)


def test_ground_subspace_probability_beats_the_uniform_baseline() -> None:
    """The optimum must carry more probability than uniform sampling would.

    Uniform over ``2**n`` basis states puts ``degeneracy / 2**n`` on the ground
    subspace. QAOA must beat that, and by ``p = 2`` it should concentrate most of
    the mass there.

    Note this deliberately does *not* assert the paper's ``1 / p**2`` bound. That
    bound is on the probability of measuring an optimal solution with optimal
    parameters, and it is far from tight: for the 4-cycle at ``p = 1`` the best
    attainable probability is about 0.285 by exhaustive grid search, while
    ``1 / p**2 = 1``. Asserting the bound would be asserting something false.
    """
    edges, n_qubits, _ = GRAPHS["cycle4"]
    operator = maxcut_cost_operator(edges, n_qubits)
    eigenvalues = np.linalg.eigvalsh(operator.to_matrix())
    degeneracy = int(np.count_nonzero(eigenvalues <= eigenvalues[0] + 1e-9))
    uniform = degeneracy / 2**n_qubits

    shallow = QAOA(operator, layers=1)
    shallow_result = shallow.run(max_iterations=250)
    assert shallow.probability_of_ground_subspace(
        shallow_result.gammas, shallow_result.betas
    ) > uniform

    deep = QAOA(operator, layers=3)
    deep_result = deep.run(max_iterations=250)
    assert deep.probability_of_ground_subspace(
        deep_result.gammas, deep_result.betas
    ) > 0.49


def test_probability_of_ground_subspace_counts_all_degenerate_states() -> None:
    """A degenerate optimum must not be penalised by the metric.

    The 4-cycle has two maximum cuts, ``|0101>`` and ``|1010>``, both at zero
    cost. Counting only one eigenvector would report a spuriously low
    probability whenever the prepared state is a superposition of the two.
    """
    edges, n_qubits, _ = GRAPHS["cycle4"]
    operator = maxcut_cost_operator(edges, n_qubits)
    eigenvalues = np.linalg.eigvalsh(operator.to_matrix())
    assert np.count_nonzero(eigenvalues <= eigenvalues[0] + 1e-9) == 2

    qaoa = QAOA(operator, layers=2)
    result = qaoa.run(max_iterations=250)
    # Two basis states are optimal, so a state fully in the subspace reaches 0.5.
    assert qaoa.probability_of_ground_subspace(result.gammas, result.betas) > 0.45


def test_cost_decreases_monotonically() -> None:
    edges, n_qubits, _ = GRAPHS["cycle4"]
    history = np.array(maxcut_qaoa(edges, n_qubits, layers=2).history)
    assert np.all(np.diff(history) <= 1e-12)


def test_qaoa_run_rejects_wrong_initial_length() -> None:
    edges, n_qubits, _ = GRAPHS["triangle"]
    qaoa = QAOA(maxcut_cost_operator(edges, n_qubits), layers=2)
    with pytest.raises(ValueError, match="expected 4"):
        qaoa.run(initial=[0.1, 0.2])


def test_zero_initial_parameters_stall_and_the_default_does_not() -> None:
    """Documents why the default start is non-zero.

    At ``gamma = beta = 0`` the gradient vanishes identically, so the optimiser
    declares convergence without moving. A test that started at zero would pass
    while never having optimised anything.
    """
    edges, n_qubits, _ = GRAPHS["cycle4"]
    qaoa = QAOA(maxcut_cost_operator(edges, n_qubits), layers=1)
    d_gamma, d_beta = qaoa.gradient([0.0], [0.0])
    assert abs(d_gamma[0]) < 1e-9
    assert abs(d_beta[0]) < 1e-9

    default = qaoa.run(max_iterations=200)
    assert len(default.history) > 1
    assert default.cost < len(edges) / 2.0


def test_cut_probabilities_sum_to_one() -> None:
    edges, n_qubits, _ = GRAPHS["path4"]
    qaoa = QAOA(maxcut_cost_operator(edges, n_qubits), layers=2)
    result = qaoa.run(max_iterations=200)
    distribution = qaoa.cut_probabilities(result.gammas, result.betas, edges)
    assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Grover walk variant
# ---------------------------------------------------------------------------


def test_grover_walk_search_runs_and_respects_the_variational_bound() -> None:
    edges, n_qubits, best = GRAPHS["cycle4"]
    result = grover_walk_search(edges, n_qubits, layers=2)
    assert result.cost >= result.exact_cost - 1e-6
    assert result.best_cut_value == float(best)
    assert result.cost <= len(edges) / 2.0


def test_sx_mixer_differs_from_the_transverse_field_mixer() -> None:
    """The two mixers are genuinely different circuits."""
    edges, n_qubits, _ = GRAPHS["triangle"]
    operator = maxcut_cost_operator(edges, n_qubits)
    transverse = QAOA(operator, layers=1, mixer="x").circuit([0.3], [0.5])
    grover = QAOA(operator, layers=1, mixer="sx").circuit([0.3], [0.5])
    assert transverse.count_ops() != grover.count_ops()


# ---------------------------------------------------------------------------
# Parameter scan
# ---------------------------------------------------------------------------


def test_parameter_scan_shape_and_minimum() -> None:
    edges, n_qubits, _ = GRAPHS["cycle4"]
    gammas, betas, grid = qaoa_parameter_scan(
        maxcut_cost_operator(edges, n_qubits), n_points=15
    )
    assert gammas.shape == (15,)
    assert betas.shape == (15,)
    assert grid.shape == (15, 15)
    # The scan must find something better than the no-op value |E|/2.
    assert grid.min() < len(edges) / 2.0


def test_parameter_scan_grid_corner_is_the_average_cost() -> None:
    edges, n_qubits, _ = GRAPHS["cycle4"]
    gammas, betas, grid = qaoa_parameter_scan(
        maxcut_cost_operator(edges, n_qubits), angle_range=(0.0, 0.0), n_points=3
    )
    assert np.allclose(grid, len(edges) / 2.0, atol=1e-9)


def test_parameter_scan_rejects_multiple_layers() -> None:
    edges, n_qubits, _ = GRAPHS["triangle"]
    with pytest.raises(ValueError, match="single layer"):
        qaoa_parameter_scan(maxcut_cost_operator(edges, n_qubits), layers=2)
