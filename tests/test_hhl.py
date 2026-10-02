"""
Tests for the HHL linear-solver reference implementation.

Every solution is checked against ``numpy.linalg.solve``, which shares no code
with the spectral path used here. Conditioning behaviour is checked against
``lambda_max / lambda_min``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qcpapers.algorithms.hhl import (
    HHLResult,
    conditioning_number,
    hhl_phase_estimates,
    hhl_solve,
    hermitian_part,
    lcu_decomposition,
)


def spd_matrix(rng: np.random.Generator, size: int, shift: float = 3.0) -> np.ndarray:
    """A symmetric positive-definite matrix with the requested size."""
    raw = rng.normal(size=(size, size))
    return raw @ raw.T + shift * np.eye(size)


# ---------------------------------------------------------------------------
# Linear systems
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[4.0, 1.0], [1.0, 3.0]]),
        np.array([[2.0, 0.5], [0.5, 2.0]]),
        np.array([[1.0]]),
        np.diag([1.0, 2.0, 5.0]),
    ],
)
def test_solution_matches_numpy_solve(matrix: np.ndarray) -> None:
    """The recovered direction must equal the exact solution's direction.

    The output is only defined up to an unknown normalisation constant, so the
    comparison is against the unit-normalised classical solution.
    """
    size = matrix.shape[0]
    rhs = np.arange(1.0, size + 1.0)
    result = hhl_solve(matrix, rhs)

    exact = np.linalg.solve(matrix, rhs)
    exact = exact / np.linalg.norm(exact)

    assert np.allclose(result.solution, exact, atol=1e-9)
    assert result.relative_error < 1e-9


def test_random_spd_matrices_are_solved() -> None:
    rng = np.random.default_rng(0)
    for size in (2, 3, 4):
        matrix = spd_matrix(rng, size)
        rhs = rng.normal(size=size)
        result = hhl_solve(matrix, rhs)
        assert result.relative_error < 1e-9, size


def test_solution_direction_is_independent_of_rhs_scale() -> None:
    """HHL normalises ``b`` internally, so scaling it must not change the answer."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    first = hhl_solve(matrix, rhs)
    second = hhl_solve(matrix, 1000.0 * rhs)
    assert np.allclose(first.solution, second.solution, atol=1e-12)


def test_singular_matrix_is_rejected() -> None:
    """``A`` with a zero eigenvalue has no solution for a general ``b``."""
    with pytest.raises(ValueError, match="positive-definite"):
        hhl_solve(np.diag([1.0, 0.0]), np.array([1.0, 1.0]))


def test_indefinite_matrix_is_rejected_with_guidance() -> None:
    """Negative eigenvalues need a shift that this implementation does not do."""
    with pytest.raises(ValueError, match="Childs-Braun-MacCleay"):
        hhl_solve(np.diag([1.0, -2.0]), np.array([1.0, 1.0]))


def test_non_hermitian_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        hhl_solve(np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([1.0, 1.0]))


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        hhl_solve(np.eye(2), np.zeros(2))


def test_mismatched_vector_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="entries"):
        hhl_solve(np.eye(3), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# LCU decomposition
# ---------------------------------------------------------------------------


def test_lcu_coefficients_are_bounded() -> None:
    """``0 < c_k <= 1`` is the property the block encoding relies on."""
    rng = np.random.default_rng(1)
    for size in (2, 3, 4):
        matrix = spd_matrix(rng, size)
        coefficients, eigenvalues, tau = lcu_decomposition(matrix)
        assert np.all(coefficients > 0)
        assert np.all(coefficients <= 1 + 1e-12)
        assert np.allclose(coefficients * tau, eigenvalues)
        assert tau == pytest.approx(float(eigenvalues.max()))


def test_lcu_reconstructs_the_matrix() -> None:
    """``tau * sum_k c_k P_k`` must equal ``A``."""
    rng = np.random.default_rng(2)
    matrix = spd_matrix(rng, 3)
    coefficients, eigenvalues, tau = lcu_decomposition(matrix)
    _, vectors = np.linalg.eigh(matrix)
    rebuilt = sum(
        c * np.outer(vectors[:, k], vectors[:, k].conj())
        for k, c in enumerate(coefficients)
    ) * tau
    assert np.allclose(rebuilt, matrix, atol=1e-9)
    assert np.allclose(eigenvalues, np.linalg.eigvalsh(matrix))


def test_lcu_rejects_tau_below_lambda_max() -> None:
    """``tau < lambda_max`` would give a coefficient above one."""
    matrix = np.diag([1.0, 4.0])
    with pytest.raises(ValueError, match="smaller than lambda_max"):
        lcu_decomposition(matrix, tau=2.0)


def test_lcu_accepts_larger_tau() -> None:
    matrix = np.diag([1.0, 4.0])
    coefficients, _, tau = lcu_decomposition(matrix, tau=8.0)
    assert np.allclose(coefficients, [0.125, 0.5])
    assert tau == 8.0


def test_hermitian_part_accepts_symmetric_input() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 3.0]])
    assert np.allclose(hermitian_part(matrix), matrix)


def test_hermitian_part_rejects_asymmetric_input() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        hermitian_part(np.array([[1.0, 2.0], [0.0, 3.0]]))


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def test_conditioning_number_matches_the_spectrum() -> None:
    assert conditioning_number(np.diag([1.0, 4.0])) == pytest.approx(4.0)
    assert conditioning_number(np.eye(3)) == pytest.approx(1.0)


def test_conditioning_number_is_infinite_for_singular_matrices() -> None:
    assert conditioning_number(np.diag([0.0, 1.0])) == math.inf


def test_ill_conditioned_systems_cost_more_post_selection() -> None:
    """The raw weight must grow as the smallest eigenvalue shrinks.

    This is the algorithm's central weakness: the ``1/lambda_k`` factors inflate
    the output amplitude, so more shots are needed to post-select on the ancilla.
    """
    weights = []
    for smallest in (2.0, 1.1, 1.01, 1.001):
        matrix = np.diag([smallest, smallest + 0.5])
        result = hhl_solve(matrix, np.array([1.0, 1.0]))
        weights.append(result.post_selection_weight)

    assert weights[0] < weights[1] < weights[2] < weights[3]


def test_post_selection_weight_is_not_a_probability() -> None:
    """Pins the naming: the weight is ``||A^{-1} b||^2`` and can exceed one.

    It is the squared norm of an un-normalised amplitude, not the probability of
    a measurement outcome. Treating it as a probability and asserting it is at
    most one would be asserting something false.
    """
    matrix = np.array([[0.05, 0.0], [0.0, 1.0]])
    result = hhl_solve(matrix, np.array([1.0, 0.0]))
    assert result.post_selection_weight > 1.0
    # The solution is still correct, because it is renormalised.
    assert result.relative_error < 1e-9


# ---------------------------------------------------------------------------
# Clock register
# ---------------------------------------------------------------------------


def test_zero_clock_bits_is_exact() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    result = hhl_solve(matrix, np.array([1.0, 2.0]), clock_bits=0)
    assert result.relative_error < 1e-12


def test_nonzero_clock_bits_adds_readout_error() -> None:
    """A finite clock register degrades the estimate, as phase estimation must."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    matrix = matrix / np.linalg.norm(matrix, 2)
    rhs = np.array([1.0, 2.0])

    coarse = hhl_solve(matrix, rhs, clock_bits=3)
    fine = hhl_solve(matrix, rhs, clock_bits=8)

    assert fine.relative_error < coarse.relative_error
    # Even the coarse run is recognisably right.
    assert coarse.relative_error < 0.2


def test_phase_estimates_converge_with_more_clock_bits() -> None:
    """``O(log(1/eps))`` clock register: each extra bit halves the error."""
    rng = np.random.default_rng(3)
    matrix = spd_matrix(rng, 2)
    matrix = matrix / np.linalg.norm(matrix, 2)

    errors = []
    for bits in (3, 6):
        estimates, exact = hhl_phase_estimates(matrix, bits)
        wrapped = np.array(
            [min((e - t + 0.5) % 1.0 - 0.5, -(e - t)) for e, t in zip(estimates, exact)]
        )
        errors.append(float(np.max(np.abs(estimates - exact - wrapped))))
    assert errors[1] < errors[0]


def test_phase_estimates_are_exact_for_grid_aligned_eigenvalues() -> None:
    """Eigenvalues on the ``2**-m`` grid must be recovered exactly."""
    for bits in (3, 4, 5):
        grid = np.array([k / 2**bits for k in (1, 3)])
        matrix = np.diag(grid)
        estimates, _ = hhl_phase_estimates(matrix, bits)
        assert np.allclose(np.sort(estimates), np.sort(grid), atol=1e-9), bits


def test_result_repr_mentions_the_relative_error() -> None:
    result = hhl_solve(np.eye(2), np.array([1.0, 0.0]))
    assert "rel_error" in repr(result)
    assert isinstance(result, HHLResult)
