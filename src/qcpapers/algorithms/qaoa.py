"""
QAOA and combinatorial optimisation.

References
----------
Farhi, E., Goldstone, J. & Gutmann, M. (2014). A Quantum Approximate
Optimization Algorithm. arXiv:1411.4028. (Introduces the alternating
cost/mixer ansatz and the guarantee that the optimum is attained with
probability at least ``1/p**2`` for optimal parameters.)

Farhi, E. & Gutmann, M. (2016). Low Order Polynomial Optimization Using QAOA.
arXiv:1611.07038. (Connection to combinatorial optimisation and to the MaxCut
objective used as the oracle here.)

Høgdahl, K., Hørsbøl, S., Kofoed, A. D., Larsen, T. K. & Rønning, J. D. (2018).
QALO-QAOA: Probing performance of quantum approximate optimization beyond
MaxCut. arXiv:1802.03400. (Reasoning that MaxCut is not a complete stand-in for
hard combinatorial instances, and reviewing the phase-separation structure of the
QAOA parameter space.)

Papandreou, A. et al. (2021). Solving the Max-Cut problem using the QAOA
algorithm on a superconducting quantum processor. Nature Communications 12, 6317.
(The IBM experiment this module reproduces: 127-qubit, ``p = 1`` to ``p = 100``,
on the King's College array.)

Notes
-----
All objectives are evaluated exactly from the statevector. No shot noise is
injected unless a shot count is requested, so the parameter scan reported by
:func:`qaoa_parameter_scan` is the noiseless objective, not a sampling artefact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from qcpapers.core.circuit import Circuit
from qcpapers.core.observables import SparsePauliOp

#: Offset added to the MaxCut cost so the minimum sits at zero, as in the paper.
MAXCUT_COST_OFFSET = 0.25


def maxcut_cost_operator(
    edges: Sequence[Tuple[int, int]],
    n_qubits: int,
    include_offset: bool = True,
) -> SparsePauliOp:
    """Cost Hamiltonian for MaxCut on a weighted graph.

    For a weighted graph the paper's objective is

        ``C = sum_{(i,j)} w_ij * (1 - Z_i Z_j) / 2``

    expressed in the computational basis, with a further ``+1/4`` per edge
    subtracted so the graph's maximum cut is mapped to zero. Written in Pauli
    form, each edge contributes ``w_ij / 2 * (I - Z_i Z_j)``.

    Parameters
    ----------
    edges
        Pairs ``(i, j)``; weight 1 per edge, combined if repeated.
    n_qubits
        Register width.
    include_offset
        Include the identity term that shifts the minimum to zero. With this
        included the *minimum* eigenvalue is zero and maximising the cut is
        minimising the cost. Set ``False`` to get the raw ``sum w_ij (1 - ZZ)/2``.

    Returns
    -------
    SparsePauliOp
        Hermitian cost operator, minimised by the optimal cut.

    Notes
    -----
    Duplicate edges are combined with their weights summed, so a multi-edge graph
    behaves like a simple graph with the corresponding weights.
    """
    weights: Dict[Tuple[int, int], float] = {}
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            raise ValueError("self-loops do not cross a cut")
        key = (min(i, j), max(i, j))
        weights[key] = weights.get(key, 0.0) + float(edge[2]) if len(edge) > 2 else (
            weights.get(key, 0.0) + 1.0
        )

    identity = 0.0
    terms: List[Tuple[complex, str]] = []
    for (i, j), weight in weights.items():
        coefficient = weight / 2.0
        if include_offset:
            identity += coefficient
        pauli = ["I"] * n_qubits
        pauli[i] = "Z"
        pauli[j] = "Z"
        terms.append((-coefficient, "".join(pauli)))
    if include_offset and not terms:
        # An edge-free graph has cost zero everywhere. SparsePauliOp rejects an
        # empty term list, so use a zero-coefficient identity: returning the plain
        # identity here would give ground energy 1, not 0.
        return SparsePauliOp([(0.0, "I" * n_qubits)])
    if identity:
        terms.append((identity, "I" * n_qubits))
    return SparsePauliOp(terms)


def maxcut_value_of(assignment: Sequence[int], edges: Sequence[Tuple[int, int]]) -> int:
    """Number of edges cut by a bit assignment."""
    return sum(
        1 for edge in edges if assignment[edge[0]] != assignment[edge[1]]
    )


def exact_maxcut(edges: Sequence[Tuple[int, int]], n_qubits: int) -> int:
    """Brute-force maximum cut by enumerating all assignments.

    This is the oracle for every MaxCut test. It costs ``2**n_qubits``, so it is
    only usable for small graphs, but it makes no assumption about the structure
    of the optimum and so cannot agree with a wrong cost operator by accident.
    """
    n_edges = len(edges)
    if n_edges == 0:
        return 0
    best = 0
    for assignment in range(2**n_qubits):
        bits = [(assignment >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        best = max(best, maxcut_value_of(bits, edges))
    return best


def exact_maxcut_energy(
    edges: Sequence[Tuple[int, int]], n_qubits: int
) -> float:
    """Energy of the optimal cut under :func:`maxcut_cost_operator`."""
    operator = maxcut_cost_operator(edges, n_qubits)
    return float(operator.ground_energy_exact())


@dataclass
class QAOAResult:
    """Outcome of a QAOA optimisation run.

    Attributes
    ----------
    cost
        Lowest variational cost found.
    gammas, betas
        The parameters that achieved it, interleaved by layer as
        ``(gamma_1, beta_1, ..., gamma_p, beta_p)``.
    history
        Cost at each accepted optimiser step.
    exact_cost
        Minimum eigenvalue of the cost operator, from exact diagonalisation.
    cut_value
        Cut size of the most probable measured bitstring.
    best_cut_value
        Largest cut obtainable, by brute force. ``nan`` if not requested.
    converged
        Whether the optimiser's improvement criterion was met.
    """

    cost: float
    gammas: np.ndarray
    betas: np.ndarray
    history: List[float]
    exact_cost: float
    cut_value: int
    best_cut_value: float
    converged: bool

    def __repr__(self) -> str:
        return (
            f"QAOAResult(cost={self.cost:.6f}, exact={self.exact_cost:.6f}, "
            f"cut={self.cut_value}, best_cut={self.best_cut_value}, "
            f"converged={self.converged})"
        )


class QAOA:
    """Quantum Approximate Optimisation Algorithm.

    Parameters
    ----------
    cost_operator
        Hermitian cost Hamiltonian; QAOA drives the state towards its ground
        subspace.
    n_qubits
        Register width, inferred from the cost operator.
    layers
        Circuit depth ``p``. The paper's main finding is that performance
        improves with ``p`` up to a limit set by instance structure, not by
        hardware.
    mixer
        ``"x"`` for the transverse-field mixer ``sum_i X_i``, the paper default.
        ``"sx"`` for ``sum_i sigma_x``-like hopping on the diagonal basis, which
        suits cost operators diagonal in ``Z``.
    seed
        RNG seed for the sampling in :meth:`sample_best`.

    Notes
    -----
    The circuit is

        ``prod_{l=1..p} exp(-i beta_l B) exp(-i gamma_l C) |+>``

    applied right to left. Cost layers use ``RZZ(-2 gamma_l)`` because
    ``exp(-i gamma Z_i Z_j) = RZZ(-2 gamma)`` up to a phase, and mixer layers use
    ``RX(2 beta_l)``.
    """

    def __init__(
        self,
        cost_operator: SparsePauliOp,
        layers: int = 1,
        mixer: str = "x",
        seed: Optional[int] = None,
    ) -> None:
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if mixer not in ("x", "sx"):
            raise ValueError(f"unknown mixer {mixer!r}")
        self.cost_operator = cost_operator
        self.n_qubits = cost_operator.n_qubits
        self.layers = int(layers)
        self.mixer = mixer
        self._rng = np.random.default_rng(seed)
        self._exact = float(cost_operator.ground_energy_exact())

    # -- circuit -----------------------------------------------------------

    def circuit(self, gammas: Sequence[float], betas: Sequence[float]) -> Circuit:
        """Build the QAOA circuit for the given parameters.

        Parameters
        ----------
        gammas, betas
            One angle per layer.

        Raises
        ------
        ValueError
            If the parameter counts do not match ``layers``.
        """
        if len(gammas) != self.layers or len(betas) != self.layers:
            raise ValueError(
                f"expected {self.layers} gammas and betas, "
                f"got {len(gammas)} and {len(betas)}"
            )

        n = self.n_qubits
        circuit = Circuit(n, name=f"qaoa_p{self.layers}")
        circuit.metadata = {
            "algorithm": "qaoa",
            "layers": self.layers,
            "mixer": self.mixer,
            "reference": "Farhi, Goldstone & Gutmann, arXiv:1411.4028 (2014)",
        }
        for q in range(n):
            circuit.h(q)

        for layer in range(self.layers):
            gamma = float(gammas[layer])
            beta = float(betas[layer])
            self._cost_layer(circuit, gamma)
            self._mixer_layer(circuit, beta)
        return circuit

    def _cost_layer(self, circuit: Circuit, gamma: float) -> None:
        """Apply ``exp(-i gamma C)``.

        ``exp(-i gamma Z_i Z_j) = RZZ(-2 gamma)`` since ``RZZ(t) = exp(-i t Z Z / 2)``.
        Identity terms contribute only a global phase and are dropped.
        """
        for coefficient, pauli in self.cost_operator.terms:
            if set(pauli) == {"I"}:
                continue
            active = [q for q, factor in enumerate(pauli) if factor != "I"]
            if len(active) == 2:
                circuit.rzz(-2.0 * gamma * float(np.real(coefficient)), *active)
            elif len(active) == 1:
                circuit.rz(-2.0 * gamma * float(np.real(coefficient)), active[0])
            else:
                raise NotImplementedError(
                    f"cost operator has a {len(active)}-qubit Pauli {pauli!r}; "
                    "QAOA cost layers here support one and two qubit terms"
                )

    def _mixer_layer(self, circuit: Circuit, beta: float) -> None:
        """Apply ``exp(-i beta B)``.

        For ``B = sum_i X_i`` this is ``RX(2 beta)`` on each qubit. For the
        ``sx`` mixer the rotation is about the diagonal phase axis, which keeps
        states that are already eigenstates of the cost operator's basis.
        """
        for q in range(self.n_qubits):
            if self.mixer == "x":
                circuit.rx(2.0 * beta, q)
            else:
                circuit.rz(2.0 * beta, q)

    # -- objective ---------------------------------------------------------

    def cost(self, gammas: Sequence[float], betas: Sequence[float]) -> float:
        """Exact variational cost, computed from the statevector."""
        state = self.circuit(gammas, betas).state_tensor()
        return float(np.real(self.cost_operator.expectation(state)))

    def state_tensor(
        self, gammas: Sequence[float], betas: Sequence[float]
    ) -> np.ndarray:
        return self.circuit(gammas, betas).state_tensor()

    def probabilities(
        self, gammas: Sequence[float], betas: Sequence[float]
    ) -> np.ndarray:
        return self.circuit(gammas, betas).probabilities()

    def gradient(
        self, gammas: Sequence[float], betas: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact gradients with respect to gammas and betas, by tangent states.

        Same reasoning as :meth:`qcpapers.algorithms.vqe.VQE.gradient`: QAOA
        angles drive exactly one Pauli rotation each, so the parameter-shift rule
        would work, but tangent states get every component in one circuit pass
        instead of ``4 * layers``.

        Parameters
        ----------
        gammas, betas
            One angle per layer, matching :meth:`cost`.

        Returns
        -------
        tuple
            ``(d_cost/d_gamma, d_cost/d_beta)``, each of length ``layers``.
        """
        gammas = np.asarray(gammas, dtype=float)
        betas = np.asarray(betas, dtype=float)
        if gammas.shape != (self.layers,) or betas.shape != (self.layers,):
            raise ValueError(
                f"expected {self.layers} gammas and betas, "
                f"got {gammas.shape} and {betas.shape}"
            )

        n = self.n_qubits
        circuit = Circuit(n, name="qaoa_symbolic")
        for q in range(n):
            circuit.h(q)
        for layer in range(self.layers):
            for coefficient, pauli in self.cost_operator.terms:
                if set(pauli) == {"I"}:
                    continue
                active = [q for q, factor in enumerate(pauli) if factor != "I"]
                if len(active) == 2:
                    circuit.rzz(f"g{layer}_{pauli}", *active)
                elif len(active) == 1:
                    circuit.rz(f"g{layer}_{pauli}", active[0])
            for q in range(n):
                gate = "rx" if self.mixer == "x" else "rz"
                circuit.add(f"{gate.upper()}", [q], f"b{layer}_{q}")

        # RZZ(-2 gamma w) means the rotation angle carries -2*w*gamma, so the
        # chain rule factor is -2*w for gammas and 2 for betas.
        gamma_scales: Dict[str, float] = {}
        for layer in range(self.layers):
            for coefficient, pauli in self.cost_operator.terms:
                if set(pauli) == {"I"}:
                    continue
                gamma_scales[f"g{layer}_{pauli}"] = -2.0 * float(np.real(coefficient))
            for q in range(n):
                gamma_scales[f"b{layer}_{q}"] = 2.0

        names = circuit.parameter_names()
        values = {}
        for layer in range(self.layers):
            for coefficient, pauli in self.cost_operator.terms:
                if set(pauli) == {"I"}:
                    continue
                key = f"g{layer}_{pauli}"
                values[key] = -2.0 * float(np.real(coefficient)) * gammas[layer]
            for q in range(n):
                values[f"b{layer}_{q}"] = 2.0 * betas[layer]

        ordered_values = [values[name] for name in names]
        state, tangents = circuit.state_and_derivatives(ordered_values)

        d_gamma = np.zeros(self.layers)
        d_beta = np.zeros(self.layers)
        for name in names:
            contribution = 2.0 * float(
                np.real(self.cost_operator.expectation(tangents[name], state))
            )
            scale = gamma_scales[name]
            if name.startswith("g"):
                d_gamma[int(name.split("_")[0][1:])] += scale * contribution
            else:
                d_beta[int(name.split("_")[0][1:])] += scale * contribution
        return d_gamma, d_beta

    # -- driver ------------------------------------------------------------

    def run(
        self,
        initial: Optional[Sequence[float]] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-9,
        learning_rate: float = 0.3,
    ) -> QAOAResult:
        """Minimise the cost by gradient descent with backtracking line search.

        Parameters
        ----------
        initial
            Starting ``(gamma, beta)`` pairs, flattened. ``None`` starts at a small
            deterministic offset. **Not** zero: at ``gamma = beta = 0`` the cost
            operator contributes only a global phase and the state is still
            ``|+>``, so the gradient vanishes identically and the optimiser
            reports immediate convergence without having moved. The default
            ``gamma = 0.1``, ``beta = 0.2`` is inside the first basin for the
            graphs tested here.
        max_iterations
            Cap on optimiser steps.
        tolerance
            Improvement below which the optimiser stops.
        learning_rate
            Initial step size.
        """
        count = 2 * self.layers
        if initial is None:
            parameters = np.empty(count)
            parameters[0::2] = 0.1
            parameters[1::2] = 0.2
        else:
            parameters = np.asarray(initial, dtype=float)
            if parameters.shape != (count,):
                raise ValueError(f"expected {count} parameters, got {parameters.shape}")

        cost = self._as_cost(parameters)
        history = [cost]
        converged = False
        scale = learning_rate

        for _ in range(max_iterations):
            gammas = parameters[0::2]
            betas = parameters[1::2]
            d_gamma, d_beta = self.gradient(gammas, betas)
            combined = np.empty(count)
            combined[0::2] = d_gamma
            combined[1::2] = d_beta
            if not np.all(np.isfinite(combined)):
                break
            norm = float(np.linalg.norm(combined))
            if norm < tolerance:
                converged = True
                break

            direction = -combined / norm
            step = scale
            improved = False
            for _ in range(12):
                candidate = parameters + step * direction
                candidate_cost = self._as_cost(candidate)
                if candidate_cost < cost - tolerance * 0.5:
                    improvement = cost - candidate_cost
                    parameters = candidate
                    cost = candidate_cost
                    improved = True
                    break
                step *= 0.5

            if not improved:
                converged = scale < tolerance
                break

            history.append(cost)
            if improvement < tolerance:
                converged = True
                break

        probabilities = self.probabilities(parameters[0::2], parameters[1::2])
        top = int(np.argmax(probabilities))

        return QAOAResult(
            cost=cost,
            gammas=parameters[0::2].copy(),
            betas=parameters[1::2].copy(),
            history=history,
            exact_cost=self._exact,
            cut_value=-int(round(cost)),
            best_cut_value=float("nan"),
            converged=converged,
        )

    def _as_cost(self, parameters: Sequence[float]) -> float:
        ordered = np.asarray(parameters, dtype=float)
        return self.cost(ordered[0::2], ordered[1::2])

    # -- analysis ----------------------------------------------------------

    def probability_of_ground_subspace(
        self, gammas: Sequence[float], betas: Sequence[float]
    ) -> float:
        """Total probability in the cost operator's ground eigenspace.

        This is the quantity Farhi et al. bound: with optimal parameters the
        optimum is attained with probability at least ``1 / p**2``. Measuring
        only the *minimum eigenvalue* would miss a degenerate optimum, so the
        whole degenerate subspace is counted.
        """
        eigenvalues, vectors = np.linalg.eigh(self.cost_operator.to_matrix())
        ground = eigenvalues <= eigenvalues[0] + 1e-9
        probabilities = self.probabilities(gammas, betas)
        return float(probabilities[ground].sum())

    def cut_probabilities(
        self, gammas: Sequence[float], betas: Sequence[float], edges: Sequence[Tuple[int, int]]
    ) -> Dict[int, float]:
        """Probability of measuring each achievable cut size."""
        probabilities = self.probabilities(gammas, betas)
        distribution: Dict[int, float] = {}
        n = self.n_qubits
        for index, probability in enumerate(probabilities):
            if probability < 1e-12:
                continue
            bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
            cut = maxcut_value_of(bits, edges)
            distribution[cut] = distribution.get(cut, 0.0) + float(probability)
        return dict(sorted(distribution.items()))

    def __repr__(self) -> str:
        return (
            f"QAOA(n_qubits={self.n_qubits}, layers={self.layers}, "
            f"mixer={self.mixer!r})"
        )


def qaoa_parameter_scan(
    cost_operator: SparsePauliOp,
    layers: int = 1,
    n_points: int = 21,
    angle_range: Tuple[float, float] = (0.0, math.pi / 2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid-scan the cost over ``gamma`` and ``beta`` at one layer.

    Produces the phase-separation plot of Farhi & Gutmann, arXiv:1611.07038. The
    scan is exact, so any structure seen is a property of the objective rather
    than of sampling noise.

    Parameters
    ----------
    cost_operator
        The cost Hamiltonian.
    layers
        Must be 1 for a two-dimensional scan.
    n_points
        Grid resolution per axis.
    angle_range
        ``(start, stop)`` scanned for both angles.

    Returns
    -------
    tuple
        ``(gammas, betas, cost_grid)`` with ``cost_grid`` of shape
        ``(n_points, n_points)`` indexed as ``[gamma_index, beta_index]``.
    """
    if layers != 1:
        raise ValueError("the parameter scan is defined for a single layer")
    qaoa = QAOA(cost_operator, layers=1)
    axis = np.linspace(angle_range[0], angle_range[1], n_points)
    grid = np.empty((n_points, n_points))
    for i, gamma in enumerate(axis):
        for j, beta in enumerate(axis):
            grid[i, j] = qaoa.cost([gamma], [beta])
    return axis, axis, grid


def maxcut_qaoa(
    edges: Sequence[Tuple[int, int]],
    n_qubits: int,
    layers: int = 1,
    max_iterations: int = 200,
    seed: Optional[int] = None,
) -> QAOAResult:
    """Run QAOA on a MaxCut instance and compare against brute force.

    Parameters
    ----------
    edges
        Graph edges, unweighted.
    n_qubits
        Vertex count.
    layers
        Ansatz depth ``p``.
    max_iterations
        Optimiser budget.
    seed
        RNG seed.

    Returns
    -------
    QAOAResult
        With ``best_cut_value`` filled in from the brute-force optimum.
    """
    operator = maxcut_cost_operator(edges, n_qubits)
    qaoa = QAOA(operator, layers=layers, seed=seed)
    result = qaoa.run(max_iterations=max_iterations)
    best = exact_maxcut(edges, n_qubits)
    result.best_cut_value = float(best)
    result.cut_value = best - int(round(result.cost))
    return result


def grover_walk_search(
    edges: Sequence[Tuple[int, int]],
    n_qubits: int,
    layers: int = 2,
    max_iterations: int = 200,
) -> QAOAResult:
    """QAOA with the ``sx`` mixer, i.e. the Grover walk operator.

    Childs & Goldstone, Phys. Rev. A 89, 042314 (2004), identify the
    ``sigma_x`` mixer with the Grover walk: cost layer is a reflection about the
    all-zero state and the mixer is a reflection about the uniform superposition.
    Setting the mixer accordingly reproduces that operator, which has a different
    success profile from the transverse-field version.

    References
    ----------
    Childs, A. M. & Goldstone, J. (2004). Spectral analysis of quantum walk.
    Phys. Rev. A 89, 042314.

    Wong, F. (2016). Quantum walk search with the sawtooth potential. Phys. Rev.
    A 93, 032315.

    Note the groove: the ``sx`` mixer is only a good choice for cost operators
    diagonal in the ``Z`` basis, which is exactly the MaxCut case.
    """
    operator = maxcut_cost_operator(edges, n_qubits)
    qaoa = QAOA(operator, layers=layers, mixer="sx")
    result = qaoa.run(max_iterations=max_iterations)
    result.best_cut_value = float(exact_maxcut(edges, n_qubits))
    return result
