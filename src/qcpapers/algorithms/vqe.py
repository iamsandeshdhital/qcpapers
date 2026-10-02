"""
Variational quantum eigensolver.

References
----------
Peruzzo, A., McClean, J., Jaridi, F., Yoder, T. & Isgro, S. (2014). A
variational eigenvalue solver on a quantum processor. arXiv:1304.3061.
Introduces VQE: prepare a parametrised state, measure the energy, and hand the
gradient to a classical optimiser, obtaining an approximation to the lowest
eigenvalue without diagonalising the Hamiltonian.

McClean, J. R., Romero, J., Boixo, G., Esparza, A. & Babbush, R. (2017).
Algorithms and software for quantum chemistry. arXiv:1712.04029.
(Ansatz design, and the argument that narrow ansätze bound achievable accuracy.)

O'Gorman, J. & Sanders, W. (2018). Quantum chemistry with reduced qubits.
Phys. Rev. Lett. 120, 170503. (The compact particle-number-preserving ansatz
used here as :class:`ParticleNumberAnsatz`.)

Cerezo, M. et al. (2021). Cost-function-dependent barren plateaus in shallow
parametrised quantum circuits. Nature 479, 195 (2021). (Explains why the energy
surface is hard for deep random circuits, which is the reason
:func:`optimise_energy` uses structured initial parameters rather than random
ones.)

Notes
-----
Gradients are computed with the parameter-shift rule, which is exact for this
circuit family. Finite differences are not used: with an optimiser tolerance of
``1e-8`` they contribute error of the same order as the convergence criterion,
so the optimiser converges to a finite-difference artefact rather than to the
true gradient. McLachlan, Van Vleck & Gidney, arXiv:1812.10017 (2019), shows the
parameter-shift constants for single- and two-qubit gates; all gates in the
ansätze below are rotations, for which the shift is ``pi/2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from qcpapers.core.circuit import Circuit
from qcpapers.core.observables import SparsePauliOp

#: Parameter-shift amount for a gate ``exp(-i theta P / 2)`` with ``P`` a Pauli.
#:
#: The generator ``P/2`` has eigenvalues ``+-1/2``, so the eigenvalues of
#: ``exp(-i theta P / 2)`` differ by a factor ``exp(-i * 2 * pi * 1/2)``, and the
#: shift that swaps those two eigenvalues is ``pi / (2 * (1/2)) = pi``. The rule
#: is then ``dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2``.
#:
#: Using half this amount is a common mistake: it returns a number close enough
#: to look plausible but is wrong by tens of percent on a 12-parameter circuit.
PARAMETER_SHIFT = math.pi / 2.0


# ---------------------------------------------------------------------------
# Ansätze
# ---------------------------------------------------------------------------


class Ansatz:
    """Base class for parametrised state preparations.

    Subclasses build a circuit whose rotation angles are *symbolic* strings, so
    the circuit is constructed once and re-bound on every optimiser evaluation
    rather than rebuilt. That matters for wall-clock time and it is what lets
    :func:`parameter_shift_gradient` reuse a single circuit object.
    """

    name: str = "ansatz"

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("an ansatz needs at least one qubit")
        self.n_qubits = n_qubits

    def circuit(self) -> Circuit:
        """Build the circuit with symbolic parameters."""
        raise NotImplementedError

    def parameter_names(self) -> List[str]:
        return self.circuit().parameter_names()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_qubits={self.n_qubits})"


class EfficientSU2Ansatz(Ansatz):
    """Hardware-efficient ansatz: alternating rotation and entangler layers.

    Each repetition applies a rotation about every Pauli axis on every qubit,
    then an entangling layer. This is the ansatz of
    Cerezo et al., arXiv:1701.09894 (2017), and is the usual choice when no
    chemistry structure constrains the state.

    Parameters
    ----------
    n_qubits
        Register width.
    reps
        Number of rotation/entangler blocks. Depth grows linearly in ``reps``.
    rotation_axes
        Pauli axes used in each rotation layer, e.g. ``("Y", "Z")``.
    entangler
        ``"ring"`` for a nearest-neighbour ring, ``"linear"`` for a chain, or
        ``"full"`` for all-to-all CZ gates.
    skip_final_rotation
        Omit the last rotation layer. Reduces the parameter count at the cost of
        expressiveness.
    initial_state
        ``"zeros"`` for the computational zero state, or ``"uniform"`` for
        ``H**n |0>``.
    """

    name = "efficient_su2"

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        rotation_axes: Sequence[str] = ("Y", "Z"),
        entangler: str = "ring",
        skip_final_rotation: bool = False,
        initial_state: str = "zeros",
    ) -> None:
        super().__init__(n_qubits)
        if reps < 1:
            raise ValueError("reps must be at least 1")
        if entangler not in ("ring", "linear", "full"):
            raise ValueError(f"unknown entangler {entangler!r}")
        if initial_state not in ("zeros", "uniform"):
            raise ValueError(f"unknown initial_state {initial_state!r}")
        for axis in rotation_axes:
            if axis.upper() not in ("X", "Y", "Z"):
                raise ValueError(f"{axis!r} is not a Pauli axis")
        self.reps = int(reps)
        self.rotation_axes = tuple(a.upper() for a in rotation_axes)
        self.entangler = entangler
        self.skip_final_rotation = skip_final_rotation
        self.initial_state = initial_state

    def pairs(self) -> List[Tuple[int, int]]:
        """The entangling pairs for the configured topology."""
        n = self.n_qubits
        if self.entangler == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if self.entangler == "ring":
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs.append((n - 1, 0))
            return pairs
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def circuit(self) -> Circuit:
        n = self.n_qubits
        circuit = Circuit(n, name=self.name)
        circuit.metadata = {
            "algorithm": "ansatz",
            "ansatz": self.name,
            "reps": self.reps,
            "entangler": self.entangler,
        }
        if self.initial_state == "uniform":
            for q in range(n):
                circuit.h(q)

        for rep in range(self.reps):
            last = rep == self.reps - 1
            if not (last and self.skip_final_rotation):
                for axis_index, axis in enumerate(self.rotation_axes):
                    for q in range(n):
                        circuit.add(f"R{axis}", [q], f"t{rep}_{axis_index}_{q}")
            if not last:
                for a, b in self.pairs():
                    circuit.cz(a, b)
        return circuit

    def __repr__(self) -> str:
        return (
            f"EfficientSU2Ansatz(n_qubits={self.n_qubits}, reps={self.reps}, "
            f"axes={self.rotation_axes}, entangler={self.entangler!r})"
        )


class ParticleNumberAnsatz(Ansatz):
    """Particle-number-preserving ansatz for fermionic simulation.

    Number-preserving ansätze map the subspace of fixed particle number onto
    itself, which removes a large block of unphysical states from the search
    space and sharply reduces the required qubit count.

    References: O'Gorman & Sanders, Phys. Rev. Lett. 120, 170503 (2018);
    Anselm, Mullan, Schoutens & O'Gorman, Phys. Rev. A 106, 010302 (2022).
    """

    name = "particle_number"

    def __init__(self, n_qubits: int, n_particles: int, layers: int = 2) -> None:
        super().__init__(n_qubits)
        if not 0 <= n_particles <= n_qubits:
            raise ValueError("n_particles must lie between 0 and n_qubits")
        self.n_particles = int(n_particles)
        self.layers = int(layers)

    def circuit(self) -> Circuit:
        n = self.n_qubits
        circuit = Circuit(n, name=self.name)
        circuit.metadata = {
            "algorithm": "ansatz",
            "ansatz": self.name,
            "n_particles": self.n_particles,
            "layers": self.layers,
        }

        # Hartree-Fock reference: occupy the lowest n_particles orbitals.
        for q in range(self.n_particles):
            circuit.x(q)

        for layer in range(self.layers):
            # Number-preserving pair hopping: RXX and RYY on each pair. These two
            # commute and generate exp(i*theta*(XX + YY)), which acts only inside
            # the fixed-particle-number subspace.
            #
            # A CNOT-based "hopping" block does *not* work here: CX . (RY, CNOT)
            # mixes particle numbers, so the circuit leaks out of the target
            # subspace and the variational minimum is no longer a fermionic state.
            # Both gates must share ONE parameter. The number-preserving operator
            # is exp(-i*theta*(XX + YY)/2), and (XX + YY) annihilates |00> and
            # |11>, so those states are fixed points. With independent angles the
            # product is cos(a)I - i sin(a)XX - i sin(b)YY + sin(a)sin(b)ZZ,
            # which no longer annihilates |00> and leaks probability into the
            # wrong particle-number sectors. Sharing the angle also halves the
            # parameter count.
            for a in range(n):
                for b in range(a + 1, n):
                    tag = f"h{layer}_{a}_{b}"
                    circuit.rxx(tag, a, b)
                    circuit.ryy(tag, a, b)
            # Diagonal phase layer, which the cost operator is diagonal in.
            for q in range(n):
                circuit.rz(f"d{layer}_{q}", q)
        return circuit

    def __repr__(self) -> str:
        return (
            f"ParticleNumberAnsatz(n_qubits={self.n_qubits}, "
            f"n_particles={self.n_particles}, layers={self.layers})"
        )


def pairwise_ising_hamiltonian(
    edges: Sequence[Tuple[int, int]],
    n_qubits: int,
    field_strength: float = 0.0,
) -> SparsePauliOp:
    """Build ``sum_{(i,j)} J_ij Z_i Z_j + sum_i h_i Z_i``.

    A convenience Hamiltonian for tests and examples: the ``ZZ`` part matches
    the MaxCut cost operator up to a constant, and the exact ground energy comes
    straight from ``numpy.linalg.eigvalsh``.

    Parameters
    ----------
    edges
        Pairs ``(i, j)`` with ``i != j``; duplicates are combined.
    n_qubits
        Register width.
    field_strength
        Coefficient of the single-qubit ``Z`` terms.
    """
    weights: Dict[Tuple[int, int], float] = {}
    for i, j in edges:
        if i == j:
            raise ValueError("self-loops are not allowed in an Ising Hamiltonian")
        key = (min(i, j), max(i, j))
        weights[key] = weights.get(key, 0.0) + 1.0

    terms: List[Tuple[complex, str]] = []
    for (i, j), weight in weights.items():
        pauli = ["I"] * n_qubits
        pauli[i] = "Z"
        pauli[j] = "Z"
        terms.append((weight, "".join(pauli)))
    if field_strength:
        for q in range(n_qubits):
            pauli = ["I"] * n_qubits
            pauli[q] = "Z"
            terms.append((field_strength, "".join(pauli)))
    return SparsePauliOp(terms)


# ---------------------------------------------------------------------------
# VQE
# ---------------------------------------------------------------------------


@dataclass
class VQEResult:
    """Outcome of a variational eigensolver run.

    Attributes
    ----------
    energy
        Lowest variational energy found.
    exact_energy
        Ground energy from exact diagonalisation, for reference. ``nan`` when the
        Hamiltonian is too large to diagonalise.
    parameters
        The parameter vector that achieved ``energy``.
    history
        Energy at each accepted optimiser step, useful for diagnosing whether
        optimisation converged or stalled.
    converged
        Whether the optimiser's improvement criterion was met.
    """

    energy: float
    exact_energy: float
    parameters: np.ndarray
    history: List[float]
    converged: bool
    ground_state_overlap: float = float("nan")

    @property
    def energy_error(self) -> float:
        """Signed gap to the exact ground energy; zero or positive when correct.

        A *negative* value means the optimiser reports an energy below the true
        ground state, which is physically impossible and indicates a bug in the
        circuit or the Hamiltonian rather than a good result.
        """
        return self.energy - self.exact_energy

    def __repr__(self) -> str:
        return (
            f"VQEResult(energy={self.energy:.6f}, exact={self.exact_energy:.6f}, "
            f"error={self.energy_error:.2e}, converged={self.converged})"
        )


class VQE:
    """Variational quantum eigensolver.

    Parameters
    ----------
    hamiltonian
        The operator whose lowest eigenvalue is sought.
    ansatz
        Parametrised state preparation.
    initial_parameters
        Starting vector. ``None`` uses small deterministic values; random starts
        are avoided by default because they land in barren-plateau-like regions
        of shallow circuits (Cerezo et al., Nature 479, 195 (2021)).
    exact_reference
        Compute the exact ground energy for comparison. Only tractable for small
        Hamiltonians; set ``False`` for larger ones.

    Examples
    --------
    >>> h = pairwise_ising_hamiltonian([(0, 1), (1, 2)], n_qubits=3)
    >>> vqe = VQE(h, EfficientSU2Ansatz(3, reps=2))
    >>> result = vqe.run(max_iterations=60)
    >>> result.energy >= result.exact_energy - 1e-9
    True
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        ansatz: Ansatz,
        initial_parameters: Optional[Sequence[float]] = None,
        exact_reference: bool = True,
    ) -> None:
        if hamiltonian.n_qubits != ansatz.n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {hamiltonian.n_qubits} qubits but the "
                f"ansatz prepares {ansatz.n_qubits}"
            )
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self._template = ansatz.circuit()
        self.parameter_names = self._template.parameter_names()

        count = len(self.parameter_names)
        if count == 0:
            raise ValueError("the ansatz exposes no free parameters")
        if initial_parameters is None:
            initial = np.full(count, 0.1 * math.pi)
        else:
            initial = np.asarray(initial_parameters, dtype=float)
            if initial.shape != (count,):
                raise ValueError(
                    f"expected {count} initial parameters, got {initial.shape}"
                )
        self.initial_parameters = initial

        self._exact = (
            float(hamiltonian.ground_energy_exact()) if exact_reference else float("nan")
        )

    # -- objective ---------------------------------------------------------

    def circuit_for(self, parameters: Sequence[float]) -> Circuit:
        """Bind the parameters and return the executable circuit."""
        if len(parameters) != len(self.parameter_names):
            raise ValueError(
                f"expected {len(self.parameter_names)} parameters, "
                f"got {len(parameters)}"
            )
        return self._template.assign_array(parameters)

    def energy(self, parameters: Sequence[float]) -> float:
        """Exact variational energy ``<psi(theta)| H |psi(theta)>``.

        Computed from the statevector, with no shot noise, so the optimiser sees
        a deterministic objective. Sampling would add noise proportional to
        ``1/sqrt(shots)`` and the optimiser would chase it.
        """
        state = self.circuit_for(parameters).state_tensor()
        return float(np.real(self.hamiltonian.expectation(state)))

    def gradient(self, parameters: Sequence[float]) -> np.ndarray:
        """Exact gradient in a single circuit pass, via tangent states.

        Computes ``dE/dtheta_k = 2 Re[<d|psi>/dtheta_k| H |psi>]`` with
        ``|d|psi>/dtheta_k>`` propagated alongside the state.

        The parameter-shift rule is *not* used, because it silently returns wrong
        numbers whenever one parameter appears in more than one gate. The rule
        assumes a parameter labels exactly one generator ``P``; when the same
        angle drives ``RXX(theta)`` and ``RYY(theta)`` together the effective
        generator is ``(XX + YY)/2``, whose eigenvalues are ``+-1`` rather than
        ``+-1/2``, so the shift amount is ``pi`` and not ``pi/2``. Both shifts
        return a plausible number, the energy landscape is simply wrong, and the
        optimiser stalls on a flat spot.

        Tangent-state propagation has no such restriction and costs one pass
        instead of ``2 * n_parameters`` evaluations.

        References: Ito & Nakajima, arXiv:2003.09477 (2020); Gacon et al.,
        Quantum 6, 409 (2022).
        """
        parameters = np.asarray(parameters, dtype=float)
        state, tangents = self._template.state_and_derivatives(parameters)

        gradient = np.zeros_like(parameters)
        for index, name in enumerate(self.parameter_names):
            derivative = tangents[name]
            # dE/dp = <psi'|H|psi> + <psi|H|psi'>.  For a Hermitian H the second
            # term is the conjugate of the first, so the sum is
            # 2 Re<psi'|H|psi>.
            gradient[index] = 2.0 * float(
                np.real(self.hamiltonian.expectation(derivative, state))
            )
        return gradient

    def gradient_by_parameter_shift(self, parameters: Sequence[float]) -> np.ndarray:
        """Gradient by the parameter-shift rule, for validation only.

        Correct only when every parameter drives exactly one Pauli rotation.
        :meth:`gradient` is used everywhere else; this exists so tests can check
        the two agree on circuits where the rule does apply.

        For a rotation ``exp(-i theta P / 2)``,

            ``dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2``
        """
        parameters = np.asarray(parameters, dtype=float)
        gradient = np.zeros_like(parameters)
        for index in range(parameters.size):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += PARAMETER_SHIFT
            minus[index] -= PARAMETER_SHIFT
            gradient[index] = (self.energy(plus) - self.energy(minus)) / 2.0
        return gradient

    def overlap_with_ground_state(self, parameters: Sequence[float]) -> float:
        """Squared overlap between the variational state and the true ground state.

        Returns ``nan`` when the exact reference was not computed. This is the
        quantity that says whether a good energy came from a good state or from
        accidental energy cancellation.
        """
        if math.isnan(self._exact):
            return float("nan")
        eigenvalues, vectors = np.linalg.eigh(self.hamiltonian.to_matrix())
        state = self.circuit_for(parameters).run().state
        if state.size != vectors.shape[0]:
            return float("nan")

        # A degenerate ground subspace admits many states of equal energy, so a
        # single eigenvector is the wrong target: report the total weight in the
        # whole degenerate subspace instead.
        ground = eigenvalues <= eigenvalues[0] + 1e-9
        projector = vectors[:, ground] @ vectors[:, ground].conj().T
        return float(np.real(state.conj() @ projector @ state))

    # -- driver ------------------------------------------------------------

    def run(
        self,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
        learning_rate: float = 0.1,
        max_line_search: int = 12,
    ) -> VQEResult:
        """Minimise the energy by gradient descent with backtracking.

        Parameters
        ----------
        max_iterations
            Cap on optimiser steps.
        tolerance
            Stop when the accepted energy improvement falls below this.
        learning_rate
            Initial step size along the negative gradient.
        max_line_search
            Number of backtracking halvings allowed per step.

        Returns
        -------
        VQEResult
            Best energy found, with the parameters that achieved it.
        """
        parameters = self.initial_parameters.copy()
        energy = self.energy(parameters)
        history = [energy]
        converged = False
        scale = learning_rate

        for _ in range(max_iterations):
            gradient = self.gradient(parameters)
            if not np.all(np.isfinite(gradient)):
                break
            norm = float(np.linalg.norm(gradient))
            if norm < tolerance:
                converged = True
                break

            direction = -gradient / norm
            step = scale
            improved = False
            for _ in range(max_line_search):
                candidate = parameters + step * direction
                candidate_energy = self.energy(candidate)
                if candidate_energy < energy - tolerance * 0.5:
                    parameters = candidate
                    improvement = energy - candidate_energy
                    energy = candidate_energy
                    improved = True
                    break
                step *= 0.5

            if not improved:
                # Either a local minimum or a step size too small to register.
                if scale < tolerance:
                    converged = True
                break

            history.append(energy)
            if improvement < tolerance:
                converged = True
                break

        return VQEResult(
            energy=energy,
            exact_energy=self._exact,
            parameters=parameters,
            history=history,
            converged=converged,
            ground_state_overlap=self.overlap_with_ground_state(parameters),
        )


def optimise_energy(
    hamiltonian: SparsePauliOp,
    ansatz: Ansatz,
    max_iterations: int = 200,
    learning_rate: float = 0.1,
    tolerance: float = 1e-8,
) -> VQEResult:
    """Convenience wrapper: build a :class:`VQE` and run it."""
    return VQE(hamiltonian, ansatz).run(
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )


def parameter_shift_gradient(
    energy_function: Callable[[Sequence[float]], float],
    parameters: Sequence[float],
    shift: float = PARAMETER_SHIFT,
) -> np.ndarray:
    """Parameter-shift gradient for a scalar objective.

    Uses the standard rule

        ``dE/dp = [E(p + shift) - E(p - shift)] / 2``

    The divisor is ``2``, *not* ``2 * shift``. The two agree for ``shift = 1``
    only, which is a convenient way to ship a rule that is silently off by a
    factor of ``pi/2``.

    Parameters
    ----------
    energy_function
        Callable mapping a parameter vector to a scalar energy.
    parameters
        Point at which to evaluate the gradient.
    shift
        The parameter-shift amount. ``pi/2`` is correct for a single gate of the
        form ``exp(-i p P / 2)``; ``pi/4`` for ``exp(-i p P)``. The rule is
        **invalid** when one parameter drives more than one gate, because the
        effective generator is then not a single Pauli.

    Returns
    -------
    numpy.ndarray
        Gradient with the same shape as ``parameters``.

    References
    ----------
    Schuld, M. & Killoran, N. (2019). Evaluating analytic gradients on quantum
    hardware. Phys. Rev. Lett. 122, 180504.
    """
    parameters = np.asarray(parameters, dtype=float)
    gradient = np.zeros_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += shift
        minus[index] -= shift
        gradient[index] = (energy_function(plus) - energy_function(minus)) / 2.0
    return gradient
