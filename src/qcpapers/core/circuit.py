"""
Statevector circuit simulator.

A dependency-light, fully vectorised reference simulator. It is deliberately
simple and exact: correctness and traceability to the papers matter more than
raw throughput, and every algorithm in :mod:`qcpapers.algorithms` is checked
against closed-form or classical oracles using this backend.

References
----------
Nielsen, M. A. & Chuang, I. L. (2010). Quantum Computation and Quantum
Information, 10th ed. Cambridge University Press.

IBM Quantum (2019). QasmSimulator: a high-performance quantum circuit simulator
with qubit reuse and no-nise default, arXiv:1908.07000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from qcpapers.core.gates import (
    apply_gate,
    apply_matrix,
    basis_state,
    controlled_gate,
    expected_param_count,
    from_tensor,
    gate_arity,
    gate_matrix,
    gate_parameter_derivative,
    to_tensor,
    zero_state,
)

#: Sentinel used in place of an unresolved symbolic parameter.
UNBOUND = "__unbound__"


@dataclass(frozen=True)
class Operation:
    """A single gate application on a circuit.

    ``params`` may contain ``str`` entries, in which case the operation is
    symbolic until :meth:`Circuit.assign` binds it to a float. This is what lets
    variational algorithms reuse one compiled circuit across optimiser
    iterations instead of rebuilding it every time.

    ``matrix`` is set only for operations added by :meth:`Circuit.unitary_gate`,
    which attaches an arbitrary dense unitary to specific qubits. This mirrors
    Qiskit's ``UnitaryGate`` and is what :func:`~qcpapers.algorithms.qpe.qpe`
    uses to apply matrix powers of a diagonal-phase unitary.
    """

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[object, ...] = ()
    matrix: Optional[np.ndarray] = None

    def with_params(self, *params: object) -> "Operation":
        return Operation(self.name, self.qubits, tuple(params), self.matrix)

    def is_bound(self) -> bool:
        return self.matrix is not None or not any(isinstance(p, str) for p in self.params)

    def __str__(self) -> str:
        qubits = ", ".join(f"q{q}" for q in self.qubits)
        if self.name == "UNITARY":
            return f"UNITARY[{self.matrix.shape}] {qubits}"
        if self.params:
            args = ", ".join(
                f"{p:.4f}" if isinstance(p, (int, float)) else str(p) for p in self.params
            )
            return f"{self.name}({args}) {qubits}"
        return f"{self.name} {qubits}"


@dataclass
class StateVectorResult:
    """Outcome of a simulation run.

    Attributes
    ----------
    state
        The final ``2**n`` complex amplitude vector, normalised to unit norm to
        within floating point precision.
    probabilities
        ``abs(state)**2``, summing to 1.
    """

    state: np.ndarray
    probabilities: np.ndarray

    @property
    def n_qubits(self) -> int:
        return int(round(math.log2(self.state.size)))

    def top_states(self, k: int = 5) -> List[Tuple[str, float]]:
        """Return the ``k`` most probable outcomes as ``(bitstring, probability)``."""
        n = self.n_qubits
        order = np.argsort(self.probabilities)[::-1][:k]
        return [(format(int(i), f"0{n}b"), float(self.probabilities[i])) for i in order]


class Circuit:
    """A quantum circuit over ``n_qubits``.

    Parameters
    ----------
    n_qubits
        Circuit width.
    name
        Free-form label, used when serialising or printing.
    seed
        Default RNG seed for :meth:`sample` and :meth:`measure`.
    metadata
        Arbitrary payload; algorithms stash paper references and circuit
        hyperparameters here.

    Examples
    --------
    >>> c = Circuit(2).h(0).cnot(0, 1)
    >>> result = c.run()
    >>> result.top_states(1)[0][0]
    '00'
    """

    def __init__(
        self,
        n_qubits: int,
        name: str = "circuit",
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("a circuit needs at least one qubit")
        self.n_qubits = int(n_qubits)
        self.name = name
        self.metadata: dict = dict(metadata or {})
        self._operations: List[Operation] = []
        self._rng = np.random.default_rng(seed)

    # -- construction ------------------------------------------------------

    def unitary_gate(self, matrix: np.ndarray, qubits: Sequence[int]) -> "Circuit":
        """Attach an arbitrary dense unitary to specific qubits.

        Used for operations that have no fixed named-gate form: matrix powers of
        a unitary (QPE), oracle matrices (Grover, amplitude amplification), and
        Hamiltonians read in the computational basis.

        Parameters
        ----------
        matrix
            Square unitary of dimension ``2**len(qubits)``. Its first factor acts
            on ``qubits[0]``.
        qubits
            Target qubits, in matrix-factor order.
        """
        matrix = np.asarray(matrix, dtype=complex)
        qubits = tuple(int(q) for q in qubits)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("unitary_gate expects a square matrix")
        if matrix.shape[0] != 2 ** len(qubits):
            raise ValueError(
                f"matrix of dimension {matrix.shape[0]} does not act on {len(qubits)} qubits"
            )
        if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-9):
            raise ValueError("unitary_gate expects a unitary matrix")
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit index {q} out of range")
        self._operations.append(Operation("UNITARY", qubits, (), matrix))
        return self

    def oracle_diagonal(self, phases: Sequence[complex], qubits: Sequence[int]) -> "Circuit":
        """Attach a diagonal unitary ``diag(phases)``.

        This is the phase oracle ``S_f`` of Grover search and of amplitude
        amplification, expressed without an ancilla register by building the
        diagonal directly. Cheaper and clearer than decomposing it into
        multi-controlled gates when only the statevector semantics matter.
        """
        phases = np.asarray(phases, dtype=complex)
        dim = 2 ** len(list(qubits))
        if phases.size != dim:
            raise ValueError(f"expected {dim} phases, got {phases.size}")
        return self.unitary_gate(np.diag(phases), qubits)

    def controlled_unitary(
        self, matrix: np.ndarray, control: int, targets: Sequence[int]
    ) -> "Circuit":
        """Apply ``matrix`` to ``targets`` conditioned on ``control`` being ``|1>``.

        Used by QPE to build the controlled matrix powers ``U**(2**k)``.
        """
        matrix = np.asarray(matrix, dtype=complex)
        full = controlled_gate(matrix, 1)
        return self.unitary_gate(full, [control, *targets])

    def add(self, name: str, qubits: Sequence[int], *params: object) -> "Circuit":
        """Append a gate and return ``self`` so calls can be chained.

        Raises
        ------
        ValueError
            If a qubit index is out of range or the gate arity is wrong.
        """
        qubits = tuple(int(q) for q in qubits)
        arity = gate_arity(name)
        if len(qubits) != arity:
            raise ValueError(f"{name} acts on {arity} qubit(s), got {len(qubits)}")
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit index {q} out of range for {self.n_qubits} qubits")

        n_params = expected_param_count(name)
        if len(params) != n_params:
            raise ValueError(
                f"{name} expects {n_params} parameter(s), got {len(params)}"
            )

        self._operations.append(Operation(name, qubits, tuple(params)))
        return self

    def barrier(self, *qubits: int) -> "Circuit":
        """Insert a barrier; a no-op for the simulator but preserved for export."""
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit index {q} out of range")
        self._operations.append(Operation("BARRIER", tuple(int(q) for q in qubits)))
        return self

    # -- convenience gate methods -----------------------------------------

    def i(self, q: int) -> "Circuit":
        return self.add("I", [q])

    def x(self, q: int) -> "Circuit":
        return self.add("X", [q])

    def y(self, q: int) -> "Circuit":
        return self.add("Y", [q])

    def z(self, q: int) -> "Circuit":
        return self.add("Z", [q])

    def h(self, q: int) -> "Circuit":
        return self.add("H", [q])

    def s(self, q: int) -> "Circuit":
        return self.add("S", [q])

    def sdg(self, q: int) -> "Circuit":
        return self.add("SDG", [q])

    def t(self, q: int) -> "Circuit":
        return self.add("T", [q])

    def tdg(self, q: int) -> "Circuit":
        return self.add("TDG", [q])

    def sx(self, q: int) -> "Circuit":
        return self.add("SX", [q])

    def rx(self, theta: object, q: int) -> "Circuit":
        return self.add("RX", [q], theta)

    def ry(self, theta: object, q: int) -> "Circuit":
        return self.add("RY", [q], theta)

    def rz(self, theta: object, q: int) -> "Circuit":
        return self.add("RZ", [q], theta)

    def p(self, theta: object, q: int) -> "Circuit":
        return self.add("P", [q], theta)

    def u(self, theta: object, phi: object, lam: object, q: int) -> "Circuit":
        return self.add("U", [q], theta, phi, lam)

    def cx(self, control: int, target: int) -> "Circuit":
        return self.add("CNOT", [control, target])

    def cnot(self, control: int, target: int) -> "Circuit":
        return self.add("CNOT", [control, target])

    def cy(self, control: int, target: int) -> "Circuit":
        return self.add("CY", [control, target])

    def cz(self, control: int, target: int) -> "Circuit":
        return self.add("CZ", [control, target])

    def ch(self, control: int, target: int) -> "Circuit":
        return self.add("CH", [control, target])

    def swap(self, q0: int, q1: int) -> "Circuit":
        return self.add("SWAP", [q0, q1])

    def iswap(self, q0: int, q1: int) -> "Circuit":
        return self.add("ISWAP", [q0, q1])

    def dcx(self, q0: int, q1: int) -> "Circuit":
        """Double controlled-NOT (the ``iSWAP``-family gate IBM names ``DCX``).

        Maps ``|01> <-> |10>`` and ``|00> -> |11>``, so ``DCX**2 = X (x) X``.
        """
        return self.add("DCX", [q0, q1])

    def rxx(self, theta: object, q0: int, q1: int) -> "Circuit":
        return self.add("RXX", [q0, q1], theta)

    def ryy(self, theta: object, q0: int, q1: int) -> "Circuit":
        return self.add("RYY", [q0, q1], theta)

    def rzz(self, theta: object, q0: int, q1: int) -> "Circuit":
        return self.add("RZZ", [q0, q1], theta)

    def rzx(self, theta: object, q0: int, q1: int) -> "Circuit":
        return self.add("RZX", [q0, q1], theta)

    def crx(self, theta: object, control: int, target: int) -> "Circuit":
        return self.add("CRX", [control, target], theta)

    def cry(self, theta: object, control: int, target: int) -> "Circuit":
        return self.add("CRY", [control, target], theta)

    def crz(self, theta: object, control: int, target: int) -> "Circuit":
        return self.add("CRZ", [control, target], theta)

    def cp(self, theta: object, control: int, target: int) -> "Circuit":
        return self.add("CP", [control, target], theta)

    def ccx(self, c0: int, c1: int, target: int) -> "Circuit":
        return self.add("CCX", [c0, c1, target])

    def cswap(self, control: int, t0: int, t1: int) -> "Circuit":
        return self.add("CSWAP", [control, t0, t1])

    # -- symbolic parameters ----------------------------------------------

    def parameter_names(self) -> List[str]:
        """Sorted unique symbolic parameter names appearing in the circuit."""
        names = {p for op in self._operations for p in op.params if isinstance(p, str)}
        return sorted(names)

    def num_parameters(self) -> int:
        return len(self.parameter_names())

    def assign(self, values: Mapping[str, float]) -> "Circuit":
        """Return a copy with every symbolic parameter bound from ``values``."""
        missing = [n for n in self.parameter_names() if n not in values]
        if missing:
            raise KeyError(f"no value supplied for parameters: {missing}")
        ops = [
            Operation(op.name, op.qubits, tuple(values[p] if isinstance(p, str) else p
                                               for p in op.params))
            for op in self._operations
        ]
        clone = Circuit(self.n_qubits, self.name, metadata=dict(self.metadata))
        clone._operations = ops
        return clone

    def assign_array(self, values: Sequence[float]) -> "Circuit":
        """Bind the circuit's symbolic parameters in sorted-name order."""
        names = self.parameter_names()
        if len(values) != len(names):
            raise ValueError(f"expected {len(names)} values, got {len(values)}")
        return self.assign(dict(zip(names, values)))

    # -- execution ---------------------------------------------------------

    def state_tensor(self, initial: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the final state as a ``(2,)*n`` tensor.

        Raises
        ------
        ValueError
            If any operation still carries an unbound parameter.
        """
        tensor = zero_state(self.n_qubits) if initial is None else self._as_tensor(initial)
        for op in self._operations:
            if op.name in ("BARRIER", "MEASURE"):
                continue
            if op.matrix is not None:
                tensor = apply_matrix(tensor, op.matrix, op.qubits)
                continue
            params = self._resolve(op.params, op)
            tensor = apply_gate(tensor, op.name, op.qubits, *params)
        return tensor

    def state_and_derivatives(
        self,
        parameters: Sequence[float],
        initial: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward-mode derivatives of the state with respect to each parameter.

        Uses the tangent-state method: alongside ``|psi>`` a tangent vector
        ``d|psi>/dp`` is propagated through the circuit, seeded at the first gate
        that carries ``p`` and carried forward by the product rule thereafter.

        This is needed because the parameter-shift rule silently returns wrong
        numbers when one parameter appears in more than one gate. It also gives
        the derivative in a single pass instead of ``2 * n_parameters`` circuit
        evaluations.

        Parameters
        ----------
        parameters
            Values for every symbolic parameter, in sorted-name order.
        initial
            Optional initial state tensor.

        Returns
        -------
        tuple
            ``(state_tensor, {parameter_name: derivative_tensor})``. Both are
            ``(2,)*n`` tensors.

        Notes
        -----
        For a gate ``U`` with ``dU/dp``, the update rule is
        ``d(U|psi>)/dp = (dU/dp)|psi> + U d|psi>/dp``.

        References: differentiation of quantum circuits, Ito & Nakajima,
        arXiv:2003.09477 (2020); Gacon et al., Quantum 6, 409 (2022), for the
        tangent-state formulation.
        """
        names = self.parameter_names()
        if len(parameters) != len(names):
            raise ValueError(f"expected {len(names)} parameters, got {len(parameters)}")
        values = dict(zip(names, [float(p) for p in parameters]))

        state = (
            zero_state(self.n_qubits) if initial is None else self._as_tensor(initial)
        )
        tangents = {name: np.zeros_like(state) for name in names}
        seeded = {name: False for name in names}

        for op in self._operations:
            if op.name in ("BARRIER", "MEASURE"):
                continue

            if op.matrix is not None:
                # A raw unitary gate carries no symbolic parameters.
                state = apply_matrix(state, op.matrix, op.qubits)
                for name in tangents:
                    tangents[name] = apply_matrix(tangents[name], op.matrix, op.qubits)
                continue

            bound = [
                values[p] if isinstance(p, str) else float(p) for p in op.params
            ]
            symbolic = [p for p in op.params if isinstance(p, str)]
            matrix = gate_matrix(op.name, *bound)

            # Carry every existing tangent through this gate first: U d|psi>.
            incoming = {
                name: apply_matrix(tangent, matrix, op.qubits)
                for name, tangent in tangents.items()
            }

            # Seed at the first gate that mentions each parameter: (dU/dp)|psi>.
            # Later occurrences flow through the product rule automatically.
            for name in symbolic:
                derivative = gate_parameter_derivative(
                    op.name, bound, self._parameter_position(op, name)
                )
                incoming[name] = incoming[name] + apply_matrix(
                    state, derivative, op.qubits
                )
                seeded[name] = True

            state = apply_matrix(state, matrix, op.qubits)
            tangents = incoming

        return state, tangents

    @staticmethod
    def _parameter_position(op: Operation, name: str) -> int:
        """Index of a parameter within its gate's own parameter list.

        ``gate_matrix`` takes a gate's parameters in order, so a symbolic slot
        must be mapped to that position in ``op.params``. Using the position among
        the *symbolic* slots alone silently differentiates the wrong Euler angle
        on multi-parameter gates such as ``U``, where ``phi`` and ``lam`` sit in
        different slots but both take one argument.
        """
        return list(op.params).index(name)

    def run(self, initial: Optional[np.ndarray] = None) -> StateVectorResult:
        """Simulate exactly and return the statevector plus probabilities."""
        tensor = self.state_tensor(initial)
        vector = from_tensor(tensor)
        return StateVectorResult(state=vector, probabilities=np.abs(vector) ** 2)

    def probabilities(self, initial: Optional[np.ndarray] = None) -> np.ndarray:
        return self.run(initial).probabilities

    def sample(self, shots: int = 1024, initial: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw ``shots`` measurement outcomes as integers, using the exact
        Born rule. No shot noise is injected beyond this sampling.
        """
        if shots < 1:
            raise ValueError("shots must be positive")
        probs = self.probabilities(initial)
        return self._rng.choice(probs.size, size=shots, p=probs)

    def measure(self, shots: int = 1024, qubits: Optional[Sequence[int]] = None) -> np.ndarray:
        """Sample bit outcomes, optionally projecting onto ``qubits``.

        When ``qubits`` is given the returned array has shape ``(shots, len(qubits))``
        of bits in the requested order. Otherwise full-register integers are
        returned, matching Qiskit's ``memory`` semantics.
        """
        samples = self.sample(shots)
        if qubits is None:
            return samples
        bits = np.empty((shots, len(qubits)), dtype=np.int8)
        for k, q in enumerate(qubits):
            bits[:, k] = (samples >> (self.n_qubits - 1 - q)) & 1
        return bits

    def counts(
        self,
        shots: int = 4096,
        qubits: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        """Measurement counts keyed by bitstring, most frequent first.

        Parameters
        ----------
        shots
            Number of samples drawn from the exact Born distribution.
        qubits
            Restrict (and order) the reported bits. ``None`` returns full-register
            bitstrings with qubit 0 as the most significant bit, matching Qiskit.
        seed
            When given, reseed the sampler for reproducible counts. Useful in
            tests; omit it in production so repeated calls differ.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if qubits is None:
            samples = self.sample(shots)
            n = self.n_qubits
            keys = [format(int(s), f"0{n}b") for s in samples]
        else:
            samples = self.measure(shots, qubits)
            keys = ["".join(map(str, row)) for row in samples]

        tally: Dict[str, int] = {}
        for key in keys:
            tally[key] = tally.get(key, 0) + 1
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))

    # -- expectation values ------------------------------------------------

    def expectation(self, pauli_string: str) -> float:
        """Exact expectation value of a Pauli string such as ``"XIZ"``.

        Uses the sparse form of the observable: apply the Pauli factors, take the
        expectation of the result, which is a sum of probabilities with signs.
        No shots are involved, so this is the reference against which
        measurement-based estimators in :mod:`qcpapers.error_mitigation` are
        validated.
        """
        result = self.expectation_state(pauli_string)
        return float(np.real(result))

    def expectation_state(self, pauli_string: str) -> complex:
        """Expectation of a Pauli string as a complex number (keeps phase info)."""
        from qcpapers.core.observables import pauli_matrix_product

        return pauli_matrix_product(self.state_tensor(), pauli_string)

    def hamiltonian_expectation(self, hamiltonian) -> float:
        """Expectation of a :class:`~qcpapers.core.observables.SparsePauliOp`."""
        return float(np.real(hamiltonian.expectation(self.state_tensor())))

    # -- structural helpers ------------------------------------------------

    def unitary(self, n_qubits: int | None = None) -> np.ndarray:
        """Dense unitary matrix of the circuit.

        Builds a ``2**n x 2**n`` matrix by applying the circuit to every column
        of the identity. Only used in tests and for unitary checks; it scales as
        ``4**n`` memory.
        """
        n = n_qubits or self.n_qubits
        dim = 2**n
        out = np.zeros((dim, dim), dtype=complex)
        for col in range(dim):
            basis = basis_state(n, col)
            out[:, col] = from_tensor(self.state_tensor(basis))
        return out

    def depth(self) -> int:
        """Circuit depth: the number of gate layers in the longest chain.

        Computed by as-soon-as-possible scheduling. Each gate is placed in the
        earliest layer strictly after every gate that already touches one of its
        qubits, so independent gates share a layer. A ``BARRIER`` on a qubit
        forces all later operations on that qubit into a fresh layer, which is
        what makes barriers meaningful for compilation.

        This matches the definition used by Qiskit's ``circuit.depth()``.
        """
        if not self._operations:
            return 0

        next_free: Dict[int, int] = {}
        deepest = -1
        for op in self._operations:
            if op.name == "BARRIER":
                layer = max(next_free.values(), default=0)
                for q in op.qubits:
                    next_free[q] = layer
                deepest = max(deepest, layer)
                continue
            # ``next_free[q]`` is the earliest layer at which qubit q is
            # unoccupied, so a gate needs no offset beyond the maximum.
            layer = max((next_free.get(q, 0) for q in op.qubits), default=0)
            for q in op.qubits:
                next_free[q] = layer + 1
            deepest = max(deepest, layer)
        return deepest + 1

    def count_ops(self) -> Dict[str, int]:
        """Tally of gate applications by name."""
        tally: Dict[str, int] = {}
        for op in self._operations:
            if op.name in ("BARRIER", "MEASURE"):
                continue
            tally[op.name] = tally.get(op.name, 0) + 1
        return tally

    def swap_needed_qubits(self) -> int:
        """Number of qubits touched by at least one non-barrier operation."""
        touched = {q for op in self._operations if op.name != "BARRIER" for q in op.qubits}
        return len(touched)

    def compose(self, other: "Circuit", front: bool = False) -> "Circuit":
        """Return a new circuit with ``other`` applied after ``self``.

        Parameters
        ----------
        other
            Circuit to append.
        front
            When ``True``, prepend ``other`` instead, so it runs first.

        Returns
        -------
        Circuit
            A **new** circuit; ``self`` is left untouched. This is deliberate:
            ``compose`` returning ``self`` invites in-place mutation of a circuit
            that is also aliased elsewhere, and makes the order of a chain of
            calls easy to get wrong. Use :meth:`add` to mutate in place.

        Examples
        --------
        >>> a = Circuit(1).h(0)
        >>> b = Circuit(1).x(0)
        >>> len(a)
        1
        >>> len(a.compose(b))
        2
        >>> len(a)
        1
        """
        if other.n_qubits > self.n_qubits:
            raise ValueError("cannot compose a wider circuit onto a narrower one")
        ops = list(self._operations) + list(other._operations)
        if front:
            ops = list(other._operations) + list(self._operations)
        out = Circuit(self.n_qubits, f"{self.name}->{other.name}", metadata=dict(self.metadata))
        out._operations = ops
        return out

    def inverse(self) -> "Circuit":
        """Return a circuit computing the adjoint of this one.

        Implemented by reversing the operation order and replacing each gate with
        its inverse sequence from
        :func:`~qcpapers.core.gates.gate_inverse_sequence`. Target qubits are left
        in their original order: the gate matrix already encodes the factor
        ordering, so ``M^dg`` is applied to the same qubit tuple. The result is an
        executable circuit, not a matrix.

        Verified in ``tests/test_core_circuit.py`` by checking
        ``U.inverse().unitary() == U.unitary().conj().T`` for circuits containing
        every gate family, including ``ISWAP`` whose inverse is not a single
        named gate.
        """
        from qcpapers.core.gates import gate_inverse_sequence

        out = Circuit(self.n_qubits, f"{self.name}.dg", metadata=dict(self.metadata))
        for op in reversed(self._operations):
            if op.name == "BARRIER":
                continue
            if op.matrix is not None:
                out.unitary_gate(op.matrix.conj().T, list(op.qubits))
                continue
            params = self._resolve(op.params, op)
            for gate, qubits, gate_params in gate_inverse_sequence(
                op.name, params, list(op.qubits)
            ):
                out.add(gate, list(qubits), *gate_params)
        return out

    def copy(self, new_name: Optional[str] = None) -> "Circuit":
        out = Circuit(self.n_qubits, new_name or self.name, metadata=dict(self.metadata))
        out._operations = list(self._operations)
        return out

    # -- introspection -----------------------------------------------------

    @property
    def operations(self) -> List[Operation]:
        return list(self._operations)

    def __len__(self) -> int:
        return len(self._operations)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self._operations)

    def __repr__(self) -> str:
        return f"Circuit(name={self.name!r}, n_qubits={self.n_qubits}, ops={len(self)})"

    def draw(self, max_lines: int = 40) -> str:
        """Plain-text listing of the circuit, useful in notebooks and logs."""
        lines = [f"{self.name}  ({self.n_qubits} qubits, {len(self._operations)} ops)"]
        shown = self._operations[:max_lines]
        for op in shown:
            lines.append(f"  {str(op)}")
        if len(self._operations) > max_lines:
            lines.append(f"  ... {len(self._operations) - max_lines} more operations")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _resolve(params: Sequence[object], op: Operation) -> Tuple[float, ...]:
        resolved = []
        for p in params:
            if isinstance(p, str):
                raise ValueError(
                    f"operation {op} still has unbound parameter {p!r}; "
                    "call Circuit.assign(...) first"
                )
            resolved.append(float(p))  # type: ignore[arg-type]
        return tuple(resolved)

    def _as_tensor(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=complex)
        if state.ndim == 1:
            tensor = to_tensor(state)
        else:
            tensor = state
        if tensor.shape != (2,) * self.n_qubits:
            raise ValueError(
                f"initial state has shape {tensor.shape}, expected {(2,) * self.n_qubits}"
            )
        return tensor
