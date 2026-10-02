"""
Gate library and tensor contraction engine.

Basis convention
----------------
Qubit 0 is the most significant bit throughout this project::

    |q0 q1 ... q(n-1)>  <->  index q0 * 2**(n-1) + q1 * 2**(n-2) + ...

Tensor convention
-----------------
Statevectors are tensors with exactly ``n`` axes of size 2, i.e. shape
``(2,)*n``. Density operators are tensors with ``2n`` axes, row axes first.
The contraction helpers here treat *every* axis of the input as a system axis;
density-operator helpers in :mod:`qcpapers.core.density` build on them.

References
----------
Aaronson, S. (2021). Quantum Computing Since Democritus. (Gate definitions,
controlled-gate construction, big-endian basis ordering.)

Nielsen, M. A. & Chuang, I. L. (2010). Quantum Computation and Quantum
Information, 10th ed. Cambridge University Press, Sec. 4.5 and Appendix A.2
(controlled operations).
"""

from __future__ import annotations

import functools
from typing import Dict, List, Sequence, Tuple

import numpy as np

ComplexArray = np.ndarray

# ---------------------------------------------------------------------------
# Fixed 1-qubit gates
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = T.conj().T
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
SXDG = SX.conj().T

_I4 = np.eye(4, dtype=complex)
_I8 = np.eye(8, dtype=complex)

#: Clifford single-qubit group (used by Gottesman-Knill and Clifford data
#: regression). Campbell, Chandler & Terrill, PRX 2, 030021 (2012).
CLIFFORD_SINGLE: Dict[str, np.ndarray] = {
    "I": I2, "X": X, "Y": Y, "Z": Z, "H": H,
    "S": S, "SDG": SDG, "T": T, "TDG": TDG, "SX": SX, "SXDG": SXDG,
}

# ---------------------------------------------------------------------------
# Fixed 2-qubit gates (first factor acts on the first listed qubit)
# ---------------------------------------------------------------------------

CZ = np.diag([1, 1, 1, -1]).astype(complex)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
CY = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex
)
CH = 0.5 * np.array(
    [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]], dtype=complex
)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)
DCX = np.array(
    [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=complex
)

TWO_QUBIT_FIXED: Dict[str, np.ndarray] = {
    "CZ": CZ, "CNOT": CNOT, "CY": CY, "CH": CH,
    "SWAP": SWAP, "ISWAP": ISWAP, "DCX": DCX,
}

# ---------------------------------------------------------------------------
# Fixed 3-qubit gates
# ---------------------------------------------------------------------------

def _swap_control_matrix(control_weight: int, target_weights: Tuple[int, int]) -> np.ndarray:
    """Build a Fredkin gate as a basis-index permutation.

    ``control_weight`` is the positional weight of the control qubit and
    ``target_weights`` are the weights of the two qubits to exchange. Constructing
    the matrix by permuting basis indices (rather than typing entries) makes the
    bit ordering explicit and removes the class of bug where a hand-written row
    is duplicated, which silently produces a non-unitary matrix.
    """
    dim = 1 << (max(control_weight, *target_weights)).bit_length()
    mapping = {}
    for col in range(dim):
        row = col ^ target_weights[0] if col & control_weight else col
        row = row ^ target_weights[1] if col & control_weight else row
        mapping[col] = row
    out = np.zeros((dim, dim), dtype=complex)
    for col, row in mapping.items():
        out[row, col] = 1.0
    return out


# With qubit 0 the most significant bit, positional weights are
# q0 = 2**(n-1), q1 = 2**(n-2), q2 = 2**(n-3).
CCX = np.array(
    [[1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0]],
    dtype=complex,
)
CSWAP = _swap_control_matrix(control_weight=4, target_weights=(2, 1))

THREE_QUBIT_FIXED: Dict[str, np.ndarray] = {"CCX": CCX, "CSWAP": CSWAP}

# ---------------------------------------------------------------------------
# Parametric gates
# ---------------------------------------------------------------------------


def rx(theta: float) -> np.ndarray:
    """Rotation about the X axis, ``exp(-i theta/2 X)``. Nielsen & Chuang eq. 4.36."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """Rotation about the Y axis, ``exp(-i theta/2 Y)``."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Rotation about the Z axis, ``exp(-i theta/2 Z)``."""
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(complex)


def phase(theta: float) -> np.ndarray:
    """Diagonal phase gate ``P(theta) = diag(1, e^{i theta})``."""
    return np.diag([1.0, np.exp(1j * theta)]).astype(complex)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """Generic single-qubit gate ``U3(theta, phi, lambda)``."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


def ising(pauli_a: str, pauli_b: str, theta: float) -> np.ndarray:
    """``exp(-i theta/2 * A (x) B)`` for ``A, B`` in ``{X, Y, Z}``.

    These are the native two-qubit gates on superconducting (CPRZZ) and
    trapped-ion (Motional Z) platforms. See Google Quantum AI, Nature 571, 489
    (2019), and Bluvstein et al., Nature 581, 124 (2020).
    """
    a = {"X": X, "Y": Y, "Z": Z}[pauli_a.upper()]
    b = {"X": X, "Y": Y, "Z": Z}[pauli_b.upper()]
    return np.cos(theta / 2.0) * _I4 - 1j * np.sin(theta / 2.0) * np.kron(a, b)


def controlled_ising(pauli: str, theta: float) -> np.ndarray:
    """Controlled ``R_P(theta)``; the control qubit is the first factor.

    The result is ``diag(1, 1, 1, R_P(theta))``: the gate applies only when the
    first factor's qubit is in ``|1>``. This is the standard neutral-atom and
    superconducting entangler family (C-RXX, C-RYY, C-RZZ), and the controlled
    rotation identity follows Barenco et al., Phys. Rev. A 52, 3457 (1995).
    """
    p = {"X": X, "Y": Y, "Z": Z}[pauli.upper()]
    block = np.cos(theta / 2.0) * I2 - 1j * np.sin(theta / 2.0) * p
    out = np.zeros((4, 4), dtype=complex)
    out[:2, :2] = I2
    out[2:, 2:] = block
    return out


def cphase(theta: float) -> np.ndarray:
    """Controlled-phase gate ``diag(1, 1, 1, e^{i theta})``."""
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


#: Ising rotations by name.
ISING_ROTATIONS = {"RXX": ("X", "X"), "RYY": ("Y", "Y"), "RZZ": ("Z", "Z")}
#: Controlled rotations by name.
CONTROLLED_ROTATIONS = {"CRX": "X", "CRY": "Y", "CRZ": "Z"}

GATE_ALIASES = {
    "CX": "CNOT",
    "CXOT": "CNOT",
    "TOFFOLI": "CCX",
    "CCNOT": "CCX",
    "FSWAP": "CSWAP",
    "FREDKIN": "CSWAP",
    "PHASE": "P",
    "U3": "U",
    "CPHASE": "CP",
    "CXX": "CRX",
    "CYY": "CRY",
    "CZZ": "CRZ",
    # Two-qubit rotations are often named with the Pauli pair spelled out.
    "XX": "RXX",
    "YY": "RYY",
    "ZZ": "RZZ",
    # Exchange family, under the names used by IonQ and Rigetti.
    "ISWAP": "ISWAP",
    "iSWAP": "ISWAP",
}

ALL_GATES: Tuple[str, ...] = (
    "I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "SX", "SXDG",
    "RX", "RY", "RZ", "P", "U",
    *TWO_QUBIT_FIXED.keys(),
    *ISING_ROTATIONS.keys(),
    "RZX", *CONTROLLED_ROTATIONS.keys(), "CP",
    *THREE_QUBIT_FIXED.keys(),
)

#: Gates admitting a closed-form circuit inverse.
#:
#: ``DCX`` (double controlled-NOT) is deliberately excluded: it maps
#: ``|01> <-> |10>`` and ``|00> -> |11>``, so ``DCX**2 = X (x) X != I`` and its
#: inverse is ``(X (x) X) @ DCX``. ``ISWAP`` is excluded for the same class of
#: reason and is handled by :data:`_INVERSE_SEQUENCES`.
SELF_INVERSE_GATES = frozenset(
    {"I", "X", "Y", "Z", "H", "SWAP", "CX", "CNOT", "CY", "CZ", "CH", "CCX", "CSWAP"}
)

#: Generators of the Clifford group.
CLIFFORD_GENERATORS = frozenset({"H", "S", "SX", "CNOT", "CZ", "SWAP"})


def _normalise(name: str) -> str:
    return GATE_ALIASES.get(name.upper().replace("-", ""), name.upper().replace("-", ""))


@functools.lru_cache(maxsize=1024)
def gate_matrix(name: str, *params: float) -> np.ndarray:
    """Resolve a gate name plus parameters into a dense unitary matrix.

    The returned matrix is cached, so callers should treat it as read-only and
    copy before mutating.

    Parameters
    ----------
    name
        Case-insensitive gate name. Vendor nomenclature (``CX``, ``TOFFOLI``,
        ``FSWAP``) is normalised through :data:`GATE_ALIASES`.
    params
        Rotation angles in radians.

    Returns
    -------
    numpy.ndarray
        A ``(2, 2)``, ``(4, 4)`` or ``(8, 8)`` complex unitary.
    """
    key = _normalise(name)

    # Validate the parameter count before dispatching. Without this, a fixed gate
    # called with stray parameters (gate_matrix("CNOT", 0.3)) falls through every
    # branch and raises a misleading "unknown gate" error instead of saying the
    # parameter count is wrong.
    expected = expected_param_count(name)
    if len(params) != expected:
        raise ValueError(
            f"{name} expects {expected} parameter(s), got {len(params)}"
        )

    if expected == 0:
        if key in CLIFFORD_SINGLE:
            return CLIFFORD_SINGLE[key]
        if key in TWO_QUBIT_FIXED:
            return TWO_QUBIT_FIXED[key]
        return THREE_QUBIT_FIXED[key]

    if key in ("RX", "RY", "RZ", "P"):
        _require(name, params, 1)
        return {"RX": rx, "RY": ry, "RZ": rz, "P": phase}[key](params[0])

    if key == "U":
        _require(name, params, 3)
        return u3(*params)

    if key in ISING_ROTATIONS:
        _require(name, params, 1)
        a, b = ISING_ROTATIONS[key]
        return ising(a, b, params[0])

    if key == "RZX":
        _require(name, params, 1)
        return ising("X", "Z", params[0])

    if key in CONTROLLED_ROTATIONS:
        _require(name, params, 1)
        return controlled_ising(CONTROLLED_ROTATIONS[key], params[0])

    if key == "CP":
        _require(name, params, 1)
        return cphase(params[0])

    raise KeyError(f"unknown gate {name!r}; supported gates: {sorted(ALL_GATES)}")


#: Pauli string behind each *pure Pauli rotation*, in matrix-factor order. A
#: ``"C"`` entry marks the control qubit, whose factor is the projector
#: ``|1><1|`` rather than a Pauli matrix.
#:
#: ``P``, ``CP`` and ``U`` are deliberately absent: their derivatives do not
#: factor as a constant matrix times the unitary. Use
#: :func:`gate_parameter_derivative` for those.
GATE_PAULI: Dict[str, Tuple[str, ...]] = {
    "RX": ("X",),
    "RY": ("Y",),
    "RZ": ("Z",),
    "U": ("X",),
    "RXX": ("X", "X"),
    "RYY": ("Y", "Y"),
    "RZZ": ("Z", "Z"),
    "RZX": ("X", "Z"),
    "CRX": ("C", "X"),
    "CRY": ("C", "Y"),
    "CRZ": ("C", "Z"),
    "CP": ("C", "CP"),
}


def gate_generator(name: str) -> np.ndarray:
    """Matrix ``G`` with ``dU/dp = G @ U`` for a Pauli rotation.

    Valid for ``U = exp(-i p P / 2)`` with ``P`` a Pauli product, whose derivative
    is ``-i P U / 2``. For a *controlled* rotation the Pauli must act only when the
    control qubit is set, so the control factor becomes the projector
    ``|1><1|`` rather than a Pauli matrix.

    Parameters
    ----------
    name
        Gate name. Must be one of the pure Pauli rotations; for the diagonal
        phase gates and the three-parameter ``U`` gate use
        :func:`gate_parameter_derivative`, whose derivative does not factor as a
        constant matrix times ``U``.

    Returns
    -------
    numpy.ndarray
        Square matrix of dimension ``2**arity`` on the gate's own qubits.
    """
    key = _normalise(name)
    if key not in GATE_PAULI or not GATE_PAULI[key]:
        raise KeyError(
            f"{name!r} is not a pure Pauli rotation and has no constant generator; "
            "use gate_parameter_derivative"
        )
    factors: List[np.ndarray] = []
    for letter in GATE_PAULI[key]:
        if letter == "C":
            factors.append(np.diag([0.0, 1.0]).astype(complex))
        else:
            factors.append({"X": X, "Y": Y, "Z": Z}[letter])
    return -0.5j * _pauli_product(tuple(factors))


def gate_parameter_derivative(
    name: str, params: Sequence[float], param_index: int = 0
) -> np.ndarray:
    """Exact matrix ``dU/dp`` for a parametric gate.

    Handles every parametric gate in the library, including the ones whose
    derivative is not of the form ``G @ U`` with ``G`` independent of the
    parameters.

    Parameters
    ----------
    name
        Gate name; see :func:`gate_matrix`.
    params
        Gate parameters, in the order :func:`gate_matrix` expects.
    param_index
        Which parameter to differentiate.

    Returns
    -------
    numpy.ndarray
        The derivative matrix, on the gate's own qubits.

    Notes
    -----
    ``P`` and ``CP`` are ``exp(+i p)`` on a subspace, so their derivative is
    ``+i`` times a projector and does **not** factor as ``G @ U`` with a constant
    ``G``: for ``P(theta) = diag(1, e^{i theta})`` the derivative is
    ``diag(0, i e^{i theta})``, whereas ``G @ U`` with ``G = i |1><1|`` gives
    ``diag(0, i)``, missing the ``e^{i theta}`` factor. ``U3`` mixes its three
    Euler angles, so its generators do not commute and no single-``G`` form
    exists either.

    This is what makes analytic differentiation possible for a parameter shared by
    more than one gate, where the parameter-shift rule does not apply.
    """
    key = _normalise(name)
    count = expected_param_count(name)
    if not 0 <= param_index < count:
        raise IndexError(
            f"{name} has {count} parameter(s), index {param_index} is out of range"
        )
    values = [float(p) for p in params]

    if key == "P":
        unitary = gate_matrix("P", values[0])
        derivative = np.zeros_like(unitary)
        derivative[1, 1] = 1j * np.exp(1j * values[0])
        return derivative

    if key == "CP":
        theta = values[0]
        unitary = gate_matrix("CP", theta)
        derivative = np.zeros_like(unitary)
        derivative[3, 3] = 1j * np.exp(1j * theta)
        return derivative

    if key == "U":
        theta, phi, lam = values
        c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
        # The theta dependence enters only through cos(theta/2) and sin(theta/2),
        # so every term picks up a factor of 1/2.
        half = 0.5
        e_phi = np.exp(1j * phi)
        e_lam = np.exp(1j * lam)
        derivative = np.zeros((2, 2), dtype=complex)
        if param_index == 0:
            derivative[0, 0] = -half * s
            derivative[0, 1] = -half * e_lam * c
            derivative[1, 0] = half * e_phi * c
            derivative[1, 1] = -half * e_phi * e_lam * s
        elif param_index == 1:
            derivative[1, 0] = 1j * e_phi * s
            derivative[1, 1] = 1j * e_phi * e_lam * c
        else:
            derivative[0, 1] = -1j * e_lam * s
            derivative[1, 1] = 1j * e_phi * e_lam * c
        return derivative

    return gate_generator(name) @ gate_matrix(name, *values)


def _pauli_product(factors: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of 2x2 matrices, left to right."""
    matrix = np.array([[1.0]], dtype=complex)
    for factor in factors:
        matrix = np.kron(matrix, factor)
    return matrix


def _require(name: str, params: Sequence[float], count: int) -> None:
    if len(params) != count:
        raise ValueError(f"{name} expects {count} parameter(s), got {len(params)}")


def gate_arity(name: str) -> int:
    """Number of qubits a gate acts on, ignoring its parameters."""
    key = _normalise(name)
    if key in CLIFFORD_SINGLE or key in ("RX", "RY", "RZ", "P", "U"):
        return 1
    if key in TWO_QUBIT_FIXED or key in ISING_ROTATIONS or key in CONTROLLED_ROTATIONS:
        return 2
    if key in ("RZX", "CP"):
        return 2
    if key in THREE_QUBIT_FIXED:
        return 3
    raise KeyError(f"unknown gate {name!r}")


def expected_param_count(name: str) -> int:
    """How many rotation parameters a gate accepts.

    ``0`` for fixed gates (Paulis, Clifford gates, Toffoli, ...), ``1`` for
    rotations and phase gates, ``3`` for the generic ``U3`` gate.
    """
    key = _normalise(name)
    if key in CLIFFORD_SINGLE or key in TWO_QUBIT_FIXED or key in THREE_QUBIT_FIXED:
        return 0
    if key in ("RX", "RY", "RZ", "P", "U"):
        return 3 if key == "U" else 1
    if key in ISING_ROTATIONS or key == "RZX" or key in CONTROLLED_ROTATIONS or key == "CP":
        return 1
    raise KeyError(f"unknown gate {name!r}")


def gate_inverse(name: str, *params: float) -> np.ndarray:
    """Matrix of the inverse of ``name``.

    Taking a plain conjugate transpose of ``S`` or ``T`` is the wrong operation
    because their inverses are the named gates ``SDG`` and ``TDG``; this helper
    returns the same matrix either way but is the name-level counterpart used by
    :func:`gate_inverse_sequence`.
    """
    key = _normalise(name)
    if key in SELF_INVERSE_GATES:
        return gate_matrix(name, *params)
    if key == "S":
        return SDG
    if key == "T":
        return TDG
    if key == "SX":
        return SXDG
    return gate_matrix(name, *params).conj().T


#: Gates whose inverse is a *different named gate* rather than the same gate with
#: negated parameters.
_INVERSE_NAME_OVERRIDES = {
    "S": "SDG", "SDG": "S",
    "T": "TDG", "TDG": "T",
    "SX": "SXDG", "SXDG": "SX",
}

#: Parametric gates whose inverse is obtained by negating every parameter.
#: Rotation generators are anti-Hermitian, so exp(-i t G)^dg = exp(+i t G).
_NEGATE_PARAMS = frozenset({
    "RX", "RY", "RZ", "P", "CP", "RXX", "RYY", "RZZ", "RZX",
    "CRX", "CRY", "CRZ",
})

#: Exact inverse sequences for gates that are neither self-inverse nor obtained
#: by a parameter rename. Verified by :func:`gate_inverse_sequence`'s unit test.
#:
#: ISWAP satisfies ISWAP**2 = Z (x) Z, hence ISWAP^dg = (Z (x) Z) @ ISWAP, which
#: is realised in time order as: apply ISWAP, then Z on each qubit.
_INVERSE_SEQUENCES: Dict[str, Tuple[Tuple[str, Tuple[int, ...]], ...]] = {
    "ISWAP": (("ISWAP", (0, 1)), ("Z", (0,)), ("Z", (1,))),
}


def gate_inverse_sequence(
    name: str, params: Sequence[float], qubits: Sequence[int]
) -> List[Tuple[str, Tuple[int, ...], Tuple[float, ...]]]:
    """Decompose the inverse of a gate into named gates, in time order.

    Returns a list of ``(gate_name, target_qubits, parameters)`` triples whose
    product equals ``gate_matrix(name, *params).conj().T``.

    Notes
    -----
    ``qubits`` is given in matrix-factor order (the first factor acts on
    ``qubits[0]``); the returned triples use the same ordering, so a caller
    reversing a circuit's operations and reversing each operation's qubit list
    gets a correct adjoint without any conjugation bookkeeping.
    """
    key = _normalise(name)
    q = list(qubits)

    if key in _INVERSE_SEQUENCES:
        sequence = _INVERSE_SEQUENCES[key]
        return [(n, tuple(q[i] for i in offsets), ()) for n, offsets in sequence]

    if key in _INVERSE_NAME_OVERRIDES:
        return [(_INVERSE_NAME_OVERRIDES[key], tuple(q), ())]

    if key in SELF_INVERSE_GATES:
        return [(key, tuple(q), tuple(float(p) for p in params))]

    if key in _NEGATE_PARAMS:
        return [(key, tuple(q), tuple(-float(p) for p in params))]

    if key == "U":
        # U3(theta, phi, lam) factors as exp(i phi/2 Z) RX(theta) exp(i lam/2 Z)
        # up to a global phase, so adjointing exchanges and negates the two Z
        # angles: U3(theta, phi, lam)^dg = U3(-theta, -lam, -phi).
        theta, phi, lam = params
        return [(key, tuple(q), (-float(theta), -float(lam), -float(phi)))]

    if key == "DCX":
        # DCX^dg = (X (x) X) @ DCX, realised as: DCX, then X on both qubits.
        return [("DCX", tuple(q), ()), ("X", (q[0],), ()), ("X", (q[1],), ())]

    # Fall back to a generic adjoint; the caller validates with unit tests.
    return [(key, tuple(q), tuple(-float(p) for p in params))]


def controlled_gate(matrix: np.ndarray, num_controls: int) -> np.ndarray:
    """Build a controlled version of an arbitrary unitary.

    Embeds ``matrix`` in the *last* block of an identity of size
    ``2**(num_controls + 1)``, i.e.

        ``U_controlled = |0...0><0...0| (x) I + |0...01><0...01| (x) I + ... + |1...1><1...1| (x) U``

    where the control qubits are the leading (most significant) factors. This is
    the primitive behind Qiskit's ``cunitary``, Q#'s ``Controlled``, and Cirq's
    ``ControlledGate``.

    Concretely the block structure is ``blockdiag(I_2, ..., I_2, U)`` with
    ``num_controls + 1`` blocks, so ``total = 2**num_controls * dim`` and ``U``
    occupies rows and columns ``[dim:, dim:]``.

    References: Barenco et al., Phys. Rev. A 52, 3457 (1995); Cormane,
    Browne & Ekert, Phys. Rev. A 63, 042301 (2001).
    """
    matrix = np.asarray(matrix, dtype=complex)
    if num_controls < 0:
        raise ValueError("num_controls must be non-negative")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("controlled_gate expects a square matrix")
    if not np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-9):
        raise ValueError("controlled_gate expects a unitary matrix")
    dim = matrix.shape[0]
    if dim < 2 or dim & (dim - 1):
        raise ValueError("matrix dimension must be a power of two >= 2")

    total = dim << num_controls
    out = np.zeros((total, total), dtype=complex)
    for control_value in range(1 << num_controls):
        start = control_value * dim
        out[start : start + dim, start : start + dim] = (
            matrix if control_value == (1 << num_controls) - 1 else np.eye(dim)
        )
    return out


def is_clifford(name: str) -> bool:
    """Whether a single fixed gate belongs to the Clifford group.

    Clifford operations preserve Pauli operators under conjugation, which is what
    makes Gottesman-Knill simulation, randomized benchmarking and Clifford-data
    regression tractable (Nielsen & Chuang 2010, Sec. 10.5.3).
    """
    return _normalise(name) in CLIFFORD_GENERATORS or _normalise(name) in CLIFFORD_SINGLE


# ---------------------------------------------------------------------------
# Tensor contraction
# ---------------------------------------------------------------------------


def _move_perm(ndim: int, axes: Sequence[int]) -> list[int]:
    axes = list(axes)
    if len(set(axes)) != len(axes):
        raise ValueError(f"repeated qubit index in {axes}")
    if any(a < 0 or a >= ndim for a in axes):
        raise ValueError(f"qubit index out of range for a {ndim}-axis tensor")
    return axes + [a for a in range(ndim) if a not in axes]


def apply_matrix(tensor: np.ndarray, matrix: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Apply ``matrix`` to the given qubit axes of a statevector tensor.

    ``tensor`` has one axis per qubit, so ``tensor.ndim`` equals the number of
    qubits. ``matrix`` must have dimension ``2**len(qubits)`` and its first
    factor acts on ``qubits[0]``.

    Implemented by moving the target axes to the front, flattening them, and
    performing one dense matmul. This is the same contraction order used by
    Qiskit's statevector backend (IBM, arXiv:1908.07000) and avoids building
    the full ``2**n x 2**n`` operator matrix.
    """
    matrix = np.asarray(matrix, dtype=complex)
    qubits = list(qubits)
    dim = matrix.shape[0] if matrix.ndim == 2 else None
    if dim is None:
        raise ValueError("matrix must be 2-dimensional")
    if dim != 2 ** len(qubits):
        raise ValueError(f"gate of dimension {dim} does not act on {len(qubits)} qubits")

    perm = _move_perm(tensor.ndim, qubits)
    flat = tensor.transpose(perm).reshape(dim, -1)
    out = matrix @ flat
    return out.reshape(tensor.shape).transpose(np.argsort(perm))


def to_tensor(vector: np.ndarray) -> np.ndarray:
    """Reshape a flat ``2**n`` statevector into a ``(2,)*n`` tensor."""
    vector = np.asarray(vector, dtype=complex)
    if vector.ndim != 1:
        raise ValueError(f"expected a flat vector, got shape {vector.shape}")
    n = int(round(np.log2(vector.size)))
    if 2**n != vector.size:
        raise ValueError(f"length {vector.size} is not a power of two")
    return vector.reshape((2,) * n)


def from_tensor(tensor: np.ndarray) -> np.ndarray:
    """Flatten a ``(2,)*n`` statevector tensor back to a ``2**n`` vector."""
    return np.asarray(tensor).reshape(-1)


def zero_state(n_qubits: int) -> np.ndarray:
    """The all-zero state ``|0...0>`` as a ``(2,)*n`` tensor."""
    return basis_state(n_qubits, 0)


def _basis_index(n_qubits: int, index: int) -> np.ndarray:
    tensor = np.zeros((2,) * n_qubits, dtype=complex)
    idx: list[int] = []
    for k in range(n_qubits - 1, -1, -1):
        idx.append((index >> k) & 1)
    tensor[tuple(idx)] = 1.0
    return tensor


def basis_state(n_qubits: int, index: int) -> np.ndarray:
    """Computational basis state ``|index>`` as a ``(2,)*n`` tensor."""
    if not 0 <= index < 2**n_qubits:
        raise ValueError(f"index {index} out of range for {n_qubits} qubits")
    return _basis_index(n_qubits, index)


def apply_gate(
    tensor: np.ndarray,
    name: str,
    qubits: Sequence[int],
    *params: float,
) -> np.ndarray:
    """Apply any supported gate to a statevector tensor.

    This is the single dispatch point used by :class:`~qcpapers.core.circuit.Circuit`
    and by every algorithm in this package.

    Parameters
    ----------
    tensor
        Statevector with shape ``(2,)*n``.
    name
        Gate name; see :func:`gate_matrix`.
    qubits
        Target qubits, in matrix-factor order.
    params
        Gate parameters in radians.
    """
    matrix = gate_matrix(name, *params)
    expected = 2 ** len(qubits)
    if matrix.shape[0] != expected:
        raise ValueError(
            f"{name} is a {gate_arity(name)}-qubit gate but was given {len(qubits)} qubits"
        )
    return apply_matrix(tensor, matrix, qubits)
