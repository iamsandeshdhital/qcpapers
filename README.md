# qcpapers

Reproducible reference implementations of landmark quantum computing papers.

Author: **iamsandeshdhital** <mesandeshdhital@gmail.com>
Repository: https://github.com/iamsandeshdhital/qcpapers

**Status: early. Six algorithm modules, 362 tests.** This README describes only
what is implemented and verified. Planned work is listed separately at the end,
clearly marked as not yet built.

---

## Design goals

1. **Traceability.** Every module carries the full citation of the paper it
   reproduces in its docstring, together with the parameters the paper reports.
2. **Runs with minimal dependencies.** The simulator is pure NumPy, so every
   test runs on a laptop with no SDK, no API key and no GPU.
3. **Exact analytic ground truth.** Every implementation ships a closed-form or
   classically derived oracle so the quantum result is checked against something
   stronger than "it looked plausible".
4. **Physically honest.** No shot noise unless a shot count is requested. No
   quantity that is not a probability is named as one.

---

## What is implemented

| Paper | Module | Oracle used in tests |
| --- | --- | --- |
| Grover, *Phys. Rev. Lett.* **79**, 533 (1996) | `algorithms/grover.py` | `sin^2((2k+1)theta)`, iteration bound `round(pi/(4 theta) - 1/2)` |
| Brassard, Hoyer & Mosca, *SIGACT News* **28**, 79 (1997) | `algorithms/grover.py` | closed form with `chi`/dim substituted for the marked count |
| Kitaev, arXiv:quant-ph/9505011 (1995) | `algorithms/qpe.py` | dense QFT matrix; exact DFT `sin(N pi d)^2 / (N sin(pi d))^2` |
| Griffiths & Niu, *Phys. Rev. Lett.* **76**, 522 (1996) | `algorithms/qpe.py` | exact agreement with full-register QPE on the grid |
| Peruzzo et al., arXiv:1304.3061 (2014) | `algorithms/vqe.py` | `numpy.linalg.eigvalsh` ground energy; finite-difference gradients |
| O'Gorman & Sanders, *Phys. Rev. Lett.* **120**, 170503 (2018) | `algorithms/vqe.py` | particle-number subspace diagonalised directly |
| Farhi, Goldstone & Gutmann, arXiv:1411.4028 (2014) | `algorithms/qaoa.py` | brute-force maximum cut over all `2^n` assignments |
| Childs & Goldstone, *Phys. Rev. A* **89**, 042314 (2004) | `algorithms/qaoa.py` | variational bound; cost evaluated on every basis state |
| Harrow, Hassenfeld & Lloyd, *Phys. Rev. Lett.* **102**, 090502 (2009) | `algorithms/hhl.py` | `numpy.linalg.solve`, entry by entry |
| Childs, Braun & MacCleary, arXiv:1804.09128 (2019) | `algorithms/hhl.py` | LCU decomposition reconstructed and compared to `A` |
| Huang, Harrow & Low, arXiv:2006.16719 (2020) | `algorithms/classical_shadows.py` | `Tr(P rho)` from the dense state |
| Chen, Harrow & Low, arXiv:2009.07472 (2021) | `algorithms/classical_shadows.py` | `1/sqrt(N)` convergence rate; shadow-norm bound |

### Supporting layer

| Module | Contents |
| --- | --- |
| `core/gates.py` | 33 gates, controlled-gate construction, exact per-parameter derivatives |
| `core/circuit.py` | Statevector simulator, symbolic parameters, dense unitary gates, forward-mode tangent states |
| `core/observables.py` | Pauli strings, `SparsePauliOp`, commuting-term grouping |

---

## Quick start

```bash
git clone https://github.com/iamsandeshdhital/qcpapers.git
cd quantum-computing-papers
python -m pip install -e ".[dev]"
pytest
```

Example, verified:

```python
from qcpapers.algorithms.qaoa import maxcut_qaoa, exact_maxcut

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
print(exact_maxcut(edges, 4))            # 4
print(maxcut_qaoa(edges, 4, layers=2))    # cost ~1e-7, cut 4, exact cost 0.0
```

---

## Testing philosophy

Every test uses an oracle that shares no code with the implementation: dense
matrix identities, closed forms from the source papers, or an independent
classical routine. Nothing asserts against a recorded output.

This matters more than it might seem. Every module written so far shipped at
least one bug that produced plausible-looking wrong numbers and converged
cleanly:

- the Grover diffuser used a single-qubit Z instead of a multi-controlled Z, so
  it was a no-op and the state stayed uniform;
- QPE assigned the wrong powers to the phase register, giving a bit-reversed
  estimate;
- `CSWAP` had a duplicated row, so it was not a permutation and not unitary;
- the classical-shadow estimator multiplied per-qubit Bloch components, which is
  correct for product states and silently wrong for entangled ones;
- the shadow Y-basis rotation collapsed onto the X branch, so the estimator
  converged to the wrong axis;
- QAOA started at zero angles, where the gradient vanishes identically, and
  reported convergence without moving.

Several of these are pinned by tests that fail loudly if reintroduced.

Two conventions worth knowing:

- **Gradients use tangent states, not the parameter-shift rule.** The rule is
  invalid when one parameter drives more than one gate, which happens in the
  number-preserving ansatz. It silently returns a wrong gradient that still
  descends, so the optimiser stalls on a flat spot.
- **Bit ordering is big-endian throughout.** Qubit 0 is the most significant bit,
  and `np.kron` is folded pairwise because it rejects three arguments.

---

## Known limitations

These are real gaps, not oversights to be discovered later:

- **QSVT is absent.** The angle construction was not derivable from the digit
  splitting alone; a least-squares search over the angle space failed to
  reproduce even a linear polynomial, which indicates the layer structure was
  wrong rather than the rounding.
- **HHL requires positive-definite matrices.** Indefinite systems need the
  Childs-Braun-MacCleay `|s><s|` shift, which is not implemented. `hhl_solve`
  raises rather than returning incorrect coefficients.
- **HHL is an exact-arithmetic reference.** It computes the spectrum classically
  and prepares the phase register exactly. It reproduces the algorithm's logic
  and conditioning behaviour, not its resource counts. `hhl_phase_estimates` does
  run real QPE so the clock-register scaling can be checked.
- **Classical shadows are slow.** `reconstruct` is O(`4^n` snapshots); fine for
  tests, not for use.
- **No stabiliser simulator,** so surface-code and Clifford-heavy work is not yet
  possible.

---

## Not yet built

Listed so the gap is explicit. None of this exists.

- **Algorithms:** QSVT, quantum walk search, adaptive QPE, portfolio optimisation
- **QEC:** surface code, Steane `[[7,1,3]]`, hypergraph product, holographic codes
- **Error mitigation:** ZNE, probabilistic error cancellation, Clifford data
  regression, M3
- **Benchmarks:** quantum volume, RB/DCRB/CLOPS, random circuit sampling,
  Gaussian boson sampling
- **Hardware models:** superconducting, trapped ion, neutral atom, topological
- **Compilers:** QIR, Quil, OpenQASM
- **Utilities:** metrology, optimal control, graphs, statistics
- **Tooling:** `cli.py`, `papers/registry.json`, runnable reproduction scripts

---

## License

Apache License 2.0. See `LICENSE`.
