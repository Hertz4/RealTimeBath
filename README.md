# RealTimeBath

`realtimebath` fits a sampled scalar real-time bath correlation with a
finite, physical set of coupled Lindblad pseudomodes:

\[
C(t) \approx g^\dagger \exp[(-iH-D)t]g,
\qquad H=H^\dagger,\quad D\succeq0.
\]

The jump-coefficient matrix returned by the package satisfies
`jumps.conj().T @ jumps == 2*damping`. This convention removes a factor-of-two
ambiguity between common forms of the Lindblad dissipator.

The package implements two complementary constructions:

- `method="sdp"`: the gauge/semidefinite physical projection of Huang, Park,
  Chan, and Lin, [arXiv:2506.10308](https://arxiv.org/abs/2506.10308).
- `method="physical"`: the exact spectral-factor and Ornstein--Uhlenbeck
  construction of Müller and Strunz,
  [arXiv:2604.06466](https://arxiv.org/abs/2604.06466).
- `method="positive"`: the same paper's rank-one positive-exponential ansatz
  (Eq. 11), optimized directly in the time domain with an analytic Jacobian. It
  uses the SDP model as a robust seed when an exact spectral factor of the
  unconstrained fit is unavailable.

`method="auto"` first uses the exact construction. If the fitted exponential
correlation is not exactly physical or is ill-conditioned, it uses the SDP
projection when CVXPY is installed. Passing `optimize=True` forces that SDP path
and applies the optional physical time-domain refinement.

## Installation

```bash
python -m pip install realtimebath
python -m pip install 'realtimebath[sdp]'
python -m pip install 'realtimebath[mosek]'  # optional; requires a MOSEK license
```

For an editable source checkout, use `python -m pip install -e '.[sdp]'`.

The base package requires NumPy and SciPy. CVXPY is only required by the SDP
backend and by the automatic fallback for noisy or slightly unphysical fits.
When available, the SDP backend prefers MOSEK, then CLARABEL, then SCS. A
specific installed solver can be selected with the `solver` argument.

## Scalar correlation

```python
import numpy as np
from realtimebath import fit_correlation

t = np.linspace(0.0, 10.0, 401)
correlation = 2.3 * np.exp(-(0.4 + 1.2j) * t)

# Choose exactly one mode count...
fit = fit_correlation(t, correlation, n_modes=1)

# ...or let the fitter choose the smallest N reaching a target error.
fit_by_accuracy = fit_correlation(t, correlation, eps=1e-8, max_modes=12)

H = fit.model.hamiltonian
D = fit.model.damping
g = fit.model.coupling
L = fit.model.jumps
reconstructed = fit.evaluate(t)
print(fit.diagnostics)
```

The SDP construction can optionally be refined directly against the input
samples in rank-one physical exponential coordinates. Physicality is retained
at every optimizer iteration. The refinement uses a vectorized analytic
Jacobian, retains the dense trust-region solver for small models, switches to
iterative trust-region solves when they become faster, and checks the full-grid
error for sustained stalls. The nonlinear evaluation budget scales with the
`4*n_modes - 1` fitted parameters while respecting
`optimization_max_nfev` as a hard user cap:

```python
effective_max_nfev = min(
    optimization_max_nfev,
    max(200, 40 * (4 * n_modes - 1)),
)
```

```python
fit = fit_correlation(
    t,
    correlation,
    n_modes=4,
    method="sdp",
    optimize=True,
    optimization_max_nfev=1_000,
)
```

The current fitter expects a uniform grid beginning at `t=0`. Users can choose
either `n_modes=N` for a fixed-order fit or `eps=...` for automatic order
selection; `eps` is the relative RMS error on the supplied time samples. The
older `tolerance` keyword remains a compatibility alias. The poles are
constrained to the stable half-plane and optionally polished by nonlinear least
squares.

The Hankel SVD also estimates how many numerically independent modes the input
grid can support. If a fixed `N` exceeds that ceiling, the package raises
`BandLimitError`. If an `eps` target cannot be reached before the ceiling, it
returns the best available fit, emits `BandLimitWarning`, and records
`band_limit_reached`, `max_supported_modes`, and the requested accuracy in
`fit.diagnostics.details`. Both messages recommend supplying finer real-time
data (a smaller time step and, when needed, a longer time window). If only
`max_modes` was exhausted while the data support more modes, the message asks
the user to increase `max_modes` instead.

## Fermionic lesser/greater hybridizations

A finite-temperature fermionic bath normally needs two positive kernels: a
filled lesser sector and an empty greater sector.

```python
from realtimebath import fit_fermionic

fit = fit_fermionic(t, delta_lesser, delta_greater)
lesser_model = fit.lesser.model       # modes begin filled
greater_model = fit.greater.model     # modes begin empty
assert fit.lesser_jump_kind == "creation"
assert fit.greater_jump_kind == "annihilation"
```

By default, inputs use the positive-correlation convention of Eq. S6 in
arXiv:2506.10308. For conventional nonequilibrium Green functions obeying
`Delta_less = +1j*C_less` and `Delta_greater = -1j*C_greater`, pass
`convention="negf"`.

## Diagnostics and limitations

The result reports time-domain error, smallest damping eigenvalue, stability
abscissa, realization condition number, conversion residual, and warnings.
The decomposition is not unique, so physically equivalent fits need not return
the same matrices.

Version 0.1 is intentionally scalar-first. Matrix-valued/multi-orbital
hybridizations and irregular time grids are rejected rather than being handled
with implicit conventions or interpolation.

## Semicircle benchmark

`benchmarks/semicircle.py` fits the transform

\[
J(\omega)=\frac{\Gamma}{\pi}\sqrt{1-(\omega/W)^2},\qquad
\Delta(t)=\Gamma\frac{J_1(Wt)}{t},
\]

with the continuous value `Delta(0) = Gamma*W/2`, using `W=10`, `Gamma=1`,
and `t` in `[0, 10]`. It compares the refined SDP
route with the positive-exponential route and reports

\[
\epsilon_1=\frac{\int_0^{10}|\Delta_{\rm fit}(t)-\Delta(t)|dt}
{\int_0^{10}|\Delta(t)|dt}.
\]

The script writes PNG/PDF fit and timing figures plus the fit, error, and timing
values as CSV files under `artifacts/`.

Run the tests and deterministic route comparison with:

```bash
pytest
python benchmarks/compare_routes.py
python benchmarks/semicircle.py
```

## One-sided finite-support benchmarks

`benchmarks/finite_support.py` applies the same route, pointwise-error, L1,
and timing comparison to two unit-normalized densities supported on
`0 <= omega <= 1`:

\[
J_{\mathrm{half}}(\omega)=\frac{4}{\pi}\sqrt{1-\omega^2},
\qquad
J_{\mathrm{box}}(\omega)=1.
\]

Their analytic transforms are used so the reference data contain no numerical
quadrature error. On the default `t in [0, 10]` grid, both Hankel matrices have
numerical rank seven in double precision. The generated mode-budget plots show
that rank-limited plateau explicitly and record both the requested budget and
active mode count in their CSV files.

```bash
python benchmarks/finite_support.py
```
