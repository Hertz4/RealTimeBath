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

The package implements two general-purpose constructions and one specialized,
experimental construction:

- `method="sdp"`: the gauge/semidefinite physical projection of Huang, Park,
  Chan, and Lin, [arXiv:2506.10308](https://arxiv.org/abs/2506.10308).
- `method="physical"`: the exact spectral-factor and Ornstein--Uhlenbeck
  construction of Müller and Strunz,
  [arXiv:2604.06466](https://arxiv.org/abs/2604.06466).
- `method="positive"`: an experimental optimizer for the same paper's rank-one
  positive-exponential ansatz (Eq. 11). Do not use this as a general-purpose
  route from sampled real-time data. A positive-exponential representation is
  appropriate only when such a representation is already known; obtaining it
  a priori is itself nontrivial. When its rates and positive Gram factor are
  known, prefer the lower-level `realize_positive_gram` interface.

> **Warning:** Do not select `method="positive"` merely from sampled
> `Delta(t)`. Use it only when a valid positive-exponential representation is
> already available; constructing that representation a priori is nontrivial.

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
error for sustained stalls. `optimization_max_nfev` is passed directly to
SciPy as the maximum number of nonlinear function evaluations; it is not
silently reduced according to the number of fitted parameters:

```python
effective_max_nfev = optimization_max_nfev
```

The validation-stall callback may still stop the optimization before this
limit when the full-grid error no longer improves meaningfully.

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

## Benchmark notebook

[`notebooks/bath_benchmarks.ipynb`](notebooks/bath_benchmarks.ipynb) is the
reproducible benchmark of the refined SDP route. It fits the full semicircle
transform

\[
J(\omega)=\frac{\Gamma}{\pi}\sqrt{1-(\omega/W)^2},\qquad
\Delta(t)=\Gamma\frac{J_1(Wt)}{t},
\]

with the continuous value `Delta(0) = Gamma*W/2`, using `W=10`, `Gamma=1`,
and also fits the unit-normalized half-semicircle and box densities supported
on `0 <= omega <= 1`. All fits use `t` in `[0, 10]` and scan requested mode
budgets `N=1,...,15`. The positive-exponential route is deliberately excluded;
the notebook reports

\[
\epsilon_1=\frac{\int_0^{10}|\Delta_{\rm fit}(t)-\Delta(t)|dt}
{\int_0^{10}|\Delta(t)|dt}.
\]

It contains the fitted real and imaginary parts at `N=6`, pointwise absolute
errors at `N=6` and `N=12`, normalized L1 error versus mode budget, and timing
plots. The executed notebook includes all six figures and numerical output;
it does not create a separate directory of generated artifacts. On the default
grid, the one-sided targets reach the estimated data rank at seven modes, and
the plots identify requested budgets above that as band-limited plateaus.

To rerun the notebook from a source checkout:

```bash
python -m pip install -e '.[notebook]'
jupyter lab notebooks/bath_benchmarks.ipynb
```

Run the unit tests and small deterministic realization comparison with:

```bash
pytest
python benchmarks/compare_routes.py
```

## License

RealTimeBath is released under the GNU General Public License, version 3 only
(`GPL-3.0-only`). See [`LICENSE`](LICENSE).
