"""Gauge/semidefinite realization from arXiv:2506.10308."""

from __future__ import annotations

import warnings as py_warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import OptionalDependencyError, RealizationError
from .realization import BackendResult
from .types import ExponentialFit, LindbladModel


_DEFAULT_SOLVER_ORDER = ("MOSEK", "CLARABEL", "SCS")


@dataclass(frozen=True)
class _SolverResult:
    solver: str
    status: str
    objective: float
    gauge: np.ndarray
    failed_attempts: tuple[str, ...]


def cvxpy_available() -> bool:
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return False
    return True


def _balanced_quasi_couplings(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitudes = np.abs(weights)
    phases = np.angle(weights)
    roots = np.sqrt(magnitudes)
    right = roots * np.exp(0.5j * phases)
    left = roots * np.exp(-0.5j * phases)
    return left, right


def _solver_options(solver: str) -> dict[str, Any]:
    options: dict[str, Any] = {"verbose": False}
    if solver == "MOSEK":
        options.update(
            mosek_params={
                "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
                "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10,
                "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
            }
        )
    elif solver == "CLARABEL":
        options.update(
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            max_iter=1_000,
        )
    elif solver == "SCS":
        options.update(eps=1e-8, max_iters=100_000)
    return options


def _solve_with_fallback(
    problem: Any,
    gauge_variable: Any,
    cp: Any,
    solver: str | None,
) -> _SolverResult:
    automatic = solver is None
    if automatic:
        installed = set(cp.installed_solvers())
        candidates = [name for name in _DEFAULT_SOLVER_ORDER if name in installed]
        if not candidates:
            raise RealizationError(
                "none of the supported SDP solvers is installed "
                f"({_DEFAULT_SOLVER_ORDER})"
            )
    else:
        candidates = [solver]

    failures: list[str] = []
    inaccurate: _SolverResult | None = None
    for candidate in candidates:
        try:
            with py_warnings.catch_warnings():
                py_warnings.filterwarnings(
                    "ignore",
                    message="Initializing a Constant with a nested list.*",
                    category=UserWarning,
                )
                py_warnings.filterwarnings(
                    "ignore",
                    message="Solution may be inaccurate.*",
                    category=UserWarning,
                )
                problem.solve(solver=candidate, **_solver_options(candidate))
        except Exception as error:  # CVXPY normalizes backend exceptions poorly.
            failure = f"{candidate}: {type(error).__name__}: {error}"
            if not automatic:
                raise RealizationError(f"the SDP solver failed ({failure})") from error
            failures.append(failure)
            continue

        if gauge_variable.value is None or problem.status not in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            failure = f"{candidate}: status {problem.status}"
            if not automatic:
                raise RealizationError(
                    f"the SDP problem did not converge ({failure})"
                )
            failures.append(failure)
            continue

        result = _SolverResult(
            solver=candidate,
            status=problem.status,
            objective=float(problem.value),
            gauge=np.asarray(gauge_variable.value, dtype=np.complex128).copy(),
            failed_attempts=tuple(failures),
        )
        if problem.status == cp.OPTIMAL:
            return result
        if inaccurate is None:
            inaccurate = result
        failures.append(f"{candidate}: status {problem.status}")

    if inaccurate is not None:
        return _SolverResult(
            solver=inaccurate.solver,
            status=inaccurate.status,
            objective=inaccurate.objective,
            gauge=inaccurate.gauge,
            failed_attempts=tuple(failures),
        )
    joined = "; ".join(failures)
    raise RealizationError(f"all automatic SDP solvers failed ({joined})")


def realize_sdp(
    exponential: ExponentialFit,
    *,
    solver: str | None = None,
    psd_margin: float = 1e-9,
    regularization: float = 1e-10,
) -> BackendResult:
    """Project an exponential fit onto a physical coupled Lindblad realization.

    The convex problem minimizes ``||l - Y r||`` subject to the dissipativity
    constraint from Huang et al. Exact physical inputs have zero residual; noisy
    or unphysical exponential fits receive a closest feasible realization.
    """

    try:
        import cvxpy as cp
    except ImportError as error:
        raise OptionalDependencyError(
            "the SDP backend requires CVXPY, which is a core RealTimeBath "
            "dependency; reinstall the package with dependencies enabled"
        ) from error

    rates = np.asarray(exponential.rates, dtype=np.complex128)
    weights = np.asarray(exponential.weights, dtype=np.complex128)
    if np.any(np.real(rates) <= 0):
        raise RealizationError("all rates must have positive real parts")
    if np.any(np.abs(weights) <= np.finfo(float).tiny):
        raise RealizationError("zero-weight exponential terms must be pruned before realization")
    amplitude_scale = float(np.max(np.abs(weights)))
    rate_scale = float(np.median(np.abs(rates)))
    scaled_rates = rates / rate_scale
    scaled_weights = weights / amplitude_scale
    left, right = _balanced_quasi_couplings(scaled_weights)
    quasi_generator = np.diag(-1j * scaled_rates)
    n_modes = rates.size

    y = cp.Variable((n_modes, n_modes), hermitian=True)
    dissipation_lmi = cp.hermitian_wrap(
        1j * (y @ quasi_generator - quasi_generator.conj().T @ y)
    )
    mismatch = left - y @ right
    objective = cp.sum_squares(cp.abs(mismatch))
    if regularization:
        objective += float(regularization) * cp.sum_squares(cp.abs(y - np.eye(n_modes)))
    constraints = [
        y >> float(psd_margin) * np.eye(n_modes),
        dissipation_lmi >> 0,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    solution = _solve_with_fallback(problem, y, cp, solver)

    y_value = solution.gauge
    y_value = (y_value + y_value.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(y_value)
    y_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-7 * y_scale:
        raise RealizationError("the SDP solver returned a non-positive gauge matrix")
    floor = max(float(psd_margin), np.finfo(float).eps * y_scale)
    eigenvalues = np.maximum(eigenvalues, floor)
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    transform = (
        eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    )
    inverse_transform = np.linalg.inv(transform)
    coupled_generator = transform @ quasi_generator @ inverse_transform
    hamiltonian = (coupled_generator + coupled_generator.conj().T) / 2.0
    damping = (coupled_generator.conj().T - coupled_generator) / (2.0j)
    damping = (damping + damping.conj().T) / 2.0
    damping_values, damping_vectors = np.linalg.eigh(damping)
    damping_scale = max(1.0, float(np.max(np.abs(damping_values))))
    if float(np.min(damping_values)) < -2e-6 * damping_scale:
        raise RealizationError(
            "the SDP solution violates the damping positivity constraint beyond solver tolerance"
        )
    damping = (
        damping_vectors
        @ np.diag(np.maximum(damping_values, 0.0))
        @ damping_vectors.conj().T
    )
    coupling = transform @ right

    model = LindbladModel(
        hamiltonian=hamiltonian * rate_scale,
        damping=damping * rate_scale,
        coupling=coupling * np.sqrt(amplitude_scale),
    )
    equality_residual = float(
        np.linalg.norm(left - y_value @ right)
        / max(np.linalg.norm(left), np.finfo(float).tiny)
    )
    check_times = np.linspace(0.0, 5.0 / float(np.min(np.real(rates))), 101)
    target = exponential.evaluate(check_times)
    model_values = model.evaluate(check_times)
    model_residual = float(
        np.linalg.norm(model_values - target)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
    )
    warnings: list[str] = []
    if solution.status == cp.OPTIMAL_INACCURATE:
        warnings.append("CVXPY reported an inaccurate optimum")
    if solution.failed_attempts:
        warnings.append(
            "automatic SDP solver fallback was used after: "
            + "; ".join(solution.failed_attempts)
        )
    if equality_residual > 1e-6:
        warnings.append(
            "the exponential fit was projected to a nearby physical correlation"
        )
    if condition_number > 1e10:
        warnings.append("the SDP gauge matrix is ill-conditioned")
    return BackendResult(
        model=model,
        condition_number=condition_number,
        conversion_residual=max(equality_residual, model_residual),
        warnings=tuple(warnings),
        details={
            "solver": solution.solver,
            "solver_status": solution.status,
            "solver_objective": solution.objective,
            "solver_failed_attempts": solution.failed_attempts,
            "gauge_eigenvalues": eigenvalues,
            "equality_residual": equality_residual,
            "model_residual": model_residual,
        },
    )
