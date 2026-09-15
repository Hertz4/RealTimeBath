"""Positive-exponential fitting suggested by arXiv:2604.06466 Eq. (11/S44)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .exceptions import RealizationError
from .rank_one_fit import optimize_rank_one_exponentials
from .realization import BackendResult
from .route_physical import realize_positive_gram
from .types import ExponentialFit, LindbladModel


def _model_to_gram(
    model: LindbladModel,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    drift = model.damping + 1j * model.hamiltonian
    rates, eigenvectors = np.linalg.eig(drift)
    if np.any(np.real(rates) <= 0):
        raise RealizationError("the initial physical model is not strictly stable")
    inverse = np.linalg.inv(eigenvectors)
    covariance = inverse @ inverse.conj().T
    transformed_coupling = eigenvectors.conj().T @ model.coupling
    left_diagonal = np.diag(transformed_coupling.conj())
    right_diagonal = np.diag(transformed_coupling)
    unit_coupling_covariance = left_diagonal @ covariance @ right_diagonal
    unit_coupling_covariance = (
        unit_coupling_covariance + unit_coupling_covariance.conj().T
    ) / 2.0
    rate_matrix = np.diag(rates)
    gram = (
        rate_matrix @ unit_coupling_covariance
        + unit_coupling_covariance @ rate_matrix.conj().T
    )
    gram = (gram + gram.conj().T) / 2.0
    eigenvalues, gram_vectors = np.linalg.eigh(gram)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-7 * scale:
        raise RealizationError("could not convert the initial model to positive Gram form")
    factor = gram_vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    return rates, factor


def _evaluate_positive(
    rates: NDArray[np.complex128],
    factor: NDArray[np.complex128],
    times: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    gram = factor @ factor.conj().T
    covariance = gram / (rates[:, None] + rates[None, :].conj())
    weights = np.sum(covariance, axis=1)
    values = np.exp(-np.multiply.outer(times, rates)) @ weights
    return values, weights


def optimize_full_positive_exponentials(
    t: ArrayLike,
    values: ArrayLike,
    initial: BackendResult,
    *,
    max_nfev: int = 300,
    max_points: int = 251,
    regularization: float = 1e-10,
) -> tuple[BackendResult, ExponentialFit]:
    """Fit the physical positive-Gram exponential ansatz in the time domain."""

    times = np.asarray(t, dtype=float)
    target = np.asarray(values, dtype=np.complex128)
    time_scale = float(times[-1])
    amplitude_scale = float(np.max(np.abs(target)))
    scaled_times = times / time_scale
    scaled_target = target / amplitude_scale
    scaled_initial_model = LindbladModel(
        initial.model.hamiltonian * time_scale,
        initial.model.damping * time_scale,
        initial.model.coupling / np.sqrt(amplitude_scale),
    )
    rates, factor = _model_to_gram(scaled_initial_model)
    initial_positive_values, initial_positive_weights = _evaluate_positive(
        rates, factor, scaled_times
    )
    n_modes = rates.size
    initial_parameters = np.concatenate(
        [
            np.log(np.real(rates)),
            np.imag(rates),
            np.real(factor).ravel(),
            np.imag(factor).ravel(),
        ]
    )
    factor_size = factor.size
    if times.size > max_points:
        indices = np.unique(np.linspace(0, times.size - 1, max_points).round().astype(int))
    else:
        indices = np.arange(times.size)
    objective_times = scaled_times[indices]
    objective_target = scaled_target[indices]
    target_norm = max(float(np.linalg.norm(objective_target)), np.finfo(float).tiny)
    nyquist = np.pi / float(scaled_times[1] - scaled_times[0])
    lower = np.concatenate(
        [
            np.full(n_modes, np.log(1e-10)),
            np.full(n_modes, -nyquist),
            np.full(2 * factor_size, -np.inf),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_modes, np.log(max(1e4, 100.0 * nyquist))),
            np.full(n_modes, nyquist),
            np.full(2 * factor_size, np.inf),
        ]
    )
    regularization_scale = np.sqrt(max(0.0, regularization) / initial_parameters.size)

    def unpack(parameters: NDArray[np.float64]):
        current_rates = (
            np.exp(parameters[:n_modes])
            + 1j * parameters[n_modes : 2 * n_modes]
        )
        offset = 2 * n_modes
        current_factor = (
            parameters[offset : offset + factor_size].reshape(factor.shape)
            + 1j
            * parameters[offset + factor_size : offset + 2 * factor_size].reshape(
                factor.shape
            )
        )
        return current_rates, current_factor

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        current_rates, current_factor = unpack(parameters)
        prediction, _ = _evaluate_positive(
            current_rates, current_factor, objective_times
        )
        difference = (prediction - objective_target) / target_norm
        pieces = [np.real(difference), np.imag(difference)]
        if regularization_scale:
            pieces.append(regularization_scale * (parameters - initial_parameters))
        return np.concatenate(pieces)

    solution = least_squares(
        residual,
        initial_parameters,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    fitted_rates, fitted_factor = unpack(solution.x)
    scaled_prediction, fitted_weights = _evaluate_positive(
        fitted_rates, fitted_factor, scaled_times
    )
    initial_error = float(
        np.linalg.norm(initial.model.evaluate(times) - target)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
    )
    fitted_error = float(
        np.linalg.norm(scaled_prediction - scaled_target)
        / max(np.linalg.norm(scaled_target), np.finfo(float).tiny)
    )
    initial_prediction = initial.model.evaluate(times)
    l1_normalization = max(
        float(np.trapezoid(np.abs(target), times)), np.finfo(float).tiny
    )
    initial_l1 = float(
        np.trapezoid(np.abs(initial_prediction - target), times) / l1_normalization
    )
    fitted_l1 = float(
        np.trapezoid(
            np.abs(scaled_prediction * amplitude_scale - target), times
        )
        / l1_normalization
    )
    if (
        fitted_error > initial_error * (1.0 + 1e-8)
        or fitted_l1 > initial_l1 * (1.0 + 1e-8)
    ):
        warnings = list(initial.warnings)
        warnings.append(
            "positive-exponential optimization was discarded because it did not "
            "improve both full-grid L1 and L2 errors"
        )
        exponential = ExponentialFit(
            rates / time_scale,
            initial_positive_weights * amplitude_scale,
            initial_error,
            float(
                np.max(np.abs(initial_positive_values * amplitude_scale - target))
            ),
        )
        return BackendResult(
            initial.model,
            initial.condition_number,
            initial.conversion_residual,
            tuple(warnings),
            {**initial.details, "positive_optimization_used_initial_model": True},
        ), exponential

    physical_rates = fitted_rates / time_scale
    physical_factor = fitted_factor * np.sqrt(amplitude_scale / time_scale)
    backend = realize_positive_gram(physical_rates, physical_factor)
    full_prediction = backend.model.evaluate(times)
    relative_error = float(
        np.linalg.norm(full_prediction - target)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
    )
    max_error = float(np.max(np.abs(full_prediction - target)))
    warnings = list(initial.warnings)
    if not solution.success:
        warnings.append(f"positive-exponential optimization stopped early: {solution.message}")
    backend = BackendResult(
        model=backend.model,
        condition_number=backend.condition_number,
        conversion_residual=backend.conversion_residual,
        warnings=tuple(warnings),
        details={
            **initial.details,
            **backend.details,
            "positive_optimization_success": bool(solution.success),
            "positive_optimization_nfev": int(solution.nfev),
            "positive_initial_relative_l2_error": initial_error,
            "positive_final_relative_l2_error": relative_error,
            "positive_initial_relative_l1_error": initial_l1,
            "positive_final_relative_l1_error": fitted_l1,
        },
    )
    exponential = ExponentialFit(
        physical_rates,
        fitted_weights * amplitude_scale,
        relative_error,
        max_error,
    )
    return backend, exponential


def optimize_positive_exponentials(
    t: ArrayLike,
    values: ArrayLike,
    initial: BackendResult,
    *,
    max_nfev: int = 1_000,
    max_points: int = 251,
    regularization: float = 1e-10,
) -> tuple[BackendResult, ExponentialFit]:
    """Fit Eq. (11) in rank-one coordinates, with the old full-Gram fallback."""

    try:
        return optimize_rank_one_exponentials(
            t,
            values,
            initial,
            max_nfev=max_nfev,
            max_points=max_points,
            regularization=regularization,
        )
    except (RealizationError, np.linalg.LinAlgError) as error:
        warnings = list(initial.warnings)
        warnings.append(
            "rank-one initialization failed; using the slower full-Gram "
            f"optimization ({error})"
        )
        fallback_initial = BackendResult(
            model=initial.model,
            condition_number=initial.condition_number,
            conversion_residual=initial.conversion_residual,
            warnings=tuple(warnings),
            details=initial.details,
        )
        return optimize_full_positive_exponentials(
            t,
            values,
            fallback_initial,
            max_nfev=min(max_nfev, 300),
            max_points=max_points,
            regularization=regularization,
        )
