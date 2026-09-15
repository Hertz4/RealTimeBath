"""Fast physical fitting in the rank-one exponential coordinates of Eq. (11)."""

from __future__ import annotations

from inspect import signature

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.optimize import least_squares

from .exceptions import RealizationError
from .realization import BackendResult
from .route_physical import _spectral_factor_residues, realize_positive_gram
from .types import ExponentialFit, LindbladModel


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_EXACT_TR_SOLVER_MAX_MODES = 4
_VALIDATION_CHECK_INTERVAL = 25
_VALIDATION_STALL_CHECKS = 8
_VALIDATION_MIN_RELATIVE_IMPROVEMENT = 1e-3
_MIN_OPTIMIZATION_NFEV = 200
_NFEV_PER_PARAMETER = 40
_LEAST_SQUARES_SUPPORTS_CALLBACK = "callback" in signature(
    least_squares
).parameters


def _effective_max_nfev(requested_max_nfev: int, n_parameters: int) -> int:
    """Return the parameter-scaled evaluation budget under the user cap."""

    scaled_budget = max(
        _MIN_OPTIMIZATION_NFEV, _NFEV_PER_PARAMETER * n_parameters
    )
    return min(requested_max_nfev, scaled_budget)


def _evaluate_rank_one(
    rates: ComplexArray,
    residues: ComplexArray,
    times: FloatArray,
) -> tuple[ComplexArray, ComplexArray]:
    denominators = rates[:, None] + rates[None, :].conj()
    covariance = np.outer(residues, residues.conj()) / denominators
    weights = np.sum(covariance, axis=1)
    values = np.exp(-np.multiply.outer(times, rates)) @ weights
    return np.asarray(values, dtype=np.complex128), weights


def _model_exponentials(model: LindbladModel) -> tuple[ComplexArray, ComplexArray]:
    drift = model.damping + 1j * model.hamiltonian
    rates, vectors = np.linalg.eig(drift)
    if np.any(np.real(rates) <= 0):
        raise RealizationError("the initial physical model is not strictly stable")
    inverse_coupling = np.linalg.solve(vectors, model.coupling)
    weights = (model.coupling.conj() @ vectors) * inverse_coupling
    order = np.lexsort((np.imag(rates), np.real(rates)))
    return rates[order], weights[order]


def _model_to_rank_one(model: LindbladModel) -> tuple[ComplexArray, ComplexArray]:
    """Convert a physical model to a well-scaled scalar spectral factor."""

    rates, weights = _model_exponentials(model)
    rate_scale = float(np.median(np.abs(rates)))
    weight_scale = float(np.max(np.abs(weights)))
    if rate_scale <= 0.0 or weight_scale <= 0.0:
        raise RealizationError("the initial model has zero rates or exponential weights")
    scaled_residues, _ = _spectral_factor_residues(
        rates / rate_scale,
        weights / weight_scale,
        positivity_tolerance=1e-10,
        root_tolerance=1e-5,
    )
    residues = scaled_residues * np.sqrt(weight_scale * rate_scale)
    reconstructed = np.sum(
        np.outer(residues, residues.conj())
        / (rates[:, None] + rates[None, :].conj()),
        axis=1,
    )
    residual = float(
        np.linalg.norm(reconstructed - weights)
        / max(np.linalg.norm(weights), np.finfo(float).tiny)
    )
    if residual > 2e-5:
        raise RealizationError(
            f"rank-one spectral coordinates are inaccurate (residual {residual:.3e})"
        )
    return rates, residues


def _pack_parameters(
    rates: ComplexArray, residues: ComplexArray
) -> tuple[FloatArray, int]:
    anchor = int(np.argmax(np.abs(residues)))
    phase = np.angle(residues[anchor])
    fixed_residues = residues * np.exp(-1j * phase)
    imaginary_indices = np.arange(residues.size) != anchor
    parameters = np.concatenate(
        [
            np.log(np.real(rates)),
            np.imag(rates),
            np.real(fixed_residues),
            np.imag(fixed_residues[imaginary_indices]),
        ]
    )
    return np.asarray(parameters, dtype=float), anchor


def _unpack_parameters(
    parameters: FloatArray, n_modes: int, anchor: int
) -> tuple[ComplexArray, ComplexArray]:
    rates = (
        np.exp(parameters[:n_modes])
        + 1j * parameters[n_modes : 2 * n_modes]
    )
    offset = 2 * n_modes
    real_residues = parameters[offset : offset + n_modes]
    offset += n_modes
    imaginary_residues = np.zeros(n_modes, dtype=float)
    imaginary_residues[np.arange(n_modes) != anchor] = parameters[offset:]
    residues = real_residues + 1j * imaginary_residues
    return rates, residues


def _prediction_jacobian(
    parameters: FloatArray,
    times: FloatArray,
    n_modes: int,
    anchor: int,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the prediction and its complex Jacobian over real parameters."""

    rates, residues = _unpack_parameters(parameters, n_modes, anchor)
    denominators = rates[:, None] + rates[None, :].conj()
    numerators = np.outer(residues, residues.conj())
    covariance = numerators / denominators
    weights = np.sum(covariance, axis=1)
    exponentials = np.exp(-np.multiply.outer(times, rates))
    prediction = exponentials @ weights
    inverse_denominators = 1.0 / denominators
    rate_kernel = numerators * inverse_denominators**2
    rate_row_sums = np.sum(rate_kernel, axis=1)

    def rate_jacobian(directions: ComplexArray) -> ComplexArray:
        weight_directions = -rate_kernel * directions.conj()[None, :]
        diagonal = np.diag_indices(n_modes)
        weight_directions[diagonal] -= directions * rate_row_sums
        exponential_directions = (
            -times[:, None]
            * exponentials
            * (directions * weights)[None, :]
        )
        return exponential_directions + exponentials @ weight_directions

    residue_kernel = residues[:, None] * inverse_denominators
    residue_row_sums = inverse_denominators @ residues.conj()

    def residue_jacobian(directions: ComplexArray) -> ComplexArray:
        weight_directions = residue_kernel * directions.conj()[None, :]
        diagonal = np.diag_indices(n_modes)
        weight_directions[diagonal] += directions * residue_row_sums
        return exponentials @ weight_directions

    imaginary_indices = np.flatnonzero(np.arange(n_modes) != anchor)
    jacobian = np.hstack(
        [
            rate_jacobian(np.real(rates).astype(np.complex128)),
            rate_jacobian(np.full(n_modes, 1j, dtype=np.complex128)),
            residue_jacobian(np.ones(n_modes, dtype=np.complex128)),
            residue_jacobian(np.full(n_modes, 1j, dtype=np.complex128))[
                :, imaginary_indices
            ],
        ]
    )
    return prediction, jacobian


def optimize_rank_one_exponentials(
    t: ArrayLike,
    values: ArrayLike,
    initial: BackendResult,
    *,
    max_nfev: int = 1_000,
    max_points: int = 251,
    regularization: float = 1e-10,
) -> tuple[BackendResult, ExponentialFit]:
    """Refine a model using the physical rank-one exponential parametrization."""

    times = np.asarray(t, dtype=float)
    target = np.asarray(values, dtype=np.complex128)
    if times.ndim != 1 or target.shape != times.shape:
        raise ValueError("t and values must be one-dimensional arrays of equal length")
    if max_nfev < 1 or max_points < 6:
        raise ValueError("max_nfev must be positive and max_points must be at least six")

    time_scale = float(times[-1])
    amplitude_scale = float(np.max(np.abs(target)))
    scaled_times = times / time_scale
    scaled_target = target / amplitude_scale
    scaled_initial_model = LindbladModel(
        hamiltonian=initial.model.hamiltonian * time_scale,
        damping=initial.model.damping * time_scale,
        coupling=initial.model.coupling / np.sqrt(amplitude_scale),
    )
    initial_rates, initial_residues = _model_to_rank_one(
        scaled_initial_model
    )
    initial_parameters, anchor = _pack_parameters(initial_rates, initial_residues)
    n_modes = initial_rates.size
    effective_max_nfev = _effective_max_nfev(
        max_nfev, initial_parameters.size
    )

    if times.size > max_points:
        indices = np.unique(
            np.linspace(0, times.size - 1, max_points).round().astype(int)
        )
    else:
        indices = np.arange(times.size)
    objective_times = scaled_times[indices]
    objective_target = scaled_target[indices]
    target_norm = max(float(np.linalg.norm(objective_target)), np.finfo(float).tiny)
    regularization_scale = np.sqrt(
        max(0.0, regularization) / initial_parameters.size
    )
    nyquist = np.pi / float(scaled_times[1] - scaled_times[0])
    lower = np.full(initial_parameters.size, -np.inf)
    upper = np.full(initial_parameters.size, np.inf)
    lower[:n_modes] = np.log(1e-10)
    upper[:n_modes] = np.log(max(1e4, 100.0 * nyquist))
    lower[n_modes : 2 * n_modes] = -nyquist
    upper[n_modes : 2 * n_modes] = nyquist

    cached_parameters: FloatArray | None = None
    cached_residual: FloatArray | None = None
    cached_jacobian: FloatArray | None = None
    progress: list[tuple[int, float, float]] = []
    iteration = 0
    stopped_for_validation_stall = False
    validation_l1_norm = max(
        float(trapezoid(np.abs(scaled_target), scaled_times)),
        np.finfo(float).tiny,
    )
    validation_l2_norm = max(
        float(np.linalg.norm(scaled_target)), np.finfo(float).tiny
    )

    def evaluate(parameters: FloatArray) -> tuple[FloatArray, FloatArray]:
        nonlocal cached_parameters, cached_residual, cached_jacobian
        if cached_parameters is not None and np.array_equal(parameters, cached_parameters):
            assert cached_residual is not None and cached_jacobian is not None
            return cached_residual, cached_jacobian
        prediction, complex_jacobian = _prediction_jacobian(
            parameters, objective_times, n_modes, anchor
        )
        difference = (prediction - objective_target) / target_norm
        residual_pieces = [np.real(difference), np.imag(difference)]
        jacobian_pieces = [
            np.real(complex_jacobian) / target_norm,
            np.imag(complex_jacobian) / target_norm,
        ]
        if regularization_scale:
            residual_pieces.append(
                regularization_scale * (parameters - initial_parameters)
            )
            jacobian_pieces.append(
                regularization_scale * np.eye(initial_parameters.size)
            )
        cached_parameters = parameters.copy()
        cached_residual = np.concatenate(residual_pieces)
        cached_jacobian = np.vstack(jacobian_pieces)
        return cached_residual, cached_jacobian

    def record_progress(intermediate_result) -> None:
        nonlocal iteration, stopped_for_validation_stall
        iteration += 1
        if iteration % _VALIDATION_CHECK_INTERVAL:
            return
        current_rates, current_residues = _unpack_parameters(
            intermediate_result.x, n_modes, anchor
        )
        current_prediction, _ = _evaluate_rank_one(
            current_rates, current_residues, scaled_times
        )
        current_l1 = float(
            trapezoid(
                np.abs(current_prediction - scaled_target), scaled_times
            )
            / validation_l1_norm
        )
        current_l2 = float(
            np.linalg.norm(current_prediction - scaled_target)
            / validation_l2_norm
        )
        progress.append((iteration, current_l1, current_l2))
        if len(progress) > _VALIDATION_STALL_CHECKS:
            previous_l1 = progress[-(_VALIDATION_STALL_CHECKS + 1)][1]
            relative_improvement = (previous_l1 - current_l1) / max(
                previous_l1, np.finfo(float).tiny
            )
            if relative_improvement < _VALIDATION_MIN_RELATIVE_IMPROVEMENT:
                stopped_for_validation_stall = True
                raise StopIteration

    tr_solver = (
        "exact" if n_modes <= _EXACT_TR_SOLVER_MAX_MODES else "lsmr"
    )
    callback_options = (
        {"callback": record_progress}
        if _LEAST_SQUARES_SUPPORTS_CALLBACK
        else {}
    )
    solution = least_squares(
        lambda parameters: evaluate(parameters)[0],
        initial_parameters,
        jac=lambda parameters: evaluate(parameters)[1],
        bounds=(lower, upper),
        method="trf",
        tr_solver=tr_solver,
        x_scale="jac",
        max_nfev=effective_max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        **callback_options,
    )
    fitted_rates, fitted_residues = _unpack_parameters(
        solution.x, n_modes, anchor
    )
    physical_rates = fitted_rates / time_scale
    physical_residues = fitted_residues * np.sqrt(amplitude_scale / time_scale)
    backend = realize_positive_gram(physical_rates, physical_residues[:, None])
    prediction = backend.model.evaluate(times)
    initial_prediction = initial.model.evaluate(times)
    l2_normalization = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    l1_normalization = max(
        float(trapezoid(np.abs(target), times)), np.finfo(float).tiny
    )
    initial_l2 = float(np.linalg.norm(initial_prediction - target) / l2_normalization)
    fitted_l2 = float(np.linalg.norm(prediction - target) / l2_normalization)
    initial_l1 = float(
        trapezoid(np.abs(initial_prediction - target), times) / l1_normalization
    )
    fitted_l1 = float(
        trapezoid(np.abs(prediction - target), times) / l1_normalization
    )
    warnings = list(initial.warnings)
    use_initial = (
        fitted_l2 > initial_l2 * (1.0 + 1e-8)
        or fitted_l1 > initial_l1 * (1.0 + 1e-8)
    )
    if use_initial:
        warnings.append(
            "rank-one optimization was discarded because it did not improve "
            "both full-grid L1 and L2 errors"
        )
        backend = initial
        prediction = initial_prediction
        fitted_l2 = initial_l2
        fitted_l1 = initial_l1
        fitted_rates = initial_rates
        fitted_residues = initial_residues
        physical_rates = fitted_rates / time_scale
        physical_residues = fitted_residues * np.sqrt(
            amplitude_scale / time_scale
        )
    elif not solution.success and not stopped_for_validation_stall:
        warnings.append(f"rank-one optimization stopped early: {solution.message}")

    _, physical_weights = _evaluate_rank_one(
        physical_rates, physical_residues, times[:1]
    )
    exponential = ExponentialFit(
        physical_rates,
        physical_weights,
        fitted_l2,
        float(np.max(np.abs(prediction - target))),
    )
    optimization_message = (
        "full-grid L1 validation improvement stalled"
        if stopped_for_validation_stall
        else str(solution.message)
    )
    result = BackendResult(
        model=backend.model,
        condition_number=backend.condition_number,
        conversion_residual=backend.conversion_residual,
        warnings=tuple(dict.fromkeys(warnings)),
        details={
            **initial.details,
            **backend.details,
            "rank_one_optimization": True,
            "rank_one_optimization_success": bool(
                solution.success or stopped_for_validation_stall
            ),
            "rank_one_optimization_status": int(solution.status),
            "rank_one_optimization_message": optimization_message,
            "rank_one_optimization_nfev": int(solution.nfev),
            "rank_one_optimization_njev": int(solution.njev or 0),
            "rank_one_requested_max_nfev": int(max_nfev),
            "rank_one_effective_max_nfev": int(effective_max_nfev),
            "rank_one_parameter_count": int(initial_parameters.size),
            "rank_one_anchor": anchor,
            "rank_one_tr_solver": tr_solver,
            "initial_relative_l2_error": initial_l2,
            "optimized_relative_l2_error": fitted_l2,
            "initial_relative_l1_error": initial_l1,
            "optimized_relative_l1_error": fitted_l1,
            "rank_one_optimization_used_initial_model": use_initial,
            "rank_one_optimization_progress": progress,
            "rank_one_optimization_validation_stall": stopped_for_validation_stall,
        },
    )
    return result, exponential
