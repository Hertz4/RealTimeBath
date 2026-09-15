"""Stable sum-of-exponentials fitting for uniformly sampled complex data."""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import hankel
from scipy.optimize import least_squares

from .exceptions import (
    BandLimitError,
    BandLimitWarning,
    InvalidSamplesError,
    RealizationError,
)
from .types import ExponentialFit


_DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ValidatedSamples:
    times: NDArray[np.float64]
    values: NDArray[np.complex128]
    time_scale: float
    amplitude_scale: float


@dataclass(frozen=True)
class _MatrixPencilData:
    left_vectors: NDArray[np.complex128]
    singular_values: NDArray[np.float64]
    right_adjoint: NDArray[np.complex128]
    shifted_hankel: NDArray[np.complex128]
    numerical_rank: int
    max_supported_modes: int


def validate_samples(t: ArrayLike, values: ArrayLike) -> ValidatedSamples:
    """Validate and scale a scalar correlation sampled on a uniform grid."""

    times = np.asarray(t, dtype=float)
    samples = np.asarray(values, dtype=np.complex128)
    if times.ndim != 1 or samples.ndim != 1 or times.shape != samples.shape:
        raise InvalidSamplesError("t and values must be one-dimensional arrays of equal length")
    if times.size < 6:
        raise InvalidSamplesError("at least six samples are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(samples)):
        raise InvalidSamplesError("samples must be finite")
    if np.any(np.diff(times) <= 0):
        raise InvalidSamplesError("t must be strictly increasing")
    step = float(np.median(np.diff(times)))
    uniform_error = float(np.max(np.abs(np.diff(times) - step)))
    if uniform_error > 1e-7 * max(1.0, abs(step)):
        raise InvalidSamplesError(
            "the current exponential fitter requires a uniformly sampled time grid"
        )
    if abs(float(times[0])) > 1e-10 * max(1.0, abs(float(times[-1]))):
        raise InvalidSamplesError("the time grid must start at t=0")
    amplitude_scale = float(np.max(np.abs(samples)))
    if amplitude_scale == 0.0:
        raise InvalidSamplesError("the identically zero correlation has no nontrivial realization")
    if abs(float(np.imag(samples[0]))) > 1e-7 * amplitude_scale:
        raise InvalidSamplesError(
            "a stationary scalar correlation must be real at t=0; check Green-function factors"
        )
    if float(np.real(samples[0])) < -1e-10 * amplitude_scale:
        raise InvalidSamplesError("a physical scalar correlation cannot be negative at t=0")
    time_scale = float(times[-1])
    if time_scale <= 0:
        raise InvalidSamplesError("the time range must be positive")
    return ValidatedSamples(times, samples, time_scale, amplitude_scale)


def _error_metrics(
    reference: NDArray[np.complex128], approximation: NDArray[np.complex128]
) -> tuple[float, float]:
    residual = approximation - reference
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    relative_rms = float(np.linalg.norm(residual) / denominator)
    max_absolute = float(np.max(np.abs(residual)))
    return relative_rms, max_absolute


def _prepare_matrix_pencil(
    samples: NDArray[np.complex128],
) -> _MatrixPencilData:
    """Build one reusable Hankel SVD and determine its mode ceiling."""

    n_samples = samples.size
    n_rows = n_samples // 2
    x = hankel(samples[:n_rows], samples[n_rows - 1 : n_samples - 1])
    y = hankel(samples[1 : n_rows + 1], samples[n_rows:n_samples])
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    cutoff = np.finfo(float).eps * max(x.shape) * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > cutoff))
    algebraic_limit = min(x.shape) - 1
    return _MatrixPencilData(
        left_vectors=np.asarray(u, dtype=np.complex128),
        singular_values=np.asarray(singular_values, dtype=float),
        right_adjoint=np.asarray(vh, dtype=np.complex128),
        shifted_hankel=np.asarray(y, dtype=np.complex128),
        numerical_rank=numerical_rank,
        max_supported_modes=min(numerical_rank, algebraic_limit),
    )


def _matrix_pencil(
    samples: NDArray[np.complex128],
    dt: float,
    rank: int,
    *,
    min_decay: float,
    max_decay: float,
    pencil: _MatrixPencilData | None = None,
) -> NDArray[np.complex128]:
    pencil = _prepare_matrix_pencil(samples) if pencil is None else pencil
    if rank > pencil.max_supported_modes:
        if rank > pencil.numerical_rank:
            reason = (
                f"the Hankel matrix has numerical rank {pencil.numerical_rank}"
            )
        else:
            reason = f"the {samples.size}-point sample window"
        raise BandLimitError(
            rank,
            pencil.max_supported_modes,
            reason=reason,
        )

    ur = pencil.left_vectors[:, :rank]
    vr = pencil.right_adjoint.conj().T[:, :rank]
    reduced_shift = (
        ur.conj().T @ pencil.shifted_hankel @ vr
    ) / pencil.singular_values[:rank][None, :]
    poles = np.linalg.eigvals(reduced_shift)
    if np.any(np.abs(poles) < np.finfo(float).tiny):
        raise RealizationError("matrix pencil returned a zero discrete-time pole")

    magnitudes = np.abs(poles)
    phases = np.angle(poles)
    lower_magnitude = np.exp(-max_decay * dt)
    upper_magnitude = np.exp(-min_decay * dt)
    magnitudes = np.clip(magnitudes, lower_magnitude, upper_magnitude)
    stabilized = magnitudes * np.exp(1j * phases)
    rates = -np.log(stabilized) / dt
    order = np.lexsort((np.imag(rates), np.real(rates)))
    return np.asarray(rates[order], dtype=np.complex128)


def _solve_weights(
    times: NDArray[np.float64],
    samples: NDArray[np.complex128],
    rates: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    design = np.exp(-np.multiply.outer(times, rates))
    weights, *_ = np.linalg.lstsq(design, samples, rcond=None)
    return np.asarray(weights, dtype=np.complex128)


def _polish(
    times: NDArray[np.float64],
    samples: NDArray[np.complex128],
    rates: NDArray[np.complex128],
    weights: NDArray[np.complex128],
    *,
    min_decay: float,
    max_decay: float,
    max_nfev: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    n_modes = rates.size
    amplitude = max(float(np.linalg.norm(samples)), np.finfo(float).tiny)
    initial = np.concatenate(
        [
            np.log(np.real(rates)),
            np.imag(rates),
            np.real(weights),
            np.imag(weights),
        ]
    )
    nyquist = np.pi / float(times[1] - times[0])
    lower = np.concatenate(
        [
            np.full(n_modes, np.log(min_decay)),
            np.full(n_modes, -nyquist),
            np.full(2 * n_modes, -np.inf),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_modes, np.log(max_decay)),
            np.full(n_modes, nyquist),
            np.full(2 * n_modes, np.inf),
        ]
    )

    cached_parameters: NDArray[np.float64] | None = None
    cached_residual: NDArray[np.float64] | None = None
    cached_jacobian: NDArray[np.float64] | None = None

    def evaluate(
        parameters: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nonlocal cached_parameters, cached_residual, cached_jacobian
        if cached_parameters is not None and np.array_equal(parameters, cached_parameters):
            assert cached_residual is not None and cached_jacobian is not None
            return cached_residual, cached_jacobian
        fitted_rates = np.exp(parameters[:n_modes]) + 1j * parameters[n_modes : 2 * n_modes]
        fitted_weights = (
            parameters[2 * n_modes : 3 * n_modes]
            + 1j * parameters[3 * n_modes : 4 * n_modes]
        )
        exponentials = np.exp(-np.multiply.outer(times, fitted_rates))
        difference = (exponentials @ fitted_weights - samples) / amplitude
        complex_jacobian = np.empty(
            (times.size, 4 * n_modes), dtype=np.complex128
        )
        rate_derivative = -times[:, None] * exponentials * fitted_weights[None, :]
        complex_jacobian[:, :n_modes] = (
            rate_derivative * np.real(fitted_rates)[None, :]
        )
        complex_jacobian[:, n_modes : 2 * n_modes] = 1j * rate_derivative
        complex_jacobian[:, 2 * n_modes : 3 * n_modes] = exponentials
        complex_jacobian[:, 3 * n_modes :] = 1j * exponentials
        cached_parameters = parameters.copy()
        cached_residual = np.concatenate([np.real(difference), np.imag(difference)])
        cached_jacobian = np.vstack(
            [np.real(complex_jacobian), np.imag(complex_jacobian)]
        ) / amplitude
        return cached_residual, cached_jacobian

    solution = least_squares(
        lambda parameters: evaluate(parameters)[0],
        initial,
        jac=lambda parameters: evaluate(parameters)[1],
        bounds=(lower, upper),
        max_nfev=max_nfev,
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    parameters = solution.x
    fitted_rates = np.exp(parameters[:n_modes]) + 1j * parameters[n_modes : 2 * n_modes]
    # Re-solve the linear part after the nonlinear pole refinement.
    fitted_weights = _solve_weights(times, samples, fitted_rates)
    order = np.lexsort((np.imag(fitted_rates), np.real(fitted_rates)))
    return fitted_rates[order], fitted_weights[order]


def _fit_fixed_order(
    times: NDArray[np.float64],
    samples: NDArray[np.complex128],
    n_modes: int,
    *,
    polish: bool,
    max_nfev: int,
    tolerance: float,
    pencil: _MatrixPencilData | None = None,
) -> ExponentialFit:
    duration = float(times[-1])
    dt = float(times[1] - times[0])
    min_decay = max(1e-10 / duration, np.finfo(float).eps)
    max_decay = max(50.0 / dt, 100.0 * min_decay)
    rates = _matrix_pencil(
        samples,
        dt,
        n_modes,
        min_decay=min_decay,
        max_decay=max_decay,
        pencil=pencil,
    )
    weights = _solve_weights(times, samples, rates)
    initial = np.exp(-np.multiply.outer(times, rates)) @ weights
    initial_relative_rms, _ = _error_metrics(samples, initial)
    if polish and initial_relative_rms > tolerance:
        rates, weights = _polish(
            times,
            samples,
            rates,
            weights,
            min_decay=min_decay,
            max_decay=max_decay,
            max_nfev=max_nfev,
        )
    approximation = np.exp(-np.multiply.outer(times, rates)) @ weights
    relative_rms, max_absolute = _error_metrics(samples, approximation)
    return ExponentialFit(rates, weights, relative_rms, max_absolute)


def fit_exponentials(
    t: ArrayLike,
    values: ArrayLike,
    *,
    n_modes: int | None = None,
    max_modes: int = 12,
    eps: float | None = None,
    tolerance: float | None = None,
    polish: bool = True,
    max_nfev: int = 2_000,
) -> ExponentialFit:
    """Fit a stable complex sum of exponentials to a real-time correlation.

    Pass ``n_modes=N`` for a fixed-order fit, or omit it and pass ``eps`` for
    automatic order selection. The latter uses the smallest model whose
    full-window relative RMS error meets ``eps``. ``tolerance`` is retained as
    a backwards-compatible alias for ``eps``.

    If fixed ``N`` exceeds the numerical information in the sampled time data,
    :class:`BandLimitError` is raised. If automatic selection reaches that same
    limit before meeting ``eps``, the best fit is returned with a
    :class:`BandLimitWarning` and matching metadata on the result.
    """

    if eps is not None and n_modes is not None:
        raise ValueError("provide either eps or n_modes, not both")
    if eps is not None and tolerance is not None:
        raise ValueError("eps and tolerance are aliases; provide only one")
    requested_tolerance = (
        _DEFAULT_TOLERANCE
        if eps is None and tolerance is None
        else float(eps if eps is not None else tolerance)
    )
    if not np.isfinite(requested_tolerance) or requested_tolerance <= 0.0:
        raise ValueError("eps must be a finite positive number")

    validated = validate_samples(t, values)
    times = validated.times / validated.time_scale
    samples = validated.values / validated.amplitude_scale
    rank_limit = min(times.size // 2, times.size - times.size // 2) - 1
    if rank_limit < 1:
        raise InvalidSamplesError("the sample window is too short")
    if n_modes is not None and (
        isinstance(n_modes, (bool, np.bool_))
        or not isinstance(n_modes, (int, np.integer))
    ):
        raise InvalidSamplesError("n_modes must be an integer")

    result_warnings: list[str] = []
    result_details: dict[str, object]
    if n_modes is not None:
        n_modes = int(n_modes)
        if n_modes < 1:
            raise InvalidSamplesError("n_modes must be positive")
        if n_modes > rank_limit:
            raise BandLimitError(
                n_modes,
                rank_limit,
                reason=f"the {times.size}-point sample window",
            )
        pencil = _prepare_matrix_pencil(samples)
        result_details = {
            "selection_mode": "fixed_n",
            "requested_n_modes": n_modes,
            "numerical_hankel_rank": pencil.numerical_rank,
            "max_supported_modes": pencil.max_supported_modes,
            "band_limit_reached": False,
        }
        try:
            selected = _fit_fixed_order(
                times,
                samples,
                n_modes,
                polish=polish,
                max_nfev=max_nfev,
                tolerance=requested_tolerance,
                pencil=pencil,
            )
        except BandLimitError:
            raise
        except (RealizationError, np.linalg.LinAlgError, ValueError) as error:
            raise RealizationError(f"N={n_modes} exponential fit failed: {error}") from error
    else:
        if max_modes < 1:
            raise InvalidSamplesError("max_modes must be positive")
        # Select order by extrapolation onto the final 20% of the time window,
        # then refit the selected order using every sample.
        training_size = max(6, int(np.ceil(0.8 * times.size)))
        training_size = min(training_size, times.size - 1)
        training_times = times[:training_size]
        training_samples = samples[:training_size]
        validation_times = times[training_size:]
        validation_samples = samples[training_size:]
        training_pencil = _prepare_matrix_pencil(training_samples)
        full_pencil = _prepare_matrix_pencil(samples)
        supported_modes = min(
            training_pencil.max_supported_modes,
            full_pencil.max_supported_modes,
            rank_limit,
        )
        orders = list(range(1, min(int(max_modes), supported_modes) + 1))
        failures: list[str] = []
        selection: list[tuple[float, float, int]] = []
        for order in orders:
            try:
                candidate = _fit_fixed_order(
                    training_times,
                    training_samples,
                    order,
                    polish=polish,
                    max_nfev=max_nfev,
                    tolerance=requested_tolerance,
                    pencil=training_pencil,
                )
            except (RealizationError, np.linalg.LinAlgError, ValueError) as error:
                failures.append(f"N={order}: {error}")
                continue
            validation_prediction = candidate.evaluate(validation_times)
            validation_error, _ = _error_metrics(
                validation_samples, validation_prediction
            )
            selection.append(
                (validation_error, candidate.relative_rms_error, order)
            )
            if (
                validation_error <= requested_tolerance
                and candidate.relative_rms_error <= requested_tolerance
            ):
                break
        if not selection:
            joined = "; ".join(failures)
            raise RealizationError(f"all exponential fits failed ({joined})")
        acceptable_orders = [
            order
            for validation_error, training_error, order in selection
            if validation_error <= requested_tolerance
            and training_error <= requested_tolerance
        ]
        if acceptable_orders:
            selected_order = min(acceptable_orders)
        else:
            selected_order = min(selection)[2]
        selected = _fit_fixed_order(
            times,
            samples,
            selected_order,
            polish=polish,
            max_nfev=max_nfev,
            tolerance=requested_tolerance,
            pencil=full_pencil,
        )
        result_details = {
            "selection_mode": "eps",
            "requested_eps": requested_tolerance,
            "selected_n_modes": selected_order,
            "numerical_hankel_rank": full_pencil.numerical_rank,
            "selection_hankel_rank": training_pencil.numerical_rank,
            "max_supported_modes": supported_modes,
            "band_limit_reached": False,
        }

    physical_rates = selected.rates / validated.time_scale
    physical_weights = selected.weights * validated.amplitude_scale
    approximation = (
        np.exp(-np.multiply.outer(validated.times, physical_rates))
        @ physical_weights
    )
    relative_rms, max_absolute = _error_metrics(validated.values, approximation)
    if n_modes is None and relative_rms > requested_tolerance:
        if int(max_modes) >= supported_modes:
            message = (
                "band limit reached: "
                f"requested eps={requested_tolerance:.3e}, but the best returned "
                f"fit has relative RMS error {relative_rms:.3e}; the supplied "
                f"data have a numerical Hankel-rank ceiling of N={supported_modes}. "
                "For a better result, provide finer real-time data (a smaller "
                "time step and, if needed, a longer time window)."
            )
            result_details["band_limit_reached"] = True
            warning_category: type[Warning] = BandLimitWarning
        else:
            message = (
                f"requested eps={requested_tolerance:.3e} was not reached with "
                f"max_modes={int(max_modes)} (best returned relative RMS error "
                f"{relative_rms:.3e}); increase max_modes to continue the search"
            )
            result_details["mode_limit_reached"] = True
            warning_category = UserWarning
        result_warnings.append(message)
        python_warnings.warn(message, warning_category, stacklevel=2)
    return ExponentialFit(
        physical_rates,
        physical_weights,
        relative_rms,
        max_absolute,
        warnings=tuple(result_warnings),
        details=result_details,
    )
