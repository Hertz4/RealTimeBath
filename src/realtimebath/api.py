"""Public fitting API."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import toeplitz

from .exceptions import OptionalDependencyError, RealizationError
from .exponential import fit_exponentials, validate_samples
from .optimization import optimize_after_sdp
from .positive_fit import optimize_positive_exponentials
from .realization import BackendResult
from .route_physical import realize_physical
from .route_sdp import cvxpy_available, realize_sdp
from .types import (
    ExponentialFit,
    FermionicLindbladFit,
    FitDiagnostics,
    LindbladFit,
)


def _relative_error(reference: np.ndarray, approximation: np.ndarray) -> tuple[float, float]:
    difference = approximation - reference
    relative = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )
    return relative, float(np.max(np.abs(difference)))


def _toeplitz_warning(values: np.ndarray) -> str | None:
    # A principal stationary-kernel matrix is a useful finite-grid diagnostic.
    sample = values[: min(values.size, 256)]
    kernel = toeplitz(sample, sample.conj())
    kernel = (kernel + kernel.conj().T) / 2.0
    minimum = float(np.min(np.linalg.eigvalsh(kernel)))
    scale = max(1.0, float(np.linalg.norm(kernel, ord=2)))
    if minimum < -1e-8 * scale:
        return (
            "input samples fail a finite-grid positive-kernel check; "
            "the returned model is a physical approximation"
        )
    return None


def _build_result(
    backend: BackendResult,
    exponential: ExponentialFit,
    times: np.ndarray,
    values: np.ndarray,
    *,
    method: str,
    tolerance: float,
    selection_exponential: ExponentialFit | None = None,
) -> LindbladFit:
    approximation = backend.model.evaluate(times)
    relative, maximum = _relative_error(values, approximation)
    damping_eigenvalues = np.linalg.eigvalsh(backend.model.damping)
    stability = float(np.max(np.real(np.linalg.eigvals(backend.model.generator))))
    selection_exponential = (
        exponential if selection_exponential is None else selection_exponential
    )
    warnings = [*selection_exponential.warnings, *backend.warnings]
    kernel_warning = _toeplitz_warning(values)
    if kernel_warning is not None:
        warnings.append(kernel_warning)
    if relative > tolerance:
        warnings.append(
            f"requested tolerance was not reached (relative RMS error {relative:.3e})"
        )
    diagnostics = FitDiagnostics(
        method=method,
        n_modes=backend.model.n_modes,
        relative_rms_error=relative,
        max_absolute_error=maximum,
        min_damping_eigenvalue=float(np.min(damping_eigenvalues)),
        stability_abscissa=stability,
        condition_number=backend.condition_number,
        conversion_residual=backend.conversion_residual,
        warnings=tuple(dict.fromkeys(warnings)),
        details={
            **selection_exponential.details,
            **backend.details,
            "exponential_relative_rms_error": exponential.relative_rms_error,
            "exponential_max_absolute_error": exponential.max_absolute_error,
        },
    )
    return LindbladFit(backend.model, exponential, diagnostics)


def fit_correlation(
    t: ArrayLike,
    values: ArrayLike,
    *,
    n_modes: int | None = None,
    max_modes: int = 12,
    eps: float | None = None,
    tolerance: float | None = None,
    method: str = "auto",
    polish: bool = True,
    max_nfev: int = 2_000,
    solver: str | None = None,
    optimize: bool = False,
    optimization_max_nfev: int = 1_000,
    optimization_max_points: int = 251,
    optimization_regularization: float = 1e-10,
) -> LindbladFit:
    """Fit a scalar real-time correlation with coupled Lindblad pseudomodes.

    Parameters
    ----------
    t, values:
        Uniform complex samples on a grid beginning at zero. Values use the
        correlation convention, not the conventional NEGF prefactors ``+/-i``.
    n_modes, eps:
        Pass ``n_modes=N`` for a fixed-order fit, or pass ``eps`` (without
        ``n_modes``) to select the smallest model meeting that relative RMS
        target. ``tolerance`` remains available as a compatibility alias.
    method:
        ``"physical"`` uses exact spectral factorization, ``"sdp"`` uses the
        convex physical projection, and ``"auto"`` tries the exact route before
        falling back to the SDP route. ``"positive"`` exposes an experimental
        optimizer for the second paper's positive-Gram ansatz; do not use it as
        a general route from sampled data. That construction is appropriate only
        when a positive-exponential representation is already known, and finding
        one a priori is nontrivial. If its rates and Gram factor are available,
        prefer ``realize_positive_gram``. Setting ``optimize=True`` forces the SDP
        path and performs a physical time-domain refinement.
    """

    if method not in {"auto", "physical", "positive", "sdp"}:
        raise ValueError("method must be 'auto', 'physical', 'positive', or 'sdp'")
    if eps is not None and n_modes is not None:
        raise ValueError("provide either eps or n_modes, not both")
    if eps is not None and tolerance is not None:
        raise ValueError("eps and tolerance are aliases; provide only one")
    requested_accuracy = (
        1e-6
        if eps is None and tolerance is None
        else eps
        if eps is not None
        else tolerance
    )
    assert requested_accuracy is not None
    fit_tolerance = float(requested_accuracy)
    if not np.isfinite(fit_tolerance) or fit_tolerance <= 0:
        raise ValueError("eps must be a finite positive number")
    if optimize and method not in {"auto", "sdp"}:
        raise ValueError("optimize=True is an optional post-SDP step")
    validated = validate_samples(t, values)
    exponential = fit_exponentials(
        validated.times,
        validated.values,
        n_modes=n_modes,
        max_modes=max_modes,
        tolerance=fit_tolerance,
        polish=polish,
        max_nfev=max_nfev,
    )

    if method == "physical":
        backend = realize_physical(exponential)
        return _build_result(
            backend,
            exponential,
            validated.times,
            validated.values,
            method="physical",
            tolerance=fit_tolerance,
        )
    if method == "positive":
        try:
            seed = realize_physical(exponential)
        except RealizationError:
            seed = realize_sdp(exponential, solver=solver)
        backend, positive_exponential = optimize_positive_exponentials(
            validated.times,
            validated.values,
            seed,
            max_nfev=optimization_max_nfev,
            max_points=optimization_max_points,
            regularization=optimization_regularization,
        )
        return _build_result(
            backend,
            positive_exponential,
            validated.times,
            validated.values,
            method="positive",
            tolerance=fit_tolerance,
            selection_exponential=exponential,
        )
    if method == "sdp":
        backend = realize_sdp(exponential, solver=solver)
        if optimize:
            backend = optimize_after_sdp(
                validated.times,
                validated.values,
                backend,
                max_nfev=optimization_max_nfev,
                max_points=optimization_max_points,
                regularization=optimization_regularization,
            )
        return _build_result(
            backend,
            exponential,
            validated.times,
            validated.values,
            method="sdp",
            tolerance=fit_tolerance,
        )

    physical_error: Exception | None = None
    if not optimize:
        try:
            backend = realize_physical(exponential)
            return _build_result(
                backend,
                exponential,
                validated.times,
                validated.values,
                method="physical",
                tolerance=fit_tolerance,
            )
        except RealizationError as error:
            physical_error = error
    else:
        physical_error = RealizationError("post-SDP optimization was requested")
    if cvxpy_available():
        backend = realize_sdp(exponential, solver=solver)
        warnings = list(backend.warnings)
        warnings.append(f"exact physical route was unavailable: {physical_error}")
        backend = BackendResult(
            model=backend.model,
            condition_number=backend.condition_number,
            conversion_residual=backend.conversion_residual,
            warnings=tuple(warnings),
            details=backend.details,
        )
        if optimize:
            backend = optimize_after_sdp(
                validated.times,
                validated.values,
                backend,
                max_nfev=optimization_max_nfev,
                max_points=optimization_max_points,
                regularization=optimization_regularization,
            )
        return _build_result(
            backend,
            exponential,
            validated.times,
            validated.values,
            method="sdp",
            tolerance=fit_tolerance,
        )
    raise OptionalDependencyError(
        "the exponential fit was not exactly physical and the SDP fallback is unavailable; "
        "install realtimebath[sdp]. Exact-route error: " + str(physical_error)
    )


def fit_fermionic(
    t: ArrayLike,
    delta_lesser: ArrayLike,
    delta_greater: ArrayLike,
    *,
    convention: str = "correlation",
    **fit_options,
) -> FermionicLindbladFit:
    """Fit the filled lesser and empty greater fermionic auxiliary sectors.

    With ``convention="correlation"`` the inputs are the positive kernels used
    in Eq. S6 of arXiv:2506.10308. With ``convention="negf"`` the inputs obey
    ``Delta_less = +i*C_less`` and ``Delta_greater = -i*C_greater``.
    """

    if convention not in {"correlation", "negf"}:
        raise ValueError("convention must be 'correlation' or 'negf'")
    lesser_values = np.asarray(delta_lesser, dtype=np.complex128)
    greater_values = np.asarray(delta_greater, dtype=np.complex128)
    if convention == "negf":
        lesser_values = -1j * lesser_values
        greater_values = 1j * greater_values
    lesser = fit_correlation(t, lesser_values, **fit_options)
    greater = fit_correlation(t, greater_values, **fit_options)
    return FermionicLindbladFit(
        lesser=lesser,
        greater=greater,
        input_convention=convention,
    )
