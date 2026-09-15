"""Physical spectral-factor realization from arXiv:2604.06466."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .exceptions import RealizationError
from .realization import BackendResult
from .types import ExponentialFit, LindbladModel


ComplexArray = NDArray[np.complex128]


def _product(polynomials: list[np.poly1d]) -> np.poly1d:
    result = np.poly1d([1.0])
    for polynomial in polynomials:
        result *= polynomial
    return result


def _spectral_numerator(rates: ComplexArray, weights: ComplexArray) -> NDArray[np.float64]:
    minus = [np.poly1d([-1j, rate]) for rate in rates]
    plus = [np.poly1d([1j, np.conj(rate)]) for rate in rates]
    numerator = np.poly1d([0.0j])
    for index, weight in enumerate(weights):
        numerator += weight * _product(minus[:index] + minus[index + 1 :] + plus)
        numerator += np.conj(weight) * _product(minus + plus[:index] + plus[index + 1 :])
    coefficients = np.asarray(numerator.c, dtype=np.complex128)
    coefficient_scale = max(1.0, float(np.max(np.abs(coefficients))))
    imaginary_error = float(np.max(np.abs(np.imag(coefficients)))) / coefficient_scale
    if imaginary_error > 1e-7:
        raise RealizationError(
            "the exponential correlation does not yield a numerically real spectrum numerator"
        )
    real_coefficients = np.real(coefficients)
    threshold = 1e-11 * max(1.0, float(np.max(np.abs(real_coefficients))))
    while real_coefficients.size > 1 and abs(float(real_coefficients[0])) <= threshold:
        real_coefficients = real_coefficients[1:]
    return np.asarray(real_coefficients, dtype=float)


def _select_factor_roots(
    coefficients: NDArray[np.float64], *, root_tolerance: float
) -> ComplexArray:
    degree = coefficients.size - 1
    if degree % 2:
        raise RealizationError("a nonnegative rational spectrum must have even numerator degree")
    if degree == 0:
        return np.empty(0, dtype=np.complex128)
    roots = list(np.roots(coefficients).astype(np.complex128))
    selected: list[complex] = []
    real_roots: list[float] = []
    for root in roots:
        scale = 1.0 + abs(float(np.real(root)))
        if abs(float(np.imag(root))) <= root_tolerance * scale:
            real_roots.append(float(np.real(root)))
        elif np.imag(root) > 0:
            selected.append(complex(root))

    real_roots.sort()
    while real_roots:
        center = real_roots.pop(0)
        cluster = [center]
        remaining: list[float] = []
        for root in real_roots:
            if abs(root - center) <= 20.0 * root_tolerance * (1.0 + abs(center)):
                cluster.append(root)
            else:
                remaining.append(root)
        real_roots = remaining
        if len(cluster) % 2:
            raise RealizationError(
                "the fitted spectrum has a real zero of odd multiplicity and is not nonnegative"
            )
        selected.extend([complex(float(np.mean(cluster)), 0.0)] * (len(cluster) // 2))

    expected = degree // 2
    if len(selected) != expected:
        raise RealizationError(
            "could not pair the spectral numerator roots into a positive factor"
        )
    return np.asarray(selected, dtype=np.complex128)


def _spectral_factor_residues(
    rates: ComplexArray,
    weights: ComplexArray,
    *,
    positivity_tolerance: float,
    root_tolerance: float,
) -> tuple[ComplexArray, float]:
    coefficients = _spectral_numerator(rates, weights)
    degree = coefficients.size - 1
    leading = float(coefficients[0])
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    if leading <= positivity_tolerance * scale:
        raise RealizationError("the fitted spectral density is not strictly positive at infinity")

    factor_roots = _select_factor_roots(coefficients, root_tolerance=root_tolerance)
    factor_coefficients = np.atleast_1d(np.poly(factor_roots)).astype(np.complex128)
    factor_coefficients *= np.sqrt(leading)
    numerator_factor = np.poly1d(factor_coefficients)

    separation = np.abs(rates[:, None] - rates[None, :])
    np.fill_diagonal(separation, np.inf)
    rate_scale = max(1.0, float(np.max(np.abs(rates))))
    if rates.size > 1 and float(np.min(separation)) < 1e-9 * rate_scale:
        raise RealizationError("spectral factorization requires distinct, well-separated rates")

    residues = np.empty(rates.size, dtype=np.complex128)
    for j, rate in enumerate(rates):
        pole = -1j * rate
        denominator = 1.0 + 0.0j
        for k, other_rate in enumerate(rates):
            if k != j:
                denominator *= other_rate - rate
        residues[j] = numerator_factor(pole) / denominator

    denominators = rates[:, None] + rates[None, :].conj()
    reconstructed = np.sum(
        np.outer(residues, residues.conj()) / denominators,
        axis=1,
    )
    residual = float(
        np.linalg.norm(reconstructed - weights)
        / max(np.linalg.norm(weights), np.finfo(float).tiny)
    )
    if residual > max(1e-6, 100.0 * positivity_tolerance):
        raise RealizationError(
            f"spectral factorization residue check failed (relative residual {residual:.3e})"
        )
    return residues, residual


def realize_physical(
    exponential: ExponentialFit,
    *,
    positivity_tolerance: float = 1e-9,
    root_tolerance: float = 1e-6,
    condition_limit: float = 1e14,
) -> BackendResult:
    """Construct an exact coupled Lindblad model for a physical exponential BCF.

    This implements the spectral-factor/Ornstein--Uhlenbeck construction of
    Müller and Strunz. It rejects exponential fits whose continuous spectrum is
    not nonnegative, rather than silently projecting them.
    """

    rates = np.asarray(exponential.rates, dtype=np.complex128)
    weights = np.asarray(exponential.weights, dtype=np.complex128)
    if np.any(np.real(rates) <= 0):
        raise RealizationError("all rates must have positive real parts")
    amplitude_scale = float(np.max(np.abs(weights)))
    rate_scale = float(np.median(np.abs(rates)))
    if amplitude_scale <= 0 or rate_scale <= 0:
        raise RealizationError("rates and weights must be nonzero")
    scaled_rates = rates / rate_scale
    scaled_weights = weights / amplitude_scale
    alpha_zero = np.sum(scaled_weights)
    if abs(float(np.imag(alpha_zero))) > positivity_tolerance * max(1.0, abs(alpha_zero)):
        raise RealizationError("the exponential fit is not real at t=0")
    if float(np.real(alpha_zero)) <= positivity_tolerance:
        raise RealizationError("the exponential fit is not positive at t=0")

    residues, factor_residual = _spectral_factor_residues(
        scaled_rates,
        scaled_weights,
        positivity_tolerance=positivity_tolerance,
        root_tolerance=root_tolerance,
    )
    denominators = scaled_rates[:, None] + scaled_rates[None, :].conj()
    covariance = np.outer(residues, residues.conj()) / denominators
    covariance = (covariance + covariance.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -positivity_tolerance * covariance_scale:
        raise RealizationError("spectral factor produced a non-positive covariance")
    floor = np.finfo(float).eps * covariance_scale
    if float(np.min(eigenvalues)) <= floor:
        raise RealizationError("the physical realization covariance is numerically singular")
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    if condition_number > condition_limit:
        raise RealizationError(
            f"the physical realization is ill-conditioned (condition {condition_number:.3e})"
        )

    # (V^dagger V)^-1 = P. U=I is a valid choice of the paper's unitary gauge.
    transform = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    inverse_transform = np.linalg.inv(transform)
    transformed_drift = transform @ np.diag(scaled_rates) @ inverse_transform
    damping = (transformed_drift + transformed_drift.conj().T) / 2.0
    hamiltonian = (transformed_drift - transformed_drift.conj().T) / (2.0j)
    damping_values, damping_vectors = np.linalg.eigh(damping)
    damping_scale = max(1.0, float(np.max(np.abs(damping_values))))
    if float(np.min(damping_values)) < -1e-8 * damping_scale:
        raise RealizationError("constructed drift has non-positive damping")
    damping = (
        damping_vectors
        @ np.diag(np.maximum(damping_values, 0.0))
        @ damping_vectors.conj().T
    )
    auxiliary_coupling = np.ones(rates.size, dtype=np.complex128)
    coupling = inverse_transform.conj().T @ auxiliary_coupling

    model = LindbladModel(
        hamiltonian=hamiltonian * rate_scale,
        damping=damping * rate_scale,
        coupling=coupling * np.sqrt(amplitude_scale),
    )
    check_times = np.linspace(0.0, 5.0 / float(np.min(np.real(rates))), 101)
    target = exponential.evaluate(check_times)
    model_values = model.evaluate(check_times)
    model_residual = float(
        np.linalg.norm(model_values - target)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
    )
    if model_residual > max(2e-6, 200.0 * positivity_tolerance):
        raise RealizationError(
            f"physical realization failed reconstruction check ({model_residual:.3e})"
        )
    return BackendResult(
        model=model,
        condition_number=condition_number,
        conversion_residual=max(factor_residual, model_residual),
        details={
            "spectral_factor_residual": factor_residual,
            "model_residual": model_residual,
            "covariance_eigenvalues": eigenvalues * amplitude_scale,
        },
    )


def realize_positive_gram(
    rates: ComplexArray,
    gram_factor: ComplexArray,
    *,
    positivity_tolerance: float = 1e-9,
    condition_limit: float = 1e14,
) -> BackendResult:
    """Realize the positive exponential parametrization of Eq. (S44).

    If ``R = gram_factor @ gram_factor.conj().T``, the exponential weights are
    ``G_j = sum_k R_jk / (rates_j + rates_k.conj())``. Unlike explicit scalar
    spectral factorization, this form also covers multiple damping channels and
    remains numerically useful near repeated spectral zeros.
    """

    rates = np.asarray(rates, dtype=np.complex128)
    factor = np.asarray(gram_factor, dtype=np.complex128)
    if rates.ndim != 1 or factor.ndim != 2 or factor.shape[0] != rates.size:
        raise ValueError("gram_factor must have one row per exponential rate")
    if np.any(np.real(rates) <= 0):
        raise RealizationError("all rates must have positive real parts")
    gram = factor @ factor.conj().T
    covariance = gram / (rates[:, None] + rates[None, :].conj())
    covariance = (covariance + covariance.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -positivity_tolerance * scale:
        raise RealizationError("positive Gram parameters produced a non-positive covariance")
    floor = np.finfo(float).eps * scale
    if float(np.min(eigenvalues)) <= floor:
        raise RealizationError("positive Gram realization is numerically singular")
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    if condition_number > condition_limit:
        raise RealizationError(
            f"positive Gram realization is ill-conditioned (condition {condition_number:.3e})"
        )
    transform = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    inverse_transform = np.linalg.inv(transform)
    drift = transform @ np.diag(rates) @ inverse_transform
    damping = (drift + drift.conj().T) / 2.0
    hamiltonian = (drift - drift.conj().T) / (2.0j)
    damping_values, damping_vectors = np.linalg.eigh(damping)
    damping_scale = max(1.0, float(np.max(np.abs(damping_values))))
    if float(np.min(damping_values)) < -1e-8 * damping_scale:
        raise RealizationError("positive Gram drift has non-positive damping")
    damping = (
        damping_vectors
        @ np.diag(np.maximum(damping_values, 0.0))
        @ damping_vectors.conj().T
    )
    coupling = inverse_transform.conj().T @ np.ones(rates.size, dtype=np.complex128)
    model = LindbladModel(hamiltonian, damping, coupling)
    return BackendResult(
        model=model,
        condition_number=condition_number,
        conversion_residual=0.0,
        details={"covariance_eigenvalues": eigenvalues},
    )
