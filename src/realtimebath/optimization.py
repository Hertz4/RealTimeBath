"""Physical time-domain refinement initialized by the SDP realization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.linalg import expm
from scipy.optimize import least_squares

from .exceptions import RealizationError
from .rank_one_fit import optimize_rank_one_exponentials
from .realization import BackendResult
from .types import LindbladModel


def _pack_hermitian(matrix: NDArray[np.complex128]) -> NDArray[np.float64]:
    n = matrix.shape[0]
    packed: list[float] = [float(np.real(matrix[i, i])) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            packed.extend([float(np.real(matrix[i, j])), float(np.imag(matrix[i, j]))])
    return np.asarray(packed, dtype=float)


def _unpack_hermitian(values: NDArray[np.float64], n: int) -> NDArray[np.complex128]:
    matrix = np.zeros((n, n), dtype=np.complex128)
    matrix[np.diag_indices(n)] = values[:n]
    offset = n
    for i in range(n):
        for j in range(i + 1, n):
            entry = values[offset] + 1j * values[offset + 1]
            matrix[i, j] = entry
            matrix[j, i] = np.conj(entry)
            offset += 2
    return matrix


def _pack_model(model: LindbladModel) -> NDArray[np.float64]:
    damping_values, damping_vectors = np.linalg.eigh(model.damping)
    damping_factor = (
        damping_vectors
        @ np.diag(np.sqrt(np.maximum(damping_values, 0.0)))
        @ damping_vectors.conj().T
    )
    return np.concatenate(
        [
            _pack_hermitian(model.hamiltonian),
            np.real(damping_factor).ravel(),
            np.imag(damping_factor).ravel(),
            np.real(model.coupling),
            np.imag(model.coupling),
        ]
    )


def _unpack_model(
    parameters: NDArray[np.float64], n: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    h_size = n * n
    factor_size = n * n
    hamiltonian = _unpack_hermitian(parameters[:h_size], n)
    offset = h_size
    damping_factor = (
        parameters[offset : offset + factor_size].reshape(n, n)
        + 1j
        * parameters[offset + factor_size : offset + 2 * factor_size].reshape(n, n)
    )
    offset += 2 * factor_size
    coupling = parameters[offset : offset + n] + 1j * parameters[offset + n : offset + 2 * n]
    damping = damping_factor @ damping_factor.conj().T
    return hamiltonian, damping, coupling


def _evaluate(
    hamiltonian: NDArray[np.complex128],
    damping: NDArray[np.complex128],
    coupling: NDArray[np.complex128],
    times: NDArray[np.float64],
) -> NDArray[np.complex128]:
    generator = -1j * hamiltonian - damping
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    if np.linalg.cond(eigenvectors) < 1e10:
        left = coupling.conj() @ eigenvectors
        right = np.linalg.solve(eigenvectors, coupling)
        return np.exp(np.multiply.outer(times, eigenvalues)) @ (left * right)
    result = np.empty(times.size, dtype=np.complex128)
    for index, time in enumerate(times):
        result[index] = np.vdot(coupling, expm(generator * time) @ coupling)
    return result


def optimize_after_sdp_full(
    t: ArrayLike,
    values: ArrayLike,
    initial: BackendResult,
    *,
    max_nfev: int = 300,
    max_points: int = 251,
    regularization: float = 1e-10,
) -> BackendResult:
    """Refine ``H, D, g`` in time while retaining Lindblad physicality.

    ``H`` is parametrized as Hermitian and ``D = B B^dagger`` during every
    objective evaluation. The optimizer therefore cannot leave the physical
    coupled-Lindblad manifold.
    """

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
    initial_model = LindbladModel(
        hamiltonian=initial.model.hamiltonian * time_scale,
        damping=initial.model.damping * time_scale,
        coupling=initial.model.coupling / np.sqrt(amplitude_scale),
    )
    initial_parameters = _pack_model(initial_model)
    n_modes = initial_model.n_modes
    if times.size > max_points:
        indices = np.unique(np.linspace(0, times.size - 1, max_points).round().astype(int))
    else:
        indices = np.arange(times.size)
    objective_times = scaled_times[indices]
    objective_target = scaled_target[indices]
    target_norm = max(float(np.linalg.norm(objective_target)), np.finfo(float).tiny)
    regularization_scale = np.sqrt(max(0.0, regularization) / initial_parameters.size)

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            hamiltonian, damping, coupling = _unpack_model(parameters, n_modes)
            difference = (
                _evaluate(hamiltonian, damping, coupling, objective_times)
                - objective_target
            ) / target_norm
            result = [np.real(difference), np.imag(difference)]
            if regularization_scale:
                result.append(regularization_scale * (parameters - initial_parameters))
            return np.concatenate(result)
        except np.linalg.LinAlgError:
            size = 2 * objective_times.size
            if regularization_scale:
                size += initial_parameters.size
            return np.full(size, 1e6, dtype=float)

    initial_residual = residual(initial_parameters)
    solution = least_squares(
        residual,
        initial_parameters,
        method="trf",
        jac="2-point",
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not np.all(np.isfinite(solution.x)):
        raise RealizationError("time-domain optimization returned non-finite parameters")
    hamiltonian, damping, coupling = _unpack_model(solution.x, n_modes)
    optimized_model = LindbladModel(
        hamiltonian=hamiltonian / time_scale,
        damping=damping / time_scale,
        coupling=coupling * np.sqrt(amplitude_scale),
    )
    initial_full = initial.model.evaluate(times)
    optimized_full = optimized_model.evaluate(times)
    normalization_l2 = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
    initial_l2 = float(np.linalg.norm(initial_full - target) / normalization_l2)
    optimized_l2 = float(np.linalg.norm(optimized_full - target) / normalization_l2)
    normalization_l1 = max(
        float(trapezoid(np.abs(target), times)), np.finfo(float).tiny
    )
    initial_l1 = float(
        trapezoid(np.abs(initial_full - target), times) / normalization_l1
    )
    optimized_l1 = float(
        trapezoid(np.abs(optimized_full - target), times) / normalization_l1
    )
    warnings = list(initial.warnings)
    if not solution.success:
        warnings.append(f"time-domain optimization stopped early: {solution.message}")
    if (
        optimized_l2 > initial_l2 * (1.0 + 1e-8)
        or optimized_l1 > initial_l1 * (1.0 + 1e-8)
    ):
        warnings.append(
            "time-domain optimization was discarded because it did not improve "
            "both full-grid L1 and L2 errors"
        )
        optimized_model = initial.model
        optimized_l2 = initial_l2
        optimized_l1 = initial_l1
    generator_vectors = np.linalg.eig(optimized_model.generator)[1]
    condition_number = float(np.linalg.cond(generator_vectors))
    return BackendResult(
        model=optimized_model,
        condition_number=condition_number,
        conversion_residual=initial.conversion_residual,
        warnings=tuple(warnings),
        details={
            **initial.details,
            "optimized": True,
            "optimization_success": bool(solution.success),
            "optimization_status": int(solution.status),
            "optimization_message": str(solution.message),
            "optimization_nfev": int(solution.nfev),
            "optimization_initial_objective_norm": float(np.linalg.norm(initial_residual)),
            "optimization_final_objective_norm": float(np.linalg.norm(solution.fun)),
            "initial_relative_l2_error": initial_l2,
            "optimized_relative_l2_error": optimized_l2,
            "initial_relative_l1_error": initial_l1,
            "optimized_relative_l1_error": optimized_l1,
        },
    )


def optimize_after_sdp(
    t: ArrayLike,
    values: ArrayLike,
    initial: BackendResult,
    *,
    max_nfev: int = 1_000,
    max_points: int = 251,
    regularization: float = 1e-10,
) -> BackendResult:
    """Refine the SDP seed in fast rank-one physical coordinates."""

    try:
        backend, _ = optimize_rank_one_exponentials(
            t,
            values,
            initial,
            max_nfev=max_nfev,
            max_points=max_points,
            regularization=regularization,
        )
        return BackendResult(
            model=backend.model,
            condition_number=backend.condition_number,
            conversion_residual=backend.conversion_residual,
            warnings=backend.warnings,
            details={**backend.details, "optimized": True},
        )
    except (RealizationError, np.linalg.LinAlgError) as error:
        warnings = list(initial.warnings)
        warnings.append(
            "rank-one refinement failed; using the slower full-matrix "
            f"optimization ({error})"
        )
        fallback_initial = BackendResult(
            model=initial.model,
            condition_number=initial.condition_number,
            conversion_residual=initial.conversion_residual,
            warnings=tuple(warnings),
            details=initial.details,
        )
        return optimize_after_sdp_full(
            t,
            values,
            fallback_initial,
            max_nfev=min(max_nfev, 300),
            max_points=max_points,
            regularization=regularization,
        )
