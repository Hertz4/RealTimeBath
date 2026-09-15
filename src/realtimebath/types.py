"""Typed result containers and the package's damping convention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExponentialFit:
    """A stable exponential representation ``sum(weights * exp(-rates*t))``."""

    rates: ComplexArray
    weights: ComplexArray
    relative_rms_error: float
    max_absolute_error: float
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rates = np.asarray(self.rates, dtype=np.complex128)
        weights = np.asarray(self.weights, dtype=np.complex128)
        if rates.ndim != 1 or weights.ndim != 1 or rates.shape != weights.shape:
            raise ValueError("rates and weights must be one-dimensional arrays of equal length")
        if np.any(np.real(rates) <= 0):
            raise ValueError("all exponential rates must have positive real parts")
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "weights", weights)

    @property
    def n_modes(self) -> int:
        return int(self.rates.size)

    def evaluate(self, t: ArrayLike) -> ComplexArray:
        times = np.asarray(t, dtype=float)
        return np.exp(-np.multiply.outer(times, self.rates)) @ self.weights

    def spectral_density(self, omega: ArrayLike) -> FloatArray:
        """Return the two-sided spectrum of the Hermitian time extension."""

        frequencies = np.asarray(omega, dtype=float)
        terms = self.weights[None, :] / (
            self.rates[None, :] - 1j * frequencies[..., None]
        )
        return np.asarray(2.0 * np.real(np.sum(terms, axis=-1)), dtype=float)


@dataclass(frozen=True)
class LindbladModel:
    r"""Coupled pseudomodes in the convention ``exp((-1j*H-D)*t)``.

    ``H`` is Hermitian, ``D`` is positive semidefinite, and the returned jump
    coefficient matrix satisfies ``jumps.conj().T @ jumps == 2*D``.
    """

    hamiltonian: ComplexArray
    damping: ComplexArray
    coupling: ComplexArray
    jumps: ComplexArray | None = None

    def __post_init__(self) -> None:
        h = np.asarray(self.hamiltonian, dtype=np.complex128)
        d = np.asarray(self.damping, dtype=np.complex128)
        g = np.asarray(self.coupling, dtype=np.complex128)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("hamiltonian must be square")
        if d.shape != h.shape or g.shape != (h.shape[0],):
            raise ValueError("damping and coupling dimensions do not match hamiltonian")
        h = (h + h.conj().T) / 2.0
        d = (d + d.conj().T) / 2.0
        jumps = self.jumps
        if jumps is None:
            values, vectors = np.linalg.eigh(d)
            scale = max(1.0, float(np.linalg.norm(d, ord=2)))
            if float(np.min(values)) < -1e-9 * scale:
                raise ValueError("damping must be positive semidefinite")
            values = np.maximum(values, 0.0)
            jumps = np.diag(np.sqrt(2.0 * values)) @ vectors.conj().T
        else:
            jumps = np.asarray(jumps, dtype=np.complex128)
            if jumps.ndim != 2 or jumps.shape[1] != h.shape[0]:
                raise ValueError("jumps must have one column per pseudomode")
        object.__setattr__(self, "hamiltonian", h)
        object.__setattr__(self, "damping", d)
        object.__setattr__(self, "coupling", g)
        object.__setattr__(self, "jumps", jumps)

    @property
    def n_modes(self) -> int:
        return int(self.coupling.size)

    @property
    def generator(self) -> ComplexArray:
        return -1j * self.hamiltonian - self.damping

    def evaluate(self, t: ArrayLike) -> ComplexArray:
        times = np.asarray(t, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eig(self.generator)
        if np.linalg.cond(eigenvectors) < 1e10:
            left = self.coupling.conj() @ eigenvectors
            right = np.linalg.solve(eigenvectors, self.coupling)
            weights = left * right
            return np.asarray(
                np.exp(np.multiply.outer(times, eigenvalues)) @ weights,
                dtype=np.complex128,
            )
        flat = times.reshape(-1)
        result = np.empty(flat.size, dtype=np.complex128)
        for index, time in enumerate(flat):
            evolved = expm(self.generator * time) @ self.coupling
            result[index] = np.vdot(self.coupling, evolved)
        return result.reshape(times.shape)

    def spectral_density(self, omega: ArrayLike) -> FloatArray:
        """Return ``int C(t) exp(i*omega*t) dt`` over the Hermitian extension."""

        frequencies = np.asarray(omega, dtype=float)
        flat = frequencies.reshape(-1)
        identity = np.eye(self.n_modes, dtype=np.complex128)
        result = np.empty(flat.size, dtype=float)
        for index, frequency in enumerate(flat):
            integral = -np.linalg.solve(
                self.generator + 1j * frequency * identity, self.coupling
            )
            result[index] = 2.0 * float(np.real(np.vdot(self.coupling, integral)))
        return result.reshape(frequencies.shape)


@dataclass(frozen=True)
class FitDiagnostics:
    method: str
    n_modes: int
    relative_rms_error: float
    max_absolute_error: float
    min_damping_eigenvalue: float
    stability_abscissa: float
    condition_number: float
    conversion_residual: float = 0.0
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LindbladFit:
    model: LindbladModel
    exponential: ExponentialFit
    diagnostics: FitDiagnostics

    def evaluate(self, t: ArrayLike) -> ComplexArray:
        return self.model.evaluate(t)

    def spectral_density(self, omega: ArrayLike) -> FloatArray:
        return self.model.spectral_density(omega)


@dataclass(frozen=True)
class FermionicLindbladFit:
    """Filled (lesser) and empty (greater) auxiliary-mode sectors."""

    lesser: LindbladFit
    greater: LindbladFit
    input_convention: str = "correlation"

    @property
    def lesser_initial_occupations(self) -> FloatArray:
        return np.ones(self.lesser.model.n_modes, dtype=float)

    @property
    def greater_initial_occupations(self) -> FloatArray:
        return np.zeros(self.greater.model.n_modes, dtype=float)

    @property
    def lesser_jump_kind(self) -> str:
        """The filled lesser sector uses creation (gain) jump operators."""

        return "creation"

    @property
    def greater_jump_kind(self) -> str:
        """The empty greater sector uses annihilation (loss) jump operators."""

        return "annihilation"

    def evaluate_lesser(self, t: ArrayLike) -> ComplexArray:
        values = self.lesser.evaluate(t)
        return 1j * values if self.input_convention == "negf" else values

    def evaluate_greater(self, t: ArrayLike) -> ComplexArray:
        values = self.greater.evaluate(t)
        return -1j * values if self.input_convention == "negf" else values
