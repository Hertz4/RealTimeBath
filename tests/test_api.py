import numpy as np
import pytest
from scipy.integrate import trapezoid

from realtimebath import BandLimitWarning, fit_correlation, fit_fermionic
from realtimebath.route_sdp import cvxpy_available


def test_unsupported_research_helpers_are_not_top_level_exports():
    import realtimebath

    unsupported = {
        "optimize_positive_exponentials",
        "realize_physical",
        "realize_positive_gram",
    }

    assert unsupported.isdisjoint(realtimebath.__all__)
    assert all(not hasattr(realtimebath, name) for name in unsupported)


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_default_method_is_sdp():
    times = np.linspace(0.0, 10.0, 401)
    values = 2.3 * np.exp(-(0.4 + 1.2j) * times)

    result = fit_correlation(
        times,
        values,
        n_modes=1,
        tolerance=1e-9,
    )

    assert result.diagnostics.method == "sdp"
    assert result.diagnostics.relative_rms_error < 1e-4
    assert result.diagnostics.n_modes == 1


def test_eps_band_limit_is_propagated_to_public_diagnostics():
    times = np.linspace(0.0, 6.0, 121)
    values = 0.8 * np.exp(-(0.5 + 0.3j) * times)

    with pytest.warns(BandLimitWarning):
        result = fit_correlation(
            times,
            values,
            eps=1e-30,
            max_modes=4,
            polish=False,
            method="physical",
        )

    assert result.diagnostics.details["selection_mode"] == "eps"
    assert result.diagnostics.details["band_limit_reached"] is True
    assert any(
        "provide finer real-time data" in warning
        for warning in result.diagnostics.warnings
    )


def test_fermionic_correlation_and_negf_conventions_agree():
    times = np.linspace(0.0, 6.0, 241)
    lesser = 0.7 * np.exp(-(0.3 - 0.4j) * times)
    greater = 1.2 * np.exp(-(0.8 + 1.1j) * times)

    correlation_result = fit_fermionic(
        times,
        lesser,
        greater,
        n_modes=1,
        method="physical",
        tolerance=1e-9,
    )
    negf_result = fit_fermionic(
        times,
        1j * lesser,
        -1j * greater,
        convention="negf",
        n_modes=1,
        method="physical",
        tolerance=1e-9,
    )

    assert np.max(np.abs(correlation_result.evaluate_lesser(times) - lesser)) < 1e-9
    assert np.max(np.abs(correlation_result.evaluate_greater(times) - greater)) < 1e-9
    assert np.max(np.abs(negf_result.evaluate_lesser(times) - 1j * lesser)) < 1e-9
    assert np.max(np.abs(negf_result.evaluate_greater(times) + 1j * greater)) < 1e-9
    assert np.all(correlation_result.lesser_initial_occupations == 1.0)
    assert np.all(correlation_result.greater_initial_occupations == 0.0)
    assert correlation_result.lesser_jump_kind == "creation"
    assert correlation_result.greater_jump_kind == "annihilation"


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_auto_falls_back_to_sdp_for_noisy_fit():
    rng = np.random.default_rng(7)
    rates = np.array([0.4 + 1.2j, 1.1 - 0.7j])
    residues = np.array([1.0 + 0.2j, 0.5 - 0.4j])
    weights = np.sum(
        np.outer(residues, residues.conj())
        / (rates[:, None] + rates[None, :].conj()),
        axis=1,
    )
    times = np.linspace(0.0, 10.0, 401)
    clean = np.exp(-np.outer(times, rates)) @ weights
    noise = 1e-5 * (rng.normal(size=times.size) + 1j * rng.normal(size=times.size))
    noise[0] = np.real(noise[0])

    result = fit_correlation(
        times,
        clean + noise,
        n_modes=2,
        method="auto",
        tolerance=1e-4,
        max_nfev=1_000,
    )

    assert result.diagnostics.method == "sdp"
    assert result.diagnostics.relative_rms_error < 1e-4
    assert any("exact physical route" in warning for warning in result.diagnostics.warnings)


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_semicircular_finite_temperature_hybridization():
    omega = np.linspace(-2.0, 2.0, 1201)
    spectral_density = np.sqrt(np.maximum(0.0, 1.0 - (omega / 2.0) ** 2)) / np.pi
    fermi = 1.0 / (1.0 + np.exp(3.0 * omega))
    times = np.linspace(0.0, 8.0, 161)
    phase = np.exp(-1j * np.outer(times, omega))
    lesser = trapezoid(phase * (spectral_density * fermi)[None, :], omega, axis=1)
    greater = trapezoid(
        phase * (spectral_density * (1.0 - fermi))[None, :], omega, axis=1
    )

    result = fit_fermionic(
        times,
        lesser,
        greater,
        n_modes=4,
        method="sdp",
        tolerance=1e-2,
        max_nfev=1_200,
    )

    assert result.lesser.diagnostics.relative_rms_error < 1e-2
    assert result.greater.diagnostics.relative_rms_error < 1e-2
    assert result.lesser.diagnostics.min_damping_eigenvalue > -1e-7
    assert result.greater.diagnostics.min_damping_eigenvalue > -1e-7


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_optional_post_sdp_optimization_is_physical_and_improves_fit():
    from scipy.special import j1

    times = np.linspace(0.0, 10.0, 201)
    semicircle = np.empty(times.size, dtype=np.complex128)
    semicircle[0] = 5.0
    semicircle[1:] = j1(10.0 * times[1:]) / times[1:]
    initial = fit_correlation(
        times, semicircle, n_modes=2, method="sdp", max_nfev=500
    )
    optimized = fit_correlation(
        times,
        semicircle,
        n_modes=2,
        method="sdp",
        optimize=True,
        max_nfev=500,
        optimization_max_nfev=80,
        optimization_max_points=101,
    )

    assert optimized.diagnostics.relative_rms_error < initial.diagnostics.relative_rms_error
    assert optimized.diagnostics.details["optimized"] is True
    assert np.min(np.linalg.eigvalsh(optimized.model.damping)) > -1e-8


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_positive_exponential_method():
    times = np.linspace(0.0, 8.0, 161)
    values = 1.4 * np.exp(-(0.5 + 0.9j) * times)

    result = fit_correlation(
        times,
        values,
        n_modes=1,
        method="positive",
        optimization_max_nfev=30,
    )

    assert result.diagnostics.method == "positive"
    assert result.diagnostics.relative_rms_error < 1e-9
    assert np.min(np.linalg.eigvalsh(result.model.damping)) > -1e-10
