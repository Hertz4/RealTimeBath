import numpy as np
import pytest

from realtimebath import (
    BandLimitError,
    BandLimitWarning,
    InvalidSamplesError,
    fit_exponentials,
)


def physical_two_mode_case():
    rates = np.array([0.4 + 1.2j, 1.1 - 0.7j])
    factor_residues = np.array([1.0 + 0.2j, 0.5 - 0.4j])
    weights = np.sum(
        np.outer(factor_residues, factor_residues.conj())
        / (rates[:, None] + rates[None, :].conj()),
        axis=1,
    )
    return rates, weights


def test_fixed_order_recovers_complex_exponentials():
    rates, weights = physical_two_mode_case()
    times = np.linspace(0.0, 10.0, 401)
    values = np.exp(-np.outer(times, rates)) @ weights

    result = fit_exponentials(times, values, n_modes=2, tolerance=1e-10)

    assert result.relative_rms_error < 1e-10
    assert np.max(np.abs(result.evaluate(times) - values)) < 1e-10
    assert np.all(np.real(result.rates) > 0.0)


def test_automatic_order_uses_smallest_accurate_model():
    rates, weights = physical_two_mode_case()
    times = np.linspace(0.0, 8.0, 321)
    values = np.exp(-np.outer(times, rates)) @ weights

    result = fit_exponentials(times, values, max_modes=5, tolerance=1e-8)

    assert result.n_modes == 2
    assert result.relative_rms_error < 1e-8


def test_eps_selects_order_and_is_mutually_exclusive_with_n_modes():
    rates, weights = physical_two_mode_case()
    times = np.linspace(0.0, 8.0, 161)
    values = np.exp(-np.outer(times, rates)) @ weights

    result = fit_exponentials(times, values, eps=1e-8, max_modes=5)

    assert result.n_modes == 2
    assert result.details["selection_mode"] == "eps"
    assert result.details["requested_eps"] == 1e-8
    with pytest.raises(ValueError, match="either eps or n_modes"):
        fit_exponentials(times, values, n_modes=2, eps=1e-8)


def test_fixed_order_above_numerical_rank_reports_band_limit():
    times = np.linspace(0.0, 8.0, 161)
    values = 1.3 * np.exp(-(0.4 + 0.7j) * times)

    with pytest.raises(BandLimitError, match="band limit reached.*finer") as caught:
        fit_exponentials(times, values, n_modes=2)

    assert caught.value.requested_modes == 2
    assert caught.value.supported_modes == 1


def test_unattainable_eps_warns_and_returns_best_band_limited_fit():
    times = np.linspace(0.0, 8.0, 161)
    values = 1.3 * np.exp(-(0.4 + 0.7j) * times)

    with pytest.warns(BandLimitWarning, match="provide finer real-time data"):
        result = fit_exponentials(
            times,
            values,
            eps=1e-30,
            max_modes=5,
            polish=False,
        )

    assert result.n_modes == 1
    assert result.details["band_limit_reached"] is True
    assert result.details["max_supported_modes"] == 1
    assert any("band limit reached" in warning for warning in result.warnings)


def test_rejects_nonuniform_grid():
    times = np.array([0.0, 0.1, 0.2, 0.31, 0.4, 0.5])
    with pytest.raises(InvalidSamplesError, match="uniformly sampled"):
        fit_exponentials(times, np.exp(-times))


def test_rejects_unconverted_negf_samples():
    times = np.linspace(0.0, 1.0, 11)
    values = 1j * np.exp(-times)
    with pytest.raises(InvalidSamplesError, match="real at t=0"):
        fit_exponentials(times, values)
