import numpy as np
import pytest

from realtimebath.rank_one_fit import (
    _effective_max_nfev,
    _evaluate_rank_one,
    _model_to_rank_one,
    _pack_parameters,
    _prediction_jacobian,
    optimize_rank_one_exponentials,
)
from realtimebath.route_physical import realize_positive_gram


def test_evaluation_budget_honors_user_request():
    assert _effective_max_nfev(1_000, 3) == 1_000
    assert _effective_max_nfev(1_000, 23) == 1_000
    assert _effective_max_nfev(2_000, 23) == 2_000
    assert _effective_max_nfev(80, 27) == 80


def test_rank_one_analytic_jacobian_matches_central_difference():
    rates = np.array([0.5 + 0.7j, 1.1 - 0.2j])
    residues = np.array([0.8 + 0.3j, -0.4 + 0.6j])
    parameters, anchor = _pack_parameters(rates, residues)
    times = np.linspace(0.0, 3.0, 13)
    _, analytic = _prediction_jacobian(parameters, times, 2, anchor)
    numerical = np.empty_like(analytic)
    step = 1e-6
    for column in range(parameters.size):
        direction = np.zeros(parameters.size)
        direction[column] = step
        plus = _prediction_jacobian(parameters + direction, times, 2, anchor)[0]
        minus = _prediction_jacobian(parameters - direction, times, 2, anchor)[0]
        numerical[:, column] = (plus - minus) / (2.0 * step)

    assert np.allclose(analytic, numerical, rtol=2e-5, atol=2e-7)


def test_physical_model_converts_to_rank_one_coordinates():
    rates = np.array([0.4 + 1.2j, 1.1 - 0.7j, 0.8 + 0.1j])
    residues = np.array([1.0 + 0.2j, 0.5 - 0.4j, -0.2 + 0.3j])
    model = realize_positive_gram(rates, residues[:, None]).model
    converted_rates, converted_residues = _model_to_rank_one(model)
    times = np.linspace(0.0, 8.0, 101)
    converted, _ = _evaluate_rank_one(converted_rates, converted_residues, times)

    assert np.max(np.abs(converted - model.evaluate(times))) < 1e-8


@pytest.mark.parametrize(
    ("n_modes", "expected_solver"), [(3, "exact"), (5, "lsmr")]
)
def test_rank_one_optimizer_uses_size_appropriate_solver(
    n_modes, expected_solver
):
    rates = np.linspace(0.4, 1.2, n_modes) + 1j * np.linspace(
        -1.5, 1.5, n_modes
    )
    residues = np.linspace(0.5, 1.0, n_modes) + 0.2j
    initial = realize_positive_gram(rates, residues[:, None])
    times = np.linspace(0.0, 5.0, 81)
    target = initial.model.evaluate(times)

    result, _ = optimize_rank_one_exponentials(
        times, target, initial, max_nfev=10, max_points=81
    )

    assert result.details["rank_one_tr_solver"] == expected_solver
    assert np.max(np.abs(result.model.evaluate(times) - target)) < 1e-8
