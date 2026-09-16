import numpy as np
import pytest

from realtimebath import ExponentialFit, RealizationError
from realtimebath.route_physical import realize_physical, realize_positive_gram
from realtimebath.route_sdp import cvxpy_available, realize_sdp


def make_physical_exponential():
    rates = np.array([0.4 + 1.2j, 1.1 - 0.7j])
    residues = np.array([1.0 + 0.2j, 0.5 - 0.4j])
    weights = np.sum(
        np.outer(residues, residues.conj())
        / (rates[:, None] + rates[None, :].conj()),
        axis=1,
    )
    return ExponentialFit(rates, weights, 0.0, 0.0)


@pytest.mark.parametrize(
    ("rates", "weights"),
    [
        (np.array([0.4 + 1.2j]), np.array([2.3 + 0.0j])),
        (make_physical_exponential().rates, make_physical_exponential().weights),
    ],
)
def test_physical_route_round_trip(rates, weights):
    exponential = ExponentialFit(rates, weights, 0.0, 0.0)
    result = realize_physical(exponential)
    times = np.linspace(0.0, 10.0, 201)

    assert np.max(np.abs(result.model.evaluate(times) - exponential.evaluate(times))) < 1e-9
    assert np.min(np.linalg.eigvalsh(result.model.damping)) > -1e-10
    assert np.max(np.real(np.linalg.eigvals(result.model.generator))) < 1e-10
    assert np.allclose(
        result.model.jumps.conj().T @ result.model.jumps,
        2.0 * result.model.damping,
        atol=1e-10,
    )
    frequencies = np.linspace(-5.0, 5.0, 301)
    assert np.min(result.model.spectral_density(frequencies)) > -1e-10


def test_physical_route_rejects_negative_spectrum():
    exponential = ExponentialFit(
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        np.array([1.0 + 0.0j, -0.9 + 0.0j]),
        0.0,
        0.0,
    )
    with pytest.raises(RealizationError):
        realize_physical(exponential)


def test_physical_route_handles_negative_exponential_weight():
    rates = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    factor_residues = np.array([1.0 + 0.0j, -1.0 + 0.0j])
    weights = np.sum(
        np.outer(factor_residues, factor_residues.conj())
        / (rates[:, None] + rates[None, :].conj()),
        axis=1,
    )
    assert weights[1] < 0
    exponential = ExponentialFit(rates, weights, 0.0, 0.0)

    result = realize_physical(exponential)
    times = np.linspace(0.0, 8.0, 161)

    assert np.max(np.abs(result.model.evaluate(times) - exponential.evaluate(times))) < 1e-10


def test_positive_gram_realization():
    rates = np.array([0.5 + 0.8j, 1.2 - 0.4j])
    factor = np.array([[0.7 + 0.1j, 0.2j], [-0.3 + 0.2j, 0.5]])
    gram = factor @ factor.conj().T
    covariance = gram / (rates[:, None] + rates[None, :].conj())
    weights = np.sum(covariance, axis=1)
    expected = ExponentialFit(rates, weights, 0.0, 0.0)

    result = realize_positive_gram(rates, factor)
    times = np.linspace(0.0, 8.0, 161)

    assert np.max(np.abs(result.model.evaluate(times) - expected.evaluate(times))) < 1e-10
    assert np.min(np.linalg.eigvalsh(result.model.damping)) > -1e-9


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_sdp_route_round_trip():
    exponential = make_physical_exponential()
    result = realize_sdp(exponential)
    times = np.linspace(0.0, 8.0, 161)

    relative = np.linalg.norm(result.model.evaluate(times) - exponential.evaluate(times))
    relative /= np.linalg.norm(exponential.evaluate(times))
    assert relative < 2e-6
    assert np.min(np.linalg.eigvalsh(result.model.damping)) > -1e-8
    assert result.conversion_residual < 2e-6


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_open_source_sdp_solvers_round_trip(solver):
    import cvxpy as cp

    if solver not in cp.installed_solvers():
        pytest.skip(f"{solver} is not installed")
    exponential = make_physical_exponential()

    result = realize_sdp(exponential, solver=solver)
    times = np.linspace(0.0, 8.0, 161)
    relative = np.linalg.norm(
        result.model.evaluate(times) - exponential.evaluate(times)
    ) / np.linalg.norm(exponential.evaluate(times))

    assert result.details["solver"] == solver
    assert relative < 2e-6
    assert result.conversion_residual < 2e-6


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_sdp_route_projects_an_unphysical_fit():
    exponential = ExponentialFit(
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        np.array([1.0 + 0.0j, -0.9 + 0.0j]),
        0.0,
        0.0,
    )
    result = realize_sdp(exponential)

    assert np.min(np.linalg.eigvalsh(result.model.damping)) > -1e-7
    assert result.conversion_residual > 1e-5
    assert result.warnings


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_automatic_sdp_solver_falls_back_after_failure(monkeypatch):
    import cvxpy as cp

    original_solve = cp.Problem.solve
    attempted = []

    def fail_mosek_then_solve(self, *args, solver=None, **kwargs):
        attempted.append(solver)
        if solver == "MOSEK":
            raise cp.error.SolverError("simulated missing MOSEK license")
        return original_solve(self, *args, solver=solver, **kwargs)

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["MOSEK", "CLARABEL"])
    monkeypatch.setattr(cp.Problem, "solve", fail_mosek_then_solve)

    result = realize_sdp(make_physical_exponential())

    assert attempted[:2] == ["MOSEK", "CLARABEL"]
    assert result.details["solver"] == "CLARABEL"
    assert result.details["solver_failed_attempts"]
    assert any("fallback" in warning for warning in result.warnings)


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_explicit_sdp_solver_failure_does_not_fallback(monkeypatch):
    import cvxpy as cp

    attempted = []

    def fail_solver(self, *args, solver=None, **kwargs):
        attempted.append(solver)
        raise cp.error.SolverError("simulated explicit solver failure")

    monkeypatch.setattr(cp.Problem, "solve", fail_solver)

    with pytest.raises(RealizationError, match="simulated explicit solver failure"):
        realize_sdp(make_physical_exponential(), solver="MOSEK")

    assert attempted == ["MOSEK"]


@pytest.mark.skipif(not cvxpy_available(), reason="CVXPY is not installed")
def test_mosek_sdp_accuracy_when_available():
    import cvxpy as cp

    if "MOSEK" not in cp.installed_solvers():
        pytest.skip("MOSEK is not installed")
    exponential = ExponentialFit(
        np.array([0.4 + 1.2j]),
        np.array([2.3 + 0.0j]),
        0.0,
        0.0,
    )

    result = realize_sdp(exponential, solver="MOSEK")
    times = np.linspace(0.0, 10.0, 401)
    relative = np.linalg.norm(
        result.model.evaluate(times) - exponential.evaluate(times)
    ) / np.linalg.norm(exponential.evaluate(times))

    assert relative < 1e-7
    assert not result.warnings
