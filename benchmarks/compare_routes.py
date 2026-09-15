"""Small deterministic comparison of the two realization routes."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from realtimebath import ExponentialFit
from realtimebath.route_physical import realize_physical
from realtimebath.route_sdp import cvxpy_available, realize_sdp


def main() -> None:
    rates = np.array([0.2 + 0.7j, 0.6 - 1.4j, 1.3 + 2.1j])
    factors = np.array([1.0 + 0.1j, -0.3 + 0.8j, 0.4 - 0.2j])
    weights = np.sum(
        np.outer(factors, factors.conj())
        / (rates[:, None] + rates[None, :].conj()),
        axis=1,
    )
    exponential = ExponentialFit(rates, weights, 0.0, 0.0)
    times = np.linspace(0.0, 20.0, 1001)
    reference = exponential.evaluate(times)

    methods = [("physical", realize_physical)]
    if cvxpy_available():
        methods.append(("sdp", realize_sdp))
    print("method     seconds    relative-error    condition")
    for name, method in methods:
        started = perf_counter()
        result = method(exponential)
        elapsed = perf_counter() - started
        error = np.linalg.norm(
            result.model.evaluate(times) - reference
        ) / np.linalg.norm(reference)
        print(f"{name:9s} {elapsed:9.4f} {error:17.6e} {result.condition_number:12.4e}")


if __name__ == "__main__":
    main()
