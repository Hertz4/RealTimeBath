import numpy as np

from benchmarks.finite_support import box_delta, half_semicircle_delta


def test_finite_support_transforms_have_unit_equal_time_weight():
    times = np.array([0.0, 0.5, 2.0])

    assert half_semicircle_delta(times)[0] == 1.0
    assert box_delta(times)[0] == 1.0


def test_box_transform_matches_closed_form():
    times = np.linspace(0.0, 10.0, 51)
    expected = np.ones(times.size, dtype=np.complex128)
    expected[1:] = (1.0 - np.exp(-1j * times[1:])) / (1j * times[1:])

    assert np.allclose(box_delta(times), expected, rtol=1e-14, atol=1e-14)


def test_half_semicircle_transform_matches_frequency_quadrature():
    times = np.array([0.2, 1.0, 4.0, 9.0])
    frequencies = np.linspace(0.0, 1.0, 20_001)
    density = 4.0 / np.pi * np.sqrt(np.maximum(0.0, 1.0 - frequencies**2))
    quadrature = np.trapezoid(
        np.exp(-1j * np.outer(times, frequencies)) * density[None, :],
        frequencies,
        axis=1,
    )

    assert np.allclose(
        half_semicircle_delta(times), quadrature, rtol=2e-7, atol=2e-7
    )
