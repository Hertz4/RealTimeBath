"""Fit one correlation and inspect the coupled Lindblad parameters."""

import numpy as np

from realtimebath import fit_correlation


times = np.linspace(0.0, 10.0, 401)
correlation = 2.3 * np.exp(-(0.4 + 1.2j) * times)

fit = fit_correlation(times, correlation, n_modes=1, tolerance=1e-9)

print("Hamiltonian H:\n", fit.model.hamiltonian)
print("Damping D:\n", fit.model.damping)
print("Coupling g:\n", fit.model.coupling)
print("Jump coefficients L:\n", fit.model.jumps)
print("Diagnostics:\n", fit.diagnostics)
