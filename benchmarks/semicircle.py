"""Reproduce the semicircular-hybridization comparison on t in [0, 10]."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/realtimebath-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/realtimebath-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1

from realtimebath import fit_correlation
from realtimebath.route_sdp import cvxpy_available


def semicircular_delta(
    times: np.ndarray, *, bandwidth: float = 10.0, coupling: float = 1.0
) -> np.ndarray:
    r"""Return the transform of J(w)=coupling/pi*sqrt(1-(w/W)^2)."""

    values = np.empty(times.size, dtype=np.complex128)
    values[0] = coupling * bandwidth / 2.0
    values[1:] = coupling * j1(bandwidth * times[1:]) / times[1:]
    return values


def relative_l1(times: np.ndarray, reference: np.ndarray, fitted: np.ndarray) -> float:
    numerator = np.trapezoid(np.abs(fitted - reference), times)
    denominator = np.trapezoid(np.abs(reference), times)
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-modes", type=int, default=15)
    parser.add_argument("--display-modes", type=int, default=6)
    parser.add_argument("--residual-modes", type=int, nargs=2, default=(6, 12))
    parser.add_argument("--optimization-max-nfev", type=int, default=1_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not 1 <= args.display_modes <= args.max_modes:
        parser.error("display-modes must be between 1 and max-modes")
    if any(not 1 <= mode <= args.max_modes for mode in args.residual_modes):
        parser.error("residual-modes must be between 1 and max-modes")

    args.output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, 10.0, 501)
    exact = semicircular_delta(times)
    mode_counts = np.arange(1, args.max_modes + 1)
    errors_sdp: list[float] = []
    errors_optimized: list[float] = []
    errors_positive: list[float] = []
    timings_sdp: list[float] = []
    timings_optimized: list[float] = []
    timings_positive: list[float] = []
    displayed: dict[int, dict[str, np.ndarray]] = {}
    retained_modes = {args.display_modes, *args.residual_modes}

    # Load CVXPY before timing so its one-time import cost does not skew N=1.
    if not cvxpy_available():
        raise RuntimeError("the semicircle comparison requires realtimebath[sdp]")
    sweep_started = perf_counter()
    for n_modes in mode_counts:
        shared = dict(
            n_modes=int(n_modes),
            tolerance=1e-3,
            max_nfev=800,
            optimization_max_nfev=args.optimization_max_nfev,
            optimization_max_points=251,
        )
        started = perf_counter()
        sdp = fit_correlation(times, exact, method="sdp", **shared)
        timings_sdp.append(perf_counter() - started)
        started = perf_counter()
        optimized = fit_correlation(
            times, exact, method="sdp", optimize=True, **shared
        )
        timings_optimized.append(perf_counter() - started)
        started = perf_counter()
        positive = fit_correlation(times, exact, method="positive", **shared)
        timings_positive.append(perf_counter() - started)
        sdp_values = sdp.evaluate(times)
        optimized_values = optimized.evaluate(times)
        positive_values = positive.evaluate(times)
        errors_sdp.append(relative_l1(times, exact, sdp_values))
        errors_optimized.append(relative_l1(times, exact, optimized_values))
        errors_positive.append(relative_l1(times, exact, positive_values))
        print(
            f"N={n_modes:2d}  SDP={errors_sdp[-1]:.6e}  "
            f"SDP+opt={errors_optimized[-1]:.6e}  "
            f"positive={errors_positive[-1]:.6e}  |  "
            f"times={timings_sdp[-1]:.3f}/{timings_optimized[-1]:.3f}/"
            f"{timings_positive[-1]:.3f} s"
        )
        if n_modes in retained_modes:
            displayed[int(n_modes)] = {
                "sdp": sdp_values,
                "optimized": optimized_values,
                "positive": positive_values,
            }

    error_table = np.column_stack(
        [mode_counts, errors_sdp, errors_optimized, errors_positive]
    )
    np.savetxt(
        args.output / "semicircle_l1.csv",
        error_table,
        delimiter=",",
        header="n_modes,sdp,sdp_optimized,positive_exponential",
        comments="",
    )
    timing_table = np.column_stack(
        [mode_counts, timings_sdp, timings_optimized, timings_positive]
    )
    np.savetxt(
        args.output / "semicircle_timing.csv",
        timing_table,
        delimiter=",",
        header="n_modes,sdp_seconds,sdp_optimized_seconds,positive_seconds",
        comments="",
    )
    primary_fit = displayed[args.display_modes]
    fit_table = np.column_stack(
        [
            times,
            np.real(exact),
            np.imag(exact),
            np.real(primary_fit["sdp"]),
            np.imag(primary_fit["sdp"]),
            np.real(primary_fit["optimized"]),
            np.imag(primary_fit["optimized"]),
            np.real(primary_fit["positive"]),
            np.imag(primary_fit["positive"]),
        ]
    )
    np.savetxt(
        args.output / "semicircle_fit.csv",
        fit_table,
        delimiter=",",
        header=(
            "t,exact_real,exact_imag,sdp_real,sdp_imag,"
            "sdp_optimized_real,sdp_optimized_imag,positive_real,positive_imag"
        ),
        comments="",
    )

    colors = {"optimized": "#1f77b4", "positive": "#e87500", "sdp": "#7f7f7f"}
    labels = {
        "optimized": "2506.10308: SDP + time optimization",
        "positive": "2604.06466: positive exponential",
        "sdp": "SDP before optimization",
    }
    figure = plt.figure(figsize=(14.6, 7.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.15))
    real_axis = figure.add_subplot(grid[0, 0])
    imag_axis = figure.add_subplot(grid[1, 0], sharex=real_axis)
    residual_axes = (
        figure.add_subplot(grid[0, 1], sharex=real_axis),
        figure.add_subplot(grid[1, 1], sharex=real_axis),
    )
    error_axis = figure.add_subplot(grid[:, 2])

    real_axis.plot(times, np.real(exact), color="black", lw=2.0, label="exact semicircle")
    imag_axis.axhline(0.0, color="black", lw=2.0, label="exact semicircle")
    for key, style in (("optimized", "-"), ("positive", "--")):
        real_axis.plot(
            times,
            np.real(primary_fit[key]),
            style,
            color=colors[key],
            lw=1.5,
            label=labels[key],
        )
        imag_axis.plot(
            times,
            np.imag(primary_fit[key]),
            style,
            color=colors[key],
            lw=1.5,
            label=labels[key],
        )

    for residual_mode, residual_axis in zip(args.residual_modes, residual_axes):
        residual_fit = displayed[residual_mode]
        for key, style in (("optimized", "-"), ("positive", "--")):
            residual_axis.semilogy(
                times,
                np.maximum(np.abs(residual_fit[key] - exact), 1e-15),
                style,
                color=colors[key],
                lw=1.5,
                label=labels[key],
            )
        residual_axis.set_title(f"Pointwise absolute error, N={residual_mode}")
        residual_axis.set_ylabel(r"$|\Delta_{\mathrm{fit}}(t)-\Delta(t)|$")

    error_axis.semilogy(
        mode_counts,
        errors_optimized,
        "-o",
        color=colors["optimized"],
        label=labels["optimized"],
    )
    error_axis.semilogy(
        mode_counts,
        errors_positive,
        "--s",
        color=colors["positive"],
        label=labels["positive"],
    )
    real_axis.set_title(f"Real part, N={args.display_modes}")
    imag_axis.set_title(f"Imaginary part, N={args.display_modes}")
    error_axis.set_title("Normalized $L^1$ error")
    real_axis.set_ylabel(r"Re $\Delta(t)$")
    imag_axis.set_ylabel(r"Im $\Delta(t)$")
    error_axis.set_ylabel(r"$\int|\Delta_{\mathrm{fit}}-\Delta|dt/\int|\Delta|dt$")
    for axis in (real_axis, imag_axis, *residual_axes):
        axis.set_xlabel("t")
        axis.set_xlim(0.0, 10.0)
        axis.grid(alpha=0.25)
    error_axis.set_xlabel("number of modes N")
    error_axis.set_xticks(mode_counts)
    error_axis.grid(alpha=0.25, which="both")
    real_axis.legend(fontsize=8)
    error_axis.legend(fontsize=8)
    figure.suptitle(
        r"Semicircular hybridization: $W=10$, $\Gamma=1$, fit window $t\in[0,10]$",
        fontsize=14,
    )
    figure.savefig(args.output / "semicircle_comparison.png", dpi=220)
    figure.savefig(args.output / "semicircle_comparison.pdf")

    timing_figure, timing_axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    timing_axis.semilogy(
        mode_counts,
        timings_optimized,
        "-o",
        color=colors["optimized"],
        label=labels["optimized"],
    )
    timing_axis.semilogy(
        mode_counts,
        timings_positive,
        "--s",
        color=colors["positive"],
        label=labels["positive"],
    )
    timing_axis.set(
        title="Semicircular hybridization fitting time",
        xlabel="number of modes N",
        ylabel="wall-clock time (seconds)",
        xticks=mode_counts,
    )
    timing_axis.grid(alpha=0.25, which="both")
    timing_axis.legend(fontsize=8)
    timing_figure.savefig(args.output / "semicircle_timing.png", dpi=220)
    timing_figure.savefig(args.output / "semicircle_timing.pdf")
    print(f"Total timed sweep: {perf_counter() - sweep_started:.3f} s")


if __name__ == "__main__":
    main()
