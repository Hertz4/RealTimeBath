"""Compare the two fitted routes for one-sided compact spectral densities."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/realtimebath-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/realtimebath-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1, struve

from realtimebath import fit_correlation
from realtimebath.exceptions import BandLimitError, RealizationError
from realtimebath.route_sdp import cvxpy_available


@dataclass(frozen=True)
class SpectralTarget:
    slug: str
    title: str
    exact_label: str
    transform: Callable[[np.ndarray], np.ndarray]


def half_semicircle_delta(times: np.ndarray) -> np.ndarray:
    r"""Transform of ``4/pi*sqrt(1-w**2)`` supported on ``0 <= w <= 1``."""

    values = np.empty(times.size, dtype=np.complex128)
    zero = times == 0.0
    values[zero] = 1.0
    nonzero_times = times[~zero]
    values[~zero] = 2.0 * (
        j1(nonzero_times) - 1j * struve(1, nonzero_times)
    ) / nonzero_times
    return values


def box_delta(times: np.ndarray) -> np.ndarray:
    r"""Transform of the unit box supported on ``0 <= w <= 1``."""

    return np.asarray(
        np.exp(-0.5j * times) * np.sinc(times / (2.0 * np.pi)),
        dtype=np.complex128,
    )


TARGETS = {
    "half-semicircle": SpectralTarget(
        slug="half_semicircle",
        title=(
            r"Half-semicircle: $J(\omega)=4\sqrt{1-\omega^2}/\pi$, "
            r"$0\leq\omega\leq1$"
        ),
        exact_label="exact half-semicircle",
        transform=half_semicircle_delta,
    ),
    "box": SpectralTarget(
        slug="box",
        title=r"Unit box: $J(\omega)=1$, $0\leq\omega\leq1$",
        exact_label="exact box",
        transform=box_delta,
    ),
}


def relative_l1(
    times: np.ndarray, reference: np.ndarray, fitted: np.ndarray
) -> float:
    numerator = np.trapezoid(np.abs(fitted - reference), times)
    denominator = np.trapezoid(np.abs(reference), times)
    return float(numerator / denominator)


def run_target(
    target: SpectralTarget,
    *,
    max_modes: int,
    display_modes: int,
    residual_modes: tuple[int, int],
    optimization_max_nfev: int,
    output: Path,
) -> None:
    times = np.linspace(0.0, 10.0, 501)
    exact = target.transform(times)
    mode_counts = np.arange(1, max_modes + 1)
    errors_sdp: list[float] = []
    errors_optimized: list[float] = []
    errors_positive: list[float] = []
    timings_sdp: list[float] = []
    timings_optimized: list[float] = []
    timings_positive: list[float] = []
    active_modes: list[int] = []
    displayed: dict[int, dict[str, np.ndarray]] = {}
    displayed_active_modes: dict[int, int] = {}
    retained_modes = {display_modes, *residual_modes}
    last_values: dict[str, np.ndarray] | None = None
    rank_limit_reason: str | None = None

    print(f"\n{target.slug}")
    sweep_started = perf_counter()
    for n_modes in mode_counts:
        if rank_limit_reason is not None:
            assert last_values is not None
            errors_sdp.append(relative_l1(times, exact, last_values["sdp"]))
            errors_optimized.append(
                relative_l1(times, exact, last_values["optimized"])
            )
            errors_positive.append(
                relative_l1(times, exact, last_values["positive"])
            )
            timings_sdp.append(np.nan)
            timings_optimized.append(np.nan)
            timings_positive.append(np.nan)
            active_modes.append(active_modes[-1])
            if n_modes in retained_modes:
                displayed[int(n_modes)] = last_values
                displayed_active_modes[int(n_modes)] = active_modes[-1]
            print(
                f"N={n_modes:2d}  rank-limited plateau with "
                f"{active_modes[-1]} active modes"
            )
            continue

        shared = dict(
            n_modes=int(n_modes),
            tolerance=1e-3,
            max_nfev=800,
            optimization_max_nfev=optimization_max_nfev,
            optimization_max_points=251,
        )
        started = perf_counter()
        try:
            sdp = fit_correlation(times, exact, method="sdp", **shared)
        except RealizationError as error:
            if last_values is None or not isinstance(error, BandLimitError):
                raise
            rank_limit_reason = str(error)
            errors_sdp.append(relative_l1(times, exact, last_values["sdp"]))
            errors_optimized.append(
                relative_l1(times, exact, last_values["optimized"])
            )
            errors_positive.append(
                relative_l1(times, exact, last_values["positive"])
            )
            timings_sdp.append(np.nan)
            timings_optimized.append(np.nan)
            timings_positive.append(np.nan)
            active_modes.append(active_modes[-1])
            if n_modes in retained_modes:
                displayed[int(n_modes)] = last_values
                displayed_active_modes[int(n_modes)] = active_modes[-1]
            print(
                f"N={n_modes:2d}  numerical-rank limit reached; using "
                f"{active_modes[-1]} active modes ({error})"
            )
            continue
        timings_sdp.append(perf_counter() - started)
        started = perf_counter()
        optimized = fit_correlation(
            times, exact, method="sdp", optimize=True, **shared
        )
        timings_optimized.append(perf_counter() - started)
        started = perf_counter()
        positive = fit_correlation(times, exact, method="positive", **shared)
        timings_positive.append(perf_counter() - started)

        values = {
            "sdp": sdp.evaluate(times),
            "optimized": optimized.evaluate(times),
            "positive": positive.evaluate(times),
        }
        last_values = values
        active_modes.append(int(n_modes))
        errors_sdp.append(relative_l1(times, exact, values["sdp"]))
        errors_optimized.append(
            relative_l1(times, exact, values["optimized"])
        )
        errors_positive.append(
            relative_l1(times, exact, values["positive"])
        )
        print(
            f"N={n_modes:2d}  SDP={errors_sdp[-1]:.6e}  "
            f"SDP+opt={errors_optimized[-1]:.6e}  "
            f"positive={errors_positive[-1]:.6e}  |  "
            f"times={timings_sdp[-1]:.3f}/{timings_optimized[-1]:.3f}/"
            f"{timings_positive[-1]:.3f} s"
        )
        if n_modes in retained_modes:
            displayed[int(n_modes)] = values
            displayed_active_modes[int(n_modes)] = int(n_modes)

    identified_modes = max(active_modes)

    np.savetxt(
        output / f"{target.slug}_l1.csv",
        np.column_stack(
            [
                mode_counts,
                active_modes,
                errors_sdp,
                errors_optimized,
                errors_positive,
            ]
        ),
        delimiter=",",
        header=(
            "mode_budget,active_modes,sdp,sdp_optimized,"
            "positive_exponential"
        ),
        comments="",
    )
    np.savetxt(
        output / f"{target.slug}_timing.csv",
        np.column_stack(
            [
                mode_counts,
                active_modes,
                timings_sdp,
                timings_optimized,
                timings_positive,
            ]
        ),
        delimiter=",",
        header=(
            "mode_budget,active_modes,sdp_seconds,sdp_optimized_seconds,"
            "positive_seconds"
        ),
        comments="",
    )

    primary_fit = displayed[display_modes]
    np.savetxt(
        output / f"{target.slug}_fit.csv",
        np.column_stack(
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
        ),
        delimiter=",",
        header=(
            "t,exact_real,exact_imag,sdp_real,sdp_imag,"
            "sdp_optimized_real,sdp_optimized_imag,positive_real,positive_imag"
        ),
        comments="",
    )

    colors = {"optimized": "#1f77b4", "positive": "#e87500"}
    labels = {
        "optimized": "2506.10308: SDP + time optimization",
        "positive": "2604.06466: positive exponential",
    }
    styles = (("optimized", "-"), ("positive", "--"))
    figure = plt.figure(figsize=(14.6, 7.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.15))
    real_axis = figure.add_subplot(grid[0, 0])
    imag_axis = figure.add_subplot(grid[1, 0], sharex=real_axis)
    residual_axes = (
        figure.add_subplot(grid[0, 1], sharex=real_axis),
        figure.add_subplot(grid[1, 1], sharex=real_axis),
    )
    error_axis = figure.add_subplot(grid[:, 2])

    real_axis.plot(
        times,
        np.real(exact),
        color="black",
        lw=2.0,
        label=target.exact_label,
    )
    imag_axis.plot(
        times,
        np.imag(exact),
        color="black",
        lw=2.0,
        label=target.exact_label,
    )
    for key, style in styles:
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

    for residual_mode, residual_axis in zip(residual_modes, residual_axes):
        residual_fit = displayed[residual_mode]
        active_residual_modes = displayed_active_modes[residual_mode]
        for key, style in styles:
            residual_axis.semilogy(
                times,
                np.maximum(np.abs(residual_fit[key] - exact), 1e-15),
                style,
                color=colors[key],
                lw=1.5,
                label=labels[key],
            )
        residual_title = f"Pointwise absolute error, N={residual_mode}"
        if active_residual_modes != residual_mode:
            residual_title += f" ({active_residual_modes} active)"
        residual_axis.set_title(residual_title)
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
    real_axis.set_title(f"Real part, N={display_modes}")
    imag_axis.set_title(f"Imaginary part, N={display_modes}")
    error_axis.set_title("Normalized $L^1$ error")
    real_axis.set_ylabel(r"Re $\Delta(t)$")
    imag_axis.set_ylabel(r"Im $\Delta(t)$")
    error_axis.set_ylabel(
        r"$\int|\Delta_{\mathrm{fit}}-\Delta|dt/\int|\Delta|dt$"
    )
    for axis in (real_axis, imag_axis, *residual_axes):
        axis.set_xlabel("t")
        axis.set_xlim(0.0, 10.0)
        axis.grid(alpha=0.25)
    error_axis.set_xlabel("mode budget N")
    error_axis.set_xticks(mode_counts)
    error_axis.grid(alpha=0.25, which="both")
    if rank_limit_reason is not None:
        error_axis.axvspan(
            identified_modes + 0.5,
            max_modes + 0.5,
            color="#eeeeee",
            zorder=-1,
        )
        error_axis.text(
            identified_modes + 0.7,
            0.02,
            f"rank-limited:\n{identified_modes} active modes",
            transform=error_axis.get_xaxis_transform(),
            fontsize=8,
            color="#555555",
        )
    real_axis.legend(fontsize=8)
    error_axis.legend(fontsize=8)
    figure.suptitle(
        target.title + r", fit window $t\in[0,10]$", fontsize=14
    )
    figure.savefig(output / f"{target.slug}_comparison.png", dpi=220)
    figure.savefig(output / f"{target.slug}_comparison.pdf")
    plt.close(figure)

    timing_figure, timing_axis = plt.subplots(
        figsize=(7.4, 4.6), constrained_layout=True
    )
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
        title=f"{target.slug.replace('_', ' ').title()} fitting time",
        xlabel="mode budget N",
        ylabel="wall-clock time (seconds)",
        xticks=mode_counts,
    )
    timing_axis.grid(alpha=0.25, which="both")
    if rank_limit_reason is not None:
        timing_axis.axvspan(
            identified_modes + 0.5,
            max_modes + 0.5,
            color="#eeeeee",
            zorder=-1,
        )
    timing_axis.legend(fontsize=8)
    timing_figure.savefig(output / f"{target.slug}_timing.png", dpi=220)
    timing_figure.savefig(output / f"{target.slug}_timing.pdf")
    plt.close(timing_figure)
    print(
        f"Total timed {target.slug} sweep: "
        f"{perf_counter() - sweep_started:.3f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(TARGETS),
        default=tuple(TARGETS),
    )
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
    if not cvxpy_available():
        raise RuntimeError(
            "the finite-support comparison requires realtimebath[sdp]"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    for target_name in args.targets:
        run_target(
            TARGETS[target_name],
            max_modes=args.max_modes,
            display_modes=args.display_modes,
            residual_modes=tuple(args.residual_modes),
            optimization_max_nfev=args.optimization_max_nfev,
            output=args.output,
        )


if __name__ == "__main__":
    main()
