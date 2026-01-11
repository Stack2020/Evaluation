from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

from calculate_score_field import calculate_score_field
from config import config as base_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute score field and draw the panel figure.")
    compute = parser.add_argument_group("score-field computation")
    compute.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional precomputed npz file. If provided, computation is skipped.",
    )
    compute.add_argument(
        "--save-data",
        type=Path,
        default=Path("score_field_times.npz"),
        help="Destination to store the computed npz payload (default: score_field_times.npz).",
    )
    compute.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(base_config["config_path"]),
        help="Path to the trained checkpoint.",
    )
    compute.add_argument(
        "--resolution",
        type=int,
        default=220,
        help="Grid resolution along each axis.",
    )
    compute.add_argument(
        "--span-sigma",
        type=float,
        default=4.0,
        help="Spatial span around the mean in multiples of std-dev.",
    )
    compute.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[round(0.01 + 0.01 * i, 3) for i in range(int((1.0 - 0.01) / 0.01) + 1)],
        help="Time instants to evaluate when computing score fields (default: 0.01, 0.02, ..., 1.00).",
    )
    compute.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for model evaluation.",
    )

    viz = parser.add_argument_group("visualization")
    viz.add_argument(
        "--figure",
        type=Path,
        default=Path("score_field_panels.png"),
        help="Output image path for the multi-panel layout.",
    )
    viz.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Colormap for the residual heatmap.",
    )
    viz.add_argument(
        "--panel-times",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        help="Time instants rendered in the multi-panel grid (default: 0.1, 0.3, 0.5).",
    )
    viz.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    viz.add_argument(
        "--lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile used to set the global color scale.",
    )
    viz.add_argument(
        "--upper-quantile",
        type=float,
        default=0.995,
        help="Upper quantile used to set the global color scale.",
    )
    viz.add_argument(
        "--norm",
        type=str,
        choices=("linear", "log", "power"),
        default="power",
        help="Normalization strategy for the colormap.",
    )
    viz.add_argument(
        "--log-eps",
        type=float,
        default=1e-5,
        help="Floor applied before log scaling.",
    )
    viz.add_argument(
        "--norm-gamma",
        type=float,
        default=0.6,
        help="Gamma exponent used when norm='power'.",
    )
    viz.add_argument(
        "--panel-contour-quantiles",
        type=float,
        nargs="+",
        default=[0.85, 0.95, 0.99],
        help="Quantiles used for contour overlays in the multi-panel view.",
    )
    viz.add_argument(
        "--contour-alpha",
        type=float,
        default=0.55,
        help="Alpha used for contour overlays.",
    )
    viz.add_argument(
        "--plot-range",
        type=float,
        nargs=2,
        default=None,
        help="Optional fixed axis limits [min max] for x/y; if omitted, use computed grid extents.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"npz file '{path}' does not exist.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def obtain_field_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.data is not None:
        if args.data.exists():
            print(f"Loading precomputed fields from {args.data}")
            return load_npz(args.data)
        print(f"Requested data path {args.data} not found. Will compute instead.")
    if args.data is None and args.save_data is not None and args.save_data.exists():
        print(f"Loading cached fields from {args.save_data}")
        return load_npz(args.save_data)

    print("Computing score field and rotational residuals...")
    return calculate_score_field(
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        span_sigma=args.span_sigma,
        times=args.times,
        batch_size=args.batch_size,
        output_path=args.save_data,
        time_varying_grid=True,
        fixed_range=(-3.0, 3.0),
    )


def select_time_index(times: np.ndarray, requested: float) -> tuple[int, float]:
    if times.ndim != 1:
        raise ValueError("times array should be 1-D.")
    idx = int(np.argmin(np.abs(times - requested)))
    actual = float(times[idx])
    return idx, actual


def select_time_indices(times: np.ndarray, requested_list: list[float]) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    actual_times: list[float] = []
    for requested in requested_list:
        idx, actual = select_time_index(times, requested)
        indices.append(idx)
        actual_times.append(actual)
    return indices, actual_times


def compute_spatial_mean(residual: np.ndarray) -> np.ndarray:
    return np.mean(residual, axis=(1, 2))


def grid_at_time(grid: np.ndarray, time_idx: int) -> np.ndarray:
    if grid.ndim == 2:
        return grid
    if grid.ndim == 3:
        if time_idx < 0 or time_idx >= grid.shape[0]:
            raise IndexError(f"time index {time_idx} outside grid range 0..{grid.shape[0]-1}")
        return grid[time_idx]
    raise ValueError(f"Unsupported grid dimension {grid.ndim}; expected 2 or 3.")


def compute_value_bounds(
    residual: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    finite = np.asarray(residual, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.clip(lower_quantile, 0.0, 1.0))
    hi = float(np.clip(upper_quantile, 0.0, 1.0))
    if lo > hi:
        lo, hi = hi, lo
    vmin = float(np.quantile(finite, lo)) if 0.0 < lo < 1.0 else float(np.min(finite))
    vmax = float(np.quantile(finite, hi)) if 0.0 < hi < 1.0 else float(np.max(finite))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise ValueError("Failed to compute finite bounds for normalization.")
    if math.isclose(vmin, vmax, rel_tol=1e-9, abs_tol=1e-12):
        if vmax == 0.0:
            vmax = 1.0
        else:
            span = abs(vmax) * 0.05
            vmin -= span
            vmax += span
    return vmin, vmax


def build_norm(norm_type: str, vmin: float, vmax: float, *, log_eps: float, gamma: float):
    from matplotlib.colors import LogNorm, Normalize, PowerNorm

    norm_type = norm_type.lower()
    if norm_type == "log":
        safe_min = max(vmin, float(log_eps))
        safe_max = max(vmax, safe_min * 1.01)
        return LogNorm(vmin=safe_min, vmax=safe_max)
    if norm_type == "power":
        safe_gamma = max(1e-3, float(gamma))
        return PowerNorm(gamma=safe_gamma, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def derive_contour_levels(data: np.ndarray, quantiles: list[float] | tuple[float, ...]) -> np.ndarray | None:
    if not quantiles:
        return None
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return None
    q = np.asarray(quantiles, dtype=np.float64)
    q = q[(0.0 <= q) & (q <= 1.0)]
    if q.size == 0:
        return None
    levels = np.quantile(finite, q)
    levels = np.unique(np.round(levels, decimals=8))
    levels = levels[levels > 0.0]
    if levels.size == 0:
        return None
    return levels


def add_contours(ax, grid_x, grid_y, data, quantiles, alpha: float) -> None:
    levels = derive_contour_levels(data, quantiles)
    if levels is None:
        return
    ax.contour(
        grid_x,
        grid_y,
        data,
        levels=levels,
        colors="white",
        linewidths=0.6,
        alpha=max(0.0, float(alpha)),
    )


def render_mean_series_axis(ax, times: np.ndarray, spatial_mean: np.ndarray) -> None:
    accent = "#000000"
    ax.set_facecolor("#ffffff")
    ax.plot(times, spatial_mean, color=accent, linewidth=2.2, label=r"mean $\mathcal{R}_{\mathrm{rot}}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"mean $\mathcal{R}_{\mathrm{rot}}$")
    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{rot}}$")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=9)


def make_panel_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    residual: np.ndarray,
    *,
    panel_indices: list[int],
    panel_times: list[float],
    cmap: str,
    norm,
    contour_quantiles: list[float],
    contour_alpha: float,
    times: np.ndarray,
    spatial_mean: np.ndarray,
    plot_range: tuple[float, float] | None,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    total_panels = len(panel_indices)
    if total_panels == 0:
        raise ValueError("At least one panel time is required.")

    cols = total_panels + 1  # reserve last column for the time series
    rows = 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.3 * cols, 3.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    heat_axes = axes_flat[:total_panels]
    series_ax = axes_flat[-1]

    last_mesh = None
    for idx, ax in enumerate(heat_axes):
        panel_idx = panel_indices[idx]
        slice_data = residual[panel_idx]
        gx = grid_at_time(grid_x, panel_idx)
        gy = grid_at_time(grid_y, panel_idx)
        mesh = ax.pcolormesh(
            gx,
            gy,
            slice_data,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        last_mesh = mesh
        add_contours(ax, gx, gy, slice_data, contour_quantiles, contour_alpha)
        ax.set_title(rf"$t={panel_times[idx]:.2f}$", fontsize=11)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        if plot_range is not None:
            ax.set_xlim(plot_range)
            ax.set_ylim(plot_range)

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=heat_axes, shrink=0.9, pad=0.02)
        cbar.set_label(r"$\mathcal{R}_{\mathrm{rot}}$")

    render_mean_series_axis(series_ax, times=times, spatial_mean=spatial_mean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = obtain_field_data(args)

    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    times = data["times"]
    rot_residual = data["rot_residual"]

    if grid_x.ndim not in (2, 3) or grid_y.ndim not in (2, 3):
        raise ValueError(f"Unexpected grid dimensions: grid_x {grid_x.shape}, grid_y {grid_y.shape}")
    if grid_x.ndim != grid_y.ndim:
        raise ValueError(f"Mismatched grid dimensions: grid_x {grid_x.shape}, grid_y {grid_y.shape}")
    if grid_x.ndim == 3 and grid_x.shape[0] != times.shape[0]:
        raise ValueError(f"grid_x time dimension {grid_x.shape[0]} does not match times {times.shape[0]}")
    if grid_y.ndim == 3 and grid_y.shape[0] != times.shape[0]:
        raise ValueError(f"grid_y time dimension {grid_y.shape[0]} does not match times {times.shape[0]}")

    panel_times = list(args.panel_times) if args.panel_times is not None else [0.1, 0.3, 0.5]
    panel_indices, actual_panel_times = select_time_indices(times, panel_times)
    spatial_mean = compute_spatial_mean(rot_residual)

    vmin, vmax = compute_value_bounds(
        rot_residual,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    norm = build_norm(
        args.norm,
        vmin,
        vmax,
        log_eps=args.log_eps,
        gamma=args.norm_gamma,
    )

    make_panel_plot(
        grid_x,
        grid_y,
        rot_residual,
        panel_indices=panel_indices,
        panel_times=actual_panel_times,
        cmap=args.cmap,
        norm=norm,
        contour_quantiles=args.panel_contour_quantiles,
        contour_alpha=args.contour_alpha,
        times=times,
        spatial_mean=spatial_mean,
        plot_range=None if args.plot_range is None else tuple(args.plot_range),
        output_path=args.figure,
        dpi=args.dpi,
    )
    summary_times = ", ".join(f"{t:.3f}" for t in actual_panel_times)
    print(f"Saved rotational residual panels at [{summary_times}] to {args.figure}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
