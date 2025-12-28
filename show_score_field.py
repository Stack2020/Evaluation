from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import math

import numpy as np

from calculate_score_field import calculate_score_field

from config_regularization import config as base_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute score field and visualize rotational residuals.")
    compute_group = parser.add_argument_group("score-field computation")
    compute_group.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional precomputed npz file. If provided, computation is skipped.",
    )
    compute_group.add_argument(
        "--save-data",
        type=Path,
        default=Path("score_field_times.npz"),
        help="Destination to store the computed npz payload (default: score_field_times.npz).",
    )
    compute_group.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(base_config["config_path"]),
        help="Path to the trained checkpoint.",
    )
    compute_group.add_argument(
        "--resolution",
        type=int,
        default=220,
        help="Grid resolution along each axis.",
    )
    compute_group.add_argument(
        "--span-sigma",
        type=float,
        default=4.0,
        help="Spatial span around the mean in multiples of std-dev.",
    )
    compute_group.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[round(0.01 + 0.01 * i, 3) for i in range(int((1.0 - 0.01) / 0.01) + 1)],
        help="Time instants to evaluate when computing score fields (default: 0.01, 0.02, ..., 1.00).",
    )
    compute_group.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for model evaluation.",
    )


    viz_group = parser.add_argument_group("visualization")
    viz_group.add_argument(
        "--visualize-time",
        type=float,
        default=0.5,
        help="Reference time highlighted in auxiliary plots.",
    )
    viz_group.add_argument(
        "--figure",
        type=Path,
        default=Path("score_field_panels.png"),
        help="Output image path for the multi-panel layout.",
    )
    viz_group.add_argument(
        "--highlight-figure",
        type=Path,
        default=None,
        help="Output path for the highlight heatmap + stats composite. If omitted, skip generating.",
    )
    viz_group.add_argument(
        "--stats-figure",
        type=Path,
        default=Path("score_field_mean_series.png"),
        help="Output path for the time-series plot of spatial mean R_rot.",
    )
    viz_group.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Colormap for the residual heatmap.",
    )
    viz_group.add_argument(
        "--panel-times",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        help="Time instants rendered in the multi-panel grid (default: 0.1, 0.3, 0.5).",
    )
    viz_group.add_argument(
        "--panel-cols",
        type=int,
        default=4,
        help="Total columns in the combined layout (heatmaps + time-series).",
    )
    viz_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    viz_group.add_argument(
        "--lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile used to set the global color scale.",
    )
    viz_group.add_argument(
        "--upper-quantile",
        type=float,
        default=0.995,
        help="Upper quantile used to set the global color scale.",
    )
    viz_group.add_argument(
        "--norm",
        type=str,
        choices=("linear", "log", "power"),
        default="power",
        help="Normalization strategy for the colormap.",
    )
    viz_group.add_argument(
        "--log-eps",
        type=float,
        default=1e-5,
        help="Floor applied before log scaling.",
    )
    viz_group.add_argument(
        "--norm-gamma",
        type=float,
        default=0.6,
        help="Gamma exponent used when norm='power'.",
    )
    viz_group.add_argument(
        "--panel-contour-quantiles",
        type=float,
        nargs="+",
        default=[0.85, 0.95, 0.99],
        help="Quantiles used for contour overlays in the multi-panel view.",
    )
    viz_group.add_argument(
        "--highlight-contour-quantiles",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 0.97],
        help="Quantiles used for contour overlays in the highlight figure.",
    )
    viz_group.add_argument(
        "--contour-alpha",
        type=float,
        default=0.55,
        help="Alpha used for contour overlays.",
    )
    viz_group.add_argument(
        "--quiver-density",
        type=int,
        default=18,
        help="Number of arrows per axis in the quiver overlay. Set <=0 to disable.",
    )
    viz_group.add_argument(
        "--quiver-scale",
        type=float,
        default=35.0,
        help="Scale factor for the quiver overlay.",
    )
    viz_group.add_argument(
        "--quiver-color",
        type=str,
        default="#1d1d1d",
        help="Color used for the quiver overlay vectors.",
    )
    viz_group.add_argument(
        "--highlight-width",
        type=float,
        default=0.015,
        help="Half-width of the shaded band around the highlighted time in stats plots.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"npz file '{path}' does not exist.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def obtain_field_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    # Prefer user-provided data path; otherwise, if a default save path exists, load it.
    if args.data is not None:
        if args.data.exists():
            print(f"Loading precomputed fields from {args.data}")
            return load_npz(args.data)
        else:
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


def compute_time_series(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial_mean = np.mean(residual, axis=(1, 2))
    mean_abs = np.mean(np.abs(residual), axis=(1, 2))
    max_abs = np.max(np.abs(residual), axis=(1, 2))
    return spatial_mean, mean_abs, max_abs


def grid_at_time(grid: np.ndarray, time_idx: int) -> np.ndarray:
    """
    Return the 2D grid slice for a specific time index.
    Supports legacy static grids (ndim=2) and time-varying grids (ndim=3).
    """
    if grid.ndim == 2:
        return grid
    if grid.ndim == 3:
        if time_idx < 0 or time_idx >= grid.shape[0]:
            raise IndexError(f"time index {time_idx} outside grid range 0..{grid.shape[0]-1}")
        return grid[time_idx]
    raise ValueError(f"Unsupported grid dimension {grid.ndim}; expected 2 or 3.")


def calculate_spatial_time_score(rot_residual: np.ndarray) -> float:
    """
    Compute the space-time aggregated rotational residual (Eq. 29):
    sqrt( mean_t mean_x R_rot(x, t)^2 ).
    """
    if rot_residual.ndim < 3:
        raise ValueError("rot_residual must have shape (T, H, W) or (T, H, W, ...).")
    spatial_mean_sq = np.mean(np.square(rot_residual), axis=(1, 2))
    return float(np.sqrt(np.mean(spatial_mean_sq)))


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


def downsample_for_quiver(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score_slice: np.ndarray | None,
    density: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if score_slice is None or density <= 0:
        return None
    if score_slice.ndim != 3 or score_slice.shape[2] != 2:
        return None
    stride_x = max(1, grid_x.shape[1] // max(1, density))
    stride_y = max(1, grid_x.shape[0] // max(1, density))
    xs = grid_x[::stride_y, ::stride_x]
    ys = grid_y[::stride_y, ::stride_x]
    u = score_slice[::stride_y, ::stride_x, 0]
    v = score_slice[::stride_y, ::stride_x, 1]
    return xs, ys, u, v


def render_stats_axis(
    ax,
    times: np.ndarray,
    spatial_mean: np.ndarray,
    mean_abs: np.ndarray,
    max_abs: np.ndarray,
    highlight_idx: int,
    highlight_time: float,
    highlight_width: float,
) -> None:
    highlight_width = max(0.0, float(highlight_width))
    if highlight_width > 0.0:
        ax.axvspan(
            highlight_time - highlight_width,
            highlight_time + highlight_width,
            color="#dddddd",
            alpha=0.4,
            linewidth=0.0,
        )
    ax.axhline(0.0, color="#bdbdbd", linestyle="--", linewidth=0.7)
    ax.plot(times, spatial_mean, label=r"mean $\mathcal{R}_{\mathrm{rot}}$", color="#0a3a5c", linewidth=2.2)
    ax.plot(times, mean_abs, label=r"mean $|\mathcal{R}_{\mathrm{rot}}|$", color="#1f77b4", linewidth=1.6, alpha=0.8)
    ax.plot(times, max_abs, label=r"max $|\mathcal{R}_{\mathrm{rot}}|$", color="#d62728", linestyle="--", linewidth=1.3)
    ax.axvline(highlight_time, color="gray", linestyle=":", linewidth=1.1)
    ax.scatter([highlight_time], [spatial_mean[highlight_idx]], color="#0a3a5c", s=32, zorder=3)
    ax.scatter([highlight_time], [mean_abs[highlight_idx]], color="#1f77b4", s=26, zorder=3)
    ax.scatter([highlight_time], [max_abs[highlight_idx]], color="#d62728", s=26, zorder=3)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{R}_{\mathrm{rot}}$")
    ax.set_title("Residual statistics over time")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    stats_text = (
        f"mean = {float(spatial_mean[highlight_idx]):+.2e}\n"
        f"mean |.| = {float(mean_abs[highlight_idx]):.2e}\n"
        f"max |.| = {float(max_abs[highlight_idx]):.2e}"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.3"),
    )


def render_mean_series_axis(
    ax,
    times: np.ndarray,
    spatial_mean: np.ndarray,
    highlight_idx: int,
    highlight_time: float,
    highlight_width: float,
) -> None:
    accent = "#0a3a5c"

    ax.set_facecolor("#f7f8fa")
    ax.plot(times, spatial_mean, color=accent, linewidth=2.2, label=r"mean $\mathcal{R}_{\mathrm{rot}}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"mean $\mathcal{R}_{\mathrm{rot}}$")
    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{rot}}$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
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
    panel_cols: int,
    contour_quantiles: list[float],
    contour_alpha: float,
    times: np.ndarray,
    spatial_mean: np.ndarray,
    highlight_idx: int,
    highlight_time: float,
    highlight_width: float,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")

    total_panels = len(panel_indices)
    if total_panels == 0:
        raise ValueError("At least one panel time is required.")

    required_cols = total_panels + 1  # reserve last column for the time series
    cols = max(required_cols, int(panel_cols))
    rows = 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.3 * cols, 3.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    heat_axes = axes_flat[: cols - 1]
    series_ax = axes_flat[cols - 1]

    last_mesh = None
    for idx, ax in enumerate(heat_axes):
        if idx >= total_panels:
            ax.axis("off")
            continue
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

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=heat_axes[:total_panels], shrink=0.9, pad=0.02)
        cbar.set_label(r"$\mathcal{R}_{\mathrm{rot}}$")

    render_mean_series_axis(
        series_ax,
        times=times,
        spatial_mean=spatial_mean,
        highlight_idx=highlight_idx,
        highlight_time=highlight_time,
        highlight_width=highlight_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_highlight_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    residual_slice: np.ndarray,
    score_slice: np.ndarray | None,
    *,
    cmap: str,
    norm,
    contour_quantiles: list[float],
    contour_alpha: float,
    quiver_density: int,
    quiver_scale: float,
    quiver_color: str,
    times: np.ndarray,
    spatial_mean: np.ndarray,
    mean_abs: np.ndarray,
    max_abs: np.ndarray,
    highlight_idx: int,
    highlight_time: float,
    highlight_width: float,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    grid_x_t = grid_at_time(grid_x, highlight_idx)
    grid_y_t = grid_at_time(grid_y, highlight_idx)
    fig = plt.figure(figsize=(11.0, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.7, 1.1], wspace=0.15)
    ax_main = fig.add_subplot(gs[0])
    mesh = ax_main.pcolormesh(
        grid_x_t,
        grid_y_t,
        residual_slice,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    add_contours(ax_main, grid_x_t, grid_y_t, residual_slice, contour_quantiles, contour_alpha)

    quiver_payload = downsample_for_quiver(grid_x_t, grid_y_t, score_slice, quiver_density)
    if quiver_payload is not None:
        xs, ys, u, v = quiver_payload
        ax_main.quiver(
            xs,
            ys,
            u,
            v,
            color=quiver_color,
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=0.002,
            alpha=0.9,
        )

    ax_main.set_xlabel("x1")
    ax_main.set_ylabel("x2")
    ax_main.set_title(rf"$\mathcal{{R}}_{{\mathrm{{rot}}}}$ at $t={highlight_time:.3f}$")
    ax_main.set_aspect("equal")
    cbar = fig.colorbar(mesh, ax=ax_main, fraction=0.046, pad=0.018)
    cbar.set_label(r"$\mathcal{R}_{\mathrm{rot}}$")

    ax_stats = fig.add_subplot(gs[1])
    render_stats_axis(
        ax_stats,
        times=times,
        spatial_mean=spatial_mean,
        mean_abs=mean_abs,
        max_abs=max_abs,
        highlight_idx=highlight_idx,
        highlight_time=highlight_time,
        highlight_width=highlight_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_time_series(
    times: np.ndarray,
    spatial_mean: np.ndarray,
    highlight_idx: int,
    highlight_time: float,
    highlight_width: float,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    render_mean_series_axis(
        ax,
        times=times,
        spatial_mean=spatial_mean,
        highlight_idx=highlight_idx,
        highlight_time=highlight_time,
        highlight_width=highlight_width,
    )
    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{rot}}$ over time")

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
    score_field = data.get("score")

    if grid_x.ndim not in (2, 3) or grid_y.ndim not in (2, 3):
        raise ValueError(f"Unexpected grid dimensions: grid_x {grid_x.shape}, grid_y {grid_y.shape}")
    if grid_x.ndim != grid_y.ndim:
        raise ValueError(f"Mismatched grid dimensions: grid_x {grid_x.shape}, grid_y {grid_y.shape}")
    if grid_x.ndim == 3 and grid_x.shape[0] != times.shape[0]:
        raise ValueError(f"grid_x time dimension {grid_x.shape[0]} does not match times {times.shape[0]}")
    if grid_y.ndim == 3 and grid_y.shape[0] != times.shape[0]:
        raise ValueError(f"grid_y time dimension {grid_y.shape[0]} does not match times {times.shape[0]}")

    panel_times = (
        list(args.panel_times)
        if args.panel_times is not None
        else [0.1, 0.3, 0.5]
    )
    panel_indices, actual_panel_times = select_time_indices(times, panel_times)
    spatial_mean, mean_abs, max_abs = compute_time_series(rot_residual)
    highlight_idx, highlight_time = select_time_index(times, args.visualize_time)

    vmin, vmax = compute_value_bounds(
        rot_residual,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    # Use a fixed 0-1 color scale for publication-ready consistency.
    vmin, vmax = 0.0, 1.0
    norm = build_norm(
        args.norm,
        vmin,
        vmax,
        log_eps=args.log_eps,
        gamma=args.norm_gamma,
    )

    if args.highlight_figure is not None:
        make_highlight_plot(
            grid_x,
            grid_y,
            residual_slice=rot_residual[highlight_idx],
            score_slice=None if score_field is None else score_field[highlight_idx],
            cmap=args.cmap,
            norm=norm,
            contour_quantiles=args.highlight_contour_quantiles,
            contour_alpha=args.contour_alpha,
            quiver_density=args.quiver_density,
            quiver_scale=args.quiver_scale,
            quiver_color=args.quiver_color,
            times=times,
            spatial_mean=spatial_mean,
            mean_abs=mean_abs,
            max_abs=max_abs,
            highlight_idx=highlight_idx,
            highlight_time=highlight_time,
            highlight_width=args.highlight_width,
            output_path=args.highlight_figure,
            dpi=args.dpi,
        )
        print(f"Saved highlight figure for t={highlight_time:.3f} to {args.highlight_figure}")

    make_panel_plot(
        grid_x,
        grid_y,
        rot_residual,
        panel_indices=panel_indices,
        panel_times=actual_panel_times,
        cmap=args.cmap,
        norm=norm,
        panel_cols=args.panel_cols,
        contour_quantiles=args.panel_contour_quantiles,
        contour_alpha=args.contour_alpha,
        times=times,
        spatial_mean=spatial_mean,
        highlight_idx=highlight_idx,
        highlight_time=highlight_time,
        highlight_width=args.highlight_width,
        output_path=args.figure,
        dpi=args.dpi,
    )
    summary_times = ", ".join(f"{t:.3f}" for t in actual_panel_times)
    print(f"Saved rotational residual panels at [{summary_times}] to {args.figure}")

    agg_score = calculate_spatial_time_score(rot_residual)
    print(f"Space-time RMS rotational residual (Eq. 29): {agg_score:.6e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
