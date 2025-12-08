from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np

from calculate_continuity_residual import calculate_continuity_residual
from config import config as base_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize continuity residual heatmaps.")
    compute_group = parser.add_argument_group("continuity computation")
    compute_group.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional precomputed npz payload. If provided, computation is skipped.",
    )
    compute_group.add_argument(
        "--save-data",
        type=Path,
        default=Path("continuity_residual_times.npz"),
        help="Destination to store the computed npz payload (default: continuity_residual_times.npz).",
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
        default=260,
        help="Grid resolution per axis.",
    )
    compute_group.add_argument(
        "--span-sigma",
        type=float,
        default=4.0,
        help="Spatial span around the Gaussian mean in std-dev units.",
    )
    compute_group.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[round(0.01 + 0.01 * i, 3) for i in range(int((1.0 - 0.01) / 0.01) + 1)],
        help="Time instants used to build finite-difference snapshots (default: 0.01, 0.02, ..., 1.00).",
    )
    compute_group.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for model queries.",
    )
    compute_group.add_argument(
        "--method",
        type=str,
        default="Heun",
        help="PF-ODE integration method (Heun/Euler/RK45).",
    )
    compute_group.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of PF-ODE integration steps.",
    )

    viz_group = parser.add_argument_group("visualization")
    viz_group.add_argument(
        "--visualize-time",
        type=float,
        default=0.5,
        help="Desired time slice for visualization.",
    )
    viz_group.add_argument(
        "--figure",
        type=Path,
        default=Path("continuity_residual_t0.5.png"),
        help="Output path for the highlight (single-time) figure.",
    )
    viz_group.add_argument(
        "--panel-figure",
        type=Path,
        default=Path("continuity_residual_panels.png"),
        help="Output path for the multi-time panel figure.",
    )
    viz_group.add_argument(
        "--stats-figure",
        type=Path,
        default=Path("continuity_residual_mean_series.png"),
        help="Output path for the spatial-mean time series figure.",
    )
    viz_group.add_argument(
        "--panel-times",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        help="Requested times for the panel view. The closest available snapshots are used.",
    )
    viz_group.add_argument(
        "--panel-cols",
        type=int,
        default=4,
        help="Number of columns in the panel grid (default 1x4: 3 heatmaps + 1 time series).",
    )
    viz_group.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Colormap for the residual heatmap.",
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
        default=1e-6,
        help="Floor applied before log scaling.",
    )
    viz_group.add_argument(
        "--norm-gamma",
        type=float,
        default=0.65,
        help="Gamma exponent used when norm='power'.",
    )
    viz_group.add_argument(
        "--contour-quantiles",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 0.97],
        help="Quantiles used for contour overlays.",
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
        default=0,
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
        help="Color used for the quiver overlay vectors, when velocity is available.",
    )
    viz_group.add_argument(
        "--highlight-width",
        type=float,
        default=0.015,
        help="Half-width of the shaded band around the highlighted time in stats plots.",
    )
    viz_group.add_argument(
        "--plot-signed",
        action="store_false",
        dest="plot_absolute",
        help="Plot the signed residual instead of its magnitude.",
    )
    viz_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.set_defaults(plot_absolute=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"npz file '{path}' does not exist.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def obtain_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    # Prefer user-provided data; otherwise, load cached data if available.
    if args.data is not None:
        if args.data.exists():
            print(f"Loading precomputed continuity residuals from {args.data}")
            return load_npz(args.data)
        else:
            print(f"Requested data path {args.data} not found. Will compute instead.")
    if args.data is None and args.save_data is not None and args.save_data.exists():
        print(f"Loading cached continuity residuals from {args.save_data}")
        return load_npz(args.save_data)

    print("Computing continuity residual fields...")
    return calculate_continuity_residual(
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        span_sigma=args.span_sigma,
        times=args.times,
        batch_size=args.batch_size,
        method=args.method,
        steps=args.steps,
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
    if residual.ndim != 3:
        raise ValueError("Residual tensor must have shape (T, H, W).")
    flat = residual.reshape(residual.shape[0], -1)
    abs_vals = np.abs(flat)
    spatial_mean = flat.mean(axis=1)
    mean_abs = abs_vals.mean(axis=1)
    max_abs = abs_vals.max(axis=1)
    return spatial_mean, mean_abs, max_abs


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


def smooth_for_contour(data: np.ndarray, sigma: float = 0.8, kernel_size: int = 5) -> np.ndarray:
    """Light Gaussian smoothing to reduce jagged contours without altering heatmap."""
    k = max(1, int(kernel_size) | 1)  # force odd
    if k < 3 or sigma <= 0:
        return data
    radius = k // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    xv, yv = np.meshgrid(coords, coords, indexing="xy")
    kernel = np.exp(-(xv * xv + yv * yv) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.pad(data, radius, mode="reflect")
    smoothed = np.zeros_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            window = padded[i : i + k, j : j + k]
            smoothed[i, j] = np.sum(window * kernel)
    return smoothed


def add_contours(ax, grid_x, grid_y, data, quantiles, alpha: float) -> None:
    levels = derive_contour_levels(data, quantiles)
    if levels is None:
        return
    data_smooth = smooth_for_contour(np.asarray(data, dtype=np.float64))
    ax.contour(
        grid_x,
        grid_y,
        data_smooth,
        levels=levels,
        colors="white",
        linewidths=0.6,
        alpha=max(0.0, float(alpha)),
    )


def downsample_for_quiver(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    vec_slice: np.ndarray | None,
    density: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if vec_slice is None or density <= 0:
        return None
    if vec_slice.ndim != 3 or vec_slice.shape[2] != 2:
        return None
    stride_x = max(1, grid_x.shape[1] // max(1, density))
    stride_y = max(1, grid_x.shape[0] // max(1, density))
    xs = grid_x[::stride_y, ::stride_x]
    ys = grid_y[::stride_y, ::stride_x]
    u = vec_slice[::stride_y, ::stride_x, 0]
    v = vec_slice[::stride_y, ::stride_x, 1]
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
    stats_label: str,
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
    ax.plot(times, spatial_mean, label="mean R", color="#0a3a5c", linewidth=2.1)
    ax.plot(times, mean_abs, label="mean |R|", color="#1f77b4", linewidth=1.6, alpha=0.85)
    ax.plot(times, max_abs, label="max |R|", color="#d62728", linestyle="--", linewidth=1.3)
    ax.axvline(highlight_time, color="gray", linestyle=":", linewidth=1.2)
    ax.scatter([highlight_time], [spatial_mean[highlight_idx]], color="#0a3a5c", s=30, zorder=3)
    ax.scatter([highlight_time], [mean_abs[highlight_idx]], color="#1f77b4", s=26, zorder=3)
    ax.scatter([highlight_time], [max_abs[highlight_idx]], color="#d62728", s=26, zorder=3)
    ax.set_xlabel("t")
    ax.set_ylabel(stats_label)
    ax.set_title("Residual stats across time")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, fontsize=9)

    ax.text(
        0.02,
        0.95,
        (
            f"mean = {float(spatial_mean[highlight_idx]):+.2e}\n"
            f"mean |.| = {float(mean_abs[highlight_idx]):.2e}\n"
            f"max |.| = {float(max_abs[highlight_idx]):.2e}"
        ),
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
    ax.axhline(0.0, color="#bdbdbd", linestyle="--", linewidth=0.7)
    ax.plot(times, spatial_mean, color=accent, linewidth=2.2, label=r"mean $\mathcal{R}_{\mathrm{cont}}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"mean $\mathcal{R}_{\mathrm{cont}}$")
    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{cont}}$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def make_highlight_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    residual_slice: np.ndarray,
    velocity_slice: np.ndarray | None,
    *,
    cmap: str,
    norm,
    contour_quantiles: list[float],
    contour_alpha: float,
    quiver_density: int,
    quiver_scale: float,
    quiver_color: str,
    field_label: str,
    stats_label: str,
    time_val: float,
    times: np.ndarray,
    spatial_mean: np.ndarray,
    mean_abs: np.ndarray,
    max_abs: np.ndarray,
    idx: int,
    highlight_width: float,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    fig, (ax_main, ax_stats) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3.5, 1.3]},
    )
    mesh = ax_main.pcolormesh(
        grid_x,
        grid_y,
        residual_slice,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    add_contours(ax_main, grid_x, grid_y, residual_slice, contour_quantiles, contour_alpha)

    quiver_payload = downsample_for_quiver(grid_x, grid_y, velocity_slice, quiver_density)
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

    cbar = fig.colorbar(mesh, ax=ax_main, shrink=0.85, pad=0.02)
    cbar.set_label(field_label)

    ax_main.set_xlabel("x1")
    ax_main.set_ylabel("x2")
    ax_main.set_title(rf"{field_label} at $t={time_val:.3f}$")
    ax_main.set_aspect("equal")

    render_stats_axis(
        ax_stats,
        times=times,
        spatial_mean=spatial_mean,
        mean_abs=mean_abs,
        max_abs=max_abs,
        highlight_idx=idx,
        highlight_time=time_val,
        highlight_width=highlight_width,
        stats_label=stats_label,
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
    ax.set_title(r"Spatial mean of $\mathcal{R}_{\mathrm{cont}}$ over time")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_panel_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    residual_data: np.ndarray,
    *,
    panel_indices: list[int],
    panel_times: list[float],
    cmap: str,
    norm,
    contour_quantiles: list[float],
    contour_alpha: float,
    panel_cols: int,
    field_label: str,
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
        raise ValueError("panel_indices must contain at least one element.")
    required_cols = total_panels + 1  # last column reserved for time series
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
        slice_data = residual_data[panel_idx]
        mesh = ax.pcolormesh(
            grid_x,
            grid_y,
            slice_data,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        last_mesh = mesh
        add_contours(ax, grid_x, grid_y, slice_data, contour_quantiles, contour_alpha)
        ax.set_title(rf"$t={panel_times[idx]:.2f}$", fontsize=11)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=heat_axes[:total_panels], shrink=0.9, pad=0.02)
        cbar.set_label(field_label)

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


def main() -> None:
    args = parse_args()
    data = obtain_data(args)

    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    times = data["times"]
    residual = data["cont_residual"]
    velocity = data.get("velocity")
    spatial_mean, mean_abs, max_abs = compute_time_series(residual)

    display_residual = np.abs(residual) if args.plot_absolute else residual
    field_label = r"$|\mathcal{R}_{\mathrm{cont}}|$" if args.plot_absolute else r"$\mathcal{R}_{\mathrm{cont}}$"
    stats_label = r"$\mathcal{R}_{\mathrm{cont}}$"

    idx, actual_time = select_time_index(times, args.visualize_time)
    velocity_slice = velocity[idx] if (velocity is not None and args.quiver_density > 0) else None

    panel_times = list(args.panel_times) if args.panel_times is not None else []
    panel_indices, actual_panel_times = select_time_indices(times, panel_times) if panel_times else ([idx], [actual_time])

    vmin, vmax = compute_value_bounds(
        display_residual,
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

    make_highlight_plot(
        grid_x,
        grid_y,
        residual_slice=display_residual[idx],
        velocity_slice=velocity_slice,
        cmap=args.cmap,
        norm=norm,
        contour_quantiles=args.contour_quantiles,
        contour_alpha=args.contour_alpha,
        quiver_density=args.quiver_density,
        quiver_scale=args.quiver_scale,
        quiver_color=args.quiver_color,
        field_label=field_label,
        stats_label=stats_label,
        time_val=actual_time,
        times=times,
        spatial_mean=spatial_mean,
        mean_abs=mean_abs,
        max_abs=max_abs,
        idx=idx,
        highlight_width=args.highlight_width,
        output_path=args.figure,
        dpi=args.dpi,
    )
    slice_min = float(np.min(display_residual[idx]))
    slice_max = float(np.max(display_residual[idx]))
    print(
        f"Saved highlight figure for t={actual_time:.4f} to {args.figure} "
        f"(display range: {slice_min:.3e} to {slice_max:.3e})"
    )

    if args.panel_figure is not None:
        make_panel_plot(
            grid_x,
            grid_y,
            residual_data=display_residual,
            panel_indices=panel_indices,
            panel_times=actual_panel_times,
            cmap=args.cmap,
            norm=norm,
        contour_quantiles=args.contour_quantiles,
        contour_alpha=args.contour_alpha,
        panel_cols=args.panel_cols,
        field_label=field_label,
        times=times,
        spatial_mean=spatial_mean,
        highlight_idx=idx,
        highlight_time=actual_time,
        highlight_width=args.highlight_width,
        output_path=args.panel_figure,
        dpi=args.dpi,
    )
        summary_times = ", ".join(f"{t:.3f}" for t in actual_panel_times)
        print(f"Saved continuity residual panels at [{summary_times}] to {args.panel_figure}")

    if args.stats_figure is not None:
        plot_time_series(
            times=times,
            spatial_mean=spatial_mean,
            highlight_idx=idx,
            highlight_time=actual_time,
            highlight_width=args.highlight_width,
            output_path=args.stats_figure,
            dpi=args.dpi,
        )
        print(f"Saved spatial-mean time series to {args.stats_figure}")


if __name__ == "__main__":
    main()
