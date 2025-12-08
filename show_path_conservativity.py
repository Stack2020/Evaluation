from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np

from calculate_path_conservativity import calculate_path_conservativity
from config import config as base_config

COLORBAR_LABEL = r"$|\widehat{I}_C| / \operatorname{length}(C)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate path-conservativity residuals on polygonal loops.")
    compute_group = parser.add_argument_group("path metric computation")
    compute_group.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional precomputed npz payload. If set, skips recomputation.",
    )
    compute_group.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="Optional destination to store computed tensors.",
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
        help="Grid resolution per axis used to build loops.",
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
        help="Time instants used for evaluating the vector field (default: 0.01, 0.02, ..., 1.00).",
    )
    compute_group.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for model evaluations.",
    )
    compute_group.add_argument(
        "--field-type",
        type=str,
        choices=("score", "drift"),
        default="drift",
        help="Vector field choice: score s_theta(x,t) or PF-ODE drift v_theta(x,t).",
    )

    viz_group = parser.add_argument_group("visualization")
    viz_group.add_argument(
        "--visualize-time",
        type=float,
        default=0.5,
        help="Desired time slice used for visualization.",
    )
    viz_group.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Destination figure path. If omitted, the per-time heatmap is skipped.",
    )
    viz_group.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Colormap for the loop residual heatmap.",
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
        "--norm-gamma",
        type=float,
        default=0.65,
        help="Gamma exponent when norm='power'.",
    )
    viz_group.add_argument(
        "--log-eps",
        type=float,
        default=1e-6,
        help="Floor applied before log scaling.",
    )
    viz_group.add_argument(
        "--panel-figure",
        type=Path,
        default=None,
        help="Destination for the multi-time panel heatmap. If omitted, panels are skipped.",
    )
    viz_group.add_argument(
        "--field-stats-figure",
        type=Path,
        default=Path("path_conservativity_field_mean.png"),
        help="Destination for the time-series plot of mean path residual over space.",
    )
    viz_group.add_argument(
        "--panel-times",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Requested times for the panel view. The closest available snapshots are used.",
    )
    viz_group.add_argument(
        "--panel-cols",
        type=int,
        default=3,
        help="Number of columns in the panel grid.",
    )
    viz_group.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Fixed lower bound for the color scale (default: 0.0).",
    )
    viz_group.add_argument(
        "--vmax",
        type=float,
        default=0.3,
        help="Fixed upper bound for the color scale (default: 0.3).",
    )
    viz_group.add_argument(
        "--loop-top-k",
        type=int,
        default=12,
        help="Number of loops with largest residuals to highlight.",
    )
    viz_group.add_argument(
        "--quiver-step",
        type=int,
        default=12,
        help="Stride for quiver down-sampling of the vector field.",
    )
    viz_group.add_argument(
        "--quiver-scale",
        type=float,
        default=0.9,
        help="Relative scaling applied to normalized quiver arrows.",
    )
    viz_group.add_argument(
        "--stream-density",
        type=float,
        default=1.4,
        help="Density parameter for matplotlib.streamplot.",
    )
    viz_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"npz file '{path}' does not exist.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def obtain_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.data is not None:
        print(f"Loading precomputed path-conservativity payload from {args.data}")
        return load_npz(args.data)
    print("Computing path-conservativity tensors ...")
    return calculate_path_conservativity(
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        span_sigma=args.span_sigma,
        times=args.times,
        batch_size=args.batch_size,
        field_type=args.field_type,
        output_path=args.save_data,
    )


def select_time_index(times: np.ndarray, requested: float) -> tuple[int, float]:
    if times.ndim != 1:
        raise ValueError("times array must be 1-D.")
    idx = int(np.argmin(np.abs(times - requested)))
    return idx, float(times[idx])


def select_time_indices(times: np.ndarray, requested_list: list[float]) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    actual_times: list[float] = []
    for requested in requested_list:
        idx, actual = select_time_index(times, requested)
        indices.append(idx)
        actual_times.append(actual)
    return indices, actual_times


def compute_value_bounds(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
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
        span = abs(vmax) * 0.05 if vmax != 0.0 else 1.0
        vmin -= span
        vmax += span
    return vmin, vmax


def build_norm(norm_type: str, vmin: float, vmax: float, *, log_eps: float, gamma: float):
    from matplotlib.colors import LogNorm, Normalize, PowerNorm

    if vmax < vmin:
        vmin, vmax = vmax, vmin
    norm_type = norm_type.lower()
    if norm_type == "log":
        safe_min = max(vmin, float(log_eps))
        safe_max = max(vmax, safe_min * 1.01)
        return LogNorm(vmin=safe_min, vmax=safe_max)
    if norm_type == "power":
        safe_gamma = max(1e-3, float(gamma))
        return PowerNorm(gamma=safe_gamma, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def extract_field_type(payload: dict[str, np.ndarray], fallback: str) -> str:
    value = payload.get("field_type")
    if value is None:
        return fallback
    if isinstance(value, np.ndarray):
        try:
            return str(value.item())
        except ValueError:
            return str(value)
    return str(value)


def gather_loop_coordinates(grid_x: np.ndarray, grid_y: np.ndarray, loop_indices: np.ndarray) -> np.ndarray:
    coords = np.stack((grid_x, grid_y), axis=-1)
    rows = loop_indices[..., 0].astype(np.int64, copy=False)
    cols = loop_indices[..., 1].astype(np.int64, copy=False)
    return coords[rows, cols]


def select_top_loops(values: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or values.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = min(int(top_k), values.size)
    indices = np.argpartition(values, -k)[-k:]
    order = np.argsort(values[indices])[::-1]
    return indices[order]


def highlight_loops(
    ax,
    loop_coords: np.ndarray,
    loop_values: np.ndarray,
    loop_ids: np.ndarray,
) -> None:
    if loop_ids.size == 0:
        return
    for rank, loop_idx in enumerate(loop_ids, start=1):
        coords = loop_coords[loop_idx]
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color="#ffbf00",
            linewidth=1.2,
            alpha=0.95 if rank == 1 else 0.7,
        )
        center = coords[:-1].mean(axis=0)
        ax.text(
            center[0],
            center[1],
            f"{loop_values[loop_idx]:.1e}",
            color="#ffbf00",
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, linewidth=0.0),
        )


def make_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cell_slice: np.ndarray,
    vector_slice: np.ndarray | None,
    *,
    path_metric: np.ndarray,
    times: np.ndarray,
    idx: int,
    time_val: float,
    field_type: str,
    cmap: str,
    norm,
    quiver_step: int,
    quiver_scale: float,
    stream_density: float,
    loop_coords: np.ndarray,
    loop_values: np.ndarray,
    top_loop_ids: np.ndarray,
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
        cell_slice,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(mesh, ax=ax_main, shrink=0.85, pad=0.02)
    cbar.set_label(COLORBAR_LABEL)

    highlight_loops(ax_main, loop_coords, loop_values, top_loop_ids)
    ax_main.set_xlabel("x1")
    ax_main.set_ylabel("x2")
    ax_main.set_aspect("equal")
    ax_main.set_title(rf"$|\widehat{{I}}_C|/\mathrm{{length}}(C)$ at $t={time_val:.3f}$ [{field_type}]")

    ax_stats.plot(times, path_metric, color="#1f77b4", linewidth=1.8, label=r"$\mathcal{R}_{\mathrm{path}}$")
    ax_stats.axvline(time_val, color="gray", linestyle=":", linewidth=1.2)
    ax_stats.scatter([time_val], [path_metric[idx]], color="#d62728", s=30, zorder=3, label="selected")
    ax_stats.set_xlabel("t")
    ax_stats.set_ylabel(r"$\mathcal{R}_{\mathrm{path}}$")
    ax_stats.set_title("Closed-loop residual summary")
    ax_stats.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax_stats.legend(frameon=False, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_panel_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    loop_metric_map: np.ndarray,
    *,
    panel_indices: list[int],
    panel_times: list[float],
    cmap: str,
    norm,
    panel_cols: int,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    total_panels = len(panel_indices)
    if total_panels == 0:
        raise ValueError("panel_indices must contain at least one element.")
    cols = max(1, min(int(panel_cols), total_panels))
    rows = int(math.ceil(total_panels / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.8 * cols, 3.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    last_mesh = None
    for idx, ax in enumerate(axes_flat):
        if idx >= total_panels:
            ax.axis("off")
            continue
        panel_idx = panel_indices[idx]
        slice_data = loop_metric_map[panel_idx]
        mesh = ax.pcolormesh(
            grid_x,
            grid_y,
            slice_data,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        last_mesh = mesh
        ax.set_title(rf"$t={panel_times[idx]:.2f}$", fontsize=11)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes_flat[:total_panels], shrink=0.9, pad=0.03)
        cbar.set_label(COLORBAR_LABEL)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def summarize(times: np.ndarray, path_metric: np.ndarray) -> None:
    print("\nEstimated path-independence residuals:")
    for t_val, metric in zip(times, path_metric):
        print(f"  t={float(t_val):.3f} -> R_path={float(metric):.4e}")


def plot_path_metric_series(
    times: np.ndarray,
    path_metric: np.ndarray,
    *,
    field_type: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    accent = "#0a3a5c"

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.set_facecolor("#f9fafb")
    ax.plot(times, path_metric, color=accent, linewidth=2.4, label="mean residual")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathcal{R}_{\mathrm{path}}$")
    ax.set_title(rf"Mean closed-loop residual over space [{field_type}]")
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=False, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def describe_slice(
    time_val: float,
    cell_slice: np.ndarray,
    loop_values: np.ndarray,
    loop_integrals: np.ndarray,
    loop_lengths: np.ndarray,
    loop_coords: np.ndarray,
    top_ids: np.ndarray,
) -> None:
    mean_val = float(cell_slice.mean())
    max_val = float(cell_slice.max())
    print(
        f"\nSlice t={time_val:.3f}: mean(|I|/len)={mean_val:.4e}, "
        f"max(|I|/len)={max_val:.4e}"
    )
    if top_ids.size == 0:
        return
    print("Top loops at this slice:")
    for rank, loop_idx in enumerate(top_ids, start=1):
        coords = loop_coords[loop_idx]
        center = coords[:-1].mean(axis=0)
        normalized_val = float(loop_values[loop_idx])
        abs_integral = float(abs(loop_integrals[loop_idx]))
        length = float(loop_lengths[loop_idx])
        print(
            f"  #{rank:02d} center=({center[0]:.3f}, {center[1]:.3f}) "
            f"|I|={abs_integral:.3e}, length={length:.3e}, |I|/len={normalized_val:.3e}"
        )


def main() -> None:
    args = parse_args()
    data = obtain_data(args)

    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    times = data["times"]
    loop_indices = data["loop_indices"]
    loop_lengths = data["loop_lengths"]
    loop_metric = data["loop_metric"]
    loop_metric_map = data.get("loop_metric_map")
    if loop_metric_map is None:
        loop_shape = tuple(int(v) for v in data.get("loop_shape", []))
        if not loop_shape:
            raise ValueError("loop_shape missing from payload.")
        loop_metric_map = loop_metric.reshape((times.shape[0], *loop_shape))
    path_metric = data.get("path_metric", loop_metric.mean(axis=1))

    idx, actual_time = select_time_index(times, args.visualize_time)
    panel_times_req = list(args.panel_times) if args.panel_times is not None else []
    panel_indices, actual_panel_times = (
        select_time_indices(times, panel_times_req) if panel_times_req else ([idx], [actual_time])
    )
    if args.vmin is not None or args.vmax is not None:
        data_min = float(np.min(loop_metric_map))
        data_max = float(np.max(loop_metric_map))
        vmin = float(args.vmin) if args.vmin is not None else data_min
        vmax = float(args.vmax) if args.vmax is not None else data_max
    else:
        vmin, vmax = compute_value_bounds(
            loop_metric_map,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
        )
    norm = build_norm(args.norm, vmin, vmax, log_eps=args.log_eps, gamma=args.norm_gamma)

    loop_coords = gather_loop_coordinates(grid_x, grid_y, loop_indices)
    loop_values = loop_metric[idx]
    loop_integrals = data["loop_integrals"][idx]
    top_ids = select_top_loops(loop_values, args.loop_top_k)
    field_type = extract_field_type(data, args.field_type)

    summarize(times, path_metric)
    if args.field_stats_figure is not None:
        plot_path_metric_series(
            times,
            path_metric,
            field_type=field_type,
            output_path=args.field_stats_figure,
            dpi=args.dpi,
        )
        print(f"Saved mean path residual time series to {args.field_stats_figure}")

    describe_slice(
        actual_time,
        loop_metric_map[idx],
        loop_values,
        loop_integrals,
        loop_lengths,
        loop_coords,
        top_ids,
    )

    if args.figure is not None:
        make_plot(
            grid_x,
            grid_y,
            loop_metric_map[idx],
            None,
            path_metric=path_metric,
            times=times,
            idx=idx,
            time_val=actual_time,
            field_type=field_type,
            cmap=args.cmap,
            norm=norm,
            quiver_step=args.quiver_step,
            quiver_scale=args.quiver_scale,
            stream_density=args.stream_density,
            loop_coords=loop_coords,
            loop_values=loop_values,
            top_loop_ids=top_ids,
            output_path=args.figure,
            dpi=args.dpi,
        )
        print(f"\nSaved path-conservativity visualization for t={actual_time:.3f} to {args.figure}")

    if args.panel_figure is not None:
        make_panel_plot(
            grid_x,
            grid_y,
            loop_metric_map,
            panel_indices=panel_indices,
            panel_times=actual_panel_times,
            cmap=args.cmap,
            norm=norm,
            panel_cols=args.panel_cols,
            output_path=args.panel_figure,
            dpi=args.dpi,
        )
        summary_times = ", ".join(f"{t:.3f}" for t in actual_panel_times)
        print(f"Saved path-conservativity panels at [{summary_times}] to {args.panel_figure}")


if __name__ == "__main__":
    main()
