from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config, get_device
from data_loader import (
    banana_log_density,
    gaussian_log_density,
    generate_test,
    get_distribution_stats,
)
from model import build_model
from sample import load_checkpoint, pc_sample, pf_sample
from util import integrate_log_probability, set_seed

DEFAULT_SETTINGS = {
    "checkpoint_path": Path(config["config_path"]),
    "resolution": 200,
    "span_sigma": 4.0,
    "batch_size": 512,
    "output_path": Path("density_heatmaps_banana.png"),
    "method": "Heun",
    "steps_override": None,
    "eps_override": None,
    "plot_bounds": (-6.0, 6.0, -6.0, 6.0),
}


def evaluate_model_log_density(
    model,
    cfg: dict,
    device: torch.device,
    points: np.ndarray,
    *,
    steps: int,
    eps: float,
    method: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate PF-ODE log-density over a set of 2D points."""
    model.eval()
    outputs: list[np.ndarray] = []
    total = points.shape[0]

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xb = torch.from_numpy(points[start:end]).to(device=device, dtype=torch.float32)
        with torch.enable_grad():
            logp = integrate_log_probability(
                model,
                xb,
                cfg,
                steps=steps,
                eps=eps,
                method=method,
            )
        outputs.append(logp.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_grid(
    cfg: dict,
    *,
    resolution: int,
    span_sigma: float,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a 2D grid; if bounds are provided, they override span-based limits."""
    if x_limits is not None and y_limits is not None:
        x_vals = np.linspace(x_limits[0], x_limits[1], resolution)
        y_vals = np.linspace(y_limits[0], y_limits[1], resolution)
    else:
        mean, cov = get_distribution_stats(cfg)

        std = np.sqrt(np.diag(cov))
        span = span_sigma * std
        x_vals = np.linspace(mean[0] - span[0], mean[0] + span[0], resolution)
        y_vals = np.linspace(mean[1] - span[1], mean[1] + span[1], resolution)
    
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, grid_points


def sanitize_for_plot(arr: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf with a finite fill value to avoid blank patches in pcolormesh."""
    cleaned = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return np.clip(cleaned, a_min=0.0, a_max=None)


def main() -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    settings = DEFAULT_SETTINGS.copy()
    checkpoint_path: Path = settings["checkpoint_path"]
    resolution: int = settings["resolution"]
    span_sigma: float = settings["span_sigma"]
    batch_size: int = settings["batch_size"]
    output_path: Path = settings["output_path"]
    method_pref: str | None = settings["method"]
    steps_override: int | None = settings["steps_override"]
    eps_override: float | None = settings["eps_override"]
    plot_bounds_setting: tuple[float, float, float, float] | None = settings.get("plot_bounds")

    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    checkpoint_config = checkpoint.get("config", {}).copy()
    merged_config = config.copy()
    merged_config.update(checkpoint_config)
    merged_config["config_path"] = str(checkpoint_path)

    if plot_bounds_setting is not None:
        x_limits = (float(plot_bounds_setting[0]), float(plot_bounds_setting[1]))
        y_limits = (float(plot_bounds_setting[2]), float(plot_bounds_setting[3]))
    else:
        # Align the heatmap bounds with the scatter plot scale for visual consistency.
        test_np = generate_test(merged_config, seed=seed)

        bounds_samples = [test_np]
        n_bounds = int(min(max(1, merged_config.get("n_gen", 5000)), 2000))
        with torch.no_grad():
            try:
                pc_bounds = pc_sample(model, merged_config, device, n_bounds).cpu().numpy()
                bounds_samples.append(pc_bounds)
            except Exception as e:
                print(f"[warn] PC sample for bounds failed: {e}")
            try:
                pf_bounds = pf_sample(model, merged_config, device, n_bounds).cpu().numpy()
                bounds_samples.append(pf_bounds)
            except Exception as e:
                print(f"[warn] PF sample for bounds failed: {e}")

        combined_bounds = np.vstack(bounds_samples)
        finite_mask = np.isfinite(combined_bounds).all(axis=1)
        if not finite_mask.any():
            y_limits = (-1.0, 1.0)
        else:
            y_vals = combined_bounds[finite_mask][:, 1]
            y_min, y_max = float(y_vals.min()), float(y_vals.max())
            span = y_max - y_min
            pad = max(0.05 * span, 1e-6) if span > 0 else max(1.0, max(abs(y_min), abs(y_max)) * 0.1 + 1e-6)
            y_limits = (y_min - pad, y_max + pad)
        x_limits = (-4.0, 4.0)

    grid_x, grid_y, grid_points = build_grid(
        merged_config,
        resolution=resolution,
        span_sigma=span_sigma,
        x_limits=x_limits,
        y_limits=y_limits,
    )

    dist = str(merged_config.get("distribution", "gaussian")).lower()
    if dist == "banana":
        a = float(merged_config.get("banana_a", 1.0))
        analytic_log_density = banana_log_density(grid_points, a).reshape(grid_x.shape)
    else:
        mean, cov = get_distribution_stats(merged_config)
        analytic_log_density = gaussian_log_density(grid_points, mean, cov).reshape(grid_x.shape)
    
    analytic_density = sanitize_for_plot(np.exp(analytic_log_density))

    steps_eval = int(steps_override or merged_config.get("density_eval_steps", 600))
    eps_eval = float(eps_override or merged_config.get("nll_eval_eps", merged_config.get("eps", 1e-3)))
    method = str(method_pref or merged_config.get("density_method", "Heun"))

    model_log_density = evaluate_model_log_density(
        model,
        merged_config,
        device,
        grid_points,
        steps=steps_eval,
        eps=eps_eval,
        method=method,
        batch_size=max(1, batch_size),
    ).reshape(grid_x.shape)
    model_density = sanitize_for_plot(np.exp(model_log_density))

    abs_error = sanitize_for_plot(np.abs(model_density - analytic_density))

    density_vmin = float(np.nanmin(np.stack([analytic_density, model_density])))
    density_vmax = float(np.nanmax(np.stack([analytic_density, model_density])))
    plot_bounds = (x_limits[0], x_limits[1], y_limits[0], y_limits[1])
    target_path = output_path

    plt.style.use("seaborn-v0_8")
    figure_size = (11.2, 3.6)

    density_cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")
    colorbar_height_frac = 0.85
    colorbar_width_scale = 0.8
    colorbar_left_shift = 0.03  # move colorbars left (figure coords) to avoid overlap

    fig = plt.figure(figsize=figure_size)
    gs = fig.add_gridspec(1, 6, width_ratios=[1.0, 0.05, 1.0, 0.05, 1.0, 0.05])
    ax_analytic = fig.add_subplot(gs[0, 0])
    cax_analytic = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    cax_model = fig.add_subplot(gs[0, 3])
    ax_error = fig.add_subplot(gs[0, 4])
    cax_error = fig.add_subplot(gs[0, 5])

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, wspace=0.28)

    def draw_panel(ax, cax, data, title, cmap, cbar_label, vmin=None, vmax=None):
        im = ax.pcolormesh(grid_x, grid_y, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        if plot_bounds is not None:
            x_min, x_max, y_min, y_max = plot_bounds
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label:
            cbar.set_label(cbar_label)
        cax_box = cax.get_position()
        new_width = cax_box.width * colorbar_width_scale
        new_height = cax_box.height * colorbar_height_frac
        new_x0 = max(cax_box.x0 - colorbar_left_shift, 0.0)
        new_y0 = cax_box.y0 + (cax_box.height - new_height) * 0.5
        cax.set_position([new_x0, new_y0, new_width, new_height])

    draw_panel(
        ax_analytic,
        cax_analytic,
        analytic_density,
        "Density (Banana)",
        density_cmap,
        "",
        vmin=density_vmin,
        vmax=density_vmax,
    )
    draw_panel(
        ax_model,
        cax_model,
        model_density,
        "PF-ODE Density",
        density_cmap,
        "",
        vmin=density_vmin,
        vmax=density_vmax,
    )
    draw_panel(
        ax_error,
        cax_error,
        abs_error,
        "Absolute Error",
        error_cmap,
        "",
    )

    fig.savefig(target_path, dpi=300)
    plt.close(fig)

    print(f"Saved density heatmaps panel to {target_path}")


if __name__ == "__main__":
    main()
