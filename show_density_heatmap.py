from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config, get_device
from data_loader import involute_params
from model import build_model
from sample import load_checkpoint
from util import integrate_log_probability
DEFAULT_SETTINGS = {
    "checkpoint_path": Path(config["config_path"]),
    "resolution": 200,
    "span_sigma": 4.0,
    "batch_size": 512,
    "output_path": Path("density_heatmaps_involute.png"),
}


def involute_log_density(
    points: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    noise_std: float,
    n_quad: int = 1024,
    chunk_size: int = 1000,
) -> np.ndarray:
    """
    Numerically evaluate the true log-density of the involute dataset:
    x | r ~ N([r*sin(2r), r*cos(2r)], noise_std^2 I), r ~ U[r_min, r_max].
    Approximated via uniform quadrature over r.
    """
    pts = np.asarray(points, dtype=np.float64)
    r = np.linspace(r_min, r_max, n_quad, dtype=np.float64)
    base_x = r * np.sin(2.0 * r)
    base_y = r * np.cos(2.0 * r)

    r_range = max(r_max - r_min, 1e-12)
    dr = r_range / max(n_quad - 1, 1)
    var = noise_std * noise_std
    log_const = math.log(dr) - math.log(r_range) - math.log(2.0 * math.pi * var)

    logp = np.empty(pts.shape[0], dtype=np.float64)

    for start in range(0, pts.shape[0], chunk_size):
        end = min(start + chunk_size, pts.shape[0])
        chunk = pts[start:end]
        dx = chunk[:, 0:1] - base_x[None, :]
        dy = chunk[:, 1:2] - base_y[None, :]
        exponent = -(dx * dx + dy * dy) / (2.0 * var)
        max_exp = np.max(exponent, axis=1, keepdims=True)
        lse = max_exp + np.log(np.exp(exponent - max_exp).sum(axis=1, keepdims=True))
        logp[start:end] = (log_const + lse.squeeze(1))
    return logp


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


def build_grid(cfg: dict, *, resolution: int, span_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a uniform 2D grid covering the involute curve plus a margin.
    span_sigma scales the noise_std to set the margin.
    """
    r_min, r_max, noise_std = involute_params(cfg)
    r = np.linspace(r_min, r_max, 2048, dtype=np.float64)
    base_x = r * np.sin(2.0 * r)
    base_y = r * np.cos(2.0 * r)

    margin = span_sigma * noise_std
    x_min, x_max = float(base_x.min() - margin), float(base_x.max() + margin)
    y_min, y_max = float(base_y.min() - margin), float(base_y.max() + margin)

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, grid_points


def main() -> None:
    settings = DEFAULT_SETTINGS.copy()
    checkpoint_path: Path = settings["checkpoint_path"]
    resolution: int = settings["resolution"]
    span_sigma: float = settings["span_sigma"]
    batch_size: int = settings["batch_size"]
    output_path: Path = settings["output_path"]

    device = get_device()
    model = build_model(config, device)
    checkpoint = load_checkpoint(model, str(checkpoint_path), device)
    checkpoint_config = checkpoint.get("config", {}).copy()
    merged_config = config.copy()
    merged_config.update(checkpoint_config)
    merged_config["config_path"] = str(checkpoint_path)

    grid_x, grid_y, grid_points = build_grid(
        merged_config,
        resolution=resolution,
        span_sigma=span_sigma,
    )

    r_min, r_max, noise_std = involute_params(merged_config)
    analytic_log_density = involute_log_density(
        grid_points,
        r_min=r_min,
        r_max=r_max,
        noise_std=noise_std,
    ).reshape(grid_x.shape)
    analytic_density = np.exp(analytic_log_density)

    steps_eval = int(merged_config.get("density_eval_steps", 1000))
    eps_eval = float(merged_config.get("nll_eval_eps", merged_config.get("eps", 1e-3)))
    method = str(merged_config.get("density_method", "Heun"))

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
    model_density = np.exp(model_log_density)

    # Clean numerical artifacts only; do not zero-out boundaries explicitly.
    analytic_density = np.nan_to_num(analytic_density, nan=0.0, posinf=0.0, neginf=0.0)
    model_density = np.nan_to_num(model_density, nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = np.abs(model_density - analytic_density)

    density_vmin = float(min(analytic_density.min(), model_density.min()))
    density_vmax = float(max(analytic_density.max(), model_density.max()))

    plt.style.use("seaborn-v0_8")
    output_dir = output_path.parent if output_path.parent != Path("") else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix if output_path.suffix else ".png"
    base_name = output_path.stem or "density_heatmap"
    figure_size = (5.0, 4.2)

    density_cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")

    def save_single_plot(
        data: np.ndarray,
        title: str,
        cmap,
        target_path: Path,
        colorbar_label: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=figure_size)
        im = ax.pcolormesh(grid_x, grid_y, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label(colorbar_label)
        fig.tight_layout()
        fig.savefig(target_path, dpi=300)
        plt.close(fig)

    analytic_path = output_dir / f"{base_name}_involute_density{suffix}"
    model_path = output_dir / f"{base_name}_pf_ode_density{suffix}"
    error_path = output_dir / f"{base_name}_abs_error{suffix}"

    save_single_plot(
        analytic_density,
        "Density (Involute)",
        density_cmap,
        analytic_path,
        "Probability density",
        vmin=density_vmin,
        vmax=density_vmax,
    )
    save_single_plot(
        model_density,
        "PF-ODE Density",
        density_cmap,
        model_path,
        "Probability density",
        vmin=density_vmin,
        vmax=density_vmax,
    )
    save_single_plot(
        abs_error,
        "Absolute Error",
        error_cmap,
        error_path,
        "Absolute error",
    )

    print(f"Saved involute density heatmap to {analytic_path}")
    print(f"Saved PF-ODE density heatmap to {model_path}")
    print(f"Saved absolute error heatmap to {error_path}")


if __name__ == "__main__":
    main()
